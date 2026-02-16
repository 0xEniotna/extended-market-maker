#!/usr/bin/env python3
"""Iteratively run the market maker, analyze results, and adjust config.

Safe-by-default:
- only an explicit allowlist of MM_* keys may be tuned
- base env file is never modified
- each iteration writes a copy like `.env.cop.iter001`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# -------------------------
# Tuning boundaries (can be overridden by CLI flags)
# -------------------------
DEFAULT_BOUNDS = {
    "MM_SPREAD_MULTIPLIER": (Decimal("0.05"), Decimal("5.0")),
    "MM_MIN_OFFSET_BPS": (Decimal("0"), Decimal("50")),
    "MM_MAX_OFFSET_BPS": (Decimal("1"), Decimal("100")),
    "MM_MIN_SPREAD_BPS": (Decimal("0"), Decimal("50")),
    "MM_REPRICE_TOLERANCE_PERCENT": (Decimal("0.05"), Decimal("2.0")),
    "MM_MIN_REPRICE_INTERVAL_S": (Decimal("0"), Decimal("10.0")),
    "MM_MIN_REPRICE_MOVE_TICKS": (Decimal("0"), Decimal("20")),
    "MM_MIN_REPRICE_EDGE_DELTA_BPS": (Decimal("0"), Decimal("20")),
    "MM_POST_ONLY_SAFETY_TICKS": (Decimal("1"), Decimal("20")),
    "MM_POF_MAX_SAFETY_TICKS": (Decimal("1"), Decimal("40")),
    "MM_POF_BACKOFF_MULTIPLIER": (Decimal("1.0"), Decimal("5.0")),
    "MM_INVENTORY_SKEW_FACTOR": (Decimal("0"), Decimal("1.0")),
    "MM_INVENTORY_DEADBAND_PCT": (Decimal("0"), Decimal("0.5")),
    "MM_SKEW_SHAPE_K": (Decimal("0"), Decimal("6.0")),
    "MM_SKEW_MAX_BPS": (Decimal("0"), Decimal("100")),
    "MM_IMBALANCE_PAUSE_THRESHOLD": (Decimal("0"), Decimal("1.0")),
    "MM_FAILURE_RATE_TRIP": (Decimal("0.05"), Decimal("1.0")),
    "MM_MIN_ATTEMPTS_FOR_BREAKER": (Decimal("1"), Decimal("200")),
    "MM_NUM_PRICE_LEVELS": (Decimal("1"), Decimal("10")),
    "MM_VOL_REGIME_CALM_BPS": (Decimal("2"), Decimal("20")),
    "MM_VOL_REGIME_ELEVATED_BPS": (Decimal("5"), Decimal("60")),
    "MM_VOL_REGIME_EXTREME_BPS": (Decimal("10"), Decimal("120")),
    "MM_VOL_OFFSET_SCALE_CALM": (Decimal("0.5"), Decimal("1.0")),
    "MM_VOL_OFFSET_SCALE_ELEVATED": (Decimal("1.0"), Decimal("3.0")),
    "MM_VOL_OFFSET_SCALE_EXTREME": (Decimal("1.0"), Decimal("5.0")),
    "MM_TREND_STRONG_THRESHOLD": (Decimal("0.2"), Decimal("1.0")),
    "MM_TREND_COUNTER_SIDE_SIZE_CUT": (Decimal("0.0"), Decimal("1.0")),
    "MM_TREND_SKEW_BOOST": (Decimal("1.0"), Decimal("3.0")),
    "MM_INVENTORY_WARN_PCT": (Decimal("0.2"), Decimal("0.9")),
    "MM_INVENTORY_CRITICAL_PCT": (Decimal("0.3"), Decimal("0.99")),
    "MM_INVENTORY_HARD_PCT": (Decimal("0.5"), Decimal("0.999")),
    "MM_FUNDING_INVENTORY_WEIGHT": (Decimal("0.0"), Decimal("3.0")),
    "MM_FUNDING_BIAS_CAP_BPS": (Decimal("0.0"), Decimal("20.0")),
}

ALLOWED_TUNING_KEYS = {
    "MM_NUM_PRICE_LEVELS",
    "MM_SPREAD_MULTIPLIER",
    "MM_MIN_OFFSET_BPS",
    "MM_MAX_OFFSET_BPS",
    "MM_MIN_SPREAD_BPS",
    "MM_REPRICE_TOLERANCE_PERCENT",
    "MM_MIN_REPRICE_INTERVAL_S",
    "MM_MIN_REPRICE_MOVE_TICKS",
    "MM_MIN_REPRICE_EDGE_DELTA_BPS",
    "MM_POST_ONLY_SAFETY_TICKS",
    "MM_POF_MAX_SAFETY_TICKS",
    "MM_POF_BACKOFF_MULTIPLIER",
    "MM_INVENTORY_SKEW_FACTOR",
    "MM_INVENTORY_DEADBAND_PCT",
    "MM_SKEW_SHAPE_K",
    "MM_SKEW_MAX_BPS",
    "MM_IMBALANCE_PAUSE_THRESHOLD",
    "MM_FAILURE_RATE_TRIP",
    "MM_MIN_ATTEMPTS_FOR_BREAKER",
    "MM_VOL_REGIME_CALM_BPS",
    "MM_VOL_REGIME_ELEVATED_BPS",
    "MM_VOL_REGIME_EXTREME_BPS",
    "MM_VOL_OFFSET_SCALE_CALM",
    "MM_VOL_OFFSET_SCALE_ELEVATED",
    "MM_VOL_OFFSET_SCALE_EXTREME",
    "MM_TREND_STRONG_THRESHOLD",
    "MM_TREND_COUNTER_SIDE_SIZE_CUT",
    "MM_TREND_SKEW_BOOST",
    "MM_INVENTORY_WARN_PCT",
    "MM_INVENTORY_CRITICAL_PCT",
    "MM_INVENTORY_HARD_PCT",
    "MM_FUNDING_INVENTORY_WEIGHT",
    "MM_FUNDING_BIAS_CAP_BPS",
}

READONLY_KEYS = {
    "MM_VAULT_ID",
    "MM_STARK_PRIVATE_KEY",
    "MM_STARK_PUBLIC_KEY",
    "MM_API_KEY",
    "MM_ENVIRONMENT",
    "MM_MARKET_NAME",
    "MM_OFFSET_MODE",
    "MM_ORDER_SIZE_MULTIPLIER",
    "MM_MAX_POSITION_SIZE",
    "MM_MAX_POSITION_NOTIONAL_USD",
    "MM_MAX_ORDER_NOTIONAL_USD",
}


@dataclass
class Metrics:
    fills: int = 0
    fill_rate_pct: Optional[Decimal] = None
    avg_edge_bps: Optional[Decimal] = None
    adverse_edges: Optional[Tuple[int, int]] = None
    avg_spread_at_fill_bps: Optional[Decimal] = None
    avg_spread_at_place_bps: Optional[Decimal] = None
    markout_5s_bps: Optional[Decimal] = None
    cancellations_pct: Optional[Decimal] = None
    rejection_pct: Optional[Decimal] = None
    post_only_reject_pct: Optional[Decimal] = None
    final_position: Optional[Decimal] = None
    last10_realized_pnl: Optional[Decimal] = None


def _decimal(val: str) -> Optional[Decimal]:
    try:
        return Decimal(val)
    except Exception:
        return None


def _slugify_token(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "unknown"


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current


def read_env_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


def parse_env(lines: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        # Keep .env compatibility with inline comments:
        # VAR=123   # comment  -> value is 123
        value_clean = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        env[key.strip()] = value_clean
    return env


def update_env_lines(lines: List[str], updates: Dict[str, str]) -> List[str]:
    remaining = dict(updates)
    out: List[str] = []
    for line in lines:
        if "=" not in line or line.strip().startswith("#"):
            out.append(line)
            continue
        key, rest = line.split("=", 1)
        key_clean = key.strip()
        if key_clean in updates:
            # Preserve any inline comment
            match = re.match(r"([^#]*)(#.*)?", rest)
            comment = match.group(2) if match else ""
            new_value = updates[key_clean]
            out.append(f"{key_clean}={new_value}{(' ' + comment) if comment else ''}")
            remaining.pop(key_clean, None)
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    return out


def ensure_bounds(
    key: str,
    value: Decimal,
    bounds: Dict[str, Tuple[Decimal, Decimal]],
) -> Decimal:
    lo, hi = bounds[key]
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def parse_metrics(path: Path) -> Metrics:
    metrics = Metrics()
    text = path.read_text()

    m = re.search(r"Total:\s+(\d+)\s+fills", text)
    if m:
        metrics.fills = int(m.group(1))

    m = re.search(r"## Fill Rate:\s+([\d.]+)%", text)
    if m:
        metrics.fill_rate_pct = _decimal(m.group(1))

    m = re.search(r"Edge \(vs mid\): avg=([\-\d.]+)bps.*adverse=(\d+)/(\d+)", text)
    if m:
        metrics.avg_edge_bps = _decimal(m.group(1))
        metrics.adverse_edges = (int(m.group(2)), int(m.group(3)))

    m = re.search(r"Spread at fill: avg=([\-\d.]+)bps", text)
    if m:
        metrics.avg_spread_at_fill_bps = _decimal(m.group(1))

    m = re.search(r"Avg spread at placement: ([\-\d.]+)bps", text)
    if m:
        metrics.avg_spread_at_place_bps = _decimal(m.group(1))

    m = re.search(r"\+5s: avg=([\-\d.]+)", text)
    if m:
        metrics.markout_5s_bps = _decimal(m.group(1))

    m = re.search(r"Cancellations: \d+ \(([\d.]+)%", text)
    if m:
        metrics.cancellations_pct = _decimal(m.group(1))

    m = re.search(r"Rejections: \d+ \(([\d.]+)%", text)
    if m:
        metrics.rejection_pct = _decimal(m.group(1))

    m = re.search(r"Post-only rejects: \d+ \(([\d.]+)%", text)
    if m:
        metrics.post_only_reject_pct = _decimal(m.group(1))

    m = re.search(r"Final position: ([\-\d.]+)", text)
    if m:
        metrics.final_position = _decimal(m.group(1))

    return metrics


def load_journal(path: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def _fill_side(fill: Dict[str, object]) -> Optional[str]:
    side = str(fill.get("side", ""))
    if "BUY" in side:
        return "BUY"
    if "SELL" in side:
        return "SELL"
    return None


def compute_realized_pnl_last_n_fills(path: Path, n: int = 10) -> Optional[Decimal]:
    events = load_journal(path)
    fills = [e for e in events if e.get("type") == "fill"]
    if len(fills) < n:
        return None
    recent = fills[-n:]

    # FIFO lots; positive qty = long, negative qty = short
    lots: List[Tuple[Decimal, Decimal]] = []  # (qty, price)
    realized = Decimal("0")

    for f in recent:
        side = _fill_side(f)
        if side is None:
            continue
        qty = _decimal(str(f.get("qty", "0"))) or Decimal("0")
        price = _decimal(str(f.get("price", "0"))) or Decimal("0")
        if qty <= 0 or price <= 0:
            continue

        incoming = qty if side == "BUY" else -qty

        # Match against opposite lots
        i = 0
        while i < len(lots) and incoming != 0:
            lot_qty, lot_price = lots[i]
            if lot_qty == 0 or (lot_qty > 0) == (incoming > 0):
                i += 1
                continue
            match_qty = min(abs(lot_qty), abs(incoming))
            if incoming < 0 and lot_qty > 0:
                # selling against long
                realized += (price - lot_price) * match_qty
            elif incoming > 0 and lot_qty < 0:
                # buying to cover short
                realized += (lot_price - price) * match_qty

            # reduce lot and incoming
            lot_qty = lot_qty + (match_qty if lot_qty < 0 else -match_qty)
            incoming = incoming + (match_qty if incoming < 0 else -match_qty)
            lots[i] = (lot_qty, lot_price)
            if lots[i][0] == 0:
                lots.pop(i)
            else:
                i += 1

        if incoming != 0:
            lots.append((incoming, price))

    return realized


def compute_updates(
    current: Dict[str, Decimal],
    metrics: Metrics,
    bounds: Dict[str, Tuple[Decimal, Decimal]],
    max_position_size: Optional[Decimal],
) -> Tuple[Dict[str, Decimal], List[str]]:
    updates: Dict[str, Decimal] = {}
    reasons: List[str] = []

    def _set_delta(key: str, delta: Decimal) -> None:
        if key not in current:
            return
        new_val = current[key] + delta
        updates[key] = ensure_bounds(key, new_val, bounds)

    def _set_mul(key: str, factor: Decimal) -> None:
        if key not in current:
            return
        new_val = current[key] * factor
        updates[key] = ensure_bounds(key, new_val, bounds)

    # Rule 1: Adverse selection -> be more conservative
    adverse_ratio = None
    if metrics.adverse_edges:
        adverse, total = metrics.adverse_edges
        if total > 0:
            adverse_ratio = Decimal(adverse) / Decimal(total)
    adverse_markout = (
        metrics.markout_5s_bps is not None and metrics.markout_5s_bps < Decimal("-0.5")
    )
    if adverse_markout or (adverse_ratio is not None and adverse_ratio > Decimal("0.3")):
        _set_mul("MM_SPREAD_MULTIPLIER", Decimal("1.10"))
        _set_delta("MM_MIN_OFFSET_BPS", Decimal("0.5"))
        _set_delta("MM_MIN_SPREAD_BPS", Decimal("0.5"))
        _set_delta("MM_MIN_REPRICE_MOVE_TICKS", Decimal("1"))
        _set_delta("MM_MIN_REPRICE_EDGE_DELTA_BPS", Decimal("0.25"))
        _set_delta("MM_MIN_REPRICE_INTERVAL_S", Decimal("0.2"))
        reasons.append("Adverse selection -> widen and require larger improvement to reprice")

    # Rule 2: Too few fills with acceptable markout -> be more aggressive
    low_fills = metrics.fills == 0 or (
        metrics.fill_rate_pct is not None and metrics.fill_rate_pct < Decimal("0.5")
    )
    positive_markout = (
        metrics.markout_5s_bps is not None and metrics.markout_5s_bps >= Decimal("0")
    )
    if low_fills and positive_markout:
        _set_mul("MM_SPREAD_MULTIPLIER", Decimal("0.95"))
        _set_delta("MM_MIN_OFFSET_BPS", Decimal("-0.3"))
        _set_delta("MM_MIN_SPREAD_BPS", Decimal("-0.5"))
        reasons.append("Low fills with positive markout -> tighten quotes slightly")

    # Rule 3: High churn -> reduce cancel/replace frequency
    if metrics.cancellations_pct is not None and metrics.cancellations_pct > Decimal("97"):
        _set_delta("MM_REPRICE_TOLERANCE_PERCENT", Decimal("0.1"))
        _set_delta("MM_MIN_REPRICE_INTERVAL_S", Decimal("0.3"))
        _set_delta("MM_MIN_REPRICE_MOVE_TICKS", Decimal("1"))
        reasons.append("High cancellation rate -> stronger reprice hysteresis")

    # Rule 4: High POST_ONLY reject rate -> increase safety/backoff
    if (
        metrics.post_only_reject_pct is not None
        and metrics.post_only_reject_pct > Decimal("1.0")
    ):
        _set_delta("MM_POST_ONLY_SAFETY_TICKS", Decimal("1"))
        _set_delta("MM_POF_MAX_SAFETY_TICKS", Decimal("1"))
        _set_mul("MM_POF_BACKOFF_MULTIPLIER", Decimal("1.1"))
        reasons.append("High post-only rejects -> raise safety ticks and cooldown backoff")

    # Rule 5: Large inventory -> stronger skew response
    if max_position_size and metrics.final_position is not None:
        if abs(metrics.final_position) > max_position_size * Decimal("0.7"):
            _set_delta("MM_INVENTORY_SKEW_FACTOR", Decimal("0.05"))
            _set_delta("MM_INVENTORY_DEADBAND_PCT", Decimal("-0.02"))
            _set_delta("MM_SKEW_MAX_BPS", Decimal("2"))
            reasons.append("Large inventory -> stronger inventory skew response")

    # Ensure max_offset >= min_offset
    if "MM_MIN_OFFSET_BPS" in updates or "MM_MAX_OFFSET_BPS" in updates:
        min_offset = updates.get("MM_MIN_OFFSET_BPS", current.get("MM_MIN_OFFSET_BPS"))
        max_offset = updates.get("MM_MAX_OFFSET_BPS", current.get("MM_MAX_OFFSET_BPS"))
        if min_offset is not None and max_offset is not None and max_offset < min_offset:
            updates["MM_MAX_OFFSET_BPS"] = ensure_bounds("MM_MAX_OFFSET_BPS", min_offset + Decimal("1"), bounds)
            reasons.append("Raised max_offset_bps to keep >= min_offset_bps")

    # Ensure volatility thresholds remain ordered: calm <= elevated <= extreme
    calm = updates.get("MM_VOL_REGIME_CALM_BPS", current.get("MM_VOL_REGIME_CALM_BPS"))
    elevated = updates.get(
        "MM_VOL_REGIME_ELEVATED_BPS", current.get("MM_VOL_REGIME_ELEVATED_BPS")
    )
    extreme = updates.get(
        "MM_VOL_REGIME_EXTREME_BPS", current.get("MM_VOL_REGIME_EXTREME_BPS")
    )
    if calm is not None and elevated is not None and elevated < calm:
        updates["MM_VOL_REGIME_ELEVATED_BPS"] = ensure_bounds(
            "MM_VOL_REGIME_ELEVATED_BPS", calm, bounds
        )
        elevated = updates["MM_VOL_REGIME_ELEVATED_BPS"]
        reasons.append("Raised vol_regime_elevated_bps to keep >= calm threshold")
    if elevated is not None and extreme is not None and extreme < elevated:
        updates["MM_VOL_REGIME_EXTREME_BPS"] = ensure_bounds(
            "MM_VOL_REGIME_EXTREME_BPS", elevated, bounds
        )
        reasons.append("Raised vol_regime_extreme_bps to keep >= elevated threshold")

    # Ensure inventory bands remain ordered: warn <= critical <= hard
    warn = updates.get("MM_INVENTORY_WARN_PCT", current.get("MM_INVENTORY_WARN_PCT"))
    critical = updates.get(
        "MM_INVENTORY_CRITICAL_PCT", current.get("MM_INVENTORY_CRITICAL_PCT")
    )
    hard = updates.get("MM_INVENTORY_HARD_PCT", current.get("MM_INVENTORY_HARD_PCT"))
    if warn is not None and critical is not None and critical < warn:
        updates["MM_INVENTORY_CRITICAL_PCT"] = ensure_bounds(
            "MM_INVENTORY_CRITICAL_PCT", warn, bounds
        )
        critical = updates["MM_INVENTORY_CRITICAL_PCT"]
        reasons.append("Raised inventory_critical_pct to keep >= inventory_warn_pct")
    if critical is not None and hard is not None and hard < critical:
        updates["MM_INVENTORY_HARD_PCT"] = ensure_bounds(
            "MM_INVENTORY_HARD_PCT", critical, bounds
        )
        reasons.append("Raised inventory_hard_pct to keep >= inventory_critical_pct")

    # Remove no-op updates
    final_updates: Dict[str, Decimal] = {}
    for key, new_val in updates.items():
        old_val = current.get(key)
        if old_val is None or new_val != old_val:
            final_updates[key] = new_val
    return final_updates, reasons


def find_latest_journal(
    journal_dir: Path,
    since: Optional[float],
    market_name: Optional[str] = None,
) -> Optional[Path]:
    pattern = f"mm_{market_name}_*.jsonl" if market_name else "mm_*.jsonl"
    files = sorted(journal_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    if since is None:
        return files[-1]
    candidates = [p for p in files if p.stat().st_mtime >= since]
    return candidates[-1] if candidates else None


def start_market_maker(repo: Path, env_file: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["ENV"] = str(env_file)
    env["PYTHONPATH"] = str(repo / "src")

    # Use the same interpreter as this controller process (venv-safe).
    cmd = [sys.executable, "scripts/run_market_maker.py"]
    return subprocess.Popen(cmd, cwd=str(repo), env=env)


def stop_market_maker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_analysis(
    repo: Path,
    target: Path,
    assumed_fee_bps: str,
    market_name: Optional[str] = None,
) -> Path:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")

    cmd = [
        sys.executable,
        "scripts/analyse_mm_journal.py",
        str(target),
        "--assumed-fee-bps",
        str(assumed_fee_bps),
    ]
    subprocess.check_call(cmd, cwd=str(repo), env=env)

    if target.is_dir():
        latest = find_latest_journal(target, since=None, market_name=market_name)
        if not latest:
            raise RuntimeError("No journal files found after analysis")
        return latest.with_suffix(".analysis.txt")
    return target.with_suffix(".analysis.txt")


def to_decimal_map(env: Dict[str, str], keys: Iterable[str]) -> Dict[str, Decimal]:
    out: Dict[str, Decimal] = {}
    for key in keys:
        if key in env:
            val = _decimal(env[key])
            if val is not None:
                out[key] = val
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MM -> analyze -> tune loop.")
    parser.add_argument("--repo", default=".", help="Repo root (default: current dir)")
    parser.add_argument("--base-env", default=".env.cop", help="Base env file (relative to repo)")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    parser.add_argument(
        "--max-run-seconds",
        "--run-seconds",
        type=int,
        default=0,
        help="Max seconds per run before forced stop (0 = no limit)",
    )
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=60,
        help="Seconds between analyses while the bot is running",
    )
    parser.add_argument(
        "--min-fills",
        type=int,
        default=10,
        help="Minimum fills required to evaluate realized PnL",
    )
    parser.add_argument(
        "--market-name",
        default=None,
        help="Market symbol used to filter journals (default: MM_MARKET_NAME in env file)",
    )
    parser.add_argument("--assumed-fee-bps", default="0", help="Fee bps for analysis")
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory")
    parser.add_argument("--dry-run", action="store_true", help="Only compute next config, no execution")
    parser.add_argument("--allow-mainnet", action="store_true", help="Allow running when MM_ENVIRONMENT=mainnet")
    args = parser.parse_args()

    repo = find_repo_root(Path(args.repo))
    base_env = (repo / args.base_env).resolve() if not Path(args.base_env).is_absolute() else Path(args.base_env)
    journal_dir = (repo / args.journal_dir).resolve()

    if not base_env.exists():
        raise SystemExit(f"Base env file not found: {base_env}")

    lines = read_env_lines(base_env)
    env_map = parse_env(lines)

    mm_env = env_map.get("MM_ENVIRONMENT", "testnet")
    if mm_env == "mainnet" and not args.allow_mainnet:
        raise SystemExit(
            "MM_ENVIRONMENT=mainnet detected. Re-run with --allow-mainnet if this is intentional."
        )

    market_name = args.market_name or env_map.get("MM_MARKET_NAME")
    if not market_name:
        raise SystemExit(
            "Unable to infer market name. Set MM_MARKET_NAME in base env or pass --market-name."
        )

    max_position_size = _decimal(env_map.get("MM_MAX_POSITION_SIZE", ""))

    # Bounds can be overridden by env vars if needed
    bounds = dict(DEFAULT_BOUNDS)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_env_slug = _slugify_token(base_env.name)
    market_slug = _slugify_token(market_name)
    log_path = journal_dir / f"mm_tuning_log_{market_slug}_{base_env_slug}_{timestamp}.jsonl"

    prev_env_file: Optional[Path] = None

    for i in range(1, args.iterations + 1):
        iter_env = base_env.parent / f"{base_env.name}.iter{i:03d}"

        if prev_env_file is None:
            shutil.copyfile(base_env, iter_env)
        elif prev_env_file.resolve() == iter_env.resolve():
            # env file already prepared for this iteration
            pass
        else:
            # Start from previous iteration env
            shutil.copyfile(prev_env_file, iter_env)

        print(f"\n[Iter {i:02d}] env={iter_env}")
        if args.dry_run:
            print("  Dry-run mode: not starting market maker")
            continue

        run_started = time.time()
        proc = start_market_maker(repo, iter_env)
        print("  Market maker started. Monitoring journal...")

        last_analysis = 0.0
        stop_reason = None
        analysis_path: Optional[Path] = None
        metrics: Optional[Metrics] = None
        realized_pnl: Optional[Decimal] = None

        while True:
            now = time.time()
            if args.max_run_seconds > 0 and now - run_started >= args.max_run_seconds:
                stop_reason = "max_run_seconds_reached"
                break

            if now - last_analysis < max(5, args.analysis_interval):
                time.sleep(1)
                continue

            last_analysis = now
            latest_journal = find_latest_journal(
                journal_dir,
                since=run_started,
                market_name=market_name,
            )
            if latest_journal is None:
                continue

            analysis_path = run_analysis(
                repo,
                latest_journal,
                args.assumed_fee_bps,
                market_name=market_name,
            )
            metrics = parse_metrics(analysis_path)
            realized_pnl = compute_realized_pnl_last_n_fills(latest_journal, n=args.min_fills)
            metrics.last10_realized_pnl = realized_pnl

            if realized_pnl is None:
                print("  Waiting for enough fills to evaluate realized PnL...")
                continue

            print(f"  last{args.min_fills}_realized_pnl={realized_pnl:.4f}")
            if realized_pnl < 0:
                stop_reason = f"last{args.min_fills}_realized_pnl_negative"
                break

        stop_market_maker(proc)
        print(f"  Market maker stopped ({stop_reason}).")

        updates: Dict[str, Decimal] = {}
        reasons: List[str] = []
        if stop_reason == f"last{args.min_fills}_realized_pnl_negative" and metrics is not None:
            current_env_lines = read_env_lines(iter_env)
            current_env_map = parse_env(current_env_lines)
            current_vals = to_decimal_map(current_env_map, ALLOWED_TUNING_KEYS)
            updates, reasons = compute_updates(current_vals, metrics, bounds, max_position_size)
            reasons.insert(0, f"Last {args.min_fills} fills realized PnL={realized_pnl:.4f} < 0 -> switching config")

        entry = {
            "iteration": i,
            "market_name": market_name,
            "env_file": str(iter_env),
            "analysis_file": str(analysis_path) if analysis_path else None,
            "stop_reason": stop_reason,
            "updates": {k: str(v) for k, v in updates.items()},
            "reasons": reasons,
            "metrics": {
                "fills": metrics.fills if metrics else None,
                "fill_rate_pct": str(metrics.fill_rate_pct) if metrics and metrics.fill_rate_pct else None,
                "avg_edge_bps": str(metrics.avg_edge_bps) if metrics and metrics.avg_edge_bps else None,
                "avg_spread_at_fill_bps": str(metrics.avg_spread_at_fill_bps) if metrics and metrics.avg_spread_at_fill_bps else None,
                "avg_spread_at_place_bps": str(metrics.avg_spread_at_place_bps) if metrics and metrics.avg_spread_at_place_bps else None,
                "markout_5s_bps": str(metrics.markout_5s_bps) if metrics and metrics.markout_5s_bps else None,
                "cancellations_pct": str(metrics.cancellations_pct) if metrics and metrics.cancellations_pct else None,
                "rejection_pct": str(metrics.rejection_pct) if metrics and metrics.rejection_pct else None,
                "post_only_reject_pct": str(metrics.post_only_reject_pct) if metrics and metrics.post_only_reject_pct else None,
                "final_position": str(metrics.final_position) if metrics and metrics.final_position else None,
                "last10_realized_pnl": str(metrics.last10_realized_pnl) if metrics and metrics.last10_realized_pnl is not None else None,
            },
            "run_started": run_started,
        }
        journal_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        if updates and i < args.iterations:
            next_env = base_env.parent / f"{base_env.name}.iter{i + 1:03d}"
            shutil.copyfile(iter_env, next_env)
            next_lines = read_env_lines(next_env)
            update_strs = {k: str(v) for k, v in updates.items()}
            next_lines = update_env_lines(next_lines, update_strs)
            next_env.write_text("\n".join(next_lines) + "\n")
            prev_env_file = next_env
            for k, v in updates.items():
                print(f"  {k} -> {v}")
            for reason in reasons:
                print(f"  reason: {reason}")
        else:
            prev_env_file = iter_env

        if stop_reason == "max_run_seconds_reached":
            break

    print(f"\nDone. Log written to: {log_path}")


if __name__ == "__main__":
    main()
