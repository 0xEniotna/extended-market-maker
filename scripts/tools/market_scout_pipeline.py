#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
SRC_DIR = PROJECT_ROOT / "src"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mm_audit_common import (  # noqa: E402
    append_jsonl,
    discover_recent_markets_from_journals,
    extract_last_json_object,
    find_latest_journal,
    iso_utc,
    load_market_jobs,
    load_policy,
    now_ts,
    parse_env,
    read_env_lines,
    read_json,
    safe_decimal,
    slugify,
    update_env_lines,
    write_json,
)

CONFIG_SNAPSHOT_KEYS: List[Tuple[str, Tuple[str, ...]]] = [
    ("MM_SPREAD_MULTIPLIER", ("MM_SPREAD_MULTIPLIER", "MM_SPREAD_PERCENT", "SPREAD_MULTIPLIER", "SPREAD_PERCENT")),
    ("MM_REPRICE_TOLERANCE_PERCENT", ("MM_REPRICE_TOLERANCE_PERCENT", "REPRICE_TOLERANCE_PERCENT")),
    ("MM_ORDER_SIZE_MULTIPLIER", ("MM_ORDER_SIZE_MULTIPLIER", "ORDER_SIZE_MULTIPLIER")),
    ("MM_INVENTORY_SKEW_FACTOR", ("MM_INVENTORY_SKEW_FACTOR", "INVENTORY_SKEW_FACTOR")),
    ("MM_IMBALANCE_PAUSE_THRESHOLD", ("MM_IMBALANCE_PAUSE_THRESHOLD", "IMBALANCE_PAUSE_THRESHOLD")),
    ("MM_MAX_POSITION_SIZE", ("MM_MAX_POSITION_SIZE", "MM_MAX_POSITION")),
    ("MM_MAX_ORDER_NOTIONAL_USD", ("MM_MAX_ORDER_NOTIONAL_USD", "MM_MAX_ORDER_NOTIONAL", "MM_MAX_NOTIONAL")),
    ("MM_MAX_POSITION_NOTIONAL_USD", ("MM_MAX_POSITION_NOTIONAL_USD", "MM_MAX_POSITION_NOTIONAL", "MM_MAX_NOTIONAL")),
]

DEFAULT_PAPER_HORIZONS_MS = [250, 1000, 5000, 30000, 120000]


def _d(value: Any, default: str = "0") -> Decimal:
    parsed = safe_decimal(value)
    if parsed is None:
        return Decimal(default)
    return parsed


def _d_optional(value: Any) -> Optional[Decimal]:
    parsed = safe_decimal(value)
    return parsed if parsed is not None else None


def _clamp_decimal(value: Decimal, lo: Decimal, hi: Decimal) -> Decimal:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _parse_int_csv(raw: str, *, fallback: List[int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for part in str(raw or "").split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out or list(fallback)


def _parse_trade_types(raw: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for part in str(raw or "").split(","):
        value = part.strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out or ["TRADE"]


def _build_paper_config(args: argparse.Namespace) -> Dict[str, Decimal]:
    return {
        "min_fills": _d(args.paper_min_fills),
        "max_toxicity_250ms": _d(args.paper_max_toxicity_250ms),
        "min_markout_250ms": _d(args.paper_min_markout_250ms),
        "min_markout_1s": _d(args.paper_min_markout_1s),
        "markout_shift_bps": _d(args.paper_markout_shift_bps),
        "markout_scale_bps": _d(args.paper_markout_scale_bps, default="1"),
        "markout_max_score": _d(args.paper_markout_max_score),
        "toxicity_center": _d(args.paper_toxicity_center),
        "toxicity_scale": _d(args.paper_toxicity_scale),
        "toxicity_max_penalty": _d(args.paper_toxicity_max_penalty),
    }


def _summarize_paper_health(
    sampled_markets: List[str],
    paper_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    sampled = [str(m).strip() for m in sampled_markets if str(m).strip()]
    sampled_set = set(sampled)
    missing_stats = sorted(m for m in sampled_set if m not in paper_stats)

    total_fills = 0
    total_trade_rows_seen = 0
    total_trade_rows_used = 0
    total_bbo_updates_seen = 0
    bbo_ready_markets = 0
    for market in sampled:
        row = paper_stats.get(market, {})
        if not isinstance(row, dict):
            continue
        total_fills += int(_d(row.get("paper_fills"), default="0"))
        total_trade_rows_seen += int(_d(row.get("paper_trade_rows_seen"), default="0"))
        total_trade_rows_used += int(_d(row.get("paper_trade_rows_used"), default="0"))
        total_bbo_updates_seen += int(_d(row.get("paper_bbo_updates_seen"), default="0"))
        if bool(row.get("paper_bbo_ready")):
            bbo_ready_markets += 1

    degraded = (
        bool(sampled)
        and total_fills == 0
        and (total_trade_rows_seen > 0 or total_bbo_updates_seen > 0)
    )
    return {
        "sampled_market_count": len(sampled),
        "sampled_markets_without_stats": missing_stats,
        "total_paper_fills": total_fills,
        "total_trade_rows_seen": total_trade_rows_seen,
        "total_trade_rows_used": total_trade_rows_used,
        "total_bbo_updates_seen": total_bbo_updates_seen,
        "markets_with_bbo_ready": bbo_ready_markets,
        "degraded": degraded,
    }


def _extract_markets_payload(stdout: str, label: str) -> Dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {"markets": [], "count": 0, "sampling": {}, "warning": f"{label}_empty_stdout"}

    # Fast-path: pure JSON stdout.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("markets"), list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Mixed stdout (human text + JSON): decode the first JSON object that has `markets`.
    decoder = json.JSONDecoder()
    brace_positions = [i for i, ch in enumerate(text) if ch == "{"]
    for pos in brace_positions:
        try:
            candidate, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and isinstance(candidate.get("markets"), list):
            return candidate

    # Fall back to the old heuristic for compatibility.
    try:
        candidate = extract_last_json_object(text)
    except RuntimeError:
        return {
            "markets": [],
            "count": 0,
            "sampling": {},
            "warning": f"{label}_json_parse_failed",
        }
    if isinstance(candidate, dict) and isinstance(candidate.get("markets"), list):
        return candidate
    return {
        "markets": [],
        "count": 0,
        "sampling": {},
        "warning": f"{label}_invalid_payload_shape",
        "payload_keys": sorted(candidate.keys()) if isinstance(candidate, dict) else [],
    }


def _run_json_stdout(
    cmd: List[str],
    label: str,
    run_env: Optional[Dict[str, str]] = None,
    timeout_s: float = 300.0,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    timeout_s = max(1.0, float(timeout_s))
    cmd_capture: Dict[str, Any] = {
        "label": label,
        "command": cmd,
        "command_text": " ".join(str(part) for part in cmd),
        "timeout_s": timeout_s,
    }
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, env=run_env)
    except subprocess.TimeoutExpired as exc:
        cmd_capture.update({
            "timed_out": True,
            "timeout_s": float(exc.timeout) if exc.timeout is not None else timeout_s,
            "returncode": None,
            "stdout": (exc.stdout if isinstance(exc.stdout, str) else "") or "",
            "stderr": (exc.stderr if isinstance(exc.stderr, str) else "") or "",
        })
        raise RuntimeError(f"{label} timed out after {exc.timeout}s") from exc
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    cmd_capture.update({
        "timed_out": False,
        "returncode": int(proc.returncode),
        "stdout": stdout,
        "stderr": stderr,
    })
    if proc.returncode != 0:
        out = (stdout + "\n" + stderr).strip()
        if (
            "No successful market snapshots collected." in out
            or "No active markets found." in out
        ):
            return {"markets": [], "count": 0, "sampling": {}, "warning": out}, cmd_capture
        raise RuntimeError(
            f"{label} failed (code={proc.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    return _extract_markets_payload(stdout, label), cmd_capture


def _resolve_env_path(repo_root: Path, raw: str) -> Optional[Path]:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    return path if path.exists() and path.is_file() else None


def _build_market_env_index(repo_root: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for env_path in sorted(repo_root.glob(".env*")):
        if not env_path.is_file():
            continue
        name = env_path.name
        if name in {".env.example", ".env.sample", ".env.template"}:
            continue
        if name.endswith(".candidate"):
            continue
        try:
            env_map = parse_env(read_env_lines(env_path))
        except Exception:
            continue
        market = str(env_map.get("MM_MARKET_NAME") or "").strip()
        if not market:
            continue
        existing = out.get(market)
        if existing is None:
            out[market] = str(env_path)
            continue
        # Prefer explicit market env files over plain ".env".
        if Path(existing).name == ".env" and name != ".env":
            out[market] = str(env_path)
    return out


def _build_subprocess_env(default_env_path: Optional[Path]) -> Dict[str, str]:
    env = dict(os.environ)
    if default_env_path is None:
        return env
    try:
        overrides = parse_env(read_env_lines(default_env_path))
    except Exception:
        return env
    env.update({k: v for k, v in overrides.items() if isinstance(k, str)})
    env["ENV"] = str(default_env_path)
    return env


def _extract_config_snapshot(env_path: Optional[str]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "available": False,
        "env_path": env_path,
        "env_hash": None,
        "values": {},
        "source_keys": {},
        "error": None,
    }
    if not env_path:
        snapshot["error"] = "env_path_missing"
        return snapshot

    env_file = Path(env_path).expanduser()
    if not env_file.exists() or not env_file.is_file():
        snapshot["error"] = "env_path_not_found"
        return snapshot

    try:
        raw_text = env_file.read_text()
        env_map = parse_env(raw_text.splitlines())
    except Exception as exc:  # pragma: no cover - best effort resilience
        snapshot["error"] = f"env_read_failed:{type(exc).__name__}"
        return snapshot

    values: Dict[str, str] = {}
    source_keys: Dict[str, str] = {}
    for canonical_key, aliases in CONFIG_SNAPSHOT_KEYS:
        for alias in aliases:
            raw = env_map.get(alias)
            value = str(raw).strip() if raw is not None else ""
            if not value:
                continue
            values[canonical_key] = value
            source_keys[canonical_key] = alias
            break

    for key in sorted(env_map):
        if not key.startswith("MM_TOXICITY_"):
            continue
        value = str(env_map.get(key) or "").strip()
        if not value:
            continue
        values[key] = value
        source_keys[key] = key

    snapshot.update(
        {
            "available": True,
            "env_path": str(env_file.resolve()),
            "env_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16],
            "values": values,
            "source_keys": source_keys,
        }
    )
    return snapshot


def _run_analysis_json(
    *,
    repo_root: Path,
    journal: Path,
    output_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "analyse_mm_journal.py"),
        str(journal),
        "--json-out",
        str(output_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "analysis_timeout"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or "analysis_failed"
    payload = read_json(output_path, default={})
    if not isinstance(payload, dict):
        return None, "analysis_json_invalid"
    return payload, None


def _run_pnl_json(
    *,
    repo_root: Path,
    market: str,
    env_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = [
        sys.executable,
        "-m", "market_maker.cli",
        "pnl", market,
        "--days", "1",
        "--max-pages", "30",
        "--json",
    ]
    if env_path:
        cmd.extend(["--env", env_path])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, "fetch_pnl_timeout"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or "fetch_pnl_failed"
    try:
        payload = extract_last_json_object(proc.stdout)
    except Exception as exc:
        return None, f"fetch_pnl_json_parse_failed: {exc}"
    return payload, None


def _build_candidate_pool(
    *,
    find_payload: Dict[str, Any],
    screen_payload: Dict[str, Any],
    policy: Dict[str, Any],
    paper_enabled: bool = False,
    paper_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    paper_cfg: Optional[Dict[str, Decimal]] = None,
    paper_sampled_markets: Optional[set[str]] = None,
) -> List[Dict[str, Any]]:
    gates = policy.get("candidate_gates", {}) if isinstance(policy, dict) else {}
    min_score = _d(gates.get("min_score", "0"))
    min_ticks = _d(gates.get("min_ticks_in_spread", "0"))
    min_cov = _d(gates.get("min_coverage_pct", "0"))
    max_p90 = _d(gates.get("max_spread_p90_bps", "0"))
    min_daily_volume = _d(gates.get("min_daily_volume", "0"))
    paper_stats = paper_stats or {}
    paper_cfg = paper_cfg or {}
    sampled_markets = {
        str(m).strip()
        for m in (paper_sampled_markets if paper_sampled_markets is not None else set(paper_stats.keys()))
        if str(m).strip()
    }

    min_fills = paper_cfg.get("min_fills", Decimal("20"))
    max_toxicity_250ms = paper_cfg.get("max_toxicity_250ms", Decimal("0.60"))
    min_markout_250ms = paper_cfg.get("min_markout_250ms", Decimal("-1"))
    min_markout_1s = paper_cfg.get("min_markout_1s", Decimal("-1"))
    markout_shift_bps = paper_cfg.get("markout_shift_bps", Decimal("2.0"))
    markout_scale_bps = paper_cfg.get("markout_scale_bps", Decimal("2.0"))
    if markout_scale_bps <= 0:
        markout_scale_bps = Decimal("1")
    markout_max_score = paper_cfg.get("markout_max_score", Decimal("3.0"))
    toxicity_center = paper_cfg.get("toxicity_center", Decimal("0.5"))
    toxicity_scale = paper_cfg.get("toxicity_scale", Decimal("6.0"))
    if toxicity_scale < 0:
        toxicity_scale = Decimal("0")
    toxicity_max_penalty = paper_cfg.get("toxicity_max_penalty", Decimal("3.0"))

    find_names = {
        str(row.get("name"))
        for row in (find_payload.get("markets") or [])
        if isinstance(row, dict) and row.get("name")
    }

    out: List[Dict[str, Any]] = []
    for row in (screen_payload.get("markets") or []):
        if not isinstance(row, dict):
            continue
        market = str(row.get("name") or "").strip()
        if not market:
            continue
        in_find = market in find_names

        score = _d(row.get("score"))
        ticks = _d(row.get("ticks_in_spread"))
        coverage = _d(row.get("coverage_3bps_pct"))
        spread_p90 = _d(row.get("spread_bps_p90"))
        daily_vol = _d(row.get("daily_vol"))

        gate_flags = {
            "passes_find_filter": in_find,
            "score": score >= min_score,
            "ticks_in_spread": ticks >= min_ticks,
            "coverage_pct": coverage >= min_cov,
            "spread_p90": (max_p90 <= 0) or (spread_p90 <= max_p90),
            "daily_volume": daily_vol >= min_daily_volume,
        }
        stage_a_pass = all(gate_flags.values())

        result_row: Dict[str, Any] = {
            "market": market,
            "score": score,
            "spread_bps": _d(row.get("spread_bps")),
            "spread_p90_bps": spread_p90,
            "coverage_pct": coverage,
            "ticks_in_spread": ticks,
            "daily_volume": daily_vol,
            "open_interest": _d(row.get("oi")),
            "in_find_pool": in_find,
            "gate_flags": gate_flags,
        }

        if not paper_enabled:
            result_row["passes_all_gates"] = stage_a_pass
            out.append(result_row)
            continue

        result_row["stageA_pass"] = stage_a_pass
        paper_sampled = market in sampled_markets
        result_row["paper_sampled"] = paper_sampled
        if not paper_sampled:
            result_row.update({
                "stageB_pass": None,
                "paper_gate_flags": None,
                "passes_all_gates": False,
                "base_score": score,
                "markout_score": None,
                "toxicity_penalty": None,
                "score2": score.quantize(Decimal("0.01")),
                "paper_fills": None,
                "paper_fill_rate_per_min": None,
                "paper_fill_rate_per_min_adjusted": None,
                "paper_markout_250ms_bps": None,
                "paper_markout_1s_bps": None,
                "paper_toxicity_250ms": None,
                "paper_toxicity_1s": None,
                "paper_trade_rows_seen": None,
                "paper_trade_rows_used": None,
                "paper_bbo_updates_seen": None,
                "paper_bbo_ready": None,
                "paper_data_quality_warnings": [],
            })
            out.append(result_row)
            continue

        paper_row = paper_stats.get(market, {}) if isinstance(paper_stats.get(market), dict) else {}
        paper_fills = int(_d(paper_row.get("paper_fills"), default="0"))
        paper_fill_rate = _d_optional(paper_row.get("paper_fill_rate_per_min"))
        paper_fill_rate_adj = _d_optional(paper_row.get("paper_fill_rate_per_min_adjusted"))
        paper_markout_250ms = _d_optional(paper_row.get("paper_markout_bps_250ms_mean"))
        paper_markout_1s = _d_optional(paper_row.get("paper_markout_bps_1s_mean"))
        paper_toxicity_250ms = _d_optional(paper_row.get("paper_toxicity_share_250ms"))
        paper_toxicity_1s = _d_optional(paper_row.get("paper_toxicity_share_1s"))

        stage_b_flags = {
            "paper_fills": Decimal(paper_fills) >= min_fills,
            "toxicity_250ms": (
                paper_toxicity_250ms is not None and paper_toxicity_250ms <= max_toxicity_250ms
            ),
            "markout_250ms": (
                paper_markout_250ms is not None and paper_markout_250ms >= min_markout_250ms
            ),
            "markout_1s": (
                paper_markout_1s is not None and paper_markout_1s >= min_markout_1s
            ),
        }
        stage_b_pass = all(stage_b_flags.values())
        passes = stage_a_pass and stage_b_pass

        markout_score = Decimal("0")
        if paper_markout_1s is not None:
            raw_markout_score = (paper_markout_1s + markout_shift_bps) / markout_scale_bps
            markout_score = _clamp_decimal(raw_markout_score, Decimal("0"), markout_max_score)

        toxicity_penalty = Decimal("0")
        if paper_toxicity_250ms is not None:
            raw_penalty = (paper_toxicity_250ms - toxicity_center) * toxicity_scale
            toxicity_penalty = _clamp_decimal(raw_penalty, Decimal("0"), toxicity_max_penalty)

        score2 = (score + markout_score - toxicity_penalty).quantize(Decimal("0.01"))

        result_row.update({
            "stageB_pass": stage_b_pass,
            "paper_gate_flags": stage_b_flags,
            "passes_all_gates": passes,
            "base_score": score,
            "markout_score": markout_score.quantize(Decimal("0.01")),
            "toxicity_penalty": toxicity_penalty.quantize(Decimal("0.01")),
            "score2": score2,
            "paper_fills": paper_fills,
            "paper_fill_rate_per_min": paper_fill_rate,
            "paper_fill_rate_per_min_adjusted": paper_fill_rate_adj,
            "paper_markout_250ms_bps": paper_markout_250ms,
            "paper_markout_1s_bps": paper_markout_1s,
            "paper_toxicity_250ms": paper_toxicity_250ms,
            "paper_toxicity_1s": paper_toxicity_1s,
            "paper_trade_rows_seen": _d_optional(paper_row.get("paper_trade_rows_seen")),
            "paper_trade_rows_used": _d_optional(paper_row.get("paper_trade_rows_used")),
            "paper_bbo_updates_seen": _d_optional(paper_row.get("paper_bbo_updates_seen")),
            "paper_bbo_ready": paper_row.get("paper_bbo_ready"),
            "paper_data_quality_warnings": paper_row.get("data_quality_warnings", []),
        })
        out.append(result_row)

    if paper_enabled:
        out.sort(
            key=lambda item: (item["passes_all_gates"], _d(item.get("score2"), default="0")),
            reverse=True,
        )
    else:
        out.sort(key=lambda item: (item["passes_all_gates"], item["score"]), reverse=True)
    return out


def _build_methodology(
    *,
    policy: Dict[str, Any],
    find_duration_s: float,
    screen_duration_s: float,
    scan_interval_s: float,
    paper_meta: Optional[Dict[str, Any]] = None,
    paper_scoring_enabled: bool = True,
) -> Dict[str, Any]:
    gates = policy.get("candidate_gates", {}) if isinstance(policy, dict) else {}
    base = {
        "scoring": {
            "source": "scripts/screen_mm_markets.py::score_market",
            "score_formula": "score = spread_score + tick_score + vol_score + oi_score + cov_score",
            "hard_liquidity_floor": "if daily_vol < 1000 then score = -1",
            "components": {
                "spread_score": {
                    "description": "Rewards median spread in a target range, penalizes too-tight and too-wide spreads.",
                    "rule": (
                        "if spread<0.5 => 0; elif spread<=50 => min(spread/5, 5); "
                        "else => max(0, 5 - (spread-50)/50); "
                        "if spread_p90 > spread*2.5 then spread_score *= 0.8"
                    ),
                    "max_points": 5,
                },
                "tick_score": {
                    "rule": "min(ticks_in_spread / 2, 3)",
                    "max_points": 3,
                },
                "vol_score": {
                    "rule": "min(log10(max(daily_vol, 1)) - 3, 4)",
                    "max_points": 4,
                },
                "oi_score": {
                    "rule": "min(log10(max(open_interest, 1)) - 3, 3)",
                    "max_points": 3,
                },
                "cov_score": {
                    "rule": "clamp(coverage_3bps_pct / 25, 0, 4)",
                    "max_points": 4,
                },
            },
        },
        "candidate_filtering": {
            "find_stage_source": "scripts/tools/find_mm_markets.py",
            "find_stage_sampling": {
                "duration_s": find_duration_s,
                "interval_s": scan_interval_s,
            },
            "find_stage_filters": {
                "min_spread_bps": gates.get("min_spread_bps", 6),
                "max_spread_bps": gates.get("max_spread_bps", 40),
                "min_coverage_pct": gates.get("min_coverage_pct", 85),
                "min_daily_volume": gates.get("min_daily_volume", 200000),
            },
            "screen_stage_source": "scripts/screen_mm_markets.py",
            "screen_stage_sampling": {
                "duration_s": screen_duration_s,
                "interval_s": scan_interval_s,
            },
            "pipeline_gate_source": "scripts/tools/market_scout_pipeline.py::_build_candidate_pool",
            "pipeline_gates": {
                "passes_find_filter": "market must exist in find_mm_markets output",
                "min_score": gates.get("min_score", 0),
                "min_ticks_in_spread": gates.get("min_ticks_in_spread", 0),
                "min_coverage_pct": gates.get("min_coverage_pct", 0),
                "max_spread_p90_bps": gates.get("max_spread_p90_bps", 0),
                "min_daily_volume": gates.get("min_daily_volume", 0),
                "pass_rule": "all gate flags must be true",
            },
            "ranking_rule": "Candidates sorted by (passes_all_gates desc, score desc).",
        },
    }
    if not (isinstance(paper_meta, dict) and paper_meta.get("enabled")):
        return base

    scoring = base.get("scoring", {})
    if isinstance(scoring, dict):
        scoring["paper_scoring_enabled"] = bool(paper_scoring_enabled)
        if paper_scoring_enabled:
            scoring["score2_formula"] = "score2 = base_score + markout_score - toxicity_penalty"
            scoring["markout_score_rule"] = (
                "clamp((paper_markout_1s_bps + markout_shift_bps) / markout_scale_bps, 0, markout_max_score)"
            )
            scoring["toxicity_penalty_rule"] = (
                "clamp((paper_toxicity_share_250ms - toxicity_center) * toxicity_scale, 0, toxicity_max_penalty)"
            )
        scoring["paper_weight_params"] = paper_meta.get("score_weights", {})

    filtering = base.get("candidate_filtering", {})
    if isinstance(filtering, dict):
        pipeline_gates = filtering.get("pipeline_gates")
        if isinstance(pipeline_gates, dict):
            pipeline_gates["stage_b_pass_rule"] = "paper_fills + toxicity + markout thresholds must all pass"
        filtering["paper_stage"] = {
            "enabled": True,
            "source": "src/market_maker/scout/paper_markout.py::run_paper_markout",
            "duration_s": paper_meta.get("duration_s"),
            "top_k": paper_meta.get("top_k"),
            "horizons_ms": paper_meta.get("horizons_ms"),
            "queue_capture": paper_meta.get("queue_capture"),
            "bbo_match_mode": paper_meta.get("bbo_match_mode"),
            "include_trade_types": paper_meta.get("include_trade_types"),
            "warmup_s": paper_meta.get("warmup_s"),
            "max_trade_lag_ms": paper_meta.get("max_trade_lag_ms"),
            "fallback_mode": paper_meta.get("fallback_mode"),
            "degraded": paper_meta.get("degraded"),
            "fallback_applied": paper_meta.get("fallback_applied"),
            "gates": paper_meta.get("gates", {}),
            "notes": [
                "Paper fills inferred from taker side and current best bid/ask only.",
                "queue_capture adjusts fill-rate expectation and does not scale markouts.",
                "Stream-level seq gaps trigger reconnect; row-level per-market seq checks reset market runtime state.",
            ],
        }
        if paper_scoring_enabled:
            filtering["ranking_rule"] = "Candidates sorted by (passes_all_gates desc, score2 desc)."
        else:
            filtering["ranking_rule"] = "Candidates sorted by (passes_all_gates desc, score desc)."
    return base


def _load_state(path: Path) -> Dict[str, Any]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("underperformance_streaks", {})
    payload.setdefault("proposed_launch_history", [])
    return payload


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    write_json(path, state)


def _scale_cap_value(value: str, multiplier: Decimal) -> str:
    dec = safe_decimal(value)
    if dec is None:
        return value
    scaled = dec * multiplier
    # Keep deterministic concise formatting.
    text = format(scaled.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _prepare_draft_env(
    *,
    template_env: Path,
    output_path: Path,
    market: str,
    launch_size_multiplier: Decimal,
    warmup_hours: int,
) -> None:
    if not template_env.exists():
        raise FileNotFoundError(f"Template env not found: {template_env}")

    lines = read_env_lines(template_env)
    env_map = parse_env(lines)
    updates: Dict[str, str] = {
        "MM_MARKET_NAME": market,
        "MM_SCOUT_CANDIDATE": "true",
        "MM_SCOUT_WARMUP_HOURS": str(warmup_hours),
    }

    for key in ("MM_MAX_POSITION_SIZE", "MM_MAX_POSITION_NOTIONAL_USD", "MM_MAX_ORDER_NOTIONAL_USD"):
        if key in env_map:
            updates[key] = _scale_cap_value(env_map[key], launch_size_multiplier)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines = update_env_lines(lines, updates)
    output_path.write_text("\n".join(output_lines) + "\n")


def _underperformance_flags(
    *,
    pnl_24h_usd: Optional[Decimal],
    markout_5s_bps: Optional[Decimal],
    fill_rate_pct: Optional[Decimal],
    policy: Dict[str, Any],
) -> Dict[str, Optional[bool]]:
    cfg = policy.get("underperformance", {}) if isinstance(policy, dict) else {}
    pnl_lt = _d(cfg.get("pnl_24h_lt", "0"))
    mo5_lte = _d(cfg.get("markout_5s_lte", "0"))
    fill_lt = _d(cfg.get("fill_rate_pct_lt", "0"))

    flags: Dict[str, Optional[bool]] = {
        "pnl_24h_lt": None,
        "markout_5s_lte": None,
        "fill_rate_pct_lt": None,
    }
    if pnl_24h_usd is not None:
        flags["pnl_24h_lt"] = pnl_24h_usd < pnl_lt
    if markout_5s_bps is not None:
        flags["markout_5s_lte"] = markout_5s_bps <= mo5_lte
    if fill_rate_pct is not None:
        flags["fill_rate_pct_lt"] = fill_rate_pct < fill_lt
    return flags


def _all_flags_true(flags: Dict[str, Optional[bool]]) -> bool:
    return all(v is True for v in flags.values())


def _build_launch_action(
    *,
    market_row: Dict[str, Any],
    created_ts: float,
    expires_ts: float,
    draft_env_path: Path,
    repo_root: Path,
    warmup_hours: int,
    launch_size_multiplier: Decimal,
) -> Dict[str, Any]:
    market = str(market_row["market"])
    slug = slugify(market.lower())
    action_id = f"launch_{slug}_{int(created_ts)}"
    env_target = repo_root / f".env.{slug}"

    commands = [
        f"cp {draft_env_path} {env_target}",
        f"cd {repo_root} && .venv/bin/mmctl start {env_target.name}",
        (
            "# Add OpenClaw cron job for this market (channel routing per your Discord setup):\n"
            f"# openclaw cron add --id mm-{slug} --agent main --schedule 'every 15m' "
            "--channel discord --to 'channel:<MM_CHANNEL_ID>'"
        ),
    ]

    return {
        "action_id": action_id,
        "created_at": iso_utc(created_ts),
        "expires_at": iso_utc(expires_ts),
        "action_type": "launch",
        "market": market,
        "reason_codes": ["candidate_passed_gates", "market_not_live", "rate_limit_available"],
        "evidence": {
            "score": market_row.get("score"),
            "score2": market_row.get("score2"),
            "spread_bps": market_row.get("spread_bps"),
            "spread_p90_bps": market_row.get("spread_p90_bps"),
            "coverage_pct": market_row.get("coverage_pct"),
            "ticks_in_spread": market_row.get("ticks_in_spread"),
            "daily_volume": market_row.get("daily_volume"),
            "gate_flags": market_row.get("gate_flags"),
            "draft_env": str(draft_env_path),
        },
        "risk_profile": {
            "launch_size_multiplier": launch_size_multiplier,
            "warmup_hours": warmup_hours,
            "authority": "recommend_only",
        },
        "commands": commands,
        "expected_evidence": {
            "env_file_exists": str(env_target),
            "cron_market": market,
            "journal_pattern": f"mm_{market}_*.jsonl",
            "journal_run_start_within_s": 1800,
        },
    }


def _build_rotate_action(
    *,
    from_market: str,
    from_score: Decimal,
    streak: int,
    candidate_row: Dict[str, Any],
    created_ts: float,
    expires_ts: float,
    draft_env_path: Path,
    repo_root: Path,
    warmup_hours: int,
    launch_size_multiplier: Decimal,
) -> Dict[str, Any]:
    to_market = str(candidate_row["market"])
    slug = slugify(to_market.lower())
    action_id = f"rotate_{slug}_{int(created_ts)}"
    env_target = repo_root / f".env.{slug}"

    commands = [
        f"# Stop underperforming market {from_market} safely first",
        "# mmctl stop <MARKET>",
        f"cp {draft_env_path} {env_target}",
        f"cd {repo_root} && .venv/bin/mmctl start {env_target.name}",
        "# Update cron mapping: disable old market job, add replacement market job",
    ]

    return {
        "action_id": action_id,
        "created_at": iso_utc(created_ts),
        "expires_at": iso_utc(expires_ts),
        "action_type": "rotate",
        "market": to_market,
        "reason_codes": ["sustained_underperformance", "replacement_score_delta"],
        "evidence": {
            "from_market": from_market,
            "from_market_score": from_score,
            "underperformance_streak": streak,
            "replacement_score": candidate_row.get("score"),
            "replacement_score2": candidate_row.get("score2"),
            "score_delta": _d(candidate_row.get("score2") or candidate_row.get("score")) - from_score,
            "replacement_gate_flags": candidate_row.get("gate_flags"),
            "draft_env": str(draft_env_path),
        },
        "risk_profile": {
            "launch_size_multiplier": launch_size_multiplier,
            "warmup_hours": warmup_hours,
            "authority": "recommend_only",
        },
        "commands": commands,
        "expected_evidence": {
            "env_file_exists": str(env_target),
            "cron_market": to_market,
            "journal_pattern": f"mm_{to_market}_*.jsonl",
            "journal_run_start_within_s": 1800,
        },
    }


def _render_markdown(report: Dict[str, Any], actions: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Market Scout Report")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")

    quality = report.get("data_quality", {})
    issues = quality.get("issues") or []
    lines.append("## Data Quality")
    lines.append(f"- ok: {quality.get('ok', False)}")
    if quality.get("default_env"):
        lines.append(f"- default_env: `{quality.get('default_env')}`")
    if issues:
        for issue in issues:
            lines.append(f"- issue: `{issue}`")
    else:
        lines.append("- issue: none")
    lines.append("")

    methodology = report.get("methodology", {}) if isinstance(report, dict) else {}
    scoring = methodology.get("scoring", {}) if isinstance(methodology, dict) else {}
    filtering = methodology.get("candidate_filtering", {}) if isinstance(methodology, dict) else {}
    paper_meta = report.get("paper_markout") if isinstance(report, dict) else None
    paper_enabled = isinstance(paper_meta, dict) and bool(paper_meta.get("enabled"))
    paper_scoring_enabled = paper_enabled and bool(paper_meta.get("scoring_enabled", True))
    lines.append("## Methodology")
    lines.append("### Scoring")
    lines.append(f"- source: `{scoring.get('source', 'n/a')}`")
    lines.append(f"- formula: `{scoring.get('score_formula', 'n/a')}`")
    lines.append(f"- liquidity_floor: `{scoring.get('hard_liquidity_floor', 'n/a')}`")
    if paper_enabled and paper_scoring_enabled:
        lines.append(f"- score2_formula: `{scoring.get('score2_formula', 'n/a')}`")
        lines.append(f"- markout_score_rule: `{scoring.get('markout_score_rule', 'n/a')}`")
        lines.append(f"- toxicity_penalty_rule: `{scoring.get('toxicity_penalty_rule', 'n/a')}`")
    components = scoring.get("components") if isinstance(scoring, dict) else None
    if isinstance(components, dict) and components:
        lines.append("| Component | Rule | Max Points |")
        lines.append("|---|---|---:|")
        for name in ("spread_score", "tick_score", "vol_score", "oi_score", "cov_score"):
            row = components.get(name)
            if not isinstance(row, dict):
                continue
            lines.append(
                "| {name} | {rule} | {max_points} |".format(
                    name=name,
                    rule=row.get("rule", "n/a"),
                    max_points=row.get("max_points", "n/a"),
                )
            )
    else:
        lines.append("- components: n/a")
    lines.append("")

    lines.append("### Candidate Filtering")
    lines.append(f"- find_stage_source: `{filtering.get('find_stage_source', 'n/a')}`")
    lines.append(f"- screen_stage_source: `{filtering.get('screen_stage_source', 'n/a')}`")
    find_filters = filtering.get("find_stage_filters") if isinstance(filtering, dict) else None
    pipeline_gates = filtering.get("pipeline_gates") if isinstance(filtering, dict) else None
    lines.append("- find_stage_filters:")
    if isinstance(find_filters, dict):
        for key, value in find_filters.items():
            lines.append(f"  - `{key}` = `{value}`")
    else:
        lines.append("  - n/a")
    lines.append("- pipeline_gates:")
    if isinstance(pipeline_gates, dict):
        for key, value in pipeline_gates.items():
            lines.append(f"  - `{key}` = `{value}`")
    else:
        lines.append("  - n/a")
    lines.append(f"- ranking_rule: `{filtering.get('ranking_rule', 'n/a')}`")
    if paper_enabled:
        paper_stage = filtering.get("paper_stage") if isinstance(filtering, dict) else None
        lines.append("- paper_stage:")
        if isinstance(paper_stage, dict):
            for key in (
                "source",
                "duration_s",
                "top_k",
                "horizons_ms",
                "queue_capture",
                "bbo_match_mode",
                "include_trade_types",
            ):
                lines.append(f"  - `{key}` = `{paper_stage.get(key, 'n/a')}`")
        else:
            lines.append("  - n/a")
    lines.append("")

    limits = report.get("rate_limits", {})
    lines.append("## Rate Limits")
    if isinstance(limits, dict) and limits.get("max_new_per_day") is not None:
        lines.append(
            f"- proposed_launches_last24h: {limits.get('proposed_launches_last24h', 0)}"
        )
        lines.append(f"- max_new_per_day: {limits.get('max_new_per_day')}")
    else:
        lines.append("- daily_launch_cap: disabled")
    lines.append(f"- max_new_per_run: {limits.get('max_new_per_run')}")
    lines.append(f"- launch_slots_this_run: {limits.get('launch_slots_this_run')}")
    lines.append("")

    lines.append("## Active Markets")
    active = report.get("active_markets", [])
    if not active:
        lines.append("- none detected from cron jobs")
    else:
        lines.append("| Market | Streak | PnL 24h | Markout +5s | Fill Rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in active:
            lines.append(
                "| {market} | {streak} | {pnl} | {mo5} | {fill_rate} |".format(
                    market=row.get("market"),
                    streak=row.get("underperformance_streak", 0),
                    pnl=row.get("pnl_24h_usd", "n/a"),
                    mo5=row.get("markout_5s_bps", "n/a"),
                    fill_rate=row.get("fill_rate_pct", "n/a"),
                )
            )
    lines.append("")

    lines.append("## Active Config Snapshot")
    if not active:
        lines.append("- none")
    else:
        lines.append("| Market | Spread | Reprice Tol | Order Size | Inv Skew | Imbalance Pause | Max Pos Notional | Toxicity Keys |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for row in active:
            snap = row.get("config_snapshot")
            values: Dict[str, Any] = {}
            if isinstance(snap, dict):
                maybe_values = snap.get("values")
                if isinstance(maybe_values, dict):
                    values = maybe_values

            toxicity_keys = ", ".join(sorted(k for k in values if k.startswith("MM_TOXICITY_"))) or "-"
            lines.append(
                "| {market} | {spread} | {reprice} | {order_sz} | {skew} | {pause} | {max_pos} | {tox} |".format(
                    market=row.get("market"),
                    spread=values.get("MM_SPREAD_MULTIPLIER", "n/a"),
                    reprice=values.get("MM_REPRICE_TOLERANCE_PERCENT", "n/a"),
                    order_sz=values.get("MM_ORDER_SIZE_MULTIPLIER", "n/a"),
                    skew=values.get("MM_INVENTORY_SKEW_FACTOR", "n/a"),
                    pause=values.get("MM_IMBALANCE_PAUSE_THRESHOLD", "n/a"),
                    max_pos=values.get("MM_MAX_POSITION_NOTIONAL_USD", "n/a"),
                    tox=toxicity_keys,
                )
            )
    lines.append("")

    lines.append("## Top Candidates")
    candidates = report.get("candidate_markets", [])[:12]
    if not candidates:
        lines.append("- no candidates")
    else:
        if paper_enabled and paper_scoring_enabled:
            lines.append(
                "| Market | Score | Score2 | Spread | P90 | Coverage | Ticks | StageA | StageB | Passes | paper_fills | paper_fill_rate/min | markout_250ms | markout_1s | toxicity_250ms | toxicity_1s |"
            )
            lines.append(
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            )
            for row in candidates:
                lines.append(
                    "| {market} | {score} | {score2} | {spread} | {p90} | {cov} | {ticks} | {stage_a} | {stage_b} | {passes} | {fills} | {fill_rate} | {mo250} | {mo1s} | {tox250} | {tox1s} |".format(
                        market=row.get("market"),
                        score=row.get("score"),
                        score2=row.get("score2"),
                        spread=row.get("spread_bps"),
                        p90=row.get("spread_p90_bps"),
                        cov=row.get("coverage_pct"),
                        ticks=row.get("ticks_in_spread"),
                        stage_a=row.get("stageA_pass"),
                        stage_b=row.get("stageB_pass"),
                        passes=row.get("passes_all_gates"),
                        fills=row.get("paper_fills"),
                        fill_rate=row.get("paper_fill_rate_per_min"),
                        mo250=row.get("paper_markout_250ms_bps"),
                        mo1s=row.get("paper_markout_1s_bps"),
                        tox250=row.get("paper_toxicity_250ms"),
                        tox1s=row.get("paper_toxicity_1s"),
                    )
                )
        else:
            lines.append("| Market | Score | Spread | P90 | Coverage | Ticks | Passes |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in candidates:
                lines.append(
                    "| {market} | {score} | {spread} | {p90} | {cov} | {ticks} | {passes} |".format(
                        market=row.get("market"),
                        score=row.get("score"),
                        spread=row.get("spread_bps"),
                        p90=row.get("spread_p90_bps"),
                        cov=row.get("coverage_pct"),
                        ticks=row.get("ticks_in_spread"),
                        passes=row.get("passes_all_gates"),
                    )
                )
    lines.append("")

    if paper_enabled:
        lines.append("## Paper Markout (Stage B)")
        lines.append("- model: infer maker fills from taker-side public trades against current BBO.")
        lines.append("- default trade types: TRADE only (liquidation/deleverage excluded unless configured).")
        lines.append("- queue_capture: adjusts expected fill rate only; markout values are not scaled.")
        if isinstance(paper_meta, dict):
            lines.append(f"- sampled_markets: `{paper_meta.get('sampled_markets', [])}`")
            lines.append(f"- target_markets: `{paper_meta.get('target_markets', [])}`")
            lines.append(f"- duration_s: `{paper_meta.get('duration_s', 'n/a')}`")
            lines.append(f"- horizons_ms: `{paper_meta.get('horizons_ms', 'n/a')}`")
            lines.append(f"- degraded: `{paper_meta.get('degraded', False)}`")
            if paper_meta.get("fallback_applied"):
                lines.append("- fallback: Stage A gating/ranking applied for this run")
            health_summary = paper_meta.get("health_summary")
            if isinstance(health_summary, dict):
                lines.append(
                    "- health: fills={fills}, trades_seen={trades}, trades_used={used}, bbo_updates={bbo}, bbo_ready_markets={ready}/{total}".format(
                        fills=health_summary.get("total_paper_fills"),
                        trades=health_summary.get("total_trade_rows_seen"),
                        used=health_summary.get("total_trade_rows_used"),
                        bbo=health_summary.get("total_bbo_updates_seen"),
                        ready=health_summary.get("markets_with_bbo_ready"),
                        total=health_summary.get("sampled_market_count"),
                    )
                )
            warnings = paper_meta.get("warnings") or []
            if warnings:
                for warning in warnings:
                    lines.append(f"- warning: `{warning}`")
        paper_rows = report.get("paper_markout_rows", []) if isinstance(report, dict) else []
        if paper_rows:
            lines.append("| Market | paper_fills | paper_fill_rate/min | markout_250ms | markout_1s | toxicity_250ms | toxicity_1s |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in paper_rows:
                lines.append(
                    "| {market} | {fills} | {fill_rate} | {mo250} | {mo1s} | {tox250} | {tox1s} |".format(
                        market=row.get("market"),
                        fills=row.get("paper_fills"),
                        fill_rate=row.get("paper_fill_rate_per_min"),
                        mo250=row.get("paper_markout_250ms_bps"),
                        mo1s=row.get("paper_markout_1s_bps"),
                        tox250=row.get("paper_toxicity_250ms"),
                        tox1s=row.get("paper_toxicity_1s"),
                    )
                )
        else:
            lines.append("- no Stage B rows sampled")
    lines.append("")

    lines.append("## Recommended Actions")
    if not actions:
        lines.append("- none")
    else:
        for action in actions:
            lines.append(
                f"- `{action['action_id']}` {action['action_type']} `{action['market']}`"
            )
            lines.append("```bash")
            for cmd in action.get("commands", []):
                lines.append(cmd)
            lines.append("```")
    lines.append("")

    raw_inputs = report.get("raw_inputs", {}) if isinstance(report, dict) else {}
    screen_output = raw_inputs.get("screen_command_output") if isinstance(raw_inputs, dict) else None
    lines.append("## Full screen_mm_markets.py Output")
    if isinstance(screen_output, dict):
        lines.append(f"- command: `{screen_output.get('command_text', 'n/a')}`")
        lines.append(f"- returncode: `{screen_output.get('returncode', 'n/a')}`")
        lines.append("### stdout")
        lines.append("```text")
        stdout_text = str(screen_output.get("stdout") or "")
        if stdout_text.strip():
            lines.extend(stdout_text.rstrip("\n").splitlines())
        else:
            lines.append("(empty)")
        lines.append("```")
        stderr_text = str(screen_output.get("stderr") or "")
        if stderr_text.strip():
            lines.append("### stderr")
            lines.append("```text")
            lines.extend(stderr_text.rstrip("\n").splitlines())
            lines.append("```")
    else:
        lines.append("- no captured screen output")
    lines.append("")

    return "\n".join(lines)


def _write_action_shell(path: Path, actions: List[Dict[str, Any]]) -> None:
    lines: List[str] = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.append("# Generated by market_scout_pipeline.py")
    lines.append(f"# generated_at={iso_utc()}")
    lines.append("")
    if not actions:
        lines.append("echo 'No recommended actions in this run.'")
    else:
        for action in actions:
            lines.append(f"# {action['action_id']} ({action['action_type']} {action['market']})")
            for cmd in action.get("commands", []):
                if cmd.startswith("#"):
                    lines.append(cmd)
                else:
                    lines.append(cmd)
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic market scout pipeline for MM.")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="Repository root path.")
    parser.add_argument(
        "--jobs-json",
        default="/home/flexouille/.openclaw/cron/jobs.json",
        help="OpenClaw jobs.json path (source of truth for active markets).",
    )
    parser.add_argument(
        "--journal-dir",
        default="data/mm_journal",
        help="Journal directory for mm_*.jsonl files.",
    )
    parser.add_argument(
        "--template-env",
        default="config/examples/mm_template.env",
        help="Conservative template env used to draft candidate market env files.",
    )
    parser.add_argument(
        "--policy",
        default="config/market_scout_policy.yaml",
        help="Policy file path (JSON-compatible YAML).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mm_audit/scout",
        help="Output directory for scout artifacts.",
    )
    parser.add_argument("--scan-duration-s", type=float, default=180.0, help="Fallback sampling duration.")
    parser.add_argument(
        "--find-duration-s",
        type=float,
        default=None,
        help="Sampling duration for find_mm_markets (defaults to --scan-duration-s).",
    )
    parser.add_argument(
        "--screen-duration-s",
        type=float,
        default=None,
        help="Sampling duration for screen_mm_markets (defaults to --scan-duration-s).",
    )
    parser.add_argument("--scan-interval-s", type=float, default=2.0, help="Sampling interval.")
    parser.add_argument(
        "--default-env",
        default=".env.eth",
        help="Fallback env file used when per-market env cannot be inferred.",
    )
    parser.add_argument("--max-new-per-day", type=int, default=None, help="Override policy max launches/day.")
    parser.add_argument(
        "--launch-size-multiplier",
        type=Decimal,
        default=None,
        help="Override policy launch size multiplier.",
    )
    parser.add_argument("--warmup-hours", type=int, default=None, help="Override policy warmup hours.")
    parser.add_argument(
        "--paper-markout",
        action="store_true",
        help="Enable Stage B paper maker-markout estimator.",
    )
    parser.add_argument(
        "--paper-duration-s",
        type=float,
        default=300.0,
        help="Stage B sampling duration in seconds (default: 300).",
    )
    parser.add_argument(
        "--paper-top-k",
        type=int,
        default=10,
        help="Run Stage B for top K Stage A candidates (default: 10).",
    )
    parser.add_argument(
        "--paper-horizons-ms",
        default="250,1000,5000,30000,120000",
        help="Comma-separated markout horizons in ms.",
    )
    parser.add_argument(
        "--paper-queue-capture",
        type=float,
        default=0.2,
        help="Queue capture factor applied to fill-rate estimate only.",
    )
    parser.add_argument(
        "--paper-include-trade-types",
        default="TRADE",
        help="Comma-separated trade types to include (default: TRADE).",
    )
    parser.add_argument(
        "--paper-min-fills",
        type=int,
        default=20,
        help="Minimum raw inferred fills for Stage B pass.",
    )
    parser.add_argument(
        "--paper-max-toxicity-250ms",
        type=float,
        default=0.60,
        help="Maximum allowed paper toxicity share at 250ms.",
    )
    parser.add_argument(
        "--paper-min-markout-250ms",
        type=float,
        default=-1.0,
        help="Minimum mean paper markout at 250ms (bps).",
    )
    parser.add_argument(
        "--paper-min-markout-1s",
        type=float,
        default=-1.0,
        help="Minimum mean paper markout at 1s (bps).",
    )
    parser.add_argument(
        "--paper-bbo-match-mode",
        choices=["strict", "loose"],
        default="strict",
        help="Trade/BBO matching mode for inferred paper fills.",
    )
    parser.add_argument(
        "--paper-warmup-s",
        type=float,
        default=8.0,
        help="Warmup window for Stage B orderbook before trade processing (default: 8s).",
    )
    parser.add_argument(
        "--paper-max-trade-lag-ms",
        type=int,
        default=5000,
        help="Max allowed |msg_ts - trade_T| before falling back to msg ts (default: 5000).",
    )
    parser.add_argument(
        "--paper-fallback-mode",
        choices=["stageA", "strict"],
        default="stageA",
        help="Behavior when Stage B data quality is degraded.",
    )
    parser.add_argument(
        "--paper-markout-shift-bps",
        type=float,
        default=2.0,
        help="score2 markout shift term in bps.",
    )
    parser.add_argument(
        "--paper-markout-scale-bps",
        type=float,
        default=2.0,
        help="score2 markout scale divisor in bps.",
    )
    parser.add_argument(
        "--paper-markout-max-score",
        type=float,
        default=3.0,
        help="score2 markout component cap.",
    )
    parser.add_argument(
        "--paper-toxicity-center",
        type=float,
        default=0.5,
        help="score2 toxicity penalty center.",
    )
    parser.add_argument(
        "--paper-toxicity-scale",
        type=float,
        default=6.0,
        help="score2 toxicity penalty slope.",
    )
    parser.add_argument(
        "--paper-toxicity-max-penalty",
        type=float,
        default=3.0,
        help="score2 toxicity penalty cap.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_policy((repo_root / args.policy).resolve())
    default_env_path = _resolve_env_path(repo_root, args.default_env) if args.default_env else None
    market_env_index = _build_market_env_index(repo_root)
    subprocess_env = _build_subprocess_env(default_env_path)

    launch_cfg = policy.get("launch", {}) if isinstance(policy, dict) else {}
    raw_max_new_per_day: Any = (
        args.max_new_per_day
        if args.max_new_per_day is not None
        else launch_cfg.get("max_new_per_day")
    )
    if raw_max_new_per_day is None:
        max_new_per_day: Optional[int] = None
    else:
        try:
            parsed_max_new_per_day = int(raw_max_new_per_day)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid launch.max_new_per_day value: {raw_max_new_per_day}") from exc
        max_new_per_day = parsed_max_new_per_day if parsed_max_new_per_day > 0 else None
    max_new_per_run = int(launch_cfg.get("max_new_per_run", 2))
    launch_size_multiplier = (
        args.launch_size_multiplier
        if args.launch_size_multiplier is not None
        else _d(launch_cfg.get("launch_size_multiplier", "0.25"))
    )
    warmup_hours = int(args.warmup_hours if args.warmup_hours is not None else launch_cfg.get("warmup_hours", 6))
    paper_enabled = bool(args.paper_markout)
    paper_horizons_ms = _parse_int_csv(
        str(args.paper_horizons_ms),
        fallback=DEFAULT_PAPER_HORIZONS_MS,
    )
    paper_trade_types = _parse_trade_types(str(args.paper_include_trade_types))
    paper_cfg = _build_paper_config(args)
    paper_fallback_mode = str(args.paper_fallback_mode)

    journal_dir = (repo_root / args.journal_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    draft_env_dir = output_dir / "draft_envs"
    tmp_dir = output_dir / "tmp"
    state_path = output_dir / "state.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    jobs_path = Path(args.jobs_json).resolve()
    active_jobs = load_market_jobs(jobs_path, repo_root)
    fallback_markets = discover_recent_markets_from_journals(journal_dir, lookback_s=86400.0)
    active_by_market = {str(j.get("market")): j for j in active_jobs}
    for idx, market in enumerate(fallback_markets):
        if market in active_by_market:
            continue
        active_jobs.append({
            "job_id": f"journal-fallback-{idx}",
            "job_name": "journal_fallback",
            "market": market,
            "env_path": None,
        })
    # Ensure each active job has the best env hint available.
    for job in active_jobs:
        market = str(job.get("market") or "")
        env_path = job.get("env_path")
        if not env_path and market:
            env_path = market_env_index.get(market)
        if not env_path and default_env_path is not None:
            env_path = str(default_env_path)
        job["env_path"] = env_path
    active_markets = {j["market"] for j in active_jobs}

    find_duration_s = float(args.find_duration_s if args.find_duration_s is not None else args.scan_duration_s)
    screen_duration_s = float(args.screen_duration_s if args.screen_duration_s is not None else args.scan_duration_s)
    scan_interval_s = float(args.scan_interval_s)

    gates = policy.get("candidate_gates", {}) if isinstance(policy, dict) else {}
    find_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "tools" / "find_mm_markets.py"),
        "--duration-s",
        str(find_duration_s),
        "--interval-s",
        str(scan_interval_s),
        "--min-spread-bps",
        str(gates.get("min_spread_bps", 6)),
        "--max-spread-bps",
        str(gates.get("max_spread_bps", 40)),
        "--min-coverage-pct",
        str(gates.get("min_coverage_pct", 85)),
        "--min-daily-volume",
        str(gates.get("min_daily_volume", 200000)),
        "--json-stdout",
    ]
    screen_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "screen_mm_markets.py"),
        "--duration-s",
        str(screen_duration_s),
        "--interval-s",
        str(scan_interval_s),
        "--json-stdout",
    ]
    # Sampling can stall on API latency spikes; keep a 10-minute minimum timeout per phase.
    find_timeout_s = max(
        600.0,
        find_duration_s + 300.0,
        find_duration_s * 2.0,
    )
    screen_timeout_s = max(
        600.0,
        screen_duration_s + 300.0,
        screen_duration_s * 2.0,
    )

    find_payload, find_command_output = _run_json_stdout(
        find_cmd,
        "find_mm_markets",
        run_env=subprocess_env,
        timeout_s=find_timeout_s,
    )
    screen_payload, screen_command_output = _run_json_stdout(
        screen_cmd,
        "screen_mm_markets",
        run_env=subprocess_env,
        timeout_s=screen_timeout_s,
    )
    paper_stats: Dict[str, Dict[str, Any]] = {}
    paper_stage_warnings: List[str] = []
    paper_target_markets: List[str] = []
    paper_health_summary: Optional[Dict[str, Any]] = None
    paper_degraded = False
    paper_fallback_applied = False
    paper_scoring_enabled = False

    if paper_enabled:
        stage_a_candidates = _build_candidate_pool(
            find_payload=find_payload,
            screen_payload=screen_payload,
            policy=policy,
        )
        stage_a_ranked = [c for c in stage_a_candidates if c.get("passes_all_gates")]
        paper_top_k = max(0, int(args.paper_top_k))
        paper_target_markets = [str(c.get("market")) for c in stage_a_ranked[:paper_top_k]]

        if paper_target_markets and float(args.paper_duration_s) > 0:
            try:
                from market_maker.scout.paper_markout import run_paper_markout

                paper_stats = run_paper_markout(
                    markets=paper_target_markets,
                    duration_s=float(args.paper_duration_s),
                    horizons_ms=paper_horizons_ms,
                    queue_capture=float(args.paper_queue_capture),
                    bbo_match_mode=str(args.paper_bbo_match_mode),
                    include_trade_types=paper_trade_types,
                    warmup_s=float(args.paper_warmup_s),
                    max_trade_lag_ms=int(args.paper_max_trade_lag_ms),
                )
            except Exception as exc:
                paper_stage_warnings.append(
                    f"paper_markout_failed:{type(exc).__name__}:{exc}"
                )
        elif not paper_target_markets:
            paper_stage_warnings.append("paper_markout_skipped_no_stage_a_candidates")
        else:
            paper_stage_warnings.append("paper_markout_skipped_non_positive_duration")

        paper_health_summary = _summarize_paper_health(paper_target_markets, paper_stats)
        paper_degraded = bool(paper_health_summary.get("degraded")) if isinstance(paper_health_summary, dict) else False
        if paper_degraded:
            if paper_fallback_mode == "stageA":
                paper_fallback_applied = True
                paper_stage_warnings.append("paper_markout_degraded_fallback_stageA")
            else:
                paper_stage_warnings.append("paper_markout_degraded_strict_mode")

        paper_scoring_enabled = not paper_fallback_applied
        if paper_scoring_enabled:
            candidates = _build_candidate_pool(
                find_payload=find_payload,
                screen_payload=screen_payload,
                policy=policy,
                paper_enabled=True,
                paper_stats=paper_stats,
                paper_cfg=paper_cfg,
                paper_sampled_markets=set(paper_target_markets),
            )
        else:
            candidates = _build_candidate_pool(
                find_payload=find_payload,
                screen_payload=screen_payload,
                policy=policy,
            )
    else:
        candidates = _build_candidate_pool(
            find_payload=find_payload,
            screen_payload=screen_payload,
            policy=policy,
        )

    screen_scores = {
        str(row.get("name")): _d(row.get("score"))
        for row in (screen_payload.get("markets") or [])
        if isinstance(row, dict) and row.get("name") is not None
    }

    state = _load_state(state_path)
    prev_streaks = state.get("underperformance_streaks", {})
    if not isinstance(prev_streaks, dict):
        prev_streaks = {}

    active_rows: List[Dict[str, Any]] = []
    new_streaks: Dict[str, int] = {}
    for job in active_jobs:
        market = str(job["market"])
        env_path = job.get("env_path")
        latest_journal = find_latest_journal(journal_dir, market)

        analysis_json = None
        analysis_error = None
        if latest_journal is not None:
            analysis_out = tmp_dir / f"analysis_{slugify(market.lower())}.json"
            analysis_json, analysis_error = _run_analysis_json(
                repo_root=repo_root,
                journal=latest_journal,
                output_path=analysis_out,
            )

        pnl_json, pnl_error = _run_pnl_json(
            repo_root=repo_root,
            market=market,
            env_path=env_path,
        )

        markout_5s = None
        fill_rate_pct = None
        if isinstance(analysis_json, dict):
            metrics = analysis_json.get("metrics")
            if isinstance(metrics, dict):
                markout_5s = safe_decimal(metrics.get("markout_5s_bps"))
                fill_rate_pct = safe_decimal(metrics.get("fill_rate_pct"))

        pnl_24h = None
        if isinstance(pnl_json, dict):
            totals = pnl_json.get("totals")
            if isinstance(totals, dict):
                pnl_24h = safe_decimal(totals.get("total_pnl_including_open_usd"))

        flags = _underperformance_flags(
            pnl_24h_usd=pnl_24h,
            markout_5s_bps=markout_5s,
            fill_rate_pct=fill_rate_pct,
            policy=policy,
        )
        triggered = _all_flags_true(flags)
        prev = int(prev_streaks.get(market, 0))
        streak = prev + 1 if triggered else 0
        new_streaks[market] = streak

        active_rows.append({
            "job_id": job.get("job_id"),
            "job_name": job.get("job_name"),
            "market": market,
            "env_path": env_path,
            "config_snapshot": _extract_config_snapshot(env_path),
            "latest_journal": str(latest_journal) if latest_journal else None,
            "score": screen_scores.get(market),
            "pnl_24h_usd": pnl_24h,
            "markout_5s_bps": markout_5s,
            "fill_rate_pct": fill_rate_pct,
            "underperformance_flags": flags,
            "underperformance_triggered": triggered,
            "underperformance_streak": streak,
            "analysis_error": analysis_error,
            "pnl_error": pnl_error,
        })

    quality_issues: List[str] = []
    find_markets = find_payload.get("markets") if isinstance(find_payload, dict) else None
    screen_markets = screen_payload.get("markets") if isinstance(screen_payload, dict) else None
    if not isinstance(find_markets, list):
        quality_issues.append("find_payload_invalid_shape")
    if not isinstance(screen_markets, list):
        quality_issues.append("screen_payload_invalid_shape")
    if isinstance(find_payload, dict) and find_payload.get("warning"):
        quality_issues.append(f"find_warning:{find_payload.get('warning')}")
    if isinstance(screen_payload, dict) and screen_payload.get("warning"):
        quality_issues.append(f"screen_warning:{screen_payload.get('warning')}")
    if isinstance(find_markets, list) and len(find_markets) == 0:
        quality_issues.append("find_zero_markets")
    if isinstance(screen_markets, list) and len(screen_markets) == 0:
        quality_issues.append("screen_zero_markets")
    if active_rows and all(row.get("pnl_error") for row in active_rows):
        quality_issues.append("all_active_markets_missing_pnl")
    if paper_enabled:
        quality_issues.extend(paper_stage_warnings)
        if paper_target_markets and not paper_stats:
            quality_issues.append("paper_markout_no_stats")
        if paper_degraded:
            quality_issues.append("paper_markout_degraded")
        if paper_fallback_applied:
            quality_issues.append("paper_markout_fallback_stageA")

    scout_data_ok = len(quality_issues) == 0

    now = now_ts()
    expires = now + int(timedelta(hours=6).total_seconds())

    history = state.get("proposed_launch_history", [])
    if not isinstance(history, list):
        history = []
    history = [
        h for h in history
        if isinstance(h, dict) and safe_decimal(h.get("ts")) is not None and float(h["ts"]) >= now - 86400
    ]
    launches_24h = len(history)
    if max_new_per_day is None:
        slots_daily = max_new_per_run
    else:
        slots_daily = max(0, max_new_per_day - launches_24h)
    slots_this_run = max(0, min(max_new_per_run, slots_daily))

    available_candidates = [
        c for c in candidates
        if c.get("passes_all_gates") and c.get("market") not in active_markets
    ]

    actions: List[Dict[str, Any]] = []
    used_candidates: set[str] = set()

    if scout_data_ok:
        for candidate in available_candidates[:slots_this_run]:
            market = str(candidate["market"])
            draft_env = draft_env_dir / f".env.{slugify(market.lower())}.candidate"
            _prepare_draft_env(
                template_env=(repo_root / args.template_env).resolve(),
                output_path=draft_env,
                market=market,
                launch_size_multiplier=launch_size_multiplier,
                warmup_hours=warmup_hours,
            )
            action = _build_launch_action(
                market_row=candidate,
                created_ts=now,
                expires_ts=expires,
                draft_env_path=draft_env,
                repo_root=repo_root,
                warmup_hours=warmup_hours,
                launch_size_multiplier=launch_size_multiplier,
            )
            actions.append(action)
            used_candidates.add(market)
            history.append({"ts": now, "market": market, "action_id": action["action_id"]})

    rotate_cfg = policy.get("rotation", {}) if isinstance(policy, dict) else {}
    min_score_delta = _d(rotate_cfg.get("min_score_delta", "1.5"))
    required_cycles = int(rotate_cfg.get("underperformance_cycles", 2))

    rotate_pool = [
        c for c in available_candidates
        if c.get("market") not in used_candidates
    ]
    rotate_score_key = "score2" if paper_scoring_enabled else "score"
    rotate_pool.sort(key=lambda c: _d(c.get(rotate_score_key), default="0"), reverse=True)

    if scout_data_ok:
        for active in sorted(active_rows, key=lambda r: r.get("underperformance_streak", 0), reverse=True):
            from_market = str(active.get("market"))
            streak = int(active.get("underperformance_streak") or 0)
            if streak < required_cycles:
                continue
            from_score = _d(active.get("score"), default="0")

            replacement = None
            for candidate in rotate_pool:
                score_delta = _d(candidate.get(rotate_score_key), default="0") - from_score
                if score_delta >= min_score_delta:
                    replacement = candidate
                    break
            if replacement is None:
                continue

            to_market = str(replacement["market"])
            draft_env = draft_env_dir / f".env.{slugify(to_market.lower())}.candidate"
            _prepare_draft_env(
                template_env=(repo_root / args.template_env).resolve(),
                output_path=draft_env,
                market=to_market,
                launch_size_multiplier=launch_size_multiplier,
                warmup_hours=warmup_hours,
            )
            action = _build_rotate_action(
                from_market=from_market,
                from_score=from_score,
                streak=streak,
                candidate_row=replacement,
                created_ts=now,
                expires_ts=expires,
                draft_env_path=draft_env,
                repo_root=repo_root,
                warmup_hours=warmup_hours,
                launch_size_multiplier=launch_size_multiplier,
            )
            actions.append(action)
            used_candidates.add(to_market)
            rotate_pool = [c for c in rotate_pool if c.get("market") != to_market]

    rate_limits: Dict[str, Any] = {
        "max_new_per_run": max_new_per_run,
        "launch_slots_this_run": slots_this_run,
    }
    if max_new_per_day is not None:
        rate_limits["proposed_launches_last24h"] = launches_24h
        rate_limits["max_new_per_day"] = max_new_per_day

    paper_methodology_meta: Optional[Dict[str, Any]] = None
    paper_report_meta: Optional[Dict[str, Any]] = None
    paper_markout_rows: List[Dict[str, Any]] = []
    if paper_enabled:
        paper_methodology_meta = {
            "enabled": True,
            "scoring_enabled": paper_scoring_enabled,
            "duration_s": float(args.paper_duration_s),
            "top_k": int(args.paper_top_k),
            "horizons_ms": paper_horizons_ms,
            "queue_capture": float(args.paper_queue_capture),
            "bbo_match_mode": str(args.paper_bbo_match_mode),
            "include_trade_types": paper_trade_types,
            "warmup_s": float(args.paper_warmup_s),
            "max_trade_lag_ms": int(args.paper_max_trade_lag_ms),
            "fallback_mode": paper_fallback_mode,
            "degraded": paper_degraded,
            "fallback_applied": paper_fallback_applied,
            "health_summary": paper_health_summary,
            "gates": {
                "paper_min_fills": int(args.paper_min_fills),
                "paper_max_toxicity_250ms": float(args.paper_max_toxicity_250ms),
                "paper_min_markout_250ms": float(args.paper_min_markout_250ms),
                "paper_min_markout_1s": float(args.paper_min_markout_1s),
            },
            "score_weights": {
                "paper_markout_shift_bps": float(args.paper_markout_shift_bps),
                "paper_markout_scale_bps": float(args.paper_markout_scale_bps),
                "paper_markout_max_score": float(args.paper_markout_max_score),
                "paper_toxicity_center": float(args.paper_toxicity_center),
                "paper_toxicity_scale": float(args.paper_toxicity_scale),
                "paper_toxicity_max_penalty": float(args.paper_toxicity_max_penalty),
            },
        }
        paper_report_meta = {
            "enabled": True,
            "scoring_enabled": paper_scoring_enabled,
            "duration_s": float(args.paper_duration_s),
            "top_k": int(args.paper_top_k),
            "horizons_ms": paper_horizons_ms,
            "queue_capture": float(args.paper_queue_capture),
            "bbo_match_mode": str(args.paper_bbo_match_mode),
            "include_trade_types": paper_trade_types,
            "warmup_s": float(args.paper_warmup_s),
            "max_trade_lag_ms": int(args.paper_max_trade_lag_ms),
            "fallback_mode": paper_fallback_mode,
            "degraded": paper_degraded,
            "fallback_applied": paper_fallback_applied,
            "health_summary": paper_health_summary,
            "target_markets": paper_target_markets,
            "sampled_markets": sorted(paper_stats.keys()),
            "warnings": paper_stage_warnings,
        }
        if paper_scoring_enabled:
            paper_markout_rows = [
                {
                    "market": c.get("market"),
                    "paper_fills": c.get("paper_fills"),
                    "paper_fill_rate_per_min": c.get("paper_fill_rate_per_min"),
                    "paper_markout_250ms_bps": c.get("paper_markout_250ms_bps"),
                    "paper_markout_1s_bps": c.get("paper_markout_1s_bps"),
                    "paper_toxicity_250ms": c.get("paper_toxicity_250ms"),
                    "paper_toxicity_1s": c.get("paper_toxicity_1s"),
                    "paper_trade_rows_seen": c.get("paper_trade_rows_seen"),
                    "paper_trade_rows_used": c.get("paper_trade_rows_used"),
                    "paper_bbo_updates_seen": c.get("paper_bbo_updates_seen"),
                    "paper_bbo_ready": c.get("paper_bbo_ready"),
                }
                for c in candidates
                if c.get("paper_sampled") is True
            ]
        else:
            paper_markout_rows = []
            for market in paper_target_markets:
                row = paper_stats.get(market, {}) if isinstance(paper_stats.get(market), dict) else {}
                paper_markout_rows.append({
                    "market": market,
                    "paper_fills": row.get("paper_fills"),
                    "paper_fill_rate_per_min": row.get("paper_fill_rate_per_min"),
                    "paper_markout_250ms_bps": row.get("paper_markout_bps_250ms_mean"),
                    "paper_markout_1s_bps": row.get("paper_markout_bps_1s_mean"),
                    "paper_toxicity_250ms": row.get("paper_toxicity_share_250ms"),
                    "paper_toxicity_1s": row.get("paper_toxicity_share_1s"),
                    "paper_trade_rows_seen": row.get("paper_trade_rows_seen"),
                    "paper_trade_rows_used": row.get("paper_trade_rows_used"),
                    "paper_bbo_updates_seen": row.get("paper_bbo_updates_seen"),
                    "paper_bbo_ready": row.get("paper_bbo_ready"),
                })

    report_payload = {
        "generated_at": iso_utc(now),
        "source": {
            "repo_root": str(repo_root),
            "jobs_json": str(jobs_path),
            "journal_dir": str(journal_dir),
        },
        "policy_snapshot": policy,
        "active_markets": active_rows,
        "candidate_markets": candidates,
        "rate_limits": rate_limits,
        "data_quality": {
            "ok": scout_data_ok,
            "issues": quality_issues,
            "default_env": str(default_env_path) if default_env_path else None,
            "market_env_index_size": len(market_env_index),
        },
        "recommendations": actions,
        "methodology": _build_methodology(
            policy=policy,
            find_duration_s=find_duration_s,
            screen_duration_s=screen_duration_s,
            scan_interval_s=scan_interval_s,
            paper_meta=paper_methodology_meta,
            paper_scoring_enabled=paper_scoring_enabled,
        ),
        "raw_inputs": {
            "find_payload": find_payload,
            "screen_payload": screen_payload,
            "find_command_output": find_command_output,
            "screen_command_output": screen_command_output,
        },
    }
    if paper_enabled:
        report_payload["paper_markout"] = paper_report_meta
        report_payload["paper_markout_rows"] = paper_markout_rows
        report_payload["raw_inputs"]["paper_markout"] = {
            "target_markets": paper_target_markets,
            "stats": paper_stats,
            "health_summary": paper_health_summary,
            "degraded": paper_degraded,
            "fallback_applied": paper_fallback_applied,
            "warnings": paper_stage_warnings,
        }

    report_json_path = output_dir / "market_scout_report.json"
    action_pack_path = output_dir / "action_pack.json"
    report_md_path = output_dir / "market_scout_report.md"
    action_sh_path = output_dir / "market_scout_actions.sh"
    run_log_path = output_dir / "market_scout_runs.jsonl"

    write_json(report_json_path, report_payload)
    write_json(action_pack_path, actions)
    report_md_path.write_text(_render_markdown(report_payload, actions) + "\n")
    _write_action_shell(action_sh_path, actions)

    append_jsonl(
        run_log_path,
        {
            "ts": now,
            "generated_at": iso_utc(now),
            "actions": [a.get("action_id") for a in actions],
            "active_markets": [a.get("market") for a in active_rows],
            "candidate_count": len(candidates),
        },
    )

    state["underperformance_streaks"] = new_streaks
    state["proposed_launch_history"] = history
    state["last_run_ts"] = now
    _save_state(state_path, state)

    print(f"report_json={report_json_path}")
    print(f"action_pack={action_pack_path}")
    print(f"report_md={report_md_path}")
    print(f"action_shell={action_sh_path}")
    print(f"actions={len(actions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
