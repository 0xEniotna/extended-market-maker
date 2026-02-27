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
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

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


def _d(value: Any, default: str = "0") -> Decimal:
    parsed = safe_decimal(value)
    if parsed is None:
        return Decimal(default)
    return parsed


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
) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=run_env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{label} timed out after {exc.timeout}s") from exc
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        out = (stdout + "\n" + stderr).strip()
        if (
            "No successful market snapshots collected." in out
            or "No active markets found." in out
        ):
            return {"markets": [], "count": 0, "sampling": {}, "warning": out}
        raise RuntimeError(
            f"{label} failed (code={proc.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    return _extract_markets_payload(stdout, label)


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
        str(repo_root / "scripts" / "tools" / "fetch_pnl.py"),
        market,
        "--days",
        "1",
        "--max-pages",
        "30",
        "--json-stdout",
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
) -> List[Dict[str, Any]]:
    gates = policy.get("candidate_gates", {}) if isinstance(policy, dict) else {}
    min_score = _d(gates.get("min_score", "0"))
    min_ticks = _d(gates.get("min_ticks_in_spread", "0"))
    min_cov = _d(gates.get("min_coverage_pct", "0"))
    max_p90 = _d(gates.get("max_spread_p90_bps", "0"))
    min_daily_volume = _d(gates.get("min_daily_volume", "0"))

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
        passes = all(gate_flags.values())

        out.append({
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
            "passes_all_gates": passes,
        })

    out.sort(key=lambda item: (item["passes_all_gates"], item["score"]), reverse=True)
    return out


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
        f"cd {repo_root} && scripts/mm_openclaw_fleet.sh start {env_target.name}",
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
        "# scripts/mm_openclaw_fleet.sh stop <ENV_OF_UNDERPERFORMING_MARKET>",
        f"cp {draft_env_path} {env_target}",
        f"cd {repo_root} && scripts/mm_openclaw_fleet.sh start {env_target.name}",
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
            "score_delta": _d(candidate_row.get("score")) - from_score,
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

    limits = report.get("rate_limits", {})
    lines.append("## Rate Limits")
    lines.append(
        f"- proposed_launches_last24h: {limits.get('proposed_launches_last24h', 0)}"
    )
    lines.append(f"- max_new_per_day: {limits.get('max_new_per_day')}")
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
    parser.add_argument("--scan-duration-s", type=float, default=180.0, help="Sampling duration.")
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
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_policy((repo_root / args.policy).resolve())
    default_env_path = _resolve_env_path(repo_root, args.default_env) if args.default_env else None
    market_env_index = _build_market_env_index(repo_root)
    subprocess_env = _build_subprocess_env(default_env_path)

    launch_cfg = policy.get("launch", {}) if isinstance(policy, dict) else {}
    max_new_per_day = int(args.max_new_per_day if args.max_new_per_day is not None else launch_cfg.get("max_new_per_day", 4))
    max_new_per_run = int(launch_cfg.get("max_new_per_run", 2))
    launch_size_multiplier = (
        args.launch_size_multiplier
        if args.launch_size_multiplier is not None
        else _d(launch_cfg.get("launch_size_multiplier", "0.25"))
    )
    warmup_hours = int(args.warmup_hours if args.warmup_hours is not None else launch_cfg.get("warmup_hours", 6))

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

    gates = policy.get("candidate_gates", {}) if isinstance(policy, dict) else {}
    find_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "tools" / "find_mm_markets.py"),
        "--duration-s",
        str(args.scan_duration_s),
        "--interval-s",
        str(args.scan_interval_s),
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
        str(args.scan_duration_s),
        "--interval-s",
        str(args.scan_interval_s),
        "--json-stdout",
    ]

    find_payload = _run_json_stdout(find_cmd, "find_mm_markets", run_env=subprocess_env)
    screen_payload = _run_json_stdout(screen_cmd, "screen_mm_markets", run_env=subprocess_env)
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
    rotate_pool.sort(key=lambda c: _d(c.get("score")), reverse=True)

    if scout_data_ok:
        for active in sorted(active_rows, key=lambda r: r.get("underperformance_streak", 0), reverse=True):
            from_market = str(active.get("market"))
            streak = int(active.get("underperformance_streak") or 0)
            if streak < required_cycles:
                continue
            from_score = _d(active.get("score"), default="0")

            replacement = None
            for candidate in rotate_pool:
                score_delta = _d(candidate.get("score")) - from_score
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
        "rate_limits": {
            "proposed_launches_last24h": launches_24h,
            "max_new_per_day": max_new_per_day,
            "max_new_per_run": max_new_per_run,
            "launch_slots_this_run": slots_this_run,
        },
        "data_quality": {
            "ok": scout_data_ok,
            "issues": quality_issues,
            "default_env": str(default_env_path) if default_env_path else None,
            "market_env_index_size": len(market_env_index),
        },
        "recommendations": actions,
        "raw_inputs": {
            "find_payload": find_payload,
            "screen_payload": screen_payload,
        },
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
