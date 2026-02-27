#!/usr/bin/env python3
"""Advisor-only MM config intelligence loop.

This script analyzes recent MM journals and emits *proposals* for config updates.
It never edits env files and never starts/stops strategy processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mm_audit_common import (  # noqa: E402
    append_jsonl,
    discover_recent_markets_from_journals,
    iso_utc,
    load_market_jobs,
    now_ts,
    parse_env,
    read_env_lines,
    read_json,
    read_jsonl,
    safe_decimal,
    slugify,
    write_json,
)

TUNABLE_KEYS = (
    "MM_SPREAD_MULTIPLIER",
    "MM_MIN_OFFSET_BPS",
    "MM_MAX_OFFSET_BPS",
    "MM_ORDER_SIZE_MULTIPLIER",
    "MM_INVENTORY_SKEW_FACTOR",
    "MM_MIN_REPRICE_INTERVAL_S",
    "MM_MIN_REPRICE_MOVE_TICKS",
    "MM_MIN_REPRICE_EDGE_DELTA_BPS",
    "MM_REPRICE_TOLERANCE_PERCENT",
)

MAX_DRIFT_FROM_BASELINE: Dict[str, Decimal] = {
    "MM_SPREAD_MULTIPLIER": Decimal("2.0"),
    "MM_MIN_OFFSET_BPS": Decimal("3.0"),
    "MM_MAX_OFFSET_BPS": Decimal("30.0"),
    "MM_ORDER_SIZE_MULTIPLIER": Decimal("0.5"),
    "MM_INVENTORY_SKEW_FACTOR": Decimal("0.3"),
    "MM_MIN_REPRICE_INTERVAL_S": Decimal("5.0"),
    "MM_MIN_REPRICE_MOVE_TICKS": Decimal("10"),
    "MM_MIN_REPRICE_EDGE_DELTA_BPS": Decimal("5.0"),
    "MM_REPRICE_TOLERANCE_PERCENT": Decimal("1.0"),
}

UPDATE_BOUNDS: Dict[str, Tuple[Decimal, Decimal]] = {
    "MM_SPREAD_MULTIPLIER": (Decimal("0.05"), Decimal("8.0")),
    "MM_MIN_OFFSET_BPS": (Decimal("0"), Decimal("100")),
    "MM_MAX_OFFSET_BPS": (Decimal("1"), Decimal("300")),
    "MM_ORDER_SIZE_MULTIPLIER": (Decimal("0.1"), Decimal("100")),
    "MM_INVENTORY_SKEW_FACTOR": (Decimal("0"), Decimal("2.0")),
    "MM_MIN_REPRICE_INTERVAL_S": (Decimal("0"), Decimal("30")),
    "MM_MIN_REPRICE_MOVE_TICKS": (Decimal("0"), Decimal("100")),
    "MM_MIN_REPRICE_EDGE_DELTA_BPS": (Decimal("0"), Decimal("50")),
    "MM_REPRICE_TOLERANCE_PERCENT": (Decimal("0.01"), Decimal("5.0")),
}


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    parsed = safe_decimal(value)
    if parsed is None:
        return Decimal(default)
    return parsed


def _decimal_str(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _clamp(key: str, value: Decimal) -> Decimal:
    bounds = UPDATE_BOUNDS.get(key)
    if bounds is None:
        return value
    lo, hi = bounds
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _read_jsonl_events(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                rows.append(event)
    return rows


def _env_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _event_mid(event: Dict[str, Any]) -> Optional[Decimal]:
    mid = safe_decimal(event.get("mid"))
    if mid is not None and mid > 0:
        return mid
    bid = safe_decimal(event.get("best_bid"))
    ask = safe_decimal(event.get("best_ask"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return (bid + ask) / Decimal("2")


def _mid_at_or_after(
    timestamps: List[float],
    mids: List[Decimal],
    target_ts: float,
    *,
    max_wait_s: float = 60.0,
) -> Optional[Decimal]:
    if not timestamps:
        return None
    lo = 0
    hi = len(timestamps)
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < target_ts:
            lo = mid + 1
        else:
            hi = mid
    if lo >= len(timestamps):
        return None
    if timestamps[lo] - target_ts > max_wait_s:
        return None
    return mids[lo]


def _avg(values: List[Decimal]) -> Optional[Decimal]:
    if not values:
        return None
    return sum(values) / Decimal(len(values))


def _pctl(values: List[Decimal], p: Decimal) -> Optional[Decimal]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = int(round((len(ordered) - 1) * float(p)))
    idx = min(len(ordered) - 1, max(0, idx))
    return ordered[idx]


def _market_env_index(repo_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for env_path in sorted(repo_root.glob(".env*")):
        if not env_path.is_file():
            continue
        if env_path.name in {".env.example", ".env.sample", ".env.template"}:
            continue
        if env_path.name.endswith(".candidate"):
            continue
        try:
            env_map = parse_env(read_env_lines(env_path))
        except Exception:
            continue
        market = str(env_map.get("MM_MARKET_NAME") or "").strip()
        if not market:
            continue
        current = out.get(market)
        if current is None:
            out[market] = env_path
        elif current.name == ".env" and env_path.name != ".env":
            out[market] = env_path
    return out


def _load_state(path: Path) -> Dict[str, Any]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("markets", {})
    return payload


def _save_state(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def _latest_run_start_config(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    for event in reversed(events):
        if event.get("type") != "run_start":
            continue
        cfg = event.get("config")
        if isinstance(cfg, dict):
            return cfg
    return {}


def _summarize_signals(
    events: List[Dict[str, Any]],
    *,
    now_ts_value: float,
    deadman_window_s: float,
    inventory_window_s: float,
) -> Dict[str, Any]:
    fills = [e for e in events if e.get("type") == "fill"]
    placements = [e for e in events if e.get("type") == "order_placed"]
    fill_rate_pct = None
    if placements:
        fill_rate_pct = (
            Decimal(len(fills)) / Decimal(len(placements)) * Decimal("100")
        )

    ts_values: List[float] = []
    mid_values: List[Decimal] = []
    for event in events:
        ts_raw = event.get("ts")
        if ts_raw is None:
            continue
        mid = _event_mid(event)
        if mid is None:
            continue
        ts_values.append(float(ts_raw))
        mid_values.append(mid)

    markout_5s: List[Decimal] = []
    for fill in fills:
        ts_raw = fill.get("ts")
        fill_px = safe_decimal(fill.get("price"))
        if ts_raw is None or fill_px is None or fill_px <= 0:
            continue
        fut_mid = _mid_at_or_after(ts_values, mid_values, float(ts_raw) + 5.0)
        if fut_mid is None:
            continue
        side = str(fill.get("side", ""))
        if "BUY" in side:
            markout_5s.append((fut_mid - fill_px) / fill_px * Decimal("10000"))
        else:
            markout_5s.append((fill_px - fut_mid) / fill_px * Decimal("10000"))

    deadman_cutoff = now_ts_value - deadman_window_s
    fills_window = 0
    reprice_window = 0
    for event in events:
        ts_raw = event.get("ts")
        if ts_raw is None:
            continue
        ts = float(ts_raw)
        if ts < deadman_cutoff:
            continue
        if event.get("type") == "fill":
            fills_window += 1
        elif event.get("type") == "reprice_decision":
            reprice_window += 1
    deadman = reprice_window > 0 and fills_window == 0

    inv_cutoff = now_ts_value - inventory_window_s
    cfg = _latest_run_start_config(events)
    max_position_size = safe_decimal(cfg.get("max_position_size"))
    hard_pct = safe_decimal(cfg.get("inventory_hard_pct")) or Decimal("0.95")
    rows: List[Tuple[float, Decimal]] = []
    for event in events:
        ts_raw = event.get("ts")
        pos = safe_decimal(event.get("position"))
        if ts_raw is None or pos is None:
            continue
        ts = float(ts_raw)
        if ts < inv_cutoff:
            continue
        rows.append((ts, pos))
    rows.sort(key=lambda row: row[0])

    util_values: List[Decimal] = []
    max_util = None
    time_above_hard_s = 0.0
    if max_position_size is not None and max_position_size > 0:
        for _, pos in rows:
            util = abs(pos) / max_position_size * Decimal("100")
            util_values.append(util)
        if util_values:
            max_util = max(util_values)
        hard_abs = max_position_size * hard_pct
        for idx in range(len(rows) - 1):
            cur_ts, cur_pos = rows[idx]
            nxt_ts, _ = rows[idx + 1]
            if abs(cur_pos) >= hard_abs:
                time_above_hard_s += max(0.0, nxt_ts - cur_ts)

    return {
        "fills_total": len(fills),
        "orders_total": len(placements),
        "fill_rate_pct": fill_rate_pct,
        "markout_5s_bps": _avg(markout_5s),
        "fills_60m": fills_window,
        "reprice_60m": reprice_window,
        "deadman": deadman,
        "utilization_p95_pct": _pctl(util_values, Decimal("0.95")) if util_values else None,
        "utilization_max_pct": max_util,
        "time_above_hard_s": time_above_hard_s if util_values else None,
    }


def _to_decimal_map(env_map: Dict[str, str], keys: Iterable[str]) -> Dict[str, Decimal]:
    out: Dict[str, Decimal] = {}
    for key in keys:
        if key not in env_map:
            continue
        parsed = safe_decimal(env_map.get(key))
        if parsed is not None:
            out[key] = parsed
    return out


def _compute_candidate_updates(
    *,
    current_env: Dict[str, str],
    baseline_params: Dict[str, str],
    signals: Dict[str, Any],
) -> Tuple[Dict[str, Decimal], List[str], str, bool, str]:
    current_vals = _to_decimal_map(current_env, TUNABLE_KEYS)
    baseline_vals = _to_decimal_map(baseline_params, TUNABLE_KEYS)
    updates: Dict[str, Decimal] = {}
    reason_codes: List[str] = []

    if signals.get("deadman"):
        for key, baseline_value in baseline_vals.items():
            current_value = current_vals.get(key)
            if current_value is None:
                continue
            if current_value != baseline_value:
                updates[key] = _clamp(key, baseline_value)
        if updates:
            return updates, ["deadman_switch", "revert_to_baseline"], "high", True, "warren"
        return {}, ["deadman_switch", "already_at_baseline"], "high", True, "warren"

    fill_rate = safe_decimal(signals.get("fill_rate_pct"))
    markout_5s = safe_decimal(signals.get("markout_5s_bps"))
    util_p95 = safe_decimal(signals.get("utilization_p95_pct"))

    if markout_5s is not None and markout_5s < Decimal("-0.5"):
        if "MM_SPREAD_MULTIPLIER" in current_vals:
            updates["MM_SPREAD_MULTIPLIER"] = _clamp(
                "MM_SPREAD_MULTIPLIER",
                current_vals["MM_SPREAD_MULTIPLIER"] * Decimal("1.10"),
            )
        if "MM_MIN_OFFSET_BPS" in current_vals:
            updates["MM_MIN_OFFSET_BPS"] = _clamp(
                "MM_MIN_OFFSET_BPS",
                current_vals["MM_MIN_OFFSET_BPS"] + Decimal("0.5"),
            )
        if "MM_MIN_REPRICE_MOVE_TICKS" in current_vals:
            updates["MM_MIN_REPRICE_MOVE_TICKS"] = _clamp(
                "MM_MIN_REPRICE_MOVE_TICKS",
                current_vals["MM_MIN_REPRICE_MOVE_TICKS"] + Decimal("1"),
            )
        if "MM_MIN_REPRICE_EDGE_DELTA_BPS" in current_vals:
            updates["MM_MIN_REPRICE_EDGE_DELTA_BPS"] = _clamp(
                "MM_MIN_REPRICE_EDGE_DELTA_BPS",
                current_vals["MM_MIN_REPRICE_EDGE_DELTA_BPS"] + Decimal("0.25"),
            )
        reason_codes.append("adverse_markout_5s")

    if (
        fill_rate is not None
        and fill_rate < Decimal("0.5")
        and (markout_5s is None or markout_5s >= Decimal("0"))
    ):
        if "MM_SPREAD_MULTIPLIER" in current_vals:
            updates["MM_SPREAD_MULTIPLIER"] = _clamp(
                "MM_SPREAD_MULTIPLIER",
                current_vals["MM_SPREAD_MULTIPLIER"] * Decimal("0.95"),
            )
        if "MM_MIN_OFFSET_BPS" in current_vals:
            updates["MM_MIN_OFFSET_BPS"] = _clamp(
                "MM_MIN_OFFSET_BPS",
                current_vals["MM_MIN_OFFSET_BPS"] - Decimal("0.3"),
            )
        reason_codes.append("low_fill_rate_with_nonnegative_markout")

    if util_p95 is not None and util_p95 > Decimal("70"):
        if "MM_INVENTORY_SKEW_FACTOR" in current_vals:
            updates["MM_INVENTORY_SKEW_FACTOR"] = _clamp(
                "MM_INVENTORY_SKEW_FACTOR",
                current_vals["MM_INVENTORY_SKEW_FACTOR"] + Decimal("0.05"),
            )
            reason_codes.append("high_inventory_utilization")

    clean_updates: Dict[str, Decimal] = {}
    for key, value in updates.items():
        if key not in current_vals:
            continue
        if value != current_vals[key]:
            clean_updates[key] = value

    if not clean_updates:
        return {}, reason_codes or ["no_actionable_signal"], "low", False, "human"

    confidence = "medium" if markout_5s is not None and fill_rate is not None else "low"
    return clean_updates, reason_codes, confidence, False, "human"


def _drift_ok(
    *,
    key: str,
    proposed: Decimal,
    baseline_value: Optional[Decimal],
) -> Tuple[bool, str]:
    if baseline_value is None:
        return False, "baseline_missing"
    if key not in MAX_DRIFT_FROM_BASELINE:
        return True, "ok"

    drift_limit = MAX_DRIFT_FROM_BASELINE[key]
    if key == "MM_SPREAD_MULTIPLIER":
        if baseline_value <= 0:
            return False, "baseline_non_positive"
        max_allowed = baseline_value * drift_limit
        min_allowed = baseline_value / drift_limit if drift_limit > 0 else baseline_value
        if proposed > max_allowed or proposed < min_allowed:
            return False, "max_drift_exceeded"
        return True, "ok"

    if key == "MM_ORDER_SIZE_MULTIPLIER":
        if baseline_value <= 0:
            return False, "baseline_non_positive"
        allowed_abs = baseline_value * drift_limit
        if abs(proposed - baseline_value) > allowed_abs:
            return False, "max_drift_exceeded"
        return True, "ok"

    if abs(proposed - baseline_value) > drift_limit:
        return False, "max_drift_exceeded"
    return True, "ok"


def _market_state(state: Dict[str, Any], market: str) -> Dict[str, Any]:
    markets = state.setdefault("markets", {})
    row = markets.get(market)
    if not isinstance(row, dict):
        row = {}
        markets[market] = row
    row.setdefault("iteration", 0)
    row.setdefault("proposal_timestamps", [])
    row.setdefault("deadman_cooldown_until_ts", 0.0)
    return row


def _prune_hourly(ts_values: List[float], now_value: float) -> List[float]:
    cutoff = now_value - 3600.0
    return [ts for ts in ts_values if ts >= cutoff]


def _proposal_row(
    *,
    proposal_id: str,
    ts: float,
    market: str,
    env_path: Path,
    iteration: int,
    key: str,
    old_value: Optional[Decimal],
    new_value: Optional[Decimal],
    baseline_value: Optional[Decimal],
    reason_codes: List[str],
    confidence: str,
    guardrail_status: str,
    guardrail_reason: str,
    deadman: bool,
    escalation_target: str,
    cooldown_until_ts: Optional[float],
    rejected: bool,
) -> Dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "ts": ts,
        "created_at": iso_utc(ts),
        "market": market,
        "env_path": str(env_path),
        "iteration": iteration,
        "param": key,
        "old": _decimal_str(old_value) if old_value is not None else None,
        "proposed": _decimal_str(new_value) if new_value is not None else None,
        "new": _decimal_str(new_value) if new_value is not None else None,
        "baseline_value": _decimal_str(baseline_value) if baseline_value is not None else None,
        "reason_codes": reason_codes,
        "confidence": confidence,
        "guardrail_status": guardrail_status,
        "guardrail_reason": guardrail_reason,
        "proposal_only": True,
        "applied": False,
        "rejected": rejected,
        "deadman": deadman,
        "escalation_target": escalation_target,
        "cooldown_until_ts": cooldown_until_ts,
        "source": "mm_advisor_loop",
    }


def _receipt_matches_applied_change(
    receipts: List[Dict[str, Any]],
    *,
    env_after_hash: str,
    min_applied_ts: float,
) -> bool:
    for row in receipts:
        if not isinstance(row, dict):
            continue
        if row.get("result") != "applied":
            continue
        if row.get("applied_by") not in {"warren_auto", "human"}:
            continue
        if str(row.get("env_after_hash") or "") != env_after_hash:
            continue
        applied_ts = safe_decimal(row.get("applied_ts"))
        if applied_ts is None:
            continue
        if float(applied_ts) < min_applied_ts:
            continue
        return True
    return False


def _refresh_baseline_if_needed(
    *,
    market: str,
    env_path: Path,
    baseline_path: Path,
    env_map: Dict[str, str],
    receipts: List[Dict[str, Any]],
    refresh_log_path: Path,
    now_value: float,
) -> Dict[str, Any]:
    baseline = read_json(baseline_path, default={})
    if not isinstance(baseline, dict):
        baseline = {}

    env_hash = _env_file_hash(env_path)
    env_mtime = env_path.stat().st_mtime
    previous_hash = str(baseline.get("env_hash") or "")
    previous_captured_ts = float(safe_decimal(baseline.get("captured_ts")) or 0.0)

    reason: Optional[str] = None
    if not baseline:
        reason = "initial"
    elif previous_hash != env_hash:
        if _receipt_matches_applied_change(
            receipts,
            env_after_hash=env_hash,
            min_applied_ts=previous_captured_ts,
        ):
            reason = "applied_change"
        else:
            reason = "external_env_change"

    if reason is not None:
        baseline = {
            "market": market,
            "env_path": str(env_path),
            "captured_at": iso_utc(now_value),
            "captured_ts": now_value,
            "env_hash": env_hash,
            "env_mtime": env_mtime,
            "refresh_reason": reason,
            "baseline_params": {
                key: env_map[key]
                for key in TUNABLE_KEYS
                if key in env_map
            },
        }
        write_json(baseline_path, baseline)
        append_jsonl(
            refresh_log_path,
            {
                "ts": now_value,
                "created_at": iso_utc(now_value),
                "market": market,
                "env_path": str(env_path),
                "reason": reason,
                "env_hash": env_hash,
            },
        )

    return baseline


def _find_latest_journal(journal_dir: Path, market: str) -> Optional[Path]:
    pattern = f"mm_{market}_*.jsonl"
    files = [
        p for p in journal_dir.glob(pattern)
        if p.is_file() and "mm_tuning_log_" not in p.name
    ]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def _market_from_env(env_path: Path) -> Optional[str]:
    try:
        env_map = parse_env(read_env_lines(env_path))
    except Exception:
        return None
    market = str(env_map.get("MM_MARKET_NAME") or "").strip()
    return market or None


def _warren_auto_apply_candidate(row: Dict[str, Any]) -> bool:
    return (
        row.get("deadman") is True
        and row.get("guardrail_status") == "passed"
        and row.get("confidence") == "high"
    )


def _resolve_markets(
    *,
    repo_root: Path,
    jobs_path: Path,
    journal_dir: Path,
    env_path_filter: Optional[Path],
    market_filter: Optional[str],
) -> List[Dict[str, Any]]:
    env_index = _market_env_index(repo_root)
    jobs = load_market_jobs(jobs_path, repo_root)

    rows: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        market = str(job.get("market") or "").strip()
        if not market:
            continue
        env_path_raw = job.get("env_path")
        env_path = Path(env_path_raw).resolve() if isinstance(env_path_raw, str) else None
        if env_path is None:
            env_path = env_index.get(market)
        rows.append({
            "market": market,
            "env_path": env_path,
            "job_id": job.get("job_id"),
        })

    if not rows:
        for market in discover_recent_markets_from_journals(journal_dir, lookback_s=86400.0):
            rows.append({
                "market": market,
                "env_path": env_index.get(market),
                "job_id": None,
            })

    if env_path_filter is not None:
        env_market = _market_from_env(env_path_filter)
        if env_market is None:
            return []
        rows = [row for row in rows if row["market"] == env_market]
        if not rows:
            rows = [{
                "market": env_market,
                "env_path": env_path_filter,
                "job_id": None,
            }]

    if market_filter:
        rows = [row for row in rows if row["market"] == market_filter]

    unique: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        market = row["market"]
        existing = unique.get(market)
        if existing is None:
            unique[market] = row
            continue
        if existing.get("env_path") is None and row.get("env_path") is not None:
            unique[market] = row

    return [unique[key] for key in sorted(unique.keys())]


def _process_market(
    *,
    market: str,
    env_path: Path,
    journal_path: Path,
    baselines_dir: Path,
    proposals_path: Path,
    changelog_path: Path,
    apply_receipts_path: Path,
    baseline_refresh_log_path: Path,
    state_row: Dict[str, Any],
    now_value: float,
    deadman_window_s: float,
    deadman_cooldown_s: float,
    inventory_window_s: float,
    max_changes_per_hour: int,
) -> Dict[str, Any]:
    env_map = parse_env(read_env_lines(env_path))
    receipts = read_jsonl(apply_receipts_path)
    baseline_path = baselines_dir / f"{slugify(market)}.json"
    baseline = _refresh_baseline_if_needed(
        market=market,
        env_path=env_path,
        baseline_path=baseline_path,
        env_map=env_map,
        receipts=receipts,
        refresh_log_path=baseline_refresh_log_path,
        now_value=now_value,
    )
    baseline_params = baseline.get("baseline_params")
    if not isinstance(baseline_params, dict):
        baseline_params = {}

    state_row["iteration"] = int(state_row.get("iteration", 0)) + 1
    iteration = state_row["iteration"]

    proposal_timestamps = [
        float(x)
        for x in state_row.get("proposal_timestamps", [])
        if safe_decimal(x) is not None
    ]
    proposal_timestamps = _prune_hourly(proposal_timestamps, now_value)
    state_row["proposal_timestamps"] = proposal_timestamps

    events = _read_jsonl_events(journal_path)
    signals = _summarize_signals(
        events,
        now_ts_value=now_value,
        deadman_window_s=deadman_window_s,
        inventory_window_s=inventory_window_s,
    )
    updates, reason_codes, confidence, deadman, escalation_target = _compute_candidate_updates(
        current_env=env_map,
        baseline_params=baseline_params,
        signals=signals,
    )

    cooldown_until = float(safe_decimal(state_row.get("deadman_cooldown_until_ts")) or 0.0)
    if now_value < cooldown_until and not deadman:
        row = _proposal_row(
            proposal_id=f"{slugify(market)}-{int(now_value)}-{iteration}-cooldown",
            ts=now_value,
            market=market,
            env_path=env_path,
            iteration=iteration,
            key="*",
            old_value=None,
            new_value=None,
            baseline_value=None,
            reason_codes=["deadman_cooldown_active"],
            confidence="low",
            guardrail_status="rejected",
            guardrail_reason="deadman_cooldown_active",
            deadman=False,
            escalation_target="human",
            cooldown_until_ts=cooldown_until,
            rejected=True,
        )
        append_jsonl(proposals_path, row)
        append_jsonl(changelog_path, row)
        return {
            "market": market,
            "rows": [row],
            "signals": signals,
            "status": "cooldown",
        }

    rows: List[Dict[str, Any]] = []
    if not updates:
        row = _proposal_row(
            proposal_id=f"{slugify(market)}-{int(now_value)}-{iteration}-noop",
            ts=now_value,
            market=market,
            env_path=env_path,
            iteration=iteration,
            key="*",
            old_value=None,
            new_value=None,
            baseline_value=None,
            reason_codes=reason_codes,
            confidence=confidence,
            guardrail_status="rejected",
            guardrail_reason="no_updates",
            deadman=deadman,
            escalation_target=escalation_target,
            cooldown_until_ts=cooldown_until if cooldown_until > now_value else None,
            rejected=True,
        )
        rows.append(row)
    else:
        for key in sorted(updates.keys()):
            old_value = safe_decimal(env_map.get(key))
            new_value = updates[key]
            baseline_value = safe_decimal(baseline_params.get(key))

            if len(proposal_timestamps) >= max_changes_per_hour:
                row = _proposal_row(
                    proposal_id=f"{slugify(market)}-{int(now_value)}-{iteration}-{slugify(key)}",
                    ts=now_value,
                    market=market,
                    env_path=env_path,
                    iteration=iteration,
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    baseline_value=baseline_value,
                    reason_codes=reason_codes + ["hourly_cap"],
                    confidence=confidence,
                    guardrail_status="rejected",
                    guardrail_reason="hourly_cap_exceeded",
                    deadman=deadman,
                    escalation_target=escalation_target,
                    cooldown_until_ts=None,
                    rejected=True,
                )
                rows.append(row)
                continue

            ok, drift_reason = _drift_ok(
                key=key,
                proposed=new_value,
                baseline_value=baseline_value,
            )
            row = _proposal_row(
                proposal_id=f"{slugify(market)}-{int(now_value)}-{iteration}-{slugify(key)}",
                ts=now_value,
                market=market,
                env_path=env_path,
                iteration=iteration,
                key=key,
                old_value=old_value,
                new_value=new_value,
                baseline_value=baseline_value,
                reason_codes=reason_codes,
                confidence=confidence,
                guardrail_status="passed" if ok else "rejected",
                guardrail_reason=drift_reason,
                deadman=deadman,
                escalation_target=escalation_target,
                cooldown_until_ts=None,
                rejected=not ok,
            )
            rows.append(row)
            if ok:
                proposal_timestamps.append(now_value)

    if deadman and any(row.get("guardrail_status") == "passed" for row in rows):
        cooldown_until = now_value + deadman_cooldown_s
        state_row["deadman_cooldown_until_ts"] = cooldown_until
        for row in rows:
            if row.get("guardrail_status") == "passed":
                row["cooldown_until_ts"] = cooldown_until

    state_row["proposal_timestamps"] = _prune_hourly(proposal_timestamps, now_value)

    for row in rows:
        append_jsonl(proposals_path, row)
        append_jsonl(changelog_path, row)

    return {
        "market": market,
        "rows": rows,
        "signals": signals,
        "status": "ok",
    }


def _render_summary(cycle_rows: List[Dict[str, Any]], *, now_value: float) -> str:
    lines: List[str] = []
    lines.append("# MM Advisor Summary")
    lines.append(f"Generated: {iso_utc(now_value)}")
    lines.append("")
    lines.append("## Handoff Contract")
    lines.append("- `deadman=true` + `guardrail_status=passed` -> escalation_target=`warren` (auto-apply)")
    lines.append("- all other proposals -> escalation_target=`human`")
    lines.append("")
    lines.append("## Markets")
    if not cycle_rows:
        lines.append("- none")
        return "\n".join(lines)

    lines.append("| Market | Passed | Rejected | Deadman | Warren Auto Candidates |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in cycle_rows:
        proposals = row.get("rows", [])
        passed = sum(1 for p in proposals if p.get("guardrail_status") == "passed")
        rejected = sum(1 for p in proposals if p.get("guardrail_status") == "rejected")
        deadman = any(p.get("deadman") is True for p in proposals)
        warren = sum(1 for p in proposals if _warren_auto_apply_candidate(p))
        lines.append(
            f"| {row.get('market')} | {passed} | {rejected} | {str(deadman).lower()} | {warren} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Advisor-only MM config proposal loop.",
    )
    parser.add_argument("--repo", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument(
        "--jobs-json",
        default="/home/flexouille/.openclaw/cron/jobs.json",
        help="OpenClaw jobs.json path.",
    )
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument(
        "--advisor-dir",
        default="data/mm_audit/advisor",
        help="Advisor output directory.",
    )
    parser.add_argument(
        "--baselines-dir",
        default="data/mm_audit/autotune_baselines",
        help="Baseline store directory.",
    )
    parser.add_argument(
        "--config-changelog",
        default="data/mm_audit/config_changelog.jsonl",
        help="Shared changelog path.",
    )
    parser.add_argument(
        "--apply-receipts",
        default="data/mm_audit/advisor/apply_receipts.jsonl",
        help="Apply receipt stream path.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of cycles (0 or negative means infinite).",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=1800.0,
        help="Sleep interval between cycles in seconds.",
    )
    parser.add_argument(
        "--deadman-window-s",
        type=float,
        default=3600.0,
        help="Dead-man detection window in seconds.",
    )
    parser.add_argument(
        "--deadman-cooldown-s",
        type=float,
        default=1800.0,
        help="Cooldown period after dead-man proposal in seconds.",
    )
    parser.add_argument(
        "--inventory-window-s",
        type=float,
        default=7200.0,
        help="Inventory summary lookback in seconds.",
    )
    parser.add_argument(
        "--max-changes-per-hour",
        type=int,
        default=10,
        help="Max passed proposals per market per rolling hour.",
    )
    parser.add_argument("--market", default=None, help="Optional single market filter.")
    parser.add_argument(
        "--env-path",
        default=None,
        help="Optional env file filter (used by per-env controller wrappers).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    jobs_path = Path(args.jobs_json).resolve()
    journal_dir = (repo_root / args.journal_dir).resolve()
    advisor_dir = (repo_root / args.advisor_dir).resolve()
    baselines_dir = (repo_root / args.baselines_dir).resolve()
    changelog_path = (repo_root / args.config_changelog).resolve()
    apply_receipts_path = (repo_root / args.apply_receipts).resolve()
    env_path_filter = Path(args.env_path).resolve() if args.env_path else None

    advisor_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    changelog_path.touch(exist_ok=True)
    apply_receipts_path.parent.mkdir(parents=True, exist_ok=True)
    apply_receipts_path.touch(exist_ok=True)

    proposals_path = advisor_dir / "proposals.jsonl"
    proposals_path.touch(exist_ok=True)
    baseline_refresh_log_path = advisor_dir / "baseline_refresh_log.jsonl"
    baseline_refresh_log_path.touch(exist_ok=True)
    state_path = advisor_dir / "state.json"
    state = _load_state(state_path)

    cycle_idx = 0
    while True:
        cycle_idx += 1
        now_value = now_ts()
        cycle_rows: List[Dict[str, Any]] = []

        targets = _resolve_markets(
            repo_root=repo_root,
            jobs_path=jobs_path,
            journal_dir=journal_dir,
            env_path_filter=env_path_filter,
            market_filter=args.market,
        )

        for target in targets:
            market = str(target["market"])
            env_path = target.get("env_path")
            if not isinstance(env_path, Path):
                continue
            if not env_path.exists():
                continue
            latest_journal = _find_latest_journal(journal_dir, market)
            if latest_journal is None:
                continue

            row_state = _market_state(state, market)
            cycle_rows.append(
                _process_market(
                    market=market,
                    env_path=env_path,
                    journal_path=latest_journal,
                    baselines_dir=baselines_dir,
                    proposals_path=proposals_path,
                    changelog_path=changelog_path,
                    apply_receipts_path=apply_receipts_path,
                    baseline_refresh_log_path=baseline_refresh_log_path,
                    state_row=row_state,
                    now_value=now_value,
                    deadman_window_s=args.deadman_window_s,
                    deadman_cooldown_s=args.deadman_cooldown_s,
                    inventory_window_s=args.inventory_window_s,
                    max_changes_per_hour=args.max_changes_per_hour,
                )
            )

        _save_state(state_path, state)
        summary_text = _render_summary(cycle_rows, now_value=now_value)
        summary_path = advisor_dir / "latest_summary.md"
        summary_path.write_text(summary_text + "\n")
        dated_summary = advisor_dir / f"summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
        dated_summary.write_text(summary_text + "\n")

        print(f"cycle={cycle_idx} markets={len(cycle_rows)} summary={summary_path}")
        if args.iterations > 0 and cycle_idx >= args.iterations:
            break
        if args.iterations <= 0 and args.sleep_s <= 0:
            break
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
