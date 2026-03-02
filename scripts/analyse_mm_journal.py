#!/usr/bin/env python
"""
Analyse Market Maker Journal

Two modes:
1) Legacy single-journal analysis (existing behavior)
2) Analyst packet generation across one or more markets over a rolling window

Usage:
    PYTHONPATH=src python scripts/analyse_mm_journal.py data/mm_journal/mm_SOL-USD_*.jsonl
    PYTHONPATH=src python scripts/analyse_mm_journal.py data/mm_journal/
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python <3.9 fallback path
    ZoneInfo = None


def _d(v) -> Decimal:
    if v is None:
        return Decimal("0")
    return Decimal(str(v))


def _d_optional(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None


def _is_buy_side(side: Any) -> bool:
    return "BUY" in str(side).upper()


def _ts_fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ts_fmt_tz(ts: Optional[float], tz_name: str) -> Optional[str]:
    if ts is None:
        return None
    try:
        if ZoneInfo is not None:
            tz = ZoneInfo(tz_name)
            return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        pass
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _event_anchor(event: Dict[str, Any], anchor: str = "mid") -> Optional[Decimal]:
    anchor_key = str(anchor or "mid").strip().lower()

    if anchor_key in {"mid", "mid_price"}:
        if event.get("mid") is not None:
            return _d_optional(event.get("mid"))
        if event.get("mark_price") is not None:
            return _d_optional(event.get("mark_price"))
        bid = _d_optional(event.get("best_bid"))
        ask = _d_optional(event.get("best_ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal("2")
        return None

    if anchor_key in {"mark", "mark_price"}:
        if event.get("mark_price") is not None:
            return _d_optional(event.get("mark_price"))
        if event.get("mid") is not None:
            return _d_optional(event.get("mid"))
        bid = _d_optional(event.get("best_bid"))
        ask = _d_optional(event.get("best_ask"))
        if bid is not None and ask is not None:
            return (bid + ask) / Decimal("2")
        return None

    # Generic key fallback.
    value = _d_optional(event.get(anchor_key))
    if value is not None:
        return value

    # Keep diagnostics usable even if an unsupported anchor is requested.
    return _event_anchor(event, "mid")


def _event_mid(event: Dict[str, Any]) -> Optional[Decimal]:
    return _event_anchor(event, "mid")


def _build_mid_series(events: List[Dict[str, Any]]) -> Tuple[List[float], List[Decimal]]:
    ts_values: List[float] = []
    mid_values: List[Decimal] = []
    for event in events:
        mid = _event_mid(event)
        ts_raw = event.get("ts")
        if mid is None or ts_raw is None:
            continue
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            continue
        ts_values.append(ts)
        mid_values.append(mid)
    return ts_values, mid_values


def _mid_at_or_after(
    ts_values: List[float],
    mid_values: List[Decimal],
    target_ts: float,
    *,
    max_wait_s: float = 60.0,
) -> Optional[Decimal]:
    if not ts_values:
        return None
    idx = bisect.bisect_left(ts_values, target_ts)
    if idx >= len(ts_values):
        return None
    if ts_values[idx] - target_ts > max_wait_s:
        return None
    return mid_values[idx]


def _mid_at_or_before(
    ts_values: Sequence[float],
    mid_values: Sequence[Decimal],
    target_ts: float,
    *,
    max_age_s: float = 300.0,
) -> Optional[Decimal]:
    if not ts_values:
        return None
    idx = bisect.bisect_right(ts_values, target_ts) - 1
    if idx < 0:
        return None
    if target_ts - ts_values[idx] > max_age_s:
        return None
    return mid_values[idx]


def _latest_position(events: List[Dict[str, Any]]) -> Optional[Decimal]:
    for event in reversed(events):
        if event.get("position") is not None:
            return _d(event["position"])
    return None


def _first_position(events: List[Dict[str, Any]]) -> Optional[Decimal]:
    for event in events:
        if event.get("position") is not None:
            return _d(event["position"])
    return None


_EXPECTED_SCHEMA_VERSION = 2


def validate_schema_versions(events: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    version_counts: Dict[int, int] = Counter()
    for e in events:
        v = e.get("schema_version")
        if v is not None:
            version_counts[int(v)] += 1
        else:
            version_counts[-1] += 1
    all_valid = True
    for version, count in sorted(version_counts.items()):
        if version == -1:
            warnings.append(f"  {count} events have no schema_version (pre-v1 format)")
            all_valid = False
        elif version != _EXPECTED_SCHEMA_VERSION:
            warnings.append(f"  {count} events have schema_version={version} (expected {_EXPECTED_SCHEMA_VERSION})")
            all_valid = False
    return all_valid, warnings


def detect_heartbeat_gaps(events: List[Dict[str, Any]], max_gap_s: float = 60.0) -> List[Dict[str, Any]]:
    heartbeats = [e for e in events if e.get("type") == "heartbeat"]
    if len(heartbeats) < 2:
        return []
    gaps: List[Dict[str, Any]] = []
    for i in range(1, len(heartbeats)):
        prev_ts = float(heartbeats[i - 1].get("ts", 0))
        curr_ts = float(heartbeats[i].get("ts", 0))
        delta = curr_ts - prev_ts
        if delta > max_gap_s:
            gaps.append({
                "start_ts": prev_ts,
                "end_ts": curr_ts,
                "gap_s": delta,
                "start_iso": _ts_fmt(prev_ts),
                "end_iso": _ts_fmt(curr_ts),
            })
    return gaps


def load_journal(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def find_latest_journal(dir_path: Path) -> Path:
    files = sorted(dir_path.glob("mm_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"No journal files found in {dir_path}")
        sys.exit(1)
    return files[-1]


_TS_SUFFIX_RE = re.compile(r"^(.+)_\d{8}_\d{6}(?:\.\d+)?$")


def _parse_market_from_filename(path: Path) -> Optional[str]:
    name = path.name
    if not name.startswith("mm_") or not name.endswith(".jsonl"):
        return None
    if "mm_tuning_log_" in name:
        return None
    inner = name[3:-6]
    if inner.endswith("_latest"):
        market = inner[:-7]
    else:
        match = _TS_SUFFIX_RE.match(inner)
        market = match.group(1) if match else inner
    market = market.strip()
    if not market:
        return None
    return market.upper()


def discover_market_journals(journal_dir: Path) -> Dict[str, Path]:
    latest_files = sorted(journal_dir.glob("mm_*_latest.jsonl"), key=lambda p: p.stat().st_mtime)
    by_market: Dict[str, Path] = {}

    for path in latest_files:
        market = _parse_market_from_filename(path)
        if market:
            by_market[market] = path

    all_files = sorted(journal_dir.glob("mm_*.jsonl"), key=lambda p: p.stat().st_mtime)
    for path in all_files:
        market = _parse_market_from_filename(path)
        if not market:
            continue
        if market not in by_market:
            by_market[market] = path

    return dict(sorted(by_market.items(), key=lambda item: item[0]))


_MARKOUT_HORIZONS = [0.25, 1.0, 5.0, 30.0, 120.0]


def _horizon_label(horizon_s: float) -> str:
    if horizon_s == 0.25:
        return "250ms"
    if horizon_s >= 60:
        return f"{int(horizon_s / 60)}m"
    return f"{int(horizon_s)}s"


def _markout_for_fill(fill, horizon_s, ts_values, mid_values) -> Optional[Decimal]:
    fill_ts = float(fill.get("ts", 0.0))
    fill_px = _d(fill.get("price"))
    if fill_px <= 0:
        return None
    fut_mid = _mid_at_or_after(ts_values, mid_values, fill_ts + horizon_s)
    if fut_mid is None:
        return None
    side = str(fill.get("side", ""))
    if "BUY" in side:
        return (fut_mid - fill_px) / fill_px * Decimal("10000")
    return (fill_px - fut_mid) / fill_px * Decimal("10000")


def _markout_for_fill_locf(
    fill: Dict[str, Any],
    horizon_s: float,
    ts_values: Sequence[float],
    mid_values: Sequence[Decimal],
) -> Optional[Decimal]:
    fill_ts = float(fill.get("ts", 0.0))
    fill_px = _d_optional(fill.get("price"))
    if fill_px is None or fill_px <= 0:
        return None
    fut_mid = _mid_at_or_before(ts_values, mid_values, fill_ts + horizon_s)
    if fut_mid is None or fut_mid <= 0:
        return None
    if _is_buy_side(fill.get("side")):
        return (fut_mid - fill_px) / fill_px * Decimal("10000")
    return (fill_px - fut_mid) / fill_px * Decimal("10000")


def _fill_context_value(fill, key):
    snap = fill.get("market_snapshot")
    if isinstance(snap, dict):
        value = snap.get(key)
        if value is not None:
            return _d(value)
    if key == "spread_bps" and fill.get("spread_bps") is not None:
        return _d(fill.get("spread_bps"))
    return None


def _bucket_spread(v):
    if v is None:
        return None
    if v < Decimal("2"):
        return "<2"
    if v < Decimal("5"):
        return "2-5"
    if v < Decimal("10"):
        return "5-10"
    return ">10"


def _bucket_micro_vol(v):
    if v is None:
        return None
    if v < Decimal("2"):
        return "<2"
    if v < Decimal("5"):
        return "2-5"
    return ">5"


def _bucket_drift(v):
    if v is None:
        return None
    if v < Decimal("-1"):
        return "negative"
    if v > Decimal("1"):
        return "positive"
    return "neutral"


def _bucket_imbalance(v):
    if v is None:
        return None
    if v < Decimal("-0.30"):
        return "ask-heavy"
    if v > Decimal("0.30"):
        return "bid-heavy"
    return "balanced"


def _format_avg(values):
    if not values:
        return "n/a"
    return f"{(sum(values) / Decimal(len(values))):.2f}"


def _avg(values):
    if not values:
        return None
    return sum(values) / Decimal(len(values))


def _to_jsonable(v):
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dict):
        return {k: _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    return v


def _avg_decimal(values: Sequence[Decimal]) -> Optional[Decimal]:
    if not values:
        return None
    return sum(values) / Decimal(len(values))


def _percentile_decimal(values: Sequence[Decimal], p: float) -> Optional[Decimal]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def _median_decimal(values: Sequence[Decimal]) -> Optional[Decimal]:
    return _percentile_decimal(values, 0.5)


def _signed_qty(fill: Dict[str, Any]) -> Decimal:
    qty = _d(fill.get("qty"))
    return qty if _is_buy_side(fill.get("side")) else -qty


def _decimal_delta(before_value: Optional[Decimal], window_values: Sequence[Decimal]) -> Optional[Decimal]:
    if before_value is None and not window_values:
        return None
    if not window_values:
        return Decimal("0")
    start = before_value if before_value is not None else window_values[0]
    end = window_values[-1]
    return end - start


def _format_md_num(value: Optional[Decimal], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return f"{value}{suffix}"


def _extract_max_position_cap_from_event(event: Dict[str, Any]) -> Optional[Decimal]:
    if event.get("type") != "run_start":
        return None
    cfg = event.get("config")
    if not isinstance(cfg, dict):
        return None

    candidates: List[Decimal] = []
    for key in (
        "max_position_size",
        "max_long_position_size",
        "max_short_position_size",
        "MM_MAX_POSITION_SIZE",
        "MAX_POSITION_SIZE",
    ):
        value = _d_optional(cfg.get(key))
        if value is not None and value != 0:
            candidates.append(abs(value))

    if not candidates:
        return None
    return max(candidates)


def _extract_pnl_fields_from_event(event: Dict[str, Any]) -> Tuple[Optional[Decimal], Optional[Decimal]]:
    realized_keys = (
        "net_realized_pnl_usd",
        "net_realised_pnl_usd",
        "realized_pnl_usd",
        "realised_pnl_usd",
        "session_realized_pnl_usd",
    )
    total_keys = (
        "total_pnl_usd",
        "pnl_usd",
        "session_pnl_usd",
        "current_pnl",
    )

    realized = None
    total = None

    for key in realized_keys:
        value = _d_optional(event.get(key))
        if value is not None:
            realized = value
            break

    for key in total_keys:
        value = _d_optional(event.get(key))
        if value is not None:
            total = value
            break

    stats = event.get("stats")
    if isinstance(stats, dict):
        if realized is None:
            for key in realized_keys:
                value = _d_optional(stats.get(key))
                if value is not None:
                    realized = value
                    break
        if total is None:
            for key in total_keys:
                value = _d_optional(stats.get(key))
                if value is not None:
                    total = value
                    break

    return realized, total


def _extract_funding_from_event(event: Dict[str, Any]) -> Dict[str, Optional[Decimal]]:
    net_components: List[Decimal] = []
    paid = Decimal("0")
    rate: Optional[Decimal] = None

    # Net funding fields (signed contribution to pnl).
    for key in ("funding_pnl_usd", "funding_pnl", "funding_net_usd"):
        value = _d_optional(event.get(key))
        if value is not None:
            net_components.append(value)

    # Cost-like fields. Assume these are paid costs.
    for key in (
        "funding_paid_usd",
        "funding_fee_usd",
        "funding_fees_usd",
        "funding_payment_usd",
        "funding_payment",
    ):
        value = _d_optional(event.get(key))
        if value is None:
            continue
        paid += abs(value)
        net_components.append(-abs(value))

    # Some payloads may carry funding fee breakdowns.
    for nested_key in ("realized_pnl_breakdown", "realised_pnl_breakdown"):
        nested = event.get(nested_key)
        if isinstance(nested, dict):
            value = _d_optional(nested.get("funding_fees"))
            if value is not None:
                paid += abs(value)
                net_components.append(-abs(value))

    # Funding rate if present.
    for key in ("funding_rate", "current_funding_rate"):
        value = _d_optional(event.get(key))
        if value is not None:
            rate = value
            break

    if rate is None:
        snap = event.get("market_snapshot")
        if isinstance(snap, dict):
            for key in ("funding_rate", "current_funding_rate"):
                value = _d_optional(snap.get(key))
                if value is not None:
                    rate = value
                    break

    net = sum(net_components) if net_components else None
    paid_value = paid if paid > 0 else None
    return {"net": net, "paid": paid_value, "rate": rate}


def _normalize_mid_points(
    points: Sequence[Tuple[float, Decimal]],
    *,
    start_ts: float,
    last_mid_before_start: Optional[Decimal],
) -> List[Tuple[float, Decimal]]:
    if not points and last_mid_before_start is None:
        return []

    out: List[Tuple[float, Decimal]] = []
    for ts, mid in sorted(points, key=lambda item: item[0]):
        if out and abs(out[-1][0] - ts) < 1e-9:
            out[-1] = (ts, mid)
        else:
            out.append((ts, mid))

    if last_mid_before_start is not None:
        if not out or out[0][0] > start_ts:
            out.insert(0, (start_ts, last_mid_before_start))
    return out


def compute_spread_capture_usd(
    fills: Sequence[Dict[str, Any]],
    ts_values: Sequence[float],
    mid_values: Sequence[Decimal],
) -> Decimal:
    """Estimate spread capture using fill-vs-mid edge at fill time.

    Positive values indicate favorable maker capture.
    """
    spread_capture = Decimal("0")
    for fill in fills:
        qty = abs(_d(fill.get("qty")))
        fill_px = _d_optional(fill.get("price"))
        if fill_px is None or fill_px <= 0 or qty <= 0:
            continue

        mid_at_fill = _event_mid(fill)
        if mid_at_fill is None:
            try:
                fill_ts = float(fill.get("ts", 0.0))
            except (TypeError, ValueError):
                continue
            mid_at_fill = _mid_at_or_before(ts_values, mid_values, fill_ts)
        if mid_at_fill is None or mid_at_fill <= 0:
            continue

        if _is_buy_side(fill.get("side")):
            edge = mid_at_fill - fill_px
        else:
            edge = fill_px - mid_at_fill
        spread_capture += edge * qty

    return spread_capture


def compute_inventory_drift_usd(
    fills: Sequence[Dict[str, Any]],
    mid_points: Sequence[Tuple[float, Decimal]],
    *,
    start_position: Decimal,
) -> Optional[Decimal]:
    """Approximate inventory drift: sum(position(t_k) * Δmid_k)."""
    if len(mid_points) < 2:
        return None

    mids = sorted(mid_points, key=lambda item: item[0])
    fills_sorted = sorted(
        fills,
        key=lambda item: float(item.get("ts", 0.0)),
    )

    position = start_position
    fill_idx = 0
    drift = Decimal("0")

    for idx in range(len(mids) - 1):
        t0, mid0 = mids[idx]
        _t1, mid1 = mids[idx + 1]

        while fill_idx < len(fills_sorted):
            fill_ts = float(fills_sorted[fill_idx].get("ts", 0.0))
            if fill_ts > t0:
                break
            position += _signed_qty(fills_sorted[fill_idx])
            fill_idx += 1

        drift += position * (mid1 - mid0)

    return drift


def aggregate_markouts(
    fills: Sequence[Dict[str, Any]],
    ts_values: Sequence[float],
    mid_values: Sequence[Decimal],
    horizons_s: Sequence[float] = _MARKOUT_HORIZONS,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for horizon_s in horizons_s:
        label = _horizon_label(float(horizon_s))
        all_vals: List[Decimal] = []
        maker_buy_vals: List[Decimal] = []
        maker_sell_vals: List[Decimal] = []

        for fill in fills:
            markout = _markout_for_fill_locf(fill, float(horizon_s), ts_values, mid_values)
            if markout is None:
                continue
            all_vals.append(markout)

            if bool(fill.get("is_taker")):
                continue
            if _is_buy_side(fill.get("side")):
                maker_buy_vals.append(markout)
            else:
                maker_sell_vals.append(markout)

        toxicity = None
        if all_vals:
            adverse = sum(1 for value in all_vals if value < 0)
            toxicity = Decimal(adverse) / Decimal(len(all_vals))

        entry: Dict[str, Any] = {
            "mean_bps": _avg_decimal(all_vals),
            "median_bps": _median_decimal(all_vals),
            "count": len(all_vals),
            "toxicity_share": toxicity,
        }

        if abs(float(horizon_s) - 0.25) < 1e-9 or abs(float(horizon_s) - 1.0) < 1e-9:
            entry["maker_buy"] = {
                "mean_bps": _avg_decimal(maker_buy_vals),
                "median_bps": _median_decimal(maker_buy_vals),
                "count": len(maker_buy_vals),
            }
            entry["maker_sell"] = {
                "mean_bps": _avg_decimal(maker_sell_vals),
                "median_bps": _median_decimal(maker_sell_vals),
                "count": len(maker_sell_vals),
            }

        out[label] = entry

    return out


def diagnose_market(metrics: Dict[str, Any]) -> Dict[str, Any]:
    actions: List[str] = []
    severity = "SEV3"

    fills_count = int(metrics.get("fills_count") or 0)
    orders_sent = int(metrics.get("orders_sent") or 0)
    zero_fill_flag = bool(metrics.get("zero_fill_flag"))

    markout_250_mean = _d_optional(metrics.get("markout_250ms_mean"))
    toxicity_250 = _d_optional(metrics.get("toxicity_share_250ms"))
    lifetime_p10_ms = _d_optional(metrics.get("lifetime_p10_ms"))

    spread_capture_usd = _d_optional(metrics.get("spread_capture_usd")) or Decimal("0")
    inventory_drift_usd = _d_optional(metrics.get("inventory_drift_usd"))
    funding_usd = _d_optional(metrics.get("funding_usd"))
    churn_ratio = _d_optional(metrics.get("churn_ratio"))

    if fills_count == 0 and zero_fill_flag and orders_sent > 0:
        actions.append(
            "Too wide / not competitive: tighten offset/spread_multiplier slightly or reduce reprice barriers to sit in queue."
        )
        severity = "SEV2"

    toxic_condition = False
    if markout_250_mean is not None and markout_250_mean < Decimal("-1"):
        toxic_condition = True
    if (
        toxicity_250 is not None
        and toxicity_250 > Decimal("0.60")
        and lifetime_p10_ms is not None
        and lifetime_p10_ms < Decimal("200")
    ):
        toxic_condition = True

    if toxic_condition:
        actions.append(
            "Toxic/picked off: widen min_offset, increase min_reprice_interval and min_reprice_move_ticks, reduce size; disable market temporarily if persistent."
        )
        severity = "SEV1"

    if inventory_drift_usd is not None and inventory_drift_usd < 0:
        dominates_inventory = abs(inventory_drift_usd) > max(abs(spread_capture_usd) * Decimal("1.2"), Decimal("5"))
        if dominates_inventory:
            actions.append(
                "Inventory risk: increase skew_factor, reduce max_position_notional, increase skew_max_bps, tighten inventory_deadband."
            )
            if severity != "SEV1":
                severity = "SEV2"

    if funding_usd is not None and funding_usd < Decimal("-5"):
        actions.append(
            "Funding drag: reduce inventory hold, bias away from paying side, lower max_position cap."
        )
        if severity != "SEV1":
            severity = "SEV2"

    if churn_ratio is not None and churn_ratio > Decimal("15"):
        actions.append(
            "Over-churning: raise min_reprice_edge_delta_bps, increase min_reprice_move_ticks, increase min_reprice_interval."
        )
        if severity != "SEV1":
            severity = "SEV2"

    if not actions:
        actions.append("No single dominant failure mode detected in this window.")

    return {
        "severity": severity,
        "actions": actions,
    }


def _analyse_market_window(
    journal_path: Path,
    *,
    market_hint: str,
    window_minutes: int,
    anchor: str,
) -> Dict[str, Any]:
    latest_ts: Optional[float] = None
    max_position_cap: Optional[Decimal] = None
    market_name = market_hint

    # Pass 1: metadata and latest timestamp only.
    for event in _iter_jsonl(journal_path):
        ts_raw = event.get("ts")
        if ts_raw is None:
            continue
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            continue
        latest_ts = ts
        if event.get("market"):
            market_name = str(event.get("market")).upper()

        inferred_cap = _extract_max_position_cap_from_event(event)
        if inferred_cap is not None and inferred_cap > 0:
            max_position_cap = inferred_cap

    if latest_ts is None:
        return {
            "market": market_name,
            "journal_file": str(journal_path),
            "window": {
                "start_ts": None,
                "end_ts": None,
                "start_iso": None,
                "end_iso": None,
                "window_minutes": window_minutes,
            },
            "fills_count": 0,
            "fills_per_hour": Decimal("0"),
            "maker_filled_notional_usd": Decimal("0"),
            "avg_fill_size_usd": None,
            "p50_fill_size_usd": None,
            "orders_sent": 0,
            "cancels": 0,
            "replaces": 0,
            "churn_ratio": Decimal("0"),
            "quote_lifetime_ms": {"p10": None, "p50": None, "p90": None, "count": 0},
            "markout": {},
            "toxicity_share": {"250ms": None, "1s": None},
            "competitiveness": {"zero_fill_flag": False, "instant_fill_flag": False},
            "inventory": {
                "current_position": None,
                "max_abs_position": None,
                "max_position_cap": max_position_cap,
                "inventory_utilization": None,
            },
            "funding": {"funding_paid_usd": None, "funding_rate_avg": None},
            "attribution": {
                "spread_capture_usd": Decimal("0"),
                "inventory_drift_usd": None,
                "funding_usd": None,
                "net_realized_pnl_usd": None,
                "net_pnl_source": None,
                "residual_usd": None,
                "spread_capture_is_estimate": True,
            },
            "diagnosis": {
                "severity": "SEV3",
                "actions": ["No events found in journal."],
            },
        }

    window_seconds = max(1, int(window_minutes)) * 60.0
    start_ts = latest_ts - window_seconds
    end_ts = latest_ts
    max_horizon_s = max(_MARKOUT_HORIZONS)

    fills: List[Dict[str, Any]] = []
    mid_points_raw: List[Tuple[float, Decimal]] = []
    last_mid_before_start: Optional[Decimal] = None

    orders_sent = 0
    cancels = 0
    replaces = 0

    funding_paid_usd = Decimal("0")
    funding_net_samples: List[Decimal] = []
    funding_rate_samples: List[Decimal] = []

    last_position_before_start: Optional[Decimal] = None
    last_position_to_end: Optional[Decimal] = None
    positions_in_window: List[Decimal] = []

    realized_before_start: Optional[Decimal] = None
    realized_window_values: List[Decimal] = []
    total_before_start: Optional[Decimal] = None
    total_window_values: List[Decimal] = []

    for event in _iter_jsonl(journal_path):
        ts_raw = event.get("ts")
        if ts_raw is None:
            continue
        try:
            ts = float(ts_raw)
        except (TypeError, ValueError):
            continue

        if event.get("market"):
            market_name = str(event.get("market")).upper()

        # Keep max position cap updated if multiple run starts exist.
        inferred_cap = _extract_max_position_cap_from_event(event)
        if inferred_cap is not None and inferred_cap > 0:
            max_position_cap = inferred_cap

        anchor_price = _event_anchor(event, anchor)
        if anchor_price is not None:
            if ts < start_ts:
                last_mid_before_start = anchor_price
            elif ts <= end_ts + max_horizon_s:
                mid_points_raw.append((ts, anchor_price))

        position_value = _d_optional(event.get("position"))
        if position_value is not None:
            if ts < start_ts:
                last_position_before_start = position_value
            if ts <= end_ts:
                last_position_to_end = position_value
            if start_ts <= ts <= end_ts:
                positions_in_window.append(position_value)

        if ts <= end_ts:
            realized_value, total_value = _extract_pnl_fields_from_event(event)
            if realized_value is not None:
                if ts < start_ts:
                    realized_before_start = realized_value
                else:
                    realized_window_values.append(realized_value)
            if total_value is not None:
                if ts < start_ts:
                    total_before_start = total_value
                else:
                    total_window_values.append(total_value)

        if not (start_ts <= ts <= end_ts):
            continue

        funding_info = _extract_funding_from_event(event)
        funding_net = funding_info.get("net")
        if funding_net is not None:
            funding_net_samples.append(funding_net)
        funding_paid = funding_info.get("paid")
        if funding_paid is not None:
            funding_paid_usd += funding_paid
        funding_rate = funding_info.get("rate")
        if funding_rate is not None:
            funding_rate_samples.append(funding_rate)

        event_type = str(event.get("type") or "")
        if event_type == "fill":
            fills.append({
                "ts": ts,
                "side": event.get("side"),
                "price": event.get("price"),
                "qty": event.get("qty"),
                "is_taker": bool(event.get("is_taker")),
                "quote_lifetime_ms": event.get("quote_lifetime_ms"),
                "position": event.get("position"),
                "mid": event.get("mid"),
                "best_bid": event.get("best_bid"),
                "best_ask": event.get("best_ask"),
                "market_snapshot": event.get("market_snapshot"),
            })
        elif event_type == "order_placed":
            orders_sent += 1
        elif event_type == "order_cancelled":
            cancels += 1
        elif event_type in {"order_replaced", "order_replace"}:
            replaces += 1
        elif event_type == "reprice_decision":
            reason = str(event.get("reason") or "")
            if reason.startswith("replace_"):
                replaces += 1

    fills.sort(key=lambda row: float(row.get("ts", 0.0)))

    mid_points = _normalize_mid_points(
        mid_points_raw,
        start_ts=start_ts,
        last_mid_before_start=last_mid_before_start,
    )
    ts_values = [ts for ts, _mid in mid_points]
    mid_values = [_mid for _ts, _mid in mid_points]

    fill_notionals: List[Decimal] = []
    maker_notional = Decimal("0")
    taker_notional = Decimal("0")
    taker_fills = 0
    lifetimes_ms: List[Decimal] = []

    for fill in fills:
        qty = abs(_d(fill.get("qty")))
        price = _d_optional(fill.get("price"))
        if price is None or price <= 0 or qty <= 0:
            continue
        notional = qty * price
        fill_notionals.append(notional)
        if bool(fill.get("is_taker")):
            taker_fills += 1
            taker_notional += notional
        else:
            maker_notional += notional

        lifetime = _d_optional(fill.get("quote_lifetime_ms"))
        if lifetime is not None and lifetime >= 0:
            lifetimes_ms.append(lifetime)

    fills_count = len(fills)
    maker_fills = fills_count - taker_fills

    window_hours = Decimal(str(window_minutes)) / Decimal("60")
    fills_per_hour = (
        Decimal(fills_count) / window_hours if window_hours > 0 else Decimal("0")
    )

    avg_fill_size_usd = _avg_decimal(fill_notionals)
    p50_fill_size_usd = _median_decimal(fill_notionals)

    churn_ratio = Decimal(cancels + replaces) / Decimal(max(1, fills_count))

    lifetime_p10 = _percentile_decimal(lifetimes_ms, 0.10)
    lifetime_p50 = _median_decimal(lifetimes_ms)
    lifetime_p90 = _percentile_decimal(lifetimes_ms, 0.90)

    markout = aggregate_markouts(fills, ts_values, mid_values, _MARKOUT_HORIZONS)
    markout_250 = markout.get("250ms", {})
    markout_1s = markout.get("1s", {})

    toxicity_250 = markout_250.get("toxicity_share")
    toxicity_1s = markout_1s.get("toxicity_share")

    high_orders_threshold = max(20, int(window_minutes) * 2)
    zero_fill_flag = fills_count == 0 and orders_sent >= high_orders_threshold
    instant_fill_flag = lifetime_p10 is not None and lifetime_p10 < Decimal("200")

    start_position = last_position_before_start
    if start_position is None and fills:
        first_fill = fills[0]
        pos_after_first = _d_optional(first_fill.get("position"))
        if pos_after_first is not None:
            start_position = pos_after_first - _signed_qty(first_fill)
    if start_position is None:
        start_position = Decimal("0")

    reconstructed_position = start_position
    reconstructed_max_abs = abs(start_position)
    for fill in fills:
        reconstructed_position += _signed_qty(fill)
        reconstructed_max_abs = max(reconstructed_max_abs, abs(reconstructed_position))

    current_position = last_position_to_end if last_position_to_end is not None else reconstructed_position

    explicit_max_abs: Optional[Decimal] = None
    if positions_in_window:
        explicit_max_abs = max(abs(value) for value in positions_in_window)

    if explicit_max_abs is None:
        max_abs_position = reconstructed_max_abs
    else:
        max_abs_position = max(explicit_max_abs, reconstructed_max_abs)

    inventory_utilization = None
    if max_position_cap is not None and max_position_cap > 0:
        inventory_utilization = max_abs_position / max_position_cap

    spread_capture_usd = compute_spread_capture_usd(fills, ts_values, mid_values)

    drift_points = [(ts, mid) for ts, mid in mid_points if start_ts <= ts <= end_ts]
    if last_mid_before_start is not None and (not drift_points or drift_points[0][0] > start_ts):
        drift_points.insert(0, (start_ts, last_mid_before_start))

    inventory_drift_usd = compute_inventory_drift_usd(
        fills,
        drift_points,
        start_position=start_position,
    )

    funding_usd = sum(funding_net_samples) if funding_net_samples else None
    funding_rate_avg = _avg_decimal(funding_rate_samples)
    funding_paid_value = funding_paid_usd if funding_paid_usd > 0 else None

    realized_delta = _decimal_delta(realized_before_start, realized_window_values)
    total_delta = _decimal_delta(total_before_start, total_window_values)
    net_pnl_source = None
    net_realized_pnl_usd = None
    if realized_delta is not None:
        net_realized_pnl_usd = realized_delta
        net_pnl_source = "realized"
    elif total_delta is not None:
        net_realized_pnl_usd = total_delta
        net_pnl_source = "total"

    residual_usd = None
    if (
        net_realized_pnl_usd is not None
        and inventory_drift_usd is not None
        and funding_usd is not None
    ):
        residual_usd = net_realized_pnl_usd - (
            spread_capture_usd + inventory_drift_usd + funding_usd
        )

    diagnosis = diagnose_market({
        "fills_count": fills_count,
        "orders_sent": orders_sent,
        "zero_fill_flag": zero_fill_flag,
        "markout_250ms_mean": markout_250.get("mean_bps"),
        "toxicity_share_250ms": toxicity_250,
        "lifetime_p10_ms": lifetime_p10,
        "spread_capture_usd": spread_capture_usd,
        "inventory_drift_usd": inventory_drift_usd,
        "funding_usd": funding_usd,
        "churn_ratio": churn_ratio,
    })

    return {
        "market": market_name,
        "journal_file": str(journal_path),
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_iso": _ts_fmt(start_ts),
            "end_iso": _ts_fmt(end_ts),
            "window_minutes": int(window_minutes),
        },
        "fills_count": fills_count,
        "fills_per_hour": fills_per_hour,
        "maker_filled_notional_usd": maker_notional,
        "taker_filled_notional_usd": taker_notional if taker_fills > 0 else None,
        "maker_fills_count": maker_fills,
        "taker_fills_count": taker_fills if taker_fills > 0 else None,
        "avg_fill_size_usd": avg_fill_size_usd,
        "p50_fill_size_usd": p50_fill_size_usd,
        "orders_sent": orders_sent,
        "cancels": cancels,
        "replaces": replaces,
        "churn_ratio": churn_ratio,
        "quote_lifetime_ms": {
            "p10": lifetime_p10,
            "p50": lifetime_p50,
            "p90": lifetime_p90,
            "count": len(lifetimes_ms),
        },
        "markout": markout,
        "toxicity_share": {
            "250ms": toxicity_250,
            "1s": toxicity_1s,
        },
        "competitiveness": {
            "zero_fill_flag": zero_fill_flag,
            "instant_fill_flag": instant_fill_flag,
        },
        "inventory": {
            "current_position": current_position,
            "start_position": start_position,
            "max_abs_position": max_abs_position,
            "max_position_cap": max_position_cap,
            "inventory_utilization": inventory_utilization,
        },
        "funding": {
            "funding_paid_usd": funding_paid_value,
            "funding_rate_avg": funding_rate_avg,
        },
        "attribution": {
            "spread_capture_usd": spread_capture_usd,
            "inventory_drift_usd": inventory_drift_usd,
            "funding_usd": funding_usd,
            "net_realized_pnl_usd": net_realized_pnl_usd,
            "net_pnl_source": net_pnl_source,
            "residual_usd": residual_usd,
            "spread_capture_is_estimate": True,
        },
        "diagnosis": diagnosis,
    }


def build_analyst_packet(
    target: Path,
    *,
    window_minutes: int,
    markets_filter: Optional[Sequence[str]],
    tz_name: str,
    anchor: str,
) -> Dict[str, Any]:
    target = target.resolve()
    requested_markets = {
        m.strip().upper()
        for m in (markets_filter or [])
        if m and m.strip()
    }

    market_journals: Dict[str, Path] = {}
    if target.is_file():
        guessed = _parse_market_from_filename(target) or target.stem.upper()
        market_journals[guessed] = target
    else:
        market_journals = discover_market_journals(target)

    if requested_markets:
        market_journals = {
            market: path
            for market, path in market_journals.items()
            if market in requested_markets
        }

    market_payload: Dict[str, Dict[str, Any]] = {}
    for market, path in sorted(market_journals.items()):
        analysis = _analyse_market_window(
            path,
            market_hint=market,
            window_minutes=window_minutes,
            anchor=anchor,
        )
        final_market_name = str(analysis.get("market") or market).upper()
        market_payload[final_market_name] = analysis

    missing_markets = []
    if requested_markets:
        missing_markets = sorted(m for m in requested_markets if m not in market_payload)

    starts = [
        data.get("window", {}).get("start_ts")
        for data in market_payload.values()
        if data.get("window", {}).get("start_ts") is not None
    ]
    ends = [
        data.get("window", {}).get("end_ts")
        for data in market_payload.values()
        if data.get("window", {}).get("end_ts") is not None
    ]

    global_start = min(starts) if starts else None
    global_end = max(ends) if ends else None

    total_fills = sum(int(data.get("fills_count") or 0) for data in market_payload.values())
    total_orders = sum(int(data.get("orders_sent") or 0) for data in market_payload.values())
    total_maker_notional = sum(
        (_d_optional(data.get("maker_filled_notional_usd")) or Decimal("0"))
        for data in market_payload.values()
    )

    total_spread_capture = sum(
        (_d_optional(data.get("attribution", {}).get("spread_capture_usd")) or Decimal("0"))
        for data in market_payload.values()
    )

    inventory_drift_values = [
        _d_optional(data.get("attribution", {}).get("inventory_drift_usd"))
        for data in market_payload.values()
    ]
    funding_values = [
        _d_optional(data.get("attribution", {}).get("funding_usd"))
        for data in market_payload.values()
    ]
    net_values = [
        _d_optional(data.get("attribution", {}).get("net_realized_pnl_usd"))
        for data in market_payload.values()
    ]

    total_inventory_drift = None
    if any(v is not None for v in inventory_drift_values):
        total_inventory_drift = sum(v for v in inventory_drift_values if v is not None)

    total_funding = None
    if any(v is not None for v in funding_values):
        total_funding = sum(v for v in funding_values if v is not None)

    total_net = None
    if any(v is not None for v in net_values):
        total_net = sum(v for v in net_values if v is not None)

    total_residual = None
    if (
        total_net is not None
        and total_inventory_drift is not None
        and total_funding is not None
    ):
        total_residual = total_net - (
            total_spread_capture + total_inventory_drift + total_funding
        )

    return {
        "generated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": {
            "window_minutes": int(window_minutes),
            "start_ts": global_start,
            "end_ts": global_end,
            "start_iso": _ts_fmt_tz(global_start, tz_name),
            "end_iso": _ts_fmt_tz(global_end, tz_name),
            "timezone": tz_name,
            "anchor": anchor,
        },
        "markets": dict(sorted(market_payload.items(), key=lambda item: item[0])),
        "totals": {
            "markets": len(market_payload),
            "fills_count": total_fills,
            "orders_sent": total_orders,
            "maker_filled_notional_usd": total_maker_notional,
            "attribution": {
                "spread_capture_usd": total_spread_capture,
                "inventory_drift_usd": total_inventory_drift,
                "funding_usd": total_funding,
                "net_realized_pnl_usd": total_net,
                "residual_usd": total_residual,
                "inventory_drift_missing_markets": sum(1 for v in inventory_drift_values if v is None),
                "funding_missing_markets": sum(1 for v in funding_values if v is None),
                "net_pnl_missing_markets": sum(1 for v in net_values if v is None),
                "spread_capture_is_estimate": True,
            },
        },
        "missing_markets": missing_markets,
    }


def _summary_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, float, float]:
    _market, payload = item
    attribution = payload.get("attribution", {})
    net_pnl = _d_optional(attribution.get("net_realized_pnl_usd"))
    markout_1s_mean = _d_optional(payload.get("markout", {}).get("1s", {}).get("mean_bps"))

    if net_pnl is not None:
        return (0, float(net_pnl), float(markout_1s_mean) if markout_1s_mean is not None else 0.0)

    if markout_1s_mean is not None:
        return (1, float(markout_1s_mean), 0.0)

    return (2, 0.0, 0.0)


def render_analyst_packet_markdown(packet: Dict[str, Any]) -> str:
    lines: List[str] = []
    window = packet.get("window", {})
    totals = packet.get("totals", {})
    markets = packet.get("markets", {})

    lines.append("# Market-Maker Analyst Packet")
    lines.append(f"Generated: {packet.get('generated_at')}")
    lines.append(
        f"Window: {window.get('window_minutes')}m | "
        f"{window.get('start_iso') or 'N/A'} -> {window.get('end_iso') or 'N/A'} | "
        f"anchor={window.get('anchor', 'mid')}"
    )
    lines.append("")

    if packet.get("missing_markets"):
        lines.append("## Missing Markets")
        lines.append(f"- {', '.join(packet['missing_markets'])}")
        lines.append("")

    lines.append("## Summary")
    lines.append(
        "| Market | Fills | Fills/hr | Maker Notional | MO250 mean | MO1s mean | "
        "SpreadCap | InvDrift | Funding | NetΔPnL | Severity |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for market, data in sorted(markets.items(), key=_summary_sort_key):
        markout = data.get("markout", {})
        attr = data.get("attribution", {})
        diagnosis = data.get("diagnosis", {})
        lines.append(
            "| {market} | {fills} | {fills_per_hour} | {maker_notional} | {mo250} | {mo1s} | "
            "{spread} | {drift} | {funding} | {net} | {sev} |".format(
                market=market,
                fills=int(data.get("fills_count") or 0),
                fills_per_hour=_format_md_num(_d_optional(data.get("fills_per_hour")), 2),
                maker_notional=_format_md_num(_d_optional(data.get("maker_filled_notional_usd")), 2),
                mo250=_format_md_num(_d_optional(markout.get("250ms", {}).get("mean_bps")), 2, "bps"),
                mo1s=_format_md_num(_d_optional(markout.get("1s", {}).get("mean_bps")), 2, "bps"),
                spread=_format_md_num(_d_optional(attr.get("spread_capture_usd")), 2),
                drift=_format_md_num(_d_optional(attr.get("inventory_drift_usd")), 2),
                funding=_format_md_num(_d_optional(attr.get("funding_usd")), 2),
                net=_format_md_num(_d_optional(attr.get("net_realized_pnl_usd")), 2),
                sev=diagnosis.get("severity", "SEV3"),
            )
        )

    lines.append("")
    lines.append("## Total Attribution")
    totals_attr = totals.get("attribution", {})
    lines.append("- Spread capture (estimate, vs mid-at-fill): " + _format_md_num(_d_optional(totals_attr.get("spread_capture_usd")), 2, " USD"))
    lines.append("- Inventory drift: " + _format_md_num(_d_optional(totals_attr.get("inventory_drift_usd")), 2, " USD"))
    lines.append("- Funding: " + _format_md_num(_d_optional(totals_attr.get("funding_usd")), 2, " USD"))
    lines.append("- Net realized/total PnL change: " + _format_md_num(_d_optional(totals_attr.get("net_realized_pnl_usd")), 2, " USD"))
    lines.append("- Residual: " + _format_md_num(_d_optional(totals_attr.get("residual_usd")), 2, " USD"))
    lines.append("")

    for market, data in sorted(markets.items(), key=_summary_sort_key):
        lines.append(f"## {market}")
        window_info = data.get("window", {})
        lines.append(
            f"Window: {window_info.get('start_iso') or 'N/A'} -> {window_info.get('end_iso') or 'N/A'} "
            f"({window_info.get('window_minutes', 'N/A')}m)"
        )

        lines.append("### KPIs")
        lines.append(f"- fills_count: {int(data.get('fills_count') or 0)}")
        lines.append(f"- fills_per_hour: {_format_md_num(_d_optional(data.get('fills_per_hour')), 2)}")
        lines.append(f"- maker_filled_notional_usd: {_format_md_num(_d_optional(data.get('maker_filled_notional_usd')), 2)}")
        lines.append(f"- avg_fill_size_usd: {_format_md_num(_d_optional(data.get('avg_fill_size_usd')), 2)}")
        lines.append(f"- p50_fill_size_usd: {_format_md_num(_d_optional(data.get('p50_fill_size_usd')), 2)}")
        lines.append(
            "- orders_sent/cancels/replaces: "
            f"{int(data.get('orders_sent') or 0)}/{int(data.get('cancels') or 0)}/{int(data.get('replaces') or 0)}"
        )
        lines.append(f"- churn_ratio: {_format_md_num(_d_optional(data.get('churn_ratio')), 2)}")

        lifetime = data.get("quote_lifetime_ms", {})
        lines.append(
            "- quote_lifetime_ms p10/p50/p90: "
            f"{_format_md_num(_d_optional(lifetime.get('p10')), 1, 'ms')} / "
            f"{_format_md_num(_d_optional(lifetime.get('p50')), 1, 'ms')} / "
            f"{_format_md_num(_d_optional(lifetime.get('p90')), 1, 'ms')}"
        )

        comp = data.get("competitiveness", {})
        lines.append(
            f"- competitiveness flags: zero_fill_flag={bool(comp.get('zero_fill_flag'))}, "
            f"instant_fill_flag={bool(comp.get('instant_fill_flag'))}"
        )

        inventory = data.get("inventory", {})
        lines.append(
            "- inventory current/max_abs/utilization: "
            f"{_format_md_num(_d_optional(inventory.get('current_position')), 6)} / "
            f"{_format_md_num(_d_optional(inventory.get('max_abs_position')), 6)} / "
            f"{_format_md_num(_d_optional(inventory.get('inventory_utilization')), 4)}"
        )

        funding = data.get("funding", {})
        lines.append(
            "- funding paid/rate_avg: "
            f"{_format_md_num(_d_optional(funding.get('funding_paid_usd')), 2)} / "
            f"{_format_md_num(_d_optional(funding.get('funding_rate_avg')), 6)}"
        )

        lines.append("### Markout")
        lines.append("| Horizon | Mean | Median | Toxicity(<0) | N |")
        lines.append("|---|---:|---:|---:|---:|")
        for horizon in ("250ms", "1s", "5s", "30s", "2m"):
            row = data.get("markout", {}).get(horizon, {})
            lines.append(
                "| {h} | {mean} | {median} | {tox} | {n} |".format(
                    h=horizon,
                    mean=_format_md_num(_d_optional(row.get("mean_bps")), 2, "bps"),
                    median=_format_md_num(_d_optional(row.get("median_bps")), 2, "bps"),
                    tox=_format_md_num(_d_optional(row.get("toxicity_share")), 3),
                    n=int(row.get("count") or 0),
                )
            )

        for horizon in ("250ms", "1s"):
            row = data.get("markout", {}).get(horizon, {})
            buy = row.get("maker_buy", {})
            sell = row.get("maker_sell", {})
            lines.append(
                f"- {horizon} maker side split: "
                f"BUY mean={_format_md_num(_d_optional(buy.get('mean_bps')), 2, 'bps')} "
                f"median={_format_md_num(_d_optional(buy.get('median_bps')), 2, 'bps')} n={int(buy.get('count') or 0)} | "
                f"SELL mean={_format_md_num(_d_optional(sell.get('mean_bps')), 2, 'bps')} "
                f"median={_format_md_num(_d_optional(sell.get('median_bps')), 2, 'bps')} n={int(sell.get('count') or 0)}"
            )

        lines.append("### Attribution")
        attr = data.get("attribution", {})
        lines.append("- Spread capture estimate (vs mid-at-fill): " + _format_md_num(_d_optional(attr.get("spread_capture_usd")), 2, " USD"))
        lines.append("- Inventory drift: " + _format_md_num(_d_optional(attr.get("inventory_drift_usd")), 2, " USD"))
        lines.append("- Funding: " + _format_md_num(_d_optional(attr.get("funding_usd")), 2, " USD"))
        source = attr.get("net_pnl_source") or "N/A"
        lines.append(
            "- Net realized/total PnL change: "
            + _format_md_num(_d_optional(attr.get("net_realized_pnl_usd")), 2, " USD")
            + f" (source={source})"
        )
        lines.append("- Residual: " + _format_md_num(_d_optional(attr.get("residual_usd")), 2, " USD"))

        diagnosis = data.get("diagnosis", {})
        lines.append(f"### Diagnosis ({diagnosis.get('severity', 'SEV3')})")
        for action in diagnosis.get("actions", []):
            lines.append(f"- {action}")

        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _default_packet_output_path(packet: Dict[str, Any]) -> Path:
    end_ts = packet.get("window", {}).get("end_ts")
    if end_ts is not None:
        stamp = datetime.fromtimestamp(float(end_ts), tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    else:
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("data/mm_audit/advisor") / f"analyst_packet_{stamp}.md"


def build_summary(events, path, assumed_fee_bps):
    fills = [e for e in events if e.get("type") == "fill"]
    orders = [e for e in events if e.get("type") == "order_placed"]
    snapshots = [e for e in events if e.get("type") == "snapshot"]
    ts_values, mid_values = _build_mid_series(events)
    market = events[0].get("market", "?") if events else "?"
    start_ts = float(events[0].get("ts", 0.0)) if events else None
    end_ts = float(events[-1].get("ts", 0.0)) if events else None
    duration_s = (end_ts - start_ts) if (start_ts and end_ts) else 0.0
    edge_values = [_d(f.get("edge_bps")) for f in fills if f.get("edge_bps") is not None]
    adverse_fill_count = sum(1 for v in edge_values if v < 0)
    adverse_fill_ratio = Decimal(adverse_fill_count) / Decimal(len(edge_values)) if edge_values else None
    markout_values = {h: [] for h in _MARKOUT_HORIZONS}
    for fill in fills:
        for h in markout_values:
            m = _markout_for_fill(fill, float(h), ts_values, mid_values)
            if m is not None:
                markout_values[h].append(m)
    fill_rate_pct = Decimal(len(fills)) / Decimal(len(orders)) * Decimal("100") if orders else None
    taker_count = sum(1 for f in fills if f.get("is_taker"))
    taker_notional = sum(_d(f.get("qty")) * _d(f.get("price")) for f in fills if f.get("is_taker"))
    total_notional = sum(_d(f.get("qty")) * _d(f.get("price")) for f in fills)
    taker_notional_ratio = taker_notional / total_notional if total_notional > 0 else Decimal("0")
    quote_lifetimes = [_d(f.get("quote_lifetime_ms")) for f in fills if f.get("quote_lifetime_ms") is not None]
    return {
        "market": market,
        "journal_file": str(path),
        "assumed_fee_bps": assumed_fee_bps,
        "counts": {
            "events": len(events),
            "fills": len(fills),
            "orders": len(orders),
            "snapshots": len(snapshots),
        },
        "window": {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "start_iso": _ts_fmt(start_ts) if start_ts else None,
            "end_iso": _ts_fmt(end_ts) if end_ts else None,
            "duration_s": duration_s,
        },
        "metrics": {
            "fill_rate_pct": fill_rate_pct,
            "avg_edge_bps": _avg(edge_values),
            "adverse_fill_count": adverse_fill_count,
            "adverse_fill_ratio": adverse_fill_ratio,
            "final_position": _latest_position(events),
            "markout_250ms_bps": _avg(markout_values[0.25]),
            "markout_1s_bps": _avg(markout_values[1.0]),
            "markout_5s_bps": _avg(markout_values[5.0]),
            "markout_30s_bps": _avg(markout_values[30.0]),
            "markout_2m_bps": _avg(markout_values[120.0]),
            "markout_counts": {_horizon_label(h): len(markout_values[h]) for h in _MARKOUT_HORIZONS},
            "taker_fill_count": taker_count,
            "taker_fill_notional_ratio": taker_notional_ratio,
            "quote_lifetime_ms_avg": _avg(quote_lifetimes),
            "quote_lifetime_ms_count": len(quote_lifetimes),
        },
    }


def analyse(events, path, assumed_fee_bps):
    from journal_analysis_sections import (
        build_fill_section,
        build_order_section,
        build_reprice_section,
    )

    fills = [e for e in events if e["type"] == "fill"]
    orders = [e for e in events if e["type"] == "order_placed"]
    reprice_decisions = [e for e in events if e["type"] == "reprice_decision"]
    snapshots = [e for e in events if e["type"] == "snapshot"]
    rejections = [e for e in events if e["type"] == "rejection"]
    cancellations = [e for e in events if e["type"] == "order_cancelled"]
    ts_values, mid_values = _build_mid_series(events)

    if not events:
        return "Empty journal."

    market = events[0].get("market", "?")
    t0 = events[0]["ts"]
    t1 = events[-1]["ts"]
    duration_m = (t1 - t0) / 60

    lines: list[str] = []
    lines.append(f"# MM Journal Analysis: {market}")
    lines.append(f"File: {path.name}")
    lines.append(f"Period: {_ts_fmt(t0)} -> {_ts_fmt(t1)}  ({duration_m:.0f} min)")
    lines.append(
        f"Events: {len(events)} total "
        f"({len(fills)} fills, {len(orders)} placements, {len(snapshots)} snapshots)"
    )
    lines.append("")

    schema_valid, schema_warnings = validate_schema_versions(events)
    if not schema_valid:
        lines.append("## Schema Warnings")
        for w in schema_warnings:
            lines.append(w)
        lines.append("  Some analysis sections may be inaccurate for incompatible events.")
        lines.append("")

    hb_gaps = detect_heartbeat_gaps(events)
    if hb_gaps:
        lines.append("## Heartbeat Gaps (potential outages)")
        for gap in hb_gaps:
            lines.append(f"  {gap['start_iso']} -> {gap['end_iso']}  ({gap['gap_s']:.0f}s gap)")
        lines.append("")

    lines.extend(build_fill_section(fills, events, ts_values, mid_values, assumed_fee_bps))
    lines.extend(build_order_section(orders, fills, rejections, cancellations, duration_m))
    lines.extend(build_reprice_section(reprice_decisions))

    if snapshots:
        lines.append("## Spread Over Time (from snapshots)")
        spreads = [_d(s.get("spread_bps")) for s in snapshots if s.get("spread_bps") is not None]
        if spreads:
            lines.append(
                f"  Samples: {len(spreads)}  avg={sum(spreads) / len(spreads):.1f}bps  "
                f"min={min(spreads):.1f}  max={max(spreads):.1f}"
            )
        positions = [_d(s.get("position")) for s in snapshots]
        if positions:
            lines.append(f"  Position range: [{min(positions)}, {max(positions)}]")
        lines.append("")

    if orders and fills:
        ratio = len(fills) / len(orders) * 100
        lines.append(f"## Fill Rate: {ratio:.1f}%  ({len(fills)} fills / {len(orders)} orders)")
        lines.append("")

    return "\n".join(lines)


def _parse_markets_arg(markets_arg: Optional[str]) -> List[str]:
    if not markets_arg:
        return []
    out: List[str] = []
    for part in markets_arg.split(","):
        market = part.strip().upper()
        if market:
            out.append(market)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse market-maker journal JSONL and produce a shareable report.",
    )
    parser.add_argument("target", help="Journal file or directory containing mm_*.jsonl")
    parser.add_argument("--assumed-fee-bps", type=Decimal, default=None)
    parser.add_argument("--json-out", default=None, help="Legacy alias for --out-json")

    # Analyst packet mode options.
    parser.add_argument("--window-minutes", type=int, default=120)
    parser.add_argument("--markets", default=None, help="Optional comma-separated market list")
    parser.add_argument("--out-md", default=None, help="Output markdown path")
    parser.add_argument("--out-json", default=None, help="Output json path")
    parser.add_argument("--tz", default="Europe/Lisbon", help="Display timezone for packet window labels")
    parser.add_argument("--anchor", default="mid", help="Price anchor for markout diagnostics")
    args = parser.parse_args()

    target = Path(args.target)
    markets_filter = _parse_markets_arg(args.markets)

    packet_mode = (
        target.is_dir()
        or bool(args.out_md)
        or bool(args.out_json)
        or bool(markets_filter)
        or int(args.window_minutes) != 120
        or str(args.tz) != "Europe/Lisbon"
        or str(args.anchor).lower() != "mid"
    )

    if not packet_mode:
        # Legacy single-journal analysis behavior.
        if target.is_dir():
            target = find_latest_journal(target)

        print(f"Analysing: {target}\n")
        events = load_journal(target)

        schema_valid, schema_warnings = validate_schema_versions(events)
        if not schema_valid:
            print("⚠ Schema version warnings:")
            for w in schema_warnings:
                print(w)
            print()

        report = analyse(events, target, args.assumed_fee_bps)
        print(report)

        report_path = target.with_suffix(".analysis.txt")
        report_path.write_text(report)
        print(f"\nReport saved to: {report_path}")

        json_out = args.out_json or args.json_out
        if json_out:
            summary = build_summary(events, target, args.assumed_fee_bps)
            json_path = Path(json_out)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(_to_jsonable(summary), indent=2) + "\n")
            print(f"JSON summary saved to: {json_path}")
        return

    packet = build_analyst_packet(
        target,
        window_minutes=args.window_minutes,
        markets_filter=markets_filter,
        tz_name=args.tz,
        anchor=args.anchor,
    )
    markdown = render_analyst_packet_markdown(packet)

    out_md_path: Optional[Path]
    if args.out_md:
        out_md_path = Path(args.out_md)
    elif target.is_dir():
        out_md_path = _default_packet_output_path(packet)
    else:
        out_md_path = None

    if out_md_path is not None:
        out_md_path.parent.mkdir(parents=True, exist_ok=True)
        out_md_path.write_text(markdown)
        print(f"Analyst packet markdown saved to: {out_md_path}")
    else:
        print(markdown)

    out_json_arg = args.out_json or args.json_out
    out_json_path: Optional[Path]
    if out_json_arg:
        out_json_path = Path(out_json_arg)
    elif out_md_path is not None:
        out_json_path = out_md_path.with_suffix(".json")
    else:
        out_json_path = None

    if out_json_path is not None:
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        out_json_path.write_text(json.dumps(_to_jsonable(packet), indent=2) + "\n")
        print(f"Analyst packet json saved to: {out_json_path}")


if __name__ == "__main__":
    main()
