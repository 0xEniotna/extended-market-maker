#!/usr/bin/env python
"""
Analyse Market Maker Journal

Reads a JSONL trade journal produced by the market maker and prints a
compact summary designed for sharing and collaborative debugging.

Usage:
    PYTHONPATH=src python scripts/analyse_mm_journal.py data/mm_journal/mm_SOL-USD_*.jsonl
    PYTHONPATH=src python scripts/analyse_mm_journal.py data/mm_journal/  # latest file
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _d(v) -> Decimal:
    if v is None:
        return Decimal("0")
    return Decimal(str(v))


def _ts_fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _event_mid(event: Dict[str, Any]) -> Optional[Decimal]:
    if event.get("mid") is not None:
        return _d(event["mid"])
    if event.get("best_bid") is not None and event.get("best_ask") is not None:
        return (_d(event["best_bid"]) + _d(event["best_ask"])) / Decimal("2")
    return None


def _build_mid_series(events: List[Dict[str, Any]]) -> Tuple[List[float], List[Decimal]]:
    ts_values: List[float] = []
    mid_values: List[Decimal] = []
    for event in events:
        mid = _event_mid(event)
        ts = event.get("ts")
        if mid is None or ts is None:
            continue
        ts_values.append(float(ts))
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
                "start_ts": prev_ts, "end_ts": curr_ts, "gap_s": delta,
                "start_iso": _ts_fmt(prev_ts), "end_iso": _ts_fmt(curr_ts),
            })
    return gaps


def load_journal(path: Path) -> List[Dict[str, Any]]:
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def find_latest_journal(dir_path: Path) -> Path:
    files = sorted(dir_path.glob("mm_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        print(f"No journal files found in {dir_path}")
        sys.exit(1)
    return files[-1]


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
        "market": market, "journal_file": str(path), "assumed_fee_bps": assumed_fee_bps,
        "counts": {"events": len(events), "fills": len(fills), "orders": len(orders), "snapshots": len(snapshots)},
        "window": {"start_ts": start_ts, "end_ts": end_ts, "start_iso": _ts_fmt(start_ts) if start_ts else None,
                    "end_iso": _ts_fmt(end_ts) if end_ts else None, "duration_s": duration_s},
        "metrics": {
            "fill_rate_pct": fill_rate_pct, "avg_edge_bps": _avg(edge_values),
            "adverse_fill_count": adverse_fill_count, "adverse_fill_ratio": adverse_fill_ratio,
            "final_position": _latest_position(events),
            "markout_250ms_bps": _avg(markout_values[0.25]), "markout_1s_bps": _avg(markout_values[1.0]),
            "markout_5s_bps": _avg(markout_values[5.0]), "markout_30s_bps": _avg(markout_values[30.0]),
            "markout_2m_bps": _avg(markout_values[120.0]),
            "markout_counts": {_horizon_label(h): len(markout_values[h]) for h in _MARKOUT_HORIZONS},
            "taker_fill_count": taker_count, "taker_fill_notional_ratio": taker_notional_ratio,
            "quote_lifetime_ms_avg": _avg(quote_lifetimes), "quote_lifetime_ms_count": len(quote_lifetimes),
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
    lines.append(f"Period: {_ts_fmt(t0)} → {_ts_fmt(t1)}  ({duration_m:.0f} min)")
    lines.append(f"Events: {len(events)} total "
                 f"({len(fills)} fills, {len(orders)} placements, {len(snapshots)} snapshots)")
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
            lines.append(f"  {gap['start_iso']} → {gap['end_iso']}  ({gap['gap_s']:.0f}s gap)")
        lines.append("")

    lines.extend(build_fill_section(fills, events, ts_values, mid_values, assumed_fee_bps))
    lines.extend(build_order_section(orders, fills, rejections, cancellations, duration_m))
    lines.extend(build_reprice_section(reprice_decisions))

    if snapshots:
        lines.append("## Spread Over Time (from snapshots)")
        spreads = [_d(s.get("spread_bps")) for s in snapshots if s.get("spread_bps") is not None]
        if spreads:
            lines.append(f"  Samples: {len(spreads)}  avg={sum(spreads) / len(spreads):.1f}bps  "
                         f"min={min(spreads):.1f}  max={max(spreads):.1f}")
        positions = [_d(s.get("position")) for s in snapshots]
        if positions:
            lines.append(f"  Position range: [{min(positions)}, {max(positions)}]")
        lines.append("")

    if orders and fills:
        ratio = len(fills) / len(orders) * 100
        lines.append(f"## Fill Rate: {ratio:.1f}%  ({len(fills)} fills / {len(orders)} orders)")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse market-maker journal JSONL and produce a shareable report.",
    )
    parser.add_argument("target", help="Journal file or directory containing mm_*.jsonl")
    parser.add_argument("--assumed-fee-bps", type=Decimal, default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    target = Path(args.target)
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
    if args.json_out:
        summary = build_summary(events, target, args.assumed_fee_bps)
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(_to_jsonable(summary), indent=2) + "\n")
        print(f"JSON summary saved to: {json_path}")


if __name__ == "__main__":
    main()
