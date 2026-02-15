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
from collections import Counter, defaultdict
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
    """Return event mid-price if available from explicit mid or BBO."""
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
    """Return first observed mid at/after target_ts within max_wait_s."""
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


def _build_mid_timeline(
    events: List[Dict[str, Any]],
) -> Tuple[List[float], List[Decimal]]:
    """Build a sorted timeline of (timestamp, mid_price) from all events with bid/ask."""
    timestamps: List[float] = []
    mids: List[Decimal] = []
    for e in events:
        bb = e.get("best_bid")
        ba = e.get("best_ask")
        if bb is not None and ba is not None:
            bid = _d(bb)
            ask = _d(ba)
            if bid > 0 and ask > 0:
                timestamps.append(e["ts"])
                mids.append((bid + ask) / 2)
    return timestamps, mids


def _lookup_mid(
    timestamps: List[float],
    mids: List[Decimal],
    target_ts: float,
    tolerance_s: float = 5.0,
) -> Optional[Decimal]:
    """Find the mid price closest to *target_ts* (within tolerance)."""
    if not timestamps:
        return None
    idx = bisect.bisect_left(timestamps, target_ts)
    best_mid = None
    best_gap = float("inf")
    for i in (idx - 1, idx):
        if 0 <= i < len(timestamps):
            gap = abs(timestamps[i] - target_ts)
            if gap < best_gap:
                best_gap = gap
                best_mid = mids[i]
    return best_mid if best_gap <= tolerance_s else None


_MARKOUT_HORIZONS = [1, 5, 30]  # seconds


def _compute_markouts(
    fills: List[Dict[str, Any]],
    timestamps: List[float],
    mids: List[Decimal],
) -> Dict[int, List[Decimal]]:
    """Compute markouts at each horizon for every fill.

    Markout = price movement in bps from fill_price, signed so positive = favorable.
    For BUY:  markout = (mid_at_T+N - fill_price) / fill_price * 10000
    For SELL: markout = (fill_price - mid_at_T+N) / fill_price * 10000
    """
    result: Dict[int, List[Decimal]] = {h: [] for h in _MARKOUT_HORIZONS}
    _10000 = Decimal("10000")

    for f in fills:
        fill_ts = f["ts"]
        fill_price = _d(f["price"])
        is_buy = "BUY" in str(f["side"])

        if fill_price <= 0:
            continue

        for horizon in _MARKOUT_HORIZONS:
            # Tolerance scales with horizon (snapshots are ~60s apart)
            tol = max(5.0, horizon * 0.5)
            future_mid = _lookup_mid(timestamps, mids, fill_ts + horizon, tol)
            if future_mid is None:
                continue
            if is_buy:
                markout = (future_mid - fill_price) / fill_price * _10000
            else:
                markout = (fill_price - future_mid) / fill_price * _10000
            result[horizon].append(markout)

    return result


def _markout_for_fill(
    fill: Dict[str, Any],
    horizon_s: float,
    ts_values: List[float],
    mid_values: List[Decimal],
) -> Optional[Decimal]:
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


def _fill_context_value(fill: Dict[str, Any], key: str) -> Optional[Decimal]:
    snap = fill.get("market_snapshot")
    if isinstance(snap, dict):
        value = snap.get(key)
        if value is not None:
            return _d(value)
    # Backward-compatible fallback for spread.
    if key == "spread_bps" and fill.get("spread_bps") is not None:
        return _d(fill.get("spread_bps"))
    return None


def _bucket_spread(spread_bps: Optional[Decimal]) -> Optional[str]:
    if spread_bps is None:
        return None
    if spread_bps < Decimal("2"):
        return "<2"
    if spread_bps < Decimal("5"):
        return "2-5"
    if spread_bps < Decimal("10"):
        return "5-10"
    return ">10"


def _bucket_micro_vol(micro_vol_bps: Optional[Decimal]) -> Optional[str]:
    if micro_vol_bps is None:
        return None
    if micro_vol_bps < Decimal("2"):
        return "<2"
    if micro_vol_bps < Decimal("5"):
        return "2-5"
    return ">5"


def _bucket_drift(micro_drift_bps: Optional[Decimal]) -> Optional[str]:
    if micro_drift_bps is None:
        return None
    if micro_drift_bps < Decimal("-1"):
        return "negative"
    if micro_drift_bps > Decimal("1"):
        return "positive"
    return "neutral"


def _bucket_imbalance(imbalance: Optional[Decimal]) -> Optional[str]:
    if imbalance is None:
        return None
    if imbalance < Decimal("-0.30"):
        return "ask-heavy"
    if imbalance > Decimal("0.30"):
        return "bid-heavy"
    return "balanced"


def _format_avg(values: List[Decimal]) -> str:
    if not values:
        return "n/a"
    return f"{(sum(values) / Decimal(len(values))):.2f}"

def analyse(events: List[Dict[str, Any]], path: Path, assumed_fee_bps: Optional[Decimal]) -> str:
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

    # ── Fill analysis ──
    if fills:
        lines.append("## Fills")
        buy_fills = [f for f in fills if f["side"] in ("BUY", "OrderSide.BUY")]
        sell_fills = [f for f in fills if f["side"] in ("SELL", "OrderSide.SELL")]
        total_buy_qty = sum(_d(f["qty"]) for f in buy_fills)
        total_sell_qty = sum(_d(f["qty"]) for f in sell_fills)
        total_buy_notional = sum(_d(f["qty"]) * _d(f["price"]) for f in buy_fills)
        total_sell_notional = sum(_d(f["qty"]) * _d(f["price"]) for f in sell_fills)
        reported_fees = sum(_d(f["fee"]) for f in fills)
        fill_notional = sum(_d(f["qty"]) * _d(f["price"]) for f in fills)
        assumed_fees = Decimal("0")
        if assumed_fee_bps is not None:
            assumed_fees = fill_notional * assumed_fee_bps / Decimal("10000")
        taker_count = sum(1 for f in fills if f.get("is_taker"))
        maker_count = len(fills) - taker_count

        lines.append(f"  Total: {len(fills)} fills "
                     f"({len(buy_fills)} buys, {len(sell_fills)} sells)")
        lines.append(f"  Buy volume:  {total_buy_qty} contracts  "
                     f"(${total_buy_notional:.2f} notional)")
        lines.append(f"  Sell volume: {total_sell_qty} contracts  "
                     f"(${total_sell_notional:.2f} notional)")
        lines.append(f"  Maker/Taker: {maker_count}/{taker_count}")
        lines.append(f"  Reported fees:  ${reported_fees:.4f}")
        if assumed_fee_bps is not None:
            lines.append(
                f"  Assumed fees @{assumed_fee_bps}bps: ${assumed_fees:.4f}"
            )

        # Edge analysis (how far from mid were fills)
        edges = [_d(f.get("edge_bps")) for f in fills if f.get("edge_bps") is not None]
        if edges:
            avg_edge = sum(edges) / len(edges)
            min_edge = min(edges)
            max_edge = max(edges)
            neg_edges = [e for e in edges if e < 0]
            lines.append(f"  Edge (vs mid): avg={avg_edge:.2f}bps  "
                         f"min={min_edge:.2f}  max={max_edge:.2f}  "
                         f"adverse={len(neg_edges)}/{len(edges)}")

        # Spread at fill time
        spreads_at_fill = [_d(f.get("spread_bps"))
                           for f in fills if f.get("spread_bps") is not None]
        if spreads_at_fill:
            avg_spread = sum(spreads_at_fill) / len(spreads_at_fill)
            lines.append(f"  Spread at fill: avg={avg_spread:.1f}bps")

        mid_ts, mid_prices = _build_mid_timeline(events)

        # Per-level breakdown
        by_level: Dict[str, list] = defaultdict(list)
        for f in fills:
            lvl = f.get("level")
            key = f"L{lvl}" if lvl is not None else "L?"
            by_level[key].append(f)
        lines.append("  Per level:")
        for lvl_key in sorted(by_level):
            lvl_fills = by_level[lvl_key]
            lvl_qty = sum(_d(f["qty"]) for f in lvl_fills)
            lines.append(f"    {lvl_key}: {len(lvl_fills)} fills, {lvl_qty} qty")
        unknown_level_fills = len(by_level.get("L?", []))
        if unknown_level_fills:
            lines.append(f"  Unknown level fills: {unknown_level_fills}/{len(fills)}")

        # MTM decomposition:
        # - Session MTM uses only inventory created by fills inside this journal.
        # - Account MTM additionally includes PnL on inventory carried into this run.
        cashflow = total_sell_notional - total_buy_notional
        net_fill_qty = total_buy_qty - total_sell_qty
        final_pos = _latest_position(events) or Decimal("0")
        start_pos_obs = _first_position(events)
        inferred_start_pos = final_pos - net_fill_qty
        start_pos_for_carry = (
            start_pos_obs if start_pos_obs is not None else inferred_start_pos
        )
        first_mid = mid_values[0] if mid_values else None
        last_mid = mid_values[-1] if mid_values else None
        session_inventory_mtm = (
            net_fill_qty * last_mid if last_mid is not None else Decimal("0")
        )
        session_gross_mtm = cashflow + session_inventory_mtm
        net_reported = session_gross_mtm - reported_fees
        net_assumed = (
            session_gross_mtm - assumed_fees if assumed_fee_bps is not None else None
        )
        carry_mtm = None
        account_gross_mtm = None
        if first_mid is not None and last_mid is not None:
            carry_mtm = start_pos_for_carry * (last_mid - first_mid)
            account_gross_mtm = session_gross_mtm + carry_mtm

        lines.append(f"  Cashflow P&L:   ${cashflow:.4f}  (sell - buy notional)")
        if last_mid is not None:
            lines.append(
                f"  Session Inventory MTM:  ${session_inventory_mtm:.4f}  "
                f"(net fills {net_fill_qty} @ mid {last_mid})"
            )
        else:
            lines.append(
                "  Session Inventory MTM:  unavailable  "
                f"(net fills {net_fill_qty}, no market mid)"
            )
        lines.append(f"  Gross MTM P&L:  ${session_gross_mtm:.4f}  (session fills only)")
        if carry_mtm is not None and account_gross_mtm is not None:
            lines.append(
                f"  Carry MTM:      ${carry_mtm:.4f}  "
                f"(start pos {start_pos_for_carry})"
            )
            lines.append(f"  Account MTM Δ:  ${account_gross_mtm:.4f}  (session + carry)")
        lines.append(f"  Net MTM P&L:    ${net_reported:.4f}  (reported fees)")
        if net_assumed is not None:
            lines.append(
                f"  Net MTM P&L:    ${net_assumed:.4f}  "
                f"(assumed fees @{assumed_fee_bps}bps)"
            )
        if start_pos_obs is not None and start_pos_obs != inferred_start_pos:
            lines.append(
                f"  Start position note: first observed={start_pos_obs}, "
                f"inferred_from_fills={inferred_start_pos}"
            )
        lines.append(f"  Final position: {final_pos}")

        # Post-fill markouts measure short-horizon selection quality.
        horizons = [1.0, 5.0, 30.0]
        markouts: Dict[float, List[Decimal]] = {h: [] for h in horizons}
        markout_coverage: Dict[float, int] = {h: 0 for h in horizons}
        for fill in fills:
            fill_ts = float(fill.get("ts", 0.0))
            fill_px = _d(fill["price"])
            side = str(fill.get("side", ""))
            for h in horizons:
                fut_mid = _mid_at_or_after(ts_values, mid_values, fill_ts + h)
                if fut_mid is None or fill_px <= 0:
                    continue
                markout_coverage[h] += 1
                if "BUY" in side:
                    value = (fut_mid - fill_px) / fill_px * Decimal("10000")
                else:
                    value = (fill_px - fut_mid) / fill_px * Decimal("10000")
                markouts[h].append(value)

        lines.append("  Markout (bps):")
        for h in horizons:
            vals = markouts[h]
            if vals:
                avg = sum(vals) / Decimal(len(vals))
                lines.append(
                    f"    +{int(h)}s: avg={avg:.2f}  min={min(vals):.2f}  "
                    f"max={max(vals):.2f}  n={len(vals)}/{len(fills)}"
                )
            else:
                lines.append(
                    f"    +{int(h)}s: unavailable  n={markout_coverage[h]}/{len(fills)}"
                )
        lines.append("  Level toxicity (+5s markout):")
        for lvl_key in sorted(by_level):
            lvl_fills = by_level[lvl_key]
            lvl_markouts_5s: List[Decimal] = []
            for fill in lvl_fills:
                fill_px = _d(fill["price"])
                if fill_px <= 0:
                    continue
                side = str(fill.get("side", ""))
                fut_mid = _mid_at_or_after(ts_values, mid_values, float(fill["ts"]) + 5.0)
                if fut_mid is None:
                    continue
                if "BUY" in side:
                    lvl_markouts_5s.append(
                        (fut_mid - fill_px) / fill_px * Decimal("10000")
                    )
                else:
                    lvl_markouts_5s.append(
                        (fill_px - fut_mid) / fill_px * Decimal("10000")
                    )
            if lvl_markouts_5s:
                adverse = sum(1 for v in lvl_markouts_5s if v < 0)
                avg_m5 = sum(lvl_markouts_5s) / Decimal(len(lvl_markouts_5s))
                lines.append(
                    f"    {lvl_key}: avg_mo5={avg_m5:.2f}bps adverse={adverse}/{len(lvl_markouts_5s)}"
                )
            else:
                lines.append(f"    {lvl_key}: mo5 unavailable")

        # Snapshot data completeness (schema v2 fill snapshots)
        snapshot_present = 0
        depth_complete = 0
        for f in fills:
            snap = f.get("market_snapshot")
            if not isinstance(snap, dict):
                continue
            snapshot_present += 1
            bids_top = snap.get("bids_top")
            asks_top = snap.get("asks_top")
            if isinstance(bids_top, list) and isinstance(asks_top, list):
                if len(bids_top) >= 5 and len(asks_top) >= 5:
                    depth_complete += 1
        lines.append("  Data completeness:")
        lines.append(
            "    fill_snapshot_present: "
            f"{snapshot_present}/{len(fills)} ({(Decimal(snapshot_present) / Decimal(len(fills)) * Decimal('100')):.1f}%)"
        )
        lines.append(
            "    fill_snapshot_top5_complete: "
            f"{depth_complete}/{len(fills)} ({(Decimal(depth_complete) / Decimal(len(fills)) * Decimal('100')):.1f}%)"
        )
        lines.append("")

        # Context/regime buckets from fill snapshots
        lines.append("## Context Regime Analysis")
        contexts = {
            "spread_bps": (
                _bucket_spread,
                ["<2", "2-5", "5-10", ">10"],
                "Spread bucket (bps)",
            ),
            "micro_vol_bps": (
                _bucket_micro_vol,
                ["<2", "2-5", ">5"],
                "Micro-vol bucket (bps)",
            ),
            "micro_drift_bps": (
                _bucket_drift,
                ["negative", "neutral", "positive"],
                "Drift bucket",
            ),
            "imbalance": (
                _bucket_imbalance,
                ["ask-heavy", "balanced", "bid-heavy"],
                "Imbalance bucket",
            ),
        }
        for context_key, (bucket_fn, bucket_order, title) in contexts.items():
            lines.append(f"  {title}:")
            for bucket in bucket_order:
                subset = [
                    f for f in fills
                    if bucket_fn(_fill_context_value(f, context_key)) == bucket
                ]
                if not subset:
                    lines.append(f"    {bucket}: n=0")
                    continue

                edges = [
                    _d(f.get("edge_bps"))
                    for f in subset
                    if f.get("edge_bps") is not None
                ]
                mo = {
                    1: [],
                    5: [],
                    30: [],
                }
                for f in subset:
                    for h in (1, 5, 30):
                        m = _markout_for_fill(f, float(h), ts_values, mid_values)
                        if m is not None:
                            mo[h].append(m)
                adverse_5 = sum(1 for v in mo[5] if v < 0)
                adverse_5_pct = (
                    Decimal(adverse_5) / Decimal(len(mo[5])) * Decimal("100")
                    if mo[5] else Decimal("0")
                )
                cashflow_bucket = Decimal("0")
                for f in subset:
                    qty = _d(f.get("qty"))
                    px = _d(f.get("price"))
                    side = str(f.get("side", ""))
                    if "SELL" in side:
                        cashflow_bucket += qty * px
                    else:
                        cashflow_bucket -= qty * px

                lines.append(
                    f"    {bucket}: n={len(subset)} "
                    f"avg_edge={_format_avg(edges)}bps "
                    f"mo1={_format_avg(mo[1])}bps "
                    f"mo5={_format_avg(mo[5])}bps "
                    f"mo30={_format_avg(mo[30])}bps "
                    f"adverse5={adverse_5_pct:.1f}% "
                    f"cashflow=${cashflow_bucket:.2f}"
                )
        lines.append("")

        # Chronological fill log (last 20)
        lines.append("## Recent Fills (last 20)")
        _10000 = Decimal("10000")
        for f in fills[-20:]:
            side_char = "B" if "BUY" in str(f["side"]) else "S"
            is_buy = "BUY" in str(f["side"])
            edge_str = ""
            if f.get("edge_bps") is not None:
                edge_str = f" edge={_d(f['edge_bps']):+.1f}bps"
            spread_str = ""
            if f.get("spread_bps") is not None:
                spread_str = f" spread={_d(f['spread_bps']):.1f}bps"
            # T+5s markout for each fill
            markout_str = ""
            fp = _d(f["price"])
            if fp > 0 and mid_ts:
                future_mid = _lookup_mid(mid_ts, mid_prices, f["ts"] + 5)
                if future_mid is not None:
                    if is_buy:
                        mo = (future_mid - fp) / fp * _10000
                    else:
                        mo = (fp - future_mid) / fp * _10000
                    markout_str = f" mo5={mo:+.1f}bps"
            taker_str = "T" if f.get("is_taker") else "M"
            lines.append(
                f"  {_ts_fmt(f['ts'])} {side_char} {f['qty']}@{f['price']} "
                f"fee={f['fee']} [{taker_str}]{edge_str}{spread_str}{markout_str}"
            )
        lines.append("")
    else:
        lines.append("## Fills: NONE")
        lines.append("")

    # ── Order placement analysis ──
    if orders:
        lines.append("## Order Placements")
        lines.append(f"  Total: {len(orders)}")
        rate = len(orders) / max(duration_m, 1)
        lines.append(f"  Rate: {rate:.1f}/min")

        spreads_at_place = [_d(o.get("spread_bps"))
                            for o in orders if o.get("spread_bps") is not None]
        if spreads_at_place:
            avg_spread = sum(spreads_at_place) / len(spreads_at_place)
            lines.append(f"  Avg spread at placement: {avg_spread:.1f}bps")

        # Offset from best (how far our orders were placed from BBO)
        offsets: list[Decimal] = []
        for o in orders:
            price = _d(o["price"])
            if "BUY" in str(o["side"]) and o.get("best_bid"):
                offsets.append((_d(o["best_bid"]) - price) / _d(o["best_bid"]) * 10000)
            elif "SELL" in str(o["side"]) and o.get("best_ask"):
                offsets.append((price - _d(o["best_ask"])) / _d(o["best_ask"]) * 10000)
        if offsets:
            avg_off = sum(offsets) / len(offsets)
            lines.append(f"  Avg offset from BBO: {avg_off:.1f}bps")

        reject_rate = Decimal("0")
        pof_count = 0
        if rejections:
            reject_rate = Decimal(len(rejections)) / Decimal(len(orders)) * Decimal("100")
            pof_count = sum(
                1
                for r in rejections
                if "POST_ONLY" in str(r.get("reason", "")).upper()
            )
            pof_rate = Decimal(pof_count) / Decimal(len(orders)) * Decimal("100")
            lines.append(
                f"  Rejections: {len(rejections)} ({reject_rate:.2f}% of placements)"
            )
            lines.append(
                f"  Post-only rejects: {pof_count} ({pof_rate:.2f}% of placements)"
            )
        if cancellations:
            cancel_rate = Decimal(len(cancellations)) / Decimal(len(orders)) * Decimal("100")
            lines.append(
                f"  Cancellations: {len(cancellations)} ({cancel_rate:.2f}% of placements)"
            )
        lines.append("")
    else:
        lines.append("## Order Placements: NONE")
        lines.append("")

    # ── Reprice decision telemetry ──
    if reprice_decisions:
        lines.append("## Reprice Decisions")
        total = len(reprice_decisions)
        lines.append(f"  Total: {total}")

        reason_counts = Counter(
            str(event.get("reason") or "unknown")
            for event in reprice_decisions
        )
        lines.append("  Reasons:")
        for reason, count in reason_counts.most_common(12):
            pct = Decimal(count) / Decimal(total) * Decimal("100")
            lines.append(f"    {reason}: {count} ({pct:.1f}%)")

        skip_toxicity = reason_counts.get("skip_toxicity", 0)
        toxicity_pct = Decimal(skip_toxicity) / Decimal(total) * Decimal("100")
        lines.append(
            f"  skip_toxicity share: {skip_toxicity}/{total} ({toxicity_pct:.1f}%)"
        )

        regime_counts = Counter(
            str(event.get("regime"))
            for event in reprice_decisions
            if event.get("regime") is not None
        )
        if regime_counts:
            lines.append("  Regimes:")
            for regime, count in regime_counts.most_common():
                pct = Decimal(count) / Decimal(total) * Decimal("100")
                lines.append(f"    {regime}: {count} ({pct:.1f}%)")

        trend_counts = Counter(
            str(event.get("trend_direction"))
            for event in reprice_decisions
            if event.get("trend_direction") is not None
        )
        if trend_counts:
            lines.append("  Trend directions:")
            for direction, count in trend_counts.most_common():
                pct = Decimal(count) / Decimal(total) * Decimal("100")
                lines.append(f"    {direction}: {count} ({pct:.1f}%)")

        band_counts = Counter(
            str(event.get("inventory_band"))
            for event in reprice_decisions
            if event.get("inventory_band") is not None
        )
        if band_counts:
            lines.append("  Inventory bands:")
            for band, count in band_counts.most_common():
                pct = Decimal(count) / Decimal(total) * Decimal("100")
                lines.append(f"    {band}: {count} ({pct:.1f}%)")

        trend_strengths = [
            _d(event.get("trend_strength"))
            for event in reprice_decisions
            if event.get("trend_strength") is not None
        ]
        if trend_strengths:
            avg_strength = sum(trend_strengths) / Decimal(len(trend_strengths))
            lines.append(f"  Avg trend strength: {avg_strength:.3f}")

        funding_bias_values = [
            _d(event.get("funding_bias_bps"))
            for event in reprice_decisions
            if event.get("funding_bias_bps") is not None
        ]
        if funding_bias_values:
            avg_funding_bias = sum(funding_bias_values) / Decimal(
                len(funding_bias_values)
            )
            lines.append(f"  Avg funding bias: {avg_funding_bias:.2f}bps")
        lines.append("")

    # ── Spread over time (from snapshots) ──
    if snapshots:
        lines.append("## Spread Over Time (from snapshots)")
        spreads = [_d(s.get("spread_bps"))
                   for s in snapshots if s.get("spread_bps") is not None]
        if spreads:
            avg = sum(spreads) / len(spreads)
            lines.append(f"  Samples: {len(spreads)}  "
                         f"avg={avg:.1f}bps  min={min(spreads):.1f}  max={max(spreads):.1f}")

        # Position evolution
        positions = [_d(s.get("position")) for s in snapshots]
        if positions:
            lines.append(f"  Position range: [{min(positions)}, {max(positions)}]")
        lines.append("")

    # ── Conversion ratio ──
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
    parser.add_argument(
        "--assumed-fee-bps",
        type=Decimal,
        default=None,
        help="Optional fee assumption per side in bps for stress-testing net MTM P&L",
    )
    args = parser.parse_args()

    target = Path(args.target)
    if target.is_dir():
        target = find_latest_journal(target)

    print(f"Analysing: {target}\n")
    events = load_journal(target)
    report = analyse(events, target, args.assumed_fee_bps)
    print(report)

    # Also write to .txt next to the journal
    report_path = target.with_suffix(".analysis.txt")
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
