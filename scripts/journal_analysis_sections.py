"""
Journal Analysis Report Sections

Extracted from ``analyse_mm_journal.py``: the long report-building
sections of the ``analyse()`` function (fill analysis, order analysis,
reprice decisions, context regime analysis).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional

from analyse_mm_journal import (
    _MARKOUT_HORIZONS,
    _bucket_drift,
    _bucket_imbalance,
    _bucket_micro_vol,
    _bucket_spread,
    _d,
    _fill_context_value,
    _first_position,
    _format_avg,
    _horizon_label,
    _latest_position,
    _markout_for_fill,
    _ts_fmt,
)


def build_fill_section(
    fills: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    ts_values: List[float],
    mid_values: List[Decimal],
    assumed_fee_bps: Optional[Decimal],
) -> List[str]:
    """Build the fill analysis section of the report."""
    if not fills:
        return ["## Fills: NONE", ""]

    lines: List[str] = []
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
    taker_notional = sum(
        _d(f["qty"]) * _d(f["price"]) for f in fills if f.get("is_taker")
    )
    taker_notional_ratio = (
        taker_notional / fill_notional if fill_notional > 0 else Decimal("0")
    )
    quote_lifetimes_ms = [
        _d(f.get("quote_lifetime_ms"))
        for f in fills
        if f.get("quote_lifetime_ms") is not None
    ]

    lines.append(f"  Total: {len(fills)} fills "
                 f"({len(buy_fills)} buys, {len(sell_fills)} sells)")
    lines.append(f"  Buy volume:  {total_buy_qty} contracts  "
                 f"(${total_buy_notional:.2f} notional)")
    lines.append(f"  Sell volume: {total_sell_qty} contracts  "
                 f"(${total_sell_notional:.2f} notional)")
    lines.append(f"  Maker/Taker: {maker_count}/{taker_count}")
    lines.append(f"  Taker notional ratio: {(taker_notional_ratio * Decimal('100')):.2f}%")
    if quote_lifetimes_ms:
        avg_lifetime = sum(quote_lifetimes_ms) / Decimal(len(quote_lifetimes_ms))
        lines.append(f"  Quote lifetime-to-fill: avg={avg_lifetime:.1f}ms n={len(quote_lifetimes_ms)}")
    lines.append(f"  Reported fees:  ${reported_fees:.4f}")
    if assumed_fee_bps is not None:
        lines.append(f"  Assumed fees @{assumed_fee_bps}bps: ${assumed_fees:.4f}")

    # Edge analysis
    edges = [_d(f.get("edge_bps")) for f in fills if f.get("edge_bps") is not None]
    if edges:
        avg_edge = sum(edges) / len(edges)
        neg_edges = [e for e in edges if e < 0]
        lines.append(f"  Edge (vs mid): avg={avg_edge:.2f}bps  "
                     f"min={min(edges):.2f}  max={max(edges):.2f}  "
                     f"adverse={len(neg_edges)}/{len(edges)}")

    spreads_at_fill = [_d(f.get("spread_bps")) for f in fills if f.get("spread_bps") is not None]
    if spreads_at_fill:
        lines.append(f"  Spread at fill: avg={sum(spreads_at_fill) / len(spreads_at_fill):.1f}bps")

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

    # MTM decomposition
    lines.extend(_build_mtm_section(
        fills, events, ts_values, mid_values,
        total_buy_qty, total_sell_qty,
        total_buy_notional, total_sell_notional,
        reported_fees, assumed_fees, assumed_fee_bps,
        fill_notional,
    ))

    # Markout analysis
    lines.extend(_build_markout_section(fills, ts_values, mid_values, by_level))

    # Data completeness
    lines.extend(_build_completeness_section(fills))
    lines.append("")

    # Context regime analysis
    lines.extend(build_context_regime_section(fills, ts_values, mid_values))

    # Recent fills
    lines.extend(_build_recent_fills_section(fills, ts_values, mid_values))
    return lines


def _build_mtm_section(
    fills, events, ts_values, mid_values,
    total_buy_qty, total_sell_qty,
    total_buy_notional, total_sell_notional,
    reported_fees, assumed_fees, assumed_fee_bps,
    fill_notional,
) -> List[str]:
    lines: List[str] = []
    cashflow = total_sell_notional - total_buy_notional
    net_fill_qty = total_buy_qty - total_sell_qty
    final_pos = _latest_position(events) or Decimal("0")
    start_pos_obs = _first_position(events)
    inferred_start_pos = final_pos - net_fill_qty
    start_pos_for_carry = start_pos_obs if start_pos_obs is not None else inferred_start_pos
    first_mid = mid_values[0] if mid_values else None
    last_mid = mid_values[-1] if mid_values else None
    session_inventory_mtm = net_fill_qty * last_mid if last_mid is not None else Decimal("0")
    session_gross_mtm = cashflow + session_inventory_mtm
    net_reported = session_gross_mtm - reported_fees
    net_assumed = session_gross_mtm - assumed_fees if assumed_fee_bps is not None else None
    carry_mtm = None
    account_gross_mtm = None
    if first_mid is not None and last_mid is not None:
        carry_mtm = start_pos_for_carry * (last_mid - first_mid)
        account_gross_mtm = session_gross_mtm + carry_mtm

    lines.append(f"  Cashflow P&L:   ${cashflow:.4f}  (sell - buy notional)")
    if last_mid is not None:
        lines.append(f"  Session Inventory MTM:  ${session_inventory_mtm:.4f}  "
                     f"(net fills {net_fill_qty} @ mid {last_mid})")
    else:
        lines.append(f"  Session Inventory MTM:  unavailable  (net fills {net_fill_qty}, no market mid)")
    lines.append(f"  Gross MTM P&L:  ${session_gross_mtm:.4f}  (session fills only)")
    if carry_mtm is not None and account_gross_mtm is not None:
        lines.append(f"  Carry MTM:      ${carry_mtm:.4f}  (start pos {start_pos_for_carry})")
        lines.append(f"  Account MTM Δ:  ${account_gross_mtm:.4f}  (session + carry)")
    lines.append(f"  Net MTM P&L:    ${net_reported:.4f}  (reported fees)")
    if net_assumed is not None:
        lines.append(f"  Net MTM P&L:    ${net_assumed:.4f}  (assumed fees @{assumed_fee_bps}bps)")
    if start_pos_obs is not None and start_pos_obs != inferred_start_pos:
        lines.append(f"  Start position note: first observed={start_pos_obs}, "
                     f"inferred_from_fills={inferred_start_pos}")
    lines.append(f"  Final position: {final_pos}")
    return lines


def _build_markout_section(fills, ts_values, mid_values, by_level) -> List[str]:
    lines: List[str] = []
    horizons = list(_MARKOUT_HORIZONS)
    markouts: Dict[float, List[Decimal]] = {h: [] for h in horizons}
    for fill in fills:
        for h in horizons:
            m = _markout_for_fill(fill, h, ts_values, mid_values)
            if m is not None:
                markouts[h].append(m)

    lines.append("  Markout (bps):")
    for h in horizons:
        vals = markouts[h]
        label = _horizon_label(h)
        if vals:
            avg = sum(vals) / Decimal(len(vals))
            lines.append(f"    +{label}: avg={avg:.2f}  min={min(vals):.2f}  "
                         f"max={max(vals):.2f}  n={len(vals)}/{len(fills)}")
        else:
            lines.append(f"    +{label}: unavailable  n=0/{len(fills)}")

    def _conditioned_markout(*, side_filter=None, taker_filter=None, horizon_s=5.0):
        vals = []
        for fill in fills:
            side = str(fill.get("side", "")).upper()
            is_taker = bool(fill.get("is_taker"))
            if side_filter is not None and side_filter not in side:
                continue
            if taker_filter is not None and taker_filter != is_taker:
                continue
            m = _markout_for_fill(fill, horizon_s, ts_values, mid_values)
            if m is not None:
                vals.append(m)
        return vals

    lines.append("  Markout +5s conditioned:")
    for label, vals in [
        ("maker", _conditioned_markout(taker_filter=False)),
        ("taker", _conditioned_markout(taker_filter=True)),
        ("buy", _conditioned_markout(side_filter="BUY")),
        ("sell", _conditioned_markout(side_filter="SELL")),
        ("buy-maker", _conditioned_markout(side_filter="BUY", taker_filter=False)),
        ("buy-taker", _conditioned_markout(side_filter="BUY", taker_filter=True)),
        ("sell-maker", _conditioned_markout(side_filter="SELL", taker_filter=False)),
        ("sell-taker", _conditioned_markout(side_filter="SELL", taker_filter=True)),
    ]:
        if vals:
            lines.append(f"    {label}: avg={(sum(vals) / Decimal(len(vals))):.2f}bps n={len(vals)}")
        else:
            lines.append(f"    {label}: n=0")

    lines.append("  Level toxicity (+5s markout):")
    for lvl_key in sorted(by_level):
        lvl_fills = by_level[lvl_key]
        lvl_m5 = [m for fill in lvl_fills
                   for m in [_markout_for_fill(fill, 5.0, ts_values, mid_values)] if m is not None]
        if lvl_m5:
            adverse = sum(1 for v in lvl_m5 if v < 0)
            lines.append(f"    {lvl_key}: avg_mo5={sum(lvl_m5) / Decimal(len(lvl_m5)):.2f}bps "
                         f"adverse={adverse}/{len(lvl_m5)}")
        else:
            lines.append(f"    {lvl_key}: mo5 unavailable")
    return lines


def _build_completeness_section(fills) -> List[str]:
    lines: List[str] = []
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
    lines.append(f"    fill_snapshot_present: "
                 f"{snapshot_present}/{len(fills)} "
                 f"({(Decimal(snapshot_present) / Decimal(len(fills)) * Decimal('100')):.1f}%)")
    lines.append(f"    fill_snapshot_top5_complete: "
                 f"{depth_complete}/{len(fills)} "
                 f"({(Decimal(depth_complete) / Decimal(len(fills)) * Decimal('100')):.1f}%)")
    return lines


def build_context_regime_section(
    fills: List[Dict[str, Any]],
    ts_values: List[float],
    mid_values: List[Decimal],
) -> List[str]:
    """Build the context regime analysis section."""
    lines: List[str] = ["## Context Regime Analysis"]
    contexts = {
        "spread_bps": (_bucket_spread, ["<2", "2-5", "5-10", ">10"], "Spread bucket (bps)"),
        "micro_vol_bps": (_bucket_micro_vol, ["<2", "2-5", ">5"], "Micro-vol bucket (bps)"),
        "micro_drift_bps": (_bucket_drift, ["negative", "neutral", "positive"], "Drift bucket"),
        "imbalance": (_bucket_imbalance, ["ask-heavy", "balanced", "bid-heavy"], "Imbalance bucket"),
    }
    for context_key, (bucket_fn, bucket_order, title) in contexts.items():
        lines.append(f"  {title}:")
        for bucket in bucket_order:
            subset = [f for f in fills
                      if bucket_fn(_fill_context_value(f, context_key)) == bucket]
            if not subset:
                lines.append(f"    {bucket}: n=0")
                continue
            edges = [_d(f.get("edge_bps")) for f in subset if f.get("edge_bps") is not None]
            mo: Dict[float, List[Decimal]] = {h: [] for h in _MARKOUT_HORIZONS}
            for f in subset:
                for h in _MARKOUT_HORIZONS:
                    m = _markout_for_fill(f, float(h), ts_values, mid_values)
                    if m is not None:
                        mo[h].append(m)
            adverse_5 = sum(1 for v in mo[5.0] if v < 0)
            adverse_5_pct = (
                Decimal(adverse_5) / Decimal(len(mo[5.0])) * Decimal("100")
                if mo[5.0] else Decimal("0")
            )
            cashflow_bucket = Decimal("0")
            for f in subset:
                qty = _d(f.get("qty"))
                px = _d(f.get("price"))
                sd = str(f.get("side", ""))
                if "SELL" in sd:
                    cashflow_bucket += qty * px
                else:
                    cashflow_bucket -= qty * px
            lines.append(
                f"    {bucket}: n={len(subset)} "
                f"avg_edge={_format_avg(edges)}bps "
                f"mo250ms={_format_avg(mo[0.25])}bps "
                f"mo1s={_format_avg(mo[1.0])}bps "
                f"mo5s={_format_avg(mo[5.0])}bps "
                f"mo30s={_format_avg(mo[30.0])}bps "
                f"mo2m={_format_avg(mo[120.0])}bps "
                f"adverse5={adverse_5_pct:.1f}% "
                f"cashflow=${cashflow_bucket:.2f}"
            )
    lines.append("")
    return lines


def _build_recent_fills_section(fills, ts_values, mid_values) -> List[str]:
    lines: List[str] = ["## Recent Fills (last 20)"]
    for f in fills[-20:]:
        side_char = "B" if "BUY" in str(f["side"]) else "S"
        edge_str = ""
        if f.get("edge_bps") is not None:
            edge_str = f" edge={_d(f['edge_bps']):+.1f}bps"
        spread_str = ""
        if f.get("spread_bps") is not None:
            spread_str = f" spread={_d(f['spread_bps']):.1f}bps"
        markout_str = ""
        mo = _markout_for_fill(f, 5.0, ts_values, mid_values)
        if mo is not None:
            markout_str = f" mo5={mo:+.1f}bps"
        taker_str = "T" if f.get("is_taker") else "M"
        lines.append(
            f"  {_ts_fmt(f['ts'])} {side_char} {f['qty']}@{f['price']} "
            f"fee={f['fee']} [{taker_str}]{edge_str}{spread_str}{markout_str}"
        )
    lines.append("")
    return lines


def build_order_section(
    orders: List[Dict[str, Any]],
    fills: List[Dict[str, Any]],
    rejections: List[Dict[str, Any]],
    cancellations: List[Dict[str, Any]],
    duration_m: float,
) -> List[str]:
    """Build the order placements section."""
    if not orders:
        return ["## Order Placements: NONE", ""]
    lines: List[str] = ["## Order Placements"]
    lines.append(f"  Total: {len(orders)}")
    rate = len(orders) / max(duration_m, 1)
    lines.append(f"  Rate: {rate:.1f}/min")

    spreads_at_place = [_d(o.get("spread_bps")) for o in orders if o.get("spread_bps") is not None]
    if spreads_at_place:
        lines.append(f"  Avg spread at placement: {sum(spreads_at_place) / len(spreads_at_place):.1f}bps")

    offsets: list[Decimal] = []
    for o in orders:
        price = _d(o["price"])
        if "BUY" in str(o["side"]) and o.get("best_bid"):
            offsets.append((_d(o["best_bid"]) - price) / _d(o["best_bid"]) * 10000)
        elif "SELL" in str(o["side"]) and o.get("best_ask"):
            offsets.append((price - _d(o["best_ask"])) / _d(o["best_ask"]) * 10000)
    if offsets:
        lines.append(f"  Avg offset from BBO: {sum(offsets) / len(offsets):.1f}bps")

    if rejections:
        reject_rate = Decimal(len(rejections)) / Decimal(len(orders)) * Decimal("100")
        pof_count = sum(1 for r in rejections if "POST_ONLY" in str(r.get("reason", "")).upper())
        pof_rate = Decimal(pof_count) / Decimal(len(orders)) * Decimal("100")
        lines.append(f"  Rejections: {len(rejections)} ({reject_rate:.2f}% of placements)")
        lines.append(f"  Post-only rejects: {pof_count} ({pof_rate:.2f}% of placements)")
    if cancellations:
        cancel_rate = Decimal(len(cancellations)) / Decimal(len(orders)) * Decimal("100")
        lines.append(f"  Cancellations: {len(cancellations)} ({cancel_rate:.2f}% of placements)")
    lines.append("")
    return lines


def build_reprice_section(
    reprice_decisions: List[Dict[str, Any]],
) -> List[str]:
    """Build the reprice decisions section."""
    if not reprice_decisions:
        return []
    lines: List[str] = ["## Reprice Decisions"]
    total = len(reprice_decisions)
    lines.append(f"  Total: {total}")

    reason_counts = Counter(str(e.get("reason") or "unknown") for e in reprice_decisions)
    lines.append("  Reasons:")
    for reason, count in reason_counts.most_common(12):
        pct = Decimal(count) / Decimal(total) * Decimal("100")
        lines.append(f"    {reason}: {count} ({pct:.1f}%)")

    skip_toxicity = reason_counts.get("skip_toxicity", 0)
    lines.append(f"  skip_toxicity share: {skip_toxicity}/{total} "
                 f"({Decimal(skip_toxicity) / Decimal(total) * Decimal('100'):.1f}%)")

    regime_counts = Counter(str(e.get("regime")) for e in reprice_decisions if e.get("regime") is not None)
    if regime_counts:
        lines.append("  Regimes:")
        for regime, count in regime_counts.most_common():
            lines.append(f"    {regime}: {count} ({Decimal(count) / Decimal(total) * Decimal('100'):.1f}%)")

    trend_counts = Counter(
        str(e.get("trend_direction")) for e in reprice_decisions if e.get("trend_direction") is not None
    )
    if trend_counts:
        lines.append("  Trend directions:")
        for d, count in trend_counts.most_common():
            lines.append(f"    {d}: {count} ({Decimal(count) / Decimal(total) * Decimal('100'):.1f}%)")

    band_counts = Counter(
        str(e.get("inventory_band")) for e in reprice_decisions if e.get("inventory_band") is not None
    )
    if band_counts:
        lines.append("  Inventory bands:")
        for band, count in band_counts.most_common():
            lines.append(f"    {band}: {count} ({Decimal(count) / Decimal(total) * Decimal('100'):.1f}%)")

    trend_strengths = [_d(e.get("trend_strength")) for e in reprice_decisions if e.get("trend_strength") is not None]
    if trend_strengths:
        lines.append(f"  Avg trend strength: {sum(trend_strengths) / Decimal(len(trend_strengths)):.3f}")

    funding_bias = [_d(e.get("funding_bias_bps")) for e in reprice_decisions if e.get("funding_bias_bps") is not None]
    if funding_bias:
        lines.append(f"  Avg funding bias: {sum(funding_bias) / Decimal(len(funding_bias)):.2f}bps")
    lines.append("")
    return lines
