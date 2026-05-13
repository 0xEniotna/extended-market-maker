#!/usr/bin/env python3
"""Per-fill markout diagnostic for MM journal data.

Question we answer: is adverse selection biting hard enough on this market
to justify Stage-2 (adverse-selection-aware quoting)?

For each ``fill`` event in a journal, we compute the MM-perspective
markout at +1s, +5s, +30s, +5min. Markout is signed so positive = good
for the MM:

    BUY fill:  markout_bps = (mid(t+Δ) - fill_price) / fill_price * 1e4
    SELL fill: markout_bps = (fill_price - mid(t+Δ)) / fill_price * 1e4

Adverse selection signature = negative mean markout (we're getting
picked off systematically). Symmetric AS = both sides bleed equally;
asymmetric AS = one side is much worse than the other.

The mid time series is built from every event that carries best_bid/
best_ask (snapshot, qtr_snapshot, fill, order_placed). Lookup is a
forward-step on a sorted ts list.

Usage:
    python scripts/diagnose_markout.py \\
        --market ETH-USD \\
        --journal data/mm_journal/mm_ETH-USD_xxx.jsonl \\
        --out docs/stage2_markout_ETH.md
"""
from __future__ import annotations

import argparse
import bisect
import json
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

# Horizons to evaluate (seconds after fill).
HORIZONS_S = [1, 5, 30, 300]


def _safe_decimal(x: str | float | int | None) -> Decimal | None:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except Exception:  # noqa: BLE001
        return None


def _mid(bid: Decimal | None, ask: Decimal | None) -> Decimal | None:
    if bid is None or ask is None:
        return None
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (bid + ask) / 2


def _build_mid_timeline(journal_path: Path) -> tuple[list[float], list[Decimal]]:
    """Extract (ts, mid) for every event that carries BBO. Sorted by ts."""
    ts_list: list[float] = []
    mid_list: list[Decimal] = []
    with journal_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = r.get("ts")
            if ts is None:
                continue
            bid = _safe_decimal(r.get("best_bid"))
            ask = _safe_decimal(r.get("best_ask"))
            if bid is None or ask is None:
                # Some events nest BBO under market_snapshot.
                ms = r.get("market_snapshot") or {}
                bid = _safe_decimal(ms.get("best_bid"))
                ask = _safe_decimal(ms.get("best_ask"))
            m = _mid(bid, ask)
            if m is None:
                continue
            ts_list.append(float(ts))
            mid_list.append(m)
    # The journal is monotonic in ts but we sort defensively.
    pairs = sorted(zip(ts_list, mid_list, strict=False), key=lambda p: p[0])
    ts_sorted = [p[0] for p in pairs]
    mid_sorted = [p[1] for p in pairs]
    return ts_sorted, mid_sorted


def _mid_at(
    ts_list: list[float], mid_list: list[Decimal], target_ts: float,
) -> Decimal | None:
    """First mid observation with ts >= target_ts (forward-step lookup)."""
    if not ts_list:
        return None
    idx = bisect.bisect_left(ts_list, target_ts)
    if idx >= len(ts_list):
        return None
    return mid_list[idx]


def _regime(spread_bps: float) -> str:
    if spread_bps < 5:
        return "calm(<5bps)"
    if spread_bps < 20:
        return "normal(5-20bps)"
    return "wide(>=20bps)"


def _edge_bucket(edge_bps: float | None) -> str:
    if edge_bps is None:
        return "unknown"
    if edge_bps < 0:
        return "neg_edge(<0)"
    if edge_bps < 5:
        return "tight(0-5)"
    if edge_bps < 15:
        return "med(5-15)"
    return "wide(>=15)"


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    n = len(values)
    s = sorted(values)
    mean = sum(values) / n
    median = s[n // 2]
    return {
        "count": n,
        "mean": mean,
        "median": median,
        "min": s[0],
        "max": s[-1],
        "p25": s[int(0.25 * (n - 1))],
        "p75": s[int(0.75 * (n - 1))],
        "p95": s[int(0.95 * (n - 1))],
        "n_neg": sum(1 for v in values if v < 0),
        "n_pos": sum(1 for v in values if v > 0),
        "stdev": statistics.stdev(values) if n >= 2 else 0.0,
    }


def diagnose(journal_path: Path) -> dict:
    ts_list, mid_list = _build_mid_timeline(journal_path)

    fills: list[dict] = []
    n_total = 0
    n_taker = 0
    sides: Counter = Counter()
    with journal_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "fill":
                continue
            n_total += 1
            try:
                ts = float(r["ts"])
                side = r["side"]
                price = float(r["price"])
                is_taker = bool(r.get("is_taker", False))
            except (KeyError, TypeError, ValueError):
                continue
            sides[side] += 1
            if is_taker:
                n_taker += 1
                # Taker fills are forced position-flat or shutdown flatten —
                # not representative of resting-order toxicity. Skip.
                continue

            edge_bps = r.get("edge_bps")
            try:
                edge_bps = float(edge_bps) if edge_bps is not None else None
            except (TypeError, ValueError):
                edge_bps = None
            spread_bps = r.get("spread_bps")
            try:
                spread_bps = float(spread_bps) if spread_bps is not None else None
            except (TypeError, ValueError):
                spread_bps = None

            # Markout at each horizon.
            markouts: dict[int, float | None] = {}
            for h in HORIZONS_S:
                m_future = _mid_at(ts_list, mid_list, ts + h)
                if m_future is None:
                    markouts[h] = None
                    continue
                if side == "BUY":
                    delta = float(m_future) - price
                else:
                    delta = price - float(m_future)
                # bps of fill price (positive = good for MM)
                markouts[h] = delta / price * 1e4

            fills.append({
                "ts": ts,
                "side": side,
                "price": price,
                "edge_bps": edge_bps,
                "spread_bps": spread_bps,
                "markouts": markouts,
            })

    return {
        "n_total_fills": n_total,
        "n_taker_fills": n_taker,
        "n_resting_fills": len(fills),
        "sides": dict(sides),
        "fills": fills,
        "mid_timeline_size": len(ts_list),
    }


def aggregate(diag: dict) -> dict:
    fills = diag["fills"]
    if not fills:
        return {"empty": True}
    out: dict = {"empty": False}

    # Overall markout per horizon.
    for h in HORIZONS_S:
        vals = [f["markouts"][h] for f in fills if f["markouts"][h] is not None]
        out[f"overall_h{h}s"] = _summary(vals)

    # By side.
    by_side: dict = {}
    for side in ("BUY", "SELL"):
        side_fills = [f for f in fills if f["side"] == side]
        by_side[side] = {
            f"h{h}s": _summary(
                [f["markouts"][h] for f in side_fills
                 if f["markouts"][h] is not None],
            ) for h in HORIZONS_S
        }
        by_side[side]["n"] = len(side_fills)
    out["by_side"] = by_side

    # By regime (use 5s as representative horizon).
    by_regime: dict = {}
    for f in fills:
        sp = f.get("spread_bps")
        if sp is None:
            continue
        reg = _regime(sp)
        by_regime.setdefault(reg, []).append(f["markouts"][5])
    out["by_regime_5s"] = {
        k: _summary([v for v in vs if v is not None])
        for k, vs in by_regime.items()
    }

    # By edge bucket (5s).
    by_edge: dict = {}
    for f in fills:
        eb = f.get("edge_bps")
        bucket = _edge_bucket(eb)
        by_edge.setdefault(bucket, []).append(f["markouts"][5])
    out["by_edge_5s"] = {
        k: _summary([v for v in vs if v is not None])
        for k, vs in by_edge.items()
    }

    return out


def render(market: str, journal_path: Path, diag: dict, agg: dict) -> str:
    lines: list[str] = [
        f"# Per-Fill Markout Diagnostic — {market}",
        "",
        f"- Journal: `{journal_path}`",
        f"- Total fills: {diag['n_total_fills']}",
        f"- Taker fills (excluded — shutdown flatten / hedge): {diag['n_taker_fills']}",
        f"- Resting-order fills analyzed: {diag['n_resting_fills']}",
        f"- Side distribution: {diag['sides']}",
        f"- Mid timeline size: {diag['mid_timeline_size']:,} observations",
        "",
        "Convention: **markout in bps, signed from MM perspective**.",
        "Positive = good for MM (we were on the right side of post-fill mid drift).",
        "Negative = adverse selection biting (we got picked off).",
        "",
    ]
    if agg.get("empty"):
        lines.append("No resting fills in window. Skipping aggregation.")
        return "\n".join(lines)

    lines += [
        "## Overall markout distribution (all resting fills)",
        "",
        "| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h in HORIZONS_S:
        s = agg[f"overall_h{h}s"]
        n = s.get("count", 0)
        if n == 0:
            lines.append(f"| +{h}s | 0 | – | – | – | – | – | – | – | – |")
            continue
        pneg = s["n_neg"] / n * 100 if n else 0
        ppos = s["n_pos"] / n * 100 if n else 0
        lines.append(
            f"| +{h}s | {n} | {s['mean']:+.3f} | {s['median']:+.3f} | "
            f"{s['p25']:+.3f} | {s['p75']:+.3f} | {s['p95']:+.3f} | "
            f"{pneg:.1f}% | {ppos:.1f}% | {s['stdev']:.3f} |",
        )

    lines += [
        "",
        "## By side (median across all horizons; look for asymmetry)",
        "",
        "| side | count | h1s mean | h5s mean | h30s mean | h300s mean |",
        "|---|---|---|---|---|---|",
    ]
    for side in ("BUY", "SELL"):
        bs = agg["by_side"][side]
        row = [side, str(bs["n"])]
        for h in HORIZONS_S:
            s = bs[f"h{h}s"]
            if s.get("count", 0) == 0:
                row.append("–")
            else:
                row.append(f"{s['mean']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## By regime at fill (5s markout)",
        "",
        "| regime | count | mean | median | %neg |",
        "|---|---|---|---|---|",
    ]
    for reg in ("calm(<5bps)", "normal(5-20bps)", "wide(>=20bps)"):
        s = agg["by_regime_5s"].get(reg, {"count": 0})
        if s.get("count", 0) == 0:
            lines.append(f"| {reg} | 0 | – | – | – |")
            continue
        pneg = s["n_neg"] / s["count"] * 100
        lines.append(
            f"| {reg} | {s['count']} | {s['mean']:+.3f} | "
            f"{s['median']:+.3f} | {pneg:.1f}% |",
        )

    lines += [
        "",
        "## By edge bucket at fill (5s markout)",
        "",
        "edge_bps = post-only edge from BBO at fill time. Tighter quote =",
        "more aggressive = should be more toxic if AS is biting.",
        "",
        "| edge bucket | count | mean | median | %neg |",
        "|---|---|---|---|---|",
    ]
    for bk in ("neg_edge(<0)", "tight(0-5)", "med(5-15)", "wide(>=15)", "unknown"):
        s = agg["by_edge_5s"].get(bk, {"count": 0})
        if s.get("count", 0) == 0:
            lines.append(f"| {bk} | 0 | – | – | – |")
            continue
        pneg = s["n_neg"] / s["count"] * 100
        lines.append(
            f"| {bk} | {s['count']} | {s['mean']:+.3f} | "
            f"{s['median']:+.3f} | {pneg:.1f}% |",
        )

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"== {args.market} ==")
    print(f"  journal: {args.journal}")
    diag = diagnose(args.journal)
    print(f"  total fills: {diag['n_total_fills']}")
    print(f"  taker fills (excluded): {diag['n_taker_fills']}")
    print(f"  resting fills analyzed: {diag['n_resting_fills']}")
    print(f"  mid observations: {diag['mid_timeline_size']:,}")

    agg = aggregate(diag)
    md = render(args.market, args.journal, diag, agg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"  report: {args.out}")
    if not agg.get("empty"):
        for h in HORIZONS_S:
            s = agg[f"overall_h{h}s"]
            if s.get("count", 0) == 0:
                continue
            print(f"  +{h}s mean markout: {s['mean']:+.3f} bps  "
                  f"(median {s['median']:+.3f}, n={s['count']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
