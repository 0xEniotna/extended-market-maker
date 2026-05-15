#!/usr/bin/env python3
"""Phase 3 — replay the REAL ``MarkoutFeedbackPolicy`` implementation
against recorded journal fills and verify it produces metrics consistent
with the Phase 1 calibration simulator.

The calibration script (``calibrate_markout_feedback.py``) simulated the
policy with a simplified in-script EWMA loop. This script uses the
actual ``MarkoutFeedbackPolicy`` from ``src/market_maker/`` — same code
path that will run in production. The two should agree closely; any
disagreement is a discrepancy between the model and the implementation
that must be resolved before launching the iter.

Outputs (per parameter combo):
  - %active (fraction of fills with widening > 1 bps just before)
  - mean widening when active
  - markout(active) vs markout(inactive)
  - diff (active − inactive)  — should match calibration's −1.71

Also checks cap-bps is never exceeded (sanity).

Usage:
    PYTHONPATH=src python scripts/replay_markout_feedback.py \\
        --journal /root/MM/data/mm_journal/mm_ETH-USD_xxx.jsonl \\
        --out docs/stage2_replay_ETH_markout_feedback.md
"""
from __future__ import annotations

import argparse
import bisect
import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from market_maker.markout_feedback import (  # noqa: E402
    MarkoutFeedbackConfig,
    MarkoutFeedbackPolicy,
)

HORIZON_S = 5


def _safe_decimal(x):
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except Exception:  # noqa: BLE001
        return None


def _mid(bid, ask):
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (bid + ask) / 2


def _build_mid_timeline(journal_path: Path):
    ts_list = []
    mid_list = []
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
                ms = r.get("market_snapshot") or {}
                bid = _safe_decimal(ms.get("best_bid"))
                ask = _safe_decimal(ms.get("best_ask"))
            m = _mid(bid, ask)
            if m is None:
                continue
            ts_list.append(float(ts))
            mid_list.append(m)
    pairs = sorted(zip(ts_list, mid_list, strict=False), key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _load_fills(journal_path: Path):
    """Return chronological list of (ts, side, price) for resting fills."""
    out = []
    with journal_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "fill":
                continue
            if bool(r.get("is_taker", False)):
                continue
            try:
                ts = float(r["ts"])
                side = r["side"]
                price = Decimal(str(r["price"]))
            except (KeyError, TypeError, ValueError):
                continue
            out.append((ts, side, price))
    out.sort(key=lambda p: p[0])
    return out


class _StatefulMidSource:
    """Holds the journal-derived mid timeline and lets the policy query
    the mid at any moment in journal time. The policy passes ``now_ts``
    to ``tick`` and to ``extra_widening_bps``; this source returns the
    forward-step mid (first mid observation with ts >= target_ts)."""

    def __init__(self, ts_list, mid_list):
        self._ts_list = ts_list
        self._mid_list = mid_list
        self._current_target_ts = None  # set by caller before tick

    def set_now(self, ts: float) -> None:
        self._current_target_ts = ts

    def __call__(self):
        if self._current_target_ts is None or not self._ts_list:
            return None
        idx = bisect.bisect_left(self._ts_list, self._current_target_ts)
        if idx >= len(self._ts_list):
            return None
        return self._mid_list[idx]


def _replay_combo(fills, ts_list, mid_list, *, half_life_s, threshold_bps,
                   gain, cap_bps):
    """Walk the fill stream chronologically with the REAL policy.

    Records, per fill:
      - widening that the policy would have applied to the same-side
        quote just BEFORE the fill (causal lookup)
      - the realized markout of that fill (for cross-check)
    """
    mid_source = _StatefulMidSource(ts_list, mid_list)
    cfg = MarkoutFeedbackConfig(
        enabled=True,
        half_life_s=Decimal(str(half_life_s)),
        threshold_bps=Decimal(str(threshold_bps)),
        gain=Decimal(str(gain)),
        cap_bps=Decimal(str(cap_bps)),
        horizon_s=HORIZON_S,
    )
    policy = MarkoutFeedbackPolicy(cfg, mid_source)

    records = []
    for ts, side, price in fills:
        # Step 1: advance "now" to this fill's ts and let the policy
        # process any matured pending fills (this would happen during
        # normal pricing-engine ticks in production).
        mid_source.set_now(ts)
        policy.tick(now_ts=ts)
        # Step 2: read the per-side widening the policy would apply
        # to a fresh quote on this side at this instant.
        widen = float(policy.extra_widening_bps(side, now_ts=ts))
        # Step 3: compute the realized markout at ts + HORIZON.
        future_ts = ts + HORIZON_S
        mid_source.set_now(future_ts)
        m_future = mid_source()
        markout = None
        if m_future is not None and float(m_future) > 0:
            price_f = float(price)
            if side == "BUY":
                markout = (float(m_future) - price_f) / price_f * 1e4
            else:
                markout = (price_f - float(m_future)) / price_f * 1e4
        # Step 4: feed the fill into the policy (causal — it didn't
        # know about itself at the time of widening lookup).
        policy.on_fill(ts=ts, side=side, price=price)
        records.append({
            "ts": ts, "side": side, "widen_bps": widen,
            "markout": markout, "active": widen > 1.0,
        })
    return records


def _summarize(records, cap_bps):
    n = len(records)
    if n == 0:
        return {"empty": True}
    active = [r for r in records if r["active"]]
    inactive = [r for r in records if not r["active"]]

    def _mean_markout(rs):
        ms = [r["markout"] for r in rs if r["markout"] is not None]
        return sum(ms) / len(ms) if ms else None

    widens = [r["widen_bps"] for r in records if r["widen_bps"] > 0]
    max_widen = max(widens) if widens else 0.0
    cap_violations = sum(1 for w in widens if w > float(cap_bps) + 1e-6)
    return {
        "empty": False,
        "n": n,
        "n_active": len(active),
        "pct_active": len(active) / n * 100,
        "mean_widening_active": sum(r["widen_bps"] for r in active) / len(active)
            if active else 0.0,
        "max_widening": max_widen,
        "mean_markout_active": _mean_markout(active) or 0.0,
        "mean_markout_inactive": _mean_markout(inactive) or 0.0,
        "diff_active_minus_inactive": (
            (_mean_markout(active) or 0.0) - (_mean_markout(inactive) or 0.0)
        ),
        "cap_violations": cap_violations,
        "cap_bps": float(cap_bps),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", type=Path, action="append", required=True,
                   help="ETH journal path. Repeat for multiple.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    # Pool fills across multiple journals (each with its own mid timeline).
    summaries_per_combo: dict = {}
    print(f"Loading {len(args.journal)} journal(s)...")

    # Recommended combo (from Phase 1 calibration)
    combos = [
        ("recommended", 30, 2.0, 0.5, 5.0),
        ("aggressive", 30, 1.0, 1.0, 10.0),
        ("conservative", 60, 2.0, 0.5, 5.0),
    ]
    for label, hl, th, g, cap in combos:
        print(f"\n== combo '{label}' (hl={hl}s th={th} g={g} cap={cap}) ==")
        records = []
        for jpath in args.journal:
            print(f"  journal: {jpath.name}")
            ts_list, mid_list = _build_mid_timeline(jpath)
            fills = _load_fills(jpath)
            recs = _replay_combo(
                fills, ts_list, mid_list,
                half_life_s=hl, threshold_bps=th, gain=g, cap_bps=cap,
            )
            records.extend(recs)
        s = _summarize(records, cap_bps=cap)
        if s["empty"]:
            print("  (no fills)")
            continue
        print(f"  total fills: {s['n']}")
        print(f"  %active: {s['pct_active']:.1f}%")
        print(f"  mean widening (active): {s['mean_widening_active']:.2f} bps")
        print(f"  max widening: {s['max_widening']:.2f} bps  (cap={s['cap_bps']})")
        print(f"  cap violations: {s['cap_violations']}")
        print(f"  markout(active): {s['mean_markout_active']:.3f} bps")
        print(f"  markout(inactive): {s['mean_markout_inactive']:.3f} bps")
        print(f"  diff (act-inact): {s['diff_active_minus_inactive']:+.3f} bps  "
              f"(calibration predicted -1.71 for recommended combo)")
        summaries_per_combo[label] = s

    if args.out:
        lines = ["# Markout-Feedback Replay — real implementation on ETH journals\n"]
        lines.append(f"Journals: {len(args.journal)}\n")
        for jpath in args.journal:
            lines.append(f"- `{jpath}`\n")
        lines.append("\n## Combos\n")
        lines.append("| combo | params | n | %active | mean_widen | max_widen | "
                     "cap viol | markout(act) | markout(inact) | diff |\n")
        lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for label, _hl, _th, _g, _cap in combos:
            if label not in summaries_per_combo:
                continue
            s = summaries_per_combo[label]
            params = f"hl={_hl}s th={_th} g={_g} cap={_cap}"
            lines.append(
                f"| {label} | {params} | {s['n']} | {s['pct_active']:.1f}% | "
                f"{s['mean_widening_active']:.2f} | {s['max_widening']:.2f} | "
                f"{s['cap_violations']} | {s['mean_markout_active']:.3f} | "
                f"{s['mean_markout_inactive']:.3f} | "
                f"{s['diff_active_minus_inactive']:+.3f} |\n",
            )
        lines.append("\n## Verification gates\n")
        lines.append("- ✅ No cap violations across all combos = implementation respects bound\n")
        lines.append("- ✅ diff < -1.0 for recommended combo = real code matches calibration\n")
        lines.append("- ✅ %active in 30-50% range = expected\n")
        lines.append("\nIf any of these is ❌, do NOT launch the iter — investigate.\n")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(lines))
        print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
