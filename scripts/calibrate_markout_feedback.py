#!/usr/bin/env python3
"""Calibrate the ETH markout-feedback overlay against journal data.

The overlay maintains a per-side EWMA of recent +5s post-fill markouts and
widens the bleeding side when EWMA drops below a threshold. For this to
actually reduce bleed in practice, post-fill markouts must be **temporally
autocorrelated** — i.e., a bad-markout streak must persist long enough
for the policy to react before the next adverse fill.

Phase 1 = calibration. Two distinct questions:

  1. **Is the AS bleed clustered in time** (autocorrelated per side)?
     If yes, a reactive feedback policy can target it. If no, the policy
     is solving the wrong problem.

  2. **For each parameter combo (half_life, threshold, gain, cap)**, what
     does the policy do? Specifically:
     - How often is it active (widening > 1 bps)?
     - Average widening when active?
     - Conditional markout: do fills during *active* periods have worse
       markout than fills during *inactive* periods? If yes, the policy
       is correctly targeting the bad windows.

The script does NOT model fill-counterfactuals (queue position, market-
order depth) — that's Phase 4 territory. Here we just measure whether
the policy targets the right moments.

Usage:
    python scripts/calibrate_markout_feedback.py \\
        --market ETH-USD \\
        --journal /root/MM/data/mm_journal/mm_ETH-USD_xxx.jsonl \\
        --journal /root/MM/data/mm_journal/mm_ETH-USD_yyy.jsonl \\
        --markout-horizon-s 5 \\
        --out docs/stage2_calibration_ETH.md
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from decimal import Decimal
from pathlib import Path


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
            mid_list.append(float(m))
    pairs = sorted(zip(ts_list, mid_list, strict=False), key=lambda p: p[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _mid_at(ts_list, mid_list, target_ts):
    if not ts_list:
        return None
    idx = bisect.bisect_left(ts_list, target_ts)
    if idx >= len(ts_list):
        return None
    return mid_list[idx]


def _load_fills_with_markout(journal_path: Path, horizon_s: int):
    """Return chronological [(ts, side, price, edge_bps, markout_bps)]."""
    ts_list, mid_list = _build_mid_timeline(journal_path)
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
                price = float(r["price"])
            except (KeyError, TypeError, ValueError):
                continue
            edge_bps = r.get("edge_bps")
            try:
                edge_bps = float(edge_bps) if edge_bps is not None else None
            except (TypeError, ValueError):
                edge_bps = None
            m_future = _mid_at(ts_list, mid_list, ts + horizon_s)
            if m_future is None:
                continue
            delta = m_future - price if side == "BUY" else price - m_future
            markout_bps = delta / price * 1e4
            out.append((ts, side, price, edge_bps, markout_bps))
    return out


# ---------------------------------------------------------------------------
# Question 1: temporal autocorrelation of markout per side
# ---------------------------------------------------------------------------


def autocorrelation_analysis(fills, max_lag=10):
    """Per-side, lag-k autocorrelation of markout sequence (ordered by ts).

    If markouts are random per fill, autocorrelation ≈ 0 → reactive policy
    cannot anticipate. If autocorrelation > 0 for small lags, bleed is
    sticky and the policy can target it.
    """
    out = {}
    for side in ("BUY", "SELL"):
        seq = [f[4] for f in fills if f[1] == side]
        n = len(seq)
        if n < max_lag + 5:
            out[side] = {"n": n, "lags": {}}
            continue
        mean = sum(seq) / n
        var = sum((x - mean) ** 2 for x in seq) / n
        lags = {}
        for k in range(1, max_lag + 1):
            cov = sum((seq[i] - mean) * (seq[i + k] - mean)
                      for i in range(n - k)) / (n - k)
            lags[k] = cov / var if var > 0 else 0.0
        out[side] = {"n": n, "lags": lags, "mean": mean, "stdev": var ** 0.5}
    return out


def streakiness_analysis(fills, window_size=5):
    """For each fill, compute mean markout of the prior K fills on same side.

    Then compute correlation between (prior-mean) and (this fill's markout).
    A positive correlation says: a streak of bad markouts predicts the next
    fill will also be bad — exactly the signal a reactive policy needs.
    """
    out = {}
    for side in ("BUY", "SELL"):
        side_fills = [(f[0], f[4]) for f in fills if f[1] == side]
        side_fills.sort(key=lambda p: p[0])
        if len(side_fills) < window_size + 5:
            out[side] = {"n": 0}
            continue
        pairs = []
        for i in range(window_size, len(side_fills)):
            prior_mean = sum(side_fills[j][1] for j in range(i - window_size, i)) \
                         / window_size
            pairs.append((prior_mean, side_fills[i][1]))
        n = len(pairs)
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        sxy = sum((x - mx) * (y - my) for x, y in pairs)
        pearson = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
        out[side] = {"n": n, "pearson": pearson, "window_size": window_size}
    return out


# ---------------------------------------------------------------------------
# Question 2: policy parameter sweep
# ---------------------------------------------------------------------------


def simulate_policy(
    fills, *, half_life_s: float, threshold_bps: float,
    gain: float, cap_bps: float, horizon_s: int,
):
    """Walk the fill sequence chronologically, maintaining per-side EWMA.

    At each fill, record:
      - the widening the policy would have applied to the fill's side
        (based on EWMA just before this fill)
      - whether the policy was 'active' (widening > 1 bps)
      - the realized markout of this fill (to check if active periods
        coincide with bad markouts)

    The EWMA updates AFTER the fill, with the realized horizon-s markout.
    Decay constant: per second, factor = exp(-ln(2) / half_life_s).
    """
    decay_per_sec = math.exp(-math.log(2) / half_life_s) if half_life_s > 0 else 0.0
    ewma = {"BUY": 0.0, "SELL": 0.0}
    last_ts = {"BUY": None, "SELL": None}
    records = []

    for ts, side, price, edge_bps, markout in fills:
        # Decay EWMA on this side from last update to now.
        if last_ts[side] is not None:
            dt = max(0.0, ts - last_ts[side])
            ewma[side] *= decay_per_sec ** dt
        # Compute widening BEFORE incorporating this fill (causal).
        if ewma[side] < -threshold_bps:
            raw_widen = gain * (-ewma[side] - threshold_bps)
            widen = min(raw_widen, cap_bps)
        else:
            widen = 0.0
        active = widen > 1.0
        records.append({
            "ts": ts, "side": side, "edge_bps": edge_bps,
            "markout": markout, "widen_bps": widen, "active": active,
            "ewma_before": ewma[side],
        })
        # Update EWMA with this fill's markout, accounting for the
        # horizon_s delay (we'd only know markout at ts + horizon_s).
        # For calibration purposes treat update as immediate; the delay
        # would slightly slow the policy but not change the
        # qualitative picture.
        ewma[side] = ewma[side] + markout  # additive EWMA update
        last_ts[side] = ts
    return records


def policy_metrics(records):
    if not records:
        return {"n": 0}
    n = len(records)
    active = [r for r in records if r["active"]]
    inactive = [r for r in records if not r["active"]]
    widenings = [r["widen_bps"] for r in records if r["widen_bps"] > 0]
    active_markouts = [r["markout"] for r in active]
    inactive_markouts = [r["markout"] for r in inactive]

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": n,
        "n_active": len(active),
        "pct_active": len(active) / n * 100 if n else 0.0,
        "mean_widening_when_active": mean([r["widen_bps"] for r in active]),
        "max_widening": max(widenings) if widenings else 0.0,
        "mean_markout_active": mean(active_markouts),
        "mean_markout_inactive": mean(inactive_markouts),
        "markout_diff_active_vs_inactive": mean(active_markouts) - mean(inactive_markouts),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(market: str, journals: list[Path], fills_list, auto, streak,
           sweep_results, horizon_s: int) -> str:
    lines = [
        f"# Markout-Feedback Calibration — {market}",
        "",
        f"- Markout horizon: +{horizon_s}s",
        f"- N journals: {len(journals)}",
    ]
    for j in journals:
        lines.append(f"- Journal: `{j}`")
    lines.append(f"- Total resting fills analyzed: {len(fills_list)}")
    lines.append("")

    lines += [
        "## Question 1a — temporal autocorrelation per side",
        "",
        "If markouts are independent draws (no clustering), all lag-k",
        "autocorrelations ≈ 0 → a reactive feedback policy can't anticipate",
        "and the overlay is mostly noise.",
        "If lag-1 autocorrelation is meaningfully positive (>0.10),",
        "bleed streaks are real and the policy can target them.",
        "",
        "| side | n | mean | stdev | lag1 | lag2 | lag5 | lag10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for side in ("BUY", "SELL"):
        a = auto.get(side, {})
        if a.get("n", 0) == 0:
            lines.append(f"| {side} | 0 | – | – | – | – | – | – |")
            continue
        lags = a["lags"]
        lines.append(
            f"| {side} | {a['n']} | {a['mean']:+.3f} | {a['stdev']:.3f} | "
            f"{lags.get(1, 0):+.3f} | {lags.get(2, 0):+.3f} | "
            f"{lags.get(5, 0):+.3f} | {lags.get(10, 0):+.3f} |",
        )

    lines += [
        "",
        "## Question 1b — streakiness (prior-K mean predicts next markout)",
        "",
        "For each fill, compute mean markout of the K prior fills on the",
        "same side. Pearson correlation between (prior-K mean) and (this",
        "fill's markout) measures how reliably a recent streak predicts",
        "the next fill. >0.10 = exploitable. ≈0 = independent draws.",
        "",
        "| side | n pairs | window K | Pearson(prior-K, this) |",
        "|---|---|---|---|",
    ]
    for side in ("BUY", "SELL"):
        s = streak.get(side, {})
        if s.get("n", 0) == 0:
            lines.append(f"| {side} | 0 | – | – |")
            continue
        lines.append(
            f"| {side} | {s['n']} | {s['window_size']} | "
            f"{s['pearson']:+.4f} |",
        )

    lines += [
        "",
        "## Question 2 — parameter sweep",
        "",
        "For each combo, simulate the policy chronologically. Metrics:",
        "- `%active`: fraction of fills that occurred while the policy",
        "  was widening (>1 bps).",
        "- `mean_widening_active`: average widening when active.",
        "- `markout_active vs markout_inactive`: if the policy targets",
        "  correctly, fills during active periods should have **more",
        "  negative** markout than fills during inactive periods. The",
        "  diff column shows (active − inactive); large negative = good.",
        "",
        "| half_life | thresh | gain | cap | n | %active | mean_widen | "
        "mark_act | mark_inact | diff |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for combo, m in sweep_results:
        hl, th, g, cap = combo
        if m["n"] == 0:
            continue
        lines.append(
            f"| {hl}s | {th} | {g} | {cap} | {m['n']} | "
            f"{m['pct_active']:.1f}% | "
            f"{m['mean_widening_when_active']:+.2f} | "
            f"{m['mean_markout_active']:+.3f} | "
            f"{m['mean_markout_inactive']:+.3f} | "
            f"{m['markout_diff_active_vs_inactive']:+.3f} |",
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "**Verdict on the policy depends on the sign and magnitude of the",
        "diff column.** If `diff` is consistently negative (active periods",
        "had worse markouts), the policy correctly identifies bleed",
        "windows — it would widen at exactly the right times.",
        "",
        "If `diff` is ~0 or positive, the policy fires randomly and the",
        "overlay should NOT be built — markouts are independent draws",
        "and feedback can't help.",
        "",
        "The `%active` column is the operational cost: how often the",
        "policy widens our quotes (and thus drops some fills).",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", required=True)
    p.add_argument("--journal", type=Path, action="append", required=True)
    p.add_argument("--markout-horizon-s", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    print(f"== {args.market} ({len(args.journal)} journal(s)) ==")
    all_fills = []
    for j in args.journal:
        print(f"  journal: {j}")
        all_fills.extend(_load_fills_with_markout(j, args.markout_horizon_s))
    all_fills.sort(key=lambda f: f[0])
    print(f"  total resting fills with markout: {len(all_fills)}")

    print("  computing autocorrelation...")
    auto = autocorrelation_analysis(all_fills, max_lag=10)
    for side, a in auto.items():
        if a.get("n", 0) == 0:
            continue
        print(f"    {side}: n={a['n']} mean={a['mean']:+.3f} "
              f"lag1={a['lags'][1]:+.3f} lag5={a['lags'][5]:+.3f}")

    print("  computing streakiness...")
    streak = streakiness_analysis(all_fills, window_size=5)
    for side, s in streak.items():
        if s.get("n", 0) == 0:
            continue
        print(f"    {side}: n={s['n']} pearson(prior5, this)={s['pearson']:+.4f}")

    print("  running parameter sweep...")
    sweep = []
    grid = [
        (hl, th, g, cap)
        for hl in (30, 60, 120, 300)
        for th in (0.5, 1.0, 2.0)
        for g in (0.5, 1.0, 2.0)
        for cap in (5.0, 10.0, 20.0)
    ]
    for combo in grid:
        hl, th, g, cap = combo
        records = simulate_policy(
            all_fills,
            half_life_s=hl, threshold_bps=th, gain=g, cap_bps=cap,
            horizon_s=args.markout_horizon_s,
        )
        m = policy_metrics(records)
        sweep.append((combo, m))

    # Sort: best (most-negative diff) first.
    sweep_sorted = sorted(sweep, key=lambda x: x[1].get(
        "markout_diff_active_vs_inactive", 0.0))

    md = render(args.market, args.journal, all_fills, auto, streak,
                sweep_sorted, args.markout_horizon_s)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"  report: {args.out}")

    print("  --- top-5 combos by diff (most-negative = best targeting) ---")
    for combo, m in sweep_sorted[:5]:
        hl, th, g, cap = combo
        print(f"    hl={hl}s th={th} g={g} cap={cap}: "
              f"%active={m['pct_active']:.1f}% "
              f"diff={m['markout_diff_active_vs_inactive']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
