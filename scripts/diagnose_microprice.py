#!/usr/bin/env python3
"""Phase 1 / A1.1 — Microprice diagnostic.

Question we answer: does microprice differ from mid in a way that
predicts where mid is heading on a short (+5s) horizon? If null on
all markets, A1.2 (live quoting change) isn't worth shipping.

For each non-taker ``fill`` event:
  - Extract bid, ask, bid_size, ask_size from ``market_snapshot.bids_top[0]``
    and ``asks_top[0]``.
  - Compute microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size).
  - mp_minus_mid_bps = (microprice - mid) / mid * 1e4.
  - mid_after = first mid observation with ts >= fill_ts + 5s.
  - raw_markout_bps = (mid_after - fill_mid) / fill_mid * 1e4
    (NOT side-flipped — positive = mid went up, regardless of fill side).

Hypothesis: corr(mp_minus_mid_bps, raw_markout_bps) > 0
            (microprice predicts mid direction)

Decision criterion (pre-registered in
``docs/stage3_microprice_diagnostic.md``):
  PROCEED to A1.2 if |Pearson r| >= 0.05 with p < 0.05 on at least one
  market with n >= 100 fills, with the sign positive (microprice leads
  mid).

Usage:
    python scripts/diagnose_microprice.py \\
        --market DOT-USD \\
        --journal /root/MM/data/mm_journal/mm_DOT-USD_xxx.jsonl \\
        --journal /root/MM/data/mm_journal/mm_DOT-USD_yyy.jsonl \\
        --out docs/stage3_microprice_DOT.md
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path
from typing import Iterable

# Markout horizon (seconds after fill). 5s matches Stage 2 markout
# diagnostic convention.
HORIZON_S = 5.0
# Maximum gap allowed for the +Δs lookup. If no mid observation falls
# within [fill_ts + 5s, fill_ts + 30s] the fill is dropped.
MAX_LOOKAHEAD_GAP_S = 30.0


def _safe_decimal(x: str | float | int | None) -> Decimal | None:
    if x is None:
        return None
    try:
        return Decimal(str(x))
    except Exception:  # noqa: BLE001
        return None


def _mid(bid: Decimal | None, ask: Decimal | None) -> Decimal | None:
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (bid + ask) / 2


def _microprice(
    bid: Decimal, ask: Decimal, bid_size: Decimal, ask_size: Decimal,
) -> Decimal | None:
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (bid * ask_size + ask * bid_size) / total


def _build_mid_timeline(
    journals: Iterable[Path],
) -> tuple[list[float], list[Decimal]]:
    """Extract (ts, mid) from every event with BBO across all journals."""
    ts_list: list[float] = []
    mid_list: list[Decimal] = []
    for journal_path in journals:
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
    pairs = sorted(
        zip(ts_list, mid_list, strict=False), key=lambda p: p[0]
    )
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _mid_at(
    ts_list: list[float], mid_list: list[Decimal],
    target_ts: float, max_gap: float,
) -> Decimal | None:
    if not ts_list:
        return None
    idx = bisect.bisect_left(ts_list, target_ts)
    if idx >= len(ts_list):
        return None
    if ts_list[idx] - target_ts > max_gap:
        return None
    return mid_list[idx]


# ---------------------------------------------------------------------
# Correlation helpers (numpy/scipy not assumed installed)
# ---------------------------------------------------------------------

def _pearson(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Returns (r, two-sided p) or None if degenerate."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    if sxx <= 0 or syy <= 0:
        return None
    r = sxy / math.sqrt(sxx * syy)
    if abs(r) >= 1.0:
        return r, 0.0
    # Two-sided p-value via Fisher z-approximation (good enough for n>=30;
    # we don't need exact-distribution accuracy for go/no-go decisions).
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # Approximate two-sided p from the t-statistic using the normal
    # approximation. For n >= 30 this is within 5% of the exact value.
    p_one = 0.5 * math.erfc(abs(t) / math.sqrt(2))
    return r, min(1.0, 2 * p_one)


def _spearman(xs: list[float], ys: list[float]) -> tuple[float, float] | None:
    """Rank-based Spearman ρ, two-sided p (asymptotic)."""
    n = len(xs)
    if n < 3:
        return None
    def _ranks(vs: list[float]) -> list[float]:
        idx = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[idx[j + 1]] == vs[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx = _ranks(xs)
    ry = _ranks(ys)
    return _pearson(rx, ry)


# ---------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------

def diagnose(
    journals: list[Path], market: str,
) -> dict:
    ts_list, mid_list = _build_mid_timeline(journals)

    records: list[dict] = []
    n_total_fills = 0
    n_taker = 0
    n_no_book = 0
    n_no_markout = 0

    for journal_path in journals:
        with journal_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "fill":
                    continue
                n_total_fills += 1
                if bool(r.get("is_taker", False)):
                    n_taker += 1
                    continue
                try:
                    ts = float(r["ts"])
                    side = r["side"]
                except (KeyError, TypeError, ValueError):
                    continue
                ms = r.get("market_snapshot") or {}
                bids_top = ms.get("bids_top") or []
                asks_top = ms.get("asks_top") or []
                if not bids_top or not asks_top:
                    n_no_book += 1
                    continue
                bid = _safe_decimal(bids_top[0].get("price"))
                ask = _safe_decimal(asks_top[0].get("price"))
                bid_size = _safe_decimal(bids_top[0].get("size"))
                ask_size = _safe_decimal(asks_top[0].get("size"))
                if (
                    bid is None or ask is None
                    or bid_size is None or ask_size is None
                ):
                    n_no_book += 1
                    continue
                fill_mid = _mid(bid, ask)
                if fill_mid is None:
                    n_no_book += 1
                    continue
                mp = _microprice(bid, ask, bid_size, ask_size)
                if mp is None:
                    n_no_book += 1
                    continue

                mp_minus_mid_bps = float(
                    (mp - fill_mid) / fill_mid * Decimal("10000")
                )

                mid_after = _mid_at(
                    ts_list, mid_list, ts + HORIZON_S, MAX_LOOKAHEAD_GAP_S,
                )
                if mid_after is None:
                    n_no_markout += 1
                    continue
                raw_markout_bps = float(
                    (mid_after - fill_mid) / fill_mid * Decimal("10000")
                )

                # MM-perspective markout: positive = good for the MM
                # (matches diagnose_markout.py convention).
                if side == "BUY":
                    mm_markout_bps = raw_markout_bps
                else:  # SELL
                    mm_markout_bps = -raw_markout_bps

                records.append({
                    "ts": ts,
                    "side": side,
                    "fill_mid": float(fill_mid),
                    "mp_minus_mid_bps": mp_minus_mid_bps,
                    "raw_markout_bps": raw_markout_bps,
                    "mm_markout_bps": mm_markout_bps,
                    "bid_size": float(bid_size),
                    "ask_size": float(ask_size),
                })

    return {
        "market": market,
        "n_journals": len(journals),
        "n_total_fills": n_total_fills,
        "n_taker_skipped": n_taker,
        "n_no_book_skipped": n_no_book,
        "n_no_markout_skipped": n_no_markout,
        "n_used": len(records),
        "records": records,
    }


def _abs_thresholds(xs: list[float], thresholds: list[float]) -> dict:
    """Fraction of |xs| above each threshold."""
    if not xs:
        return {f"abs_ge_{t}": 0.0 for t in thresholds}
    out: dict = {}
    for t in thresholds:
        out[f"abs_ge_{t}"] = sum(1 for x in xs if abs(x) >= t) / len(xs)
    return out


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[int(q * (len(s) - 1))]


def analyze(result: dict) -> dict:
    recs = result["records"]
    mp_mm = [r["mp_minus_mid_bps"] for r in recs]
    raw_mk = [r["raw_markout_bps"] for r in recs]
    mm_mk = [r["mm_markout_bps"] for r in recs]
    buy_recs = [r for r in recs if r["side"] == "BUY"]
    sell_recs = [r for r in recs if r["side"] == "SELL"]

    summary = {
        "n_used": len(recs),
        "n_buy": len(buy_recs),
        "n_sell": len(sell_recs),
        "mp_minus_mid_bps": {
            "mean": (sum(mp_mm) / len(mp_mm)) if mp_mm else None,
            "stdev": statistics.stdev(mp_mm) if len(mp_mm) >= 2 else None,
            "p05": _pct(mp_mm, 0.05),
            "p50": _pct(mp_mm, 0.50),
            "p95": _pct(mp_mm, 0.95),
            **_abs_thresholds(mp_mm, [1.0, 3.0, 5.0]),
        },
        "mm_markout_bps": {
            "mean": (sum(mm_mk) / len(mm_mk)) if mm_mk else None,
            "stdev": statistics.stdev(mm_mk) if len(mm_mk) >= 2 else None,
            "p05": _pct(mm_mk, 0.05),
            "p50": _pct(mm_mk, 0.50),
            "p95": _pct(mm_mk, 0.95),
        },
        "pearson_mp_vs_raw_markout": _pearson(mp_mm, raw_mk),
        "spearman_mp_vs_raw_markout": _spearman(mp_mm, raw_mk),
    }

    # Per-side breakdowns: BUY uses signed markout (raw, since for a BUY
    # we want the same-sign relationship); SELL flips.
    for label, sub in [("buy", buy_recs), ("sell", sell_recs)]:
        if len(sub) >= 3:
            sub_mp = [r["mp_minus_mid_bps"] for r in sub]
            sub_raw = [r["raw_markout_bps"] for r in sub]
            summary[f"pearson_mp_vs_raw_markout_{label}"] = _pearson(
                sub_mp, sub_raw
            )
            summary[f"spearman_mp_vs_raw_markout_{label}"] = _spearman(
                sub_mp, sub_raw
            )
        else:
            summary[f"pearson_mp_vs_raw_markout_{label}"] = None

    return summary


def _fmt_corr(c: tuple[float, float] | None) -> str:
    if c is None:
        return "n/a"
    r, p = c
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    return f"{r:+.4f} (p={p:.4g}) {sig}".strip()


def _fmt_float(x: float | None, ndigits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{ndigits}f}"


def render_report(market: str, result: dict, summary: dict) -> str:
    lines = [
        f"# Stage 3 — Microprice Diagnostic — {market}",
        "",
        "## Funnel",
        "",
        f"- Journals scanned: {result['n_journals']}",
        f"- Total `fill` events: {result['n_total_fills']}",
        f"- Skipped (taker): {result['n_taker_skipped']}",
        f"- Skipped (no book snapshot): {result['n_no_book_skipped']}",
        f"- Skipped (no +5s mid lookup): {result['n_no_markout_skipped']}",
        f"- **Used in analysis: {result['n_used']}** "
        f"(BUY {summary['n_buy']} / SELL {summary['n_sell']})",
        "",
        "## (microprice − mid) distribution (bps of mid)",
        "",
        f"- mean: {_fmt_float(summary['mp_minus_mid_bps']['mean'])} bps",
        f"- stdev: {_fmt_float(summary['mp_minus_mid_bps']['stdev'])} bps",
        f"- p05 / p50 / p95: "
        f"{_fmt_float(summary['mp_minus_mid_bps']['p05'])} / "
        f"{_fmt_float(summary['mp_minus_mid_bps']['p50'])} / "
        f"{_fmt_float(summary['mp_minus_mid_bps']['p95'])} bps",
        f"- fraction \\|mp−mid\\| ≥ 1 bps: "
        f"{summary['mp_minus_mid_bps']['abs_ge_1.0']:.1%}",
        f"- fraction \\|mp−mid\\| ≥ 3 bps: "
        f"{summary['mp_minus_mid_bps']['abs_ge_3.0']:.1%}",
        f"- fraction \\|mp−mid\\| ≥ 5 bps: "
        f"{summary['mp_minus_mid_bps']['abs_ge_5.0']:.1%}",
        "",
        "## +5s raw-markout vs (microprice − mid)",
        "",
        "Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is "
        "directional (positive = mid went up), not side-flipped. Test: "
        "does microprice predict the direction of subsequent mid moves?",
        "",
        "- Pearson r (pooled): "
        f"{_fmt_corr(summary['pearson_mp_vs_raw_markout'])}",
        "- Spearman ρ (pooled): "
        f"{_fmt_corr(summary['spearman_mp_vs_raw_markout'])}",
        "- Pearson r (BUY fills): "
        f"{_fmt_corr(summary.get('pearson_mp_vs_raw_markout_buy'))}",
        "- Pearson r (SELL fills): "
        f"{_fmt_corr(summary.get('pearson_mp_vs_raw_markout_sell'))}",
        "",
        "## MM-perspective markout (sign-flipped; positive = good for MM)",
        "",
        f"- mean: {_fmt_float(summary['mm_markout_bps']['mean'])} bps",
        f"- stdev: {_fmt_float(summary['mm_markout_bps']['stdev'])} bps",
        f"- p05 / p50 / p95: "
        f"{_fmt_float(summary['mm_markout_bps']['p05'])} / "
        f"{_fmt_float(summary['mm_markout_bps']['p50'])} / "
        f"{_fmt_float(summary['mm_markout_bps']['p95'])} bps",
        "",
        "## Pre-registered decision criterion",
        "",
        "PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least",
        "one market with n ≥ 100 fills, sign positive (microprice leads mid).",
        "",
    ]
    r_pair = summary["pearson_mp_vs_raw_markout"]
    if summary["n_used"] < 100:
        lines.append(
            f"**Verdict for this market**: INCONCLUSIVE — only "
            f"{summary['n_used']} fills (need ≥100). Pool with other markets."
        )
    elif r_pair is None:
        lines.append("**Verdict for this market**: degenerate (no variance).")
    else:
        r, p = r_pair
        if abs(r) >= 0.05 and p < 0.05 and r > 0:
            lines.append(
                f"**Verdict for this market**: PASS "
                f"(r={r:+.4f}, p={p:.4g})."
            )
        elif abs(r) >= 0.05 and p < 0.05 and r < 0:
            lines.append(
                f"**Verdict for this market**: WRONG SIGN "
                f"(r={r:+.4f}, p={p:.4g}) — microprice anti-predicts mid. "
                f"Investigate before shipping."
            )
        else:
            lines.append(
                f"**Verdict for this market**: NULL "
                f"(r={r:+.4f}, p={p:.4g})."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", required=True)
    ap.add_argument(
        "--journal", action="append", required=True, type=Path,
        help="Journal file (repeatable, pooled).",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    journals = [p for p in args.journal if p.exists()]
    missing = [p for p in args.journal if not p.exists()]
    for p in missing:
        print(f"WARNING: skipping missing journal {p}")

    result = diagnose(journals, args.market)
    summary = analyze(result)
    report = render_report(args.market, result, summary)

    print(report)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
