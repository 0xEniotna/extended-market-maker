#!/usr/bin/env python3
"""Stage 4 — OFI diagnostic. Does order-flow imbalance predict the next
mid move on our markets, and in which sign?

Uses the `book_change` journal events (top-of-book mutations: bid, bid_qty,
ask, ask_qty). Reconstructs two candidate signals and correlates each with
the subsequent forward mid return:

  1. Flow-OFI (Cont-Kukanov-Stoikov, per consecutive book event):
       e = bid_qty           if bid_t  >  bid_{t-1}
           bid_qty - bid_qty_{t-1}  if bid_t == bid_{t-1}
          -bid_qty_{t-1}      if bid_t  <  bid_{t-1}
       f = ask_qty           if ask_t  <  ask_{t-1}
           ask_qty - ask_qty_{t-1}  if ask_t == ask_{t-1}
          -ask_qty_{t-1}      if ask_t  >  ask_{t-1}
       OFI_t = e - f       (positive = net buy pressure)
     Trailing OFI over window W = sum of OFI_t in [t-W, t].

  2. Depth-imbalance (instantaneous, the simpler proxy already in the bot):
       imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)

Forward signal: mid return over horizon H, in bps:
       fwd = (mid(t+H) - mid(t)) / mid(t) * 1e4

VERDICT (the whole point — what's the sign?):
  - corr > 0  : OFI predicts CONTINUATION (trend). For MM: lean DEFENSIVELY
                — pull the quote on the side flow is pushing toward (else
                you get run over). This is the crypto-trending case.
  - corr < 0  : OFI predicts REVERSION. For MM: lean AGGRESSIVELY into the
                imbalance (the Brief-18 / equity-index case).
  - corr ~ 0  : OFI carries no exploitable short-horizon signal here.

To avoid overlapping-window autocorrelation inflating significance, points
are sampled at a spacing ≥ H so forward windows don't overlap.

Usage (on VPS):
    python scripts/diagnose_ofi.py --journal /root/MM/data/mm_journal/mm_DOT-USD_xxx.jsonl --out docs/stage4_ofi_DOT.md
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

WINDOWS_S = [5.0, 30.0]
HORIZONS_S = [5.0, 30.0]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _load_book_changes(journal: str):
    """Return chronological list of (ts, bid, bid_qty, ask, ask_qty, mid)."""
    out = []
    with open(journal) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "book_change":
                continue
            bid = _f(r.get("bid"))
            ask = _f(r.get("ask"))
            bq = _f(r.get("bid_qty"))
            aq = _f(r.get("ask_qty"))
            ts = r.get("ts")
            if None in (bid, ask, bq, aq, ts):
                continue
            if bid <= 0 or ask <= 0 or ask <= bid or bq <= 0 or aq <= 0:
                continue
            out.append((float(ts), bid, bq, ask, aq, (bid + ask) / 2))
    return out


def _ofi_contrib(prev, curr):
    """Cont-Kukanov-Stoikov OFI for one consecutive book pair."""
    _, pbid, pbq, pask, paq, _ = prev
    _, bid, bq, ask, aq, _ = curr
    if bid > pbid:
        e = bq
    elif bid == pbid:
        e = bq - pbq
    else:
        e = -pbq
    if ask < pask:
        f = aq
    elif ask == pask:
        f = aq - paq
    else:
        f = -paq
    return e - f


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0, n
    return sxy / math.sqrt(sxx * syy), n


def _spearman(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    rho, _ = _pearson(rx, ry)
    return rho


def _mid_at(ts_list, mid_list, target):
    import bisect
    idx = bisect.bisect_left(ts_list, target)
    if idx >= len(ts_list):
        return None
    return mid_list[idx]


def analyze(events, window_s, horizon_s):
    """Sample points spaced >= horizon; correlate trailing OFI / depth-imb
    with forward mid return."""
    import bisect
    ts_list = [e[0] for e in events]
    mid_list = [e[5] for e in events]
    # Precompute per-event OFI contributions.
    contribs = [0.0]
    for i in range(1, len(events)):
        contribs.append(_ofi_contrib(events[i - 1], events[i]))

    ofi_x, imb_x, fwd_y = [], [], []
    last_sample_ts = None
    for i in range(len(events)):
        ts, bid, bq, ask, aq, mid = events[i]
        # Space samples by >= horizon to avoid overlapping forward windows.
        if last_sample_ts is not None and ts - last_sample_ts < horizon_s:
            continue
        # Trailing OFI over [ts-window, ts].
        lo = bisect.bisect_left(ts_list, ts - window_s)
        trailing_ofi = sum(contribs[lo:i + 1])
        # Depth imbalance now.
        imb = (bq - aq) / (bq + aq)
        # Forward return over horizon.
        m_fwd = _mid_at(ts_list, mid_list, ts + horizon_s)
        if m_fwd is None:
            continue
        fwd = (m_fwd - mid) / mid * 1e4
        ofi_x.append(trailing_ofi)
        imb_x.append(imb)
        fwd_y.append(fwd)
        last_sample_ts = ts

    r_ofi, n = _pearson(ofi_x, fwd_y)
    rho_ofi = _spearman(ofi_x, fwd_y)
    r_imb, _ = _pearson(imb_x, fwd_y)
    rho_imb = _spearman(imb_x, fwd_y)
    return {
        "n": n,
        "ofi_pearson": r_ofi, "ofi_spearman": rho_ofi,
        "imb_pearson": r_imb, "imb_spearman": rho_imb,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", action="append", required=True,
                   help="book_change journal(s); repeat to pool a market's rotations.")
    p.add_argument("--market", default="?")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    events = []
    for j in args.journal:
        ev = _load_book_changes(j)
        print(f"  {os.path.basename(j)}: {len(ev):,} book_change events")
        events.extend(ev)
    events.sort(key=lambda e: e[0])
    print(f"  total: {len(events):,} events")

    lines = [f"# Stage 4 OFI diagnostic — {args.market}", "",
             f"book_change events: {len(events):,}", "",
             "Sign convention: corr>0 = OFI predicts CONTINUATION (trend → "
             "MM leans defensively); corr<0 = REVERSION (MM leans aggressively); "
             "~0 = no exploitable signal.", "",
             "| window | horizon | n | OFI Pearson | OFI Spearman | depth-imb Pearson | depth-imb Spearman |",
             "|---|---|---|---|---|---|---|"]
    print(f"\n{'W':>4} {'H':>4} {'n':>6} {'OFI_r':>8} {'OFI_rho':>8} {'imb_r':>8} {'imb_rho':>8}")
    for w in WINDOWS_S:
        for h in HORIZONS_S:
            res = analyze(events, w, h)
            print(f"{w:>4.0f} {h:>4.0f} {res['n']:>6} "
                  f"{res['ofi_pearson']:>+8.4f} {res['ofi_spearman']:>+8.4f} "
                  f"{res['imb_pearson']:>+8.4f} {res['imb_spearman']:>+8.4f}")
            lines.append(
                f"| {w:.0f}s | {h:.0f}s | {res['n']} | "
                f"{res['ofi_pearson']:+.4f} | {res['ofi_spearman']:+.4f} | "
                f"{res['imb_pearson']:+.4f} | {res['imb_spearman']:+.4f} |")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
