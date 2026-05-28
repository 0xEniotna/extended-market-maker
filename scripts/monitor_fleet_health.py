#!/usr/bin/env python3
"""Fleet-health monitor — flags the kill-pattern triggers on LIVE markets.

WHY: the discovery screen (`scripts/tools/find_mm_markets.py`) is point-in-time
and CANNOT predict the post-launch *wake-up collapse* that killed MU and EDGE
(they screened fine while wide, then their spread collapsed into the toxic band
once they got active). This monitor watches RUNNING markets' recent activity and
flags the three kill-pattern triggers (see project_fleet.md "KILL PATTERN"):

  1. TOXIC-BAND — live median spread <= --toxic-bps (default 8): spread
     collapsed into the toxic zone where our slow MM gets picked off.
  2. ADVERSE   — MM +5s markout <= --adverse-bps (default -2) on >= --min-fills:
     we're being adversely selected.
  3. ONE-SIDED — fill balance >= --oneside-pct (default 80): accumulating a
     directional inventory (the NEAR / one-sided-bleed mode).

Verdict per market: OK / WATCH (1 trigger) / KILL? (TOXIC-BAND, or >=2 triggers).
Read-only — it never trades or stops anything; it just surfaces the triggers so
the operator (or a cron) can act. Pairs with the hardened discovery screen.

Usage (on the VPS):
  PYTHONPATH=src python scripts/monitor_fleet_health.py --window-min 180
  PYTHONPATH=src python scripts/monitor_fleet_health.py --json
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import time

_MARKET_RE = re.compile(r"(.+?)_\d{8}_\d{6}")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _market_from_journal(path: str):
    base = os.path.basename(path)
    if not base.startswith("mm_"):
        return None
    m = _MARKET_RE.match(base[3:])  # strip "mm_"
    return m.group(1) if m else None


def analyze(journal: str, window_start: float, horizon_s: float) -> dict:
    """Recent-window stats for one market journal: spread, markout, one-sidedness."""
    mids = []          # (ts, mid) from book_change
    spreads = []       # bps
    fills = []         # (ts, fill_mid, is_buy)
    buys = sells = 0
    with open(journal) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = r.get("ts")
            t = r.get("type")
            if ts is None or ts < window_start:
                continue
            if t == "book_change":
                b, a = _f(r.get("bid")), _f(r.get("ask"))
                if b and a and a > b > 0:
                    mid = (a + b) / 2
                    mids.append((ts, mid))
                    spreads.append((a - b) / mid * 1e4)
            elif t == "fill":
                s = str(r.get("side", ""))
                is_buy = s.endswith("BUY") or s == "BUY"
                buys += int(is_buy)
                sells += int(not is_buy)
                ms = r.get("market_snapshot") or {}
                fb, fa = _f(ms.get("best_bid")), _f(ms.get("best_ask"))
                fm = (fa + fb) / 2 if fb and fa and fa > fb > 0 else None
                fills.append((ts, fm, is_buy))

    # MM-perspective +Hs markout (positive = good for MM; negative = picked off)
    mids.sort()
    mts = [m[0] for m in mids]
    mvals = [m[1] for m in mids]
    markouts = []
    for ft, fm, is_buy in fills:
        if fm is None or fm <= 0:
            continue
        i = bisect.bisect_left(mts, ft + horizon_s)
        if i < len(mvals):
            raw = (mvals[i] - fm) / fm * 1e4
            markouts.append(raw if is_buy else -raw)

    n = buys + sells
    return {
        "n_book": len(spreads),
        "med_spread_bps": (sorted(spreads)[len(spreads) // 2] if spreads else None),
        "fills": n,
        "oneside_pct": (max(buys, sells) / n * 100 if n else None),
        "markout_n": len(markouts),
        "mean_markout_5s": (sum(markouts) / len(markouts) if markouts else None),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--journals-dir", default="/root/MM/data/mm_journal")
    p.add_argument("--window-min", type=float, default=180.0,
                   help="Look-back window in minutes (also defines 'running' = journal modified within it).")
    p.add_argument("--horizon-s", type=float, default=5.0)
    p.add_argument("--toxic-bps", type=float, default=8.0)
    p.add_argument("--adverse-bps", type=float, default=-2.0)
    p.add_argument("--oneside-pct", type=float, default=80.0)
    p.add_argument("--min-fills", type=int, default=10)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    now = time.time()
    window_start = now - args.window_min * 60.0

    # Latest non-symlink journal per market, modified within the window (= active).
    latest: dict[str, str] = {}
    for jp in glob.glob(os.path.join(args.journals_dir, "mm_*.jsonl")):
        if os.path.islink(jp) or os.path.getmtime(jp) < window_start:
            continue
        mk = _market_from_journal(jp)
        if not mk:
            continue
        if mk not in latest or os.path.getmtime(jp) > os.path.getmtime(latest[mk]):
            latest[mk] = jp

    rows = []
    for mk, jp in sorted(latest.items()):
        r = analyze(jp, window_start, args.horizon_s)
        r["market"] = mk
        trig = []
        if r["med_spread_bps"] is not None and r["med_spread_bps"] <= args.toxic_bps:
            trig.append("TOXIC-BAND")
        if (r["mean_markout_5s"] is not None and r["markout_n"] >= args.min_fills
                and r["mean_markout_5s"] <= args.adverse_bps):
            trig.append("ADVERSE")
        if (r["oneside_pct"] is not None and r["fills"] >= args.min_fills
                and r["oneside_pct"] >= args.oneside_pct):
            trig.append("ONE-SIDED")
        r["triggers"] = trig
        r["verdict"] = "KILL?" if ("TOXIC-BAND" in trig or len(trig) >= 2) else ("WATCH" if trig else "OK")
        rows.append(r)

    if args.json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    print(f"Fleet health — window {args.window_min:.0f} min | triggers: "
          f"spread<={args.toxic_bps} / +5s_markout<={args.adverse_bps} / "
          f"1-sided>={args.oneside_pct}% (>= {args.min_fills} fills)")
    print(f"{'market':<22}{'spread':>9}{'+5s_mkout':>11}{'fills':>7}{'1side%':>8}  verdict / triggers")
    print("-" * 78)
    order = {"KILL?": 0, "WATCH": 1, "OK": 2}
    for r in sorted(rows, key=lambda x: (order[x["verdict"]], x["market"])):
        sp = f"{r['med_spread_bps']:.1f}" if r["med_spread_bps"] is not None else "-"
        mko = f"{r['mean_markout_5s']:+.1f}" if r["mean_markout_5s"] is not None else "-"
        osd = f"{r['oneside_pct']:.0f}" if r["oneside_pct"] is not None else "-"
        print(f"{r['market']:<22}{sp:>9}{mko:>11}{r['fills']:>7}{osd:>8}  "
              f"{r['verdict']}{(' ' + ','.join(r['triggers'])) if r['triggers'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
