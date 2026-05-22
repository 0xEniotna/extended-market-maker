#!/usr/bin/env python3
"""Reconcile bot-computed PnL against the Extended exchange UI's "realized".

WHY THIS EXISTS
---------------
The Extended UI shows "realized PnL" = cash booked from CLOSED positions
only. The bot's ``pnl_attribution`` events expose a ``total_usd`` that is
**closed + open mark-to-market** (it includes unrealized PnL on positions
still held). So the two won't match directly. The relationship is:

    exchange_realized  ≈  bot_total_usd  −  current_open_unrealized

Verified 2026-05-22: bot total_usd over 7d = +$25.39, open unrealized =
+$9.14, → realized estimate +$16.25 vs UI +$17.25 (≈$1 noise).

DO NOT use the bot's ``realized_pnl_usd`` field to compare with the UI —
it's an internal attribution component that offsets ``spread_capture_usd``
(it read −$174.94 over the same window while the true realized was +$17).
The correct quantity is ``total_usd − open_unrealized``.

WHAT THIS SCRIPT DOES
---------------------
1. Sums the bot's ``total_usd`` over a window (default 7 days), across ALL
   markets found in the journal dir, correctly handling:
     - journal rotation (``.jsonl``, ``.jsonl.1``, ...): pnl_attribution
       is cumulative *within* a journal, so we sum the per-journal delta
     - window boundary: for a journal that started before the cutoff, we
       subtract the last pre-cutoff event's value (baseline) from the last
       event's value
2. Optionally queries current open unrealized via ``mmctl pnl <market>``
   for each market that traded in the window, and subtracts it to produce
   the realized estimate to compare with the UI.

USAGE (run on the VPS, from /root/MM)
-------------------------------------
    python scripts/reconcile_pnl.py --days 7
    python scripts/reconcile_pnl.py --days 7 --no-unrealized   # journal-only
    python scripts/reconcile_pnl.py --days 7 --mmctl /root/MM/.venv/bin/mmctl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

_PA = "pnl_attribution"
_MARKET_RE = re.compile(r"(.+?)_\d{8}_\d{6}")
# Components we surface. total_usd is the headline; the rest are diagnostic.
_FIELDS = [
    "total_usd",
    "spread_capture_usd",
    "inventory_pnl_usd",
    "realized_pnl_usd",
    "fee_pnl_usd",
    "funding_pnl_usd",
    "total_volume_usd",
]


def _journals_by_market(journals_dir: Path) -> dict[str, list[str]]:
    by_market: dict[str, list[str]] = defaultdict(list)
    for j in glob(str(journals_dir / "mm_*.jsonl")):
        if os.path.islink(j):
            continue  # the *_latest.jsonl symlink points at a real file we already have
        base = os.path.basename(j)[3:]  # strip "mm_"
        m = _MARKET_RE.match(base)
        if m:
            by_market[m.group(1)].append(j)
    return by_market


def _window_delta(journal: str, cutoff: float) -> dict[str, float]:
    """Per-journal window delta of each cumulative field.

    pnl_attribution is cumulative within a journal. If the journal spans
    the cutoff, the window contribution is (last − last_pre_cutoff). If it
    started after the cutoff, it's the full last value.
    """
    first_pre = None  # last event strictly before cutoff (baseline)
    last_in = None  # last event at or after cutoff
    started_in_window = True
    with open(journal) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != _PA:
                continue
            ts = r.get("ts", 0)
            if ts < cutoff:
                first_pre = r
                started_in_window = False
            else:
                last_in = r
    out: dict[str, float] = dict.fromkeys(_FIELDS, 0.0)
    if last_in is None:
        return out
    for fld in _FIELDS:
        last_v = float(last_in.get(fld, 0) or 0)
        if not started_in_window and first_pre is not None:
            base_v = float(first_pre.get(fld, 0) or 0)
            out[fld] = last_v - base_v
        else:
            out[fld] = last_v
    return out


def _fills_in_window(journal: str, cutoff: float) -> tuple[int, int, float]:
    """(resting_fills, taker_fills, fee_sum) for the window from raw fills."""
    resting = taker = 0
    fee = 0.0
    with open(journal) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "fill" or r.get("ts", 0) < cutoff:
                continue
            if r.get("is_taker"):
                taker += 1
            else:
                resting += 1
            try:
                fee += float(r.get("fee", 0) or 0)
            except (TypeError, ValueError):
                pass
    return resting, taker, fee


def _query_unrealized(mmctl: str, market: str) -> float | None:
    """Shell out to ``mmctl pnl <market>`` and parse unrealized_component."""
    try:
        res = subprocess.run(
            [mmctl, "pnl", market],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    for line in res.stdout.splitlines():
        if "unrealized_component:" in line:
            m = re.search(r"-?\d+\.\d+", line)
            if m:
                return float(m.group(0))
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=float, default=7.0, help="Lookback window in days.")
    p.add_argument(
        "--journals-dir", type=Path,
        default=Path("/root/MM/data/mm_journal"),
        help="Directory of mm_*.jsonl journals.",
    )
    p.add_argument(
        "--mmctl", default="/root/MM/.venv/bin/mmctl",
        help="Path to mmctl for the unrealized query.",
    )
    p.add_argument(
        "--no-unrealized", action="store_true",
        help="Skip the mmctl unrealized query (journal-only; prints gross total).",
    )
    args = p.parse_args()

    now = datetime.now(timezone.utc).timestamp()
    cutoff = now - args.days * 24 * 3600
    print(
        f"Window: last {args.days:g}d "
        f"({datetime.fromtimestamp(cutoff, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} "
        f"-> now)\n"
    )

    by_market = _journals_by_market(args.journals_dir)
    rows: list[tuple[str, dict[str, float], int, int, float]] = []
    grand: dict[str, float] = defaultdict(float)

    for market, journals in sorted(by_market.items()):
        agg: dict[str, float] = defaultdict(float)
        resting = taker = 0
        fee = 0.0
        for j in sorted(journals):
            d = _window_delta(j, cutoff)
            for k, v in d.items():
                agg[k] += v
            rf, tf, fe = _fills_in_window(j, cutoff)
            resting += rf
            taker += tf
            fee += fe
        if abs(agg.get("total_usd", 0)) < 0.01 and resting == 0:
            continue  # market had no activity in the window
        rows.append((market, agg, resting, taker, fee))
        for k, v in agg.items():
            grand[k] += v

    # --- Journal-based table ---
    print(f"{'market':<24} {'total_usd':>10} {'spread':>9} {'invent':>9} "
          f"{'volume':>11} {'fills':>6} {'fee':>6}")
    print("-" * 90)
    for market, agg, resting, taker, fee in sorted(rows, key=lambda x: x[1]["total_usd"]):
        print(f"{market:<24} {agg['total_usd']:>+10.2f} "
              f"{agg['spread_capture_usd']:>+9.2f} {agg['inventory_pnl_usd']:>+9.2f} "
              f"{agg['total_volume_usd']:>11,.0f} {resting:>6} {fee:>6.2f}")
    print("-" * 90)
    print(f"{'TOTAL':<24} {grand['total_usd']:>+10.2f} "
          f"{grand['spread_capture_usd']:>+9.2f} {grand['inventory_pnl_usd']:>+9.2f} "
          f"{grand['total_volume_usd']:>11,.0f}")

    gross_total = grand["total_usd"]

    if args.no_unrealized:
        print(f"\nBot total_usd (gross, incl open mark-to-market): ${gross_total:+.2f}")
        print("Run without --no-unrealized to subtract open unrealized and "
              "get the realized estimate to compare with the Extended UI.")
        return 0

    # --- Unrealized adjustment via mmctl ---
    print("\nQuerying current open unrealized via mmctl ...")
    total_unrealized = 0.0
    market_names = {market for market, *_ in rows}
    # also include current fleet markets even if no window activity captured
    for market in sorted(market_names):
        # market dir name uses underscores already matching the exchange symbol
        mname = market if market.endswith("-USD") else f"{market}-USD"
        u = _query_unrealized(args.mmctl, mname)
        if u is None:
            print(f"  {mname:<24} unrealized = (query failed, skipped)")
            continue
        if abs(u) > 0.001:
            print(f"  {mname:<24} unrealized = {u:+.2f}")
        total_unrealized += u

    realized_est = gross_total - total_unrealized
    print("\n=== RECONCILIATION ===")
    print(f"  Bot total_usd (gross):        ${gross_total:+.2f}")
    print(f"  − Open unrealized:            ${-total_unrealized:+.2f}")
    print(f"  = Realized estimate:          ${realized_est:+.2f}")
    print("  Compare this to the Extended UI 'realized' figure.")
    print("  (Residual of ~$1-2 is expected: window-boundary alignment, "
          "unrealized-at-start ≠ 0, killed-market final closes.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
