#!/usr/bin/env python3
"""Report since-restart PnL for each running market.

`mmctl pnl <MARKET>` returns lifetime cumulative PnL from the exchange,
which mixes prior runs and the current run. This script subtracts a
per-market baseline (snapshot at last restart) from the current value
to give the PnL attributable to the current run only.

The baseline lives at `data/pnl_baselines.json` and is updated by
`scripts/update_pnl_baseline.py` whenever a bot is started.

Usage:
    python scripts/pnl_since_restart.py                # all markets
    python scripts/pnl_since_restart.py --market DOT-USD
    python scripts/pnl_since_restart.py --json         # machine-readable
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

BASELINES_PATH = Path("/root/MM/data/pnl_baselines.json")


def _run_mmctl_pnl(market: str) -> dict[str, float]:
    """Parse `mmctl pnl <market>` output into numeric fields."""
    proc = subprocess.run(
        ["mmctl", "pnl", market],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mmctl pnl {market} failed: {proc.stderr.strip()}")

    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        # "trade_pnl: 102.015000 USD"
        for key in ("trade_pnl", "funding_fees", "close_fees"):
            prefix = f"{key}: "
            if line.startswith(prefix):
                rest = line[len(prefix):].split()[0]
                try:
                    out[key] = float(rest)
                except ValueError:
                    pass
        # "total_pnl_including_open=104.318805 USD"
        if line.startswith("total_pnl_including_open="):
            rest = line.split("=", 1)[1].split()[0]
            try:
                out["total_pnl_including_open"] = float(rest)
            except ValueError:
                pass
    return out


def _read_current_run_id(market: str) -> str | None:
    """Extract run_id from the market's latest journal's run_start event."""
    journal = Path(f"/root/MM/data/mm_journal/mm_{market}_latest.jsonl")
    if not journal.exists():
        return None
    try:
        with journal.open() as f:
            first = f.readline()
            if not first:
                return None
            event = json.loads(first)
            return event.get("run_id")
    except (OSError, json.JSONDecodeError):
        return None


def compute_delta(market: str, baseline: dict[str, Any]) -> dict[str, Any]:
    current = _run_mmctl_pnl(market)
    current_run_id = _read_current_run_id(market)

    baseline_run_id = baseline.get("run_id")
    run_id_match = (
        baseline_run_id is not None
        and current_run_id is not None
        and baseline_run_id == current_run_id
    )

    base_total = baseline.get("baseline_total_pnl_including_open", 0.0)
    cur_total = current.get("total_pnl_including_open", 0.0)
    delta_total = cur_total - base_total

    base_trade = baseline.get("baseline_trade_pnl", 0.0)
    cur_trade = current.get("trade_pnl", 0.0)
    delta_trade = cur_trade - base_trade

    base_funding = baseline.get("baseline_funding_fees", 0.0)
    cur_funding = current.get("funding_fees", 0.0)
    delta_funding = cur_funding - base_funding

    base_fees = baseline.get("baseline_close_fees", 0.0)
    cur_fees = current.get("close_fees", 0.0)
    delta_fees = cur_fees - base_fees

    return {
        "market": market,
        "baseline_run_id": baseline_run_id,
        "current_run_id": current_run_id,
        "run_id_match": run_id_match,
        "baseline_captured_at": baseline.get("captured_at_utc"),
        "delta_total_pnl": round(delta_total, 4),
        "delta_trade_pnl": round(delta_trade, 4),
        "delta_funding_fees": round(delta_funding, 4),
        "delta_close_fees": round(delta_fees, 4),
        "current_total_pnl": round(cur_total, 4),
        "baseline_total_pnl": round(base_total, 4),
    }


def _fmt_money(x: float) -> str:
    sign = "-" if x < 0 else " "
    return f"{sign}${abs(x):>8,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default=None, help="Single market (default: all in baselines file)")
    ap.add_argument(
        "--baselines",
        type=Path,
        default=BASELINES_PATH,
        help=f"Baselines file (default: {BASELINES_PATH})",
    )
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    if not args.baselines.exists():
        print(f"ERROR: baselines file not found: {args.baselines}", file=sys.stderr)
        return 1

    baselines = json.loads(args.baselines.read_text())["baselines"]
    if args.market:
        if args.market not in baselines:
            print(f"ERROR: no baseline for {args.market}", file=sys.stderr)
            return 1
        markets = [args.market]
    else:
        markets = list(baselines.keys())

    results = []
    for m in markets:
        try:
            results.append(compute_delta(m, baselines[m]))
        except Exception as exc:
            results.append({"market": m, "error": str(exc)})

    if args.json:
        print(json.dumps({"markets": results}, indent=2))
        return 0

    # Table output
    print(f"{'Market':<16} {'Δ Total':>11} {'Δ Trade':>11} {'Δ Funding':>11} {'Δ Fees':>11}  Match  Since")
    print("-" * 92)
    total_delta = 0.0
    for r in results:
        if "error" in r:
            print(f"{r['market']:<16}  ERROR: {r['error']}")
            continue
        total_delta += r["delta_total_pnl"]
        match = "✓" if r["run_id_match"] else "✗ STALE"
        since = r["baseline_captured_at"] or "-"
        print(
            f"{r['market']:<16} "
            f"{_fmt_money(r['delta_total_pnl'])} "
            f"{_fmt_money(r['delta_trade_pnl'])} "
            f"{_fmt_money(r['delta_funding_fees'])} "
            f"{_fmt_money(r['delta_close_fees'])}  "
            f"{match:<7} {since}"
        )
    print("-" * 92)
    print(f"{'FLEET TOTAL':<16} {_fmt_money(total_delta)}")
    print()
    print("Match column: ✓ means current run_id matches baseline run_id (delta is accurate).")
    print("              ✗ STALE means the bot has been restarted since the baseline was")
    print("              captured — run scripts/update_pnl_baseline.py --market <M> to refresh.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
