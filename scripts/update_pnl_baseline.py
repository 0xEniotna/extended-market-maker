#!/usr/bin/env python3
"""Update PnL baselines after a bot restart.

Workflow:
    # After (re)starting a bot:
    python scripts/update_pnl_baseline.py --market DOT-USD --context "iter002 microprice live"

This will:
  1. Read the current `mmctl pnl <MARKET>` values
  2. Read the current run_id from the journal's run_start event
  3. Overwrite the baseline for that market in pnl_baselines.json
  4. Print the new baseline

Use --all to refresh every market in the baselines file (be careful — this
zeroes out the delta for every bot).

Usage:
    python scripts/update_pnl_baseline.py --market DOT-USD
    python scripts/update_pnl_baseline.py --market DOT-USD --context "iter002 launch"
    python scripts/update_pnl_baseline.py --all  # be careful
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASELINES_PATH = Path("/root/MM/data/pnl_baselines.json")


def _run_mmctl_pnl(market: str) -> dict[str, float]:
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
        for key in ("trade_pnl", "funding_fees", "close_fees"):
            prefix = f"{key}: "
            if line.startswith(prefix):
                rest = line[len(prefix):].split()[0]
                try:
                    out[key] = float(rest)
                except ValueError:
                    pass
        if line.startswith("total_pnl_including_open="):
            rest = line.split("=", 1)[1].split()[0]
            try:
                out["total_pnl_including_open"] = float(rest)
            except ValueError:
                pass
    return out


def _read_current_run_id(market: str) -> str | None:
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


def update_one(
    market: str, baselines: dict, context: str | None = None,
) -> dict:
    current = _run_mmctl_pnl(market)
    run_id = _read_current_run_id(market)
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    new_baseline = {
        "run_id": run_id,
        "baseline_total_pnl_including_open": current.get("total_pnl_including_open", 0.0),
        "baseline_trade_pnl": current.get("trade_pnl", 0.0),
        "baseline_funding_fees": current.get("funding_fees", 0.0),
        "baseline_close_fees": current.get("close_fees", 0.0),
        "captured_at_utc": now_iso,
        "snapshot_source": "update_pnl_baseline.py",
        "iter_context": context or baselines.get(market, {}).get("iter_context", "manual update"),
    }
    baselines[market] = new_baseline
    return new_baseline


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--market", help="Update a single market")
    g.add_argument("--all", action="store_true", help="Update every market in baselines file")
    ap.add_argument("--context", default=None, help="Free-text iter_context note")
    ap.add_argument(
        "--baselines",
        type=Path,
        default=BASELINES_PATH,
        help=f"Baselines file (default: {BASELINES_PATH})",
    )
    args = ap.parse_args()

    if not args.baselines.exists():
        print(f"ERROR: baselines file not found: {args.baselines}", file=sys.stderr)
        return 1

    data = json.loads(args.baselines.read_text())
    baselines = data["baselines"]

    if args.market:
        markets = [args.market]
    else:
        markets = list(baselines.keys())
        print(f"Refreshing baselines for ALL {len(markets)} markets. Press Ctrl-C to abort.")
        time.sleep(2)

    for m in markets:
        try:
            new = update_one(m, baselines, context=args.context)
            print(
                f"{m}: baseline_total={new['baseline_total_pnl_including_open']:.4f} "
                f"run_id={new['run_id']} ts={new['captured_at_utc']}"
            )
        except Exception as exc:
            print(f"{m}: ERROR {exc}", file=sys.stderr)

    args.baselines.write_text(json.dumps(data, indent=2) + "\n")
    print(f"\nWrote {args.baselines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
