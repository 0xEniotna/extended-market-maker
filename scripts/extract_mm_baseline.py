#!/usr/bin/env python3
"""Compute schema-v2 crypto baseline metrics from MM journals."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


_ANALYSIS_FILL_RATE_RE = re.compile(r"^## Fill Rate: ([0-9.]+)%  \(([0-9]+) fills / ([0-9]+) orders\)")
_ANALYSIS_MO5_RE = re.compile(r"^\s*\+5s: avg=([-0-9.]+)")
_ANALYSIS_CANCEL_RE = re.compile(r"^\s*Cancellations: [0-9]+ \(([0-9.]+)%")


def _is_crypto_market(market: str) -> bool:
    if "_24_5-" in market:
        return False
    if market in {"SPX500m-USD", "XAG-USD"}:
        return False
    return True


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _run_start(path: Path) -> Optional[dict]:
    for event in _iter_jsonl(path):
        if event.get("type") == "run_start":
            return event
    return None


def _parse_analysis(path: Path) -> tuple[Optional[int], Optional[int], Optional[float], Optional[float]]:
    fills = None
    orders = None
    mo5 = None
    canc_pct = None
    if not path.exists():
        return fills, orders, mo5, canc_pct

    with path.open() as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if fills is None:
                m = _ANALYSIS_FILL_RATE_RE.match(line)
                if m:
                    fills = int(m.group(2))
                    orders = int(m.group(3))
                    continue
            if mo5 is None:
                m = _ANALYSIS_MO5_RE.match(line)
                if m:
                    mo5 = float(m.group(1))
                    continue
            if canc_pct is None:
                m = _ANALYSIS_CANCEL_RE.match(line)
                if m:
                    canc_pct = float(m.group(1))
                    continue
    return fills, orders, mo5, canc_pct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal_dir",
        nargs="?",
        default="data/mm_journal",
        help="Directory containing mm_*.jsonl journals",
    )
    args = parser.parse_args()

    root = Path(args.journal_dir)
    files = sorted(root.glob("mm_*.jsonl"))

    selected: list[Path] = []
    for file in files:
        if "mm_tuning_log_" in file.name:
            continue
        rs = _run_start(file)
        if rs is None:
            continue
        if int(rs.get("schema_version", 0)) != 2:
            continue
        market = str(rs.get("market") or rs.get("config", {}).get("market_name") or "")
        if not market or not _is_crypto_market(market):
            continue
        selected.append(file)

    fills_total = 0
    orders_total = 0
    mo5_weighted_num = 0.0
    mo5_weighted_den = 0
    canc_weighted_num = 0.0
    canc_weighted_den = 0
    reasons = Counter()

    for journal in selected:
        analysis = journal.with_suffix(".analysis.txt")
        fills, orders, mo5, canc_pct = _parse_analysis(analysis)
        if fills is not None and orders is not None:
            fills_total += fills
            orders_total += orders
        if fills and mo5 is not None:
            mo5_weighted_num += mo5 * fills
            mo5_weighted_den += fills
        if orders and canc_pct is not None:
            canc_weighted_num += canc_pct * orders
            canc_weighted_den += orders

        for event in _iter_jsonl(journal):
            if event.get("type") != "reprice_decision":
                continue
            reason = str(event.get("reason") or "unknown")
            reasons[reason] += 1

    print(f"schema2_crypto_runs={len(selected)}")
    if orders_total > 0:
        print(f"weighted_fill_rate_pct={100.0 * fills_total / orders_total:.2f}")
    else:
        print("weighted_fill_rate_pct=n/a")

    if mo5_weighted_den > 0:
        print(f"weighted_mo5_bps={mo5_weighted_num / mo5_weighted_den:.2f}")
    else:
        print("weighted_mo5_bps=n/a")

    if canc_weighted_den > 0:
        print(f"weighted_cancellation_pct={canc_weighted_num / canc_weighted_den:.2f}")
    else:
        print("weighted_cancellation_pct=n/a")

    total_reprice = sum(reasons.values())
    print(f"reprice_decisions={total_reprice}")
    if total_reprice > 0:
        print("reprice_reason_distribution:")
        for reason, count in reasons.most_common():
            pct = 100.0 * count / total_reprice
            print(f"  {reason}: {count} ({pct:.2f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
