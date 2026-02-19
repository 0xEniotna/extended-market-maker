#!/usr/bin/env python3
"""Export inventory/position time-series artifacts from MM journals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _d(v: Any) -> Optional[Decimal]:
    if v is None:
        return None
    try:
        return Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug(value: str) -> str:
    out = "".join(ch if ch.isalnum() else "_" for ch in value.strip())
    out = "_".join(part for part in out.split("_") if part)
    return out or "unknown"


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, dict):
        return {k: _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_to_jsonable(x) for x in v]
    return v


def _percentile(values: List[Decimal], p: float) -> Optional[Decimal]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
    return vals[idx]


def _iter_journals(target: Path) -> List[Path]:
    if target.is_file():
        return [target]
    files = [p for p in target.rglob("mm_*.jsonl") if "mm_tuning_log_" not in p.name]
    return sorted(files, key=lambda p: (p.stat().st_mtime, str(p)))


def load_position_rows(
    *,
    target: Path,
    market: Optional[str],
    start_ts: Optional[float],
    end_ts: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Decimal]]]:
    rows: List[Dict[str, Any]] = []
    inferred: Dict[str, Optional[Decimal]] = {
        "max_position_size": None,
        "inventory_warn_pct": None,
        "inventory_critical_pct": None,
        "inventory_hard_pct": None,
    }

    for path in _iter_journals(target):
        with path.open() as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                ts_raw = event.get("ts")
                if ts_raw is None:
                    continue
                ts = float(ts_raw)
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts:
                    continue

                event_market = str(event.get("market") or "")
                if market and event_market and event_market != market:
                    continue

                if event.get("type") == "run_start":
                    cfg = event.get("config", {})
                    if isinstance(cfg, dict):
                        for key in inferred.keys():
                            val = _d(cfg.get(key))
                            if val is not None:
                                inferred[key] = val

                pos = _d(event.get("position"))
                if pos is None:
                    continue

                abs_pos = abs(pos)
                row: Dict[str, Any] = {
                    "ts": ts,
                    "ts_iso": _iso(ts),
                    "market": event_market or (market or "unknown"),
                    "run_id": event.get("run_id"),
                    "source_event": event.get("type"),
                    "position": pos,
                    "abs_position": abs_pos,
                    "best_bid": _d(event.get("best_bid")),
                    "best_ask": _d(event.get("best_ask")),
                    "spread_bps": _d(event.get("spread_bps")),
                    "circuit_open": event.get("circuit_open"),
                    "journal_file": str(path),
                }
                rows.append(row)

    rows.sort(key=lambda r: r["ts"])
    return rows, inferred


def attach_utilization(rows: List[Dict[str, Any]], max_position_size: Optional[Decimal]) -> None:
    for row in rows:
        if max_position_size is None or max_position_size <= 0:
            row["utilization_pct"] = None
            continue
        util = (row["abs_position"] / max_position_size) * Decimal("100")
        row["utilization_pct"] = util


def bucket_rows(rows: List[Dict[str, Any]], bucket_seconds: int) -> List[Dict[str, Any]]:
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be > 0")
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        bucket_ts = int(math.floor(row["ts"] / bucket_seconds) * bucket_seconds)
        buckets.setdefault(bucket_ts, []).append(row)

    out: List[Dict[str, Any]] = []
    for bucket_ts in sorted(buckets.keys()):
        samples = buckets[bucket_ts]
        pos_open = samples[0]["position"]
        pos_close = samples[-1]["position"]
        abs_max = max(s["abs_position"] for s in samples)
        util_values = [s["utilization_pct"] for s in samples if s.get("utilization_pct") is not None]
        util_close = samples[-1].get("utilization_pct")
        util_max = max(util_values) if util_values else None
        out.append({
            "bucket_ts": bucket_ts,
            "bucket_iso": _iso(float(bucket_ts)),
            "market": samples[-1]["market"],
            "samples": len(samples),
            "position_open": pos_open,
            "position_close": pos_close,
            "abs_position_max": abs_max,
            "utilization_close_pct": util_close,
            "utilization_max_pct": util_max,
            "source_event_last": samples[-1].get("source_event"),
        })
    return out


def summarize_rows(
    rows: List[Dict[str, Any]],
    *,
    max_position_size: Optional[Decimal],
    warn_pct: Decimal,
    critical_pct: Decimal,
    hard_pct: Decimal,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "observations": len(rows),
        "start_ts": None,
        "end_ts": None,
        "start_iso": None,
        "end_iso": None,
        "duration_s": 0.0,
        "max_position_size": max_position_size,
        "warn_pct": warn_pct,
        "critical_pct": critical_pct,
        "hard_pct": hard_pct,
        "max_abs_position": None,
        "p95_abs_position": None,
        "p95_utilization_pct": None,
        "final_position": None,
        "time_nonflat_s": 0.0,
        "nonflat_ratio": None,
        "time_above_warn_s": None,
        "time_above_critical_s": None,
        "time_above_hard_s": None,
        "longest_drift_s": 0.0,
    }
    if not rows:
        return summary

    abs_positions = [r["abs_position"] for r in rows]
    util_values = [r["utilization_pct"] for r in rows if r.get("utilization_pct") is not None]
    start_ts = float(rows[0]["ts"])
    end_ts = float(rows[-1]["ts"])
    duration = max(0.0, end_ts - start_ts)
    summary.update({
        "start_ts": start_ts,
        "end_ts": end_ts,
        "start_iso": _iso(start_ts),
        "end_iso": _iso(end_ts),
        "duration_s": duration,
        "max_abs_position": max(abs_positions),
        "p95_abs_position": _percentile(abs_positions, 0.95),
        "p95_utilization_pct": _percentile(util_values, 0.95) if util_values else None,
        "final_position": rows[-1]["position"],
    })

    warn_abs: Optional[Decimal] = None
    critical_abs: Optional[Decimal] = None
    hard_abs: Optional[Decimal] = None
    if max_position_size is not None and max_position_size > 0:
        warn_abs = max_position_size * warn_pct
        critical_abs = max_position_size * critical_pct
        hard_abs = max_position_size * hard_pct
        summary["time_above_warn_s"] = 0.0
        summary["time_above_critical_s"] = 0.0
        summary["time_above_hard_s"] = 0.0

    longest_drift = 0.0
    current_drift = 0.0
    current_sign = 0
    time_nonflat = 0.0

    for idx in range(len(rows) - 1):
        cur = rows[idx]
        nxt = rows[idx + 1]
        dt = max(0.0, float(nxt["ts"]) - float(cur["ts"]))
        pos = cur["position"]
        abs_pos = cur["abs_position"]

        sign = 1 if pos > 0 else -1 if pos < 0 else 0
        if sign == 0:
            longest_drift = max(longest_drift, current_drift)
            current_drift = 0.0
            current_sign = 0
        elif sign == current_sign:
            current_drift += dt
        else:
            longest_drift = max(longest_drift, current_drift)
            current_sign = sign
            current_drift = dt

        if abs_pos > 0:
            time_nonflat += dt

        if warn_abs is not None:
            if abs_pos >= warn_abs:
                summary["time_above_warn_s"] += dt
            if abs_pos >= critical_abs:
                summary["time_above_critical_s"] += dt
            if abs_pos >= hard_abs:
                summary["time_above_hard_s"] += dt

    longest_drift = max(longest_drift, current_drift)
    summary["time_nonflat_s"] = time_nonflat
    summary["longest_drift_s"] = longest_drift
    if duration > 0:
        summary["nonflat_ratio"] = time_nonflat / duration
    return summary


def write_events_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(_to_jsonable(row)) + "\n")


def write_bucket_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket_ts",
        "bucket_iso",
        "market",
        "samples",
        "position_open",
        "position_close",
        "abs_position_max",
        "utilization_close_pct",
        "utilization_max_pct",
        "source_event_last",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _to_jsonable(row.get(k)) for k in fieldnames})


def write_summary_json(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(_to_jsonable(summary), f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export inventory time-series artifacts from MM journals.")
    parser.add_argument(
        "target",
        nargs="?",
        default="data/mm_journal",
        help="Journal file or directory (default: data/mm_journal).",
    )
    parser.add_argument("--market", default=None, help="Market filter (e.g., AMZN_24_5-USD).")
    parser.add_argument("--start-ts", type=float, default=None, help="Start timestamp (unix seconds).")
    parser.add_argument("--end-ts", type=float, default=None, help="End timestamp (unix seconds).")
    parser.add_argument("--max-position-size", default=None, help="Override max position size.")
    parser.add_argument("--warn-pct", default=None, help="Override warn threshold ratio (default from run_start or 0.5).")
    parser.add_argument(
        "--critical-pct",
        default=None,
        help="Override critical threshold ratio (default from run_start or 0.8).",
    )
    parser.add_argument("--hard-pct", default=None, help="Override hard threshold ratio (default from run_start or 0.95).")
    parser.add_argument("--bucket-seconds", type=int, default=60, help="Bucket size in seconds (default: 60).")
    parser.add_argument(
        "--output-dir",
        default="data/mm_audit/inventory",
        help="Output directory (default: data/mm_audit/inventory).",
    )
    args = parser.parse_args()

    target = Path(args.target).resolve()
    if not target.exists():
        raise SystemExit(f"Target not found: {target}")
    output_dir = Path(args.output_dir).resolve()

    rows, inferred = load_position_rows(
        target=target,
        market=args.market,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
    )

    if args.max_position_size is not None:
        max_pos = _d(args.max_position_size)
    else:
        max_pos = inferred.get("max_position_size")

    warn_pct = _d(args.warn_pct) or inferred.get("inventory_warn_pct") or Decimal("0.5")
    critical_pct = _d(args.critical_pct) or inferred.get("inventory_critical_pct") or Decimal("0.8")
    hard_pct = _d(args.hard_pct) or inferred.get("inventory_hard_pct") or Decimal("0.95")
    attach_utilization(rows, max_pos)

    buckets = bucket_rows(rows, args.bucket_seconds)
    summary = summarize_rows(
        rows,
        max_position_size=max_pos,
        warn_pct=warn_pct,
        critical_pct=critical_pct,
        hard_pct=hard_pct,
    )
    summary["market"] = args.market or (rows[-1]["market"] if rows else "unknown")
    summary["bucket_seconds"] = args.bucket_seconds

    prefix = _slug(str(summary["market"]))
    events_path = output_dir / f"{prefix}_events.jsonl"
    bucket_path = output_dir / f"{prefix}_{args.bucket_seconds}s.csv"
    summary_path = output_dir / f"{prefix}_summary.json"
    write_events_jsonl(events_path, rows)
    write_bucket_csv(bucket_path, buckets)
    write_summary_json(summary_path, summary)
    print(f"events={events_path}")
    print(f"buckets={bucket_path}")
    print(f"summary={summary_path}")
    print(f"observations={len(rows)}")


if __name__ == "__main__":
    main()
