#!/usr/bin/env python3
"""Surface the config history of a market by parsing every ``run_start``
event in its journals.

The MM bot writes a ``run_start`` event at the top of each journal file
containing a snapshot of the live settings (credentials redacted). This
script collects all run_starts for a market across one or more journal
directories, sorts them chronologically, and renders:

  1. A timeline of (ts, run_id, journal_file, key knobs)
  2. A diff between consecutive runs (which knobs changed)
  3. Optionally a JSON dump for downstream tooling

Usage:
    python scripts/journal_config_history.py --market MU_24_5-USD \\
        --journals-dir /root/MM/data/mm_journal \\
        --journals-dir /root/MM-funding-aware/data/mm_journal \\
        --out docs/config_history_MU.md

Without --market, the script processes every distinct market it finds.

Why this exists:
The MM operator (human or AI) needs to be able to look at a past
journal slice and know exactly what configuration produced it. Without
this, .env files (which are mutable) get out of sync with what was
actually live, and post-hoc analysis becomes unreliable.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Knobs we surface in the markdown timeline. Order matters (left-to-right
# in the rendered table). The rest of the config dict is still kept for
# diffing.
SURFACED_KNOBS = [
    "market_profile",
    "num_price_levels",
    "offset_mode",
    "spread_multiplier",
    "min_offset_bps",
    "max_offset_bps",
    "order_size_multiplier",
    "max_position_size",
    "max_position_notional_usd",
    "max_order_notional_usd",
    "inventory_skew_factor",
    "skew_max_bps",
    "min_acceptable_markout_bps",
    "min_reprice_interval_s",
    "drawdown_stop_enabled",
    "drawdown_stop_pct_of_max_notional",
    "funding_bias_enabled",
    "funding_aware_enabled",
    "trend_enabled",
    "deadman_enabled",
]


def _find_run_starts(journal_path: Path) -> list[dict]:
    """Return every run_start record in a journal (usually 0 or 1)."""
    out: list[dict] = []
    with journal_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "run_start":
                out.append(r)
    return out


def _collect_runs(journals_dirs: list[Path], market: str | None) -> dict[str, list[dict]]:
    """Walk every journal file under the given dirs and group by market."""
    by_market: dict[str, list[dict]] = defaultdict(list)
    for jdir in journals_dirs:
        for p in glob.glob(str(jdir / "mm_*.jsonl")):
            if not os.path.isfile(p):
                continue  # skip symlinks (we'll pick up the underlying file)
            for r in _find_run_starts(Path(p)):
                m = r.get("market", "?")
                if market and m != market:
                    continue
                r["_journal"] = os.path.basename(p)
                r["_journal_full"] = p
                by_market[m].append(r)
    for m in by_market:
        by_market[m].sort(key=lambda x: x.get("ts", 0))
    return by_market


def _config_diff(prev: dict, curr: dict) -> dict[str, tuple]:
    """Return {knob: (old, new)} for differing values (skip credentials)."""
    diff: dict[str, tuple] = {}
    redacted_marker = "***redacted***"
    all_keys = set(prev.keys()) | set(curr.keys())
    for k in sorted(all_keys):
        a = prev.get(k)
        b = curr.get(k)
        if a == redacted_marker or b == redacted_marker:
            continue
        if a != b:
            diff[k] = (a, b)
    return diff


def render(market: str, runs: list[dict]) -> str:
    lines = [
        f"# Config History — {market}",
        "",
        f"Reconstructed from `run_start` events. **{len(runs)} runs** found.",
        "",
        "Each row shows the config that was live at the moment the run "
        "started. The MM bot does NOT re-emit a `run_start` on hot reload "
        "(SIGHUP), so a long-running process with reloaded config is NOT "
        "represented here — assume the values held until the next run_start "
        "in the same journal.",
        "",
        "## Timeline (key knobs only — full config in raw events)",
        "",
    ]

    header = ["ts (UTC)", "run_id", "journal"] + SURFACED_KNOBS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in runs:
        ts = r.get("ts", 0)
        run_id_short = (r.get("run_id") or "")[:8]
        journal = r.get("_journal", "?")
        cfg = r.get("config") or {}
        row = [
            datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            run_id_short,
            f"`{journal}`",
        ]
        for k in SURFACED_KNOBS:
            v = cfg.get(k, "—")
            if isinstance(v, bool):
                v = "true" if v else "false"
            row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Diffs between consecutive runs",
        "",
        "Only diffs are shown (constant knobs omitted). Useful to verify a "
        "deliberate config change took effect, or to spot an unintended one.",
        "",
    ]

    for i in range(1, len(runs)):
        prev_cfg = runs[i - 1].get("config") or {}
        curr_cfg = runs[i].get("config") or {}
        diff = _config_diff(prev_cfg, curr_cfg)

        prev_ts = datetime.fromtimestamp(runs[i - 1]["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        curr_ts = datetime.fromtimestamp(runs[i]["ts"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        prev_run_id = (runs[i - 1].get("run_id") or "")[:8]
        curr_run_id = (runs[i].get("run_id") or "")[:8]

        lines.append(f"### {prev_ts} ({prev_run_id}) → {curr_ts} ({curr_run_id})")
        if not diff:
            lines.append("")
            lines.append("_No config changes (identical settings)._")
            lines.append("")
            continue
        lines.append("")
        lines.append("| knob | before | after |")
        lines.append("|---|---|---|")
        for k, (a, b) in diff.items():
            lines.append(f"| `{k}` | `{a}` | `{b}` |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", help="Filter to a single market (e.g. MU_24_5-USD)")
    p.add_argument(
        "--journals-dir", type=Path, action="append", required=True,
        help="Directory containing mm_*.jsonl files. Repeat for multiple dirs.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="If set, write the report to this path. Otherwise print to stdout.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Also print a JSON dump (run_id → config) to stdout.",
    )
    args = p.parse_args()

    by_market = _collect_runs(args.journals_dir, args.market)

    if not by_market:
        print(f"No run_start events found "
              f"({'for ' + args.market if args.market else ''}).")
        return 1

    if args.market:
        markets = [args.market]
    else:
        markets = sorted(by_market.keys())

    full_md = []
    for m in markets:
        runs = by_market.get(m, [])
        if not runs:
            continue
        full_md.append(render(m, runs))

    out_text = "\n\n---\n\n".join(full_md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out_text)
        print(f"Wrote: {args.out}  ({len(runs)} runs across {len(markets)} market(s))")
    else:
        print(out_text)

    if args.json:
        # Lightweight JSON suitable for diffing or piping
        dump = {
            m: [
                {
                    "ts": r.get("ts"),
                    "run_id": r.get("run_id"),
                    "journal": r.get("_journal"),
                    "config": {k: v for k, v in (r.get("config") or {}).items()
                               if v != "***redacted***"},
                }
                for r in runs
            ]
            for m, runs in by_market.items()
        }
        print("\n--- JSON ---")
        print(json.dumps(dump, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
