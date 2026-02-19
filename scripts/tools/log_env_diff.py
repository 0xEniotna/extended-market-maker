#!/usr/bin/env python3
"""Append timestamped config diffs between two .env files to a JSONL changelog."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def read_env_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


def parse_env(lines: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value_clean = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        env[key.strip()] = value_clean
    return env


def build_rows(
    *,
    before_env: Dict[str, str],
    after_env: Dict[str, str],
    env_before_path: Path,
    env_after_path: Path,
    market: str,
    agent: str,
    source: str,
    mm_only: bool,
) -> List[Dict[str, Optional[str]]]:
    params = sorted(set(before_env.keys()) | set(after_env.keys()))
    rows: List[Dict[str, Optional[str]]] = []
    ts = time.time()

    for param in params:
        if mm_only and not param.startswith("MM_"):
            continue
        old = before_env.get(param)
        new = after_env.get(param)
        if old == new:
            continue
        rows.append({
            "ts": ts,
            "market": market,
            "agent": agent,
            "source": source,
            "iteration": None,
            "env_before": str(env_before_path),
            "env_after": str(env_after_path),
            "param": param,
            "old": old,
            "new": new,
            "reasons": ["manual_env_diff"],
            "stop_reason": None,
            "run_started": None,
            "analysis_file": None,
            "tuning_log_file": None,
            "trigger": None,
        })
    return rows


def append_rows(path: Path, rows: List[Dict[str, Optional[str]]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Log env config diffs to append-only changelog.")
    parser.add_argument("--before", required=True, help="Path to old env file.")
    parser.add_argument("--after", required=True, help="Path to new env file.")
    parser.add_argument(
        "--output",
        default="data/mm_audit/config_changelog.jsonl",
        help="JSONL changelog path (default: data/mm_audit/config_changelog.jsonl).",
    )
    parser.add_argument("--agent", default="manual", help="Agent or operator label.")
    parser.add_argument("--source", default="log_env_diff", help="Change source label.")
    parser.add_argument("--market", default=None, help="Market override.")
    parser.add_argument(
        "--all-keys",
        action="store_true",
        help="Include non-MM_* keys (default only logs MM_* keys).",
    )
    args = parser.parse_args()

    before_path = Path(args.before).resolve()
    after_path = Path(args.after).resolve()
    output_path = Path(args.output).resolve()
    if not before_path.exists():
        raise SystemExit(f"Before env file not found: {before_path}")
    if not after_path.exists():
        raise SystemExit(f"After env file not found: {after_path}")

    before_env = parse_env(read_env_lines(before_path))
    after_env = parse_env(read_env_lines(after_path))
    market = args.market or after_env.get("MM_MARKET_NAME") or before_env.get("MM_MARKET_NAME") or "unknown"
    rows = build_rows(
        before_env=before_env,
        after_env=after_env,
        env_before_path=before_path,
        env_after_path=after_path,
        market=market,
        agent=args.agent,
        source=args.source,
        mm_only=not args.all_keys,
    )
    append_rows(output_path, rows)
    print(f"changes_logged={len(rows)} output={output_path}")


if __name__ == "__main__":
    main()
