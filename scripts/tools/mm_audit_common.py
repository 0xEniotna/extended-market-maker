#!/usr/bin/env python3
"""Shim — re-exports from market_maker.audit_common (canonical location)."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from market_maker.audit_common import (  # noqa: F401, E402
    append_jsonl,
    discover_recent_markets_from_journals,
    extract_last_json_object,
    find_latest_journal,
    iso_utc,
    load_market_jobs,
    load_policy,
    now_ts,
    parse_env,
    read_env_lines,
    read_json,
    read_jsonl,
    resolve_env_path,
    safe_decimal,
    slugify,
    to_jsonable,
    update_env_lines,
    write_json,
)
