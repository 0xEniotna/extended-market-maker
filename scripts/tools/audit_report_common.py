#!/usr/bin/env python3
"""Shim — re-exports from market_maker.audit_report (canonical location)."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from market_maker.audit_report import (  # noqa: F401, E402
    collect_mid_series,
    compute_time_above_util,
    discover_market_journals,
    format_utc,
    is_block_reason,
    iso_utc,
    load_do_not_restart,
    load_market_env,
    load_recent_entries,
    median,
    mid_at_or_after,
    normalize_market_name,
    parse_env_map,
    pct,
    quantile_average,
    rolling_price_change,
    round_or_none,
    to_decimal,
    to_float,
    truncate_markdown,
    zero_crossings,
)
