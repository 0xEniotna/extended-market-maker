#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mm_audit_common import parse_env, read_env_lines, safe_decimal  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENV_MAP: Dict[str, str] = {
    "ETH-USD": ".env.eth",
    "XPT-USD": ".env.xpt",
    "XNG-USD": ".env.xng",
    "XCU-USD": ".env.xcu",
    "NEAR-USD": ".env.near",
    "MON-USD": ".env.mon",
    "SPX500M-USD": ".env.spx500m",
    "SPX500m-USD": ".env.spx500m",
    "GOOG_24_5-USD": ".env.goog",
}

_TS_SUFFIX_RE = re.compile(r"^(.+)_\d{8}_\d{6}$")


def normalize_market_name(value: str) -> str:
    return value.strip().upper()


def format_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = datetime.now(tz=timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def iso_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = datetime.now(tz=timezone.utc).timestamp()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_float(value: Any) -> Optional[float]:
    dec = safe_decimal(value)
    if dec is None:
        return None
    return float(dec)


def to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    dec = safe_decimal(value)
    return dec if dec is not None else default


def pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def truncate_markdown(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    suffix = "\n...(truncated)"
    while lines and len("\n".join(lines)) + len(suffix) > limit:
        lines.pop()
    out = "\n".join(lines).rstrip()
    if not out:
        return text[: max(0, limit - 1)] + "…"
    if len(out) + len(suffix) > limit:
        out = out[: max(0, limit - len(suffix) - 1)].rstrip()
    return out + suffix


def load_do_not_restart(path: Path) -> set[str]:
    if not path.exists():
        return set()
    markets: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        markets.add(normalize_market_name(line))
    return markets


def _parse_market_from_filename(path: Path) -> Optional[str]:
    name = path.name
    if not name.startswith("mm_") or not name.endswith(".jsonl"):
        return None
    if "mm_tuning_log_" in name:
        return None
    inner = name[3:-6]
    if inner.endswith("_latest"):
        market = inner[:-7]
    else:
        match = _TS_SUFFIX_RE.match(inner)
        market = match.group(1) if match else inner
    if not market:
        return None
    return normalize_market_name(market)


def discover_market_journals(journal_dir: Path) -> Dict[str, Path]:
    latest_files = sorted(journal_dir.glob("mm_*_latest.jsonl"), key=lambda p: p.stat().st_mtime)
    by_market: Dict[str, Path] = {}

    for path in latest_files:
        market = _parse_market_from_filename(path)
        if market:
            by_market[market] = path
    if by_market:
        return dict(sorted(by_market.items(), key=lambda item: item[0]))

    all_files = sorted(journal_dir.glob("mm_*.jsonl"), key=lambda p: p.stat().st_mtime)
    for path in all_files:
        market = _parse_market_from_filename(path)
        if market:
            by_market[market] = path
    return dict(sorted(by_market.items(), key=lambda item: item[0]))


def load_recent_entries(path: Path, min_ts: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            ts = to_float(obj.get("ts"))
            if ts is None or ts < min_ts:
                continue
            out.append(obj)
    out.sort(key=lambda row: to_float(row.get("ts")) or 0.0)
    return out


def parse_env_map(env_map_arg: Optional[str]) -> Dict[str, Path]:
    mapping = {
        normalize_market_name(market): _resolve_path(path)
        for market, path in DEFAULT_ENV_MAP.items()
    }
    if not env_map_arg:
        return mapping

    raw = env_map_arg.strip()
    if not raw:
        return mapping

    custom: Dict[str, str] = {}
    maybe_path = Path(raw)
    if maybe_path.exists():
        text = maybe_path.read_text().strip()
        custom = _parse_env_map_text(text)
    else:
        custom = _parse_env_map_text(raw)

    for market, env_path in custom.items():
        mapping[normalize_market_name(market)] = _resolve_path(env_path)
    return mapping


def _parse_env_map_text(text: str) -> Dict[str, str]:
    if not text:
        return {}
    if text.lstrip().startswith("{"):
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("--env-map JSON must decode to an object")
        out: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            out[str(key)] = str(value)
        return out

    items: List[str] = []
    for chunk in text.replace("\n", ",").split(","):
        candidate = chunk.strip()
        if candidate:
            items.append(candidate)

    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env-map entry: {item}")
        market, env_path = item.split("=", 1)
        market = market.strip()
        env_path = env_path.strip()
        if not market or not env_path:
            raise ValueError(f"Invalid --env-map entry: {item}")
        out[market] = env_path
    return out


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / value).resolve()


def load_market_env(market: str, env_map: Dict[str, Path]) -> Tuple[Optional[Path], Dict[str, str]]:
    path = env_map.get(normalize_market_name(market))
    if path is None or not path.exists():
        return (path, {})
    return (path, parse_env(read_env_lines(path)))


def is_block_reason(reason: str) -> bool:
    return reason.startswith("skip_") or reason.startswith("hold_")


def collect_mid_series(events: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    ts_values: List[float] = []
    mids: List[float] = []
    for event in events:
        ts = to_float(event.get("ts"))
        if ts is None:
            continue
        mid = to_float(event.get("mid"))
        if mid is None:
            bid = to_float(event.get("best_bid"))
            ask = to_float(event.get("best_ask"))
            if bid is None or ask is None:
                price = to_float(event.get("price"))
                mid = price
            else:
                mid = (bid + ask) / 2.0
        if mid is None:
            continue
        ts_values.append(ts)
        mids.append(mid)
    return ts_values, mids


def mid_at_or_after(
    ts_values: Sequence[float],
    mids: Sequence[float],
    target_ts: float,
    *,
    max_wait_s: float = 120.0,
) -> Optional[float]:
    if not ts_values:
        return None
    idx = bisect_left(ts_values, target_ts)
    if idx >= len(ts_values):
        return None
    if ts_values[idx] - target_ts > max_wait_s:
        return None
    return mids[idx]


def quantile_average(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    count = max(1, int(len(values) * fraction))
    return sum(values[:count]) / float(count)


def zero_crossings(values: Sequence[float], *, epsilon: float = 1e-9) -> int:
    crossings = 0
    prev_sign = 0
    for value in values:
        sign = 0
        if value > epsilon:
            sign = 1
        elif value < -epsilon:
            sign = -1
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            crossings += 1
        prev_sign = sign
    return crossings


def compute_time_above_util(
    positions: Sequence[Tuple[float, float]],
    *,
    max_position: float,
    threshold: float,
    window_end_ts: float,
) -> float:
    if max_position <= 0 or not positions:
        return 0.0

    sorted_positions = sorted(positions, key=lambda item: item[0])
    total = 0.0
    for idx, (ts, position) in enumerate(sorted_positions):
        next_ts = window_end_ts if idx + 1 >= len(sorted_positions) else sorted_positions[idx + 1][0]
        if next_ts <= ts:
            continue
        util = abs(position) / max_position
        if util > threshold:
            total += (next_ts - ts)
    return total


def rolling_price_change(events: Sequence[Dict[str, Any]]) -> float:
    ts_values, mids = collect_mid_series(events)
    if len(mids) < 2:
        return 0.0
    return mids[-1] - mids[0]


def round_or_none(value: Optional[float], digits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def median(values: Iterable[float]) -> float:
    items = sorted(values)
    if not items:
        return 0.0
    mid = len(items) // 2
    if len(items) % 2 == 1:
        return float(items[mid])
    return float((items[mid - 1] + items[mid]) / 2.0)
