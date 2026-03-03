"""mmctl markets — Market discovery, info, and screening."""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from market_maker.cli.common import PROJECT_ROOT, to_jsonable


# ---------------------------------------------------------------------------
# mmctl markets info (lifted from scripts/tools/fetch_market_info.py)
# ---------------------------------------------------------------------------


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, Decimal):
        if value == 0:
            return "0"
        return format(value.normalize(), "f").rstrip("0").rstrip(".")
    return str(value)


def _load_env(args) -> None:
    """Load the .env file specified by --env, or the default .env."""
    env_file = getattr(args, "env", None)
    if env_file:
        path = Path(env_file)
        if not path.exists():
            print(f"Warning: env file {path} not found", file=sys.stderr)
        else:
            load_dotenv(path, override=True)
    else:
        load_dotenv()


def _resolve_api_base(api_base: Optional[str]) -> str:
    from market_maker.public_markets import MAINNET_API_BASE, TESTNET_API_BASE

    if api_base:
        return api_base.rstrip("/")
    explicit = os.getenv("EXTENDED_API_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")
    env = (os.getenv("MM_ENVIRONMENT") or os.getenv("EXTENDED_ENV") or "mainnet").strip().lower()
    if env == "testnet":
        return TESTNET_API_BASE
    return MAINNET_API_BASE


def _fetch_market(api_base: str, market_name: str) -> Dict[str, Any]:
    import requests

    from market_maker.public_markets import MARKETS_PATH

    url = f"{api_base}{MARKETS_PATH}"
    resp = requests.get(
        url, params={"market": market_name},
        headers={"User-Agent": "extended-market-maker/0.1"}, timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected markets payload: {payload!r}")
    if str(payload.get("status", "")).lower() != "ok":
        raise RuntimeError(f"Extended API error: {payload!r}")

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"Market not found: {market_name}")
    if len(data) == 1:
        return data[0]

    for market in data:
        if str(market.get("name", "")).upper() == market_name.upper():
            return market
    raise RuntimeError(f"Market not found: {market_name}")


def _run_market_info(args) -> int:
    _load_env(args)
    api_base = _resolve_api_base(getattr(args, "api_base", None))
    market = _fetch_market(api_base, args.info_market)

    if getattr(args, "json", False):
        print(json.dumps(market, indent=2, sort_keys=True))
        return 0

    stats = market.get("marketStats") or {}
    trading = market.get("tradingConfig") or {}
    min_order_size = _to_decimal(trading.get("minOrderSize"))
    min_price_change = _to_decimal(trading.get("minPriceChange"))
    bid = _to_decimal(stats.get("bidPrice"))
    ask = _to_decimal(stats.get("askPrice"))
    mark = _to_decimal(stats.get("markPrice"))

    mid: Optional[Decimal] = None
    if bid > 0 and ask > 0:
        mid = (bid + ask) / Decimal("2")
    elif mark > 0:
        mid = mark

    tick_bps: Optional[Decimal] = None
    if mid and mid > 0 and min_price_change > 0:
        tick_bps = (min_price_change / mid) * Decimal("10000")

    print(f"Market: {market.get('name')}")
    print(f"Status: {market.get('status')} active={market.get('active')}")
    print(f"Asset: {market.get('assetName')} collateral={market.get('collateralAssetName')}")
    print(f"\nTrading config:")
    print(f"  minOrderSize: {_fmt(min_order_size)}")
    print(f"  minOrderSizeChange: {_fmt(trading.get('minOrderSizeChange'))}")
    print(f"  minPriceChange: {_fmt(min_price_change)}")
    print(f"  maxPositionValue: {_fmt(trading.get('maxPositionValue'))}")
    print(f"  maxLeverage: {_fmt(trading.get('maxLeverage'))}")
    print(f"\nMarket stats:")
    print(f"  bidPrice: {_fmt(stats.get('bidPrice'))}")
    print(f"  askPrice: {_fmt(stats.get('askPrice'))}")
    print(f"  markPrice: {_fmt(stats.get('markPrice'))}")
    print(f"  dailyVolume: {_fmt(stats.get('dailyVolume'))}")
    print(f"  openInterest: {_fmt(stats.get('openInterest'))}")
    print(f"\nMM inputs:")
    print(f"  tick_size: {_fmt(min_price_change)}")
    print(f"  min_order_size: {_fmt(min_order_size)}")
    if tick_bps is not None:
        print(f"  tick_size_bps: {_fmt(tick_bps)}")
    return 0


# ---------------------------------------------------------------------------
# mmctl markets find (delegates to scripts/tools/find_mm_markets.py)
# ---------------------------------------------------------------------------


def _run_markets_find(args) -> int:
    _load_env(args)
    scripts_tools = PROJECT_ROOT / "scripts" / "tools"
    if str(scripts_tools) not in sys.path:
        sys.path.insert(0, str(scripts_tools))

    from find_mm_markets import main as find_main

    argv = []
    if hasattr(args, "duration_s") and args.duration_s is not None:
        argv.extend(["--duration-s", str(args.duration_s)])
    if hasattr(args, "interval_s") and args.interval_s is not None:
        argv.extend(["--interval-s", str(args.interval_s)])
    if hasattr(args, "limit") and args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    if getattr(args, "json", False):
        argv.append("--json-stdout")
    if getattr(args, "api_base", None):
        argv.extend(["--api-base", args.api_base])

    # Monkey-patch sys.argv for the standalone main()
    old_argv = sys.argv
    sys.argv = ["find_mm_markets"] + argv
    try:
        return find_main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# mmctl markets screen (delegates to scripts/screen_mm_markets.py)
# ---------------------------------------------------------------------------


def _run_markets_screen(args) -> int:
    _load_env(args)
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from screen_mm_markets import main as screen_main

    argv = []
    if hasattr(args, "duration_s") and args.duration_s is not None:
        argv.extend(["--duration-s", str(args.duration_s)])
    if hasattr(args, "limit") and args.limit is not None:
        argv.extend(["--limit", str(args.limit)])
    if getattr(args, "json", False):
        argv.append("--json-stdout")
    if getattr(args, "api_base", None):
        argv.extend(["--api-base", args.api_base])

    old_argv = sys.argv
    sys.argv = ["screen_mm_markets"] + argv
    try:
        return screen_main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    markets_parser = subparsers.add_parser("markets", help="Market discovery and info")
    markets_sub = markets_parser.add_subparsers(dest="markets_command")

    # info
    info_p = markets_sub.add_parser("info", help="Market trading config and stats")
    info_p.add_argument("info_market", metavar="market", help="Market name (e.g. ETH-USD)")
    info_p.add_argument("--json", action="store_true", help="Raw JSON output")
    info_p.add_argument("--api-base", default=None, help="Override API base URL")
    info_p.add_argument("--env", default=None, help="Path to .env file (e.g. .env.eth)")
    info_p.set_defaults(func=_run_market_info)

    # find
    find_p = markets_sub.add_parser("find", help="Find suitable MM markets via rolling spread sampling")
    find_p.add_argument("--json", action="store_true", help="JSON output")
    find_p.add_argument("--duration-s", type=float, default=120.0, help="Sampling window seconds")
    find_p.add_argument("--interval-s", type=float, default=2.0, help="Snapshot interval seconds")
    find_p.add_argument("--limit", type=int, default=40, help="Max rows")
    find_p.add_argument("--api-base", default=None, help="Override API base URL")
    find_p.add_argument("--env", default=None, help="Path to .env file (e.g. .env.eth)")
    find_p.set_defaults(func=_run_markets_find)

    # screen
    screen_p = markets_sub.add_parser("screen", help="Screen markets for MM suitability")
    screen_p.add_argument("--json", action="store_true", help="JSON output")
    screen_p.add_argument("--duration-s", type=float, default=180.0, help="Sampling window seconds")
    screen_p.add_argument("--limit", type=int, default=80, help="Max rows")
    screen_p.add_argument("--api-base", default=None, help="Override API base URL")
    screen_p.add_argument("--env", default=None, help="Path to .env file (e.g. .env.eth)")
    screen_p.set_defaults(func=_run_markets_screen)

    markets_parser.set_defaults(func=lambda args: _markets_help(markets_parser, args))


def _markets_help(parser, args) -> int:
    if not getattr(args, "markets_command", None):
        parser.print_help()
        return 1
    return 0
