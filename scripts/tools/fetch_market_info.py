#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_maker.public_markets import (  # noqa: E402
    MAINNET_API_BASE,
    MARKETS_PATH,
    TESTNET_API_BASE,
)


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


def _resolve_api_base(api_base: Optional[str]) -> str:
    if api_base:
        return api_base.rstrip("/")
    explicit = os.getenv("EXTENDED_API_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")

    env = (os.getenv("MM_ENVIRONMENT") or os.getenv("EXTENDED_ENV") or "testnet").strip().lower()
    if env == "mainnet":
        return MAINNET_API_BASE
    return TESTNET_API_BASE


def _fetch_market(api_base: str, market_name: str) -> Dict[str, Any]:
    url = f"{api_base}{MARKETS_PATH}"
    resp = requests.get(
        url,
        params={"market": market_name},
        headers={"User-Agent": "extended-market-maker/0.1"},
        timeout=10,
    )
    resp.raise_for_status()
    payload = resp.json()

    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected markets payload (not a dict): {payload!r}")

    status = str(payload.get("status", "")).lower()
    if status != "ok":
        raise SystemExit(f"Extended API error: {payload!r}")

    data = payload.get("data")
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected `data` in markets payload: {data!r}")
    if not data:
        raise SystemExit(f"Market not found: {market_name}")

    if len(data) == 1:
        return data[0]

    for market in data:
        if str(market.get("name", "")).upper() == market_name.upper():
            return market

    raise SystemExit(f"Market not found: {market_name}")


def _print_summary(market: Dict[str, Any]) -> None:
    name = market.get("name")
    status = market.get("status")
    active = market.get("active")
    asset = market.get("assetName")
    collateral = market.get("collateralAssetName")

    stats = market.get("marketStats") or {}
    trading = market.get("tradingConfig") or {}

    min_order_size = _to_decimal(trading.get("minOrderSize"))
    min_order_size_change = _to_decimal(trading.get("minOrderSizeChange"))
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

    min_order_notional: Optional[Decimal] = None
    if mark > 0 and min_order_size > 0:
        min_order_notional = min_order_size * mark

    print(f"Market: {name}")
    print(f"Status: {status} active={active}")
    print(f"Asset: {asset} collateral={collateral}")

    print("\nTrading config:")
    print(f"  minOrderSize: {_fmt(min_order_size)}")
    print(f"  minOrderSizeChange: {_fmt(min_order_size_change)}")
    print(f"  minPriceChange: {_fmt(min_price_change)}")
    print(f"  maxPositionValue: {_fmt(trading.get('maxPositionValue'))}")
    print(f"  maxLeverage: {_fmt(trading.get('maxLeverage'))}")
    print(f"  maxNumOrders: {_fmt(trading.get('maxNumOrders'))}")

    print("\nMarket stats:")
    print(f"  bidPrice: {_fmt(stats.get('bidPrice'))}")
    print(f"  askPrice: {_fmt(stats.get('askPrice'))}")
    print(f"  markPrice: {_fmt(stats.get('markPrice'))}")
    print(f"  dailyVolume: {_fmt(stats.get('dailyVolume'))}")
    print(f"  openInterest: {_fmt(stats.get('openInterest'))}")

    print("\nMM inputs:")
    print(f"  tick_size: {_fmt(min_price_change)}")
    print(f"  min_order_size: {_fmt(min_order_size)}")
    print(f"  min_order_size_change: {_fmt(min_order_size_change)}")
    if tick_bps is not None:
        print(f"  tick_size_bps: {_fmt(tick_bps)}")
    if min_order_notional is not None:
        print(f"  min_order_notional: {_fmt(min_order_notional)}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Extended market info for a specific market.",
    )
    parser.add_argument(
        "market",
        help="Market name (e.g. BTC-USD)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (e.g. https://api.starknet.extended.exchange/api/v1)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw JSON for the market instead of a summary.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_base = _resolve_api_base(args.api_base)

    market = _fetch_market(api_base, args.market)

    if args.raw:
        print(json.dumps(market, indent=2, sort_keys=True))
        return 0

    _print_summary(market)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
