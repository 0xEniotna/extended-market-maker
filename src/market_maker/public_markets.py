"""
Public market-data helpers for Extended Exchange.

This module is intentionally standalone for the OSS MM repository and does not
depend on the legacy ``extended_pairs_bot`` package.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List

import requests

MAINNET_API_BASE = "https://api.starknet.extended.exchange/api/v1"
TESTNET_API_BASE = "https://api.starknet.sepolia.extended.exchange/api/v1"
MARKETS_PATH = "/info/markets"


def resolve_default_api_base() -> str:
    """Resolve API base URL from environment with safe fallbacks.

    Priority:
    1) ``EXTENDED_API_BASE`` when explicitly set
    2) ``MM_ENVIRONMENT`` / ``EXTENDED_ENV`` equals ``mainnet`` or ``testnet``
    3) testnet default
    """
    explicit = os.getenv("EXTENDED_API_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")

    env = (
        os.getenv("MM_ENVIRONMENT")
        or os.getenv("EXTENDED_ENV")
        or "testnet"
    ).strip().lower()
    if env == "mainnet":
        return MAINNET_API_BASE
    return TESTNET_API_BASE


@dataclass
class PublicMarketsClient:
    """Simple wrapper around the public markets endpoint."""

    api_base: str

    @classmethod
    def default(cls) -> "PublicMarketsClient":
        return cls(api_base=resolve_default_api_base())

    def fetch_all_markets(self) -> List[Dict[str, Any]]:
        url = f"{self.api_base}{MARKETS_PATH}"
        resp = requests.get(
            url,
            headers={"User-Agent": "extended-market-maker/0.1"},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()

        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected markets payload (not a dict): {payload!r}")

        status = str(payload.get("status", "")).lower()
        if status != "ok":
            raise RuntimeError(f"Extended API error: {payload!r}")

        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected `data` in markets payload: {data!r}")

        return data


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def rank_markets_by_liquidity(
    markets: List[Dict[str, Any]],
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Filter to active tradable markets and rank by volume/open interest."""
    filtered: List[Dict[str, Any]] = []
    for market in markets:
        if not market.get("active") or market.get("status") != "ACTIVE":
            continue

        stats = market.get("marketStats") or {}
        market["_daily_volume"] = _to_decimal(stats.get("dailyVolume", "0"))
        market["_open_interest"] = _to_decimal(stats.get("openInterest", "0"))
        filtered.append(market)

    filtered.sort(
        key=lambda item: (item["_daily_volume"], item["_open_interest"]),
        reverse=True,
    )
    return filtered[:top_n]

