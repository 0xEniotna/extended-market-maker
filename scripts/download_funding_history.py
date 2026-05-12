#!/usr/bin/env python3
"""Download Extended funding-rate history per market and cache to JSON.

Each market's series is saved to data/funding_history/<market>.json with
one entry per funding period (typically hourly) of the form:

    [{"timestamp": <epoch_ms>, "funding_rate": "<decimal>"}, ...]

The Extended endpoint returns at most N entries per call (observed ~48 for
a 2-day window). We page by adjusting the time window.

Usage:
    PYTHONPATH=src python scripts/download_funding_history.py \
        --markets ETH-USD DOT-USD SPX500m-USD MU_24_5-USD \
        --days 7 \
        --out data/funding_history
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Use the same settings/factory the bot uses so credentials and base URLs
# are resolved the same way.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv  # noqa: E402


def _load_env(env_file: str | None) -> None:
    if env_file:
        load_dotenv(env_file)
    elif os.path.exists(".env.eth"):
        load_dotenv(".env.eth")
    elif os.path.exists(".env"):
        load_dotenv(".env")


def _decimal_default(o: object) -> str:
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError(f"unsupported: {type(o)}")


async def _fetch_window(client, market: str, start: datetime, end: datetime) -> list[dict]:
    resp = await client.markets_info.get_funding_rates_history(
        market_name=market, start_time=start, end_time=end,
    )
    data = resp.data if hasattr(resp, "data") else resp
    return [
        {"timestamp": int(e.timestamp), "funding_rate": str(e.funding_rate)}
        for e in (data or [])
    ]


async def _fetch_full(client, market: str, days: int) -> list[dict]:
    """Page in 2-day chunks because the endpoint caps response size."""
    end = datetime.now(timezone.utc)
    seen_ts: set[int] = set()
    out: list[dict] = []
    days_remaining = days
    while days_remaining > 0:
        chunk = min(2, days_remaining)
        start = end - timedelta(days=chunk)
        try:
            entries = await _fetch_window(client, market, start, end)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: window {start.date()}..{end.date()} failed: {exc}")
            entries = []
        for e in entries:
            ts = e["timestamp"]
            if ts not in seen_ts:
                seen_ts.add(ts)
                out.append(e)
        end = start
        days_remaining -= chunk
    out.sort(key=lambda r: r["timestamp"])
    return out


async def _run(markets: list[str], days: int, out_dir: Path, env_file: str | None) -> None:
    _load_env(env_file)
    os.environ.setdefault("MM_ENVIRONMENT", "mainnet")
    from market_maker.config import MarketMakerSettings
    from market_maker.strategy_factory import StrategyFactory

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = MarketMakerSettings()
    client = StrategyFactory(settings).build_trading_client()
    try:
        for market in markets:
            print(f"== {market} ==")
            entries = await _fetch_full(client, market, days)
            path = out_dir / f"{market}.json"
            with path.open("w") as f:
                json.dump(
                    {"market": market, "days": days, "entries": entries},
                    f, default=_decimal_default,
                )
            if entries:
                first_ts = entries[0]["timestamp"]
                last_ts = entries[-1]["timestamp"]
                print(
                    f"  {len(entries)} entries  "
                    f"{datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).isoformat()} → "
                    f"{datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).isoformat()}  "
                    f"→ {path}"
                )
            else:
                print(f"  0 entries → {path}")
    finally:
        await client.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--markets", nargs="+", required=True)
    p.add_argument("--days", type=int, default=7, help="Lookback in days")
    p.add_argument("--out", type=Path, default=Path("data/funding_history"))
    p.add_argument("--env-file", default=None, help="Override env file path")
    args = p.parse_args()
    asyncio.run(_run(args.markets, args.days, args.out, args.env_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
