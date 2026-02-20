#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from x10.perpetual.accounts import StarkPerpetualAccount  # noqa: E402
from x10.perpetual.positions import PositionSide  # noqa: E402
from x10.perpetual.trading_client import PerpetualTradingClient  # noqa: E402
from x10.utils.http import ResponseStatus  # noqa: E402

from market_maker.config import MarketMakerSettings  # noqa: E402


def _resolve_env_file(env_value: str) -> Path:
    """Resolve ENV argument to a concrete file path.

    Supports:
    - explicit file path (relative or absolute), e.g. `.env.silver`
    - short name, e.g. `silver` -> `.env.silver`
    """
    raw = env_value.strip()
    candidates = [raw]
    if raw and not raw.startswith("."):
        candidates.insert(0, f".{raw}")

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / candidate
        if path.exists():
            return path

    # Fall back to first candidate under project root for a clear error message.
    path = Path(candidates[0])
    if not path.is_absolute():
        path = PROJECT_ROOT / candidates[0]
    return path


def _fmt_decimal(value: Decimal) -> str:
    return f"{value:.6f}"


def _fmt_ts(ms: Optional[int]) -> str:
    if not ms:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ms / 1000))


def _to_ms(days: Optional[float]) -> Optional[int]:
    if days is None:
        return None
    now_ms = int(time.time() * 1000)
    return now_ms - int(days * 24 * 60 * 60 * 1000)


def _ensure_ok(resp, label: str) -> None:
    if resp.status != ResponseStatus.OK:
        raise RuntimeError(f"{label} failed: status={resp.status} error={resp.error}")


def _to_jsonable(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


async def _resolve_market_name(
    client: PerpetualTradingClient, requested_market: str
) -> str:
    """Resolve market name to the exact exchange casing/symbol."""
    market = requested_market.strip()
    if not market:
        raise RuntimeError("Market cannot be empty.")

    markets = await client.markets_info.get_markets_dict()
    if market in markets:
        return market

    lower_map = {name.lower(): name for name in markets.keys()}
    exact_ci = lower_map.get(market.lower())
    if exact_ci:
        return exact_ci

    # Friendly hints for common symbol confusion.
    market_norm = market.replace("-", "").lower()
    suggestions = [
        name
        for name in markets.keys()
        if market_norm in name.replace("-", "").lower()
    ]
    suggestions = sorted(suggestions)[:8]
    if suggestions:
        raise RuntimeError(
            f"Market not found: {requested_market}. Did you mean: {', '.join(suggestions)}"
        )
    raise RuntimeError(f"Market not found: {requested_market}")


@dataclass
class HistoryStats:
    count: int = 0
    realized_total: Decimal = Decimal("0")
    trade_pnl_total: Decimal = Decimal("0")
    funding_fees_total: Decimal = Decimal("0")
    open_fees_total: Decimal = Decimal("0")
    close_fees_total: Decimal = Decimal("0")
    wins: int = 0
    losses: int = 0
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None

    def observe(self, pnl: Decimal, ts_ms: Optional[int]) -> None:
        self.count += 1
        self.realized_total += pnl
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1
        if ts_ms is not None:
            if self.first_ts is None or ts_ms < self.first_ts:
                self.first_ts = ts_ms
            if self.last_ts is None or ts_ms > self.last_ts:
                self.last_ts = ts_ms


async def _run(args: argparse.Namespace) -> int:
    market_input = args.market.strip()
    since_ms = _to_ms(args.days)

    if args.env:
        env_file = _resolve_env_file(args.env)
        if not env_file.exists():
            raise RuntimeError(f"Env file not found: {env_file}")
        os.environ["ENV"] = str(env_file)
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()

    settings = MarketMakerSettings()
    if not settings.is_configured:
        raise RuntimeError(
            "Missing MM credentials. Ensure MM_VAULT_ID/MM_STARK_PRIVATE_KEY/"
            "MM_STARK_PUBLIC_KEY/MM_API_KEY are set."
        )

    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    client = PerpetualTradingClient(settings.endpoint_config, account)
    market = await _resolve_market_name(client, market_input)

    stats = HistoryStats()
    pages = 0
    cursor: Optional[int] = None
    seen_cursors: set[int] = set()

    open_realized = Decimal("0")
    open_unrealized = Decimal("0")
    open_notional = Decimal("0")
    net_open_size = Decimal("0")

    try:
        while pages < args.max_pages:
            resp = await client.account.get_positions_history(
                market_names=[market],
                cursor=cursor,
                limit=args.page_size,
            )
            _ensure_ok(resp, "get_positions_history")
            rows = resp.data or []

            for row in rows:
                # History endpoint also includes currently open positions with
                # closed_time=None. Keep closed-history and open-position
                # accounting separate to avoid double-counting.
                if row.closed_time is None:
                    continue

                ts_ms = row.closed_time
                if since_ms is not None and ts_ms is not None and ts_ms < since_ms:
                    continue

                stats.observe(row.realised_pnl, ts_ms)
                stats.trade_pnl_total += row.realised_pnl_breakdown.trade_pnl
                stats.funding_fees_total += row.realised_pnl_breakdown.funding_fees
                stats.open_fees_total += row.realised_pnl_breakdown.open_fees
                stats.close_fees_total += row.realised_pnl_breakdown.close_fees

            pages += 1
            next_cursor = resp.pagination.cursor if resp.pagination else None
            if next_cursor is None or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        pos_resp = await client.account.get_positions(market_names=[market])
        _ensure_ok(pos_resp, "get_positions")
        positions = pos_resp.data or []

        for pos in positions:
            if pos.market.upper() != market:
                continue
            sign = Decimal("1") if pos.side == PositionSide.LONG else Decimal("-1")
            net_open_size += sign * pos.size
            open_realized += pos.realised_pnl
            open_unrealized += pos.unrealised_pnl
            open_notional += pos.value

    finally:
        await client.close()

    overall = stats.realized_total + open_realized + open_unrealized
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "market": market,
        "environment": settings.environment.value,
        "window_days": args.days,
        "closed_positions": {
            "count": stats.count,
            "wins": stats.wins,
            "losses": stats.losses,
            "realized_pnl_usd": stats.realized_total,
            "trade_pnl_usd": stats.trade_pnl_total,
            "funding_fees_usd": stats.funding_fees_total,
            "open_fees_usd": stats.open_fees_total,
            "close_fees_usd": stats.close_fees_total,
            "first_close_ms": stats.first_ts,
            "last_close_ms": stats.last_ts,
        },
        "open_positions": {
            "net_size": net_open_size,
            "notional_usd": open_notional,
            "realized_component_usd": open_realized,
            "unrealized_component_usd": open_unrealized,
        },
        "totals": {
            "closed_realized_pnl_usd": stats.realized_total,
            "open_realized_pnl_usd": open_realized,
            "open_unrealized_pnl_usd": open_unrealized,
            "total_pnl_including_open_usd": overall,
        },
    }

    print(f"PnL summary for {market}")
    print(f"Environment: {settings.environment.value}")
    if args.days is None:
        print("Window: all available history")
    else:
        print(f"Window: last {args.days} days")

    print("\nClosed positions (history):")
    print(f"  count: {stats.count} (wins={stats.wins} losses={stats.losses})")
    print(f"  realized_pnl: {_fmt_decimal(stats.realized_total)} USD")
    print(f"  trade_pnl: {_fmt_decimal(stats.trade_pnl_total)} USD")
    print(f"  funding_fees: {_fmt_decimal(stats.funding_fees_total)} USD")
    print(f"  open_fees: {_fmt_decimal(stats.open_fees_total)} USD")
    print(f"  close_fees: {_fmt_decimal(stats.close_fees_total)} USD")
    print(f"  first_close: {_fmt_ts(stats.first_ts)}")
    print(f"  last_close: {_fmt_ts(stats.last_ts)}")

    print("\nOpen positions (current):")
    print(f"  net_size: {_fmt_decimal(net_open_size)}")
    print(f"  notional: {_fmt_decimal(open_notional)} USD")
    print(f"  realized_component: {_fmt_decimal(open_realized)} USD")
    print(f"  unrealized_component: {_fmt_decimal(open_unrealized)} USD")

    print("\nTotals:")
    print(
        "  total_pnl_including_open="
        f"{_fmt_decimal(overall)} USD "
        "(closed_realized + open_realized + open_unrealized)"
    )
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n")
    if args.json_stdout:
        print(json.dumps(_to_jsonable(payload), indent=2))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch PnL summary for one market using account position history and current positions."
        )
    )
    parser.add_argument("market", help="Market symbol, e.g. BTC-USD")
    parser.add_argument(
        "--env",
        default=None,
        help=(
            "Env file to load (e.g. .env.silver or silver). "
            "Overrides process env with values from that file."
        ),
    )
    parser.add_argument(
        "--days",
        type=float,
        default=None,
        help="Optional lookback window in days for closed-position history (default: all history).",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Page size for positions history fetch (default: 200).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Safety cap on history pages to fetch (default: 100).",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write structured JSON summary.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print structured JSON summary to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
