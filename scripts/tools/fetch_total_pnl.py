#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
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

_MS_PER_DAY = Decimal("86400000")


def _resolve_env_file(env_value: str) -> Path:
    """Resolve ENV argument to a concrete file path."""
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

    path = Path(candidates[0])
    if not path.is_absolute():
        path = PROJECT_ROOT / candidates[0]
    return path


def _ensure_ok(resp, label: str) -> None:
    if resp.status != ResponseStatus.OK:
        raise RuntimeError(f"{label} failed: status={resp.status} error={resp.error}")


def _fmt_decimal(value: Decimal) -> str:
    return f"{value:.6f}"


def _fmt_pct(value: Decimal) -> str:
    return f"{(value * Decimal('100')):.4f}%"


def _fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_since_timestamp(raw: str) -> int:
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError("timestamp cannot be empty")

    if value.isdigit():
        ts = int(value)
        # Treat small integer values as epoch seconds.
        return ts if ts >= 1_000_000_000_000 else ts * 1000

    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "invalid timestamp. Use epoch seconds/ms or ISO-8601, "
            "e.g. 1704067200, 1704067200000, 2026-02-01T00:00:00Z"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _parse_decimal(raw: Optional[str], *, label: str) -> Optional[Decimal]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid {label}: {raw}") from exc


def _infer_starting_equity(current_equity: Decimal, total_pnl: Decimal) -> Decimal:
    return current_equity - total_pnl


def _annualized_returns(
    *,
    total_pnl: Decimal,
    starting_equity: Decimal,
    period_days: Decimal,
) -> tuple[Decimal, Decimal, Optional[Decimal]]:
    if starting_equity <= 0:
        raise ValueError("starting_equity must be > 0")
    if period_days <= 0:
        raise ValueError("period_days must be > 0")

    total_return = total_pnl / starting_equity
    annual_factor = Decimal("365") / period_days
    apr = total_return * annual_factor

    apy: Optional[Decimal] = None
    one_plus_return = Decimal("1") + total_return
    if one_plus_return > 0:
        apy_float = math.pow(float(one_plus_return), float(annual_factor)) - 1.0
        apy = Decimal(str(apy_float))

    return total_return, apr, apy


async def _run(args: argparse.Namespace) -> int:
    since_ms = _parse_since_timestamp(args.since)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    until_ms = _parse_since_timestamp(args.until) if args.until else now_ms
    if until_ms > now_ms:
        raise RuntimeError("`--until` must be earlier than current time")
    if since_ms >= until_ms:
        if args.until:
            raise RuntimeError("`--since` must be earlier than `--until`")
        raise RuntimeError("`--since` must be earlier than current time")
    period_days = Decimal(until_ms - since_ms) / _MS_PER_DAY
    historical_until = args.until is not None and until_ms < now_ms

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

    closed_realized_total = Decimal("0")
    trade_pnl_total = Decimal("0")
    funding_fees_total = Decimal("0")
    open_fees_total = Decimal("0")
    close_fees_total = Decimal("0")
    closed_count = 0

    open_realized_total = Decimal("0")
    open_unrealized_total = Decimal("0")
    open_notional_total = Decimal("0")
    net_open_size_total = Decimal("0")
    open_count_included = 0
    open_count_skipped_preexisting = 0

    per_market_closed = defaultdict(Decimal)
    per_market_open_realized = defaultdict(Decimal)
    per_market_open_unrealized = defaultdict(Decimal)

    pages = 0
    cursor: Optional[int] = None
    seen_cursors: set[int] = set()

    current_equity: Optional[Decimal] = None

    try:
        while pages < args.max_pages:
            resp = await client.account.get_positions_history(
                market_names=None,
                cursor=cursor,
                limit=args.page_size,
            )
            _ensure_ok(resp, "get_positions_history")
            rows = resp.data or []

            for row in rows:
                # History endpoint also returns currently open positions with
                # closed_time=None. Those are accounted for via get_positions().
                if row.closed_time is None:
                    continue

                ts_ms = row.closed_time
                if ts_ms is None or ts_ms < since_ms or ts_ms > until_ms:
                    continue

                market = str(row.market).upper()
                realized = Decimal(str(row.realised_pnl))
                closed_count += 1
                closed_realized_total += realized
                per_market_closed[market] += realized

                trade_pnl_total += Decimal(str(row.realised_pnl_breakdown.trade_pnl))
                funding_fees_total += Decimal(str(row.realised_pnl_breakdown.funding_fees))
                open_fees_total += Decimal(str(row.realised_pnl_breakdown.open_fees))
                close_fees_total += Decimal(str(row.realised_pnl_breakdown.close_fees))

            pages += 1
            next_cursor = resp.pagination.cursor if resp.pagination else None
            if next_cursor is None or next_cursor in seen_cursors:
                break
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        if not historical_until:
            pos_resp = await client.account.get_positions(market_names=None)
            _ensure_ok(pos_resp, "get_positions")
            positions = pos_resp.data or []

            for pos in positions:
                created_at = getattr(pos, "created_at", None)
                if (
                    created_at is not None
                    and created_at < since_ms
                    and not args.include_preexisting_open
                ):
                    open_count_skipped_preexisting += 1
                    continue
                if created_at is not None and created_at > until_ms:
                    continue

                market = str(pos.market).upper()
                realized = Decimal(str(pos.realised_pnl))
                unrealized = Decimal(str(pos.unrealised_pnl))
                value = Decimal(str(pos.value))
                sign = Decimal("1") if pos.side == PositionSide.LONG else Decimal("-1")

                open_count_included += 1
                open_realized_total += realized
                open_unrealized_total += unrealized
                open_notional_total += value
                net_open_size_total += sign * Decimal(str(pos.size))

                per_market_open_realized[market] += realized
                per_market_open_unrealized[market] += unrealized

            bal_resp = await client.account.get_balance()
            _ensure_ok(bal_resp, "get_balance")
            if bal_resp.data is not None:
                current_equity = Decimal(str(bal_resp.data.equity))
    finally:
        await client.close()

    total_pnl = closed_realized_total + open_realized_total + open_unrealized_total
    starting_equity = _parse_decimal(args.starting_equity_usd, label="starting equity")
    starting_equity_source = "provided"
    if starting_equity is None:
        if current_equity is not None and not historical_until:
            starting_equity = _infer_starting_equity(current_equity, total_pnl)
            starting_equity_source = "inferred(current_equity - total_pnl)"
        elif historical_until:
            starting_equity_source = "unavailable(historical --until window)"
        else:
            starting_equity_source = "unavailable"

    total_return: Optional[Decimal] = None
    apr: Optional[Decimal] = None
    apy: Optional[Decimal] = None
    annualized_error: Optional[str] = None
    if starting_equity is not None:
        try:
            total_return, apr, apy = _annualized_returns(
                total_pnl=total_pnl,
                starting_equity=starting_equity,
                period_days=period_days,
            )
        except ValueError as exc:
            annualized_error = str(exc)

    print("PnL summary across all markets")
    print(f"Environment: {settings.environment.value}")
    print(f"Since: {_fmt_ts(since_ms)} ({since_ms})")
    if args.until:
        print(f"Until: {_fmt_ts(until_ms)} ({until_ms})")
    else:
        print(f"Now:   {_fmt_ts(now_ms)} ({now_ms})")
    print(f"Period: {_fmt_decimal(period_days)} days")

    print("\nClosed positions (history):")
    print(f"  count: {closed_count}")
    print(f"  realized_pnl: {_fmt_decimal(closed_realized_total)} USD")
    print(f"  trade_pnl: {_fmt_decimal(trade_pnl_total)} USD")
    print(f"  funding_fees: {_fmt_decimal(funding_fees_total)} USD")
    print(f"  open_fees: {_fmt_decimal(open_fees_total)} USD")
    print(f"  close_fees: {_fmt_decimal(close_fees_total)} USD")

    print("\nOpen positions (current):")
    if historical_until:
        print("  unavailable for historical --until window")
        print("  included_positions: 0")
        print("  skipped_preexisting_positions: 0")
        print("  net_size: 0.000000")
        print("  notional: 0.000000 USD")
        print("  realized_component: 0.000000 USD")
        print("  unrealized_component: 0.000000 USD")
    else:
        print(f"  included_positions: {open_count_included}")
        print(f"  skipped_preexisting_positions: {open_count_skipped_preexisting}")
        print(f"  net_size: {_fmt_decimal(net_open_size_total)}")
        print(f"  notional: {_fmt_decimal(open_notional_total)} USD")
        print(f"  realized_component: {_fmt_decimal(open_realized_total)} USD")
        print(f"  unrealized_component: {_fmt_decimal(open_unrealized_total)} USD")

    print("\nTotals:")
    print(
        "  total_pnl="
        f"{_fmt_decimal(total_pnl)} USD "
        "(closed_realized + open_realized + open_unrealized)"
    )

    print("\nPerformance:")
    if starting_equity is None:
        print(f"  starting_equity: unavailable ({starting_equity_source})")
        print("  total_return: unavailable")
        print("  APR: unavailable")
        print("  APY: unavailable")
    else:
        print(
            f"  starting_equity: {_fmt_decimal(starting_equity)} USD "
            f"[{starting_equity_source}]"
        )
        if annualized_error:
            print("  total_return: unavailable")
            print(f"  APR: unavailable ({annualized_error})")
            print("  APY: unavailable")
        else:
            assert total_return is not None and apr is not None
            print(f"  total_return: {_fmt_pct(total_return)}")
            print(f"  APR(simple annualized): {_fmt_pct(apr)}")
            if apy is None:
                print("  APY(compounded): unavailable (1 + return <= 0)")
            else:
                print(f"  APY(compounded): {_fmt_pct(apy)}")

    all_markets = set(per_market_closed) | set(per_market_open_realized) | set(per_market_open_unrealized)
    ranked = []
    for market in all_markets:
        closed = per_market_closed[market]
        open_realized = per_market_open_realized[market]
        open_unrealized = per_market_open_unrealized[market]
        total = closed + open_realized + open_unrealized
        ranked.append((market, total, closed, open_realized, open_unrealized))
    ranked.sort(key=lambda row: abs(row[1]), reverse=True)

    print(f"\nPer-market PnL (top {args.top_markets_by_abs_pnl} by |total|):")
    for idx, row in enumerate(ranked[: args.top_markets_by_abs_pnl], start=1):
        market, total, closed, open_realized, open_unrealized = row
        print(
            f"  {idx:02d}. {market:<14} "
            f"total={_fmt_decimal(total)} "
            f"closed={_fmt_decimal(closed)} "
            f"open_realized={_fmt_decimal(open_realized)} "
            f"open_unrealized={_fmt_decimal(open_unrealized)}"
        )

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch total account PnL across all markets between --since and "
            "optional --until, and compute annualized APR/APY."
        )
    )
    parser.add_argument(
        "--since",
        required=True,
        help=(
            "Start timestamp (inclusive). Accepts epoch seconds, epoch milliseconds, "
            "or ISO-8601 (e.g. 2026-02-01T00:00:00Z)."
        ),
    )
    parser.add_argument(
        "--env",
        default=None,
        help=(
            "Env file to load (e.g. .env.amzn or amzn). "
            "Overrides process env with values from that file."
        ),
    )
    parser.add_argument(
        "--until",
        default=None,
        help=(
            "Optional end timestamp (inclusive). Accepts epoch seconds, epoch "
            "milliseconds, or ISO-8601 (e.g. 2026-02-01T12:00:00Z). "
            "Defaults to current time."
        ),
    )
    parser.add_argument(
        "--starting-equity-usd",
        default=None,
        help=(
            "Optional starting equity used for return/APR/APY calculation. "
            "If omitted, inferred as current_equity - total_pnl."
        ),
    )
    parser.add_argument(
        "--include-preexisting-open",
        action="store_true",
        help=(
            "Include currently open positions created before --since. "
            "Disabled by default for strict window semantics."
        ),
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
        "--top-markets-by-abs-pnl",
        type=int,
        default=20,
        help="How many markets to print in per-market breakdown (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
