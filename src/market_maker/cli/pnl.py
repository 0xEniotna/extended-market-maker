"""mmctl pnl — PnL reporting (per-market, account-wide, daily scorecard)."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from market_maker.cli.common import (
    PROJECT_ROOT,
    create_trading_client,
    ensure_ok,
    fmt_decimal,
    fmt_ts,
    json_output,
    load_env_and_settings,
    resolve_market_name,
    run_async,
    to_jsonable,
)

_MS_PER_DAY = Decimal("86400000")


# ---------------------------------------------------------------------------
# Per-market PnL (lifted from scripts/tools/fetch_pnl.py)
# ---------------------------------------------------------------------------


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


def _to_ms(days: Optional[float]) -> Optional[int]:
    if days is None:
        return None
    now_ms = int(time.time() * 1000)
    return now_ms - int(days * 24 * 60 * 60 * 1000)


async def _run_market_pnl(args) -> int:
    from x10.perpetual.positions import PositionSide

    settings = load_env_and_settings(args.env)
    client = create_trading_client(settings)
    market = await resolve_market_name(client, args.market)
    since_ms = _to_ms(args.days)

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
                market_names=[market], cursor=cursor, limit=args.page_size,
            )
            ensure_ok(resp, "get_positions_history")
            rows = resp.data or []

            for row in rows:
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
        ensure_ok(pos_resp, "get_positions")
        for pos in (pos_resp.data or []):
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
            "count": stats.count, "wins": stats.wins, "losses": stats.losses,
            "realized_pnl_usd": stats.realized_total,
            "trade_pnl_usd": stats.trade_pnl_total,
            "funding_fees_usd": stats.funding_fees_total,
            "open_fees_usd": stats.open_fees_total,
            "close_fees_usd": stats.close_fees_total,
            "first_close_ms": stats.first_ts, "last_close_ms": stats.last_ts,
        },
        "open_positions": {
            "net_size": net_open_size, "notional_usd": open_notional,
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

    json_output(payload, json_flag=getattr(args, "json", False),
                json_out_path=getattr(args, "json_out", None))

    if not getattr(args, "json", False):
        print(f"PnL summary for {market}")
        print(f"Environment: {settings.environment.value}")
        if args.days is None:
            print("Window: all available history")
        else:
            print(f"Window: last {args.days} days")
        print(f"\nClosed positions (history):")
        print(f"  count: {stats.count} (wins={stats.wins} losses={stats.losses})")
        print(f"  realized_pnl: {fmt_decimal(stats.realized_total)} USD")
        print(f"  trade_pnl: {fmt_decimal(stats.trade_pnl_total)} USD")
        print(f"  funding_fees: {fmt_decimal(stats.funding_fees_total)} USD")
        print(f"  open_fees: {fmt_decimal(stats.open_fees_total)} USD")
        print(f"  close_fees: {fmt_decimal(stats.close_fees_total)} USD")
        print(f"  first_close: {fmt_ts(stats.first_ts)}")
        print(f"  last_close: {fmt_ts(stats.last_ts)}")
        print(f"\nOpen positions (current):")
        print(f"  net_size: {fmt_decimal(net_open_size)}")
        print(f"  notional: {fmt_decimal(open_notional)} USD")
        print(f"  realized_component: {fmt_decimal(open_realized)} USD")
        print(f"  unrealized_component: {fmt_decimal(open_unrealized)} USD")
        print(f"\nTotals:")
        print(f"  total_pnl_including_open={fmt_decimal(overall)} USD")

    return 0


# ---------------------------------------------------------------------------
# Account-wide PnL (lifted from scripts/tools/fetch_total_pnl.py)
# ---------------------------------------------------------------------------


def _parse_since_timestamp(raw: str) -> int:
    value = raw.strip()
    if not value:
        raise ValueError("timestamp cannot be empty")
    if value.isdigit():
        ts = int(value)
        return ts if ts >= 1_000_000_000_000 else ts * 1000

    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            "invalid timestamp. Use epoch seconds/ms or ISO-8601, "
            "e.g. 1704067200, 2026-02-01T00:00:00Z"
        ) from exc

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _fmt_pct(value: Decimal) -> str:
    return f"{(value * Decimal('100')):.4f}%"


def _fmt_ts_total(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _annualized_returns(
    *, total_pnl: Decimal, starting_equity: Decimal, period_days: Decimal,
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


async def _run_total_pnl(args) -> int:
    from x10.perpetual.positions import PositionSide

    since_ms = _parse_since_timestamp(args.since)
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    until_ms = _parse_since_timestamp(args.until) if args.until else now_ms
    if until_ms > now_ms:
        raise RuntimeError("`--until` must be earlier than current time")
    if since_ms >= until_ms:
        raise RuntimeError("`--since` must be earlier than `--until` or current time")
    period_days = Decimal(until_ms - since_ms) / _MS_PER_DAY
    historical_until = args.until is not None and until_ms < now_ms

    settings = load_env_and_settings(args.env)
    client = create_trading_client(settings)

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
    per_market_closed: Dict[str, Decimal] = defaultdict(Decimal)
    per_market_open_realized: Dict[str, Decimal] = defaultdict(Decimal)
    per_market_open_unrealized: Dict[str, Decimal] = defaultdict(Decimal)
    pages = 0
    cursor: Optional[int] = None
    seen_cursors: set[int] = set()
    current_equity: Optional[Decimal] = None

    try:
        while pages < args.max_pages:
            resp = await client.account.get_positions_history(
                market_names=None, cursor=cursor, limit=args.page_size,
            )
            ensure_ok(resp, "get_positions_history")
            rows = resp.data or []

            for row in rows:
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
            ensure_ok(pos_resp, "get_positions")
            for pos in (pos_resp.data or []):
                created_at = getattr(pos, "created_at", None)
                if (
                    created_at is not None
                    and created_at < since_ms
                    and not getattr(args, "include_preexisting_open", False)
                ):
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
            ensure_ok(bal_resp, "get_balance")
            if bal_resp.data is not None:
                current_equity = Decimal(str(bal_resp.data.equity))
    finally:
        await client.close()

    total_pnl = closed_realized_total + open_realized_total + open_unrealized_total
    starting_equity: Optional[Decimal] = None
    if current_equity is not None and not historical_until:
        starting_equity = current_equity - total_pnl

    total_return: Optional[Decimal] = None
    apr: Optional[Decimal] = None
    apy: Optional[Decimal] = None
    if starting_equity is not None and starting_equity > 0:
        try:
            total_return, apr, apy = _annualized_returns(
                total_pnl=total_pnl, starting_equity=starting_equity, period_days=period_days,
            )
        except ValueError:
            pass

    # Build per-market breakdown
    all_markets = set(per_market_closed) | set(per_market_open_realized) | set(per_market_open_unrealized)
    ranked = []
    for m in all_markets:
        closed = per_market_closed[m]
        o_real = per_market_open_realized[m]
        o_unreal = per_market_open_unrealized[m]
        total = closed + o_real + o_unreal
        ranked.append((m, total, closed, o_real, o_unreal))
    ranked.sort(key=lambda r: abs(r[1]), reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": settings.environment.value,
        "since_ms": since_ms,
        "until_ms": until_ms,
        "period_days": period_days,
        "closed_count": closed_count,
        "closed_realized_pnl_usd": closed_realized_total,
        "open_realized_pnl_usd": open_realized_total,
        "open_unrealized_pnl_usd": open_unrealized_total,
        "total_pnl_usd": total_pnl,
        "starting_equity_usd": starting_equity,
        "current_equity_usd": current_equity,
        "total_return": total_return,
        "apr": apr,
        "apy": apy,
        "per_market": [
            {"market": m, "total": t, "closed": c, "open_realized": o_r, "open_unrealized": o_u}
            for m, t, c, o_r, o_u in ranked[:getattr(args, "top_markets", 20)]
        ],
    }

    json_output(payload, json_flag=getattr(args, "json", False))

    if not getattr(args, "json", False):
        print("PnL summary across all markets")
        print(f"Environment: {settings.environment.value}")
        print(f"Since: {_fmt_ts_total(since_ms)}")
        print(f"Period: {fmt_decimal(period_days)} days")
        print(f"\nClosed positions: {closed_count}")
        print(f"  realized_pnl: {fmt_decimal(closed_realized_total)} USD")
        print(f"  trade_pnl: {fmt_decimal(trade_pnl_total)} USD")
        print(f"  funding_fees: {fmt_decimal(funding_fees_total)} USD")
        print(f"\nOpen positions: {open_count_included}")
        print(f"  realized: {fmt_decimal(open_realized_total)} USD")
        print(f"  unrealized: {fmt_decimal(open_unrealized_total)} USD")
        print(f"\nTotal PnL: {fmt_decimal(total_pnl)} USD")
        if total_return is not None:
            print(f"  return: {_fmt_pct(total_return)}")
        if apr is not None:
            print(f"  APR: {_fmt_pct(apr)}")
        if apy is not None:
            print(f"  APY: {_fmt_pct(apy)}")

        print(f"\nPer-market (top {min(20, len(ranked))} by |total|):")
        for idx, (m, total, closed, o_r, o_u) in enumerate(ranked[:20], start=1):
            print(
                f"  {idx:02d}. {m:<14} total={fmt_decimal(total)} "
                f"closed={fmt_decimal(closed)} open_r={fmt_decimal(o_r)} open_u={fmt_decimal(o_u)}"
            )

    return 0


# ---------------------------------------------------------------------------
# Scorecard (lifted from scripts/tools/audit_daily_scorecard.py)
# ---------------------------------------------------------------------------


def _run_scorecard(args) -> int:
    """Run the daily scorecard — delegates to the existing module."""
    import sys

    # The scorecard has complex journal analysis logic. Import and call it.
    scripts_tools = PROJECT_ROOT / "scripts" / "tools"
    if str(scripts_tools) not in sys.path:
        sys.path.insert(0, str(scripts_tools))

    from audit_daily_scorecard import build_parser as sc_build_parser, run as sc_run

    # Map mmctl args to scorecard args
    sc_parser = sc_build_parser()
    sc_argv = []
    if hasattr(args, "lookback_hours") and args.lookback_hours is not None:
        sc_argv.extend(["--lookback-hours", str(args.lookback_hours)])
    if hasattr(args, "env_map") and args.env_map:
        sc_argv.extend(["--env-map", args.env_map])

    sc_args = sc_parser.parse_args(sc_argv)
    return sc_run(sc_args)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def _handle_pnl(args) -> int:
    if getattr(args, "scorecard", False):
        return _run_scorecard(args)
    if getattr(args, "all", False):
        if not args.since:
            print("error: --since is required with --all", file=__import__("sys").stderr)
            return 1
        return run_async(_run_total_pnl(args))
    if not args.market:
        print("error: market argument required (or use --all/--scorecard)", file=__import__("sys").stderr)
        return 1
    return run_async(_run_market_pnl(args))


def register(subparsers) -> None:
    pnl_parser = subparsers.add_parser("pnl", help="PnL reporting")
    pnl_parser.add_argument("market", nargs="?", help="Market symbol (e.g. ETH-USD)")
    pnl_parser.add_argument("--all", action="store_true", help="Account-wide PnL across all markets")
    pnl_parser.add_argument("--scorecard", action="store_true", help="Daily fleet scorecard")
    pnl_parser.add_argument("--since", help="Start timestamp (ISO-8601 or epoch)")
    pnl_parser.add_argument("--until", default=None, help="End timestamp (optional)")
    pnl_parser.add_argument("--days", type=float, default=None, help="Lookback window in days")
    pnl_parser.add_argument("--env", default=None, help="Env file (e.g. .env.eth or eth)")
    pnl_parser.add_argument("--json", action="store_true", help="JSON output")
    pnl_parser.add_argument("--json-out", default=None, help="Write JSON to file")
    pnl_parser.add_argument("--page-size", type=int, default=200, help="History page size")
    pnl_parser.add_argument("--max-pages", type=int, default=100, help="Max history pages")
    pnl_parser.add_argument("--top-markets", type=int, default=20, help="Top N markets in breakdown")
    pnl_parser.add_argument("--include-preexisting-open", action="store_true")
    pnl_parser.add_argument("--lookback-hours", type=float, default=24.0, help="Scorecard lookback")
    pnl_parser.add_argument("--env-map", default=None, help="Market-to-env mapping for scorecard")
    pnl_parser.set_defaults(func=_handle_pnl)
