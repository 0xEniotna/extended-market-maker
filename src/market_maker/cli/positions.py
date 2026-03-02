"""mmctl positions / close — Position risk overview and flatten."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from market_maker.cli.common import (
    PROJECT_ROOT,
    create_trading_client,
    ensure_ok,
    json_output,
    load_env_and_settings,
    resolve_market_name,
    run_async,
    to_jsonable,
)


# ---------------------------------------------------------------------------
# mmctl positions (lifted from scripts/tools/audit_position_risk.py)
# ---------------------------------------------------------------------------


def _run_positions(args) -> int:
    """Position risk audit across all markets from journal data."""
    scripts_tools = PROJECT_ROOT / "scripts" / "tools"
    if str(scripts_tools) not in sys.path:
        sys.path.insert(0, str(scripts_tools))

    from audit_position_risk import build_parser as pr_build_parser, run as pr_run

    pr_parser = pr_build_parser()
    pr_argv = []
    if hasattr(args, "lookback_hours") and args.lookback_hours is not None:
        pr_argv.extend(["--lookback-hours", str(args.lookback_hours)])
    if hasattr(args, "env_map") and args.env_map:
        pr_argv.extend(["--env-map", args.env_map])

    pr_args = pr_parser.parse_args(pr_argv)
    return pr_run(pr_args)


# ---------------------------------------------------------------------------
# mmctl close (lifted from scripts/tools/close_mm_position.py)
# ---------------------------------------------------------------------------


def _decimal_arg(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal value: {raw}") from exc


def _extract_level_price(levels: Any) -> Optional[Decimal]:
    if not levels:
        return None
    first = levels[0]
    if isinstance(first, dict):
        for key in ("price", "p"):
            if key in first:
                return Decimal(str(first[key]))
        return None
    value = getattr(first, "price", None)
    if value is not None:
        return Decimal(str(value))
    value = getattr(first, "p", None)
    if value is not None:
        return Decimal(str(value))
    return None


def _extract_top_prices(orderbook: Any) -> tuple[Optional[Decimal], Optional[Decimal]]:
    bid_levels = getattr(orderbook, "bid", None)
    ask_levels = getattr(orderbook, "ask", None)
    if isinstance(orderbook, dict):
        bid_levels = orderbook.get("bid") or orderbook.get("b")
        ask_levels = orderbook.get("ask") or orderbook.get("a")
    return _extract_level_price(bid_levels), _extract_level_price(ask_levels)


def _signed_position_for_market(positions: Iterable[Any], market: str) -> Decimal:
    from x10.perpetual.positions import PositionSide

    target = market.upper()
    signed = Decimal("0")
    for pos in positions:
        market_name = str(getattr(pos, "market", "")).upper()
        if market_name != target:
            continue
        side = getattr(pos, "side", None)
        size = Decimal(str(getattr(pos, "size", "0")))
        if side == PositionSide.LONG or str(side).upper() == "LONG":
            signed += size
        elif side == PositionSide.SHORT or str(side).upper() == "SHORT":
            signed -= size
    return signed


async def _fetch_signed_position(client: Any, market: str) -> Decimal:
    resp = await client.account.get_positions(market_names=[market])
    ensure_ok(resp, "get_positions")
    return _signed_position_for_market(resp.data or [], market)


async def _fetch_best_prices(
    client: Any, market: str, market_info: Any,
) -> tuple[Optional[Decimal], Optional[Decimal], str, Optional[str]]:
    best_bid: Optional[Decimal] = None
    best_ask: Optional[Decimal] = None
    source = "orderbook_snapshot"
    warning: Optional[str] = None

    try:
        ob_resp = await client.markets_info.get_orderbook_snapshot(market_name=market)
        ensure_ok(ob_resp, "get_orderbook_snapshot")
        best_bid, best_ask = _extract_top_prices(ob_resp.data)
    except Exception as exc:
        warning = f"orderbook_fetch_failed:{exc}"

    if best_bid is not None and best_bid <= 0:
        best_bid = None
    if best_ask is not None and best_ask <= 0:
        best_ask = None

    if best_bid is None:
        stats_bid = Decimal(str(market_info.market_stats.bid_price))
        if stats_bid > 0:
            best_bid = stats_bid
            source = "market_stats_fallback"
    if best_ask is None:
        stats_ask = Decimal(str(market_info.market_stats.ask_price))
        if stats_ask > 0:
            best_ask = stats_ask
            source = "market_stats_fallback"

    return best_bid, best_ask, source, warning


async def _run_close(args) -> int:
    from x10.perpetual.orders import OrderSide

    settings = load_env_and_settings(args.env)
    market_input = args.market.strip() if args.market else settings.market_name
    slippage_bps = args.slippage_bps if args.slippage_bps is not None else settings.shutdown_flatten_slippage_bps
    max_attempts = args.max_attempts if args.max_attempts is not None else settings.shutdown_flatten_retries
    retry_delay_s = args.retry_delay_s if args.retry_delay_s is not None else settings.shutdown_flatten_retry_delay_s

    client = create_trading_client(settings)
    from market_maker.order_manager import OrderManager

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": settings.environment.value,
        "market": None, "initial_position": None, "final_position": None,
        "status": "error", "cancel_open_orders": args.cancel_open_orders,
        "attempts": [], "dry_run": args.dry_run,
    }

    try:
        market = await resolve_market_name(client, market_input)
        payload["market"] = market

        markets = await client.markets_info.get_markets_dict()
        market_info = markets.get(market)
        if market_info is None:
            raise RuntimeError(f"Market metadata unavailable for {market}")

        tick_size = Decimal(str(market_info.trading_config.min_price_change))
        min_order_size = Decimal(str(market_info.trading_config.min_order_size))
        size_step = Decimal(str(market_info.trading_config.min_order_size_change))
        order_mgr = OrderManager(client, market)

        initial_position = await _fetch_signed_position(client, market)
        payload["initial_position"] = initial_position

        best_bid, best_ask, price_source, price_warning = await _fetch_best_prices(
            client, market, market_info
        )
        payload["price_source"] = price_source
        payload["best_bid"] = best_bid
        payload["best_ask"] = best_ask

        if args.dry_run:
            payload["status"] = "dry_run"
            payload["final_position"] = initial_position
            if initial_position == 0:
                payload["status"] = "already_flat"
            else:
                side = OrderSide.SELL if initial_position > 0 else OrderSide.BUY
                close_size = order_mgr._round_down_to_step(abs(initial_position), size_step)
                payload["attempts"].append({
                    "attempt": 1, "position_before": initial_position,
                    "side": str(side), "size": close_size, "dry_run": True,
                })
        else:
            if args.cancel_open_orders:
                try:
                    cancel_resp = await client.orders.mass_cancel(markets=[market])
                    ensure_ok(cancel_resp, "mass_cancel")
                except Exception as exc:
                    payload["cancel_open_orders_error"] = str(exc)

            for attempt in range(1, max_attempts + 1):
                position_before = await _fetch_signed_position(client, market)
                if position_before == 0:
                    payload["final_position"] = Decimal("0")
                    payload["status"] = "ok"
                    break

                best_bid, best_ask, price_source, price_warning = await _fetch_best_prices(
                    client, market, market_info
                )
                flatten_result = await order_mgr.flatten_position(
                    signed_position=position_before,
                    best_bid=best_bid, best_ask=best_ask,
                    tick_size=tick_size, min_order_size=min_order_size,
                    size_step=size_step, slippage_bps=slippage_bps,
                )
                payload["attempts"].append({
                    "attempt": attempt,
                    "position_before": position_before,
                    "flatten": {
                        "attempted": flatten_result.attempted,
                        "success": flatten_result.success,
                        "reason": flatten_result.reason,
                        "side": str(flatten_result.side) if flatten_result.side else None,
                        "size": flatten_result.size, "price": flatten_result.price,
                    },
                })
                if flatten_result.reason in {"below_min_order_size", "missing_orderbook_price"}:
                    break
                if attempt < max_attempts and retry_delay_s > 0:
                    await asyncio.sleep(retry_delay_s)
            else:
                final_position = await _fetch_signed_position(client, market)
                payload["final_position"] = final_position
                payload["status"] = "ok" if final_position == 0 else "error"
    finally:
        await client.close()

    json_output(payload, json_flag=getattr(args, "json", False),
                json_out_path=getattr(args, "json_out", None))

    if not getattr(args, "json", False):
        status = payload.get("status", "error")
        market = payload.get("market") or "-"
        initial = payload.get("initial_position")
        final = payload.get("final_position")
        attempts = len(payload.get("attempts", []))
        print(f"status={status} market={market} initial_position={initial} "
              f"final_position={final} attempts={attempts}")

    return 0 if payload.get("status") in ("ok", "dry_run", "already_flat") else 2


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def _handle_positions(args) -> int:
    return _run_positions(args)


def _handle_close(args) -> int:
    return run_async(_run_close(args))


def register(subparsers) -> None:
    # positions
    pos_parser = subparsers.add_parser("positions", help="Position risk overview across all markets")
    pos_parser.add_argument("--json", action="store_true", help="JSON output")
    pos_parser.add_argument("--lookback-hours", type=float, default=2.0, help="Lookback window")
    pos_parser.add_argument("--env-map", default=None, help="Market-to-env mapping")
    pos_parser.set_defaults(func=_handle_positions)

    # close
    close_parser = subparsers.add_parser("close", help="Flatten a market position")
    close_parser.add_argument("market", nargs="?", help="Market to flatten (default: from env)")
    close_parser.add_argument("--env", help="Env file")
    close_parser.add_argument("--dry-run", action="store_true", help="Preview without submitting")
    close_parser.add_argument("--json", action="store_true", help="JSON output")
    close_parser.add_argument("--json-out", default=None, help="Write JSON to file")
    close_parser.add_argument("--slippage-bps", type=_decimal_arg, default=None)
    close_parser.add_argument("--max-attempts", type=int, default=None)
    close_parser.add_argument("--retry-delay-s", type=float, default=None)
    close_parser.set_defaults(cancel_open_orders=True)
    close_parser.add_argument("--no-cancel-open-orders", action="store_false", dest="cancel_open_orders")
    close_parser.set_defaults(func=_handle_close)
