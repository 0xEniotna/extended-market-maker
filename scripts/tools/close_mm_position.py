#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from x10.perpetual.accounts import StarkPerpetualAccount  # noqa: E402
from x10.perpetual.orders import OrderSide  # noqa: E402
from x10.perpetual.positions import PositionSide  # noqa: E402
from x10.perpetual.trading_client import PerpetualTradingClient  # noqa: E402
from x10.utils.http import ResponseStatus  # noqa: E402

from market_maker.config import MarketMakerSettings  # noqa: E402
from market_maker.order_manager import OrderManager  # noqa: E402


def _decimal_arg(raw: str) -> Decimal:
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid decimal value: {raw}") from exc


def _resolve_env_file(env_value: str) -> Path:
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


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _ensure_ok(resp, label: str) -> None:
    if resp.status != ResponseStatus.OK:
        raise RuntimeError(f"{label} failed: status={resp.status} error={resp.error}")


async def _resolve_market_name(client: PerpetualTradingClient, requested_market: str) -> str:
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

    best_bid = _extract_level_price(bid_levels)
    best_ask = _extract_level_price(ask_levels)
    return best_bid, best_ask


def _signed_position_for_market(positions: Iterable[Any], market: str) -> Decimal:
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


async def _fetch_signed_position(client: PerpetualTradingClient, market: str) -> Decimal:
    resp = await client.account.get_positions(market_names=[market])
    _ensure_ok(resp, "get_positions")
    return _signed_position_for_market(resp.data or [], market)


async def _fetch_best_prices(client: PerpetualTradingClient, market: str, market_info) -> tuple[Optional[Decimal], Optional[Decimal], str, Optional[str]]:
    best_bid: Optional[Decimal] = None
    best_ask: Optional[Decimal] = None
    source = "orderbook_snapshot"
    warning: Optional[str] = None

    try:
        ob_resp = await client.markets_info.get_orderbook_snapshot(market_name=market)
        _ensure_ok(ob_resp, "get_orderbook_snapshot")
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


def _preview_flatten(
    *,
    order_mgr: OrderManager,
    signed_position: Decimal,
    best_bid: Optional[Decimal],
    best_ask: Optional[Decimal],
    tick_size: Decimal,
    min_order_size: Decimal,
    size_step: Decimal,
    slippage_bps: Decimal,
) -> dict[str, Any]:
    if signed_position == 0:
        return {
            "attempted": False,
            "success": True,
            "reason": "already_flat",
            "side": None,
            "size": Decimal("0"),
            "price": None,
        }

    side = OrderSide.SELL if signed_position > 0 else OrderSide.BUY
    close_size = order_mgr._round_down_to_step(abs(signed_position), size_step)
    if close_size < min_order_size:
        return {
            "attempted": False,
            "success": False,
            "reason": "below_min_order_size",
            "side": str(side),
            "size": close_size,
            "price": None,
        }

    ref_price = best_bid if side == OrderSide.SELL else best_ask
    if ref_price is None or ref_price <= 0:
        ref_price = best_ask if side == OrderSide.SELL else best_bid
    if ref_price is None or ref_price <= 0:
        return {
            "attempted": False,
            "success": False,
            "reason": "missing_orderbook_price",
            "side": str(side),
            "size": close_size,
            "price": None,
        }

    bps = max(Decimal("0"), slippage_bps) / Decimal("10000")
    if side == OrderSide.SELL:
        target_price = ref_price * (Decimal("1") - bps)
    else:
        target_price = ref_price * (Decimal("1") + bps)

    price = order_mgr._round_to_tick_for_side(target_price, tick_size, side)
    if price <= 0:
        price = tick_size if tick_size > 0 else Decimal("1")

    return {
        "attempted": True,
        "success": True,
        "reason": "preview",
        "side": str(side),
        "size": close_size,
        "price": price,
    }


async def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
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

    market_input = args.market.strip() if args.market else settings.market_name
    slippage_bps = args.slippage_bps if args.slippage_bps is not None else settings.shutdown_flatten_slippage_bps
    max_attempts = args.max_attempts if args.max_attempts is not None else settings.shutdown_flatten_retries
    retry_delay_s = args.retry_delay_s if args.retry_delay_s is not None else settings.shutdown_flatten_retry_delay_s

    if max_attempts < 1:
        raise RuntimeError("max_attempts must be >= 1")
    if retry_delay_s < 0:
        raise RuntimeError("retry_delay_s must be >= 0")

    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    client = PerpetualTradingClient(settings.endpoint_config, account)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": settings.environment.value,
        "market": None,
        "initial_position": None,
        "final_position": None,
        "status": "error",
        "cancel_open_orders": args.cancel_open_orders,
        "cancel_open_orders_error": None,
        "attempts": [],
        "dry_run": args.dry_run,
    }

    try:
        market = await _resolve_market_name(client, market_input)
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

        best_bid, best_ask, price_source, price_warning = await _fetch_best_prices(client, market, market_info)
        payload["price_source"] = price_source
        payload["price_warning"] = price_warning
        payload["best_bid"] = best_bid
        payload["best_ask"] = best_ask

        if args.dry_run:
            payload["attempts"].append({
                "attempt": 1,
                "position_before": initial_position,
                "flatten": _preview_flatten(
                    order_mgr=order_mgr,
                    signed_position=initial_position,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    tick_size=tick_size,
                    min_order_size=min_order_size,
                    size_step=size_step,
                    slippage_bps=slippage_bps,
                ),
                "price_source": price_source,
                "price_warning": price_warning,
            })
            payload["final_position"] = initial_position
            payload["status"] = "dry_run"
            return 0, payload

        if args.cancel_open_orders:
            try:
                cancel_resp = await client.orders.mass_cancel(markets=[market])
                _ensure_ok(cancel_resp, "mass_cancel")
            except Exception as exc:
                payload["cancel_open_orders_error"] = str(exc)
                if args.fail_on_cancel_error:
                    payload["status"] = "error"
                    payload["final_position"] = initial_position
                    return 2, payload

        for attempt in range(1, max_attempts + 1):
            position_before = await _fetch_signed_position(client, market)
            if position_before == 0:
                payload["final_position"] = Decimal("0")
                payload["status"] = "ok"
                return 0, payload

            best_bid, best_ask, price_source, price_warning = await _fetch_best_prices(client, market, market_info)
            flatten_result = await order_mgr.flatten_position(
                signed_position=position_before,
                best_bid=best_bid,
                best_ask=best_ask,
                tick_size=tick_size,
                min_order_size=min_order_size,
                size_step=size_step,
                slippage_bps=slippage_bps,
            )

            payload["attempts"].append({
                "attempt": attempt,
                "position_before": position_before,
                "price_source": price_source,
                "price_warning": price_warning,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "flatten": {
                    "attempted": flatten_result.attempted,
                    "success": flatten_result.success,
                    "reason": flatten_result.reason,
                    "side": str(flatten_result.side) if flatten_result.side is not None else None,
                    "size": flatten_result.size,
                    "price": flatten_result.price,
                },
            })

            if flatten_result.reason in {"below_min_order_size", "missing_orderbook_price"}:
                break

            if attempt < max_attempts and retry_delay_s > 0:
                await asyncio.sleep(retry_delay_s)

        final_position = await _fetch_signed_position(client, market)
        payload["final_position"] = final_position
        if final_position == 0:
            payload["status"] = "ok"
            return 0, payload

        submitted = any(
            bool(row.get("flatten", {}).get("success"))
            for row in payload["attempts"]
        )
        if submitted and args.allow_submitted:
            payload["status"] = "submitted_not_confirmed_flat"
            return 0, payload

        payload["status"] = "error"
        return 2, payload

    finally:
        await client.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Close one market position using reduce-only MARKET+IOC orders. "
            "Designed for automation/agent usage."
        )
    )
    parser.add_argument(
        "--env",
        help="Env file path or short name (e.g. .env.eth or env.eth).",
    )
    parser.add_argument(
        "--market",
        help="Market to flatten (default: MM_MARKET_NAME from env).",
    )
    parser.add_argument(
        "--slippage-bps",
        type=_decimal_arg,
        default=None,
        help="Flatten slippage in bps (default from MM_SHUTDOWN_FLATTEN_SLIPPAGE_BPS).",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Flatten retries (default from MM_SHUTDOWN_FLATTEN_RETRIES).",
    )
    parser.add_argument(
        "--retry-delay-s",
        type=float,
        default=None,
        help="Delay between retries in seconds (default from MM_SHUTDOWN_FLATTEN_RETRY_DELAY_S).",
    )
    parser.set_defaults(cancel_open_orders=True)
    parser.add_argument(
        "--cancel-open-orders",
        action="store_true",
        dest="cancel_open_orders",
        help="Cancel resting orders for this market before flattening (default: enabled).",
    )
    parser.add_argument(
        "--no-cancel-open-orders",
        action="store_false",
        dest="cancel_open_orders",
        help="Do not mass-cancel resting orders before flattening.",
    )
    parser.add_argument(
        "--fail-on-cancel-error",
        action="store_true",
        help="Fail immediately if market mass-cancel fails.",
    )
    parser.add_argument(
        "--allow-submitted",
        action="store_true",
        help="Return success when a flatten order was submitted but flat position is not yet confirmed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the flatten plan without submitting any order.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write JSON result payload.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print JSON payload to stdout.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        rc, payload = asyncio.run(_run(args))
    except Exception as exc:
        payload = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "error",
            "error": str(exc),
        }
        rc = 1

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n")

    if args.json_stdout:
        print(json.dumps(_to_jsonable(payload), indent=2))
    else:
        status = payload.get("status", "error")
        market = payload.get("market") or "-"
        initial = payload.get("initial_position")
        final = payload.get("final_position")
        attempts = len(payload.get("attempts", [])) if isinstance(payload.get("attempts"), list) else 0
        print(
            f"status={status} market={market} initial_position={initial} "
            f"final_position={final} attempts={attempts}"
        )
        if payload.get("error"):
            print(f"error={payload['error']}")
        if payload.get("cancel_open_orders_error"):
            print(f"cancel_open_orders_error={payload['cancel_open_orders_error']}")

    raise SystemExit(rc)


if __name__ == "__main__":
    main()
