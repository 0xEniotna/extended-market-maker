"""
Position Flattener

Encapsulates the shutdown-flatten logic previously inlined in
``OrderManager.flatten_position``.  Handles one-sided book fallbacks,
progressive slippage, and fill-verification polling.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Optional

from x10.perpetual.orders import OrderSide, OrderType, TimeInForce

logger = logging.getLogger(__name__)


@dataclass
class FlattenResult:
    """Result of a shutdown flatten attempt."""

    attempted: bool
    success: bool
    reason: str
    side: Optional[OrderSide] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    remaining_position: Optional[Decimal] = None


def round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def round_to_tick_for_side(price: Decimal, tick_size: Decimal, side: OrderSide) -> Decimal:
    if tick_size <= 0:
        return price
    rounding = ROUND_UP if side == OrderSide.BUY else ROUND_DOWN
    return (price / tick_size).to_integral_value(rounding=rounding) * tick_size


def extract_exchange_id(resp) -> Optional[str]:
    """Extract the exchange order ID from the SDK response."""
    data = resp.data if hasattr(resp, "data") else resp
    if hasattr(data, "id") and data.id is not None:
        return str(data.id)
    if isinstance(data, dict):
        for key in ("id", "order_id", "orderId"):
            if key in data and data[key] is not None:
                return str(data[key])
    return None


async def _wait_for_position_change(
    *,
    risk_mgr: Any,
    initial_position: Decimal,
    timeout_s: float,
    poll_interval_s: float = 0.25,
) -> Decimal:
    """Poll risk_mgr position until it differs from *initial_position* or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current: Decimal = risk_mgr.get_current_position()
        if current != initial_position:
            logger.info(
                "Flatten fill confirmed: position moved %s -> %s",
                initial_position,
                current,
            )
            return current
        await asyncio.sleep(poll_interval_s)

    current = risk_mgr.get_current_position()
    if current == initial_position:
        logger.warning(
            "Flatten wait timed out after %.1fs: position still %s",
            timeout_s,
            current,
        )
    return Decimal(str(current))


async def flatten_position(
    client,
    market_name: str,
    fee_resolver,
    *,
    signed_position: Decimal,
    best_bid: Optional[Decimal],
    best_ask: Optional[Decimal],
    tick_size: Decimal,
    min_order_size: Decimal,
    size_step: Decimal,
    slippage_bps: Decimal,
    risk_mgr=None,
    wait_for_fill_s: float = 0,
    last_known_mid: Optional[Decimal] = None,
    rate_limit_hit_callback=None,
    cancel_all_callback=None,
) -> FlattenResult:
    """Submit a reduce-only MARKET+IOC order to flatten a signed position.

    When *risk_mgr* and *wait_for_fill_s* > 0 are given, poll the risk
    manager's cached position after submission to verify the fill actually
    reduced the position.

    One-sided book handling:
    - If the natural BBO side is missing, use the available side with extra
      slippage (``slippage_bps`` is applied to whichever price is used).
    - If both sides are missing, fall back to ``last_known_mid`` with 50 bps
      slippage.
    - Only give up if no price reference exists at all.
    """
    if signed_position == 0:
        return FlattenResult(
            attempted=False,
            success=True,
            reason="already_flat",
            remaining_position=Decimal("0"),
        )

    side = OrderSide.SELL if signed_position > 0 else OrderSide.BUY
    close_size = round_down_to_step(abs(signed_position), size_step)
    if close_size < min_order_size:
        logger.warning(
            "Skipping flatten for market=%s: |position|=%s rounds to size=%s < min_order_size=%s",
            market_name,
            signed_position,
            close_size,
            min_order_size,
        )
        return FlattenResult(
            attempted=False,
            success=False,
            reason="below_min_order_size",
            side=side,
            size=close_size,
        )

    # --- Price resolution with one-sided book fallback ---
    ref_price = best_bid if side == OrderSide.SELL else best_ask
    effective_slippage = slippage_bps
    if ref_price is None or ref_price <= 0:
        ref_price = best_ask if side == OrderSide.SELL else best_bid
        if ref_price is not None and ref_price > 0:
            logger.warning(
                "Flatten using opposite-side BBO for market=%s: "
                "natural side missing, using ref_price=%s with slippage=%sbps",
                market_name,
                ref_price,
                effective_slippage,
            )
    if ref_price is None or ref_price <= 0:
        if last_known_mid is not None and last_known_mid > 0:
            ref_price = last_known_mid
            effective_slippage = max(slippage_bps, Decimal("50"))
            logger.warning(
                "Flatten using last known mid for market=%s: "
                "both BBO sides missing, ref_price=%s slippage=%sbps",
                market_name,
                ref_price,
                effective_slippage,
            )
        else:
            logger.error(
                "Cannot flatten position for market=%s: "
                "no price reference at all (bid=%s ask=%s last_mid=%s)",
                market_name,
                best_bid,
                best_ask,
                last_known_mid,
            )
            return FlattenResult(
                attempted=False,
                success=False,
                reason="missing_orderbook_price",
                side=side,
                size=close_size,
            )

    bps = max(Decimal("0"), effective_slippage) / Decimal("10000")
    if side == OrderSide.SELL:
        target_price = ref_price * (Decimal("1") - bps)
    else:
        target_price = ref_price * (Decimal("1") + bps)
    price = round_to_tick_for_side(target_price, tick_size, side)
    if price <= 0:
        price = tick_size if tick_size > 0 else Decimal("1")

    external_id = f"mm-flat-{uuid.uuid4().hex[:12]}"
    fee_cfg = None
    if fee_resolver is not None:
        fee_cfg = await fee_resolver.resolve_order_fees(
            post_only=False,
            fail_closed=False,
        )

    try:
        from x10.utils.http import RateLimitException
    except Exception:
        class RateLimitException(Exception):  # type: ignore[no-redef]
            pass

    try:
        resp = await client.place_order(
            market_name=market_name,
            amount_of_synthetic=close_size,
            price=price,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            post_only=False,
            reduce_only=True,
            external_id=external_id,
            max_fee_rate=fee_cfg.max_fee_rate if fee_cfg is not None else None,
            builder_fee_rate=fee_cfg.builder_fee_rate if fee_cfg is not None else None,
            builder_id=fee_cfg.builder_id if fee_cfg is not None else None,
        )
        if hasattr(resp, "status") and hasattr(resp, "error"):
            if resp.status != "OK" or resp.error is not None:
                logger.error(
                    "Flatten order rejected: market=%s side=%s size=%s price=%s error=%s",
                    market_name,
                    side,
                    close_size,
                    price,
                    resp.error,
                )
                return FlattenResult(
                    attempted=True,
                    success=False,
                    reason=f"rejected:{resp.error}",
                    side=side,
                    size=close_size,
                    price=price,
                )
        logger.warning(
            "Submitted shutdown flatten order: market=%s side=%s size=%s price=%s ext_id=%s",
            market_name,
            side,
            close_size,
            price,
            external_id,
        )

        remaining_position: Optional[Decimal] = None
        if risk_mgr is not None and wait_for_fill_s > 0:
            remaining_position = await _wait_for_position_change(
                risk_mgr=risk_mgr,
                initial_position=signed_position,
                timeout_s=wait_for_fill_s,
            )

        return FlattenResult(
            attempted=True,
            success=True,
            reason="submitted",
            side=side,
            size=close_size,
            price=price,
            remaining_position=remaining_position,
        )
    except RateLimitException as exc:
        if rate_limit_hit_callback is not None:
            should_halt = rate_limit_hit_callback()
            if should_halt and cancel_all_callback is not None:
                asyncio.create_task(cancel_all_callback())
        logger.exception(
            "Rate-limited flatten order submit: market=%s side=%s size=%s price=%s",
            market_name,
            side,
            close_size,
            price,
        )
        return FlattenResult(
            attempted=True,
            success=False,
            reason=f"rate_limited:{exc}",
            side=side,
            size=close_size,
            price=price,
        )
    except Exception as exc:
        logger.exception(
            "Failed to submit shutdown flatten order: market=%s side=%s size=%s price=%s",
            market_name,
            side,
            close_size,
            price,
        )
        return FlattenResult(
            attempted=True,
            success=False,
            reason=f"exception:{exc}",
            side=side,
            size=close_size,
            price=price,
        )
