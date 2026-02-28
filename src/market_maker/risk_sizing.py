"""
Risk Sizing

Extracted from ``RiskManager``: the ``allowed_order_size`` logic and its
helper methods for balance-aware sizing, gross exposure checks, and
per-side position limits.
"""
from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal
from typing import Optional

from x10.perpetual.orders import OrderSide

logger = logging.getLogger(__name__)


def get_orderbook_mid_price(risk: object) -> Optional[Decimal]:
    """Fetch mid price from the orderbook manager if available and fresh."""
    ob = getattr(risk, "_orderbook_mgr", None)
    if ob is None:
        return None
    if ob.is_stale():
        return None
    bid = ob.best_bid()
    ask = ob.best_ask()
    if bid is None or ask is None:
        return None
    if bid.price <= 0 or ask.price <= 0:
        return None
    return Decimal(str((bid.price + ask.price) / 2))


def is_balance_stale(risk: object) -> bool:
    """Return True if cached balance is older than staleness threshold."""
    import time
    max_s = getattr(risk, "_balance_staleness_max_s", 0.0)
    if max_s <= 0:
        return False
    updated_at = getattr(risk, "_cached_balance_updated_at", None)
    if updated_at is None:
        return True
    return (time.monotonic() - updated_at) > max_s


def effective_max_position_for_side(risk: object, side: OrderSide) -> Decimal:
    """Return the effective max position size for the given side."""
    from .risk_manager import RiskManager
    is_buy = RiskManager._is_buy_side(side)
    max_long = getattr(risk, "_max_long_position_size", Decimal("0"))
    max_short = getattr(risk, "_max_short_position_size", Decimal("0"))
    max_pos = getattr(risk, "_max_position_size", Decimal("0"))
    if is_buy and max_long > 0:
        return max_long
    if not is_buy and max_short > 0:
        return max_short
    return max_pos


def allowed_order_size(
    risk: object,
    side: OrderSide,
    requested_size: Decimal,
    reference_price: Decimal,
    reserved_same_side_qty: Decimal = Decimal("0"),
    reserved_open_notional_usd: Decimal = Decimal("0"),
) -> Decimal:
    """Return the maximum safe size that can be placed for this order.

    Clips by position size limit, per-order notional cap, total position
    notional cap, gross exposure limit, reserved same-side quantity,
    in-flight order notional, and account available_for_trade headroom.
    """
    if requested_size <= 0:
        return Decimal("0")

    ob_mid = get_orderbook_mid_price(risk)
    if ob_mid is not None:
        reference_price = ob_mid
    elif getattr(risk, "_orderbook_mgr", None) is not None and risk._orderbook_mgr.is_stale():
        logger.info("Order size zeroed: orderbook is stale, refusing to size")
        return Decimal("0")

    clipped = requested_size
    current = risk._cached_position
    reserved_same_side_qty = max(Decimal("0"), reserved_same_side_qty)
    reserved_open_notional_usd = max(Decimal("0"), reserved_open_notional_usd)
    inflight_notional = risk.total_inflight_notional()

    # Per-side position size headroom.
    effective_max = effective_max_position_for_side(risk, side)
    if effective_max > 0:
        if side == OrderSide.BUY:
            qty_headroom = effective_max - current - reserved_same_side_qty
        else:
            qty_headroom = effective_max + current - reserved_same_side_qty
        clipped = min(clipped, max(Decimal("0"), qty_headroom))

    if reference_price > 0:
        # Per-order notional cap.
        max_order_notional = getattr(risk, "_max_order_notional_usd", Decimal("0"))
        if max_order_notional > 0:
            per_order_max_size = max_order_notional / reference_price
            clipped = min(clipped, max(Decimal("0"), per_order_max_size))

        # Absolute position notional cap.
        max_pos_notional = getattr(risk, "_max_position_notional_usd", Decimal("0"))
        if max_pos_notional > 0:
            current_notional = abs(current) * reference_price
            reserved_notional = reserved_same_side_qty * reference_price
            remaining_notional = max_pos_notional - current_notional - reserved_notional
            if remaining_notional <= 0:
                clipped = Decimal("0")
            else:
                clipped = min(clipped, max(Decimal("0"), remaining_notional / reference_price))

        # Gross exposure limit.
        gross_limit = getattr(risk, "_gross_exposure_limit_usd", Decimal("0"))
        if gross_limit > 0:
            position_notional = abs(current) * reference_price
            total_order_notional = reserved_open_notional_usd + inflight_notional
            current_gross = position_notional + total_order_notional
            gross_headroom = gross_limit - current_gross
            if gross_headroom <= 0:
                clipped = Decimal("0")
            else:
                clipped = min(clipped, max(Decimal("0"), gross_headroom / reference_price))

        from .risk_manager import RiskManager
        reducing_qty, opening_qty = RiskManager._split_reducing_and_opening_qty(
            side=side, current_position=current, size=clipped,
        )

        # Balance-aware sizing (with staleness guard).
        balance_available = risk._cached_available_for_trade
        balance_stale_flag = is_balance_stale(risk)
        balance_enabled = getattr(risk, "_balance_aware_sizing_enabled", True)
        if balance_stale_flag and balance_enabled:
            stale_action = getattr(risk, "_balance_stale_action", "reduce")
            if stale_action == "halt":
                opening_qty = Decimal("0")
                clipped = reducing_qty + opening_qty
            elif stale_action == "reduce":
                opening_qty = (opening_qty / 2).quantize(
                    Decimal("1"), rounding=ROUND_DOWN
                ) if opening_qty > 0 else opening_qty
                clipped = reducing_qty + opening_qty
            balance_available = None
        elif balance_stale_flag:
            balance_available = None

        if (
            balance_enabled
            and balance_available is not None
            and opening_qty > 0
        ):
            usage_factor = getattr(risk, "_balance_usage_factor", Decimal("0.95"))
            min_avail = getattr(risk, "_balance_min_available_usd", Decimal("0"))
            notional_mult = getattr(risk, "_balance_notional_multiplier", Decimal("1.0"))
            balance_headroom = (balance_available * usage_factor) - min_avail
            notional_headroom = (
                balance_headroom * notional_mult
            ) - reserved_open_notional_usd - inflight_notional
            if notional_headroom <= 0:
                opening_qty = Decimal("0")
            else:
                max_size_from_balance = notional_headroom / reference_price
                opening_qty = min(opening_qty, max(Decimal("0"), max_size_from_balance))
            clipped = reducing_qty + opening_qty

    clipped = max(Decimal("0"), clipped)
    if clipped < requested_size:
        reducing_qty, opening_qty = RiskManager._split_reducing_and_opening_qty(
            side=side, current_position=current, size=clipped,
        )
        logger.info(
            "Order size clipped: side=%s requested=%s allowed=%s reducing=%s opening=%s "
            "reserved_qty=%s reserved_notional=%s inflight_notional=%s "
            "avail_for_trade=%s ref_price=%s",
            side, requested_size, clipped, reducing_qty, opening_qty,
            reserved_same_side_qty, reserved_open_notional_usd, inflight_notional,
            risk._cached_available_for_trade, reference_price,
        )
    return clipped
