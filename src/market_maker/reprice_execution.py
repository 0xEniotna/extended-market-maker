"""
Reprice Execution

Extracted from ``RepricePipeline``: the order construction and placement
phase of the reprice cycle.  Handles risk-adjusted sizing, post-only
safety, and the cancel-then-place flow.
"""
from __future__ import annotations

import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from .decision_models import RepriceMarketContext
from .types import StrategyContext

if TYPE_CHECKING:
    from .reprice_pipeline import RepricePipeline, RiskAdjustedOrder


def compute_risk_adjusted_order(
    pipeline: RepricePipeline,
    strategy: StrategyContext,
    side,
    level: int,
    *,
    level_ctx: Any,
    quote_inputs: Any,
    market_ctx: RepriceMarketContext,
) -> RiskAdjustedOrder:
    """Compute target price and risk-clipped size for a level."""
    from .reprice_pipeline import RiskAdjustedOrder

    target_price = pipeline._pricing.compute_target_price(
        side,
        level,
        quote_inputs.current_best,
        extra_offset_bps=quote_inputs.extra_offset_bps,
        regime_scale=market_ctx.regime.offset_scale,
        trend=market_ctx.trend,
        funding_bias_bps=market_ctx.funding_bias_bps,
    )

    requested_size = pipeline._pricing.level_size(level)
    counter_side = strategy._counter_trend_side(market_ctx.trend)
    if (
        strategy._settings.market_profile == "crypto"
        and counter_side is not None
        and level_ctx.side_name == counter_side
    ):
        inventory_override = pipeline._inventory_override_active(strategy, side, market_ctx)
        if inventory_override:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="allow_trend_size_inventory_override",
                current_best=quote_inputs.current_best,
                target_price=target_price,
                **pipeline._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=quote_inputs.spread_bps,
                    extra_offset_bps=quote_inputs.extra_offset_bps,
                ),
            )
        else:
            if market_ctx.trend.strength >= strategy._settings.trend_strong_threshold and level == 0:
                requested_size = Decimal("0")
            else:
                cut = strategy._settings.trend_counter_side_size_cut * market_ctx.trend.strength
                requested_size = strategy._quantize_size(
                    requested_size * max(Decimal("0"), Decimal("1") - cut)
                )

    reserved_same_side_qty, reserved_open_notional_usd = strategy._orders.reserved_exposure(
        side=side,
        exclude_external_id=level_ctx.prev_ext_id,
    )
    level_size = strategy._quantize_size(
        strategy._risk.allowed_order_size(
            side,
            requested_size,
            target_price,
            reserved_same_side_qty=reserved_same_side_qty,
            reserved_open_notional_usd=reserved_open_notional_usd,
        )
    )
    return RiskAdjustedOrder(
        target_price=target_price,
        requested_size=requested_size,
        level_size=level_size,
    )


async def execute_replace_if_needed(
    pipeline: RepricePipeline,
    strategy: StrategyContext,
    side,
    level: int,
    *,
    level_ctx: Any,
    quote_inputs: Any,
    order_plan: Any,
    market_ctx: RepriceMarketContext,
) -> None:
    """Cancel-then-place with post-only safety and risk checks."""
    if order_plan.level_size < strategy._market_min_order_size:
        if level_ctx.prev_ext_id is not None:
            await strategy._cancel_level_order(
                key=level_ctx.key,
                external_id=level_ctx.prev_ext_id,
                side=side,
                level=level,
                reason="risk_limit",
            )
        return

    if level_ctx.prev_ext_id is not None:
        cancelled = await strategy._cancel_level_order(
            key=level_ctx.key,
            external_id=level_ctx.prev_ext_id,
            side=side,
            level=level,
            reason="reprice",
        )
        if not cancelled:
            return
        if strategy._orders.in_rate_limit_degraded:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="cancel_only_rate_limit_degraded",
                **pipeline._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=quote_inputs.spread_bps,
                    extra_offset_bps=quote_inputs.extra_offset_bps,
                ),
            )
            return

    if strategy._ob.is_stale():
        return
    fresh_bid = strategy._ob.best_bid()
    fresh_ask = strategy._ob.best_ask()
    if fresh_bid is None or fresh_ask is None:
        return

    safe_target_price = strategy._apply_post_only_safety(
        side=side,
        target_price=order_plan.target_price,
        bid_price=fresh_bid.price,
        ask_price=fresh_ask.price,
        safety_ticks=strategy._effective_safety_ticks(level_ctx.key),
    )
    if safe_target_price is None:
        return

    ext_id = await strategy._orders.place_order(
        side=side,
        price=safe_target_price,
        size=order_plan.level_size,
        level=level,
    )
    if ext_id is None:
        return

    strategy._on_successful_quote(level_ctx.key)
    strategy._level_ext_ids[level_ctx.key] = ext_id
    strategy._level_order_created_at[level_ctx.key] = time.monotonic()
    strategy._level_last_reprice_at[level_ctx.key] = time.monotonic()
    order_info = strategy._orders.get_active_order(ext_id)
    strategy._journal.record_order_placed(
        external_id=ext_id,
        exchange_id=order_info.exchange_order_id if order_info else None,
        side=strategy._normalise_side(str(side)),
        price=safe_target_price,
        size=order_plan.level_size,
        level=level,
        best_bid=fresh_bid.price if fresh_bid else None,
        best_ask=fresh_ask.price if fresh_ask else None,
        position=strategy._risk.get_current_position(),
    )
    qtr = getattr(strategy, "_qtr", None)
    if qtr is not None:
        qtr.record_quote()
