from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

from .decision_models import GuardDecision, RegimeState, RepriceMarketContext, TrendState
from .reprice_execution import (
    compute_risk_adjusted_order as _compute_risk_adjusted,
)
from .reprice_execution import (
    execute_replace_if_needed as _execute_replace,
)
from .types import StrategyContext


@dataclass
class LevelContext:
    key: tuple[str, int]
    side_name: str
    prev_ext_id: Optional[str]
    prev_order: Any = None


@dataclass(frozen=True)
class QuoteInputs:
    bid: Any
    ask: Any
    spread_bps: Optional[Decimal]
    extra_offset_bps: Decimal
    current_best: Decimal


@dataclass(frozen=True)
class RiskAdjustedOrder:
    target_price: Decimal
    requested_size: Decimal
    level_size: Decimal


class RepricePipeline:
    """Target shift and tolerance gates for cancel/replace decisions."""

    def __init__(self, settings, tick_size: Decimal, pricing_engine) -> None:
        self._settings = settings
        self._tick_size = tick_size
        self._pricing = pricing_engine

    def needs_reprice(
        self,
        side,
        prev_price: Decimal,
        current_best: Decimal,
        level: int,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend=None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> tuple[bool, str]:
        if current_best == 0:
            return True, "replace_target_shift"

        target_price = self._pricing.compute_target_price(
            side,
            level,
            current_best,
            extra_offset_bps=extra_offset_bps,
            regime_scale=regime_scale,
            trend=trend,
            funding_bias_bps=funding_bias_bps,
        )

        target_offset = self._pricing.compute_offset(
            level,
            current_best,
            regime_scale=regime_scale,
        )
        if extra_offset_bps > 0:
            target_offset += current_best * extra_offset_bps / Decimal("10000")
        if target_offset == 0:
            return True, "replace_target_shift"

        tolerance = self._settings.reprice_tolerance_percent
        max_deviation = target_offset * tolerance

        price_diff = abs(prev_price - target_price)
        if price_diff <= max_deviation:
            return False, "hold_within_tolerance"

        min_move_ticks = self._settings.min_reprice_move_ticks
        if (
            min_move_ticks > 0
            and self._tick_size > 0
            and price_diff < (self._tick_size * Decimal(min_move_ticks))
        ):
            return False, "hold_below_tick_gate"

        prev_edge = self._pricing.theoretical_edge_bps(side, prev_price, current_best)
        target_edge = self._pricing.theoretical_edge_bps(side, target_price, current_best)
        edge_delta = abs(target_edge - prev_edge)
        if edge_delta < self._settings.min_reprice_edge_delta_bps:
            return False, "hold_below_edge_gate"

        return True, "replace_target_shift"

    def build_level_context(self, strategy: StrategyContext, side, level: int) -> LevelContext:
        key = (str(side), level)
        prev_ext_id = strategy._level_ext_ids.get(key)
        prev_order = strategy._orders.get_active_order(prev_ext_id)
        return LevelContext(
            key=key,
            side_name=strategy._normalise_side(str(side)),
            prev_ext_id=prev_ext_id,
            prev_order=prev_order,
        )

    @staticmethod
    def _inventory_override_active(strategy: StrategyContext, side, market_ctx: RepriceMarketContext) -> bool:
        reduces_inventory = not strategy._increases_inventory(side)
        inventory_elevated = market_ctx.inventory_band in {"WARN", "CRITICAL", "HARD"}
        return reduces_inventory and inventory_elevated

    async def evaluate_blocking_conditions(
        self,
        strategy: StrategyContext,
        side,
        level: int,
        *,
        now: float,
        level_ctx: LevelContext,
        market_ctx: RepriceMarketContext,
    ) -> bool:
        if strategy._level_cancel_pending_ext_id.get(level_ctx.key) is not None:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_cancel_pending",
            )
            return False

        pof_until = strategy._level_pof_until.get(level_ctx.key, 0.0)
        if now < pof_until:
            strategy._record_reprice_decision(side=side, level=level, reason="skip_pof_cooldown")
            return False

        inventory_override = self._inventory_override_active(strategy, side, market_ctx)
        if strategy._is_strong_counter_trend_side(level_ctx.side_name, market_ctx.trend):
            if not inventory_override:
                strategy._record_reprice_decision(
                    side=side,
                    level=level,
                    reason="skip_trend_counter_strong",
                    regime=market_ctx.regime.regime,
                    trend_direction=market_ctx.trend.direction,
                    trend_strength=market_ctx.trend.strength,
                    inventory_band=market_ctx.inventory_band,
                    funding_bias_bps=market_ctx.funding_bias_bps,
                )
                if level_ctx.prev_ext_id is not None and strategy._settings.trend_cancel_counter_on_strong:
                    await strategy._cancel_level_order(
                        key=level_ctx.key,
                        external_id=level_ctx.prev_ext_id,
                        side=side,
                        level=level,
                        reason="trend_counter_strong",
                    )
                return False

            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="allow_trend_counter_inventory_override",
                regime=market_ctx.regime.regime,
                trend_direction=market_ctx.trend.direction,
                trend_strength=market_ctx.trend.strength,
                inventory_band=market_ctx.inventory_band,
                funding_bias_bps=market_ctx.funding_bias_bps,
            )

        if market_ctx.min_reprice_interval_s > 0:
            last_reprice = strategy._level_last_reprice_at.get(level_ctx.key, 0.0)
            if (now - last_reprice) < market_ctx.min_reprice_interval_s:
                return False

        if strategy._ob.is_stale():
            strategy._record_reprice_decision(side=side, level=level, reason="skip_stale")
            if strategy._settings.cancel_on_stale_book and level_ctx.prev_ext_id is not None:
                stale_since = strategy._level_stale_since.get(level_ctx.key)
                if stale_since is None:
                    stale_since = now
                    strategy._level_stale_since[level_ctx.key] = stale_since
                if (now - stale_since) >= strategy._settings.stale_cancel_grace_s:
                    await strategy._cancel_level_order(
                        key=level_ctx.key,
                        external_id=level_ctx.prev_ext_id,
                        side=side,
                        level=level,
                        reason="stale_orderbook",
                    )
            return False

        strategy._level_stale_since[level_ctx.key] = None
        return True

    def _decision_fields(
        self,
        *,
        market_ctx: RepriceMarketContext,
        spread_bps: Optional[Decimal],
        extra_offset_bps: Decimal,
    ) -> dict[str, Any]:
        return {
            "spread_bps": spread_bps,
            "extra_offset_bps": extra_offset_bps,
            "regime": market_ctx.regime.regime,
            "trend_direction": market_ctx.trend.direction,
            "trend_strength": market_ctx.trend.strength,
            "inventory_band": market_ctx.inventory_band,
            "funding_bias_bps": market_ctx.funding_bias_bps,
        }

    async def _prepare_quote_inputs(
        self,
        strategy: StrategyContext,
        side,
        level: int,
        *,
        level_ctx: LevelContext,
        market_ctx: RepriceMarketContext,
    ) -> Optional[QuoteInputs]:
        bid = strategy._ob.best_bid()
        ask = strategy._ob.best_ask()
        if bid is None or ask is None:
            return None
        if strategy._ob.is_stale():
            return None

        spread_bps = strategy._ob.spread_bps()
        imbalance = strategy._ob.orderbook_imbalance(strategy._settings.imbalance_window_s)
        inventory_override = self._inventory_override_active(strategy, side, market_ctx)
        guard: GuardDecision = strategy._guards.check(
            side=level_ctx.side_name,
            level=level,
            spread_bps=spread_bps,
            imbalance=imbalance,
            regime=market_ctx.regime,
            inventory_override=inventory_override,
        )
        if guard.allow and guard.reason != "allow":
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=guard.reason,
                **self._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=spread_bps,
                    extra_offset_bps=guard.extra_offset_bps,
                ),
            )
        if not guard.allow:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=guard.reason,
                **self._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=spread_bps,
                    extra_offset_bps=guard.extra_offset_bps,
                ),
            )
            if level_ctx.prev_ext_id is not None:
                await strategy._cancel_level_order(
                    key=level_ctx.key,
                    external_id=level_ctx.prev_ext_id,
                    side=side,
                    level=level,
                    reason=(
                        f"toxicity:{market_ctx.regime.regime.lower()}"
                        if guard.reason == "skip_toxicity"
                        else guard.reason.replace("skip_", "")
                    ),
                )
            return None

        current_best = bid.price if str(side).endswith("BUY") or str(side) == "BUY" else ask.price

        # Fold in POF-spread correlation boost if available.
        pof_boost = Decimal("0")
        post_only = getattr(strategy, "_post_only", None)
        if post_only is not None:
            pof_boost = post_only.pof_offset_boost_bps
        rate_limit_boost = getattr(strategy._orders, "rate_limit_extra_offset_bps", Decimal("0"))
        if not isinstance(rate_limit_boost, Decimal):
            rate_limit_boost = Decimal("0")

        return QuoteInputs(
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            extra_offset_bps=guard.extra_offset_bps + pof_boost + rate_limit_boost,
            current_best=current_best,
        )

    async def _evaluate_existing_order(
        self,
        strategy: StrategyContext,
        side,
        level: int,
        *,
        level_ctx: LevelContext,
        market_ctx: RepriceMarketContext,
        quote_inputs: QuoteInputs,
        target_price: Decimal,
    ) -> bool:
        if level_ctx.prev_order is None:
            return True

        if strategy._order_age_exceeded(level_ctx.key, max_age_s=market_ctx.max_order_age_s):
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="replace_max_age",
                prev_price=level_ctx.prev_order.price,
                **self._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=quote_inputs.spread_bps,
                    extra_offset_bps=quote_inputs.extra_offset_bps,
                ),
            )
            if level_ctx.prev_ext_id is None:
                return True
            cancelled = await strategy._cancel_level_order(
                key=level_ctx.key,
                external_id=level_ctx.prev_ext_id,
                side=side,
                level=level,
                reason="max_order_age",
            )
            if not cancelled:
                return False
            level_ctx.prev_ext_id = None
            level_ctx.prev_order = None
            return True

        should_reprice, decision_reason = self.needs_reprice(
            side,
            level_ctx.prev_order.price,
            quote_inputs.current_best,
            level,
            extra_offset_bps=quote_inputs.extra_offset_bps,
            regime_scale=market_ctx.regime.offset_scale,
            trend=market_ctx.trend,
            funding_bias_bps=market_ctx.funding_bias_bps,
        )
        strategy._record_reprice_decision(
            side=side,
            level=level,
            reason=decision_reason,
            current_best=quote_inputs.current_best,
            prev_price=level_ctx.prev_order.price,
            target_price=target_price,
            **self._decision_fields(
                market_ctx=market_ctx,
                spread_bps=quote_inputs.spread_bps,
                extra_offset_bps=quote_inputs.extra_offset_bps,
            ),
        )
        return bool(should_reprice)

    def compute_risk_adjusted_order(
        self, strategy, side, level, *, level_ctx, quote_inputs, market_ctx,
    ) -> RiskAdjustedOrder:
        return _compute_risk_adjusted(
            self, strategy, side, level,
            level_ctx=level_ctx, quote_inputs=quote_inputs, market_ctx=market_ctx,
        )

    async def execute_replace_if_needed(
        self, strategy, side, level, *, level_ctx, quote_inputs, order_plan, market_ctx,
    ) -> None:
        await _execute_replace(
            self, strategy, side, level,
            level_ctx=level_ctx, quote_inputs=quote_inputs,
            order_plan=order_plan, market_ctx=market_ctx,
        )

    def _resolve_market_context(
        self,
        strategy: StrategyContext,
        market_ctx: Optional[RepriceMarketContext],
    ) -> RepriceMarketContext:
        if market_ctx is not None:
            return market_ctx
        if strategy._settings.market_profile == "crypto":
            regime = strategy._volatility.evaluate()
            trend = strategy._trend_signal.evaluate()
        else:
            regime = RegimeState(regime="NORMAL")
            trend = TrendState()
        min_interval, max_order_age_s = strategy._volatility.cadence(regime)
        rate_limit_multiplier = getattr(strategy._orders, "rate_limit_reprice_multiplier", Decimal("1"))
        if not isinstance(rate_limit_multiplier, Decimal):
            rate_limit_multiplier = Decimal("1")
        min_interval *= float(rate_limit_multiplier)
        return RepriceMarketContext(
            regime=regime,
            trend=trend,
            min_reprice_interval_s=min_interval,
            max_order_age_s=max_order_age_s,
            funding_bias_bps=strategy._funding_bias_bps(),
            inventory_band=strategy._pricing.inventory_band(),
        )

    async def evaluate(
        self,
        strategy: StrategyContext,
        side,
        level: int,
        *,
        market_ctx: Optional[RepriceMarketContext] = None,
    ) -> None:
        """Evaluate whether the order at (side, level) needs repricing."""
        now = time.monotonic()
        market_ctx = self._resolve_market_context(strategy, market_ctx)
        level_ctx = self.build_level_context(strategy, side, level)
        allowed = await self.evaluate_blocking_conditions(
            strategy,
            side,
            level,
            now=now,
            level_ctx=level_ctx,
            market_ctx=market_ctx,
        )
        if not allowed:
            return

        quote_inputs = await self._prepare_quote_inputs(
            strategy,
            side,
            level,
            level_ctx=level_ctx,
            market_ctx=market_ctx,
        )
        if quote_inputs is None:
            return

        if market_ctx.inventory_band in {"CRITICAL", "HARD"} and strategy._increases_inventory(side):
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=(
                    "skip_inventory_hard"
                    if market_ctx.inventory_band == "HARD"
                    else "skip_inventory_critical"
                ),
                **self._decision_fields(
                    market_ctx=market_ctx,
                    spread_bps=quote_inputs.spread_bps,
                    extra_offset_bps=quote_inputs.extra_offset_bps,
                ),
            )
            if level_ctx.prev_ext_id is not None:
                await strategy._cancel_level_order(
                    key=level_ctx.key,
                    external_id=level_ctx.prev_ext_id,
                    side=side,
                    level=level,
                    reason=market_ctx.inventory_band.lower(),
                )
            return

        order_plan = self.compute_risk_adjusted_order(
            strategy,
            side,
            level,
            level_ctx=level_ctx,
            quote_inputs=quote_inputs,
            market_ctx=market_ctx,
        )
        should_continue = await self._evaluate_existing_order(
            strategy,
            side,
            level,
            level_ctx=level_ctx,
            market_ctx=market_ctx,
            quote_inputs=quote_inputs,
            target_price=order_plan.target_price,
        )
        if not should_continue:
            return

        await self.execute_replace_if_needed(
            strategy,
            side,
            level,
            level_ctx=level_ctx,
            quote_inputs=quote_inputs,
            order_plan=order_plan,
            market_ctx=market_ctx,
        )
