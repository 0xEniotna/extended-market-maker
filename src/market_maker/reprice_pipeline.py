from __future__ import annotations

import time
from decimal import Decimal

from .decision_models import RegimeState, TrendState


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

    async def evaluate(self, strategy, side, level: int) -> None:
        """Evaluate whether the order at (side, level) needs repricing."""
        key = (str(side), level)
        now = time.monotonic()

        # --- POF cooldown: don't re-quote a level that just got rejected ---
        pof_until = strategy._level_pof_until.get(key, 0.0)
        if now < pof_until:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_pof_cooldown",
            )
            return

        side_name = strategy._normalise_side(str(side))
        regime = (
            strategy._volatility.evaluate()
            if strategy._settings.market_profile == "crypto"
            else RegimeState(regime="NORMAL")
        )
        trend = (
            strategy._trend_signal.evaluate()
            if strategy._settings.market_profile == "crypto"
            else TrendState()
        )
        min_interval, max_order_age_s = strategy._volatility.cadence(regime)

        # --- Min reprice interval: prevent cancel/place churn ---
        if min_interval > 0:
            last_reprice = strategy._level_last_reprice_at.get(key, 0.0)
            if (now - last_reprice) < min_interval:
                return

        active_orders = strategy._orders.get_active_orders()
        prev_ext_id = strategy._level_ext_ids.get(key)
        prev_order = active_orders.get(prev_ext_id) if prev_ext_id else None

        # --- Stale orderbook fail-safe ---
        if strategy._ob.is_stale():
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_stale",
            )
            if strategy._settings.cancel_on_stale_book and prev_ext_id is not None:
                stale_since = strategy._level_stale_since.get(key)
                if stale_since is None:
                    stale_since = now
                    strategy._level_stale_since[key] = stale_since
                if (now - stale_since) >= strategy._settings.stale_cancel_grace_s:
                    await strategy._cancel_level_order(
                        key=key,
                        external_id=prev_ext_id,
                        side=side,
                        level=level,
                        reason="stale_orderbook",
                    )
            return

        strategy._level_stale_since[key] = None

        bid = strategy._ob.best_bid()
        ask = strategy._ob.best_ask()
        if bid is None or ask is None:
            return

        # --- Double-check staleness right before any quoting logic ---
        if strategy._ob.is_stale():
            return

        spread_bps = strategy._ob.spread_bps()
        imbalance = strategy._ob.orderbook_imbalance(strategy._settings.imbalance_window_s)
        guard = strategy._guards.check(
            side=side_name,
            level=level,
            spread_bps=spread_bps,
            imbalance=imbalance,
            regime=regime,
        )
        if not guard.allow:
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=guard.reason,
                spread_bps=spread_bps,
                regime=regime.regime,
                trend_direction=trend.direction,
                trend_strength=trend.strength,
                inventory_band=strategy._pricing.inventory_band(),
                funding_bias_bps=strategy._funding_bias_bps(),
            )
            if prev_ext_id is not None:
                await strategy._cancel_level_order(
                    key=key,
                    external_id=prev_ext_id,
                    side=side,
                    level=level,
                    reason=(
                        f"toxicity:{regime.regime.lower()}"
                        if guard.reason == "skip_toxicity"
                        else guard.reason.replace("skip_", "")
                    ),
                )
            return
        extra_offset_bps = guard.extra_offset_bps

        # Enforce a maximum resting age for stale orders.
        if prev_order is not None and strategy._order_age_exceeded(
            key, max_age_s=max_order_age_s
        ):
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason="replace_max_age",
                prev_price=prev_order.price,
                spread_bps=spread_bps,
                regime=regime.regime,
                trend_direction=trend.direction,
                trend_strength=trend.strength,
                inventory_band=strategy._pricing.inventory_band(),
                funding_bias_bps=strategy._funding_bias_bps(),
            )
            await strategy._cancel_level_order(
                key=key,
                external_id=prev_ext_id,
                side=side,
                level=level,
                reason="max_order_age",
            )
            prev_ext_id = None
            prev_order = None

        current_best = bid.price if str(side).endswith("BUY") or str(side) == "BUY" else ask.price
        target_price = strategy._compute_target_price(
            side,
            level,
            current_best,
            extra_offset_bps=extra_offset_bps,
            regime_scale=regime.offset_scale,
            trend=trend,
            funding_bias_bps=strategy._funding_bias_bps(),
        )

        if prev_order is not None:
            should_reprice, decision_reason = strategy._needs_reprice(
                side,
                prev_order.price,
                current_best,
                level,
                extra_offset_bps=extra_offset_bps,
                regime_scale=regime.offset_scale,
                trend=trend,
                funding_bias_bps=strategy._funding_bias_bps(),
            )
            if not should_reprice:
                strategy._record_reprice_decision(
                    side=side,
                    level=level,
                    reason=decision_reason,
                    current_best=current_best,
                    prev_price=prev_order.price,
                    target_price=target_price,
                    spread_bps=spread_bps,
                    extra_offset_bps=extra_offset_bps,
                    regime=regime.regime,
                    trend_direction=trend.direction,
                    trend_strength=trend.strength,
                    inventory_band=strategy._pricing.inventory_band(),
                    funding_bias_bps=strategy._funding_bias_bps(),
                )
                return  # Order is still within tolerance / gates
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=decision_reason,
                current_best=current_best,
                prev_price=prev_order.price,
                target_price=target_price,
                spread_bps=spread_bps,
                extra_offset_bps=extra_offset_bps,
                regime=regime.regime,
                trend_direction=trend.direction,
                trend_strength=trend.strength,
                inventory_band=strategy._pricing.inventory_band(),
                funding_bias_bps=strategy._funding_bias_bps(),
            )

        inventory_band = strategy._pricing.inventory_band()
        if inventory_band in {"CRITICAL", "HARD"} and strategy._increases_inventory(side):
            strategy._record_reprice_decision(
                side=side,
                level=level,
                reason=(
                    "skip_inventory_hard"
                    if inventory_band == "HARD"
                    else "skip_inventory_critical"
                ),
                spread_bps=spread_bps,
                regime=regime.regime,
                trend_direction=trend.direction,
                trend_strength=trend.strength,
                inventory_band=inventory_band,
                funding_bias_bps=strategy._funding_bias_bps(),
            )
            if prev_ext_id is not None:
                await strategy._cancel_level_order(
                    key=key,
                    external_id=prev_ext_id,
                    side=side,
                    level=level,
                    reason=inventory_band.lower(),
                )
            return

        # Compute per-level order size (pyramid scaling)
        requested_size = strategy._level_size(level)
        if strategy._settings.market_profile == "crypto":
            counter_side = None
            if trend.direction == "BULLISH":
                counter_side = "SELL"
            elif trend.direction == "BEARISH":
                counter_side = "BUY"
            if counter_side is not None and side_name == counter_side:
                if (
                    trend.strength >= strategy._settings.trend_strong_threshold
                    and level == 0
                ):
                    requested_size = Decimal("0")
                else:
                    cut = strategy._settings.trend_counter_side_size_cut * trend.strength
                    multiplier = max(Decimal("0"), Decimal("1") - cut)
                    requested_size = strategy._quantize_size(requested_size * multiplier)

        reserved_same_side_qty = sum(
            info.size
            for ext_id, info in active_orders.items()
            if str(info.side) == str(side) and ext_id != prev_ext_id
        )
        reserved_open_notional_usd = sum(
            info.size * info.price
            for ext_id, info in active_orders.items()
            if ext_id != prev_ext_id
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

        # Cancel existing order if we can no longer hold this side.
        if level_size < strategy._market_min_order_size:
            if prev_ext_id is not None:
                await strategy._cancel_level_order(
                    key=key,
                    external_id=prev_ext_id,
                    side=side,
                    level=level,
                    reason="risk_limit",
                )
            return

        if prev_ext_id is not None:
            await strategy._cancel_level_order(
                key=key,
                external_id=prev_ext_id,
                side=side,
                level=level,
                reason="reprice",
            )

        # Final post-only safety recheck right before placement.
        if strategy._ob.is_stale():
            return
        fresh_bid = strategy._ob.best_bid()
        fresh_ask = strategy._ob.best_ask()
        if fresh_bid is None or fresh_ask is None:
            return
        safe_target_price = strategy._apply_post_only_safety(
            side=side,
            target_price=target_price,
            bid_price=fresh_bid.price,
            ask_price=fresh_ask.price,
            safety_ticks=strategy._effective_safety_ticks(key),
        )
        if safe_target_price is None:
            return

        ext_id = await strategy._orders.place_order(
            side=side,
            price=safe_target_price,
            size=level_size,
            level=level,
        )
        if ext_id is not None:
            strategy._on_successful_quote(key)
            strategy._level_ext_ids[key] = ext_id
            strategy._level_order_created_at[key] = time.monotonic()
            strategy._level_last_reprice_at[key] = time.monotonic()
            order_info = strategy._orders.get_active_orders().get(ext_id)
            strategy._journal.record_order_placed(
                external_id=ext_id,
                exchange_id=order_info.exchange_order_id if order_info else None,
                side=strategy._normalise_side(str(side)),
                price=safe_target_price,
                size=level_size,
                level=level,
                best_bid=fresh_bid.price if fresh_bid else None,
                best_ask=fresh_ask.price if fresh_ask else None,
                position=strategy._risk.get_current_position(),
            )
