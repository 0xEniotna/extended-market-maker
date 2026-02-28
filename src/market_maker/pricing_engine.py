from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Optional, cast

from .types import OrderbookLike, PriceLevelLike, PricingSettingsLike, RiskManagerLike
from .utils import safe_decimal, safe_float


class PricingEngine:
    """Quote price + size calculation with inventory/funding/trend adjustments.

    Hot-path methods (``compute_offset``, ``_skew_component``,
    ``compute_target_price``, ``theoretical_edge_bps``) use **float**
    arithmetic internally for ~10-50x speed-up on ARM (Raspberry Pi).
    Decimal precision is used only for final tick rounding and size
    quantisation — both exchange-critical operations.
    """

    # Aliases for backward compatibility with callers using the static methods.
    _to_decimal = staticmethod(safe_decimal)
    _to_float = staticmethod(safe_float)

    def __init__(
        self,
        settings: object,
        orderbook_mgr: object,
        risk_mgr: RiskManagerLike,
        tick_size: Decimal,
        base_order_size: Decimal,
        min_order_size_step: Decimal,
    ) -> None:
        self._settings = cast(PricingSettingsLike, settings)
        self._ob = cast(OrderbookLike, orderbook_mgr)
        self._risk = risk_mgr
        self._tick_size = tick_size
        self._tick_size_f = float(tick_size)
        self._base_order_size = base_order_size
        self._min_order_size_step = min_order_size_step

    def _offset_mode(self) -> str:
        mode = self._settings.offset_mode
        if isinstance(mode, str):
            return mode
        value = getattr(mode, "value", mode)
        return str(value)

    def round_to_tick(self, price: Decimal, side: Optional[object] = None) -> Decimal:
        """Round price to the nearest tick — stays Decimal for exchange precision."""
        if self._tick_size <= 0:
            return price
        side_name = str(side)
        rounding = ROUND_UP if side_name.endswith("SELL") or side_name == "SELL" else ROUND_DOWN
        return (price / self._tick_size).quantize(
            Decimal("1"), rounding=rounding
        ) * self._tick_size

    # ------------------------------------------------------------------
    # Hot path — float arithmetic
    # ------------------------------------------------------------------

    def compute_offset(
        self,
        level: int,
        best_price: Decimal,
        *,
        regime_scale: Decimal = Decimal("1"),
    ) -> Decimal:
        best_f = float(best_price)
        offset_f = self._compute_offset_f(level, best_f, float(regime_scale))
        return Decimal(str(offset_f))

    def _compute_offset_f(
        self,
        level: int,
        best_price_f: float,
        regime_scale_f: float = 1.0,
    ) -> float:
        """Pure-float offset computation for the hot path."""
        if self._offset_mode() == "dynamic":
            spread_bps_raw = self._ob.spread_bps_ema()
            spread_bps_f = float(spread_bps_raw) if spread_bps_raw is not None and spread_bps_raw > 0 else self._to_float(self._settings.min_offset_bps)

            multiplier_f = self._to_float(self._settings.spread_multiplier)
            min_offset_f = self._to_float(self._settings.min_offset_bps)
            max_offset_f = self._to_float(self._settings.max_offset_bps)

            level_mult = level + 1
            per_level_bps = spread_bps_f * multiplier_f * level_mult
            floor = min_offset_f * level_mult
            ceiling = max_offset_f * level_mult
            per_level_bps = max(floor, min(per_level_bps, ceiling))
            per_level_bps *= max(0.0, regime_scale_f)
            return best_price_f * per_level_bps / 10000.0

        offset_pct_f = self._to_float(self._settings.price_offset_per_level_percent) * (level + 1)
        return best_price_f * offset_pct_f / 100.0

    def inventory_norm(self) -> Decimal:
        max_pos = self._to_decimal(self._settings.max_position_size)
        if max_pos > 0:
            val = self._risk.get_current_position() / max_pos
        else:
            val = Decimal("0")
        return max(Decimal("-1"), min(Decimal("1"), val))

    def _inventory_norm_f(self) -> float:
        """Float version of inventory_norm for hot-path use."""
        max_pos = self._to_float(self._settings.max_position_size)
        if max_pos > 0:
            val = float(self._risk.get_current_position()) / max_pos
        else:
            val = 0.0
        return max(-1.0, min(1.0, val))

    def inventory_band(self) -> str:
        abs_norm = abs(self.inventory_norm())
        if abs_norm >= self._to_decimal(self._settings.inventory_hard_pct):
            return "HARD"
        if abs_norm >= self._to_decimal(self._settings.inventory_critical_pct):
            return "CRITICAL"
        if abs_norm >= self._to_decimal(self._settings.inventory_warn_pct):
            return "WARN"
        return "NORMAL"

    def _skew_component(self, trend=None) -> Decimal:
        """Inventory skew in bps — returns Decimal for API compat."""
        return Decimal(str(self._skew_component_f(trend)))

    def _skew_component_f(self, trend=None) -> float:
        """Pure-float skew computation for the hot path."""
        inv_norm = self._inventory_norm_f()
        deadband = max(0.0, min(1.0, self._to_float(self._settings.inventory_deadband_pct)))
        abs_norm = abs(inv_norm)

        if abs_norm <= deadband:
            shaped = 0.0
        else:
            if deadband >= 1.0:
                normalized = 0.0
            else:
                normalized = (abs_norm - deadband) / (1.0 - deadband)
            sign = 1.0 if inv_norm >= 0 else -1.0
            shape_k = max(0.0, self._to_float(self._settings.skew_shape_k))
            if shape_k == 0:
                curve = normalized
            else:
                denom = math.tanh(shape_k)
                curve = 0.0 if denom == 0 else math.tanh(shape_k * normalized) / denom
            shaped = sign * curve

        max_skew_bps = (
            self._to_float(self._settings.skew_max_bps)
            * self._to_float(self._settings.inventory_skew_factor)
        )
        if trend is not None and str(self._settings.market_profile) == "crypto":
            boost = self._to_float(self._settings.trend_skew_boost)
            strength = float(trend.strength) if hasattr(trend, "strength") else 0.0
            max_skew_bps *= 1.0 + (boost - 1.0) * strength

        if self.inventory_band() in {"WARN", "CRITICAL", "HARD"}:
            max_skew_bps *= 1.25

        return shaped * max_skew_bps

    def compute_target_price(
        self,
        side: object,
        level: int,
        best_price: Decimal,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend=None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> Decimal:
        best_f = float(best_price)
        extra_f = float(extra_offset_bps)
        regime_f = float(regime_scale)
        funding_f = float(funding_bias_bps)

        # Offset (float)
        offset_f = self._compute_offset_f(level, best_f, regime_f)
        if extra_f > 0:
            offset_f += best_f * extra_f / 10000.0

        # Skew + funding (float)
        skew_bps_f = self._skew_component_f(trend)
        skew_offset_f = best_f * skew_bps_f / 10000.0
        funding_offset_f = best_f * funding_f / 10000.0

        # Compute raw price (float)
        side_name = str(side)
        is_buy = side_name.endswith("BUY") or side_name == "BUY"
        if is_buy:
            raw_f = best_f - offset_f - skew_offset_f - funding_offset_f
        else:
            raw_f = best_f + offset_f - skew_offset_f - funding_offset_f

        # Clamp against BBO (float comparison for speed, Decimal for precision edge)
        bid = cast(Optional[PriceLevelLike], self._ob.best_bid())
        ask = cast(Optional[PriceLevelLike], self._ob.best_ask())
        if bid is not None and ask is not None:
            if is_buy and raw_f >= float(ask.price):
                raw_f = float(ask.price) - self._tick_size_f
            elif not is_buy and raw_f <= float(bid.price):
                raw_f = float(bid.price) + self._tick_size_f

        # Convert back to Decimal for tick rounding (exchange precision).
        raw = Decimal(str(raw_f))
        return self.round_to_tick(raw, side)

    def theoretical_edge_bps(self, side: object, quote_price: Decimal, current_best: Decimal) -> Decimal:
        best_f = float(current_best)
        if best_f <= 0:
            return Decimal("0")
        quote_f = float(quote_price)
        side_name = str(side)
        if side_name.endswith("BUY") or side_name == "BUY":
            edge_f = (best_f - quote_f) / best_f * 10000.0
        else:
            edge_f = (quote_f - best_f) / best_f * 10000.0
        return Decimal(str(edge_f))

    def level_size(self, level: int) -> Decimal:
        """Level size — stays Decimal for exchange-critical quantisation."""
        scale = self._to_decimal(self._settings.size_scale_per_level, default="1")
        if scale == 1 or level == 0:
            return self._base_order_size
        raw = self._base_order_size * scale ** level
        return (raw / self._min_order_size_step).quantize(
            Decimal("1"), rounding=ROUND_DOWN,
        ) * self._min_order_size_step
