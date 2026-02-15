from __future__ import annotations

import math
from decimal import ROUND_DOWN, ROUND_UP, Decimal


class PricingEngine:
    """Quote price + size calculation with inventory/funding/trend adjustments."""

    def __init__(
        self,
        settings,
        orderbook_mgr,
        risk_mgr,
        tick_size: Decimal,
        base_order_size: Decimal,
        min_order_size_step: Decimal,
    ) -> None:
        self._settings = settings
        self._ob = orderbook_mgr
        self._risk = risk_mgr
        self._tick_size = tick_size
        self._base_order_size = base_order_size
        self._min_order_size_step = min_order_size_step

    def round_to_tick(self, price: Decimal, side=None) -> Decimal:
        if self._tick_size <= 0:
            return price
        side_name = str(side)
        rounding = ROUND_UP if side_name.endswith("SELL") or side_name == "SELL" else ROUND_DOWN
        return (price / self._tick_size).quantize(
            Decimal("1"), rounding=rounding
        ) * self._tick_size

    def compute_offset(
        self,
        level: int,
        best_price: Decimal,
        *,
        regime_scale: Decimal = Decimal("1"),
    ) -> Decimal:
        _100 = Decimal("100")
        _10000 = Decimal("10000")

        if self._settings.offset_mode.value == "dynamic":
            spread_bps = self._ob.spread_bps_ema()
            if spread_bps is None or spread_bps <= 0:
                spread_bps = self._settings.min_offset_bps

            per_level_bps = spread_bps * self._settings.spread_multiplier * (level + 1)
            floor = self._settings.min_offset_bps * (level + 1)
            ceiling = self._settings.max_offset_bps * (level + 1)
            per_level_bps = max(floor, min(per_level_bps, ceiling))
            per_level_bps *= max(Decimal("0"), regime_scale)
            return best_price * per_level_bps / _10000

        offset_pct = self._settings.price_offset_per_level_percent * (level + 1)
        return best_price * offset_pct / _100

    def inventory_norm(self) -> Decimal:
        max_pos = self._settings.max_position_size
        if max_pos > 0:
            val = self._risk.get_current_position() / max_pos
        else:
            val = Decimal("0")
        return max(Decimal("-1"), min(Decimal("1"), val))

    def inventory_band(self) -> str:
        abs_norm = abs(self.inventory_norm())
        if abs_norm >= self._settings.inventory_hard_pct:
            return "HARD"
        if abs_norm >= self._settings.inventory_critical_pct:
            return "CRITICAL"
        if abs_norm >= self._settings.inventory_warn_pct:
            return "WARN"
        return "NORMAL"

    def _skew_component(self, trend=None) -> Decimal:
        skew_norm = self.inventory_norm()
        deadband = max(Decimal("0"), min(Decimal("1"), self._settings.inventory_deadband_pct))
        abs_norm = abs(skew_norm)
        if abs_norm <= deadband:
            shaped = Decimal("0")
        else:
            if deadband >= Decimal("1"):
                normalized = Decimal("0")
            else:
                normalized = (abs_norm - deadband) / (Decimal("1") - deadband)
            sign = Decimal("1") if skew_norm >= 0 else Decimal("-1")
            shape_k = float(max(Decimal("0"), self._settings.skew_shape_k))
            if shape_k == 0:
                curve = float(normalized)
            else:
                denom = math.tanh(shape_k)
                curve = 0.0 if denom == 0 else math.tanh(shape_k * float(normalized)) / denom
            shaped = sign * Decimal(str(curve))

        max_skew_bps = self._settings.skew_max_bps * self._settings.inventory_skew_factor
        if trend is not None and self._settings.market_profile == "crypto":
            max_skew_bps *= (
                Decimal("1")
                + (self._settings.trend_skew_boost - Decimal("1")) * trend.strength
            )
        if self.inventory_band() in {"WARN", "CRITICAL", "HARD"}:
            max_skew_bps *= Decimal("1.25")
        return shaped * max_skew_bps

    def compute_target_price(
        self,
        side,
        level: int,
        best_price: Decimal,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend=None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> Decimal:
        offset = self.compute_offset(level, best_price, regime_scale=regime_scale)
        if extra_offset_bps > 0:
            offset += best_price * extra_offset_bps / Decimal("10000")

        skew_bps = self._skew_component(trend)
        skew_offset = best_price * skew_bps / Decimal("10000")
        funding_offset = best_price * funding_bias_bps / Decimal("10000")

        side_name = str(side)
        if side_name.endswith("BUY") or side_name == "BUY":
            raw = best_price - offset - skew_offset - funding_offset
        else:
            raw = best_price + offset - skew_offset - funding_offset

        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        if bid is not None and ask is not None:
            if (side_name.endswith("BUY") or side_name == "BUY") and raw >= ask.price:
                raw = ask.price - self._tick_size
            elif (side_name.endswith("SELL") or side_name == "SELL") and raw <= bid.price:
                raw = bid.price + self._tick_size

        return self.round_to_tick(raw, side)

    def theoretical_edge_bps(self, side, quote_price: Decimal, current_best: Decimal) -> Decimal:
        if current_best <= 0:
            return Decimal("0")
        side_name = str(side)
        if side_name.endswith("BUY") or side_name == "BUY":
            return (current_best - quote_price) / current_best * Decimal("10000")
        return (quote_price - current_best) / current_best * Decimal("10000")

    def level_size(self, level: int) -> Decimal:
        scale = self._settings.size_scale_per_level
        if scale == 1 or level == 0:
            return self._base_order_size
        return (self._base_order_size * scale ** level).quantize(
            self._min_order_size_step,
            rounding=ROUND_DOWN,
        )
