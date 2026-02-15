from __future__ import annotations

from decimal import Decimal

from .decision_models import TrendState


def _ema(values: list[Decimal], alpha: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    out = values[0]
    one = Decimal("1")
    for v in values[1:]:
        out = alpha * v + (one - alpha) * out
    return out


class TrendSignal:
    """Estimate trend direction from short/medium mid-price EMAs."""

    def __init__(self, settings, orderbook_mgr) -> None:
        self._settings = settings
        self._ob = orderbook_mgr

    def evaluate(self) -> TrendState:
        if not self._settings.trend_enabled:
            return TrendState()

        window = max(
            float(self._settings.trend_slow_ema_s),
            float(self._settings.vol_regime_long_window_s),
        )
        mids = self._ob.mid_prices(window)
        if len(mids) < 3:
            return TrendState()

        fast_n = max(2, int(float(self._settings.trend_fast_ema_s) / 2.0))
        slow_n = max(fast_n + 1, int(float(self._settings.trend_slow_ema_s) / 2.0))
        if len(mids) < slow_n:
            return TrendState()

        fast_alpha = Decimal("2") / Decimal(str(fast_n + 1))
        slow_alpha = Decimal("2") / Decimal(str(slow_n + 1))

        fast = _ema(mids[-slow_n:], fast_alpha)
        slow = _ema(mids[-slow_n:], slow_alpha)
        if slow <= 0:
            return TrendState()

        diff_bps = (fast - slow) / slow * Decimal("10000")
        abs_bps = abs(diff_bps)
        # Normalize on short-horizon micro-vol so strength is regime-aware.
        ref_vol = self._ob.micro_volatility_bps(self._settings.vol_regime_medium_window_s)
        if ref_vol is None or ref_vol <= 0:
            ref_vol = Decimal("10")

        strength = abs_bps / ref_vol
        strength = max(Decimal("0"), min(Decimal("1"), strength))

        if diff_bps > 0:
            direction = "BULLISH"
        elif diff_bps < 0:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return TrendState(direction=direction, strength=strength)
