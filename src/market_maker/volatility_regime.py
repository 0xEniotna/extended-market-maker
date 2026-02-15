from __future__ import annotations

from decimal import Decimal

from .decision_models import RegimeState


class VolatilityRegime:
    """Classify market microstructure into a volatility regime."""

    def __init__(self, settings, orderbook_mgr) -> None:
        self._settings = settings
        self._ob = orderbook_mgr

    def evaluate(self) -> RegimeState:
        if not self._settings.vol_regime_enabled:
            return RegimeState(regime="NORMAL")

        vol_short = self._ob.micro_volatility_bps(self._settings.vol_regime_short_window_s)
        vol_medium = self._ob.micro_volatility_bps(self._settings.vol_regime_medium_window_s)
        drift_short = self._ob.micro_drift_bps(self._settings.vol_regime_short_window_s)

        signal_vol = vol_medium if vol_medium is not None else vol_short
        if signal_vol is None:
            return RegimeState(
                regime="NORMAL",
                vol_short_bps=vol_short,
                vol_medium_bps=vol_medium,
                drift_short_bps=drift_short,
                offset_scale=Decimal("1"),
                pause=False,
            )

        calm = self._settings.vol_regime_calm_bps
        elevated = self._settings.vol_regime_elevated_bps
        extreme = self._settings.vol_regime_extreme_bps

        regime = "NORMAL"
        scale = Decimal("1")
        pause = False

        if signal_vol >= extreme:
            regime = "EXTREME"
            scale = self._settings.vol_offset_scale_extreme
            pause = True
        elif signal_vol >= elevated:
            regime = "ELEVATED"
            scale = self._settings.vol_offset_scale_elevated
        elif signal_vol <= calm:
            regime = "CALM"
            scale = self._settings.vol_offset_scale_calm

        return RegimeState(
            regime=regime,
            vol_short_bps=vol_short,
            vol_medium_bps=vol_medium,
            drift_short_bps=drift_short,
            offset_scale=scale,
            pause=pause,
        )

    def cadence(self, regime: RegimeState) -> tuple[float, float]:
        """Return (min_reprice_interval_s, max_order_age_s) for current regime."""
        if self._settings.market_profile != "crypto":
            return self._settings.min_reprice_interval_s, self._settings.max_order_age_s

        if regime.regime == "CALM":
            return 0.4, 20.0
        if regime.regime == "NORMAL":
            return 0.6, 15.0
        if regime.regime == "ELEVATED":
            return 1.0, 10.0
        return 2.0, 5.0
