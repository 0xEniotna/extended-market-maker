from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from market_maker.decision_models import RegimeState
from market_maker.volatility_regime import VolatilityRegime


class _FakeOrderbook:
    def __init__(self, vol_short: Decimal, vol_medium: Decimal, drift_short: Decimal):
        self._vol_short = vol_short
        self._vol_medium = vol_medium
        self._drift_short = drift_short

    def micro_volatility_bps(self, window_s: float):
        return self._vol_short if window_s <= 15 else self._vol_medium

    def micro_drift_bps(self, window_s: float):
        _ = window_s
        return self._drift_short


def _settings(**overrides):
    base = {
        "vol_regime_enabled": True,
        "vol_regime_short_window_s": 15.0,
        "vol_regime_medium_window_s": 60.0,
        "vol_regime_calm_bps": Decimal("8"),
        "vol_regime_elevated_bps": Decimal("20"),
        "vol_regime_extreme_bps": Decimal("45"),
        "vol_offset_scale_calm": Decimal("0.8"),
        "vol_offset_scale_elevated": Decimal("1.5"),
        "vol_offset_scale_extreme": Decimal("2.2"),
        "market_profile": "crypto",
        "min_reprice_interval_s": 0.5,
        "max_order_age_s": 15.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_classifies_calm_regime():
    regime = VolatilityRegime(
        _settings(),
        _FakeOrderbook(Decimal("5"), Decimal("6"), Decimal("1")),
    ).evaluate()
    assert regime.regime == "CALM"
    assert regime.offset_scale == Decimal("0.8")
    assert not regime.pause


def test_classifies_extreme_regime_and_pause():
    regime = VolatilityRegime(
        _settings(),
        _FakeOrderbook(Decimal("60"), Decimal("55"), Decimal("3")),
    ).evaluate()
    assert regime.regime == "EXTREME"
    assert regime.pause
    assert regime.offset_scale == Decimal("2.2")


def test_crypto_cadence_by_regime():
    classifier = VolatilityRegime(_settings(), _FakeOrderbook(Decimal("0"), Decimal("0"), Decimal("0")))
    assert classifier.cadence(RegimeState(regime="CALM")) == (0.4, 20.0)
    assert classifier.cadence(RegimeState(regime="NORMAL")) == (0.6, 15.0)
    assert classifier.cadence(RegimeState(regime="ELEVATED")) == (1.0, 10.0)
    assert classifier.cadence(RegimeState(regime="EXTREME")) == (2.0, 5.0)


def test_legacy_cadence_uses_settings():
    settings = _settings(market_profile="legacy", min_reprice_interval_s=1.3, max_order_age_s=9.0)
    classifier = VolatilityRegime(settings, _FakeOrderbook(Decimal("0"), Decimal("0"), Decimal("0")))
    assert classifier.cadence(RegimeState(regime="ELEVATED")) == (1.3, 9.0)
