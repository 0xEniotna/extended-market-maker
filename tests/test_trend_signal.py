from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from market_maker.trend_signal import TrendSignal


class _FakeOrderbook:
    def __init__(self, mids, ref_vol: Decimal):
        self._mids = mids
        self._ref_vol = ref_vol

    def mid_prices(self, window_s: float):
        _ = window_s
        return list(self._mids)

    def micro_volatility_bps(self, window_s: float):
        _ = window_s
        return self._ref_vol


def _settings(**overrides):
    base = {
        "trend_enabled": True,
        "trend_fast_ema_s": 15.0,
        "trend_slow_ema_s": 60.0,
        "vol_regime_long_window_s": 120.0,
        "vol_regime_medium_window_s": 60.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_detects_bullish_trend():
    mids = [Decimal("100") + Decimal(i) * Decimal("0.1") for i in range(80)]
    signal = TrendSignal(_settings(), _FakeOrderbook(mids, Decimal("5")))
    trend = signal.evaluate()
    assert trend.direction == "BULLISH"
    assert Decimal("0") <= trend.strength <= Decimal("1")


def test_detects_bearish_trend():
    mids = [Decimal("108") - Decimal(i) * Decimal("0.1") for i in range(80)]
    signal = TrendSignal(_settings(), _FakeOrderbook(mids, Decimal("5")))
    trend = signal.evaluate()
    assert trend.direction == "BEARISH"
    assert Decimal("0") <= trend.strength <= Decimal("1")


def test_returns_neutral_when_insufficient_data():
    signal = TrendSignal(_settings(), _FakeOrderbook([Decimal("100"), Decimal("101")], Decimal("5")))
    trend = signal.evaluate()
    assert trend.direction == "NEUTRAL"
    assert trend.strength == Decimal("0")
