from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from market_maker.pricing_engine import PricingEngine


class _FakeOrderbook:
    def __init__(self):
        self._spread = Decimal("10")
        self._bid = SimpleNamespace(price=Decimal("100"), size=Decimal("10"))
        self._ask = SimpleNamespace(price=Decimal("100.10"), size=Decimal("10"))

    def spread_bps_ema(self):
        return self._spread

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask


class _FakeRisk:
    def __init__(self, pos: Decimal):
        self._pos = pos

    def get_current_position(self):
        return self._pos


def _settings(**overrides):
    base = {
        "offset_mode": SimpleNamespace(value="dynamic"),
        "spread_multiplier": Decimal("1.0"),
        "min_offset_bps": Decimal("3"),
        "max_offset_bps": Decimal("100"),
        "price_offset_per_level_percent": Decimal("0.3"),
        "max_position_size": Decimal("100"),
        "inventory_warn_pct": Decimal("0.5"),
        "inventory_critical_pct": Decimal("0.8"),
        "inventory_hard_pct": Decimal("0.95"),
        "inventory_deadband_pct": Decimal("0.1"),
        "skew_shape_k": Decimal("2.0"),
        "skew_max_bps": Decimal("20"),
        "inventory_skew_factor": Decimal("0.5"),
        "market_profile": "crypto",
        "trend_skew_boost": Decimal("1.5"),
        "size_scale_per_level": Decimal("1.2"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _engine(pos: Decimal = Decimal("0"), **setting_overrides):
    return PricingEngine(
        settings=_settings(**setting_overrides),
        orderbook_mgr=_FakeOrderbook(),
        risk_mgr=_FakeRisk(pos),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("10"),
        min_order_size_step=Decimal("0.1"),
    )


def test_dynamic_offset_respects_regime_scale():
    engine = _engine()
    offset = engine.compute_offset(
        level=1,
        best_price=Decimal("100"),
        regime_scale=Decimal("1.5"),
    )
    assert offset == Decimal("0.30")


def test_inventory_band_thresholds():
    assert _engine(pos=Decimal("10")).inventory_band() == "NORMAL"
    assert _engine(pos=Decimal("55")).inventory_band() == "WARN"
    assert _engine(pos=Decimal("85")).inventory_band() == "CRITICAL"
    assert _engine(pos=Decimal("97")).inventory_band() == "HARD"


def test_funding_bias_shifts_buy_price_down():
    engine = _engine()
    no_bias = engine.compute_target_price(
        "BUY",
        level=0,
        best_price=Decimal("100"),
        funding_bias_bps=Decimal("0"),
    )
    with_bias = engine.compute_target_price(
        "BUY",
        level=0,
        best_price=Decimal("100"),
        funding_bias_bps=Decimal("5"),
    )
    assert with_bias < no_bias


def test_level_size_scales_by_level():
    engine = _engine()
    assert engine.level_size(0) == Decimal("10")
    assert engine.level_size(2) == Decimal("14.4")
