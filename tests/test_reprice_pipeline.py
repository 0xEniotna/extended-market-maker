from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from market_maker.reprice_pipeline import RepricePipeline


class _PricingStub:
    def compute_target_price(
        self,
        side,
        level: int,
        current_best: Decimal,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend=None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> Decimal:
        _ = (side, level, extra_offset_bps, regime_scale, trend, funding_bias_bps)
        return current_best - Decimal("1")

    def compute_offset(
        self,
        level: int,
        current_best: Decimal,
        *,
        regime_scale: Decimal = Decimal("1"),
    ) -> Decimal:
        _ = (level, current_best, regime_scale)
        return Decimal("1")

    def theoretical_edge_bps(self, side, quote_price: Decimal, current_best: Decimal) -> Decimal:
        _ = side
        return abs(current_best - quote_price)


def _settings(**overrides):
    base = {
        "reprice_tolerance_percent": Decimal("0.5"),
        "min_reprice_move_ticks": 0,
        "min_reprice_edge_delta_bps": Decimal("0"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_holds_within_tolerance():
    pipeline = RepricePipeline(_settings(), Decimal("1"), _PricingStub())
    should, reason = pipeline.needs_reprice("BUY", Decimal("99.2"), Decimal("100"), 0)
    assert not should
    assert reason == "hold_within_tolerance"


def test_replaces_when_far_from_target():
    pipeline = RepricePipeline(_settings(), Decimal("1"), _PricingStub())
    should, reason = pipeline.needs_reprice("BUY", Decimal("97"), Decimal("100"), 0)
    assert should
    assert reason == "replace_target_shift"


def test_holds_below_tick_gate():
    pipeline = RepricePipeline(
        _settings(min_reprice_move_ticks=3, min_reprice_edge_delta_bps=Decimal("0")),
        Decimal("1"),
        _PricingStub(),
    )
    should, reason = pipeline.needs_reprice("BUY", Decimal("97.8"), Decimal("100"), 0)
    assert not should
    assert reason == "hold_below_tick_gate"


def test_holds_below_edge_gate():
    pipeline = RepricePipeline(
        _settings(
            reprice_tolerance_percent=Decimal("0.1"),
            min_reprice_move_ticks=0,
            min_reprice_edge_delta_bps=Decimal("2"),
        ),
        Decimal("0.01"),
        _PricingStub(),
    )
    should, reason = pipeline.needs_reprice("BUY", Decimal("98.7"), Decimal("100"), 0)
    assert not should
    assert reason == "hold_below_edge_gate"
