from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from market_maker.decision_models import (
    GuardDecision,
    RegimeState,
    RepriceMarketContext,
    TrendState,
)
from market_maker.reprice_pipeline import LevelContext, RepricePipeline


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


@pytest.mark.asyncio
async def test_prepare_quote_inputs_includes_latency_widening():
    pipeline = RepricePipeline(_settings(), Decimal("1"), _PricingStub())
    settings = _settings(imbalance_window_s=2.0)

    ob = SimpleNamespace(
        best_bid=lambda: SimpleNamespace(price=Decimal("99"), size=Decimal("10")),
        best_ask=lambda: SimpleNamespace(price=Decimal("101"), size=Decimal("10")),
        is_stale=lambda: False,
        spread_bps=lambda: Decimal("20"),
        orderbook_imbalance=lambda window_s: Decimal("0"),
    )
    strategy = SimpleNamespace(
        _ob=ob,
        _settings=settings,
        _guards=SimpleNamespace(
            check=lambda **kwargs: GuardDecision(
                allow=True,
                reason="allow",
                extra_offset_bps=Decimal("2"),
            )
        ),
        _orders=SimpleNamespace(rate_limit_extra_offset_bps=Decimal("3")),
        _post_only=SimpleNamespace(pof_offset_boost_bps=Decimal("4")),
        _latency_monitor=SimpleNamespace(extra_offset_bps=Decimal("5")),
        _increases_inventory=lambda side: True,
        _record_reprice_decision=lambda **kwargs: None,
    )

    quote_inputs = await pipeline._prepare_quote_inputs(
        strategy,
        "BUY",
        0,
        level_ctx=LevelContext(key=("BUY", 0), side_name="BUY", prev_ext_id=None),
        market_ctx=RepriceMarketContext(
            regime=RegimeState(),
            trend=TrendState(),
            min_reprice_interval_s=0.0,
            max_order_age_s=30.0,
            funding_bias_bps=Decimal("0"),
            inventory_band="NORMAL",
        ),
    )

    assert quote_inputs is not None
    assert quote_inputs.extra_offset_bps == Decimal("14")


@pytest.mark.asyncio
async def test_execute_prepares_order_capacity_before_final_bbo_check():
    events: list[str] = []
    permit = SimpleNamespace(name="permit")
    pipeline = RepricePipeline(_settings(), Decimal("1"), _PricingStub())

    ob = SimpleNamespace(
        is_stale=lambda: events.append("is_stale") or False,
        best_bid=lambda: events.append("best_bid") or SimpleNamespace(price=Decimal("99")),
        best_ask=lambda: events.append("best_ask") or SimpleNamespace(price=Decimal("101")),
    )
    orders = SimpleNamespace(
        prepare_place_order=AsyncMock(
            side_effect=lambda: events.append("prepare") or permit
        ),
        place_order=AsyncMock(return_value="ext-1"),
        in_rate_limit_degraded=False,
        get_active_order=lambda ext_id: SimpleNamespace(exchange_order_id="1"),
    )
    strategy = SimpleNamespace(
        _ob=ob,
        _orders=orders,
        _market_min_order_size=Decimal("1"),
        _apply_post_only_safety=lambda **kwargs: Decimal("98"),
        _effective_safety_ticks=lambda key: 1,
        _on_successful_quote=lambda key: None,
        _level_ext_ids={},
        _level_order_created_at={},
        _level_last_reprice_at={},
        _journal=SimpleNamespace(record_order_placed=lambda **kwargs: None),
        _risk=SimpleNamespace(get_current_position=lambda: Decimal("0")),
        _normalise_side=lambda side: side,
    )

    await pipeline.execute_replace_if_needed(
        strategy,
        "BUY",
        0,
        level_ctx=LevelContext(key=("BUY", 0), side_name="BUY", prev_ext_id=None),
        quote_inputs=SimpleNamespace(spread_bps=Decimal("1"), extra_offset_bps=Decimal("0")),
        order_plan=SimpleNamespace(target_price=Decimal("98"), level_size=Decimal("1")),
        market_ctx=RepriceMarketContext(
            regime=RegimeState(),
            trend=TrendState(),
            min_reprice_interval_s=0.0,
            max_order_age_s=30.0,
            funding_bias_bps=Decimal("0"),
            inventory_band="NORMAL",
        ),
    )

    assert events[:2] == ["prepare", "is_stale"]
    orders.place_order.assert_awaited_once()
    assert orders.place_order.await_args.kwargs["permit"] is permit
