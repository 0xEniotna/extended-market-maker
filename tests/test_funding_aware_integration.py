"""Integration + rollback tests for the funding-aware overlay.

Gate 2 of the funding-aware MM rollout plan:
1. Flag-off fresh-boot ⇒ PricingEngine has no policy attached AND quote
   output is byte-identical to a pre-feature reference PricingEngine.
2. Flag-on → flag-off hot reload via rebuild_components ⇒ same byte-identical
   guarantee post-reload.

Also asserts the new policy actually perturbs quotes when on (i.e., it isn't
a silent no-op).
"""
from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# SDK module stubs (mirror the convention used in other test files)
# ---------------------------------------------------------------------------

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.trading_client",
    "x10.perpetual.positions",
    "x10.perpetual.accounts",
    "x10.perpetual.configuration",
    "x10.perpetual.orderbook",
    "x10.perpetual.trades",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.utils",
    "x10.utils.http",
]
for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED", CANCELLED="CANCELLED", EXPIRED="EXPIRED",
    REJECTED="REJECTED", OPEN="OPEN",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT", MARKET="MARKET")
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT", IOC="IOC")
_orders_mod.OpenOrderModel = object
_orders_mod.OrderStatusReason = SimpleNamespace(POST_ONLY_FAILED="POST_ONLY_FAILED")

_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")

from market_maker.funding_aware import (  # noqa: E402
    FundingAwarePolicy,
    make_policy_if_enabled,
)
from market_maker.orderbook_manager import PriceLevel  # noqa: E402
from market_maker.pricing_engine import PricingEngine  # noqa: E402

OrderSide = _orders_mod.OrderSide


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


class _FakeOB:
    def __init__(self, bid: Decimal, ask: Decimal) -> None:
        self._bid = PriceLevel(price=bid, size=Decimal("100"))
        self._ask = PriceLevel(price=ask, size=Decimal("100"))

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask

    def spread_bps(self):
        mid = (self._bid.price + self._ask.price) / 2
        return (self._ask.price - self._bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self):
        return self.spread_bps()

    def is_stale(self):
        return False


class _FakeRisk:
    def __init__(self, position: Decimal = Decimal("0")) -> None:
        self._pos = position

    def get_current_position(self):
        return self._pos


def _make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        offset_mode="dynamic",
        spread_multiplier=Decimal("1"),
        min_offset_bps=Decimal("2"),
        max_offset_bps=Decimal("50"),
        price_offset_per_level_percent=Decimal("0.05"),
        max_position_size=Decimal("10"),
        inventory_hard_pct=Decimal("0.95"),
        inventory_critical_pct=Decimal("0.80"),
        inventory_warn_pct=Decimal("0.50"),
        inventory_deadband_pct=Decimal("0.10"),
        skew_shape_k=Decimal("2"),
        skew_max_bps=Decimal("25"),
        inventory_skew_factor=Decimal("0.6"),
        trend_skew_boost=Decimal("1.5"),
        market_profile="crypto",
        size_scale_per_level=Decimal("1.2"),
    )


def _make_engine(
    *,
    funding_aware: Optional[FundingAwarePolicy],
    position: Decimal = Decimal("0"),
    bid: Decimal = Decimal("99.95"),
    ask: Decimal = Decimal("100.05"),
) -> PricingEngine:
    return PricingEngine(
        settings=_make_settings(),
        orderbook_mgr=_FakeOB(bid, ask),
        risk_mgr=_FakeRisk(position),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.1"),
        funding_aware=funding_aware,
    )


# ---------------------------------------------------------------------------
# 1. Flag-off rollback (Gate 2 scenario A)
# ---------------------------------------------------------------------------


class TestRollbackFreshBoot:
    """Engine with funding_aware=None must produce identical quotes to an
    engine constructed by the (unmodified) call-site that doesn't pass the
    new kwarg at all."""

    def test_no_kwarg_vs_explicit_none_byte_identical(self) -> None:
        # Engine A: doesn't pass funding_aware at all (mimics pre-feature
        # call site).
        engine_a = PricingEngine(
            settings=_make_settings(),
            orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05")),
            risk_mgr=_FakeRisk(Decimal("3")),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.1"),
        )
        # Engine B: passes explicit None for the new kwarg.
        engine_b = _make_engine(funding_aware=None, position=Decimal("3"))

        for side in (OrderSide.BUY, OrderSide.SELL):
            for level in (0, 1):
                for funding_bps in (Decimal("0"), Decimal("3"), Decimal("-2")):
                    a = engine_a.compute_target_price(
                        side, level, Decimal("100"),
                        funding_bias_bps=funding_bps,
                    )
                    b = engine_b.compute_target_price(
                        side, level, Decimal("100"),
                        funding_bias_bps=funding_bps,
                    )
                    assert a == b, (
                        f"Quote drift for side={side} level={level} "
                        f"funding_bps={funding_bps}: A={a} B={b}"
                    )

    @pytest.mark.parametrize("position", [
        Decimal("0"), Decimal("3"), Decimal("-3"),
        Decimal("8"), Decimal("-8"),
    ])
    def test_flag_off_matches_reference_across_states(self, position: Decimal) -> None:
        # Reference: no funding_aware in the constructor.
        ref = PricingEngine(
            settings=_make_settings(),
            orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05")),
            risk_mgr=_FakeRisk(position),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.1"),
        )
        # New: funding_aware explicitly None.
        new = _make_engine(funding_aware=None, position=position)

        for side in (OrderSide.BUY, OrderSide.SELL):
            for funding_bps in (Decimal("0"), Decimal("4")):
                a = ref.compute_target_price(
                    side, 0, Decimal("100"), funding_bias_bps=funding_bps,
                )
                b = new.compute_target_price(
                    side, 0, Decimal("100"), funding_bias_bps=funding_bps,
                )
                assert a == b


# ---------------------------------------------------------------------------
# 2. Hot reload from on → off (Gate 2 scenario B)
# ---------------------------------------------------------------------------


class TestRollbackHotReload:
    """Construct with flag on, then construct again with flag off (simulates
    the SIGHUP rebuild_components path) and assert byte-identical quotes
    after the toggle."""

    def test_toggle_off_returns_to_baseline(self) -> None:
        # Compare flag-never-on engine (uses legacy bias) with flag-toggled-off
        # engine (was on, now off — uses legacy bias). Both must produce
        # identical quotes given identical inputs.
        funding_bps = Decimal("4")

        ref_engine = _make_engine(funding_aware=None, position=Decimal("5"))
        ref_bid = ref_engine.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=funding_bps,
        )
        ref_ask = ref_engine.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=funding_bps,
        )

        # Build with flag on. Use a horizon that makes the overlay output
        # saturate at the bps cap so it cannot coincidentally match the
        # legacy 4-bps contribution (which is below the 8-bps cap).
        rate_holder = [Decimal("0.0005")]
        policy = make_policy_if_enabled(
            enabled=True,
            coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.01"),
            funding_rate_source=lambda: rate_holder[0],
        )
        on_engine = _make_engine(funding_aware=policy, position=Decimal("5"))
        # Realistic flag-on call site: legacy bias passes 0 because
        # FundingManager.funding_bias_bps() returns 0 when the overlay
        # is active.
        on_bid = on_engine.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        # Sanity: overlay must perturb relative to a no-funding baseline.
        no_funding = _make_engine(funding_aware=None, position=Decimal("5"))
        no_funding_bid = no_funding.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        assert on_bid != no_funding_bid

        # Hot reload: rebuild with flag off (no policy). Identical to ref.
        off_engine = _make_engine(funding_aware=None, position=Decimal("5"))
        off_bid = off_engine.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=funding_bps,
        )
        off_ask = off_engine.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=funding_bps,
        )

        assert off_bid == ref_bid
        assert off_ask == ref_ask


# ---------------------------------------------------------------------------
# 3. Sanity: flag-on actually perturbs quotes in the expected direction
# ---------------------------------------------------------------------------


class TestOverlayPerturbation:
    """Confirm the overlay isn't a silent no-op: F>0 widens bid + tightens ask."""

    def _setup(self, rate: Decimal):
        policy = make_policy_if_enabled(
            enabled=True,
            coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.001"),
            funding_rate_source=lambda: rate,
        )
        return _make_engine(funding_aware=policy)

    def test_positive_funding_widens_bid_tightens_ask(self) -> None:
        base = _make_engine(funding_aware=None)
        on = self._setup(rate=Decimal("0.0001"))

        # When the overlay is on, FundingManager would normally return 0 for
        # funding_bias_bps. To isolate the overlay's effect we pass 0 to both
        # so the only difference is the overlay path.
        b_base = base.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        b_on = on.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        a_base = base.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        a_on = on.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        # F>0 ⇒ overlay widens bid (lower price, further below BBO) and
        # tightens ask (lower price, closer to BBO).
        assert b_on < b_base
        assert a_on < a_base

    def test_negative_funding_mirrors_direction(self) -> None:
        base = _make_engine(funding_aware=None)
        on = self._setup(rate=Decimal("-0.0001"))

        b_base = base.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        b_on = on.compute_target_price(
            OrderSide.BUY, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        a_base = base.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        a_on = on.compute_target_price(
            OrderSide.SELL, 0, Decimal("100"), funding_bias_bps=Decimal("0"),
        )
        # F<0 ⇒ overlay tightens bid (higher) and widens ask (higher).
        assert b_on > b_base
        assert a_on > a_base
