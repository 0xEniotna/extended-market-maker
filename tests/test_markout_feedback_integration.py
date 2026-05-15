"""Integration tests for the markout-feedback overlay.

Covers:
  1. Rollback safety: PricingEngine with `markout_feedback=None` produces
     byte-identical quotes to one without the kwarg (Gate 2 freshboot).
  2. Combination with funding_aware: both overlays can coexist.
  3. End-to-end: a bad fill leads to widening on the right side at the
     subsequent compute_target_price call.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import pytest

from market_maker.funding_aware import (
    FundingAwareConfig,
    FundingAwarePolicy,
)
from market_maker.markout_feedback import (
    MarkoutFeedbackConfig,
    MarkoutFeedbackPolicy,
)
from market_maker.pricing_engine import PricingEngine

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _Price:
    price: Decimal
    size: Decimal = Decimal("1")


class _FakeOB:
    def __init__(self, bid: Decimal, ask: Decimal) -> None:
        self._bid = _Price(bid)
        self._ask = _Price(ask)

    def best_bid(self) -> _Price:
        return self._bid

    def best_ask(self) -> _Price:
        return self._ask

    def spread_bps(self) -> Decimal:
        mid = (self._bid.price + self._ask.price) / 2
        return (self._ask.price - self._bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self) -> Decimal:
        return self.spread_bps()

    def is_stale(self) -> bool:
        return False


class _FakeRisk:
    def __init__(self, position: Decimal) -> None:
        self._pos = position

    def get_current_position(self) -> Decimal:
        return self._pos


def _settings() -> object:
    from types import SimpleNamespace
    return SimpleNamespace(
        offset_mode="dynamic",
        spread_multiplier=Decimal("1.0"),
        min_offset_bps=Decimal("4"),
        max_offset_bps=Decimal("90"),
        price_offset_per_level_percent=Decimal("0.05"),
        max_position_size=Decimal("10"),
        inventory_hard_pct=Decimal("0.95"),
        inventory_critical_pct=Decimal("0.80"),
        inventory_warn_pct=Decimal("0.50"),
        inventory_deadband_pct=Decimal("0.10"),
        skew_shape_k=Decimal("2.0"),
        skew_max_bps=Decimal("25"),
        inventory_skew_factor=Decimal("0.6"),
        trend_skew_boost=Decimal("1.5"),
        market_profile="crypto",
        size_scale_per_level=Decimal("1.2"),
    )


def _engine(
    *,
    funding_aware: Optional[FundingAwarePolicy] = None,
    markout_feedback: Optional[MarkoutFeedbackPolicy] = None,
    position: Decimal = Decimal("0"),
    bid: Decimal = Decimal("99.95"),
    ask: Decimal = Decimal("100.05"),
) -> PricingEngine:
    return PricingEngine(
        settings=_settings(),
        orderbook_mgr=_FakeOB(bid, ask),
        risk_mgr=_FakeRisk(position),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.1"),
        funding_aware=funding_aware,
        markout_feedback=markout_feedback,
    )


def _make_mf(
    *, enabled: bool = True, half_life_s: Decimal = Decimal("30"),
    threshold_bps: Decimal = Decimal("2"), gain: Decimal = Decimal("0.5"),
    cap_bps: Decimal = Decimal("5"), horizon_s: int = 5,
    mid: Decimal = Decimal("100"),
) -> MarkoutFeedbackPolicy:
    cfg = MarkoutFeedbackConfig(
        enabled=enabled, half_life_s=half_life_s,
        threshold_bps=threshold_bps, gain=gain, cap_bps=cap_bps,
        horizon_s=horizon_s,
    )
    return MarkoutFeedbackPolicy(cfg, mid_source=lambda: mid)


# ---------------------------------------------------------------------------
# 1. Rollback safety: flag-off byte-identical to no-overlay engine
# ---------------------------------------------------------------------------


class TestRollbackFreshBoot:
    """When markout_feedback=None, quotes must match a baseline engine
    constructed without the new kwarg at all."""

    def test_kwarg_default_vs_explicit_none(self) -> None:
        # Engine A: doesn't pass markout_feedback (uses default None)
        eng_a = PricingEngine(
            settings=_settings(),
            orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05")),
            risk_mgr=_FakeRisk(Decimal("0")),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.1"),
        )
        eng_b = _engine(markout_feedback=None)
        for side in ("BUY", "SELL"):
            for level in (0, 1):
                a = eng_a.compute_target_price(side, level, Decimal("99.95"))
                b = eng_b.compute_target_price(side, level, Decimal("99.95"))
                assert a == b, f"side={side} level={level}: {a} != {b}"

    @pytest.mark.parametrize("position", [
        Decimal("0"), Decimal("3"), Decimal("-3"), Decimal("9"), Decimal("-9"),
    ])
    def test_flag_off_matches_reference_across_inventory(
        self, position: Decimal,
    ) -> None:
        ref = PricingEngine(
            settings=_settings(),
            orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05")),
            risk_mgr=_FakeRisk(position),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.1"),
        )
        new = _engine(markout_feedback=None, position=position)
        for side in ("BUY", "SELL"):
            a = ref.compute_target_price(side, 0, Decimal("99.95"))
            b = new.compute_target_price(side, 0, Decimal("99.95"))
            assert a == b


# ---------------------------------------------------------------------------
# 2. Both overlays off → equivalent to baseline
# ---------------------------------------------------------------------------


class TestBothOverlaysOff:
    """Funding-aware None + markout-feedback None ⇒ baseline quotes."""

    def test_both_none_byte_identical_to_baseline(self) -> None:
        baseline = PricingEngine(
            settings=_settings(),
            orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05")),
            risk_mgr=_FakeRisk(Decimal("0")),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.1"),
        )
        with_both = _engine(funding_aware=None, markout_feedback=None)
        for side in ("BUY", "SELL"):
            for level in (0, 1):
                a = baseline.compute_target_price(side, level, Decimal("99.95"))
                b = with_both.compute_target_price(side, level, Decimal("99.95"))
                assert a == b


# ---------------------------------------------------------------------------
# 3. End-to-end: bad fill leads to widening on subsequent quote
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_bad_buy_fill_widens_subsequent_bid(self) -> None:
        # Setup: policy with mid that produces a bad markout on BUY
        cfg = MarkoutFeedbackConfig(
            enabled=True, half_life_s=Decimal("30"),
            threshold_bps=Decimal("0"),  # any negative markout triggers
            gain=Decimal("1.0"), cap_bps=Decimal("20"), horizon_s=5,
        )
        # Mid = 99.80 → BUY at 100 = -20 bps markout
        policy = MarkoutFeedbackPolicy(cfg, mid_source=lambda: Decimal("99.80"))

        eng_ref = _engine(markout_feedback=None)
        eng_test = _engine(markout_feedback=policy)

        # Reference BUY at mid 100
        ref_bid = eng_ref.compute_target_price(
            "BUY", 0, Decimal("99.95"),
        )

        # Now simulate a bad BUY fill (mid moved against us)
        import time
        now = time.time()
        policy.on_fill(ts=now - 10, side="BUY", price=Decimal("100"))
        # Force the tick so pending fill is processed
        policy.tick(now_ts=now)

        # Engine should now widen the BID side
        widened_bid = eng_test.compute_target_price(
            "BUY", 0, Decimal("99.95"),
        )
        # Widened bid < ref bid (further from BBO)
        assert widened_bid < ref_bid

    def test_bad_sell_fill_widens_subsequent_ask(self) -> None:
        cfg = MarkoutFeedbackConfig(
            enabled=True, half_life_s=Decimal("30"),
            threshold_bps=Decimal("0"),
            gain=Decimal("1.0"), cap_bps=Decimal("20"), horizon_s=5,
        )
        # Mid = 100.20 → SELL at 100 = -20 bps markout (MM perspective)
        policy = MarkoutFeedbackPolicy(cfg, mid_source=lambda: Decimal("100.20"))

        eng_ref = _engine(markout_feedback=None)
        eng_test = _engine(markout_feedback=policy)

        ref_ask = eng_ref.compute_target_price(
            "SELL", 0, Decimal("100.05"),
        )

        import time
        now = time.time()
        policy.on_fill(ts=now - 10, side="SELL", price=Decimal("100"))
        policy.tick(now_ts=now)

        widened_ask = eng_test.compute_target_price(
            "SELL", 0, Decimal("100.05"),
        )
        # Widened ask > ref ask (further from BBO)
        assert widened_ask > ref_ask

    def test_bad_buy_does_not_widen_ask(self) -> None:
        # The widening must be per-side. A bad BUY only widens BUY.
        cfg = MarkoutFeedbackConfig(
            enabled=True, half_life_s=Decimal("30"),
            threshold_bps=Decimal("0"), gain=Decimal("1.0"),
            cap_bps=Decimal("20"), horizon_s=5,
        )
        policy = MarkoutFeedbackPolicy(cfg, mid_source=lambda: Decimal("99.80"))

        eng_ref = _engine(markout_feedback=None)
        eng_test = _engine(markout_feedback=policy)

        ref_ask = eng_ref.compute_target_price(
            "SELL", 0, Decimal("100.05"),
        )

        import time
        now = time.time()
        policy.on_fill(ts=now - 10, side="BUY", price=Decimal("100"))
        policy.tick(now_ts=now)

        ask_after_buy = eng_test.compute_target_price(
            "SELL", 0, Decimal("100.05"),
        )
        # SELL side untouched
        assert ask_after_buy == ref_ask


# ---------------------------------------------------------------------------
# 4. Compatibility with funding_aware
# ---------------------------------------------------------------------------


class TestCoexistsWithFundingAware:
    def test_both_on_no_exceptions(self) -> None:
        """Smoke test: both overlays on, engine still produces valid quotes."""
        fa_cfg = FundingAwareConfig(
            enabled=True, coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.001"),
        )
        fa_policy = FundingAwarePolicy(
            fa_cfg, funding_rate_source=lambda: Decimal("0.0001"),
        )
        mf_policy = _make_mf()

        eng = _engine(funding_aware=fa_policy, markout_feedback=mf_policy)
        bid = eng.compute_target_price("BUY", 0, Decimal("99.95"))
        ask = eng.compute_target_price("SELL", 0, Decimal("100.05"))
        assert bid > 0
        assert ask > 0
        assert ask > bid  # spread maintained

    def test_markout_widening_stacks_on_top_of_funding_signal(self) -> None:
        """When both fire, the resulting widening is roughly additive
        (small-perturbation regime). The exact value depends on rounding
        and clamps, but the bid moves further away than with either
        overlay alone."""
        fa_cfg = FundingAwareConfig(
            enabled=True, coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("1"),  # disable dollar cap
        )
        fa_policy = FundingAwarePolicy(
            fa_cfg, funding_rate_source=lambda: Decimal("0.0001"),
        )

        mf_cfg = MarkoutFeedbackConfig(
            enabled=True, half_life_s=Decimal("30"),
            threshold_bps=Decimal("0"), gain=Decimal("1.0"),
            cap_bps=Decimal("10"), horizon_s=5,
        )
        mf_policy = MarkoutFeedbackPolicy(
            mf_cfg, mid_source=lambda: Decimal("99.80"),  # bad BUY markout
        )

        # Bad fill to seed the EWMA on BUY
        import time
        now = time.time()
        mf_policy.on_fill(ts=now - 10, side="BUY", price=Decimal("100"))
        mf_policy.tick(now_ts=now)

        eng_baseline = _engine()  # both off
        eng_fa_only = _engine(funding_aware=fa_policy)
        eng_both = _engine(funding_aware=fa_policy, markout_feedback=mf_policy)

        bid_baseline = eng_baseline.compute_target_price("BUY", 0, Decimal("99.95"))
        bid_fa_only = eng_fa_only.compute_target_price("BUY", 0, Decimal("99.95"))
        bid_both = eng_both.compute_target_price("BUY", 0, Decimal("99.95"))

        # Both overlays push bid down (F>0 widens bid, bad markout widens bid).
        # Both should be ≤ baseline bid; "both" should be ≤ FA-only.
        assert bid_fa_only <= bid_baseline
        assert bid_both <= bid_fa_only
