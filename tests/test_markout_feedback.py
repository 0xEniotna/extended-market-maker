"""Tests for the markout-feedback overlay policy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from market_maker.markout_feedback import (
    MarkoutFeedbackConfig,
    MarkoutFeedbackPolicy,
    make_policy_if_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(
    *,
    enabled: bool = True,
    half_life_s: Decimal = Decimal("30"),
    threshold_bps: Decimal = Decimal("2"),
    gain: Decimal = Decimal("0.5"),
    cap_bps: Decimal = Decimal("5"),
    horizon_s: int = 5,
    mid: Decimal = Decimal("100"),
) -> MarkoutFeedbackPolicy:
    cfg = MarkoutFeedbackConfig(
        enabled=enabled,
        half_life_s=half_life_s,
        threshold_bps=threshold_bps,
        gain=gain,
        cap_bps=cap_bps,
        horizon_s=horizon_s,
    )
    return MarkoutFeedbackPolicy(cfg, mid_source=lambda: mid)


# ---------------------------------------------------------------------------
# 1. Flag off → no behavior
# ---------------------------------------------------------------------------


class TestDisabled:
    def test_factory_returns_none_when_disabled(self) -> None:
        p = make_policy_if_enabled(
            enabled=False,
            half_life_s=Decimal("30"),
            threshold_bps=Decimal("2"),
            gain=Decimal("0.5"),
            cap_bps=Decimal("5"),
            horizon_s=5,
            mid_source=lambda: Decimal("100"),
        )
        assert p is None

    def test_factory_returns_policy_when_enabled(self) -> None:
        p = make_policy_if_enabled(
            enabled=True,
            half_life_s=Decimal("30"),
            threshold_bps=Decimal("2"),
            gain=Decimal("0.5"),
            cap_bps=Decimal("5"),
            horizon_s=5,
            mid_source=lambda: Decimal("100"),
        )
        assert p is not None
        assert p.enabled

    def test_disabled_policy_returns_zero(self) -> None:
        pol = _make(enabled=False)
        assert pol.extra_widening_bps("BUY") == Decimal("0")
        assert pol.extra_widening_bps("SELL") == Decimal("0")

    def test_disabled_policy_ignores_on_fill(self) -> None:
        pol = _make(enabled=False)
        pol.on_fill(ts=0.0, side="BUY", price=Decimal("100"))
        assert pol.pending_count() == 0

    def test_disabled_policy_tick_noop(self) -> None:
        pol = _make(enabled=False)
        assert pol.tick(now_ts=1000.0) == 0


# ---------------------------------------------------------------------------
# 2. No fills / no markout signal → no widening
# ---------------------------------------------------------------------------


class TestZeroSignal:
    def test_no_fills_no_widen(self) -> None:
        pol = _make()
        assert pol.extra_widening_bps("BUY", now_ts=1000.0) == Decimal("0")
        assert pol.extra_widening_bps("SELL", now_ts=1000.0) == Decimal("0")

    def test_invalid_side_returns_zero(self) -> None:
        pol = _make()
        assert pol.extra_widening_bps("BOGUS") == Decimal("0")
        assert pol.extra_widening_bps("") == Decimal("0")

    def test_fill_within_horizon_not_yet_processed(self) -> None:
        pol = _make(horizon_s=5)
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        assert pol.pending_count() == 1
        # Tick BEFORE deadline: pending stays
        n = pol.tick(now_ts=1001.0)
        assert n == 0
        assert pol.pending_count() == 1
        # No EWMA update → no widen
        assert pol.extra_widening_bps("BUY", now_ts=1001.0) == Decimal("0")

    def test_fill_with_positive_markout_does_not_widen(self) -> None:
        # We bought at 100, mid moved up to 100.05 → markout +5 bps (good)
        pol = _make()
        pol._mid_source = lambda: Decimal("100.05")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        assert pol.extra_widening_bps("BUY", now_ts=1005.0) == Decimal("0")


# ---------------------------------------------------------------------------
# 3. Negative markout → side widening
# ---------------------------------------------------------------------------


class TestWidening:
    def test_bad_buy_fill_widens_buy_side(self) -> None:
        # bought at 100, mid dropped to 99.95 → -5 bps markout
        pol = _make(threshold_bps=Decimal("2"), gain=Decimal("0.5"), cap_bps=Decimal("5"))
        pol._mid_source = lambda: Decimal("99.95")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        # widen = 0.5 * (|-5| - 2) = 1.5 bps
        widen = pol.extra_widening_bps("BUY", now_ts=1005.0)
        assert widen > 0
        assert abs(float(widen) - 1.5) < 1e-3

    def test_bad_buy_does_not_widen_sell(self) -> None:
        pol = _make()
        pol._mid_source = lambda: Decimal("99.95")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        assert pol.extra_widening_bps("SELL", now_ts=1005.0) == Decimal("0")

    def test_bad_sell_widens_sell_side(self) -> None:
        # sold at 100, mid rose to 100.05 → -5 bps markout from MM perspective
        pol = _make(threshold_bps=Decimal("2"), gain=Decimal("0.5"), cap_bps=Decimal("5"))
        pol._mid_source = lambda: Decimal("100.05")
        pol.on_fill(ts=1000.0, side="SELL", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen = pol.extra_widening_bps("SELL", now_ts=1005.0)
        assert widen > 0
        assert abs(float(widen) - 1.5) < 1e-3

    def test_widening_is_non_negative(self) -> None:
        # Even with positive markout the widening should be 0, never negative.
        pol = _make()
        pol._mid_source = lambda: Decimal("100.20")  # +20 bps good for BUY
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        assert pol.extra_widening_bps("BUY", now_ts=1005.0) >= 0


# ---------------------------------------------------------------------------
# 4. Cap behavior
# ---------------------------------------------------------------------------


class TestCap:
    def test_cap_clamps_extreme_markout(self) -> None:
        # Very bad markout: bought at 100, mid dropped to 99 → -100 bps
        # gain=0.5, threshold=2 → raw widen = 0.5 * (100 - 2) = 49 bps
        # cap=5 → should clamp to 5
        pol = _make(threshold_bps=Decimal("2"), gain=Decimal("0.5"), cap_bps=Decimal("5"))
        pol._mid_source = lambda: Decimal("99")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen = pol.extra_widening_bps("BUY", now_ts=1005.0)
        assert abs(float(widen) - 5.0) < 1e-3

    def test_zero_cap_means_no_widening(self) -> None:
        pol = _make(cap_bps=Decimal("0"))
        pol._mid_source = lambda: Decimal("99")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        assert pol.extra_widening_bps("BUY", now_ts=1005.0) == Decimal("0")


# ---------------------------------------------------------------------------
# 5. EWMA decay
# ---------------------------------------------------------------------------


class TestDecay:
    def test_ewma_decays_after_half_life(self) -> None:
        # Bad fill, then check that after half_life_s the EWMA is halved.
        pol = _make(half_life_s=Decimal("30"), threshold_bps=Decimal("0"),
                    gain=Decimal("1"), cap_bps=Decimal("100"))
        pol._mid_source = lambda: Decimal("99.96")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen_at_5 = pol.extra_widening_bps("BUY", now_ts=1005.0)
        # 30s later, EWMA should be halved (half_life decay)
        widen_at_35 = pol.extra_widening_bps("BUY", now_ts=1035.0)
        # ratio should be ~0.5 (the decay is multiplicative)
        ratio = float(widen_at_35) / float(widen_at_5)
        assert 0.45 < ratio < 0.55

    def test_zero_half_life_decays_instantly(self) -> None:
        # half_life_s = 0 → infinite decay rate → any time elapsed
        # after the EWMA update wipes it. The update itself is additive,
        # so the EWMA is briefly non-zero at update time but is decayed
        # to 0 on the next query at any later timestamp.
        pol = _make(half_life_s=Decimal("0"))
        pol._mid_source = lambda: Decimal("99.95")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        # Query at the same ts: no decay applied, EWMA still non-zero.
        # Query a moment later: instant decay kicks in → 0.
        widen_later = pol.extra_widening_bps("BUY", now_ts=1005.1)
        assert widen_later == Decimal("0")

    def test_accumulation_compounds_bad_fills(self) -> None:
        # Two consecutive bad BUY fills compound the EWMA (additive update).
        pol = _make(threshold_bps=Decimal("2"), gain=Decimal("0.5"),
                    cap_bps=Decimal("50"))
        pol._mid_source = lambda: Decimal("99.96")  # each fill = -4 bps
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen_after_1 = pol.extra_widening_bps("BUY", now_ts=1005.0)
        pol.on_fill(ts=1006.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1011.0)
        widen_after_2 = pol.extra_widening_bps("BUY", now_ts=1011.0)
        # Second widen should be larger (compound EWMA with some decay).
        assert widen_after_2 > widen_after_1


# ---------------------------------------------------------------------------
# 6. Stale orderbook handling
# ---------------------------------------------------------------------------


class TestStaleOrderbook:
    def test_mid_none_defers_to_next_tick(self) -> None:
        # mid_source returns None → fill stays pending.
        pol = _make()
        pol._mid_source = lambda: None
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        n = pol.tick(now_ts=1005.0)
        assert n == 0
        assert pol.pending_count() == 1
        # Once mid returns, the fill is processed.
        pol._mid_source = lambda: Decimal("99.95")
        n = pol.tick(now_ts=1006.0)
        assert n == 1
        assert pol.pending_count() == 0

    def test_zero_mid_defers_to_next_tick(self) -> None:
        pol = _make()
        pol._mid_source = lambda: Decimal("0")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        n = pol.tick(now_ts=1005.0)
        assert n == 0
        assert pol.pending_count() == 1


# ---------------------------------------------------------------------------
# 7. Edge cases / robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_unknown_side_in_on_fill_dropped(self) -> None:
        pol = _make()
        pol.on_fill(ts=1000.0, side="BOGUS", price=Decimal("100"))
        assert pol.pending_count() == 0

    def test_negative_price_in_on_fill_dropped(self) -> None:
        pol = _make()
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("-1"))
        assert pol.pending_count() == 0

    def test_zero_price_in_on_fill_dropped(self) -> None:
        pol = _make()
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("0"))
        assert pol.pending_count() == 0

    @pytest.mark.parametrize("side", ["BUY", "buy", "Buy"])
    def test_side_is_case_insensitive(self, side: str) -> None:
        pol = _make()
        pol._mid_source = lambda: Decimal("99.95")
        pol.on_fill(ts=1000.0, side=side, price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        assert pol.extra_widening_bps("BUY", now_ts=1005.0) > 0

    def test_processed_fills_removed_from_pending(self) -> None:
        pol = _make()
        pol._mid_source = lambda: Decimal("99.95")
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.on_fill(ts=1001.0, side="SELL", price=Decimal("100"))
        assert pol.pending_count() == 2
        pol.tick(now_ts=1010.0)
        assert pol.pending_count() == 0


# ---------------------------------------------------------------------------
# 8. Property tests
# ---------------------------------------------------------------------------


class TestProperty:
    @pytest.mark.parametrize("markout_bps", [-10, -5, -2, 0, 5, 10])
    def test_widening_non_negative_for_any_markout(self, markout_bps: float) -> None:
        # mid that produces the given markout for a BUY fill at 100
        mid = 100 * (1 + markout_bps / 1e4)
        pol = _make()
        pol._mid_source = lambda mid=mid: Decimal(str(mid))
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen = pol.extra_widening_bps("BUY", now_ts=1005.0)
        assert widen >= Decimal("0")

    @pytest.mark.parametrize("ewma_input", [-10, -5, -3, -2.5])
    def test_widening_monotone_in_negative_ewma(self, ewma_input: float) -> None:
        # Larger |negative markout| should produce larger widening.
        pol = _make(threshold_bps=Decimal("0"), gain=Decimal("1"), cap_bps=Decimal("100"))
        # Compute mid that would produce ewma_input for BUY at 100
        mid = 100 * (1 + ewma_input / 1e4)
        pol._mid_source = lambda mid=mid: Decimal(str(mid))
        pol.on_fill(ts=1000.0, side="BUY", price=Decimal("100"))
        pol.tick(now_ts=1005.0)
        widen = pol.extra_widening_bps("BUY", now_ts=1005.0)
        assert widen >= Decimal("0")
        # Should be approximately |ewma_input| × gain (with gain=1)
        assert abs(float(widen) - abs(ewma_input)) < 0.1
