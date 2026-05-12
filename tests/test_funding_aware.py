"""Unit tests for the funding-aware LQ overlay policy."""
from __future__ import annotations

from decimal import Decimal

import pytest

from market_maker.funding_aware import (
    FundingAwareConfig,
    FundingAwarePolicy,
    make_policy_if_enabled,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(
    *,
    enabled: bool = True,
    coupling_bps_max: Decimal = Decimal("8"),
    hold_horizon_periods: Decimal = Decimal("4"),
    dollar_cap_pct_of_notional: Decimal = Decimal("0.001"),
    funding_rate: Decimal = Decimal("0.0001"),
) -> FundingAwarePolicy:
    cfg = FundingAwareConfig(
        enabled=enabled,
        coupling_bps_max=coupling_bps_max,
        hold_horizon_periods=hold_horizon_periods,
        dollar_cap_pct_of_notional=dollar_cap_pct_of_notional,
    )
    return FundingAwarePolicy(cfg, funding_rate_source=lambda: funding_rate)


_MID = Decimal("100")
_MAX_POS = Decimal("10")


# ---------------------------------------------------------------------------
# 1. Flag-off → no signal
# ---------------------------------------------------------------------------


class TestDisabled:
    def test_disabled_returns_zero(self) -> None:
        pol = _make(enabled=False, funding_rate=Decimal("0.01"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")
        assert not pol.enabled

    def test_factory_returns_none_when_disabled(self) -> None:
        out = make_policy_if_enabled(
            enabled=False,
            coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.001"),
            funding_rate_source=lambda: Decimal("0.001"),
        )
        assert out is None

    def test_factory_returns_policy_when_enabled(self) -> None:
        out = make_policy_if_enabled(
            enabled=True,
            coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.001"),
            funding_rate_source=lambda: Decimal("0.001"),
        )
        assert out is not None
        assert out.enabled


# ---------------------------------------------------------------------------
# 2. Zero/edge inputs → no signal
# ---------------------------------------------------------------------------


class TestZeroInputs:
    def test_zero_funding_rate(self) -> None:
        pol = _make(funding_rate=Decimal("0"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")

    def test_zero_mid_price(self) -> None:
        pol = _make()
        out = pol.compute_funding_signal_bps(
            mid_price=Decimal("0"), max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")

    def test_negative_mid_price(self) -> None:
        pol = _make()
        out = pol.compute_funding_signal_bps(
            mid_price=Decimal("-1"), max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")

    def test_zero_horizon(self) -> None:
        pol = _make(hold_horizon_periods=Decimal("0"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")

    def test_zero_max_position(self) -> None:
        pol = _make()
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=Decimal("0"),
        )
        assert out == Decimal("0")


# ---------------------------------------------------------------------------
# 3. Sign of the signal tracks sign(F), not sign(q)
# ---------------------------------------------------------------------------


class TestSignBehavior:
    def test_positive_funding_returns_positive_signal(self) -> None:
        pol = _make(funding_rate=Decimal("0.0001"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out > 0

    def test_negative_funding_returns_negative_signal(self) -> None:
        pol = _make(funding_rate=Decimal("-0.0001"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out < 0

    def test_sign_is_symmetric_around_zero(self) -> None:
        pol_pos = _make(funding_rate=Decimal("0.00005"))
        pol_neg = _make(funding_rate=Decimal("-0.00005"))
        a = pol_pos.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        b = pol_neg.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert a == -b


# ---------------------------------------------------------------------------
# 4. Saturation: bps cap and dollar cap
# ---------------------------------------------------------------------------


class TestSaturation:
    def test_bps_cap_clamps_large_signal(self) -> None:
        # Huge funding rate ⇒ should hit cap.
        pol = _make(
            funding_rate=Decimal("0.5"),  # 50% per period, absurd
            coupling_bps_max=Decimal("8"),
            dollar_cap_pct_of_notional=Decimal("1"),  # disable dollar saturation
        )
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("8")

    def test_bps_cap_clamps_large_negative(self) -> None:
        pol = _make(
            funding_rate=Decimal("-0.5"),
            coupling_bps_max=Decimal("8"),
            dollar_cap_pct_of_notional=Decimal("1"),
        )
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("-8")

    def test_dollar_cap_dominates_when_smaller(self) -> None:
        # Set dollar_cap_pct very small so it kicks in before bps cap.
        # max_notional = 10 * 100 = 1000. dollar_cap = 0.0001 * 1000 = 0.1 USD.
        # With F=0.001, H=4, mid=100: funding_dollar raw = 100 * 0.001 * 4 = 0.4 USD
        # → clamped to 0.1 USD → signal_bps = 0.1 / 100 * 1e4 = 10 bps
        # → still > coupling_bps_max of 8 ⇒ final = 8 bps.
        pol = _make(
            funding_rate=Decimal("0.001"),
            coupling_bps_max=Decimal("8"),
            dollar_cap_pct_of_notional=Decimal("0.0001"),
        )
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        # Dollar cap clamps to 0.1 USD → 10 bps raw → bps cap reduces to 8.
        assert out == Decimal("8")

    def test_dollar_cap_pct_zero_returns_zero(self) -> None:
        pol = _make(dollar_cap_pct_of_notional=Decimal("0"))
        out = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out == Decimal("0")


# ---------------------------------------------------------------------------
# 5. Horizon scaling
# ---------------------------------------------------------------------------


class TestHorizonScaling:
    def test_doubling_horizon_doubles_signal_below_cap(self) -> None:
        # Pick F + caps so we stay below the cap at H=2 and at H=4.
        # F=1e-6, H=2: funding_dollar = 100 * 1e-6 * 2 = 2e-4 USD per contract.
        #   max_notional = 1000, dollar_cap_pct = 0.001 ⇒ dollar_cap = 1 USD.
        #   signal_bps = 2e-4 / 100 * 1e4 = 0.02 bps. Far under 8 bps cap.
        pol_h2 = _make(
            funding_rate=Decimal("0.000001"),
            hold_horizon_periods=Decimal("2"),
        )
        pol_h4 = _make(
            funding_rate=Decimal("0.000001"),
            hold_horizon_periods=Decimal("4"),
        )
        out_h2 = pol_h2.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        out_h4 = pol_h4.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        # Allow tiny Decimal rounding tolerance.
        assert abs(out_h4 - 2 * out_h2) < Decimal("1e-12")


# ---------------------------------------------------------------------------
# 6. Decimal precision: no float drift across many evaluations
# ---------------------------------------------------------------------------


class TestPrecision:
    @pytest.mark.parametrize(
        "mid",
        [Decimal("0.0001"), Decimal("1.23456789"), Decimal("100"), Decimal("100000")],
    )
    def test_no_decimal_drift(self, mid: Decimal) -> None:
        pol = _make(funding_rate=Decimal("0.0001"))
        out = pol.compute_funding_signal_bps(
            mid_price=mid, max_position_size=_MAX_POS,
        )
        # Re-evaluation must produce the exact same Decimal.
        out2 = pol.compute_funding_signal_bps(
            mid_price=mid, max_position_size=_MAX_POS,
        )
        assert out == out2

    def test_funding_rate_source_is_re_read_each_call(self) -> None:
        # The policy reads the source callable every call, so a funding
        # rate update propagates without re-construction.
        rate = [Decimal("0.0001")]
        cfg = FundingAwareConfig(
            enabled=True,
            coupling_bps_max=Decimal("8"),
            hold_horizon_periods=Decimal("4"),
            dollar_cap_pct_of_notional=Decimal("0.001"),
        )
        pol = FundingAwarePolicy(cfg, funding_rate_source=lambda: rate[0])
        out_a = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        rate[0] = Decimal("-0.0001")
        out_b = pol.compute_funding_signal_bps(
            mid_price=_MID, max_position_size=_MAX_POS,
        )
        assert out_a > 0
        assert out_b < 0
        assert out_a == -out_b
