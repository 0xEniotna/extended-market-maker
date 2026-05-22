"""Integration + rollback tests for the microprice recentering (Stage 3 / A1.2).

Mirrors ``test_funding_aware_integration.py``. Asserts:
1. Flag-off ⇒ quotes byte-identical to a pre-feature reference engine, even
   with a strongly imbalanced book and non-zero inventory.
2. Flag-on + crypto ⇒ quotes shift in the correct direction (bid-heavy book
   ⇒ both quotes up; ask-heavy ⇒ both down) by exactly (microprice − mid).
3. Flag-on + balanced book ⇒ no-op (microprice == mid).
4. Flag-on + non-crypto profile ⇒ no-op (the crypto gate blocks it), so
   enabling the flag on a TradFi market can never apply the wrong-sign shift.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# SDK module stubs (mirror the convention used in other test files)
# ---------------------------------------------------------------------------

_SDK_MODULES = [
    "x10", "x10.perpetual", "x10.perpetual.orders",
    "x10.perpetual.trading_client", "x10.perpetual.positions",
    "x10.perpetual.accounts", "x10.perpetual.configuration",
    "x10.perpetual.orderbook", "x10.perpetual.trades",
    "x10.perpetual.stream_client", "x10.perpetual.stream_client.stream_client",
    "x10.utils", "x10.utils.http",
]
for _mod_name in _SDK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from market_maker.pricing_engine import PricingEngine  # noqa: E402

_BUY = "BUY"
_SELL = "SELL"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _FakeOB:
    """Top-of-book with independently configurable bid/ask sizes."""

    def __init__(
        self,
        bid: Decimal,
        ask: Decimal,
        bid_size: Decimal = Decimal("100"),
        ask_size: Decimal = Decimal("100"),
    ) -> None:
        self._bid = SimpleNamespace(price=bid, size=bid_size)
        self._ask = SimpleNamespace(price=ask, size=ask_size)

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


def _make_settings(market_profile: str = "crypto") -> SimpleNamespace:
    # Deliberately omits ``use_microprice`` — proves the engine reads the
    # flag from its constructor, not from settings (so legacy fixtures and
    # call sites that never heard of the field keep working unchanged).
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
        market_profile=market_profile,
        size_scale_per_level=Decimal("1.2"),
    )


def _make_engine(
    *,
    use_microprice: bool,
    market_profile: str = "crypto",
    position: Decimal = Decimal("0"),
    bid: Decimal = Decimal("99.95"),
    ask: Decimal = Decimal("100.05"),
    bid_size: Decimal = Decimal("100"),
    ask_size: Decimal = Decimal("100"),
    microprice_cap_bps: Decimal = Decimal("10"),
) -> PricingEngine:
    return PricingEngine(
        settings=_make_settings(market_profile),
        orderbook_mgr=_FakeOB(bid, ask, bid_size, ask_size),
        risk_mgr=_FakeRisk(position),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.1"),
        use_microprice=use_microprice,
        microprice_cap_bps=microprice_cap_bps,
    )


def _reference_engine(
    *,
    market_profile: str = "crypto",
    position: Decimal = Decimal("0"),
    bid_size: Decimal = Decimal("100"),
    ask_size: Decimal = Decimal("100"),
) -> PricingEngine:
    # Mimics the pre-feature call site: does NOT pass use_microprice at all.
    return PricingEngine(
        settings=_make_settings(market_profile),
        orderbook_mgr=_FakeOB(Decimal("99.95"), Decimal("100.05"), bid_size, ask_size),
        risk_mgr=_FakeRisk(position),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.1"),
    )


# ---------------------------------------------------------------------------
# 1. Flag-off rollback — byte-identical to the pre-feature engine
# ---------------------------------------------------------------------------


class TestRollbackFlagOff:
    @pytest.mark.parametrize("position", [
        Decimal("0"), Decimal("3"), Decimal("-3"), Decimal("8"), Decimal("-8"),
    ])
    @pytest.mark.parametrize("bid_size,ask_size", [
        (Decimal("100"), Decimal("100")),   # balanced
        (Decimal("900"), Decimal("100")),   # bid-heavy
        (Decimal("100"), Decimal("900")),   # ask-heavy
    ])
    def test_flag_off_matches_reference(
        self, position: Decimal, bid_size: Decimal, ask_size: Decimal,
    ) -> None:
        ref = _reference_engine(position=position, bid_size=bid_size, ask_size=ask_size)
        off = _make_engine(
            use_microprice=False, position=position,
            bid_size=bid_size, ask_size=ask_size,
        )
        for side in (_BUY, _SELL):
            for level in (0, 1):
                a = ref.compute_target_price(side, level, Decimal("100"))
                b = off.compute_target_price(side, level, Decimal("100"))
                assert a == b, (
                    f"Quote drift side={side} level={level} pos={position} "
                    f"sizes=({bid_size},{ask_size}): ref={a} off={b}"
                )


# ---------------------------------------------------------------------------
# 2. Flag-on + crypto — correct direction and exact magnitude
# ---------------------------------------------------------------------------


class TestPerturbationDirection:
    """bid_size=900, ask_size=100 ⇒ microprice = 100.04, mid = 100.00 ⇒
    shift = +0.04 (exactly 4 ticks), so both quotes move UP by exactly 0.04.
    ask-heavy mirrors it downward."""

    def test_bid_heavy_shifts_both_quotes_up(self) -> None:
        base = _make_engine(use_microprice=False,
                            bid_size=Decimal("900"), ask_size=Decimal("100"))
        on = _make_engine(use_microprice=True,
                          bid_size=Decimal("900"), ask_size=Decimal("100"))
        b_base = base.compute_target_price(_BUY, 0, Decimal("100"))
        b_on = on.compute_target_price(_BUY, 0, Decimal("100"))
        a_base = base.compute_target_price(_SELL, 0, Decimal("100"))
        a_on = on.compute_target_price(_SELL, 0, Decimal("100"))
        # microprice (100.04) > mid (100.00) ⇒ lean up.
        assert b_on > b_base
        assert a_on > a_base
        # Exact: shift is a whole number of ticks, so round-to-tick is linear.
        assert b_on - b_base == Decimal("0.04")
        assert a_on - a_base == Decimal("0.04")

    def test_ask_heavy_shifts_both_quotes_down(self) -> None:
        base = _make_engine(use_microprice=False,
                            bid_size=Decimal("100"), ask_size=Decimal("900"))
        on = _make_engine(use_microprice=True,
                          bid_size=Decimal("100"), ask_size=Decimal("900"))
        b_base = base.compute_target_price(_BUY, 0, Decimal("100"))
        b_on = on.compute_target_price(_BUY, 0, Decimal("100"))
        a_base = base.compute_target_price(_SELL, 0, Decimal("100"))
        a_on = on.compute_target_price(_SELL, 0, Decimal("100"))
        # microprice (99.96) < mid (100.00) ⇒ lean down.
        assert b_on < b_base
        assert a_on < a_base
        assert b_base - b_on == Decimal("0.04")
        assert a_base - a_on == Decimal("0.04")


# ---------------------------------------------------------------------------
# 3. Flag-on + balanced book — no perturbation
# ---------------------------------------------------------------------------


class TestBalancedBookNoOp:
    def test_balanced_book_is_byte_identical(self) -> None:
        off = _make_engine(use_microprice=False,
                          bid_size=Decimal("250"), ask_size=Decimal("250"))
        on = _make_engine(use_microprice=True,
                         bid_size=Decimal("250"), ask_size=Decimal("250"))
        for side in (_BUY, _SELL):
            assert (off.compute_target_price(side, 0, Decimal("100"))
                    == on.compute_target_price(side, 0, Decimal("100")))


# ---------------------------------------------------------------------------
# 4. Flag-on + non-crypto profile — crypto gate blocks the (wrong-sign) shift
# ---------------------------------------------------------------------------


class TestShiftCap:
    """A dislocated book (wide spread, 10000:1 imbalance) would shift ~50 bps
    uncapped; microprice_cap_bps must clip it. Guards the 75 bps tail the
    Stage 3 DOT replay surfaced."""

    _BOOK = dict(bid=Decimal("100"), ask=Decimal("101"),
                 bid_size=Decimal("10000"), ask_size=Decimal("1"))

    def test_cap_clips_dislocation_tail(self) -> None:
        off = _make_engine(use_microprice=False, **self._BOOK)
        capped = _make_engine(use_microprice=True,
                              microprice_cap_bps=Decimal("2"), **self._BOOK)
        uncapped = _make_engine(use_microprice=True,
                                microprice_cap_bps=Decimal("1000"), **self._BOOK)
        b_off = off.compute_target_price(_BUY, 0, Decimal("100"))
        b_capped = capped.compute_target_price(_BUY, 0, Decimal("100"))
        b_uncapped = uncapped.compute_target_price(_BUY, 0, Decimal("100"))

        shift_capped = abs(b_capped - b_off)
        shift_uncapped = abs(b_uncapped - b_off)
        # The cap must bite hard: ~2 bps of mid (100.5) ≈ 0.02, vs ~50 bps
        # uncapped ≈ 0.50.
        assert shift_capped < shift_uncapped
        assert shift_capped <= Decimal("0.03")    # ≈ 2 bps + 1 tick
        assert shift_uncapped >= Decimal("0.10")  # uncapped tail is large


class TestCryptoGate:
    @pytest.mark.parametrize("bid_size,ask_size", [
        (Decimal("900"), Decimal("100")),
        (Decimal("100"), Decimal("900")),
    ])
    def test_legacy_profile_ignores_microprice(
        self, bid_size: Decimal, ask_size: Decimal,
    ) -> None:
        off = _make_engine(use_microprice=False, market_profile="legacy",
                          bid_size=bid_size, ask_size=ask_size)
        on = _make_engine(use_microprice=True, market_profile="legacy",
                         bid_size=bid_size, ask_size=ask_size)
        for side in (_BUY, _SELL):
            assert (off.compute_target_price(side, 0, Decimal("100"))
                    == on.compute_target_price(side, 0, Decimal("100"))), (
                "Microprice must be a no-op on non-crypto profiles "
                "(Stage 3 showed the wrong sign on TradFi)."
            )
