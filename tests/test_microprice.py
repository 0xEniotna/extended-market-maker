"""Unit tests for the microprice fair-value estimator (Stoikov 2018).

Covers formula correctness, the cross-weighting direction (bid-heavy ⇒
microprice above mid), degenerate-book fallbacks to mid, exact-bounds, and
Decimal precision (no float drift). The live-quoting integration / rollback
guarantees live in ``test_microprice_integration.py``.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from market_maker.microprice import microprice


def _mid(bid: Decimal, ask: Decimal) -> Decimal:
    return (bid + ask) / 2


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------


class TestFormula:
    def test_balanced_book_equals_mid(self) -> None:
        bid, ask = Decimal("99.95"), Decimal("100.05")
        mp = microprice(bid, ask, Decimal("100"), Decimal("100"))
        assert mp == _mid(bid, ask)

    def test_known_exact_value(self) -> None:
        # micro = (99·1 + 101·3) / (3 + 1) = 402/4 = 100.5
        mp = microprice(Decimal("99"), Decimal("101"), Decimal("3"), Decimal("1"))
        assert mp == Decimal("100.5")

    def test_bid_heavy_book_leans_above_mid(self) -> None:
        # More size resting on the bid ⇒ predicted up-move ⇒ microprice > mid.
        bid, ask = Decimal("99.95"), Decimal("100.05")
        mp = microprice(bid, ask, Decimal("900"), Decimal("100"))
        assert mp > _mid(bid, ask)

    def test_ask_heavy_book_leans_below_mid(self) -> None:
        bid, ask = Decimal("99.95"), Decimal("100.05")
        mp = microprice(bid, ask, Decimal("100"), Decimal("900"))
        assert mp < _mid(bid, ask)

    def test_always_within_bid_ask(self) -> None:
        bid, ask = Decimal("100"), Decimal("100.10")
        for bq, aq in [
            (Decimal("1"), Decimal("999")),
            (Decimal("999"), Decimal("1")),
            (Decimal("7"), Decimal("13")),
        ]:
            mp = microprice(bid, ask, bq, aq)
            assert bid <= mp <= ask

    def test_monotone_in_bid_share(self) -> None:
        # Increasing bid_qty (holding ask_qty) monotonically raises microprice.
        bid, ask = Decimal("50"), Decimal("50.20")
        prev = microprice(bid, ask, Decimal("1"), Decimal("100"))
        for bq in (Decimal("10"), Decimal("100"), Decimal("1000")):
            cur = microprice(bid, ask, bq, Decimal("100"))
            assert cur > prev
            prev = cur


# ---------------------------------------------------------------------------
# Degenerate books fall back to mid
# ---------------------------------------------------------------------------


class TestFallback:
    @pytest.mark.parametrize("bq,aq", [
        (Decimal("0"), Decimal("100")),   # empty bid
        (Decimal("100"), Decimal("0")),   # empty ask
        (Decimal("0"), Decimal("0")),     # empty both
        (Decimal("-5"), Decimal("100")),  # defensive: negative qty
        (Decimal("100"), Decimal("-5")),
    ])
    def test_nonpositive_qty_returns_mid(self, bq: Decimal, aq: Decimal) -> None:
        bid, ask = Decimal("99.95"), Decimal("100.05")
        assert microprice(bid, ask, bq, aq) == _mid(bid, ask)


# ---------------------------------------------------------------------------
# Decimal precision — the function must not silently use float arithmetic
# ---------------------------------------------------------------------------


class TestPrecision:
    def test_returns_decimal(self) -> None:
        mp = microprice(Decimal("99.95"), Decimal("100.05"),
                        Decimal("100"), Decimal("100"))
        assert isinstance(mp, Decimal)

    def test_no_float_drift(self) -> None:
        # 100.1 + 100.3 = 200.4 exactly in Decimal (→ /2 = 100.2), but in
        # IEEE-754 float it is 200.39999999999998. A float implementation
        # would return 100.19999999999999; Decimal must return exactly 100.2.
        mp = microprice(Decimal("100.1"), Decimal("100.3"),
                        Decimal("1"), Decimal("1"))
        assert mp == Decimal("100.2")
        assert mp != Decimal(str((100.1 + 100.3) / 2))

    def test_high_value_tick_precision(self) -> None:
        # TECH100m-style mid ~$29k with a 2-tick spread; balanced book ⇒
        # exact mid with no rounding artifact.
        bid, ask = Decimal("29000.01"), Decimal("29000.03")
        mp = microprice(bid, ask, Decimal("5"), Decimal("5"))
        assert mp == Decimal("29000.02")
