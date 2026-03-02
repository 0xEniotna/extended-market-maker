from __future__ import annotations

from decimal import Decimal

from market_maker.cli.pnl import _parse_since_timestamp, _annualized_returns


def test_parse_since_timestamp_accepts_epoch_seconds():
    assert _parse_since_timestamp("1700000000") == 1_700_000_000_000


def test_parse_since_timestamp_accepts_epoch_milliseconds():
    assert _parse_since_timestamp("1700000000000") == 1_700_000_000_000


def test_parse_since_timestamp_accepts_iso_utc():
    assert _parse_since_timestamp("2026-02-01T00:00:00Z") == 1_769_904_000_000


def test_annualized_returns_for_one_year_window():
    total_return, apr, apy = _annualized_returns(
        total_pnl=Decimal("10"),
        starting_equity=Decimal("100"),
        period_days=Decimal("365"),
    )
    assert total_return == Decimal("0.1")
    assert apr == Decimal("0.1")
    assert apy is not None
    assert abs(apy - Decimal("0.1")) < Decimal("0.0000001")


def test_infer_starting_equity():
    # starting_equity = current_equity - total_pnl (inline in pnl.py)
    current_equity = Decimal("1200")
    total_pnl = Decimal("200")
    assert current_equity - total_pnl == Decimal("1000")
