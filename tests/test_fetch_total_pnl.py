from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path


def _load_module():
    path = Path("scripts/tools/fetch_total_pnl.py")
    spec = importlib.util.spec_from_file_location("fetch_total_pnl_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_since_timestamp_accepts_epoch_seconds():
    mod = _load_module()
    assert mod._parse_since_timestamp("1700000000") == 1_700_000_000_000


def test_parse_since_timestamp_accepts_epoch_milliseconds():
    mod = _load_module()
    assert mod._parse_since_timestamp("1700000000000") == 1_700_000_000_000


def test_parse_since_timestamp_accepts_iso_utc():
    mod = _load_module()
    assert mod._parse_since_timestamp("2026-02-01T00:00:00Z") == 1_769_904_000_000


def test_annualized_returns_for_one_year_window():
    mod = _load_module()
    total_return, apr, apy = mod._annualized_returns(
        total_pnl=Decimal("10"),
        starting_equity=Decimal("100"),
        period_days=Decimal("365"),
    )
    assert total_return == Decimal("0.1")
    assert apr == Decimal("0.1")
    assert apy is not None
    assert abs(apy - Decimal("0.1")) < Decimal("0.0000001")


def test_infer_starting_equity():
    mod = _load_module()
    assert mod._infer_starting_equity(Decimal("1200"), Decimal("200")) == Decimal("1000")
