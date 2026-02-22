from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path("scripts/tools/close_mm_position.py")
    spec = importlib.util.spec_from_file_location("close_mm_position_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_top_prices_from_dict_payload() -> None:
    mod = _load_module()
    orderbook = {
        "b": [{"p": "99.5", "q": "1"}],
        "a": [{"p": "100.5", "q": "1"}],
    }

    best_bid, best_ask = mod._extract_top_prices(orderbook)

    assert best_bid == Decimal("99.5")
    assert best_ask == Decimal("100.5")


def test_extract_top_prices_from_object_payload() -> None:
    mod = _load_module()
    bid_level = SimpleNamespace(price=Decimal("12.3"))
    ask_level = SimpleNamespace(price=Decimal("12.7"))
    orderbook = SimpleNamespace(bid=[bid_level], ask=[ask_level])

    best_bid, best_ask = mod._extract_top_prices(orderbook)

    assert best_bid == Decimal("12.3")
    assert best_ask == Decimal("12.7")


def test_signed_position_for_market_sums_long_and_short() -> None:
    mod = _load_module()
    positions = [
        SimpleNamespace(market="ETH-USD", side="LONG", size="2.5"),
        SimpleNamespace(market="ETH-USD", side="SHORT", size="1.0"),
        SimpleNamespace(market="BTC-USD", side="SHORT", size="5.0"),
    ]

    signed = mod._signed_position_for_market(positions, "ETH-USD")

    assert signed == Decimal("1.5")
