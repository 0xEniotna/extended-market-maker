from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.positions",
    "x10.perpetual.trading_client",
]

for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")

_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED")


from market_maker.risk_manager import RiskManager  # noqa: E402

OrderSide = _orders_mod.OrderSide


def _make_rm(
    *,
    max_position_size: Decimal = Decimal("100"),
    max_order_notional_usd: Decimal = Decimal("0"),
    max_position_notional_usd: Decimal = Decimal("0"),
) -> RiskManager:
    return RiskManager(
        trading_client=MagicMock(),
        market_name="TEST-USD",
        max_position_size=max_position_size,
        max_order_notional_usd=max_order_notional_usd,
        max_position_notional_usd=max_position_notional_usd,
    )


def test_allowed_order_size_clips_by_position_size_headroom():
    rm = _make_rm(max_position_size=Decimal("100"))
    rm._cached_position = Decimal("95")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("10"),
        reference_price=Decimal("10"),
    )
    assert allowed == Decimal("5")


def test_allowed_order_size_clips_by_order_notional():
    rm = _make_rm(max_order_notional_usd=Decimal("250"))
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("5"),
        reference_price=Decimal("100"),
    )
    assert allowed == Decimal("2.5")


def test_allowed_order_size_clips_by_position_notional():
    rm = _make_rm(max_position_notional_usd=Decimal("2500"))
    rm._cached_position = Decimal("20")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("10"),
        reference_price=Decimal("100"),
    )
    assert allowed == Decimal("5")


def test_allowed_order_size_zero_when_position_notional_limit_already_reached():
    rm = _make_rm(max_position_notional_usd=Decimal("1000"))
    rm._cached_position = Decimal("10")
    allowed = rm.allowed_order_size(
        side=OrderSide.SELL,
        requested_size=Decimal("3"),
        reference_price=Decimal("100"),
    )
    assert allowed == Decimal("0")


def test_allowed_order_size_accounts_for_reserved_same_side_qty():
    rm = _make_rm(max_position_size=Decimal("100"))
    rm._cached_position = Decimal("70")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("40"),
        reference_price=Decimal("10"),
        reserved_same_side_qty=Decimal("20"),
    )
    assert allowed == Decimal("10")
