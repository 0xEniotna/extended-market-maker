from __future__ import annotations

import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

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
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")


from market_maker.risk_manager import RiskManager  # noqa: E402

OrderSide = _orders_mod.OrderSide


def _make_rm(
    *,
    max_position_size: Decimal = Decimal("100"),
    max_order_notional_usd: Decimal = Decimal("0"),
    max_position_notional_usd: Decimal = Decimal("0"),
    balance_aware_sizing_enabled: bool = False,
    balance_usage_factor: Decimal = Decimal("0.95"),
    balance_notional_multiplier: Decimal = Decimal("1.0"),
    balance_min_available_usd: Decimal = Decimal("0"),
    balance_staleness_max_s: float = 0,
) -> RiskManager:
    return RiskManager(
        trading_client=MagicMock(),
        market_name="TEST-USD",
        max_position_size=max_position_size,
        max_order_notional_usd=max_order_notional_usd,
        max_position_notional_usd=max_position_notional_usd,
        balance_aware_sizing_enabled=balance_aware_sizing_enabled,
        balance_usage_factor=balance_usage_factor,
        balance_notional_multiplier=balance_notional_multiplier,
        balance_min_available_usd=balance_min_available_usd,
        balance_staleness_max_s=balance_staleness_max_s,
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


def test_allowed_order_size_clips_by_available_for_trade_headroom():
    rm = _make_rm(
        balance_aware_sizing_enabled=True,
        balance_usage_factor=Decimal("0.9"),
        balance_notional_multiplier=Decimal("2.0"),
        balance_min_available_usd=Decimal("10"),
    )
    rm._cached_available_for_trade = Decimal("100")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("10"),
        reference_price=Decimal("20"),
        reserved_open_notional_usd=Decimal("100"),
    )
    # usable_notional = (100*0.9 - 10) * 2 - 100 = 60 => size 3
    assert allowed == Decimal("3")


def test_handle_balance_update_refreshes_cached_available_for_trade():
    rm = _make_rm(balance_aware_sizing_enabled=True)
    rm.handle_balance_update(
        SimpleNamespace(
            available_for_trade=Decimal("123.45"),
            equity=Decimal("150"),
            initial_margin=Decimal("12"),
        )
    )
    assert rm.get_available_for_trade() == Decimal("123.45")


def test_reducing_order_bypasses_balance_clip_when_not_crossing_flat():
    rm = _make_rm(
        balance_aware_sizing_enabled=True,
        balance_usage_factor=Decimal("1"),
        balance_notional_multiplier=Decimal("1"),
        balance_min_available_usd=Decimal("0"),
    )
    rm._cached_position = Decimal("-68")
    rm._cached_available_for_trade = Decimal("1500")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("20"),
        reference_price=Decimal("100"),
        reserved_open_notional_usd=Decimal("14000"),
    )
    assert allowed == Decimal("20")


def test_reducing_order_only_balance_clips_opening_remainder():
    rm = _make_rm(
        balance_aware_sizing_enabled=True,
        balance_usage_factor=Decimal("1"),
        balance_notional_multiplier=Decimal("1"),
        balance_min_available_usd=Decimal("0"),
    )
    rm._cached_position = Decimal("-68")
    rm._cached_available_for_trade = Decimal("1500")
    allowed = rm.allowed_order_size(
        side=OrderSide.BUY,
        requested_size=Decimal("80"),
        reference_price=Decimal("100"),
        reserved_open_notional_usd=Decimal("1400"),
    )
    # Reducing component (68) is preserved, opening component (12) is clipped by
    # remaining notional headroom: (1500 - 1400) / 100 = 1.
    assert allowed == Decimal("69")


def test_handle_position_update_tracks_total_position_pnl():
    rm = _make_rm()
    rm.handle_position_update(
        SimpleNamespace(
            market="TEST-USD",
            status=_positions_mod.PositionStatus.OPENED,
            side=_positions_mod.PositionSide.LONG,
            size=Decimal("7"),
            realised_pnl=Decimal("12.5"),
            unrealised_pnl=Decimal("-2.0"),
        )
    )

    assert rm.get_current_position() == Decimal("7")
    assert rm.get_position_realized_pnl() == Decimal("12.5")
    assert rm.get_position_unrealized_pnl() == Decimal("-2.0")
    assert rm.get_position_total_pnl() == Decimal("10.5")


def test_handle_position_update_closed_resets_position_pnl():
    rm = _make_rm()
    rm.handle_position_update(
        SimpleNamespace(
            market="TEST-USD",
            status=_positions_mod.PositionStatus.OPENED,
            side=_positions_mod.PositionSide.LONG,
            size=Decimal("2"),
            realised_pnl=Decimal("1"),
            unrealised_pnl=Decimal("3"),
        )
    )
    rm.handle_position_update(
        SimpleNamespace(
            market="TEST-USD",
            status=_positions_mod.PositionStatus.CLOSED,
            side=_positions_mod.PositionSide.SHORT,
            size=Decimal("0"),
            realised_pnl=Decimal("0"),
            unrealised_pnl=Decimal("0"),
        )
    )

    assert rm.get_current_position() == Decimal("0")
    assert rm.get_position_total_pnl() == Decimal("0")


@pytest.mark.asyncio
async def test_refresh_position_updates_cached_position_pnl():
    rm = _make_rm()
    rm._client.account.get_positions = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(
                    market="TEST-USD",
                    size=Decimal("5"),
                    side="LONG",
                    realised_pnl=Decimal("-4.5"),
                    unrealised_pnl=Decimal("1.0"),
                )
            ]
        )
    )

    await rm.refresh_position()

    assert rm.get_current_position() == Decimal("5")
    assert rm.get_position_total_pnl() == Decimal("-3.5")
