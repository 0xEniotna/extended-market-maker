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
    "x10.perpetual.trading_client",
]

for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED",
    CANCELLED="CANCELLED",
    EXPIRED="EXPIRED",
    REJECTED="REJECTED",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT")
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT")
_orders_mod.OpenOrderModel = object


from market_maker.order_manager import OrderInfo, OrderManager  # noqa: E402

OrderSide = _orders_mod.OrderSide
OrderStatus = _orders_mod.OrderStatus


@pytest.mark.asyncio
async def test_place_order_non_ok_increments_failures():
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(status="ERR", error="bad", data=None)
    )
    manager = OrderManager(client, "TEST-USD")

    result = await manager.place_order(
        side=OrderSide.BUY,
        price=Decimal("10"),
        size=Decimal("1"),
        level=0,
    )

    assert result is None
    assert manager.consecutive_failures == 1


@pytest.mark.asyncio
async def test_place_order_success_resets_failures_and_indexes_exchange_id():
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(
            status="OK",
            error=None,
            data=SimpleNamespace(id=12345),
        )
    )
    manager = OrderManager(client, "TEST-USD")
    manager.consecutive_failures = 3

    ext_id = await manager.place_order(
        side=OrderSide.SELL,
        price=Decimal("11"),
        size=Decimal("2"),
        level=1,
    )

    assert ext_id is not None
    assert manager.consecutive_failures == 0
    info = manager.find_order_by_exchange_id("12345")
    assert info is not None
    assert info.level == 1


def test_rejected_terminal_update_increments_failures_and_keeps_mapping():
    manager = OrderManager(MagicMock(), "TEST-USD")
    info = OrderInfo(
        external_id="ext-1",
        side=OrderSide.BUY,
        price=Decimal("12"),
        size=Decimal("1"),
        level=0,
        exchange_order_id="999",
    )
    manager._active_orders["ext-1"] = info

    callback_args = []

    def _cb(*args, **kwargs):
        callback_args.append((args, kwargs))

    manager.on_level_freed(_cb)

    manager.handle_order_update(
        SimpleNamespace(
            external_id="ext-1",
            status=OrderStatus.REJECTED,
            status_reason="POST_ONLY_FAILED",
        )
    )

    assert manager.consecutive_failures == 1
    found = manager.find_order_by_exchange_id("999")
    assert found is not None
    assert found.external_id == "ext-1"
    assert callback_args
    _, kwargs = callback_args[0]
    assert kwargs["rejected"] is True
    assert "POST_ONLY_FAILED" in kwargs["reason"]


def test_failure_window_stats_and_reset():
    manager = OrderManager(MagicMock(), "TEST-USD")
    manager._record_attempt()
    manager._record_attempt()
    manager._record_failure()

    stats = manager.failure_window_stats(60.0)
    assert stats["attempts"] == 2.0
    assert stats["failures"] == 1.0
    assert stats["failure_rate"] == 0.5

    manager.reset_failure_tracking()
    stats_after = manager.failure_window_stats(60.0)
    assert stats_after["attempts"] == 0.0
    assert stats_after["failures"] == 0.0
    assert stats_after["failure_rate"] == 0.0
