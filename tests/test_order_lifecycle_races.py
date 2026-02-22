"""Tests for order lifecycle race condition fixes.

Covers:
1. cancel_all_orders pending_cancel state + timeout sweep
2. Fill deduplication via _seen_trade_ids
3. Pending-placement buffer (stream arrives before place_order returns)
4. Zombie order detection
5. flatten_position wait_for_fill mode
"""
from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from decimal import Decimal
from types import SimpleNamespace
from typing import Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# SDK module stubs (must precede imports from market_maker)
# ---------------------------------------------------------------------------

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.trading_client",
    "x10.perpetual.positions",
    "x10.perpetual.accounts",
    "x10.perpetual.configuration",
    "x10.perpetual.orderbook",
    "x10.perpetual.trades",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.utils",
    "x10.utils.http",
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
    OPEN="OPEN",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT", MARKET="MARKET")
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT", IOC="IOC")
_orders_mod.OpenOrderModel = object
_orders_mod.OrderStatusReason = SimpleNamespace(POST_ONLY_FAILED="POST_ONLY_FAILED")

_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")

from market_maker.order_manager import OrderInfo, OrderManager  # noqa: E402
from market_maker.strategy_callbacks import on_fill, on_level_freed  # noqa: E402
from market_maker.account_stream import FillEvent  # noqa: E402

OrderSide = _orders_mod.OrderSide
OrderStatus = _orders_mod.OrderStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> OrderManager:
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(
            status="OK",
            error=None,
            data=SimpleNamespace(id=1),
        )
    )
    client.orders = MagicMock()
    client.orders.cancel_order_by_external_id = AsyncMock()
    client.orders.mass_cancel = AsyncMock()
    return OrderManager(client, "TEST-USD")


def _add_order(
    mgr: OrderManager,
    ext_id: str,
    *,
    side=None,
    price: Decimal = Decimal("100"),
    size: Decimal = Decimal("1"),
    level: int = 0,
    placed_at: float | None = None,
    last_stream_update_at: float | None = None,
) -> OrderInfo:
    if side is None:
        side = OrderSide.BUY
    info = OrderInfo(
        external_id=ext_id,
        side=side,
        price=price,
        size=size,
        level=level,
        placed_at=placed_at if placed_at is not None else time.monotonic(),
        last_stream_update_at=last_stream_update_at,
    )
    mgr._active_orders[ext_id] = info
    return info


def _make_strategy_stub():
    """Minimal object that satisfies strategy_callbacks.on_fill expectations."""
    stub = SimpleNamespace()
    stub._ob = MagicMock()
    stub._ob.best_bid = MagicMock(return_value=SimpleNamespace(price=Decimal("100")))
    stub._ob.best_ask = MagicMock(return_value=SimpleNamespace(price=Decimal("100.1")))
    stub._ob.market_snapshot = MagicMock(return_value={})
    stub._orders = MagicMock()
    stub._orders.find_order_by_exchange_id = MagicMock(return_value=None)
    stub._risk = MagicMock()
    stub._risk.get_current_position = MagicMock(return_value=Decimal("0"))
    stub._journal = MagicMock()
    stub._settings = SimpleNamespace(
        fill_snapshot_depth=5,
        micro_vol_window_s=5.0,
        micro_drift_window_s=3.0,
        imbalance_window_s=2.0,
    )
    stub._reset_pof_state = MagicMock()
    stub._seen_trade_ids: deque = deque()
    stub._seen_trade_ids_set: Set[int] = set()
    return stub


# ===================================================================
# 1. cancel_all_orders: pending_cancel state with timeout sweep
# ===================================================================

class TestCancelAllPendingCancel:
    """cancel_all_orders should NOT clear _active_orders immediately."""

    @pytest.mark.asyncio
    async def test_cancel_all_marks_pending_cancel_instead_of_clearing(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")
        _add_order(mgr, "ext-2")

        await mgr.cancel_all_orders()

        # Orders should still be in _active_orders
        assert "ext-1" in mgr._active_orders
        assert "ext-2" in mgr._active_orders
        # And should be in _pending_cancel
        assert "ext-1" in mgr._pending_cancel
        assert "ext-2" in mgr._pending_cancel

    @pytest.mark.asyncio
    async def test_stream_confirmation_clears_pending_cancel(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")

        await mgr.cancel_all_orders()
        assert "ext-1" in mgr._pending_cancel

        # Simulate stream delivering CANCELLED
        mgr.handle_order_update(
            SimpleNamespace(
                external_id="ext-1",
                status=OrderStatus.CANCELLED,
                status_reason=None,
            )
        )

        assert "ext-1" not in mgr._active_orders
        assert "ext-1" not in mgr._pending_cancel

    def test_sweep_force_removes_timed_out_orders(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-old")

        # Simulate a cancel issued 20 seconds ago
        mgr._pending_cancel["ext-old"] = time.monotonic() - 20

        removed = mgr.sweep_pending_cancels(timeout_s=10.0)

        assert removed == 1
        assert "ext-old" not in mgr._active_orders
        assert "ext-old" not in mgr._pending_cancel
        # Should be moved to recent
        assert "ext-old" in mgr._recent_orders_by_external_id

    def test_sweep_does_not_remove_within_timeout(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-fresh")
        mgr._pending_cancel["ext-fresh"] = time.monotonic()

        removed = mgr.sweep_pending_cancels(timeout_s=10.0)

        assert removed == 0
        assert "ext-fresh" in mgr._active_orders

    def test_is_pending_cancel(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")
        assert mgr.is_pending_cancel("ext-1") is False

        mgr._pending_cancel["ext-1"] = time.monotonic()
        assert mgr.is_pending_cancel("ext-1") is True

    def test_reserved_exposure_excludes_pending_cancel(self):
        mgr = _make_manager()
        _add_order(mgr, "buy-1", side=OrderSide.BUY, price=Decimal("10"), size=Decimal("5"))
        _add_order(mgr, "buy-2", side=OrderSide.BUY, price=Decimal("10"), size=Decimal("3"))

        # Mark buy-1 as pending cancel
        mgr._pending_cancel["buy-1"] = time.monotonic()

        qty, notional = mgr.reserved_exposure(side=OrderSide.BUY)
        # Only buy-2 should be counted
        assert qty == Decimal("3")
        assert notional == Decimal("30")

    @pytest.mark.asyncio
    async def test_cancel_all_fallback_on_mass_cancel_failure(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")
        mgr._client.orders.mass_cancel = AsyncMock(side_effect=Exception("network error"))

        await mgr.cancel_all_orders()

        # On failure, pending_cancel should be cleared and individual
        # cancel_order calls should have been made
        assert len(mgr._pending_cancel) == 0
        mgr._client.orders.cancel_order_by_external_id.assert_called_once()


# ===================================================================
# 2. Fill deduplication via _seen_trade_ids
# ===================================================================

class TestFillDeduplication:
    """on_fill should skip duplicate trade_ids."""

    def test_duplicate_fill_is_skipped(self):
        strategy = _make_strategy_stub()

        fill = FillEvent(
            trade_id=12345,
            order_id=100,
            market="TEST-USD",
            side=OrderSide.BUY,
            price=Decimal("100"),
            qty=Decimal("1"),
            fee=Decimal("0.05"),
            is_taker=False,
            timestamp=1000000,
        )

        # First call should record
        on_fill(strategy, fill)
        assert strategy._journal.record_fill.call_count == 1

        # Second call with same trade_id should be skipped
        on_fill(strategy, fill)
        assert strategy._journal.record_fill.call_count == 1

    def test_different_trade_ids_are_both_recorded(self):
        strategy = _make_strategy_stub()

        fill1 = FillEvent(
            trade_id=1, order_id=100, market="TEST-USD",
            side=OrderSide.BUY, price=Decimal("100"), qty=Decimal("1"),
            fee=Decimal("0.05"), is_taker=False, timestamp=1000000,
        )
        fill2 = FillEvent(
            trade_id=2, order_id=101, market="TEST-USD",
            side=OrderSide.SELL, price=Decimal("101"), qty=Decimal("2"),
            fee=Decimal("0.05"), is_taker=False, timestamp=1000001,
        )

        on_fill(strategy, fill1)
        on_fill(strategy, fill2)
        assert strategy._journal.record_fill.call_count == 2

    def test_seen_trade_ids_capped_at_max_size(self):
        strategy = _make_strategy_stub()

        # Insert more than the cap
        for i in range(10_500):
            fill = FillEvent(
                trade_id=i, order_id=i, market="TEST-USD",
                side=OrderSide.BUY, price=Decimal("100"), qty=Decimal("1"),
                fee=Decimal("0"), is_taker=False, timestamp=i,
            )
            on_fill(strategy, fill)

        # Set should be capped to 10k
        assert len(strategy._seen_trade_ids_set) <= 10_000
        assert len(strategy._seen_trade_ids) <= 10_000

        # Old IDs should have been evicted — ID 0 should no longer be present
        assert 0 not in strategy._seen_trade_ids_set
        # Recent ID should still be present
        assert 10_499 in strategy._seen_trade_ids_set


# ===================================================================
# 3. Pending-placement buffer
# ===================================================================

class TestPendingPlacementBuffer:
    """When stream delivers confirmation before place_order returns."""

    @pytest.mark.asyncio
    async def test_stream_update_before_place_order_returns(self):
        """Simulate: place_order sends HTTP, stream delivers confirmation
        before the HTTP response arrives. handle_order_update should promote
        the order from _pending_placements to _active_orders."""
        mgr = _make_manager()
        ext_id_holder = {}

        async def _slow_place(**kwargs):
            # By the time this runs, place_order has already registered
            # the external_id in _pending_placements.  Simulate the stream
            # delivering the open confirmation before we return.
            ext_id = kwargs["external_id"]
            ext_id_holder["id"] = ext_id

            # Simulate stream update arriving
            mgr.handle_order_update(
                SimpleNamespace(
                    external_id=ext_id,
                    status="OPEN",
                    id=42,
                )
            )

            return SimpleNamespace(
                status="OK",
                error=None,
                data=SimpleNamespace(id=42),
            )

        mgr._client.place_order = _slow_place

        result_ext_id = await mgr.place_order(
            side=OrderSide.BUY,
            price=Decimal("100"),
            size=Decimal("5"),
            level=0,
        )

        assert result_ext_id is not None
        # Should be in active orders
        assert result_ext_id in mgr._active_orders
        # Should NOT be in pending placements
        assert result_ext_id not in mgr._pending_placements
        # Exchange ID should have been set from the stream update
        info = mgr._active_orders[result_ext_id]
        assert info.exchange_order_id == "42"

    @pytest.mark.asyncio
    async def test_stream_terminal_before_place_order_returns(self):
        """Stream delivers FILLED before place_order HTTP returns.
        The order should still be properly tracked in recent."""
        mgr = _make_manager()
        callback_calls = []

        def _cb(*args, **kwargs):
            callback_calls.append((args, kwargs))

        mgr.on_level_freed(_cb)

        async def _slow_place(**kwargs):
            ext_id = kwargs["external_id"]

            # Stream delivers FILLED before we return
            mgr.handle_order_update(
                SimpleNamespace(
                    external_id=ext_id,
                    status=OrderStatus.FILLED,
                    id=99,
                    status_reason=None,
                )
            )

            return SimpleNamespace(
                status="OK",
                error=None,
                data=SimpleNamespace(id=99),
            )

        mgr._client.place_order = _slow_place

        result_ext_id = await mgr.place_order(
            side=OrderSide.SELL,
            price=Decimal("200"),
            size=Decimal("2"),
            level=1,
        )

        assert result_ext_id is not None
        # Should be in active (place_order added it back after return)
        # The key insight: the order was promoted to active and then
        # removed by handle_order_update's terminal status logic.
        # But since place_order also adds to active, it ends up there.
        # The callback should have been fired once.
        assert len(callback_calls) == 1
        # Should be findable in recent
        assert mgr.find_order_by_external_id(result_ext_id) is not None

    @pytest.mark.asyncio
    async def test_pending_placement_cleaned_on_failure(self):
        """Failed place_order should remove from _pending_placements."""
        mgr = _make_manager()
        mgr._client.place_order = AsyncMock(side_effect=Exception("network"))

        result = await mgr.place_order(
            side=OrderSide.BUY,
            price=Decimal("100"),
            size=Decimal("1"),
            level=0,
        )

        assert result is None
        assert len(mgr._pending_placements) == 0

    @pytest.mark.asyncio
    async def test_pending_placement_cleaned_on_rejection(self):
        """Rejected place_order should remove from _pending_placements."""
        mgr = _make_manager()
        mgr._client.place_order = AsyncMock(
            return_value=SimpleNamespace(status="ERR", error="rejected", data=None)
        )

        result = await mgr.place_order(
            side=OrderSide.BUY,
            price=Decimal("100"),
            size=Decimal("1"),
            level=0,
        )

        assert result is None
        assert len(mgr._pending_placements) == 0

    def test_handle_order_update_records_stream_timestamp(self):
        """Non-terminal update should set last_stream_update_at."""
        mgr = _make_manager()
        _add_order(mgr, "ext-1")

        before = time.monotonic()
        mgr.handle_order_update(
            SimpleNamespace(
                external_id="ext-1",
                status="OPEN",
                id=42,
            )
        )
        after = time.monotonic()

        info = mgr._active_orders["ext-1"]
        assert info.last_stream_update_at is not None
        assert before <= info.last_stream_update_at <= after


# ===================================================================
# 4. Zombie order detection
# ===================================================================

class TestZombieOrderDetection:
    """Orders without stream updates should be detected as zombies."""

    def test_find_zombie_orders_detects_old_unconfirmed(self):
        mgr = _make_manager()
        old_time = time.monotonic() - 120  # 2 minutes ago
        _add_order(
            mgr, "zombie-1",
            placed_at=old_time,
            last_stream_update_at=None,
        )
        # This one got a stream update — NOT a zombie
        _add_order(
            mgr, "healthy-1",
            placed_at=old_time,
            last_stream_update_at=time.monotonic() - 5,
        )
        # This one is recent — NOT a zombie
        _add_order(
            mgr, "fresh-1",
            placed_at=time.monotonic(),
            last_stream_update_at=None,
        )

        zombies = mgr.find_zombie_orders(max_age_s=60.0)

        assert len(zombies) == 1
        assert zombies[0].external_id == "zombie-1"

    def test_find_zombie_orders_excludes_pending_cancel(self):
        mgr = _make_manager()
        old_time = time.monotonic() - 120
        _add_order(
            mgr, "cancelling-1",
            placed_at=old_time,
            last_stream_update_at=None,
        )
        mgr._pending_cancel["cancelling-1"] = time.monotonic()

        zombies = mgr.find_zombie_orders(max_age_s=60.0)
        assert len(zombies) == 0

    def test_find_zombie_orders_empty_when_no_zombies(self):
        mgr = _make_manager()
        _add_order(
            mgr, "normal-1",
            placed_at=time.monotonic(),
            last_stream_update_at=None,
        )
        zombies = mgr.find_zombie_orders(max_age_s=60.0)
        assert len(zombies) == 0


# ===================================================================
# 5. flatten_position wait_for_fill mode
# ===================================================================

class TestFlattenPositionWaitForFill:
    """flatten_position should optionally poll risk_mgr for fill confirmation."""

    @pytest.mark.asyncio
    async def test_wait_for_fill_detects_position_change(self):
        mgr = _make_manager()
        mgr._client.place_order = AsyncMock(
            return_value=SimpleNamespace(
                status="OK", error=None, data=SimpleNamespace(id=42)
            )
        )

        call_count = 0
        def _get_position():
            nonlocal call_count
            call_count += 1
            # First call returns original position, then returns 0
            if call_count <= 1:
                return Decimal("10")
            return Decimal("0")

        risk_mgr = MagicMock()
        risk_mgr.get_current_position = _get_position

        result = await mgr.flatten_position(
            signed_position=Decimal("10"),
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            tick_size=Decimal("0.1"),
            min_order_size=Decimal("1"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
            risk_mgr=risk_mgr,
            wait_for_fill_s=2.0,
        )

        assert result.success is True
        assert result.remaining_position == Decimal("0")

    @pytest.mark.asyncio
    async def test_wait_for_fill_timeout_returns_current_position(self):
        mgr = _make_manager()
        mgr._client.place_order = AsyncMock(
            return_value=SimpleNamespace(
                status="OK", error=None, data=SimpleNamespace(id=42)
            )
        )

        # Position never changes
        risk_mgr = MagicMock()
        risk_mgr.get_current_position = MagicMock(return_value=Decimal("10"))

        result = await mgr.flatten_position(
            signed_position=Decimal("10"),
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            tick_size=Decimal("0.1"),
            min_order_size=Decimal("1"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
            risk_mgr=risk_mgr,
            wait_for_fill_s=0.3,
        )

        assert result.success is True
        assert result.remaining_position == Decimal("10")

    @pytest.mark.asyncio
    async def test_flatten_without_wait_for_fill_returns_none_remaining(self):
        """When wait_for_fill_s is 0, remaining_position should be None."""
        mgr = _make_manager()
        mgr._client.place_order = AsyncMock(
            return_value=SimpleNamespace(
                status="OK", error=None, data=SimpleNamespace(id=42)
            )
        )

        result = await mgr.flatten_position(
            signed_position=Decimal("5"),
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            tick_size=Decimal("0.1"),
            min_order_size=Decimal("1"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
        )

        assert result.success is True
        assert result.remaining_position is None

    @pytest.mark.asyncio
    async def test_flatten_zero_position_returns_zero_remaining(self):
        mgr = _make_manager()

        result = await mgr.flatten_position(
            signed_position=Decimal("0"),
            best_bid=Decimal("100"),
            best_ask=Decimal("101"),
            tick_size=Decimal("0.1"),
            min_order_size=Decimal("1"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
        )

        assert result.success is True
        assert result.remaining_position == Decimal("0")

    @pytest.mark.asyncio
    async def test_flatten_result_has_remaining_position_field(self):
        """FlattenResult dataclass should include remaining_position."""
        from market_maker.order_manager import FlattenResult

        result = FlattenResult(
            attempted=True,
            success=True,
            reason="submitted",
            remaining_position=Decimal("3"),
        )
        assert result.remaining_position == Decimal("3")


# ===================================================================
# Backward compatibility: existing tests should still pass
# ===================================================================

class TestBackwardCompatibility:

    @pytest.mark.asyncio
    async def test_place_order_success_still_tracks_and_resets(self):
        mgr = _make_manager()
        mgr.consecutive_failures = 3

        ext_id = await mgr.place_order(
            side=OrderSide.SELL,
            price=Decimal("11"),
            size=Decimal("2"),
            level=1,
        )

        assert ext_id is not None
        assert mgr.consecutive_failures == 0
        assert ext_id in mgr._active_orders

    def test_handle_order_update_terminal_still_fires_callbacks(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")

        callback_args = []
        mgr.on_level_freed(lambda *a, **kw: callback_args.append((a, kw)))

        mgr.handle_order_update(
            SimpleNamespace(
                external_id="ext-1",
                status=OrderStatus.FILLED,
                status_reason=None,
                id=42,
            )
        )

        assert len(callback_args) == 1
        assert "ext-1" not in mgr._active_orders

    def test_reserved_exposure_excludes_target_order(self):
        mgr = _make_manager()
        _add_order(mgr, "buy-1", side=OrderSide.BUY, price=Decimal("10"), size=Decimal("2"))
        _add_order(mgr, "buy-2", side=OrderSide.BUY, price=Decimal("11"), size=Decimal("3"))

        qty, notional = mgr.reserved_exposure(
            side=OrderSide.BUY,
            exclude_external_id="buy-1",
        )
        assert qty == Decimal("3")
        assert notional == Decimal("33")

    @pytest.mark.asyncio
    async def test_cancel_order_does_not_pop_from_active(self):
        mgr = _make_manager()
        _add_order(mgr, "ext-1")

        result = await mgr.cancel_order("ext-1")
        assert result is True
        # Should still be in active — stream will clean up
        assert "ext-1" in mgr._active_orders
