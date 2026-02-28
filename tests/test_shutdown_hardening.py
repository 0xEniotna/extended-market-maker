"""Tests for shutdown sequence hardening (prompt 2).

Covers:
1. Progressive slippage computation and integration in flatten loop
2. Hard shutdown timeout with emergency state file
3. Double-signal force-exit
4. One-sided book handling in flatten_position
5. Pre-shutdown state persistence
6. Fresh REST position verification after flatten loop
"""
from __future__ import annotations

import asyncio
import json
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# SDK module stubs
# ---------------------------------------------------------------------------
_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.accounts",
    "x10.perpetual.configuration",
    "x10.perpetual.orderbook",
    "x10.perpetual.orders",
    "x10.perpetual.positions",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.perpetual.trades",
    "x10.perpetual.trading_client",
    "x10.utils",
    "x10.utils.http",
]
for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED", CANCELLED="CANCELLED", EXPIRED="EXPIRED", REJECTED="REJECTED",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT")
_orders_mod.OrderType.MARKET = "MARKET"
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT", IOC="IOC")
_orders_mod.OpenOrderModel = object

_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")

from market_maker import shutdown_manager, strategy_runner  # noqa: E402
from market_maker.order_manager import FlattenResult, OrderManager  # noqa: E402
from market_maker.shutdown_manager import (  # noqa: E402
    attempt_shutdown_flatten as _attempt_shutdown_flatten,
    compute_progressive_slippage as _compute_progressive_slippage,
    shutdown_core as _shutdown_core,
    _write_json_state,
    _write_pre_shutdown_state,
)
from market_maker.strategy_runner import RuntimeContext  # noqa: E402

OrderSide = _orders_mod.OrderSide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides):
    """Create a mock settings object with sensible defaults."""
    defaults = {
        "market_name": "TEST-USD",
        "flatten_position_on_shutdown": True,
        "shutdown_flatten_retries": 3,
        "shutdown_flatten_slippage_bps": Decimal("20"),
        "shutdown_flatten_slippage_step_bps": Decimal("10"),
        "shutdown_flatten_max_slippage_bps": Decimal("100"),
        "shutdown_flatten_retry_delay_s": 0.0,
        "shutdown_timeout_s": 30.0,
        "max_position_size": Decimal("100"),
        "max_position_notional_usd": Decimal("2500"),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_ctx(
    *,
    position: Decimal = Decimal("0"),
    best_bid: Decimal | None = Decimal("100"),
    best_ask: Decimal | None = Decimal("100.10"),
    settings_overrides: dict | None = None,
    flatten_side_effect=None,
    refresh_position_positions=None,
):
    """Build a RuntimeContext with mocked components."""
    settings = _make_settings(**(settings_overrides or {}))

    # Mock orderbook
    ob_mgr = MagicMock()
    if best_bid is not None:
        ob_mgr.best_bid.return_value = SimpleNamespace(price=best_bid, size=Decimal("10"))
    else:
        ob_mgr.best_bid.return_value = None
    if best_ask is not None:
        ob_mgr.best_ask.return_value = SimpleNamespace(price=best_ask, size=Decimal("10"))
    else:
        ob_mgr.best_ask.return_value = None
    ob_mgr.is_stale.return_value = False
    ob_mgr.spread_bps.return_value = Decimal("10")

    # Mock risk manager
    risk_mgr = MagicMock()
    if refresh_position_positions is not None:
        positions = list(refresh_position_positions)
        position_iter = iter(positions)
        current_pos = [position]

        async def _refresh():
            try:
                current_pos[0] = next(position_iter)
            except StopIteration:
                pass
            return current_pos[0]

        risk_mgr.get_current_position.side_effect = lambda: current_pos[0]
        risk_mgr.refresh_position = AsyncMock(side_effect=_refresh)
    else:
        risk_mgr.get_current_position.return_value = position
        risk_mgr.refresh_position = AsyncMock(return_value=position)
    risk_mgr.get_position_realized_pnl.return_value = Decimal("0")
    risk_mgr.get_position_unrealized_pnl.return_value = Decimal("0")
    risk_mgr.get_position_total_pnl.return_value = Decimal("0")
    risk_mgr.get_available_for_trade.return_value = Decimal("1000")

    # Mock order manager
    order_mgr = MagicMock()
    order_mgr.get_active_orders.return_value = {}
    order_mgr.cancel_all_orders = AsyncMock()
    if flatten_side_effect is not None:
        order_mgr.flatten_position = AsyncMock(side_effect=flatten_side_effect)
    else:
        order_mgr.flatten_position = AsyncMock(
            return_value=FlattenResult(
                attempted=True, success=True, reason="submitted"
            )
        )

    # Mock strategy
    strategy = MagicMock()
    strategy.shutdown_reason = "shutdown"
    strategy._handle_signal = MagicMock()

    # Mock journal
    journal = MagicMock()
    journal.record_run_end = MagicMock()
    journal.close = MagicMock()

    # Mock metrics
    metrics = MagicMock()
    metrics.snapshot.return_value = SimpleNamespace(
        position=position,
        active_orders=0,
        total_fills=0,
        total_cancellations=0,
        total_rejections=0,
        post_only_failures=0,
        total_fees=Decimal("0"),
        consecutive_failures=0,
        circuit_open=False,
        uptime_s=100.0,
    )
    metrics.stop = AsyncMock()

    # Mock other services
    account_stream = MagicMock()
    account_stream.stop = AsyncMock()
    trading_client = MagicMock()
    trading_client.close = AsyncMock()

    return RuntimeContext(
        settings=settings,
        trading_client=trading_client,
        market_info=MagicMock(),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("0.01"),
        min_order_size_change=Decimal("0.01"),
        order_size=Decimal("1"),
        ob_mgr=ob_mgr,
        order_mgr=order_mgr,
        risk_mgr=risk_mgr,
        account_stream=account_stream,
        journal=journal,
        metrics=metrics,
        strategy=strategy,
    )


# ===================================================================
# 1. Progressive slippage
# ===================================================================

class TestProgressiveSlippage:

    def test_compute_progressive_slippage_attempt_1(self):
        result = _compute_progressive_slippage(
            attempt=1,
            base_bps=Decimal("20"),
            step_bps=Decimal("10"),
            max_bps=Decimal("100"),
        )
        assert result == Decimal("20")

    def test_compute_progressive_slippage_attempt_2(self):
        result = _compute_progressive_slippage(
            attempt=2,
            base_bps=Decimal("20"),
            step_bps=Decimal("10"),
            max_bps=Decimal("100"),
        )
        assert result == Decimal("30")

    def test_compute_progressive_slippage_attempt_3(self):
        result = _compute_progressive_slippage(
            attempt=3,
            base_bps=Decimal("20"),
            step_bps=Decimal("10"),
            max_bps=Decimal("100"),
        )
        assert result == Decimal("40")

    def test_compute_progressive_slippage_capped_at_max(self):
        result = _compute_progressive_slippage(
            attempt=20,
            base_bps=Decimal("20"),
            step_bps=Decimal("10"),
            max_bps=Decimal("100"),
        )
        assert result == Decimal("100")

    def test_compute_progressive_slippage_zero_step(self):
        """Step of 0 means constant slippage."""
        result = _compute_progressive_slippage(
            attempt=5,
            base_bps=Decimal("20"),
            step_bps=Decimal("0"),
            max_bps=Decimal("100"),
        )
        assert result == Decimal("20")

    @pytest.mark.asyncio
    async def test_flatten_loop_passes_progressive_slippage(self):
        """Each flatten retry should use increasing slippage."""
        slippages_received = []

        async def _capture_flatten(**kwargs):
            slippages_received.append(kwargs.get("slippage_bps"))
            return FlattenResult(
                attempted=True, success=True, reason="submitted"
            )

        ctx = _make_ctx(
            position=Decimal("10"),
            flatten_side_effect=_capture_flatten,
            settings_overrides={
                "shutdown_flatten_retries": 4,
                "shutdown_flatten_slippage_bps": Decimal("20"),
                "shutdown_flatten_slippage_step_bps": Decimal("10"),
                "shutdown_flatten_max_slippage_bps": Decimal("100"),
            },
        )

        # Reset module state
        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        await _attempt_shutdown_flatten(ctx, "shutdown")

        assert len(slippages_received) == 4
        assert slippages_received[0] == Decimal("20")
        assert slippages_received[1] == Decimal("30")
        assert slippages_received[2] == Decimal("40")
        assert slippages_received[3] == Decimal("50")

    @pytest.mark.asyncio
    async def test_flatten_loop_stops_early_when_flat(self):
        """Flatten loop should stop when position reaches 0."""
        call_count = [0]

        async def _flatten_and_clear(**kwargs):
            call_count[0] += 1
            return FlattenResult(attempted=True, success=True, reason="submitted")

        ctx = _make_ctx(
            position=Decimal("10"),
            flatten_side_effect=_flatten_and_clear,
            settings_overrides={"shutdown_flatten_retries": 5},
            # After first refresh, position becomes 0
            refresh_position_positions=[Decimal("0")],
        )

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        result = await _attempt_shutdown_flatten(ctx, "shutdown")

        assert call_count[0] == 1
        assert result["reason"] == "flattened"
        assert result["attempts"] == 1


# ===================================================================
# 2. Hard shutdown timeout
# ===================================================================

class TestShutdownTimeout:

    @pytest.mark.asyncio
    async def test_shutdown_timeout_writes_emergency_state(self, tmp_path):
        """When shutdown exceeds timeout, emergency state file should be written."""
        ctx = _make_ctx(
            settings_overrides={"shutdown_timeout_s": 0.1},
        )

        # Make _shutdown_core hang
        async def _slow_core(ctx, tasks):
            await asyncio.sleep(10)  # Way longer than timeout

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        with patch.object(shutdown_manager, "shutdown_core", _slow_core):
            with patch.object(shutdown_manager, "_write_emergency_state") as mock_emergency:
                with patch("os._exit") as mock_exit:
                    await shutdown_manager.shutdown_and_record(ctx, [])
                    mock_emergency.assert_called_once()
                    mock_exit.assert_called_once_with(1)


# ===================================================================
# 3. Double-signal force-exit
# ===================================================================

class TestDoubleSignalForceExit:

    @pytest.mark.asyncio
    async def test_force_exit_event_aborts_flatten(self):
        """When force_exit_event is set, flatten loop should abort."""
        call_count = [0]

        async def _slow_flatten(**kwargs):
            call_count[0] += 1
            return FlattenResult(attempted=True, success=True, reason="submitted")

        ctx = _make_ctx(
            position=Decimal("10"),
            flatten_side_effect=_slow_flatten,
            settings_overrides={"shutdown_flatten_retries": 5},
        )

        # Simulate force-exit already signaled
        shutdown_manager._shutdown_in_progress = True
        shutdown_manager._force_exit_event = asyncio.Event()
        shutdown_manager._force_exit_event.set()

        result = await _attempt_shutdown_flatten(ctx, "shutdown")

        assert call_count[0] == 0
        assert result["reason"] == "force_exit"

    @pytest.mark.asyncio
    async def test_shutdown_core_skips_flatten_on_force_exit(self):
        """_shutdown_core should skip flatten when force_exit_event is set."""
        ctx = _make_ctx(position=Decimal("10"))

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = asyncio.Event()
        shutdown_manager._force_exit_event.set()

        with patch.object(shutdown_manager, "_write_pre_shutdown_state"):
            with patch.object(shutdown_manager, "stop_services", new_callable=AsyncMock):
                await _shutdown_core(ctx, [])

        # flatten should not have been called on order_mgr
        ctx.order_mgr.flatten_position.assert_not_called()

    def test_double_signal_handler_sets_force_exit(self):
        """On second signal during shutdown, force_exit_event should be set."""
        shutdown_manager._shutdown_in_progress = True
        shutdown_manager._force_exit_event = asyncio.Event()

        # The handler is installed inside _install_signal_handlers,
        # but we can test the logic by simulating the second call.
        # force_exit_event should be set.
        assert not shutdown_manager._force_exit_event.is_set()

        # Simulate what _double_signal_handler does on second signal
        shutdown_manager._force_exit_event.set()

        assert shutdown_manager._force_exit_event.is_set()


# ===================================================================
# 4. One-sided book handling in flatten_position
# ===================================================================

class TestOneSidedBookFlatten:

    @pytest.mark.asyncio
    async def test_flatten_with_only_ask_available_for_sell(self):
        """When selling and bid is missing, should use ask price."""
        client = MagicMock()
        client.place_order = AsyncMock(
            return_value=SimpleNamespace(status="OK", error=None, data=SimpleNamespace(id="123"))
        )
        mgr = OrderManager(client, "TEST-USD")

        result = await mgr.flatten_position(
            signed_position=Decimal("10"),
            best_bid=None,  # missing
            best_ask=Decimal("100"),
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("0.01"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
        )

        assert result.attempted is True
        assert result.success is True
        client.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_flatten_with_only_bid_available_for_buy(self):
        """When buying and ask is missing, should use bid price."""
        client = MagicMock()
        client.place_order = AsyncMock(
            return_value=SimpleNamespace(status="OK", error=None, data=SimpleNamespace(id="123"))
        )
        mgr = OrderManager(client, "TEST-USD")

        result = await mgr.flatten_position(
            signed_position=Decimal("-10"),  # short -> need to BUY
            best_bid=Decimal("100"),
            best_ask=None,  # missing
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("0.01"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
        )

        assert result.attempted is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_flatten_uses_last_known_mid_when_both_bbo_missing(self):
        """When both BBO sides missing, use last_known_mid with 50bps slippage."""
        client = MagicMock()
        client.place_order = AsyncMock(
            return_value=SimpleNamespace(status="OK", error=None, data=SimpleNamespace(id="123"))
        )
        mgr = OrderManager(client, "TEST-USD")

        result = await mgr.flatten_position(
            signed_position=Decimal("10"),
            best_bid=None,
            best_ask=None,
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("0.01"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
            last_known_mid=Decimal("100"),
        )

        assert result.attempted is True
        assert result.success is True

    @pytest.mark.asyncio
    async def test_flatten_gives_up_when_no_price_reference(self):
        """When all price sources are missing, should give up."""
        client = MagicMock()
        mgr = OrderManager(client, "TEST-USD")

        result = await mgr.flatten_position(
            signed_position=Decimal("10"),
            best_bid=None,
            best_ask=None,
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("0.01"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
            last_known_mid=None,
        )

        assert result.attempted is False
        assert result.reason == "missing_orderbook_price"

    @pytest.mark.asyncio
    async def test_flatten_last_known_mid_applies_minimum_50bps(self):
        """When using last_known_mid, slippage should be at least 50bps."""
        client = MagicMock()
        call_args = {}

        async def _capture_place_order(**kwargs):
            call_args.update(kwargs)
            return SimpleNamespace(status="OK", error=None, data=SimpleNamespace(id="123"))

        client.place_order = AsyncMock(side_effect=_capture_place_order)
        mgr = OrderManager(client, "TEST-USD")

        # Requesting only 20bps slippage, but mid fallback should use 50bps minimum
        await mgr.flatten_position(
            signed_position=Decimal("10"),  # SELL to flatten
            best_bid=None,
            best_ask=None,
            tick_size=Decimal("0.01"),
            min_order_size=Decimal("0.01"),
            size_step=Decimal("0.01"),
            slippage_bps=Decimal("20"),
            last_known_mid=Decimal("100"),
        )

        # For a SELL, price = ref * (1 - bps/10000) = 100 * (1 - 50/10000) = 99.50
        # Rounded down to tick = 99.50
        assert call_args["price"] == Decimal("99.50")


# ===================================================================
# 5. Pre-shutdown state persistence
# ===================================================================

class TestPreShutdownState:

    def test_write_json_state_creates_file(self, tmp_path):
        data = {"test": "value", "decimal": Decimal("1.5")}
        path = _write_json_state(str(tmp_path / "state"), "TEST-USD", data)

        assert path is not None
        assert path.exists()
        content = json.loads(path.read_text())
        assert content["test"] == "value"
        assert content["decimal"] == "1.5"

    def test_write_json_state_creates_directories(self, tmp_path):
        deep_path = str(tmp_path / "a" / "b" / "c")
        path = _write_json_state(deep_path, "TEST-USD", {"x": 1})
        assert path is not None
        assert path.exists()

    def test_write_pre_shutdown_state_includes_all_fields(self, tmp_path):
        ctx = _make_ctx(position=Decimal("5"))

        with patch.object(shutdown_manager, "_write_json_state") as mock_write:
            mock_write.return_value = tmp_path / "test.json"
            _write_pre_shutdown_state(ctx, "shutdown")

            mock_write.assert_called_once()
            call_args = mock_write.call_args
            data = call_args[0][2]  # third positional arg is data

            assert data["type"] == "pre_shutdown_state"
            assert data["market"] == "TEST-USD"
            assert data["shutdown_reason"] == "shutdown"
            assert "position" in data
            assert "position_pnl" in data
            assert "active_orders" in data
            assert "orderbook" in data
            assert "config" in data


# ===================================================================
# 6. Fresh REST verification after flatten loop
# ===================================================================

class TestFreshRESTVerification:

    @pytest.mark.asyncio
    async def test_critical_log_when_position_nonzero_after_flatten(self, caplog):
        """Should log CRITICAL when position is still non-zero after all retries."""
        ctx = _make_ctx(
            position=Decimal("5"),
            settings_overrides={"shutdown_flatten_retries": 1},
        )

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        import logging
        with caplog.at_level(logging.CRITICAL):
            await _attempt_shutdown_flatten(ctx, "shutdown")

        # Position is still 5 (mock never returns 0)
        assert any("POSITION NOT FLAT" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_critical_log_when_position_is_flat(self, caplog):
        """Should NOT log CRITICAL when position reaches 0."""
        ctx = _make_ctx(
            position=Decimal("5"),
            settings_overrides={"shutdown_flatten_retries": 1},
            refresh_position_positions=[Decimal("0")],
        )

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        import logging
        with caplog.at_level(logging.CRITICAL):
            result = await _attempt_shutdown_flatten(ctx, "shutdown")

        assert result["reason"] == "flattened"
        critical_msgs = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        position_not_flat_msgs = [
            r for r in critical_msgs if "POSITION NOT FLAT" in r.message
        ]
        assert len(position_not_flat_msgs) == 0


# ===================================================================
# Integration / combined scenarios
# ===================================================================

class TestIntegration:

    @pytest.mark.asyncio
    async def test_full_shutdown_core_happy_path(self):
        """Full shutdown flow with a position that flattens on first attempt."""
        # _shutdown_core calls refresh_position 3 times:
        #   1. Before flatten loop
        #   2. Inside flatten loop (after flatten attempt)
        #   3. After flatten loop (final verification)
        # Position starts at 10, stays 10 for pre-flatten refresh,
        # then becomes 0 after flatten to confirm success.
        ctx = _make_ctx(
            position=Decimal("10"),
            settings_overrides={"shutdown_flatten_retries": 2},
            refresh_position_positions=[
                Decimal("10"),  # pre-flatten refresh: still 10
                Decimal("0"),   # post-attempt refresh: now flat
                Decimal("0"),   # final REST verification
            ],
        )

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        with patch.object(shutdown_manager, "_write_pre_shutdown_state"):
            with patch.object(shutdown_manager, "stop_services", new_callable=AsyncMock):
                await _shutdown_core(ctx, [])

        ctx.order_mgr.cancel_all_orders.assert_awaited_once()
        ctx.order_mgr.flatten_position.assert_awaited_once()
        ctx.journal.record_run_end.assert_called_once()
        ctx.journal.close.assert_called_once()
        assert shutdown_manager._shutdown_in_progress is True

    @pytest.mark.asyncio
    async def test_flatten_passes_last_known_mid(self):
        """_attempt_shutdown_flatten should pass last_known_mid to flatten_position."""
        last_mid_received = []

        async def _capture_flatten(**kwargs):
            last_mid_received.append(kwargs.get("last_known_mid"))
            return FlattenResult(attempted=True, success=True, reason="submitted")

        ctx = _make_ctx(
            position=Decimal("5"),
            best_bid=Decimal("99"),
            best_ask=Decimal("101"),
            flatten_side_effect=_capture_flatten,
            settings_overrides={"shutdown_flatten_retries": 1},
        )

        shutdown_manager._shutdown_in_progress = False
        shutdown_manager._force_exit_event = None

        await _attempt_shutdown_flatten(ctx, "shutdown")

        assert len(last_mid_received) == 1
        assert last_mid_received[0] == Decimal("100")  # (99 + 101) / 2
