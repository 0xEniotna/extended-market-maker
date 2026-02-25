"""
Account Stream Manager

Subscribes to the exchange's real-time account-update WebSocket and
propagates order fills, cancellations, rejections, position updates, and
trade events to the rest of the market-maker.

This is the **single source of truth** for order lifecycle events.  The
``OrderManager`` and ``RiskManager`` register callbacks so their internal
state stays consistent without periodic polling.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Awaitable, Callable, Deque, List, Optional

from x10.perpetual.accounts import AccountStreamDataModel
from x10.perpetual.configuration import EndpointConfig
from x10.perpetual.orders import OpenOrderModel, OrderSide, OrderStatus, OrderStatusReason
from x10.perpetual.positions import PositionModel
from x10.perpetual.stream_client.stream_client import PerpetualStreamClient
from x10.perpetual.trades import AccountTradeModel
from x10.utils.http import WrappedStreamResponse

logger = logging.getLogger(__name__)

# Backoff constants for account stream reconnection.
_STREAM_BACKOFF_BASE_S = 2.0
_STREAM_BACKOFF_MAX_S = 120.0
_STREAM_JITTER_MAX_S = 1.0


# ---------------------------------------------------------------------------
# Public data-classes exposed to consumers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FillEvent:
    """Represents a single fill (trade) on one of our orders."""

    trade_id: int
    order_id: int
    market: str
    side: OrderSide
    price: Decimal
    qty: Decimal
    fee: Decimal
    is_taker: bool
    timestamp: int


@dataclass
class AccountStreamMetrics:
    """Running counters updated by the stream manager."""

    fills: int = 0
    cancellations: int = 0
    rejections: int = 0
    post_only_failures: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    last_event_ts: float = 0.0


# Callback signatures
OrderUpdateCallback = Callable[[OpenOrderModel], None]
PositionUpdateCallback = Callable[[PositionModel], None]
FillCallback = Callable[[FillEvent], None]
BalanceUpdateCallback = Callable[[object], None]
FailSafeCallback = Callable[[str], Awaitable[None] | None]


class AccountStreamManager:
    """
    Subscribes to the account WebSocket and dispatches events:

    * **Order updates** → ``OrderManager`` (remove filled/cancelled orders,
      detect POST_ONLY_FAILED rejections).
    * **Position updates** → ``RiskManager`` (instant position refresh).
    * **Trade updates** → P&L / metrics tracking.
    """

    def __init__(
        self,
        endpoint_config: EndpointConfig,
        api_key: str,
        market_name: str,
    ) -> None:
        self._stream_client = PerpetualStreamClient(api_url=endpoint_config.stream_url)
        self._api_key = api_key
        self._market_name = market_name

        self._task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

        # Callbacks registered by other components
        self._order_callbacks: List[OrderUpdateCallback] = []
        self._position_callbacks: List[PositionUpdateCallback] = []
        self._fill_callbacks: List[FillCallback] = []
        self._balance_callbacks: List[BalanceUpdateCallback] = []

        # Optional references set by the runner for health monitoring.
        self._order_mgr: Optional[object] = None
        self._journal: Optional[object] = None
        self._fail_safe_handler: Optional[FailSafeCallback] = None

        self._connected: bool = False
        self._sequence_healthy: bool = True
        self._last_seq: Optional[int] = None
        self._rolling_fill_window_1m: Deque[tuple[float, Decimal, bool]] = deque()

        # Public metrics
        self.metrics = AccountStreamMetrics()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_order_update(self, cb: OrderUpdateCallback) -> None:
        """Register a callback invoked on every order status change."""
        self._order_callbacks.append(cb)

    def on_position_update(self, cb: PositionUpdateCallback) -> None:
        """Register a callback invoked when a position update arrives."""
        self._position_callbacks.append(cb)

    def on_fill(self, cb: FillCallback) -> None:
        """Register a callback invoked on every trade/fill."""
        self._fill_callbacks.append(cb)

    def on_balance_update(self, cb: BalanceUpdateCallback) -> None:
        """Register a callback invoked on every balance update."""
        self._balance_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_order_manager(self, order_mgr: object) -> None:
        """Set the order manager reference for connection health monitoring."""
        self._order_mgr = order_mgr

    def set_journal(self, journal: object) -> None:
        """Set the trade journal reference for event recording."""
        self._journal = journal

    def set_fail_safe_handler(self, cb: FailSafeCallback) -> None:
        """Register async callback invoked on stream sequence/disconnect faults."""
        self._fail_safe_handler = cb

    def is_connected(self) -> bool:
        return self._connected

    def is_sequence_healthy(self) -> bool:
        return self._sequence_healthy and self._connected

    def taker_fill_count_1m(self) -> int:
        self._prune_fill_window()
        return sum(1 for _, _, is_taker in self._rolling_fill_window_1m if is_taker)

    def taker_fill_notional_ratio_1m(self) -> Decimal:
        self._prune_fill_window()
        if not self._rolling_fill_window_1m:
            return Decimal("0")
        total = sum(notional for _, notional, _ in self._rolling_fill_window_1m)
        if total <= 0:
            return Decimal("0")
        taker = sum(
            notional for _, notional, is_taker in self._rolling_fill_window_1m if is_taker
        )
        return taker / total

    def _prune_fill_window(self) -> None:
        cutoff = time.monotonic() - 60.0
        while self._rolling_fill_window_1m and self._rolling_fill_window_1m[0][0] < cutoff:
            self._rolling_fill_window_1m.popleft()

    async def _trigger_fail_safe(self, reason: str) -> None:
        """Cancel orders and notify strategy when stream health is compromised."""
        if self._order_mgr is not None:
            count_fn = getattr(self._order_mgr, "active_order_count", None)
            has_active = bool(count_fn() > 0) if count_fn is not None else False
            if has_active:
                cancel_fn = getattr(self._order_mgr, "cancel_all_orders", None)
                if cancel_fn is not None:
                    try:
                        await cancel_fn()
                    except Exception as exc:
                        logger.error("Fail-safe cancel_all_orders failed: %s", exc)
        if self._fail_safe_handler is not None:
            try:
                maybe = self._fail_safe_handler(reason)
                if asyncio.iscoroutine(maybe):
                    await maybe
            except Exception as exc:
                logger.error("Fail-safe callback error: %s", exc)

    async def start(self) -> None:
        """Start the background stream task and connection health watchdog."""
        self._task = asyncio.create_task(self._stream_loop(), name="mm-account-stream")
        self._health_task = asyncio.create_task(
            self._connection_health_watchdog(), name="mm-stream-health"
        )
        logger.info("Account stream manager started for %s", self._market_name)

    async def stop(self) -> None:
        """Cancel the background stream and health tasks."""
        for task in (self._task, self._health_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._task = None
        self._health_task = None
        logger.info("Account stream manager stopped")

    # ------------------------------------------------------------------
    # Stream loop (reconnects automatically)
    # ------------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Connect to the account stream and process events.

        Reconnects on drop with exponential backoff and jitter, mirroring
        the pattern in ``OrderbookManager._reconnect_watchdog``.  Backoff
        resets after a successful connection delivers at least one event.
        """
        consecutive_failures = 0
        while True:
            got_event = False
            try:
                async with self._stream_client.subscribe_to_account_updates(
                    self._api_key
                ) as stream:
                    logger.info("Account stream connected")
                    self._connected = True
                    self._last_seq = None
                    consecutive_failures = 0  # connected successfully
                    async for event in stream:
                        got_event = True
                        await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._connected = False
                self._sequence_healthy = False
                await self._trigger_fail_safe("account_stream_error")
                consecutive_failures += 1
                backoff = min(
                    _STREAM_BACKOFF_BASE_S * (2 ** (consecutive_failures - 1)),
                    _STREAM_BACKOFF_MAX_S,
                )
                jitter = random.uniform(0.0, _STREAM_JITTER_MAX_S)
                delay = backoff + jitter
                logger.error(
                    "Account stream error: %s — reconnecting in %.1fs "
                    "(attempt %d, backoff=%.0fs jitter=%.1fs)",
                    exc, delay, consecutive_failures, backoff, jitter,
                )
                await asyncio.sleep(delay)
                continue

            # Stream ended cleanly (no exception) — reconnect.
            if got_event:
                # Received at least one event: reset backoff.
                consecutive_failures = 0
                delay = _STREAM_BACKOFF_BASE_S + random.uniform(0.0, _STREAM_JITTER_MAX_S)
            else:
                consecutive_failures += 1
                backoff = min(
                    _STREAM_BACKOFF_BASE_S * (2 ** (consecutive_failures - 1)),
                    _STREAM_BACKOFF_MAX_S,
                )
                delay = backoff + random.uniform(0.0, _STREAM_JITTER_MAX_S)
            self._connected = False
            self._sequence_healthy = False
            await self._trigger_fail_safe("account_stream_disconnect")
            logger.warning(
                "Account stream ended — reconnecting in %.1fs", delay,
            )
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Connection health watchdog
    # ------------------------------------------------------------------

    async def _connection_health_watchdog(self) -> None:
        """Monitor account stream liveness and take protective action.

        - 30s with no event while orders are active → warning log.
        - 60s with no event while orders are active → cancel_all_orders
          as a safety measure (we cannot trust order state without stream
          confirmation).
        """
        _WARNING_THRESHOLD_S = 30.0
        _CANCEL_THRESHOLD_S = 60.0
        warned = False
        cancelled = False
        while True:
            try:
                await asyncio.sleep(5.0)
                last_event_ts = self.metrics.last_event_ts
                if last_event_ts == 0.0:
                    # No event received yet (initial startup).
                    continue

                elapsed = time.monotonic() - last_event_ts
                has_active_orders = False
                if self._order_mgr is not None:
                    count_fn = getattr(self._order_mgr, "active_order_count", None)
                    if count_fn is not None:
                        has_active_orders = count_fn() > 0

                if not has_active_orders:
                    warned = False
                    cancelled = False
                    continue

                if elapsed >= _CANCEL_THRESHOLD_S and not cancelled:
                    logger.critical(
                        "Account stream silent for %.0fs with %s active orders — "
                        "cancelling all orders as safety measure",
                        elapsed,
                        "unknown" if not has_active_orders else "active",
                    )
                    cancel_fn = getattr(self._order_mgr, "cancel_all_orders", None)
                    if cancel_fn is not None:
                        await cancel_fn()
                    if self._journal is not None:
                        record_fn = getattr(self._journal, "record_exchange_event", None)
                        if record_fn is not None:
                            record_fn(
                                event_type="stream_health_cancel",
                                details={
                                    "elapsed_s": elapsed,
                                    "reason": "account_stream_silent",
                                },
                            )
                    cancelled = True
                    warned = True
                elif elapsed >= _WARNING_THRESHOLD_S and not warned:
                    logger.warning(
                        "Account stream silent for %.0fs with active orders — "
                        "order state may be stale",
                        elapsed,
                    )
                    warned = True

                if elapsed < _WARNING_THRESHOLD_S:
                    warned = False
                    cancelled = False

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Connection health watchdog error: %s", exc)
                await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _handle_event(
        self, event: WrappedStreamResponse[AccountStreamDataModel]
    ) -> None:
        current_seq = int(getattr(event, "seq", 0))
        if self._last_seq is not None and current_seq != (self._last_seq + 1):
            self._sequence_healthy = False
            logger.critical(
                "Account stream sequence gap detected for %s: prev=%s current=%s",
                self._market_name,
                self._last_seq,
                current_seq,
            )
            await self._trigger_fail_safe("account_stream_seq_gap")
            raise RuntimeError(
                f"account_stream_seq_gap prev={self._last_seq} current={current_seq}"
            )
        self._last_seq = current_seq
        self._sequence_healthy = True

        data: Optional[AccountStreamDataModel] = event.data
        if data is None:
            return

        self.metrics.last_event_ts = time.monotonic()

        # --- Order updates ---
        if data.orders:
            for order in data.orders:
                self._dispatch_order_update(order)

        # --- Position updates ---
        if data.positions:
            for pos in data.positions:
                if pos.market == self._market_name:
                    self._dispatch_position_update(pos)

        # --- Trade (fill) updates ---
        if data.trades:
            for trade in data.trades:
                if trade.market == self._market_name:
                    self._dispatch_trade(trade)

        # --- Balance updates ---
        if data.balance is not None:
            self._dispatch_balance_update(data.balance)

    def _dispatch_order_update(self, order: OpenOrderModel) -> None:
        """Process a single order update from the stream."""
        if order.market != self._market_name:
            return

        # Detect terminal states
        if order.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
            OrderStatus.REJECTED,
        ):
            if order.status == OrderStatus.CANCELLED:
                self.metrics.cancellations += 1

            if order.status == OrderStatus.REJECTED:
                self.metrics.rejections += 1
                reason = getattr(order, "status_reason", None)
                if reason == OrderStatusReason.POST_ONLY_FAILED:
                    self.metrics.post_only_failures += 1
                    logger.warning(
                        "POST_ONLY_FAILED: order ext_id=%s side=%s price=%s — "
                        "would have crossed the spread",
                        order.external_id,
                        order.side,
                        order.price,
                    )

        for cb in self._order_callbacks:
            try:
                cb(order)
            except Exception as exc:
                logger.error("Order callback error: %s", exc)

    def _dispatch_position_update(self, pos: PositionModel) -> None:
        for cb in self._position_callbacks:
            try:
                cb(pos)
            except Exception as exc:
                logger.error("Position callback error: %s", exc)

    def _dispatch_trade(self, trade: AccountTradeModel) -> None:
        self.metrics.fills += 1
        self.metrics.total_fees += trade.fee
        self._rolling_fill_window_1m.append(
            (time.monotonic(), trade.qty * trade.price, bool(trade.is_taker))
        )
        self._prune_fill_window()

        fill = FillEvent(
            trade_id=trade.id,
            order_id=trade.order_id,
            market=trade.market,
            side=trade.side,
            price=trade.price,
            qty=trade.qty,
            fee=trade.fee,
            is_taker=trade.is_taker,
            timestamp=trade.created_time,
        )

        logger.info(
            "Fill: side=%s price=%s qty=%s fee=%s taker=%s",
            fill.side,
            fill.price,
            fill.qty,
            fill.fee,
            fill.is_taker,
        )

        for cb in self._fill_callbacks:
            try:
                cb(fill)
            except Exception as exc:
                logger.error("Fill callback error: %s", exc)

    def _dispatch_balance_update(self, balance) -> None:
        for cb in self._balance_callbacks:
            try:
                cb(balance)
            except Exception as exc:
                logger.error("Balance callback error: %s", exc)
