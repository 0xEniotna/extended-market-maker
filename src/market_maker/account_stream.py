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
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, List, Optional

from x10.perpetual.accounts import AccountStreamDataModel
from x10.perpetual.configuration import EndpointConfig
from x10.perpetual.orders import OpenOrderModel, OrderSide, OrderStatus, OrderStatusReason
from x10.perpetual.positions import PositionModel
from x10.perpetual.stream_client.stream_client import PerpetualStreamClient
from x10.perpetual.trades import AccountTradeModel
from x10.utils.http import WrappedStreamResponse

logger = logging.getLogger(__name__)


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

        # Callbacks registered by other components
        self._order_callbacks: List[OrderUpdateCallback] = []
        self._position_callbacks: List[PositionUpdateCallback] = []
        self._fill_callbacks: List[FillCallback] = []

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background stream task."""
        self._task = asyncio.create_task(self._stream_loop(), name="mm-account-stream")
        logger.info("Account stream manager started for %s", self._market_name)

    async def stop(self) -> None:
        """Cancel the background stream task."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Account stream manager stopped")

    # ------------------------------------------------------------------
    # Stream loop (reconnects automatically)
    # ------------------------------------------------------------------

    async def _stream_loop(self) -> None:
        """Connect to the account stream and process events.  Reconnects on drop."""
        while True:
            try:
                async with self._stream_client.subscribe_to_account_updates(
                    self._api_key
                ) as stream:
                    logger.info("Account stream connected")
                    async for event in stream:
                        await self._handle_event(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Account stream error: %s — reconnecting in 2s", exc)
            await asyncio.sleep(2.0)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    async def _handle_event(
        self, event: WrappedStreamResponse[AccountStreamDataModel]
    ) -> None:
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
