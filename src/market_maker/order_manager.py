"""
Order Manager

Tracks active orders and provides place / cancel operations through the
x10 PerpetualTradingClient.  All orders are placed with post_only=True
(limit + GTT) to ensure maker status.

Integrates with AccountStreamManager — order fills, cancellations and
rejections are propagated via ``handle_order_update()`` so the internal
tracking dict stays accurate in real-time.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Callable, Dict, List, Optional

from x10.perpetual.orders import (
    OpenOrderModel,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from x10.perpetual.trading_client import PerpetualTradingClient

logger = logging.getLogger(__name__)

# Terminal order statuses that mean the order is no longer live
_TERMINAL_STATUSES = frozenset({
    OrderStatus.FILLED,
    OrderStatus.CANCELLED,
    OrderStatus.EXPIRED,
    OrderStatus.REJECTED,
})


@dataclass
class OrderInfo:
    """Metadata for a tracked order."""

    external_id: str
    side: OrderSide
    price: Decimal
    size: Decimal
    level: int
    exchange_order_id: Optional[str] = None
    placed_at: float = field(default_factory=time.monotonic)


@dataclass
class FlattenResult:
    """Result of a shutdown flatten attempt."""

    attempted: bool
    success: bool
    reason: str
    side: Optional[OrderSide] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None


# Callback invoked when an order for a specific level is removed (filled / cancelled).
# Signature: (side_value: str, level: int, external_id: str, *, rejected: bool)
LevelFreedCallback = Callable[..., None]


class OrderManager:
    """
    Places and cancels limit orders, tracking them by external_id.
    """

    def __init__(
        self,
        trading_client: PerpetualTradingClient,
        market_name: str,
    ) -> None:
        self._client = trading_client
        self._market_name = market_name
        self._active_orders: Dict[str, OrderInfo] = {}
        self._recent_orders_by_external_id: Dict[str, OrderInfo] = {}
        self._orders_by_exchange_id: Dict[str, OrderInfo] = {}

        # Callbacks fired when a level's order is removed by the stream
        self._level_freed_callbacks: List[LevelFreedCallback] = []

        # Running counter of consecutive placement failures (for circuit breaker)
        self.consecutive_failures: int = 0
        self._attempt_timestamps: deque[float] = deque()
        self._failure_timestamps: deque[float] = deque()

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_level_freed(self, cb: LevelFreedCallback) -> None:
        """Register a callback invoked when an order is removed from tracking."""
        self._level_freed_callbacks.append(cb)

    # ------------------------------------------------------------------
    # Account-stream integration
    # ------------------------------------------------------------------

    def handle_order_update(self, order: OpenOrderModel) -> None:
        """Called by AccountStreamManager on every order status change.

        Removes terminal orders from tracking so the dict stays accurate.
        """
        tracked = self._active_orders.get(order.external_id)
        if tracked is not None and tracked.exchange_order_id is None:
            exchange_id = getattr(order, "id", None)
            if exchange_id is None:
                exchange_id = getattr(order, "order_id", None)
            if exchange_id is not None:
                tracked.exchange_order_id = str(exchange_id)
                self._orders_by_exchange_id[str(exchange_id)] = tracked

        if order.status not in _TERMINAL_STATUSES:
            return

        # Find by external_id
        info = self._active_orders.pop(order.external_id, None)
        if info is None:
            return
        self._recent_orders_by_external_id[info.external_id] = info
        if info.exchange_order_id:
            self._orders_by_exchange_id[info.exchange_order_id] = info

        is_rejected = order.status == OrderStatus.REJECTED
        if is_rejected:
            self.consecutive_failures += 1
            self._record_failure()

        logger.info(
            "Order %s (ext_id=%s, level=%d) reached terminal status: %s",
            order.status,
            info.external_id,
            info.level,
            order.status,
        )

        # Notify strategy that this level's slot is now free
        status_reason = getattr(order, "status_reason", None)
        for cb in self._level_freed_callbacks:
            try:
                cb(
                    str(info.side),
                    info.level,
                    info.external_id,
                    rejected=is_rejected,
                    status=str(order.status),
                    reason=str(status_reason) if status_reason is not None else None,
                    price=info.price,
                )
            except Exception as exc:
                logger.error("level_freed callback error: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def place_order(
        self,
        side: OrderSide,
        price: Decimal,
        size: Decimal,
        level: int,
    ) -> Optional[str]:
        """
        Place a post-only limit order.

        Returns the external_id on success, None on failure.
        """
        external_id = self._generate_external_id()
        self._record_attempt()
        try:
            resp = await self._client.place_order(
                market_name=self._market_name,
                amount_of_synthetic=size,
                price=price,
                side=side,
                order_type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTT,
                post_only=True,
                external_id=external_id,
            )

            # Validate SDK response status
            if hasattr(resp, "status") and hasattr(resp, "error"):
                if resp.status != "OK" or resp.error is not None:
                    self.consecutive_failures += 1
                    self._record_failure()
                    side_name = getattr(side, "value", str(side))
                    logger.warning(
                        "Order rejected: side=%s price=%s error=%s failures=%d",
                        side_name,
                        price,
                        resp.error,
                        self.consecutive_failures,
                    )
                    return None

            # Extract exchange order id
            exchange_id = self._extract_exchange_id(resp)

            info = OrderInfo(
                external_id=external_id,
                side=side,
                price=price,
                size=size,
                level=level,
                exchange_order_id=exchange_id,
            )
            self._active_orders[external_id] = info
            self._recent_orders_by_external_id[external_id] = info
            if exchange_id is not None:
                self._orders_by_exchange_id[exchange_id] = info
            self.consecutive_failures = 0

            logger.info(
                "Order placed: side=%s price=%s size=%s level=%d ext_id=%s exch_id=%s",
                side,
                price,
                size,
                level,
                external_id,
                exchange_id,
            )
            return external_id

        except Exception as exc:
            self.consecutive_failures += 1
            self._record_failure()
            logger.error(
                "Failed to place order: side=%s price=%s size=%s level=%d error=%s",
                side,
                price,
                size,
                level,
                exc,
            )
            return None

    async def cancel_order(self, external_id: str) -> bool:
        """Cancel a single order by its external_id."""
        info = self._active_orders.get(external_id)
        if info is None:
            return False

        try:
            await self._client.orders.cancel_order_by_external_id(
                order_external_id=external_id,
            )
            logger.info(
                "Order cancel requested: ext_id=%s exchange_id=%s",
                external_id,
                info.exchange_order_id,
            )
            # Do NOT pop from _active_orders here — the account stream
            # will deliver the CANCELLED event and handle_order_update()
            # will do the cleanup.
            return True
        except Exception as exc:
            logger.warning(
                "Cancel failed for ext_id=%s exchange_id=%s: %s",
                external_id,
                info.exchange_order_id,
                exc,
            )
            return False

    async def cancel_all_orders(self) -> None:
        """Cancel every order for this market using mass-cancel (single API call)."""
        try:
            await self._client.orders.mass_cancel(
                markets=[self._market_name],
            )
            logger.info(
                "Mass cancel issued for market=%s (%d tracked orders)",
                self._market_name,
                len(self._active_orders),
            )
            # The stream will confirm each cancellation; clear tracking optimistically
            self._active_orders.clear()
        except Exception as exc:
            logger.error(
                "Mass cancel failed for market=%s: %s — falling back to per-order cancel",
                self._market_name,
                exc,
            )
            # Fallback: cancel individually
            ext_ids = list(self._active_orders.keys())
            for ext_id in ext_ids:
                await self.cancel_order(ext_id)

    async def flatten_position(
        self,
        *,
        signed_position: Decimal,
        best_bid: Optional[Decimal],
        best_ask: Optional[Decimal],
        tick_size: Decimal,
        min_order_size: Decimal,
        size_step: Decimal,
        slippage_bps: Decimal,
    ) -> FlattenResult:
        """Submit a reduce-only MARKET+IOC order to flatten a signed position."""
        if signed_position == 0:
            return FlattenResult(
                attempted=False,
                success=True,
                reason="already_flat",
            )

        side = OrderSide.SELL if signed_position > 0 else OrderSide.BUY
        close_size = self._round_down_to_step(abs(signed_position), size_step)
        if close_size < min_order_size:
            logger.warning(
                "Skipping flatten for market=%s: |position|=%s rounds to size=%s < min_order_size=%s",
                self._market_name,
                signed_position,
                close_size,
                min_order_size,
            )
            return FlattenResult(
                attempted=False,
                success=False,
                reason="below_min_order_size",
                side=side,
                size=close_size,
            )

        ref_price = best_bid if side == OrderSide.SELL else best_ask
        if ref_price is None or ref_price <= 0:
            # Fallback to opposite side if one side is temporarily missing.
            ref_price = best_ask if side == OrderSide.SELL else best_bid
        if ref_price is None or ref_price <= 0:
            logger.error(
                "Cannot flatten position for market=%s: missing usable orderbook price (bid=%s ask=%s)",
                self._market_name,
                best_bid,
                best_ask,
            )
            return FlattenResult(
                attempted=False,
                success=False,
                reason="missing_orderbook_price",
                side=side,
                size=close_size,
            )

        bps = max(Decimal("0"), slippage_bps) / Decimal("10000")
        if side == OrderSide.SELL:
            target_price = ref_price * (Decimal("1") - bps)
        else:
            target_price = ref_price * (Decimal("1") + bps)
        price = self._round_to_tick_for_side(target_price, tick_size, side)
        if price <= 0:
            price = tick_size if tick_size > 0 else Decimal("1")

        external_id = f"mm-flat-{uuid.uuid4().hex[:12]}"
        try:
            resp = await self._client.place_order(
                market_name=self._market_name,
                amount_of_synthetic=close_size,
                price=price,
                side=side,
                order_type=OrderType.MARKET,
                time_in_force=TimeInForce.IOC,
                post_only=False,
                reduce_only=True,
                external_id=external_id,
            )
            if hasattr(resp, "status") and hasattr(resp, "error"):
                if resp.status != "OK" or resp.error is not None:
                    logger.error(
                        "Flatten order rejected: market=%s side=%s size=%s price=%s error=%s",
                        self._market_name,
                        side,
                        close_size,
                        price,
                        resp.error,
                    )
                    return FlattenResult(
                        attempted=True,
                        success=False,
                        reason=f"rejected:{resp.error}",
                        side=side,
                        size=close_size,
                        price=price,
                    )
            logger.warning(
                "Submitted shutdown flatten order: market=%s side=%s size=%s price=%s ext_id=%s",
                self._market_name,
                side,
                close_size,
                price,
                external_id,
            )
            return FlattenResult(
                attempted=True,
                success=True,
                reason="submitted",
                side=side,
                size=close_size,
                price=price,
            )
        except Exception as exc:
            logger.exception(
                "Failed to submit shutdown flatten order: market=%s side=%s size=%s price=%s",
                self._market_name,
                side,
                close_size,
                price,
            )
            return FlattenResult(
                attempted=True,
                success=False,
                reason=f"exception:{exc}",
                side=side,
                size=close_size,
                price=price,
            )

    def get_active_orders(self) -> Dict[str, OrderInfo]:
        return dict(self._active_orders)

    def find_order_by_exchange_id(self, exchange_order_id: str) -> Optional[OrderInfo]:
        """Return tracked metadata for an exchange order id, if known."""
        return self._orders_by_exchange_id.get(exchange_order_id)

    def find_order_by_external_id(self, external_id: str) -> Optional[OrderInfo]:
        """Return tracked metadata for an external id (active or recent)."""
        return self._active_orders.get(external_id) or self._recent_orders_by_external_id.get(external_id)

    def remove_order(self, external_id: str) -> None:
        """Remove an order from tracking without cancelling (e.g. after fill)."""
        self._active_orders.pop(external_id, None)

    def failure_window_stats(self, window_s: float) -> Dict[str, float]:
        """Return rolling attempt/failure stats for *window_s* seconds."""
        self._prune_failure_window(window_s)
        attempts = len(self._attempt_timestamps)
        failures = len(self._failure_timestamps)
        rate = float(failures / attempts) if attempts > 0 else 0.0
        return {
            "attempts": float(attempts),
            "failures": float(failures),
            "failure_rate": rate,
        }

    def reset_failure_tracking(self) -> None:
        """Reset consecutive and rolling failure counters."""
        self.consecutive_failures = 0
        self._attempt_timestamps.clear()
        self._failure_timestamps.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_external_id() -> str:
        return f"mm-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _round_down_to_step(value: Decimal, step: Decimal) -> Decimal:
        if step <= 0:
            return value
        return (value / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _round_to_tick_for_side(price: Decimal, tick_size: Decimal, side: OrderSide) -> Decimal:
        if tick_size <= 0:
            return price
        rounding = ROUND_UP if side == OrderSide.BUY else ROUND_DOWN
        return (price / tick_size).to_integral_value(rounding=rounding) * tick_size

    @staticmethod
    def _extract_exchange_id(resp) -> Optional[str]:
        """Extract the exchange order ID from the SDK response."""
        data = resp.data if hasattr(resp, "data") else resp
        # Object with .id attribute (most common SDK path)
        if hasattr(data, "id") and data.id is not None:
            return str(data.id)
        # Dict fallback
        if isinstance(data, dict):
            for key in ("id", "order_id", "orderId"):
                if key in data and data[key] is not None:
                    return str(data[key])
        return None

    def _record_attempt(self) -> None:
        self._attempt_timestamps.append(time.monotonic())

    def _record_failure(self) -> None:
        self._failure_timestamps.append(time.monotonic())

    def _prune_failure_window(self, window_s: float) -> None:
        cutoff = time.monotonic() - max(0.0, float(window_s))
        while self._attempt_timestamps and self._attempt_timestamps[0] < cutoff:
            self._attempt_timestamps.popleft()
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()
