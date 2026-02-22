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

import asyncio
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

# Default timeout for pending-cancel orders before force-removal.
_PENDING_CANCEL_TIMEOUT_S = 10.0


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
    last_stream_update_at: Optional[float] = None


@dataclass
class FlattenResult:
    """Result of a shutdown flatten attempt."""

    attempted: bool
    success: bool
    reason: str
    side: Optional[OrderSide] = None
    size: Optional[Decimal] = None
    price: Optional[Decimal] = None
    remaining_position: Optional[Decimal] = None


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

        # Pending placements: orders submitted to the exchange but whose
        # place_order() call has not yet returned.  The account stream may
        # deliver a confirmation before the HTTP response arrives.
        self._pending_placements: Dict[str, OrderInfo] = {}

        # Orders marked as pending-cancel by cancel_all_orders().
        # Value is the monotonic time at which the cancel was issued.
        self._pending_cancel: Dict[str, float] = {}

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
        Also checks ``_pending_placements`` for orders whose stream
        confirmation arrived before ``place_order()`` returned.
        """
        ext_id = order.external_id

        # --- Promote from pending_placements if not yet in active_orders ---
        if ext_id not in self._active_orders and ext_id in self._pending_placements:
            info = self._pending_placements.pop(ext_id)
            self._active_orders[ext_id] = info
            self._recent_orders_by_external_id[ext_id] = info
            logger.info(
                "Promoted pending placement to active: ext_id=%s",
                ext_id,
            )

        tracked = self._active_orders.get(ext_id)
        if tracked is not None:
            # Record stream activity timestamp for zombie detection.
            tracked.last_stream_update_at = time.monotonic()
            if tracked.exchange_order_id is None:
                exchange_id = getattr(order, "id", None)
                if exchange_id is None:
                    exchange_id = getattr(order, "order_id", None)
                if exchange_id is not None:
                    tracked.exchange_order_id = str(exchange_id)
                    self._orders_by_exchange_id[str(exchange_id)] = tracked

        if order.status not in _TERMINAL_STATUSES:
            return

        # Find by external_id
        info = self._active_orders.pop(ext_id, None)
        # Also clear from pending_cancel if present.
        self._pending_cancel.pop(ext_id, None)
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

        # Register in pending_placements *before* the await so that
        # handle_order_update can find the order if the stream delivers
        # the confirmation before the HTTP response.
        pending_info = OrderInfo(
            external_id=external_id,
            side=side,
            price=price,
            size=size,
            level=level,
        )
        self._pending_placements[external_id] = pending_info

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
                    # Clean up pending placement on rejection
                    self._pending_placements.pop(external_id, None)
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

            # The order may have already been promoted from
            # _pending_placements by handle_order_update.
            if external_id in self._active_orders:
                info = self._active_orders[external_id]
                if exchange_id is not None and info.exchange_order_id is None:
                    info.exchange_order_id = exchange_id
                    self._orders_by_exchange_id[exchange_id] = info
            else:
                info = self._pending_placements.pop(external_id, None) or pending_info
                if exchange_id is not None:
                    info.exchange_order_id = exchange_id
                self._active_orders[external_id] = info
                if exchange_id is not None:
                    self._orders_by_exchange_id[exchange_id] = info

            self._recent_orders_by_external_id[external_id] = info
            # Always clear from pending now that it's in active
            self._pending_placements.pop(external_id, None)
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
            self._pending_placements.pop(external_id, None)
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
        """Cancel every order for this market using mass-cancel (single API call).

        Instead of clearing ``_active_orders`` immediately, marks every order
        as *pending_cancel* and lets the account stream confirm each
        cancellation.  A timeout sweeper (``sweep_pending_cancels``) handles
        stragglers that never receive a stream confirmation.
        """
        try:
            now = time.monotonic()
            for ext_id in list(self._active_orders):
                self._pending_cancel[ext_id] = now

            await self._client.orders.mass_cancel(
                markets=[self._market_name],
            )
            logger.info(
                "Mass cancel issued for market=%s (%d orders marked pending_cancel)",
                self._market_name,
                len(self._pending_cancel),
            )
        except Exception as exc:
            logger.error(
                "Mass cancel failed for market=%s: %s — falling back to per-order cancel",
                self._market_name,
                exc,
            )
            # Clear pending_cancel state since mass cancel failed
            self._pending_cancel.clear()
            # Fallback: cancel individually
            ext_ids = list(self._active_orders.keys())
            for ext_id in ext_ids:
                await self.cancel_order(ext_id)

    def sweep_pending_cancels(
        self,
        timeout_s: float = _PENDING_CANCEL_TIMEOUT_S,
    ) -> int:
        """Force-remove orders stuck in pending_cancel beyond *timeout_s*.

        Returns the number of orders force-removed.
        """
        now = time.monotonic()
        removed = 0
        for ext_id in list(self._pending_cancel):
            cancel_ts = self._pending_cancel[ext_id]
            if (now - cancel_ts) < timeout_s:
                continue
            info = self._active_orders.pop(ext_id, None)
            self._pending_cancel.pop(ext_id, None)
            if info is not None:
                self._recent_orders_by_external_id[ext_id] = info
                logger.warning(
                    "Force-removed pending_cancel order after %.1fs: "
                    "ext_id=%s exchange_id=%s level=%d",
                    now - cancel_ts,
                    ext_id,
                    info.exchange_order_id,
                    info.level,
                )
                removed += 1
        return removed

    def is_pending_cancel(self, external_id: str) -> bool:
        """Return True if the order is awaiting cancel confirmation."""
        return external_id in self._pending_cancel

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
        risk_mgr=None,
        wait_for_fill_s: float = 0,
        last_known_mid: Optional[Decimal] = None,
    ) -> FlattenResult:
        """Submit a reduce-only MARKET+IOC order to flatten a signed position.

        When *risk_mgr* and *wait_for_fill_s* > 0 are given, poll the risk
        manager's cached position after submission to verify the fill actually
        reduced the position.  ``remaining_position`` in the result reflects
        the position observed after the wait.

        One-sided book handling:
        - If the natural BBO side is missing, use the available side with extra
          slippage (``slippage_bps`` is applied to whichever price is used).
        - If both sides are missing, fall back to ``last_known_mid`` with 50 bps
          slippage.
        - Only give up if no price reference exists at all.
        """
        if signed_position == 0:
            return FlattenResult(
                attempted=False,
                success=True,
                reason="already_flat",
                remaining_position=Decimal("0"),
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

        # --- Price resolution with one-sided book fallback ---
        ref_price = best_bid if side == OrderSide.SELL else best_ask
        effective_slippage = slippage_bps
        if ref_price is None or ref_price <= 0:
            # Fallback to opposite side with extra slippage.
            ref_price = best_ask if side == OrderSide.SELL else best_bid
            if ref_price is not None and ref_price > 0:
                logger.warning(
                    "Flatten using opposite-side BBO for market=%s: "
                    "natural side missing, using ref_price=%s with slippage=%sbps",
                    self._market_name,
                    ref_price,
                    effective_slippage,
                )
        if ref_price is None or ref_price <= 0:
            # Both sides missing: fall back to last known mid with 50bps slippage.
            if last_known_mid is not None and last_known_mid > 0:
                ref_price = last_known_mid
                effective_slippage = max(slippage_bps, Decimal("50"))
                logger.warning(
                    "Flatten using last known mid for market=%s: "
                    "both BBO sides missing, ref_price=%s slippage=%sbps",
                    self._market_name,
                    ref_price,
                    effective_slippage,
                )
            else:
                logger.error(
                    "Cannot flatten position for market=%s: "
                    "no price reference at all (bid=%s ask=%s last_mid=%s)",
                    self._market_name,
                    best_bid,
                    best_ask,
                    last_known_mid,
                )
                return FlattenResult(
                    attempted=False,
                    success=False,
                    reason="missing_orderbook_price",
                    side=side,
                    size=close_size,
                )

        bps = max(Decimal("0"), effective_slippage) / Decimal("10000")
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

            # --- Wait-for-fill verification ---
            remaining_position: Optional[Decimal] = None
            if risk_mgr is not None and wait_for_fill_s > 0:
                remaining_position = await self._wait_for_position_change(
                    risk_mgr=risk_mgr,
                    initial_position=signed_position,
                    timeout_s=wait_for_fill_s,
                )

            return FlattenResult(
                attempted=True,
                success=True,
                reason="submitted",
                side=side,
                size=close_size,
                price=price,
                remaining_position=remaining_position,
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

    def get_active_order(self, external_id: Optional[str]) -> Optional[OrderInfo]:
        """Return one active order by external id without copying the order map."""
        if external_id is None:
            return None
        return self._active_orders.get(external_id)

    def active_order_count(self) -> int:
        """Return active order count without allocating a copied dict."""
        return len(self._active_orders)

    def reserved_exposure(
        self,
        *,
        side: OrderSide,
        exclude_external_id: Optional[str] = None,
    ) -> tuple[Decimal, Decimal]:
        """Return (reserved_same_side_qty, reserved_open_notional_usd).

        The exclusion id lets callers ignore the currently tracked level order
        while sizing a replacement order.  Orders pending cancellation are
        excluded since they will not be filled.
        """
        side_name = str(side)
        same_side_qty = Decimal("0")
        open_notional = Decimal("0")
        for ext_id, info in self._active_orders.items():
            if exclude_external_id is not None and ext_id == exclude_external_id:
                continue
            # Skip orders that are awaiting cancel confirmation
            if ext_id in self._pending_cancel:
                continue
            open_notional += info.size * info.price
            if str(info.side) == side_name:
                same_side_qty += info.size
        return same_side_qty, open_notional

    def find_order_by_exchange_id(self, exchange_order_id: str) -> Optional[OrderInfo]:
        """Return tracked metadata for an exchange order id, if known."""
        return self._orders_by_exchange_id.get(exchange_order_id)

    def find_order_by_external_id(self, external_id: str) -> Optional[OrderInfo]:
        """Return tracked metadata for an external id (active or recent)."""
        return self._active_orders.get(external_id) or self._recent_orders_by_external_id.get(external_id)

    def remove_order(self, external_id: str) -> None:
        """Remove an order from tracking without cancelling (e.g. after fill)."""
        self._active_orders.pop(external_id, None)

    def find_zombie_orders(self, max_age_s: float) -> List[OrderInfo]:
        """Return orders older than *max_age_s* that never received a stream update.

        These "zombie" orders were likely lost — the exchange may have
        processed them but the stream confirmation never arrived.
        """
        now = time.monotonic()
        zombies: List[OrderInfo] = []
        for info in self._active_orders.values():
            age = now - info.placed_at
            if age < max_age_s:
                continue
            # Skip orders awaiting cancel
            if info.external_id in self._pending_cancel:
                continue
            # A zombie is one that never received *any* stream update
            if info.last_stream_update_at is None:
                zombies.append(info)
        return zombies

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

    @staticmethod
    async def _wait_for_position_change(
        *,
        risk_mgr,
        initial_position: Decimal,
        timeout_s: float,
        poll_interval_s: float = 0.25,
    ) -> Decimal:
        """Poll risk_mgr position until it differs from *initial_position* or timeout."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = risk_mgr.get_current_position()
            if current != initial_position:
                logger.info(
                    "Flatten fill confirmed: position moved %s -> %s",
                    initial_position,
                    current,
                )
                return current
            await asyncio.sleep(poll_interval_s)

        current = risk_mgr.get_current_position()
        if current == initial_position:
            logger.warning(
                "Flatten wait timed out after %.1fs: position still %s",
                timeout_s,
                current,
            )
        return current

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
