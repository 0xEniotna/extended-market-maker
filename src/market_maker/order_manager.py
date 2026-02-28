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
from decimal import Decimal
from typing import Callable, Dict, List, Optional

from x10.perpetual.orders import (
    OpenOrderModel,
    OrderSide,
    OrderStatus,
)
from x10.perpetual.trading_client import PerpetualTradingClient

from .fee_resolver import FeeResolver
from .order_lifecycle import (
    cancel_all_orders as _cancel_all,
)
from .order_lifecycle import (
    cancel_order as _cancel_one,
)
from .order_lifecycle import (
    place_order as _place,
)
from .order_lifecycle import (
    sweep_pending_cancels as _sweep,
)
from .order_rate_state import OrderRateState
from .order_tracking import (
    FailureTracker,
    LatencyTracker,
)
from .order_tracking import (
    find_zombie_orders as _find_zombies,
)
from .order_tracking import (
    reserved_exposure as _reserved_exposure,
)
from .position_flattener import (
    FlattenResult,
    extract_exchange_id,
    round_down_to_step,
    round_to_tick_for_side,
)
from .position_flattener import (
    flatten_position as _do_flatten,
)

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
    last_stream_update_at: Optional[float] = None


# Re-export FlattenResult for backwards compatibility.
FlattenResult = FlattenResult  # noqa: F811

# Callback invoked when an order for a specific level is removed.
LevelFreedCallback = Callable[..., None]


class OrderManager:
    """Places and cancels limit orders, tracking them by external_id."""

    def __init__(
        self,
        trading_client: PerpetualTradingClient,
        market_name: str,
        *,
        max_orders_per_second: float = 10.0,
        maintenance_pause_s: float = 60.0,
        fee_resolver: Optional[FeeResolver] = None,
        rate_limit_degraded_s: float = 20.0,
        rate_limit_halt_window_s: float = 60.0,
        rate_limit_halt_hits: int = 5,
        rate_limit_halt_s: float = 30.0,
        rate_limit_extra_offset_bps: Decimal = Decimal("5"),
        rate_limit_reprice_multiplier: Decimal = Decimal("2"),
    ) -> None:
        self._client = trading_client
        self._market_name = market_name
        self._fee_resolver = fee_resolver
        self._active_orders: Dict[str, OrderInfo] = {}
        self._recent_orders_by_external_id: Dict[str, OrderInfo] = {}
        self._orders_by_exchange_id: Dict[str, OrderInfo] = {}
        self._pending_placements: Dict[str, OrderInfo] = {}
        self._pending_cancel: Dict[str, float] = {}
        self._level_freed_callbacks: List[LevelFreedCallback] = []

        self._failures = FailureTracker()
        self._latency = LatencyTracker()
        self._rate_state = OrderRateState(
            market_name,
            max_orders_per_second=max_orders_per_second,
            maintenance_pause_s=maintenance_pause_s,
            rate_limit_degraded_s=rate_limit_degraded_s,
            rate_limit_halt_window_s=rate_limit_halt_window_s,
            rate_limit_halt_hits=rate_limit_halt_hits,
            rate_limit_halt_s=rate_limit_halt_s,
            rate_limit_extra_offset_bps=rate_limit_extra_offset_bps,
            rate_limit_reprice_multiplier=rate_limit_reprice_multiplier,
        )

        self._inflight_count: int = 0
        self._inflight_zero_event: asyncio.Event = asyncio.Event()
        self._inflight_zero_event.set()

    @property
    def consecutive_failures(self) -> int:
        return self._failures.consecutive_failures

    @consecutive_failures.setter
    def consecutive_failures(self, value: int) -> None:
        self._failures.consecutive_failures = value

    @property
    def _latency_samples(self) -> deque:
        return self._latency.samples

    def set_journal(self, journal: object) -> None:
        self._rate_state.set_journal(journal)

    def start_rate_limiter(self) -> None:
        self._rate_state.start_rate_limiter()

    async def stop_rate_limiter(self) -> None:
        await self._rate_state.stop_rate_limiter()

    @property
    def in_maintenance(self) -> bool:
        return self._rate_state.in_maintenance

    @property
    def maintenance_remaining_s(self) -> float:
        return self._rate_state.maintenance_remaining_s

    @property
    def in_rate_limit_degraded(self) -> bool:
        return self._rate_state.in_rate_limit_degraded

    @property
    def in_rate_limit_halt(self) -> bool:
        return self._rate_state.in_rate_limit_halt

    @property
    def rate_limit_extra_offset_bps(self) -> Decimal:
        return self._rate_state.rate_limit_extra_offset_bps

    @property
    def rate_limit_reprice_multiplier(self) -> Decimal:
        return self._rate_state.rate_limit_reprice_multiplier

    def rate_limit_hits_in_window(self) -> int:
        return self._rate_state.rate_limit_hits_in_window()

    def _record_rate_limit_hit(self) -> bool:
        return self._rate_state.record_rate_limit_hit()

    def _check_maintenance_response(self, resp=None, exc: Optional[Exception] = None) -> bool:
        return self._rate_state.check_maintenance_response(resp=resp, exc=exc)

    # In-flight operation tracking

    def _begin_inflight(self) -> None:
        self._inflight_count += 1
        self._inflight_zero_event.clear()

    def _end_inflight(self) -> None:
        self._inflight_count = max(0, self._inflight_count - 1)
        if self._inflight_count == 0:
            self._inflight_zero_event.set()

    async def wait_for_inflight(self, timeout_s: float = 5.0) -> bool:
        try:
            await asyncio.wait_for(
                self._inflight_zero_event.wait(), timeout=timeout_s,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for %d in-flight operations after %.1fs",
                self._inflight_count, timeout_s,
            )
            return False

    def on_level_freed(self, cb: LevelFreedCallback) -> None:
        self._level_freed_callbacks.append(cb)

    # Account-stream integration

    def handle_order_update(self, order: OpenOrderModel) -> None:
        """Process order status change from the account stream."""
        ext_id = order.external_id

        if ext_id not in self._active_orders and ext_id in self._pending_placements:
            info = self._pending_placements.pop(ext_id)
            self._active_orders[ext_id] = info
            self._recent_orders_by_external_id[ext_id] = info
            logger.info("Promoted pending placement to active: ext_id=%s", ext_id)

        tracked = self._active_orders.get(ext_id)
        if tracked is not None:
            tracked.last_stream_update_at = time.monotonic()
            self._latency.record_ack(ext_id)
            if tracked.exchange_order_id is None:
                exchange_id = getattr(order, "id", None)
                if exchange_id is None:
                    exchange_id = getattr(order, "order_id", None)
                if exchange_id is not None:
                    tracked.exchange_order_id = str(exchange_id)
                    self._orders_by_exchange_id[str(exchange_id)] = tracked

        if order.status not in _TERMINAL_STATUSES:
            return

        removed = self._active_orders.pop(ext_id, None)
        self._pending_cancel.pop(ext_id, None)
        if removed is None:
            return
        self._recent_orders_by_external_id[removed.external_id] = removed
        if removed.exchange_order_id:
            self._orders_by_exchange_id[removed.exchange_order_id] = removed

        is_rejected = order.status == OrderStatus.REJECTED
        if is_rejected:
            self.consecutive_failures += 1
            self._record_failure()

        logger.info(
            "Order %s (ext_id=%s, level=%d) reached terminal status: %s",
            order.status, removed.external_id, removed.level, order.status,
        )

        status_reason = getattr(order, "status_reason", None)
        for cb in self._level_freed_callbacks:
            try:
                cb(
                    str(removed.side), removed.level, removed.external_id,
                    rejected=is_rejected,
                    status=str(order.status),
                    reason=str(status_reason) if status_reason is not None else None,
                    price=removed.price,
                )
            except Exception as exc:
                logger.error("level_freed callback error: %s", exc)

    # Order lifecycle (delegated to order_lifecycle module)

    async def place_order(self, side, price, size, level) -> Optional[str]:
        return await _place(self, side, price, size, level)

    async def cancel_order(self, external_id: str) -> bool:
        return await _cancel_one(self, external_id)

    async def cancel_all_orders(self) -> None:
        await _cancel_all(self)

    def sweep_pending_cancels(self, timeout_s: float = 10.0) -> int:
        return _sweep(self, timeout_s=timeout_s)

    def is_pending_cancel(self, external_id: str) -> bool:
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
        return await _do_flatten(
            self._client, self._market_name, self._fee_resolver,
            signed_position=signed_position,
            best_bid=best_bid, best_ask=best_ask,
            tick_size=tick_size, min_order_size=min_order_size,
            size_step=size_step, slippage_bps=slippage_bps,
            risk_mgr=risk_mgr, wait_for_fill_s=wait_for_fill_s,
            last_known_mid=last_known_mid,
            rate_limit_hit_callback=self._record_rate_limit_hit,
            cancel_all_callback=self.cancel_all_orders,
        )

    def get_active_orders(self) -> Dict[str, OrderInfo]:
        return dict(self._active_orders)

    def get_active_order(self, external_id: Optional[str]) -> Optional[OrderInfo]:
        if external_id is None:
            return None
        return self._active_orders.get(external_id)

    def active_order_count(self) -> int:
        return len(self._active_orders)

    def reserved_exposure(
        self, *, side: OrderSide, exclude_external_id: Optional[str] = None,
    ) -> tuple[Decimal, Decimal]:
        return _reserved_exposure(
            self._active_orders, self._pending_cancel,
            side=side, exclude_external_id=exclude_external_id,
        )

    def find_order_by_exchange_id(self, exchange_order_id: str) -> Optional[OrderInfo]:
        return self._orders_by_exchange_id.get(exchange_order_id)

    def find_order_by_external_id(self, external_id: str) -> Optional[OrderInfo]:
        return (self._active_orders.get(external_id)
                or self._recent_orders_by_external_id.get(external_id))

    def remove_order(self, external_id: str) -> None:
        self._active_orders.pop(external_id, None)

    def find_zombie_orders(self, max_age_s: float) -> List[OrderInfo]:
        return _find_zombies(self._active_orders, self._pending_cancel, max_age_s)

    def failure_window_stats(self, window_s: float) -> Dict[str, float]:
        return self._failures.failure_window_stats(window_s)

    def reset_failure_tracking(self) -> None:
        self._failures.reset()

    def avg_placement_latency_ms(self) -> float:
        return self._latency.avg_ms()

    def latency_sample_count(self) -> int:
        return self._latency.sample_count()

    async def _maintenance_cancel_orders(self) -> None:
        try:
            await self.cancel_all_orders()
            logger.info(
                "Cancelled resting orders due to exchange maintenance for %s",
                self._market_name,
            )
        except Exception as exc:
            logger.error("Failed to cancel orders during maintenance: %s", exc)

    @staticmethod
    def _generate_external_id() -> str:
        return f"mm-{uuid.uuid4().hex[:12]}"

    _round_down_to_step = staticmethod(round_down_to_step)
    _round_to_tick_for_side = staticmethod(round_to_tick_for_side)
    _extract_exchange_id = staticmethod(extract_exchange_id)

    def _record_attempt(self) -> None:
        self._failures.record_attempt()

    def _record_failure(self) -> None:
        self._failures.record_failure()
