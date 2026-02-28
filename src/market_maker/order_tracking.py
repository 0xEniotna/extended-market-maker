"""
Order Tracking

Provides query methods, failure tracking, latency tracking, and zombie
detection that were previously inlined in ``OrderManager``.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Dict, Optional

from x10.perpetual.orders import OrderSide

logger = logging.getLogger(__name__)


class FailureTracker:
    """Tracks consecutive and rolling-window order placement failures."""

    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        self._attempt_timestamps: deque[float] = deque()
        self._failure_timestamps: deque[float] = deque()

    def record_attempt(self) -> None:
        self._attempt_timestamps.append(time.monotonic())

    def record_failure(self) -> None:
        self._failure_timestamps.append(time.monotonic())

    def failure_window_stats(self, window_s: float) -> Dict[str, float]:
        self._prune_failure_window(window_s)
        attempts = len(self._attempt_timestamps)
        failures = len(self._failure_timestamps)
        rate = float(failures / attempts) if attempts > 0 else 0.0
        return {
            "attempts": float(attempts),
            "failures": float(failures),
            "failure_rate": rate,
        }

    def reset(self) -> None:
        self.consecutive_failures = 0
        self._attempt_timestamps.clear()
        self._failure_timestamps.clear()

    def _prune_failure_window(self, window_s: float) -> None:
        cutoff = time.monotonic() - max(0.0, float(window_s))
        while self._attempt_timestamps and self._attempt_timestamps[0] < cutoff:
            self._attempt_timestamps.popleft()
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()


class LatencyTracker:
    """Rolling window of order placement latencies (ms)."""

    def __init__(self, maxlen: int = 50) -> None:
        self._samples: deque[float] = deque(maxlen=maxlen)
        self._send_ts: Dict[str, float] = {}

    @property
    def samples(self) -> deque[float]:
        """Expose raw samples for external consumers."""
        return self._samples

    def record_send(self, ext_id: str) -> None:
        self._send_ts[ext_id] = time.monotonic()

    def record_ack(self, ext_id: str) -> Optional[float]:
        send_ts = self._send_ts.pop(ext_id, None)
        if send_ts is None:
            return None
        latency_ms = (time.monotonic() - send_ts) * 1000.0
        self._samples.append(latency_ms)
        return latency_ms

    def discard(self, ext_id: str) -> None:
        self._send_ts.pop(ext_id, None)

    def avg_ms(self) -> float:
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def sample_count(self) -> int:
        return len(self._samples)


def find_zombie_orders(
    active_orders: Dict[str, object],
    pending_cancel: Dict[str, float],
    max_age_s: float,
) -> list:
    """Return orders older than *max_age_s* that never received a stream update."""
    now = time.monotonic()
    zombies = []
    for info in active_orders.values():
        age = now - info.placed_at  # type: ignore[attr-defined]
        if age < max_age_s:
            continue
        if info.external_id in pending_cancel:  # type: ignore[attr-defined]
            continue
        if info.last_stream_update_at is None:  # type: ignore[attr-defined]
            zombies.append(info)
    return zombies


def reserved_exposure(
    active_orders: Dict[str, object],
    pending_cancel: Dict[str, float],
    *,
    side: OrderSide,
    exclude_external_id: Optional[str] = None,
) -> tuple[Decimal, Decimal]:
    """Return (reserved_same_side_qty, reserved_open_notional_usd)."""
    side_name = str(side)
    same_side_qty = Decimal("0")
    open_notional = Decimal("0")
    for ext_id, info in active_orders.items():
        if exclude_external_id is not None and ext_id == exclude_external_id:
            continue
        if ext_id in pending_cancel:
            continue
        open_notional += info.size * info.price  # type: ignore[attr-defined]
        if str(info.side) == side_name:  # type: ignore[attr-defined]
            same_side_qty += info.size  # type: ignore[attr-defined]
    return same_side_qty, open_notional
