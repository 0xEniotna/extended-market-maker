"""
Order Rate State

Tracks rate-limit and exchange-maintenance state for the order manager.
Extracted to keep OrderManager focused on order lifecycle logic.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


class OrderRateState:
    """Manages token-bucket rate limiting, maintenance detection, and rate-limit defense."""

    # HTTP status codes / error strings that indicate exchange maintenance.
    _MAINTENANCE_CODES = frozenset({503, "503"})
    _MAINTENANCE_ERROR_PATTERNS = ("maintenance", "service unavailable", "503")

    def __init__(
        self,
        market_name: str,
        *,
        max_orders_per_second: float = 10.0,
        maintenance_pause_s: float = 60.0,
        rate_limit_degraded_s: float = 20.0,
        rate_limit_halt_window_s: float = 60.0,
        rate_limit_halt_hits: int = 5,
        rate_limit_halt_s: float = 30.0,
        rate_limit_extra_offset_bps: Decimal = Decimal("5"),
        rate_limit_reprice_multiplier: Decimal = Decimal("2"),
    ) -> None:
        self._market_name = market_name

        # --- Token-bucket rate limiter ---
        self._max_orders_per_second = max(0.1, max_orders_per_second)
        self._rate_tokens = int(max(1, max_orders_per_second))
        self._rate_semaphore: asyncio.Semaphore = asyncio.Semaphore(self._rate_tokens)
        self._rate_replenish_task: Optional[asyncio.Task] = None

        # --- Exchange maintenance detection ---
        self._maintenance_pause_s = maintenance_pause_s
        self._maintenance_until: float = 0.0

        # --- Rate-limit defensive state ---
        self._rate_limit_degraded_s = max(0.0, float(rate_limit_degraded_s))
        self._rate_limit_halt_window_s = max(1.0, float(rate_limit_halt_window_s))
        self._rate_limit_halt_hits = max(1, int(rate_limit_halt_hits))
        self._rate_limit_halt_s = max(1.0, float(rate_limit_halt_s))
        self._rate_limit_extra_offset_bps = max(Decimal("0"), Decimal(str(rate_limit_extra_offset_bps)))
        self._rate_limit_reprice_multiplier = max(
            Decimal("1"), Decimal(str(rate_limit_reprice_multiplier))
        )
        self._rate_limit_hit_ts: deque[float] = deque()
        self._rate_limit_degraded_until: float = 0.0
        self._rate_limit_halt_until: float = 0.0

        # Optional journal for event recording.
        self._journal: Optional[object] = None

    def set_journal(self, journal: object) -> None:
        self._journal = journal

    # ------------------------------------------------------------------
    # Rate limiter lifecycle
    # ------------------------------------------------------------------

    def start_rate_limiter(self) -> None:
        if self._rate_replenish_task is None:
            self._rate_replenish_task = asyncio.create_task(
                self._replenish_tokens(), name="mm-rate-limiter"
            )

    async def stop_rate_limiter(self) -> None:
        if self._rate_replenish_task is not None:
            self._rate_replenish_task.cancel()
            try:
                await self._rate_replenish_task
            except asyncio.CancelledError:
                pass
            self._rate_replenish_task = None

    async def _replenish_tokens(self) -> None:
        interval = 1.0 / self._max_orders_per_second
        while True:
            try:
                await asyncio.sleep(interval)
                if self._rate_semaphore._value < self._rate_tokens:
                    self._rate_semaphore.release()
            except asyncio.CancelledError:
                return

    async def acquire_rate_token(self, timeout: float = 2.0) -> bool:
        """Acquire a rate-limit token. Returns False on timeout."""
        try:
            await asyncio.wait_for(self._rate_semaphore.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------
    # Maintenance state
    # ------------------------------------------------------------------

    @property
    def in_maintenance(self) -> bool:
        return time.monotonic() < self._maintenance_until

    @property
    def maintenance_remaining_s(self) -> float:
        return max(0.0, self._maintenance_until - time.monotonic())

    @property
    def in_rate_limit_degraded(self) -> bool:
        return time.monotonic() < self._rate_limit_degraded_until

    @property
    def in_rate_limit_halt(self) -> bool:
        return time.monotonic() < self._rate_limit_halt_until

    @property
    def rate_limit_extra_offset_bps(self) -> Decimal:
        return self._rate_limit_extra_offset_bps if self.in_rate_limit_degraded else Decimal("0")

    @property
    def rate_limit_reprice_multiplier(self) -> Decimal:
        return self._rate_limit_reprice_multiplier if self.in_rate_limit_degraded else Decimal("1")

    def rate_limit_hits_in_window(self) -> int:
        self._prune_rate_limit_hits()
        return len(self._rate_limit_hit_ts)

    def _prune_rate_limit_hits(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._rate_limit_halt_window_s
        while self._rate_limit_hit_ts and self._rate_limit_hit_ts[0] < cutoff:
            self._rate_limit_hit_ts.popleft()

    def record_rate_limit_hit(self) -> bool:
        now = time.monotonic()
        self._rate_limit_hit_ts.append(now)
        self._prune_rate_limit_hits(now)
        if self._rate_limit_degraded_s > 0:
            self._rate_limit_degraded_until = max(
                self._rate_limit_degraded_until,
                now + self._rate_limit_degraded_s,
            )

        should_halt = len(self._rate_limit_hit_ts) >= self._rate_limit_halt_hits
        if should_halt:
            self._rate_limit_halt_until = max(
                self._rate_limit_halt_until,
                now + self._rate_limit_halt_s,
            )
            logger.error(
                "Rate-limit halt triggered for %s: hits=%d window=%.0fs halt=%.0fs",
                self._market_name,
                len(self._rate_limit_hit_ts),
                self._rate_limit_halt_window_s,
                self._rate_limit_halt_s,
            )
        else:
            logger.warning(
                "Rate-limit degraded mode for %s: hits=%d degraded_until=%.1f",
                self._market_name,
                len(self._rate_limit_hit_ts),
                self._rate_limit_degraded_until,
            )
        return should_halt

    # ------------------------------------------------------------------
    # Maintenance detection
    # ------------------------------------------------------------------

    def check_maintenance_response(self, resp=None, exc: Optional[Exception] = None) -> bool:
        """Return True and enter maintenance state if response/exception indicates 503."""
        if resp is not None:
            status_code = getattr(resp, "status_code", getattr(resp, "status", None))
            if status_code in self._MAINTENANCE_CODES:
                self._enter_maintenance("http_503")
                return True
            error = getattr(resp, "error", None)
            if error is not None and self._is_maintenance_error(str(error)):
                self._enter_maintenance(f"error:{error}")
                return True
        if exc is not None:
            exc_str = str(exc).lower()
            if any(p in exc_str for p in self._MAINTENANCE_ERROR_PATTERNS):
                self._enter_maintenance(f"exception:{exc}")
                return True
            status_code = getattr(exc, "status_code", getattr(exc, "status", None))
            if status_code in self._MAINTENANCE_CODES:
                self._enter_maintenance(f"exception_status:{status_code}")
                return True
        return False

    @staticmethod
    def _is_maintenance_error(error_str: str) -> bool:
        lower = error_str.lower()
        return any(p in lower for p in OrderRateState._MAINTENANCE_ERROR_PATTERNS)

    def _enter_maintenance(self, reason: str) -> None:
        self._maintenance_until = time.monotonic() + self._maintenance_pause_s
        logger.warning(
            "Exchange maintenance detected for market=%s: reason=%s — "
            "pausing order placement for %.0fs",
            self._market_name, reason, self._maintenance_pause_s,
        )
        if self._journal is not None:
            record_fn = getattr(self._journal, "record_exchange_event", None)
            if record_fn is not None:
                record_fn(
                    event_type="exchange_maintenance",
                    details={
                        "reason": reason,
                        "pause_s": self._maintenance_pause_s,
                        "market": self._market_name,
                    },
                )
