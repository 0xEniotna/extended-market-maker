"""
Venue Latency SLA Monitoring

Monitors order placement latency against configurable SLA thresholds.
When latency exceeds thresholds, the monitor can:
- Trigger wider spreads (via extra offset BPS)
- Halt quoting entirely (via quote halt signal)

On a Raspberry Pi with WiFi, latency can spike from 50ms to 2000ms+
during interference.  During high-latency periods, quotes are stale
by the time they reach the exchange, increasing adverse selection.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_S = 60.0
_MAX_SAMPLES = 200


@dataclass(frozen=True)
class LatencySLASnapshot:
    """Point-in-time latency SLA status."""

    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    sample_count: int = 0
    extra_offset_bps: Decimal = Decimal("0")
    halt_quoting: bool = False
    degraded: bool = False


class LatencyMonitor:
    """Rolling-window latency monitor with SLA thresholds.

    Parameters
    ----------
    warn_ms : float
        Latency above which spreads are widened.
    critical_ms : float
        Latency above which quoting should be halted.
    extra_offset_bps_per_ms : Decimal
        Additional offset BPS per ms of excess latency above warn_ms.
    max_extra_offset_bps : Decimal
        Cap on additional offset from latency.
    window_s : float
        Rolling window for latency statistics.
    """

    def __init__(
        self,
        *,
        warn_ms: float = 200.0,
        critical_ms: float = 1000.0,
        extra_offset_bps_per_ms: Decimal = Decimal("0.01"),
        max_extra_offset_bps: Decimal = Decimal("5"),
        window_s: float = _DEFAULT_WINDOW_S,
    ) -> None:
        self._warn_ms = warn_ms
        self._critical_ms = critical_ms
        self._extra_offset_bps_per_ms = extra_offset_bps_per_ms
        self._max_extra_offset_bps = max_extra_offset_bps
        self._window_s = max(1.0, window_s)

        self._samples: deque[tuple[float, float]] = deque(maxlen=_MAX_SAMPLES)

        self._degraded = False
        self._halt = False

    def record_latency(self, latency_ms: float) -> None:
        """Record an order placement latency sample."""
        self._samples.append((time.monotonic(), latency_ms))

    def _prune(self, now: float) -> list[float]:
        """Prune old samples and return active latency values."""
        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return [s[1] for s in self._samples]

    def evaluate(self) -> LatencySLASnapshot:
        """Evaluate current latency SLA status."""
        now = time.monotonic()
        values = self._prune(now)

        if not values:
            self._degraded = False
            self._halt = False
            return LatencySLASnapshot()

        avg = sum(values) / len(values)
        sorted_vals = sorted(values)
        p95_idx = max(0, int(len(sorted_vals) * 0.95) - 1)
        p95 = sorted_vals[p95_idx]
        max_val = sorted_vals[-1]

        # Determine SLA state based on p95 (not average — catches spikes).
        extra_offset = Decimal("0")
        halt = False
        degraded = False

        if p95 >= self._critical_ms:
            halt = True
            degraded = True
        elif p95 >= self._warn_ms:
            degraded = True
            excess_ms = Decimal(str(p95 - self._warn_ms))
            extra_offset = min(
                self._max_extra_offset_bps,
                excess_ms * self._extra_offset_bps_per_ms,
            )

        was_halt = self._halt
        was_degraded = self._degraded
        self._halt = halt
        self._degraded = degraded

        if halt and not was_halt:
            logger.warning(
                "LATENCY SLA CRITICAL: p95=%.0fms avg=%.0fms — halting quoting",
                p95, avg,
            )
        elif degraded and not was_degraded:
            logger.info(
                "LATENCY SLA DEGRADED: p95=%.0fms avg=%.0fms — widening by %.1fbps",
                p95, avg, extra_offset,
            )
        elif was_degraded and not degraded:
            logger.info(
                "LATENCY SLA OK: p95=%.0fms avg=%.0fms — normal spreads",
                p95, avg,
            )

        return LatencySLASnapshot(
            avg_latency_ms=avg,
            p95_latency_ms=p95,
            max_latency_ms=max_val,
            sample_count=len(values),
            extra_offset_bps=extra_offset,
            halt_quoting=halt,
            degraded=degraded,
        )

    @property
    def should_halt(self) -> bool:
        return self._halt

    @property
    def is_degraded(self) -> bool:
        return self._degraded
