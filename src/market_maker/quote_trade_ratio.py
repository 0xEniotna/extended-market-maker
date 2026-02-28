"""
Quote-to-Trade Ratio Monitoring

Tracks the ratio of order modifications (place + cancel) to fills, a key
metric that exchanges monitor for market makers.  High ratios can result
in API throttling, fee penalties, or account suspension.

The tracker maintains a rolling window and can trigger warnings or
automatic spread widening when the ratio exceeds configurable thresholds.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

logger = logging.getLogger(__name__)

# Rolling window for QTR computation.
_DEFAULT_WINDOW_S = 300.0  # 5 minutes


@dataclass(frozen=True)
class QTRSnapshot:
    """Point-in-time quote-to-trade ratio metrics."""

    quotes_in_window: int = 0
    fills_in_window: int = 0
    ratio: Decimal = Decimal("0")
    window_s: float = _DEFAULT_WINDOW_S
    warn_active: bool = False
    critical_active: bool = False


class QuoteTradeRatioTracker:
    """Rolling-window quote-to-trade ratio monitor.

    A "quote" is any order placement or cancel-and-replace.
    A "fill" is any maker fill.

    Parameters
    ----------
    window_s : float
        Rolling window in seconds (default 300s = 5 min).
    warn_threshold : float
        Ratio above which a warning is logged.
    critical_threshold : float
        Ratio above which critical alert is triggered
        (can be used to widen spreads or pause quoting).
    """

    def __init__(
        self,
        *,
        window_s: float = _DEFAULT_WINDOW_S,
        warn_threshold: float = 50.0,
        critical_threshold: float = 100.0,
    ) -> None:
        self._window_s = max(1.0, window_s)
        self._warn_threshold = warn_threshold
        self._critical_threshold = critical_threshold

        self._quote_timestamps: deque[float] = deque()
        self._fill_timestamps: deque[float] = deque()

        self._warn_active = False
        self._critical_active = False

    def record_quote(self) -> None:
        """Record an order placement or cancel-and-replace event."""
        self._quote_timestamps.append(time.monotonic())

    def record_fill(self) -> None:
        """Record a maker fill event."""
        self._fill_timestamps.append(time.monotonic())

    def _prune(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._quote_timestamps and self._quote_timestamps[0] < cutoff:
            self._quote_timestamps.popleft()
        while self._fill_timestamps and self._fill_timestamps[0] < cutoff:
            self._fill_timestamps.popleft()

    def evaluate(self) -> QTRSnapshot:
        """Compute current QTR and update warn/critical state."""
        now = time.monotonic()
        self._prune(now)

        quotes = len(self._quote_timestamps)
        fills = len(self._fill_timestamps)

        if fills > 0:
            ratio = Decimal(str(quotes)) / Decimal(str(fills))
        elif quotes > 0:
            ratio = Decimal("Infinity")
        else:
            ratio = Decimal("0")

        # Update state transitions.
        was_warn = self._warn_active
        was_critical = self._critical_active

        self._warn_active = float(ratio) >= self._warn_threshold if ratio != Decimal("Infinity") else quotes > 0
        self._critical_active = float(ratio) >= self._critical_threshold if ratio != Decimal("Infinity") else quotes > 0

        if self._critical_active and not was_critical:
            logger.warning(
                "QTR CRITICAL: ratio=%.1f (quotes=%d fills=%d in %.0fs window) — "
                "consider widening spreads or reducing reprice frequency",
                float(ratio) if ratio != Decimal("Infinity") else float("inf"),
                quotes, fills, self._window_s,
            )
        elif self._warn_active and not was_warn:
            logger.info(
                "QTR WARNING: ratio=%.1f (quotes=%d fills=%d in %.0fs window)",
                float(ratio) if ratio != Decimal("Infinity") else float("inf"),
                quotes, fills, self._window_s,
            )
        elif was_warn and not self._warn_active:
            logger.info("QTR normalized: ratio=%.1f", float(ratio) if ratio != Decimal("Infinity") else 0.0)

        return QTRSnapshot(
            quotes_in_window=quotes,
            fills_in_window=fills,
            ratio=ratio,
            window_s=self._window_s,
            warn_active=self._warn_active,
            critical_active=self._critical_active,
        )

    @property
    def is_critical(self) -> bool:
        return self._critical_active

    @property
    def is_warn(self) -> bool:
        return self._warn_active
