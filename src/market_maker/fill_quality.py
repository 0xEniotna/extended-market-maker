"""
Fill Quality Tracker

Tracks per-level fill metrics: edge at fill time, markout at +1s and +5s,
and adverse fill classification.  Exposes running averages consumed by
MetricsCollector for status-line logging and monitoring.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Maximum samples per level to keep in memory.
_MAX_SAMPLES_PER_LEVEL = 200


@dataclass
class LevelFillQuality:
    """Aggregated fill quality metrics for one (side, level) slot."""

    avg_edge_bps: Decimal = Decimal("0")
    avg_markout_1s: Decimal = Decimal("0")
    avg_markout_5s: Decimal = Decimal("0")
    adverse_fill_pct: Decimal = Decimal("0")
    sample_count: int = 0


class FillQualityTracker:
    """Track fill quality metrics with deferred markout callbacks."""

    def __init__(self, orderbook_mgr) -> None:
        self._ob = orderbook_mgr
        # Per-level running samples: key → deque of (edge_bps, markout_1s, markout_5s)
        self._edge_samples: dict[tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES_PER_LEVEL)
        )
        self._markout_1s_samples: dict[tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES_PER_LEVEL)
        )
        self._markout_5s_samples: dict[tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES_PER_LEVEL)
        )
        # Callback for automatic offset widening on poor markout.
        self._offset_widen_callback: Optional[Callable] = None

    def set_offset_widen_callback(self, cb) -> None:
        """Register a callback(key, reason) for auto-widening."""
        self._offset_widen_callback = cb

    def record_fill(
        self,
        *,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
    ) -> None:
        """Record a fill and schedule markout callbacks.

        ``key`` is (side_name, level).
        """
        mid = self._get_mid()
        if mid is not None and mid > 0:
            if side_name.upper() == "BUY":
                edge_bps = (mid - fill_price) / mid * Decimal("10000")
            else:
                edge_bps = (fill_price - mid) / mid * Decimal("10000")
            self._edge_samples[key].append(edge_bps)

        # Schedule markout callbacks
        try:
            loop = asyncio.get_running_loop()
            loop.call_later(
                1.0,
                self._record_markout,
                key,
                fill_price,
                side_name,
                "1s",
            )
            loop.call_later(
                5.0,
                self._record_markout,
                key,
                fill_price,
                side_name,
                "5s",
            )
        except RuntimeError:
            # No running loop (e.g. in tests) — skip markout scheduling.
            pass

    def _record_markout(
        self,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
        horizon: str,
    ) -> None:
        """Compute markout vs current mid and store sample."""
        mid = self._get_mid()
        if mid is None or mid <= 0:
            return

        if side_name.upper() == "BUY":
            markout_bps = (mid - fill_price) / fill_price * Decimal("10000")
        else:
            markout_bps = (fill_price - mid) / fill_price * Decimal("10000")

        if horizon == "1s":
            self._markout_1s_samples[key].append(markout_bps)
        elif horizon == "5s":
            self._markout_5s_samples[key].append(markout_bps)
            # Check for adverse markout → auto-widen
            min_markout = getattr(self, "_min_acceptable_markout_bps", Decimal("-2"))
            avg_5s = self._avg(self._markout_5s_samples[key])
            if avg_5s < min_markout and self._offset_widen_callback is not None:
                logger.warning(
                    "Adverse 5s markout for %s: avg=%.2fbps < threshold=%sbps — widening offset",
                    key,
                    avg_5s,
                    min_markout,
                )
                try:
                    self._offset_widen_callback(key, "adverse_markout")
                except Exception as exc:
                    logger.error("Offset widen callback error: %s", exc)

    def set_min_acceptable_markout_bps(self, value: Decimal) -> None:
        """Set the minimum acceptable 5s markout before auto-widening."""
        self._min_acceptable_markout_bps = value

    def level_quality(self, key: tuple[str, int]) -> LevelFillQuality:
        """Return aggregated fill quality for a single level."""
        edges = self._edge_samples.get(key, deque())
        m1s = self._markout_1s_samples.get(key, deque())
        m5s = self._markout_5s_samples.get(key, deque())

        adverse_count = sum(1 for v in m5s if v < 0)
        total_m5s = len(m5s)

        return LevelFillQuality(
            avg_edge_bps=self._avg(edges),
            avg_markout_1s=self._avg(m1s),
            avg_markout_5s=self._avg(m5s),
            adverse_fill_pct=(
                Decimal(str(adverse_count)) / Decimal(str(total_m5s)) * Decimal("100")
                if total_m5s > 0
                else Decimal("0")
            ),
            sample_count=len(edges),
        )

    def all_level_qualities(self) -> dict[tuple[str, int], LevelFillQuality]:
        """Return quality for all levels that have any data."""
        keys = set(self._edge_samples.keys()) | set(self._markout_1s_samples.keys()) | set(self._markout_5s_samples.keys())
        return {k: self.level_quality(k) for k in keys}

    def record_markout_sync(
        self,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
        horizon: str,
    ) -> None:
        """Synchronous markout recording (for testing without event loop)."""
        self._record_markout(key, fill_price, side_name, horizon)

    def _get_mid(self) -> Optional[Decimal]:
        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        if bid is None or ask is None:
            return None
        bp = getattr(bid, "price", None)
        ap = getattr(ask, "price", None)
        if bp is None or ap is None or bp <= 0 or ap <= 0:
            return None
        return Decimal(str((bp + ap) / 2))

    @staticmethod
    def _avg(samples: deque) -> Decimal:
        if not samples:
            return Decimal("0")
        return Decimal(str(sum(samples) / Decimal(str(len(samples)))))
