"""
Fill Quality Tracker

Tracks per-level fill metrics and delayed markouts across multiple
horizons.  Also exposes maker/taker and side-conditioned diagnostics used
for adverse-selection monitoring.
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
_MAX_SEGMENT_SAMPLES = 2000

_MARKOUT_HORIZONS_S: dict[str, float] = {
    "250ms": 0.25,
    "1s": 1.0,
    "5s": 5.0,
    "30s": 30.0,
    "2m": 120.0,
}


@dataclass
class LevelFillQuality:
    """Aggregated fill quality metrics for one (side, level) slot."""

    avg_edge_bps: Decimal = Decimal("0")
    avg_markout_1s: Decimal = Decimal("0")
    avg_markout_5s: Decimal = Decimal("0")
    adverse_fill_pct: Decimal = Decimal("0")
    sample_count: int = 0


@dataclass(frozen=True)
class SegmentedMarkoutSummary:
    """Average markouts for one horizon conditioned by side/maker-taker."""

    horizon: str
    avg_all: Decimal
    avg_maker: Decimal
    avg_taker: Decimal
    avg_buy: Decimal
    avg_sell: Decimal
    avg_buy_maker: Decimal
    avg_buy_taker: Decimal
    avg_sell_maker: Decimal
    avg_sell_taker: Decimal
    count_all: int
    count_maker: int
    count_taker: int


class FillQualityTracker:
    """Track fill quality metrics with deferred markout callbacks."""

    def __init__(self, orderbook_mgr) -> None:
        self._ob = orderbook_mgr
        self._edge_samples: dict[tuple[str, int], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES_PER_LEVEL)
        )
        self._markout_samples_by_horizon: dict[str, dict[tuple[str, int], deque]] = {
            horizon: defaultdict(lambda: deque(maxlen=_MAX_SAMPLES_PER_LEVEL))
            for horizon in _MARKOUT_HORIZONS_S
        }
        # Backward-compatible aliases used by existing tests/metrics.
        self._markout_1s_samples = self._markout_samples_by_horizon["1s"]
        self._markout_5s_samples = self._markout_samples_by_horizon["5s"]

        self._segmented_markouts: dict[tuple[str, str, bool], deque] = defaultdict(
            lambda: deque(maxlen=_MAX_SEGMENT_SAMPLES)
        )
        self._quote_lifetime_ms: deque = deque(maxlen=_MAX_SEGMENT_SAMPLES)

        # Callback for automatic offset widening on poor markout.
        self._offset_widen_callback: Optional[Callable] = None
        self._min_acceptable_markout_bps: Decimal = Decimal("-2")

    def set_offset_widen_callback(self, cb) -> None:
        """Register a callback(key, reason) for auto-widening."""
        self._offset_widen_callback = cb

    def record_fill(
        self,
        *,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
        is_taker: bool = False,
        quote_lifetime_ms: Optional[Decimal] = None,
    ) -> None:
        """Record a fill and schedule markout callbacks."""
        side = self._normalize_side(side_name)
        mid = self._get_mid()
        if mid is not None and mid > 0:
            if side == "BUY":
                edge_bps = (mid - fill_price) / mid * Decimal("10000")
            else:
                edge_bps = (fill_price - mid) / mid * Decimal("10000")
            self._edge_samples[key].append(edge_bps)

        if quote_lifetime_ms is not None and quote_lifetime_ms >= 0:
            self._quote_lifetime_ms.append(quote_lifetime_ms)

        try:
            loop = asyncio.get_running_loop()
            for horizon, delay_s in _MARKOUT_HORIZONS_S.items():
                loop.call_later(
                    delay_s,
                    self._record_markout,
                    key,
                    fill_price,
                    side,
                    horizon,
                    bool(is_taker),
                )
        except RuntimeError:
            # No running loop (e.g. in tests) — skip async scheduling.
            pass

    def _record_markout(
        self,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
        horizon: str,
        is_taker: bool = False,
    ) -> None:
        """Compute markout vs current reference and store sample."""
        mid = self._get_mid()
        if mid is None or mid <= 0:
            return
        if fill_price <= 0:
            return

        side = self._normalize_side(side_name)
        if side == "BUY":
            markout_bps = (mid - fill_price) / fill_price * Decimal("10000")
        else:
            markout_bps = (fill_price - mid) / fill_price * Decimal("10000")

        per_horizon = self._markout_samples_by_horizon.get(horizon)
        if per_horizon is not None:
            per_horizon[key].append(markout_bps)
        self._segmented_markouts[(horizon, side, bool(is_taker))].append(markout_bps)

        if horizon == "5s":
            avg_5s = self._avg(self._markout_5s_samples[key])
            if avg_5s < self._min_acceptable_markout_bps and self._offset_widen_callback is not None:
                logger.warning(
                    "Adverse 5s markout for %s: avg=%.2fbps < threshold=%sbps — widening offset",
                    key,
                    avg_5s,
                    self._min_acceptable_markout_bps,
                )
                try:
                    self._offset_widen_callback(key, "adverse_markout")
                except Exception as exc:  # pragma: no cover - defensive logging path
                    logger.error("Offset widen callback error: %s", exc)

    def set_min_acceptable_markout_bps(self, value: Decimal) -> None:
        """Set the minimum acceptable 5-second markout before auto-widening."""
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
        keys = set(self._edge_samples.keys())
        for horizon_samples in self._markout_samples_by_horizon.values():
            keys |= set(horizon_samples.keys())
        return {k: self.level_quality(k) for k in keys}

    def record_markout_sync(
        self,
        key: tuple[str, int],
        fill_price: Decimal,
        side_name: str,
        horizon: str,
        is_taker: bool = False,
    ) -> None:
        """Synchronous markout recording (for testing without event loop)."""
        self._record_markout(key, fill_price, side_name, horizon, is_taker)

    def quote_lifetime_avg_ms(self) -> Decimal:
        return self._avg(self._quote_lifetime_ms)

    def quote_lifetime_count(self) -> int:
        return len(self._quote_lifetime_ms)

    def segmented_markout_summary(self, horizon: str) -> SegmentedMarkoutSummary:
        """Return maker/taker and side-conditioned markout summary for one horizon."""
        horizon_key = str(horizon)

        def _vals(side: Optional[str], is_taker: Optional[bool]) -> list[Decimal]:
            out: list[Decimal] = []
            sides = ["BUY", "SELL"] if side is None else [side]
            roles = [False, True] if is_taker is None else [is_taker]
            for side_name in sides:
                for taker_flag in roles:
                    out.extend(
                        self._segmented_markouts.get(
                            (horizon_key, side_name, taker_flag),
                            (),
                        )
                    )
            return out

        all_vals = _vals(None, None)
        maker_vals = _vals(None, False)
        taker_vals = _vals(None, True)
        buy_vals = _vals("BUY", None)
        sell_vals = _vals("SELL", None)
        buy_maker_vals = _vals("BUY", False)
        buy_taker_vals = _vals("BUY", True)
        sell_maker_vals = _vals("SELL", False)
        sell_taker_vals = _vals("SELL", True)

        return SegmentedMarkoutSummary(
            horizon=horizon_key,
            avg_all=self._avg(all_vals),
            avg_maker=self._avg(maker_vals),
            avg_taker=self._avg(taker_vals),
            avg_buy=self._avg(buy_vals),
            avg_sell=self._avg(sell_vals),
            avg_buy_maker=self._avg(buy_maker_vals),
            avg_buy_taker=self._avg(buy_taker_vals),
            avg_sell_maker=self._avg(sell_maker_vals),
            avg_sell_taker=self._avg(sell_taker_vals),
            count_all=len(all_vals),
            count_maker=len(maker_vals),
            count_taker=len(taker_vals),
        )

    @staticmethod
    def markout_horizons() -> tuple[str, ...]:
        return tuple(_MARKOUT_HORIZONS_S.keys())

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
    def _avg(samples) -> Decimal:
        if not samples:
            return Decimal("0")
        return Decimal(str(sum(samples) / Decimal(str(len(samples)))))

    @staticmethod
    def _normalize_side(side_name: str) -> str:
        side_upper = str(side_name).upper()
        if "BUY" in side_upper:
            return "BUY"
        if "SELL" in side_upper:
            return "SELL"
        return side_upper
