"""
Market Maker Metrics & Observability

Periodically logs a compact status summary and exposes counters that can
be consumed by external monitoring (Prometheus, Telegram, etc.).

Designed to be lightweight and self-contained.  The heavy lifting (fill
counting, fee tracking) is done by ``AccountStreamManager.metrics``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional

from .account_stream import AccountStreamManager
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

_STATUS_LOG_INTERVAL_S = 60.0  # log a summary every N seconds


@dataclass
class LevelFillQualitySnapshot:
    """Per-level fill quality metrics."""

    avg_edge_bps: Decimal = Decimal("0")
    avg_markout_1s: Decimal = Decimal("0")
    avg_markout_5s: Decimal = Decimal("0")
    adverse_fill_pct: Decimal = Decimal("0")
    sample_count: int = 0


@dataclass
class StrategySnapshot:
    """Point-in-time snapshot of the strategy state for monitoring."""

    timestamp: float = 0.0
    position: Decimal = Decimal("0")
    best_bid: Optional[Decimal] = None
    best_ask: Optional[Decimal] = None
    spread_bps: Optional[Decimal] = None
    active_orders: int = 0
    total_fills: int = 0
    total_cancellations: int = 0
    total_rejections: int = 0
    post_only_failures: int = 0
    total_fees: Decimal = Decimal("0")
    consecutive_failures: int = 0
    circuit_open: bool = False
    uptime_s: float = 0.0
    avg_placement_latency_ms: float = 0.0
    pof_offset_boost_bps: Decimal = Decimal("0")
    level_fill_quality: Dict[str, LevelFillQualitySnapshot] = field(
        default_factory=dict
    )


class MetricsCollector:
    """
    Collects metrics from all market-maker components and produces
    periodic status summaries.
    """

    def __init__(
        self,
        orderbook_mgr: OrderbookManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
        account_stream: AccountStreamManager,
        journal: Optional[TradeJournal] = None,
    ) -> None:
        self._ob = orderbook_mgr
        self._orders = order_mgr
        self._risk = risk_mgr
        self._stream = account_stream
        self._journal = journal
        self._start_ts = time.monotonic()
        self._task: Optional[asyncio.Task] = None

        # External flag set by strategy
        self.circuit_open: bool = False

        # Optional fill quality tracker (set by strategy after construction)
        self._fill_quality = None

        # Optional post_only safety (for POF offset boost)
        self._post_only = None

    def set_fill_quality_tracker(self, tracker) -> None:
        self._fill_quality = tracker

    def set_post_only_safety(self, post_only) -> None:
        self._post_only = post_only

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._log_loop(), name="mm-metrics")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> StrategySnapshot:
        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        m = self._stream.metrics

        level_quality: Dict[str, LevelFillQualitySnapshot] = {}
        if self._fill_quality is not None:
            for key, fq in self._fill_quality.all_level_qualities().items():
                str_key = f"{key[0]}_L{key[1]}"
                level_quality[str_key] = LevelFillQualitySnapshot(
                    avg_edge_bps=fq.avg_edge_bps,
                    avg_markout_1s=fq.avg_markout_1s,
                    avg_markout_5s=fq.avg_markout_5s,
                    adverse_fill_pct=fq.adverse_fill_pct,
                    sample_count=fq.sample_count,
                )

        pof_boost = Decimal("0")
        if self._post_only is not None:
            pof_boost = self._post_only.pof_offset_boost_bps

        return StrategySnapshot(
            timestamp=time.time(),
            position=self._risk.get_current_position(),
            best_bid=bid.price if bid else None,
            best_ask=ask.price if ask else None,
            spread_bps=self._ob.spread_bps(),
            active_orders=self._orders.active_order_count(),
            total_fills=m.fills,
            total_cancellations=m.cancellations,
            total_rejections=m.rejections,
            post_only_failures=m.post_only_failures,
            total_fees=m.total_fees,
            consecutive_failures=self._orders.consecutive_failures,
            circuit_open=self.circuit_open,
            uptime_s=time.monotonic() - self._start_ts,
            avg_placement_latency_ms=self._orders.avg_placement_latency_ms(),
            pof_offset_boost_bps=pof_boost,
            level_fill_quality=level_quality,
        )

    # ------------------------------------------------------------------
    # Periodic logger
    # ------------------------------------------------------------------

    async def _log_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_STATUS_LOG_INTERVAL_S)
                snap = self.snapshot()
                spread_str = (
                    f"{snap.spread_bps:.1f}bps" if snap.spread_bps is not None else "N/A"
                )
                logger.info(
                    "STATUS | pos=%s | bid=%s ask=%s spread=%s | "
                    "orders=%d | fills=%d cancel=%d reject=%d pof=%d | "
                    "fees=%s | fails=%d cb=%s | latency=%.0fms pof_boost=%sbps | uptime=%.0fs",
                    snap.position,
                    snap.best_bid or "N/A",
                    snap.best_ask or "N/A",
                    spread_str,
                    snap.active_orders,
                    snap.total_fills,
                    snap.total_cancellations,
                    snap.total_rejections,
                    snap.post_only_failures,
                    snap.total_fees,
                    snap.consecutive_failures,
                    "OPEN" if snap.circuit_open else "closed",
                    snap.avg_placement_latency_ms,
                    snap.pof_offset_boost_bps,
                    snap.uptime_s,
                )
                # Log per-level fill quality if available.
                for level_key, fq in snap.level_fill_quality.items():
                    if fq.sample_count > 0:
                        logger.info(
                            "  FILL_QUALITY %s | edge=%.1fbps markout_1s=%.1fbps "
                            "markout_5s=%.1fbps adverse=%.0f%% samples=%d",
                            level_key,
                            fq.avg_edge_bps,
                            fq.avg_markout_1s,
                            fq.avg_markout_5s,
                            fq.adverse_fill_pct,
                            fq.sample_count,
                        )
                if self._journal is not None:
                    self._journal.record_snapshot(
                        position=snap.position,
                        best_bid=snap.best_bid,
                        best_ask=snap.best_ask,
                        spread_bps=snap.spread_bps,
                        active_orders=snap.active_orders,
                        total_fills=snap.total_fills,
                        total_fees=snap.total_fees,
                        circuit_open=snap.circuit_open,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Metrics log error: %s", exc)
