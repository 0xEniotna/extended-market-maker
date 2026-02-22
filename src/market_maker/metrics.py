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
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from .account_stream import AccountStreamManager
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

_STATUS_LOG_INTERVAL_S = 60.0  # log a summary every N seconds


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
                    "fees=%s | fails=%d cb=%s | uptime=%.0fs",
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
                    snap.uptime_s,
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
