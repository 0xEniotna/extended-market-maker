"""
Monitoring Tasks

Background coroutines for institutional monitoring, KPI watchdog,
balance/position refresh, and config rollback.  Extracted from
``MarketMakerStrategy`` to keep the strategy focused on quoting logic.

All tasks follow the same pattern: loop on shutdown_event, do work,
sleep for an interval.  They accept a ``StrategyRef`` protocol
(satisfied by MarketMakerStrategy) so they can access shared state.
"""
from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_BALANCE_REFRESH_INTERVAL_S = 10.0
_POSITION_REFRESH_INTERVAL_S = 30.0
_KPI_WATCHDOG_INTERVAL_S = 5.0
_QTR_EVALUATE_INTERVAL_S = 30.0
_LATENCY_SLA_INTERVAL_S = 10.0
_CONFIG_ROLLBACK_INTERVAL_S = 60.0
_PNL_ATTRIBUTION_LOG_INTERVAL_S = 60.0


class StrategyRef(Protocol):
    """Minimal protocol that monitoring tasks need from the strategy."""

    _settings: Any
    _shutdown_event: asyncio.Event
    _ob: Any
    _risk: Any
    _orders: Any
    _account_stream: Any
    _journal: Any
    _metrics: Any
    _pnl_attribution: Any
    _qtr: Any
    _latency_monitor: Any
    _config_rollback: Any
    _fill_quality: Any
    _halt_mgr: Any
    _runtime_mode: str
    _last_taker_leakage_warn_at: float
    _funding_mgr: Any

    def _sync_quote_halt_state(self) -> None: ...
    def _is_normal_quoting_mode(self) -> bool: ...
    def _set_quote_halt(self, reason: str) -> None: ...
    def _clear_quote_halt(self, reason: str) -> None: ...


async def balance_refresh_task(s: Any) -> None:
    """Periodically refresh available-for-trade collateral headroom."""
    while not s._shutdown_event.is_set():
        try:
            await s._risk.refresh_balance()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("Balance refresh failed", exc_info=True)
        await asyncio.sleep(_BALANCE_REFRESH_INTERVAL_S)


async def position_refresh_task(s: Any) -> None:
    """Periodically refresh the cached position from the exchange."""
    while not s._shutdown_event.is_set():
        try:
            await s._risk.refresh_position()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Position refresh error: %s", exc)
            s._journal.record_error(
                component="position_refresh",
                exception_type=type(exc).__name__,
                message=str(exc),
                stack_trace_hash=s._journal.make_stack_trace_hash(exc),
            )
        await asyncio.sleep(_POSITION_REFRESH_INTERVAL_S)


async def kpi_watchdog_task(s: Any) -> None:
    """Monitor taker leakage and sync halt state."""
    while not s._shutdown_event.is_set():
        try:
            s._sync_quote_halt_state()

            taker_count = (
                int(s._account_stream.taker_fill_count_1m())
                if hasattr(s._account_stream, "taker_fill_count_1m")
                else 0
            )
            taker_ratio = (
                Decimal(str(s._account_stream.taker_fill_notional_ratio_1m()))
                if hasattr(s._account_stream, "taker_fill_notional_ratio_1m")
                else Decimal("0")
            )

            now = time.monotonic()
            if (
                taker_count > 0
                and s._is_normal_quoting_mode()
                and (now - s._last_taker_leakage_warn_at) >= 30.0
            ):
                s._last_taker_leakage_warn_at = now
                logger.warning(
                    "Taker leakage detected for %s: count_1m=%d notional_ratio_1m=%.2f%%",
                    s._settings.market_name,
                    taker_count,
                    float(taker_ratio * Decimal("100")),
                )
                s._journal.record_exchange_event(
                    event_type="taker_leakage_warning",
                    details={
                        "count_1m": taker_count,
                        "notional_ratio_1m": taker_ratio,
                        "runtime_mode": s._runtime_mode,
                    },
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("KPI watchdog task error: %s", exc, exc_info=True)
        await asyncio.sleep(_KPI_WATCHDOG_INTERVAL_S)


async def pnl_attribution_task(s: Any) -> None:
    """Periodically log P&L attribution breakdown to the journal."""
    while not s._shutdown_event.is_set():
        try:
            bid = s._ob.best_bid()
            ask = s._ob.best_ask()
            mid = None
            if bid is not None and ask is not None and bid.price > 0:
                mid = (bid.price + ask.price) / 2
            snap = s._pnl_attribution.snapshot(current_mid=mid)
            if snap.fill_count > 0:
                s._journal.record_exchange_event(
                    event_type="pnl_attribution",
                    details={
                        "spread_capture_usd": str(snap.spread_capture_usd),
                        "inventory_pnl_usd": str(snap.inventory_pnl_usd),
                        "fee_pnl_usd": str(snap.fee_pnl_usd),
                        "funding_pnl_usd": str(snap.funding_pnl_usd),
                        "total_usd": str(snap.total_usd),
                        "fill_count": snap.fill_count,
                        "buy_fills": snap.buy_fill_count,
                        "sell_fills": snap.sell_fill_count,
                        "maker_fills": snap.maker_fill_count,
                        "taker_fills": snap.taker_fill_count,
                        "avg_spread_capture_bps": str(snap.avg_spread_capture_bps),
                        "total_volume_usd": str(snap.total_volume_usd),
                    },
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("P&L attribution task error: %s", exc)
        await asyncio.sleep(_PNL_ATTRIBUTION_LOG_INTERVAL_S)


async def qtr_monitor_task(s: Any) -> None:
    """Periodically evaluate quote-to-trade ratio."""
    while not s._shutdown_event.is_set():
        try:
            snap = s._qtr.evaluate()
            if snap.quotes_in_window > 0:
                s._journal.record_exchange_event(
                    event_type="qtr_snapshot",
                    details={
                        "quotes": snap.quotes_in_window,
                        "fills": snap.fills_in_window,
                        "ratio": str(snap.ratio),
                        "warn": snap.warn_active,
                        "critical": snap.critical_active,
                    },
                )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("QTR monitor task error: %s", exc)
        await asyncio.sleep(_QTR_EVALUATE_INTERVAL_S)


async def latency_sla_task(s: Any) -> None:
    """Periodically evaluate venue latency SLA and adjust quoting."""
    while not s._shutdown_event.is_set():
        try:
            for sample in getattr(s._orders, "_latency_samples", []):
                s._latency_monitor.record_latency(sample)

            snap = s._latency_monitor.evaluate()
            if snap.halt_quoting:
                s._set_quote_halt("latency_sla")
            else:
                s._clear_quote_halt("latency_sla")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Latency SLA task error: %s", exc)
        await asyncio.sleep(_LATENCY_SLA_INTERVAL_S)


async def config_rollback_task(s: Any) -> None:
    """Periodically evaluate whether a recent config change degraded performance."""
    while not s._shutdown_event.is_set():
        try:
            if s._config_rollback.has_pending_evaluation:
                fq = getattr(s, "_fill_quality", None)
                avg_markout_1s = Decimal("0")
                if fq is not None:
                    summary = fq.segmented_markout_summary("1s")
                    avg_markout_1s = summary.avg_all

                pnl_snap = s._pnl_attribution.snapshot()
                decision = s._config_rollback.evaluate(
                    avg_markout_1s_bps=avg_markout_1s,
                    fill_count=pnl_snap.fill_count,
                    session_pnl_usd=s._risk.get_session_pnl(),
                    avg_spread_capture_bps=pnl_snap.avg_spread_capture_bps,
                )
                if decision.should_rollback:
                    good_config = s._config_rollback.last_known_good_config
                    s._journal.record_exchange_event(
                        event_type="config_rollback",
                        details={
                            "reason": decision.reason,
                            "degraded_metrics": {
                                k: str(v) for k, v in decision.degraded_metrics.items()
                            },
                        },
                    )
                    logger.warning(
                        "Config rollback triggered: %s (good config available: %s)",
                        decision.reason,
                        good_config is not None,
                    )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Config rollback task error: %s", exc)
        await asyncio.sleep(_CONFIG_ROLLBACK_INTERVAL_S)
