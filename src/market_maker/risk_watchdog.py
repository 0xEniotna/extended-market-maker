"""
Risk Watchdog

Consolidates the three safety-loop tasks previously spread across
MarketMakerStrategy:

- **Drawdown stop**: monitors session P&L and triggers shutdown if
  drawdown exceeds a configured threshold.
- **Margin guard**: monitors collateral / margin utilisation and halts
  quoting (or triggers shutdown) when constraints are breached.
- **Circuit breaker**: monitors order-placement failure rates, sweeps
  pending-cancel state, and detects zombie orders.

Each task is exposed as a coroutine that the strategy runner creates
as an ``asyncio.Task``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_DRAWDOWN_CHECK_INTERVAL_S = 1.0
_MARGIN_GUARD_CHECK_INTERVAL_S = 1.0


class RiskWatchdog:
    """Encapsulates drawdown, margin-guard, and circuit-breaker loops."""

    def __init__(
        self,
        *,
        settings: object,
        risk_mgr: object,
        orders: object,
        halt_mgr: object,
        metrics: object,
        journal: object,
        post_only: object,
        drawdown_stop: object,
        request_shutdown_fn: object,
    ) -> None:
        self._settings = settings
        self._risk = risk_mgr
        self._orders = orders
        self._halt_mgr = halt_mgr
        self._metrics = metrics
        self._journal = journal
        self._post_only = post_only
        self._drawdown_stop = drawdown_stop
        self._request_shutdown = request_shutdown_fn
        self._circuit_open = False

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    # ------------------------------------------------------------------
    # Drawdown stop
    # ------------------------------------------------------------------

    def evaluate_drawdown_stop(self) -> bool:
        """Return True if drawdown threshold was triggered (initiates shutdown)."""
        state = self._drawdown_stop.evaluate(self._risk.get_session_pnl())  # type: ignore[attr-defined]
        if not state.triggered:
            return False

        action = "cancel_all_flatten_terminate"
        logger.error(
            "DRAWDOWN STOP TRIGGERED: market=%s current_pnl=%s peak_pnl=%s drawdown=%s "
            "threshold=%s action=%s",
            self._settings.market_name,  # type: ignore[attr-defined]
            state.current_pnl,
            state.peak_pnl,
            state.drawdown,
            state.threshold_usd,
            action,
        )
        self._journal.record_drawdown_stop(  # type: ignore[attr-defined]
            current_pnl=state.current_pnl,
            peak_pnl=state.peak_pnl,
            drawdown=state.drawdown,
            threshold_usd=state.threshold_usd,
            action=action,
        )
        self._request_shutdown("drawdown_stop")  # type: ignore[operator]
        return True

    async def drawdown_watchdog_task(self, shutdown_event: asyncio.Event) -> None:
        """Async loop monitoring drawdown state."""
        while not shutdown_event.is_set():
            try:
                if self.evaluate_drawdown_stop():
                    return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Drawdown watchdog error", exc_info=True)
                self._journal.record_error(  # type: ignore[attr-defined]
                    component="drawdown_watchdog",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=self._journal.make_stack_trace_hash(exc),  # type: ignore[attr-defined]
                )
            await asyncio.sleep(_DRAWDOWN_CHECK_INTERVAL_S)

    # ------------------------------------------------------------------
    # Margin guard
    # ------------------------------------------------------------------

    def margin_guard_breach(self) -> tuple[bool, list[str], dict[str, Optional[Decimal]]]:
        """Check all margin constraints and return (breached, reasons, snapshot)."""
        snapshot = self._risk.margin_snapshot()  # type: ignore[attr-defined]
        reasons: list[str] = []

        available = snapshot.get("available_for_trade")
        equity = snapshot.get("equity")
        initial_margin = snapshot.get("initial_margin")
        available_ratio = snapshot.get("available_ratio")
        margin_utilization = snapshot.get("margin_utilization")
        liq_distance_bps = snapshot.get("liq_distance_bps")
        current_position = self._risk.get_current_position()  # type: ignore[attr-defined]

        if available is None:
            reasons.append("available_for_trade_missing")
        elif available < self._settings.min_available_balance_for_trading:  # type: ignore[attr-defined]
            reasons.append("available_for_trade")

        if self._settings.min_available_balance_ratio > 0:  # type: ignore[attr-defined]
            if available_ratio is None or equity is None or equity <= 0:
                reasons.append("available_ratio_missing")
            elif available_ratio < self._settings.min_available_balance_ratio:  # type: ignore[attr-defined]
                reasons.append("available_ratio")

        if self._settings.max_margin_utilization > 0:  # type: ignore[attr-defined]
            if margin_utilization is None or initial_margin is None or equity is None or equity <= 0:
                reasons.append("margin_utilization_missing")
            elif margin_utilization > self._settings.max_margin_utilization:  # type: ignore[attr-defined]
                reasons.append("margin_utilization")

        if current_position != 0 and self._settings.min_liq_distance_bps > 0:  # type: ignore[attr-defined]
            if liq_distance_bps is None:
                reasons.append("liq_distance_missing")
            elif liq_distance_bps < self._settings.min_liq_distance_bps:  # type: ignore[attr-defined]
                reasons.append("liq_distance")

        return bool(reasons), reasons, snapshot

    async def margin_guard_task(self, shutdown_event: asyncio.Event) -> None:
        """Async loop monitoring margin status."""
        while not shutdown_event.is_set():
            try:
                if not self._settings.margin_guard_enabled:  # type: ignore[attr-defined]
                    await asyncio.sleep(_MARGIN_GUARD_CHECK_INTERVAL_S)
                    continue

                breached, reasons, snapshot = self.margin_guard_breach()
                now = time.monotonic()
                if breached:
                    self._metrics.set_margin_guard_breached(True)  # type: ignore[attr-defined]
                    self._halt_mgr.set_halt("margin_guard")  # type: ignore[attr-defined]
                    if self._halt_mgr.margin_breach_since is None:  # type: ignore[attr-defined]
                        self._halt_mgr.margin_breach_since = now  # type: ignore[attr-defined]
                        self._journal.record_exchange_event(  # type: ignore[attr-defined]
                            event_type="margin_guard_breach",
                            details={
                                "reasons": reasons,
                                "snapshot": snapshot,
                            },
                        )
                        if self._orders.active_order_count() > 0:  # type: ignore[attr-defined]
                            await self._orders.cancel_all_orders()  # type: ignore[attr-defined]
                    breach_elapsed = now - self._halt_mgr.margin_breach_since  # type: ignore[attr-defined]
                    if (
                        self._settings.margin_guard_shutdown_breach_s > 0  # type: ignore[attr-defined]
                        and breach_elapsed >= self._settings.margin_guard_shutdown_breach_s  # type: ignore[attr-defined]
                    ):
                        self._journal.record_exchange_event(  # type: ignore[attr-defined]
                            event_type="margin_guard_shutdown",
                            details={
                                "reasons": reasons,
                                "breach_elapsed_s": breach_elapsed,
                                "snapshot": snapshot,
                            },
                        )
                        self._request_shutdown("margin_guard")  # type: ignore[operator]
                        return
                else:
                    if self._halt_mgr.margin_breach_since is not None:  # type: ignore[attr-defined]
                        self._journal.record_exchange_event(  # type: ignore[attr-defined]
                            event_type="margin_guard_cleared",
                            details={"snapshot": snapshot},
                        )
                    self._halt_mgr.margin_breach_since = None  # type: ignore[attr-defined]
                    self._metrics.set_margin_guard_breached(False)  # type: ignore[attr-defined]
                    self._halt_mgr.clear_halt("margin_guard")  # type: ignore[attr-defined]
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Margin guard task error: %s", exc, exc_info=True)
                self._journal.record_error(  # type: ignore[attr-defined]
                    component="margin_guard",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=self._journal.make_stack_trace_hash(exc),  # type: ignore[attr-defined]
                )
            await asyncio.sleep(_MARGIN_GUARD_CHECK_INTERVAL_S)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    async def circuit_breaker_task(self, shutdown_event: asyncio.Event) -> None:
        """Monitor consecutive failures, sweep pending cancels, and detect zombies."""
        max_failures = self._settings.circuit_breaker_max_failures  # type: ignore[attr-defined]
        cooldown = self._settings.circuit_breaker_cooldown_s  # type: ignore[attr-defined]
        failure_window_s = self._settings.failure_window_s  # type: ignore[attr-defined]
        failure_rate_trip = float(self._settings.failure_rate_trip)  # type: ignore[attr-defined]
        min_attempts = self._settings.min_attempts_for_breaker  # type: ignore[attr-defined]

        while not shutdown_event.is_set():
            try:
                self._orders.sweep_pending_cancels()  # type: ignore[attr-defined]

                zombie_threshold_s = self._settings.max_order_age_s * 2  # type: ignore[attr-defined]
                if zombie_threshold_s > 0:
                    zombies = self._orders.find_zombie_orders(zombie_threshold_s)  # type: ignore[attr-defined]
                    for zombie in zombies:
                        logger.warning(
                            "Zombie order detected (no stream update in %.0fs): "
                            "ext_id=%s exchange_id=%s side=%s level=%d — cancelling",
                            time.monotonic() - zombie.placed_at,
                            zombie.external_id,
                            zombie.exchange_order_id,
                            zombie.side,
                            zombie.level,
                        )
                        await self._orders.cancel_order(zombie.external_id)  # type: ignore[attr-defined]

                placement_stats_60s = self._orders.failure_window_stats(60.0)  # type: ignore[attr-defined]
                total_placements_60s = int(placement_stats_60s["attempts"])
                self._post_only.update_pof_offset_boost(total_placements_60s)  # type: ignore[attr-defined]

                window_stats = self._orders.failure_window_stats(failure_window_s)  # type: ignore[attr-defined]
                attempts = int(window_stats["attempts"])
                failure_rate = float(window_stats["failure_rate"])
                window_trip = (
                    attempts >= min_attempts and failure_rate >= failure_rate_trip
                )
                if (
                    not self._circuit_open
                    and (
                        self._orders.consecutive_failures >= max_failures  # type: ignore[attr-defined]
                        or window_trip
                    )
                ):
                    self._circuit_open = True
                    self._metrics.circuit_open = True  # type: ignore[attr-defined]
                    logger.warning(
                        "CIRCUIT BREAKER OPEN: consecutive_failures=%d attempts=%d "
                        "failure_rate=%.2f — pausing for %.0fs",
                        self._orders.consecutive_failures,  # type: ignore[attr-defined]
                        attempts,
                        failure_rate,
                        cooldown,
                    )
                    await asyncio.sleep(cooldown)
                    self._orders.reset_failure_tracking()  # type: ignore[attr-defined]
                    self._circuit_open = False
                    self._metrics.circuit_open = False  # type: ignore[attr-defined]
                    logger.info("Circuit breaker reset — resuming")
                else:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Circuit breaker task error: %s", exc)
                self._journal.record_error(  # type: ignore[attr-defined]
                    component="circuit_breaker",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=self._journal.make_stack_trace_hash(exc),  # type: ignore[attr-defined]
                )
                await asyncio.sleep(1.0)
