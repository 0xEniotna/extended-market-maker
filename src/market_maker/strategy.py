"""
Market Maker Strategy

Main orchestration for the market making bot.  Spawns one async task per
(level, side) pair.  Each task waits for a price change on the relevant
side of the book, computes the target price, and reprices its order when
the current order drifts outside a tolerance band.

Features:
- Inventory-skewed pricing (Avellaneda-Stoikov style)
- Real-time fill/order tracking via account WebSocket stream
- Atomic cancel-and-replace via ``previous_order_id``
- Minimum spread check to avoid quoting inside the fee
- Circuit breaker on consecutive order-placement failures
- Staleness detection on orderbook data

Entry point: ``MarketMakerStrategy.run()``
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Any, Dict, Optional

from x10.perpetual.accounts import StarkPerpetualAccount
from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager, FillEvent
from .config import ENV_FILE, MarketMakerSettings, OffsetMode
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

# Refresh the exchange position every N seconds as a safety net.
# The account stream handles real-time updates; this is a fallback.
_POSITION_REFRESH_INTERVAL_S = 30.0

# Default circuit-breaker settings
_CIRCUIT_BREAKER_MAX_FAILURES = 5
_CIRCUIT_BREAKER_COOLDOWN_S = 30.0


class MarketMakerStrategy:
    """
    Market making strategy that quotes on multiple price levels per side.
    """

    def __init__(
        self,
        settings: MarketMakerSettings,
        trading_client: PerpetualTradingClient,
        orderbook_mgr: OrderbookManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
        account_stream: AccountStreamManager,
        metrics: MetricsCollector,
        journal: TradeJournal,
        tick_size: Decimal,
        base_order_size: Decimal,
        market_min_order_size: Decimal,
        min_order_size_step: Decimal,
    ) -> None:
        self._settings = settings
        self._client = trading_client
        self._ob = orderbook_mgr
        self._orders = order_mgr
        self._risk = risk_mgr
        self._account_stream = account_stream
        self._metrics = metrics
        self._journal = journal
        self._tick_size = tick_size
        self._base_order_size = base_order_size
        self._market_min_order_size = market_min_order_size
        self._min_order_size_step = min_order_size_step

        # Keyed by (side_name, level)
        self._level_ext_ids: Dict[tuple[str, int], Optional[str]] = {}
        self._level_order_created_at: Dict[tuple[str, int], Optional[float]] = {}
        # Reprice rate limiting: monotonic timestamp of last reprice per slot
        self._level_last_reprice_at: Dict[tuple[str, int], float] = {}
        # POF cooldown: monotonic timestamp until which a level should not re-quote
        self._level_pof_until: Dict[tuple[str, int], float] = {}
        self._level_pof_streak: Dict[tuple[str, int], int] = {}
        self._level_pof_last_ts: Dict[tuple[str, int], float] = {}
        self._level_dynamic_safety_ticks: Dict[tuple[str, int], int] = {}
        # Stale-book grace tracking by slot
        self._level_stale_since: Dict[tuple[str, int], Optional[float]] = {}
        self._level_imbalance_pause_until: Dict[tuple[str, int], float] = {}
        # Cancel reason tracking for journaling once cancellation is confirmed
        self._pending_cancel_reasons: Dict[str, str] = {}

        self._shutdown_event = asyncio.Event()

        # Circuit-breaker state
        self._circuit_open = False

    # ------------------------------------------------------------------
    # Class-level entry point
    # ------------------------------------------------------------------

    @classmethod
    async def run(cls) -> None:
        """Load config, initialise components, and run the strategy."""

        # 1. Load config
        settings = MarketMakerSettings()

        # 2. Setup logging
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )

        logger.info(
            "Market maker starting: market=%s env=%s levels=%d offset_mode=%s skew=%.2f max_age=%ss",
            settings.market_name,
            settings.environment.value,
            settings.num_price_levels,
            settings.offset_mode.value,
            settings.inventory_skew_factor,
            settings.max_order_age_s or "off",
        )
        if settings.offset_mode == OffsetMode.FIXED:
            logger.info(
                "  Fixed offset: %.4f%% per level",
                settings.price_offset_per_level_percent,
            )
        else:
            logger.info(
                "  Dynamic offset: spread_mult=%.2f min=%sbps max=%sbps",
                settings.spread_multiplier,
                settings.min_offset_bps,
                settings.max_offset_bps,
            )

        # 3. Kill switch
        if not settings.enabled:
            logger.warning("MM_ENABLED is false — exiting")
            return

        # 4. Validate credentials
        if not settings.is_configured:
            logger.error(
                "Missing credentials (MM_VAULT_ID, MM_STARK_PRIVATE_KEY, "
                "MM_STARK_PUBLIC_KEY, MM_API_KEY). Exiting."
            )
            return

        # 5. Initialise SDK trading client
        account = StarkPerpetualAccount(
            vault=int(settings.vault_id),
            private_key=settings.stark_private_key,
            public_key=settings.stark_public_key,
            api_key=settings.api_key,
        )
        trading_client = PerpetualTradingClient(settings.endpoint_config, account)

        # 6. Fetch market info and validate the market exists
        markets = await trading_client.markets_info.get_markets_dict()
        market_info = markets.get(settings.market_name)
        if market_info is None:
            logger.error("Market %s not found on exchange", settings.market_name)
            await trading_client.close()
            return

        tick_size = Decimal(str(market_info.trading_config.min_price_change))
        min_order_size = Decimal(str(market_info.trading_config.min_order_size))
        min_order_size_change = Decimal(
            str(market_info.trading_config.min_order_size_change)
        )

        logger.info(
            "Market %s: tick_size=%s min_order_size=%s step=%s",
            settings.market_name,
            tick_size,
            min_order_size,
            min_order_size_change,
        )

        # 7. Compute order size
        order_size = (min_order_size * settings.order_size_multiplier).quantize(
            min_order_size_change, rounding=ROUND_DOWN
        )
        if order_size < min_order_size:
            order_size = min_order_size
        logger.info("Order size per level: %s", order_size)

        # 8. Build components
        ob_mgr = OrderbookManager(settings.endpoint_config, settings.market_name)
        order_mgr = OrderManager(trading_client, settings.market_name)
        risk_mgr = RiskManager(
            trading_client,
            settings.market_name,
            settings.max_position_size,
            max_order_notional_usd=settings.max_order_notional_usd,
            max_position_notional_usd=settings.max_position_notional_usd,
        )
        account_stream = AccountStreamManager(
            settings.endpoint_config, settings.api_key, settings.market_name
        )
        run_id = uuid.uuid4().hex
        journal = TradeJournal(settings.market_name, run_id=run_id, schema_version=2)
        metrics = MetricsCollector(
            orderbook_mgr=ob_mgr,
            order_mgr=order_mgr,
            risk_mgr=risk_mgr,
            account_stream=account_stream,
            journal=journal,
        )

        strategy = cls(
            settings=settings,
            trading_client=trading_client,
            orderbook_mgr=ob_mgr,
            order_mgr=order_mgr,
            risk_mgr=risk_mgr,
            account_stream=account_stream,
            metrics=metrics,
            journal=journal,
            tick_size=tick_size,
            base_order_size=order_size,
            market_min_order_size=min_order_size,
            min_order_size_step=min_order_size_change,
        )
        journal.record_run_start(
            environment=settings.environment.value,
            config=strategy._sanitized_run_config(settings),
            market_static={
                "tick_size": tick_size,
                "min_order_size": min_order_size,
                "min_order_size_change": min_order_size_change,
            },
            provenance=strategy._run_provenance(),
        )

        # 8b. Wire account stream → order manager + risk manager
        account_stream.on_order_update(order_mgr.handle_order_update)
        account_stream.on_position_update(risk_mgr.handle_position_update)

        # Wire level-freed callback so strategy can re-quote immediately
        order_mgr.on_level_freed(strategy._on_level_freed)

        # Wire fill callback for trade journal
        account_stream.on_fill(strategy._on_fill)

        # 9. Start orderbook + account stream + metrics
        await ob_mgr.start()
        await account_stream.start()
        await metrics.start()

        # Wait for initial prices (use has_data() which skips the staleness
        # check — we just need the first snapshot to arrive)
        logger.info("Waiting for orderbook data...")
        for _ in range(100):
            if ob_mgr.has_data():
                break
            await asyncio.sleep(0.2)
        else:
            logger.error("Timed out waiting for orderbook data — exiting")
            await metrics.stop()
            await account_stream.stop()
            await ob_mgr.stop()
            await trading_client.close()
            journal.record_run_end(reason="startup_timeout")
            journal.close()
            return

        bid_lvl = ob_mgr.best_bid()
        ask_lvl = ob_mgr.best_ask()
        logger.info(
            "Orderbook ready: bid=%s ask=%s",
            bid_lvl,
            ask_lvl,
        )

        # Log market diagnostics to help with configuration
        if bid_lvl and ask_lvl and bid_lvl.price > 0:
            raw_spread = ask_lvl.price - bid_lvl.price
            mid = (bid_lvl.price + ask_lvl.price) / 2
            spread_bps_val = raw_spread / mid * Decimal("10000")
            logger.info(
                "MARKET DIAGNOSTICS for %s:\n"
                "  mid_price  = %s\n"
                "  spread     = %s (%.1f bps)\n"
                "  tick_size  = %s (%.2f bps of mid)\n"
                "  L0 offset  = %s (%.1f bps)  [%s mode]",
                settings.market_name,
                mid,
                raw_spread,
                spread_bps_val,
                tick_size,
                tick_size / mid * Decimal("10000"),
                (
                    f"{settings.price_offset_per_level_percent}%"
                    if settings.offset_mode == OffsetMode.FIXED
                    else f"~{settings.spread_multiplier}x spread"
                ),
                (
                    settings.price_offset_per_level_percent * Decimal("100")
                    if settings.offset_mode == OffsetMode.FIXED
                    else spread_bps_val * settings.spread_multiplier
                ),
                settings.offset_mode.value,
            )
            if (
                settings.offset_mode == OffsetMode.FIXED
                and settings.price_offset_per_level_percent * Decimal("100")
                > spread_bps_val * Decimal("10")
            ):
                logger.warning(
                    "L0 offset (%.0f bps) is >10x the current spread (%.1f bps). "
                    "Orders will likely never fill. Consider using "
                    "MM_OFFSET_MODE=dynamic or lowering "
                    "MM_PRICE_OFFSET_PER_LEVEL_PERCENT.",
                    settings.price_offset_per_level_percent * Decimal("100"),
                    spread_bps_val,
                )

        # 10. Refresh position once before starting
        await risk_mgr.refresh_position()
        logger.info("Current position: %s", risk_mgr.get_current_position())

        # 11. Setup signal handlers for graceful shutdown + hot reload
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, strategy._handle_signal)
        loop.add_signal_handler(signal.SIGHUP, strategy._handle_reload)

        # 12. Build tasks: one per (side, level) + position refresh + circuit breaker
        tasks: list[asyncio.Task] = []
        for level in range(settings.num_price_levels):
            tasks.append(
                asyncio.create_task(
                    strategy._level_task(OrderSide.BUY, level),
                    name=f"mm-buy-L{level}",
                )
            )
            tasks.append(
                asyncio.create_task(
                    strategy._level_task(OrderSide.SELL, level),
                    name=f"mm-sell-L{level}",
                )
            )
        tasks.append(
            asyncio.create_task(
                strategy._position_refresh_task(),
                name="mm-pos-refresh",
            )
        )
        tasks.append(
            asyncio.create_task(
                strategy._circuit_breaker_task(),
                name="mm-circuit-breaker",
            )
        )

        logger.info("Market maker running with %d tasks", len(tasks))

        # 13. Wait until shutdown
        try:
            await strategy._shutdown_event.wait()
        finally:
            logger.info("Shutting down — cancelling tasks and orders...")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            final_snap = metrics.snapshot()
            await order_mgr.cancel_all_orders()
            await metrics.stop()
            await account_stream.stop()
            await ob_mgr.stop()
            await trading_client.close()
            journal.record_run_end(
                reason="shutdown",
                stats={
                    "position": final_snap.position,
                    "active_orders": final_snap.active_orders,
                    "fills": final_snap.total_fills,
                    "cancellations": final_snap.total_cancellations,
                    "rejections": final_snap.total_rejections,
                    "post_only_failures": final_snap.post_only_failures,
                    "total_fees": final_snap.total_fees,
                    "consecutive_failures": final_snap.consecutive_failures,
                    "circuit_open": final_snap.circuit_open,
                    "uptime_s": Decimal(str(final_snap.uptime_s)),
                },
            )
            journal.close()
            logger.info("Market maker stopped")

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_signal(self) -> None:
        logger.info("Signal received, initiating shutdown")
        self._shutdown_event.set()

    def _handle_reload(self) -> None:
        """SIGHUP handler — reload config from environment / .env file."""
        try:
            new_settings = MarketMakerSettings()
            self._settings = new_settings
            logger.info(
                "Config reloaded: offset_mode=%s skew=%.2f spread_min=%s levels=%d max_age=%ss",
                new_settings.offset_mode.value,
                new_settings.inventory_skew_factor,
                new_settings.min_spread_bps,
                new_settings.num_price_levels,
                new_settings.max_order_age_s,
            )
        except Exception as exc:
            logger.error("Config reload failed: %s", exc)

    @staticmethod
    def _sanitized_run_config(settings: MarketMakerSettings) -> Dict[str, Any]:
        """Return a config snapshot safe to persist in journals."""
        data = settings.model_dump(mode="python")
        redact_keys = {
            "vault_id",
            "stark_private_key",
            "stark_public_key",
            "api_key",
        }
        for key in redact_keys:
            if key in data:
                data[key] = "***redacted***"
        return data

    @staticmethod
    def _run_provenance() -> Dict[str, Any]:
        """Best-effort runtime provenance for reproducibility."""
        provenance: Dict[str, Any] = {
            "env_file": str(ENV_FILE),
            "cwd": os.getcwd(),
        }
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            sha = res.stdout.strip()
            if res.returncode == 0 and sha:
                provenance["git_sha"] = sha
        except Exception:
            # Keep provenance optional and never break strategy startup.
            pass
        return provenance

    # ------------------------------------------------------------------
    # Fill callback (from account stream → trade journal)
    # ------------------------------------------------------------------

    def _on_fill(self, fill: FillEvent) -> None:
        """Record every fill to the trade journal with orderbook context."""
        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        market_snapshot = self._ob.market_snapshot(
            depth=self._settings.fill_snapshot_depth,
            micro_vol_window_s=self._settings.micro_vol_window_s,
            micro_drift_window_s=self._settings.micro_drift_window_s,
            imbalance_window_s=self._settings.imbalance_window_s,
        )

        order_info = self._orders.find_order_by_exchange_id(str(fill.order_id))
        level = order_info.level if order_info is not None else None
        if order_info is not None:
            key = (str(order_info.side), order_info.level)
            self._reset_pof_state(key)

        self._journal.record_fill(
            trade_id=fill.trade_id,
            order_id=fill.order_id,
            side=str(fill.side),
            price=fill.price,
            qty=fill.qty,
            fee=fill.fee,
            is_taker=fill.is_taker,
            level=level,
            best_bid=bid.price if bid else None,
            best_ask=ask.price if ask else None,
            position=self._risk.get_current_position(),
            market_snapshot=market_snapshot,
        )

    # ------------------------------------------------------------------
    # Level-freed callback (from account stream → order manager)
    # ------------------------------------------------------------------

    def _on_level_freed(
        self,
        side_value: str,
        level: int,
        external_id: str,
        *,
        rejected: bool = False,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        price: Optional[Decimal] = None,
    ) -> None:
        """Called when an order at (side, level) reaches a terminal state.

        Clears the level slot so the next iteration of _level_task will
        immediately place a new order instead of waiting for a price change.
        If the order was rejected (e.g. POST_ONLY_FAILED), apply a cooldown
        to prevent rapid retry storms.
        """
        key = (side_value, level)
        current = self._level_ext_ids.get(key)
        status_upper = str(status or "").upper()
        side_for_journal = self._normalise_side(side_value)
        order_info = self._orders.find_order_by_external_id(external_id)
        exchange_id = (
            order_info.exchange_order_id if order_info is not None else None
        )
        if current == external_id:
            self._clear_level_slot(key)
        cancel_reason = self._pending_cancel_reasons.pop(
            external_id,
            reason or "terminal",
        )

        if rejected:
            self._journal.record_rejection(
                external_id=external_id,
                exchange_id=exchange_id,
                side=side_for_journal,
                price=price if price is not None else Decimal("0"),
                reason=reason or "REJECTED",
            )
        elif status_upper in {"CANCELLED", "EXPIRED"}:
            self._journal.record_order_cancelled(
                external_id=external_id,
                exchange_id=exchange_id,
                side=side_for_journal,
                level=level,
                reason=cancel_reason,
            )

        # Apply POF cooldown to prevent immediate retry
        if (
            rejected
            and self._settings.pof_cooldown_s > 0
            and (reason is None or "POST_ONLY_FAILED" in str(reason).upper())
        ):
            if self._settings.adaptive_pof_enabled:
                self._apply_adaptive_pof_reject(key)
            else:
                self._level_pof_until[key] = (
                    time.monotonic() + self._settings.pof_cooldown_s
                )

    # ------------------------------------------------------------------
    # Per-level task
    # ------------------------------------------------------------------

    async def _level_task(self, side: OrderSide, level: int) -> None:
        """Continuously quote on one (side, level) slot."""
        key = (str(side), level)
        self._clear_level_slot(key)

        condition = (
            self._ob.best_bid_condition
            if side == OrderSide.BUY
            else self._ob.best_ask_condition
        )

        while not self._shutdown_event.is_set():
            # Pause while circuit breaker is open
            if self._circuit_open:
                await asyncio.sleep(1.0)
                continue

            try:
                # Wait for a price change on this side of the book
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass  # Re-evaluate anyway every few seconds
            except asyncio.CancelledError:
                return

            try:
                await self._maybe_reprice(side, level)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(
                    "Error in level task %s L%d: %s", side, level, exc,
                    exc_info=True,
                )
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Repricing logic
    # ------------------------------------------------------------------

    async def _maybe_reprice(self, side: OrderSide, level: int) -> None:
        """Evaluate whether the order at (side, level) needs repricing."""
        key = (str(side), level)
        now = time.monotonic()

        # --- POF cooldown: don't re-quote a level that just got rejected ---
        pof_until = self._level_pof_until.get(key, 0.0)
        if now < pof_until:
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_pof_cooldown",
            )
            return

        # --- Min reprice interval: prevent cancel/place churn ---
        min_interval = self._settings.min_reprice_interval_s
        if min_interval > 0:
            last_reprice = self._level_last_reprice_at.get(key, 0.0)
            if (now - last_reprice) < min_interval:
                return

        active_orders = self._orders.get_active_orders()
        prev_ext_id = self._level_ext_ids.get(key)
        prev_order = active_orders.get(prev_ext_id) if prev_ext_id else None

        # --- Stale orderbook fail-safe ---
        if self._ob.is_stale():
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_stale",
            )
            if self._settings.cancel_on_stale_book and prev_ext_id is not None:
                stale_since = self._level_stale_since.get(key)
                if stale_since is None:
                    stale_since = now
                    self._level_stale_since[key] = stale_since
                if (now - stale_since) >= self._settings.stale_cancel_grace_s:
                    logger.info(
                        "Orderbook stale for %.1fs — cancelling %s L%d",
                        now - stale_since,
                        side,
                        level,
                    )
                    await self._cancel_level_order(
                        key=key,
                        external_id=prev_ext_id,
                        side=side,
                        level=level,
                        reason="stale_orderbook",
                    )
            return

        self._level_stale_since[key] = None

        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        if bid is None or ask is None:
            return

        # --- Double-check staleness right before any quoting logic ---
        # Guards against race where stale cancel fires but a new order
        # slips through using cached data from the same evaluation cycle.
        if self._ob.is_stale():
            return

        # --- Minimum spread check ---
        spread_bps = self._ob.spread_bps()
        if self._settings.min_spread_bps > 0:
            if spread_bps is not None and spread_bps < self._settings.min_spread_bps:
                self._record_reprice_decision(
                    side=side,
                    level=level,
                    reason="skip_min_spread",
                    spread_bps=spread_bps,
                )
                # Cancel any resting order so it cannot be filled in thin-spread regimes.
                if prev_ext_id is not None:
                    logger.info(
                        "Spread %.2fbps below min %.2fbps — cancelling %s L%d",
                        spread_bps,
                        self._settings.min_spread_bps,
                        side,
                        level,
                    )
                    await self._cancel_level_order(
                        key=key,
                        external_id=prev_ext_id,
                        side=side,
                        level=level,
                        reason="min_spread",
                    )
                return

        # --- One-sided imbalance pause ---
        imbalance = self._ob.orderbook_imbalance(self._settings.imbalance_window_s)
        pause_threshold = self._settings.imbalance_pause_threshold
        pause_until = self._level_imbalance_pause_until.get(key, 0.0)
        if now < pause_until:
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_imbalance_pause",
                spread_bps=spread_bps,
            )
            return
        if (
            imbalance is not None
            and pause_threshold > 0
            and (
                (side == OrderSide.SELL and imbalance > pause_threshold)
                or (side == OrderSide.BUY and imbalance < -pause_threshold)
            )
        ):
            pause_for = max(1.0, float(self._settings.imbalance_window_s))
            self._level_imbalance_pause_until[key] = now + pause_for
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_imbalance",
                spread_bps=spread_bps,
            )
            return

        # --- Toxicity guard: widen in mild stress, pause in severe stress ---
        extra_offset_bps, pause_reason = self._toxicity_adjustment()
        if pause_reason is not None:
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="skip_toxicity",
                spread_bps=spread_bps,
                extra_offset_bps=extra_offset_bps,
            )
            if prev_ext_id is not None:
                logger.info(
                    "Toxicity guard (%s) — cancelling %s L%d",
                    pause_reason,
                    side,
                    level,
                )
                await self._cancel_level_order(
                    key=key,
                    external_id=prev_ext_id,
                    side=side,
                    level=level,
                    reason=f"toxicity:{pause_reason}",
                )
            return

        # Enforce a maximum resting age for stale orders.
        if prev_order is not None and self._order_age_exceeded(key):
            placed_ts = self._level_order_created_at.get(key)
            age_s = (
                time.monotonic() - placed_ts
                if placed_ts is not None
                else self._settings.max_order_age_s
            )
            logger.info(
                "Order max age exceeded (%.1fs > %.1fs) — refreshing %s L%d",
                age_s,
                self._settings.max_order_age_s,
                side,
                level,
            )
            self._record_reprice_decision(
                side=side,
                level=level,
                reason="replace_max_age",
                prev_price=prev_order.price,
                spread_bps=spread_bps,
            )
            await self._cancel_level_order(
                key=key,
                external_id=prev_ext_id,
                side=side,
                level=level,
                reason="max_order_age",
            )
            prev_ext_id = None
            prev_order = None

        current_best = bid.price if side == OrderSide.BUY else ask.price
        target_price = self._compute_target_price(
            side,
            level,
            current_best,
            extra_offset_bps=extra_offset_bps,
        )

        if prev_order is not None:
            should_reprice, decision_reason = self._needs_reprice(
                side,
                prev_order.price,
                current_best,
                level,
                extra_offset_bps=extra_offset_bps,
            )
            if not should_reprice:
                self._record_reprice_decision(
                    side=side,
                    level=level,
                    reason=decision_reason,
                    current_best=current_best,
                    prev_price=prev_order.price,
                    target_price=target_price,
                    spread_bps=spread_bps,
                    extra_offset_bps=extra_offset_bps,
                )
                return  # Order is still within tolerance / gates
            self._record_reprice_decision(
                side=side,
                level=level,
                reason=decision_reason,
                current_best=current_best,
                prev_price=prev_order.price,
                target_price=target_price,
                spread_bps=spread_bps,
                extra_offset_bps=extra_offset_bps,
            )

        # --- Reprice needed ---
        logger.info(
            "Reprice %s L%d: best=%s target=%s prev=%s",
            side,
            level,
            current_best,
            target_price,
            prev_order.price if prev_order else "none",
        )

        # Compute per-level order size (pyramid scaling)
        requested_size = self._level_size(level)
        reserved_same_side_qty = sum(
            info.size
            for ext_id, info in active_orders.items()
            if str(info.side) == str(side) and ext_id != prev_ext_id
        )
        level_size = self._quantize_size(
            self._risk.allowed_order_size(
                side,
                requested_size,
                target_price,
                reserved_same_side_qty=reserved_same_side_qty,
            )
        )

        # Cancel the existing order if we can no longer hold this side.
        if level_size < self._market_min_order_size:
            logger.info(
                "Allowed size below market minimum: requested=%s allowed=%s min=%s (%s L%d)",
                requested_size,
                level_size,
                self._market_min_order_size,
                side,
                level,
            )
            # Cancel the existing order if we can no longer hold this side
            if prev_ext_id is not None:
                await self._cancel_level_order(
                    key=key,
                    external_id=prev_ext_id,
                    side=side,
                    level=level,
                    reason="risk_limit",
                )
            return

        # Cancel previous order before placing the replacement.
        # We do NOT use cancel_id / previous_order_id because the exchange
        # rejects the NEW order with PREV_ORDER_NOT_FOUND if the old one is
        # already filled/cancelled — losing the replacement entirely.
        if prev_ext_id is not None:
            await self._cancel_level_order(
                key=key,
                external_id=prev_ext_id,
                side=side,
                level=level,
                reason="reprice",
            )

        # Final post-only safety recheck right before placement.
        # Also re-check staleness to prevent orphaned orders if the book
        # went stale between the top-of-function check and now.
        if self._ob.is_stale():
            return
        fresh_bid = self._ob.best_bid()
        fresh_ask = self._ob.best_ask()
        if fresh_bid is None or fresh_ask is None:
            return
        safe_target_price = self._apply_post_only_safety(
            side=side,
            target_price=target_price,
            bid_price=fresh_bid.price,
            ask_price=fresh_ask.price,
            safety_ticks=self._effective_safety_ticks(key),
        )
        if safe_target_price is None:
            return

        # Place new order
        ext_id = await self._orders.place_order(
            side=side,
            price=safe_target_price,
            size=level_size,
            level=level,
        )
        if ext_id is not None:
            self._on_successful_quote(key)
            self._level_ext_ids[key] = ext_id
            self._level_order_created_at[key] = time.monotonic()
            self._level_last_reprice_at[key] = time.monotonic()
            order_info = self._orders.get_active_orders().get(ext_id)
            self._journal.record_order_placed(
                external_id=ext_id,
                exchange_id=order_info.exchange_order_id if order_info else None,
                side=self._normalise_side(str(side)),
                price=safe_target_price,
                size=level_size,
                level=level,
                best_bid=fresh_bid.price if fresh_bid else None,
                best_ask=fresh_ask.price if fresh_ask else None,
                position=self._risk.get_current_position(),
            )

    def _clear_level_slot(self, key: tuple[str, int]) -> None:
        """Clear tracking for one (side, level) slot."""
        self._level_ext_ids[key] = None
        self._level_order_created_at[key] = None
        self._level_stale_since[key] = None

    def _record_reprice_decision(
        self,
        *,
        side: OrderSide,
        level: int,
        reason: str,
        current_best: Optional[Decimal] = None,
        prev_price: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
        spread_bps: Optional[Decimal] = None,
        extra_offset_bps: Optional[Decimal] = None,
    ) -> None:
        if not self._settings.journal_reprice_decisions:
            return
        self._journal.record_reprice_decision(
            side=self._normalise_side(str(side)),
            level=level,
            reason=reason,
            current_best=current_best,
            prev_price=prev_price,
            target_price=target_price,
            spread_bps=spread_bps,
            extra_offset_bps=extra_offset_bps,
        )

    async def _cancel_level_order(
        self,
        *,
        key: tuple[str, int],
        external_id: str,
        side: OrderSide,
        level: int,
        reason: str,
    ) -> None:
        """Request cancel for a level order and store a structured reason."""
        self._pending_cancel_reasons[external_id] = reason
        ok = await self._orders.cancel_order(external_id)
        if not ok:
            self._pending_cancel_reasons.pop(external_id, None)
        self._clear_level_slot(key)

    def _quantize_size(self, size: Decimal) -> Decimal:
        """Quantize order size to market step size."""
        if size <= 0:
            return Decimal("0")
        if self._min_order_size_step <= 0:
            return size
        return size.quantize(self._min_order_size_step, rounding=ROUND_DOWN)

    def _effective_safety_ticks(self, key: tuple[str, int]) -> int:
        base_ticks = max(1, self._settings.post_only_safety_ticks)
        if not self._settings.adaptive_pof_enabled:
            return base_ticks
        dynamic_ticks = self._level_dynamic_safety_ticks.get(key, base_ticks)
        return max(base_ticks, dynamic_ticks)

    def _apply_adaptive_pof_reject(self, key: tuple[str, int]) -> None:
        now = time.monotonic()
        last_ts = self._level_pof_last_ts.get(key)
        reset_window_s = self._settings.pof_streak_reset_s
        if (
            last_ts is None
            or (reset_window_s > 0 and (now - last_ts) > reset_window_s)
        ):
            streak = 0
        else:
            streak = self._level_pof_streak.get(key, 0)
        streak += 1
        self._level_pof_streak[key] = streak
        self._level_pof_last_ts[key] = now

        base_ticks = max(1, self._settings.post_only_safety_ticks)
        max_ticks = max(base_ticks, self._settings.pof_max_safety_ticks)
        dynamic_ticks = min(max_ticks, base_ticks + streak)
        self._level_dynamic_safety_ticks[key] = dynamic_ticks

        multiplier = max(Decimal("1"), self._settings.pof_backoff_multiplier)
        cooldown_s = float(
            Decimal(str(self._settings.pof_cooldown_s))
            * (multiplier ** (streak - 1))
        )
        cooldown_s = min(cooldown_s, 120.0)
        self._level_pof_until[key] = now + cooldown_s

    def _on_successful_quote(self, key: tuple[str, int]) -> None:
        """Decay adaptive POF state after successful accepted placement."""
        if not self._settings.adaptive_pof_enabled:
            return
        streak = self._level_pof_streak.get(key, 0)
        if streak <= 0:
            return
        streak -= 1
        self._level_pof_streak[key] = streak
        base_ticks = max(1, self._settings.post_only_safety_ticks)
        self._level_dynamic_safety_ticks[key] = min(
            max(base_ticks, self._settings.pof_max_safety_ticks),
            base_ticks + streak,
        )
        if streak == 0:
            self._level_pof_until[key] = 0.0

    def _reset_pof_state(self, key: tuple[str, int]) -> None:
        self._level_pof_streak[key] = 0
        self._level_pof_until[key] = 0.0
        self._level_pof_last_ts[key] = time.monotonic()
        self._level_dynamic_safety_ticks[key] = max(
            1, self._settings.post_only_safety_ticks
        )

    def _apply_post_only_safety(
        self,
        *,
        side: OrderSide,
        target_price: Decimal,
        bid_price: Decimal,
        ask_price: Decimal,
        safety_ticks: Optional[int] = None,
    ) -> Optional[Decimal]:
        """Clamp target to a safe post-only price just before placement."""
        if bid_price <= 0 or ask_price <= 0:
            return None
        if self._tick_size <= 0:
            return target_price

        safety_ticks = max(
            1,
            safety_ticks
            if safety_ticks is not None
            else self._settings.post_only_safety_ticks,
        )
        safety_buffer = self._tick_size * Decimal(safety_ticks)

        if side == OrderSide.BUY:
            max_buy = ask_price - safety_buffer
            safe = min(target_price, max_buy)
            safe = self._round_to_tick(safe, side)
            if safe >= ask_price:
                safe = ask_price - self._tick_size
        else:
            min_sell = bid_price + safety_buffer
            safe = max(target_price, min_sell)
            safe = self._round_to_tick(safe, side)
            if safe <= bid_price:
                safe = bid_price + self._tick_size

        if safe <= 0:
            return None
        return safe

    def _toxicity_adjustment(self) -> tuple[Decimal, Optional[str]]:
        """Return (extra_offset_bps, pause_reason) from microstructure stress."""
        vol_bps = self._ob.micro_volatility_bps(self._settings.micro_vol_window_s)
        drift_bps = self._ob.micro_drift_bps(self._settings.micro_drift_window_s)
        drift_abs = abs(drift_bps) if drift_bps is not None else None

        vol_limit = self._settings.micro_vol_max_bps
        drift_limit = self._settings.micro_drift_max_bps

        # Hard pause on severe micro-regime stress.
        if (
            vol_bps is not None
            and vol_limit > 0
            and vol_bps > (vol_limit * Decimal("1.25"))
        ):
            return Decimal("0"), "volatility_spike"
        if (
            drift_abs is not None
            and drift_limit > 0
            and drift_abs > (drift_limit * Decimal("1.25"))
        ):
            return Decimal("0"), "drift_spike"

        # Moderate widening above soft thresholds.
        extra_bps = Decimal("0")
        if vol_bps is not None and vol_limit > 0 and vol_bps > vol_limit:
            extra_bps += (vol_bps - vol_limit) * self._settings.volatility_offset_multiplier
        if drift_abs is not None and drift_limit > 0 and drift_abs > drift_limit:
            extra_bps += (
                drift_abs - drift_limit
            ) * self._settings.volatility_offset_multiplier
        return max(Decimal("0"), extra_bps), None

    @staticmethod
    def _normalise_side(side_value: str) -> str:
        side_upper = side_value.upper()
        if "BUY" in side_upper:
            return "BUY"
        if "SELL" in side_upper:
            return "SELL"
        return side_value

    def _order_age_exceeded(self, key: tuple[str, int]) -> bool:
        """Return True if the tracked order at *key* exceeded max_order_age_s."""
        max_age_s = self._settings.max_order_age_s
        if max_age_s <= 0:
            return False
        placed_ts = self._level_order_created_at.get(key)
        if placed_ts is None:
            return False
        return (time.monotonic() - placed_ts) > max_age_s

    def _compute_offset(self, level: int, best_price: Decimal) -> Decimal:
        """Compute the raw price offset for *level*.

        In FIXED mode this is simply ``best_price * offset_pct * (level+1) / 100``.
        In DYNAMIC mode it is derived from the live spread, clamped between
        ``min_offset_bps`` and ``max_offset_bps``.
        """
        _100 = Decimal("100")
        _10000 = Decimal("10000")

        if self._settings.offset_mode == OffsetMode.DYNAMIC:
            # Use EMA-smoothed spread to avoid noisy offset from transient spikes
            spread_bps = self._ob.spread_bps_ema()
            if spread_bps is None or spread_bps <= 0:
                # Fallback to floor when spread is unavailable
                spread_bps = self._settings.min_offset_bps

            per_level_bps = spread_bps * self._settings.spread_multiplier * (level + 1)

            # Clamp
            floor = self._settings.min_offset_bps * (level + 1)
            ceiling = self._settings.max_offset_bps * (level + 1)
            per_level_bps = max(floor, min(per_level_bps, ceiling))

            return best_price * per_level_bps / _10000
        else:
            offset_pct = self._settings.price_offset_per_level_percent * (level + 1)
            return best_price * offset_pct / _100

    def _compute_target_price(
        self,
        side: OrderSide,
        level: int,
        best_price: Decimal,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
    ) -> Decimal:
        """Calculate the target price for a given level with inventory skew."""
        offset = self._compute_offset(level, best_price)
        if extra_offset_bps > 0:
            offset += best_price * extra_offset_bps / Decimal("10000")

        # --- Inventory skew (shaped + deadband, capped in bps) ---
        max_pos = self._settings.max_position_size
        if max_pos > 0:
            skew_norm = self._risk.get_current_position() / max_pos
        else:
            skew_norm = Decimal("0")
        skew_norm = max(Decimal("-1"), min(Decimal("1"), skew_norm))

        deadband = max(Decimal("0"), min(Decimal("1"), self._settings.inventory_deadband_pct))
        abs_norm = abs(skew_norm)
        if abs_norm <= deadband:
            shaped = Decimal("0")
        else:
            if deadband >= Decimal("1"):
                normalized = Decimal("0")
            else:
                normalized = (abs_norm - deadband) / (Decimal("1") - deadband)
            sign = Decimal("1") if skew_norm >= 0 else Decimal("-1")
            shape_k = float(max(Decimal("0"), self._settings.skew_shape_k))
            if shape_k == 0:
                curve = float(normalized)
            else:
                denom = math.tanh(shape_k)
                curve = 0.0 if denom == 0 else math.tanh(shape_k * float(normalized)) / denom
            shaped = sign * Decimal(str(curve))

        max_skew_bps = self._settings.skew_max_bps * self._settings.inventory_skew_factor
        skew_offset = best_price * (shaped * max_skew_bps) / Decimal("10000")

        if side == OrderSide.BUY:
            # When long (skew > 0), shift bid further down → less aggressive buying
            raw = best_price - offset - skew_offset
        else:
            # When long (skew > 0), shift ask further down → more aggressive selling
            raw = best_price + offset - skew_offset

        # --- Safety clamp: never cross the spread ---
        # Prevents inventory skew from pushing orders past the opposite BBO
        # which would cause POST_ONLY_FAILED rejections.
        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        if bid is not None and ask is not None:
            if side == OrderSide.BUY and raw >= ask.price:
                raw = ask.price - self._tick_size
            elif side == OrderSide.SELL and raw <= bid.price:
                raw = bid.price + self._tick_size

        # Round to tick: BUY rounds DOWN (safer, further from mid),
        # SELL rounds UP (safer, further from mid).
        return self._round_to_tick(raw, side)

    def _round_to_tick(
        self, price: Decimal, side: Optional[OrderSide] = None
    ) -> Decimal:
        """Round *price* to the nearest tick.

        BUY orders round DOWN (floor) to stay further below mid.
        SELL orders round UP (ceil) to stay further above mid.
        """
        if self._tick_size <= 0:
            return price
        rounding = ROUND_UP if side == OrderSide.SELL else ROUND_DOWN
        return (price / self._tick_size).quantize(
            Decimal("1"), rounding=rounding
        ) * self._tick_size

    def _level_size(self, level: int) -> Decimal:
        """Compute order size for *level* using per-level scaling.

        L0 = min_order_size, L1 = min_order_size * scale, L2 = min_order_size * scale^2 ...
        """
        scale = self._settings.size_scale_per_level
        if scale == 1 or level == 0:
            return self._base_order_size
        return (self._base_order_size * scale ** level).quantize(
            self._min_order_size_step, rounding=ROUND_DOWN
        )

    def _needs_reprice(
        self,
        side: OrderSide,
        prev_price: Decimal,
        current_best: Decimal,
        level: int,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
    ) -> tuple[bool, str]:
        """Determine whether the existing order needs to be replaced.

        Compares the existing order price against the *full* target price
        (including inventory skew).  Reprices if the difference exceeds the
        tolerance band around the target offset.

        This ensures that position changes (fills) that shift the skew
        component trigger repricing, not just BBO movements.
        """
        if current_best == 0:
            return True, "replace_target_shift"

        target_price = self._compute_target_price(
            side,
            level,
            current_best,
            extra_offset_bps=extra_offset_bps,
        )

        # Compute the target offset in price terms for tolerance scaling
        target_offset = self._compute_offset(level, current_best)
        if extra_offset_bps > 0:
            target_offset += current_best * extra_offset_bps / Decimal("10000")
        if target_offset == 0:
            return True, "replace_target_shift"

        tolerance = self._settings.reprice_tolerance_percent
        max_deviation = target_offset * tolerance

        # How far is the current order from the ideal target?
        price_diff = abs(prev_price - target_price)
        if price_diff <= max_deviation:
            return False, "hold_within_tolerance"

        min_move_ticks = self._settings.min_reprice_move_ticks
        if (
            min_move_ticks > 0
            and self._tick_size > 0
            and price_diff < (self._tick_size * Decimal(min_move_ticks))
        ):
            return False, "hold_below_tick_gate"

        prev_edge = self._theoretical_edge_bps(side, prev_price, current_best)
        target_edge = self._theoretical_edge_bps(side, target_price, current_best)
        edge_delta = abs(target_edge - prev_edge)
        if edge_delta < self._settings.min_reprice_edge_delta_bps:
            return False, "hold_below_edge_gate"

        return True, "replace_target_shift"

    @staticmethod
    def _theoretical_edge_bps(
        side: OrderSide,
        quote_price: Decimal,
        current_best: Decimal,
    ) -> Decimal:
        """Return quote edge in bps vs same-side best price."""
        if current_best <= 0:
            return Decimal("0")
        if side == OrderSide.BUY:
            return (current_best - quote_price) / current_best * Decimal("10000")
        return (quote_price - current_best) / current_best * Decimal("10000")

    # ------------------------------------------------------------------
    # Background position refresh (safety-net)
    # ------------------------------------------------------------------

    async def _position_refresh_task(self) -> None:
        """Periodically refresh the cached position from the exchange.

        This is a safety net — the account stream provides real-time updates.
        """
        while not self._shutdown_event.is_set():
            try:
                await self._risk.refresh_position()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Position refresh error: %s", exc)
            await asyncio.sleep(_POSITION_REFRESH_INTERVAL_S)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    async def _circuit_breaker_task(self) -> None:
        """Monitor consecutive failures and pause quoting when threshold is hit."""
        max_failures = self._settings.circuit_breaker_max_failures
        cooldown = self._settings.circuit_breaker_cooldown_s
        failure_window_s = self._settings.failure_window_s
        failure_rate_trip = float(self._settings.failure_rate_trip)
        min_attempts = self._settings.min_attempts_for_breaker

        while not self._shutdown_event.is_set():
            try:
                window_stats = self._orders.failure_window_stats(failure_window_s)
                attempts = int(window_stats["attempts"])
                failure_rate = float(window_stats["failure_rate"])
                window_trip = (
                    attempts >= min_attempts and failure_rate >= failure_rate_trip
                )
                if (
                    not self._circuit_open
                    and (
                        self._orders.consecutive_failures >= max_failures
                        or window_trip
                    )
                ):
                    self._circuit_open = True
                    self._metrics.circuit_open = True
                    logger.warning(
                        "CIRCUIT BREAKER OPEN: consecutive_failures=%d attempts=%d "
                        "failure_rate=%.2f — pausing for %.0fs",
                        self._orders.consecutive_failures,
                        attempts,
                        failure_rate,
                        cooldown,
                    )
                    await asyncio.sleep(cooldown)
                    self._orders.reset_failure_tracking()
                    self._circuit_open = False
                    self._metrics.circuit_open = False
                    logger.info("Circuit breaker reset — resuming")
                else:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Circuit breaker task error: %s", exc)
                await asyncio.sleep(1.0)
