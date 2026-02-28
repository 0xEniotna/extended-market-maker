from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any, Type

from x10.perpetual.accounts import StarkPerpetualAccount
from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager
from .config import MarketMakerSettings, OffsetMode
from .fee_resolver import FeeResolver
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

# Module-level flag for double-signal detection.
_shutdown_in_progress = False
_force_exit_event: asyncio.Event | None = None


class _DecimalEncoder(json.JSONEncoder):
    """Encode Decimal as string to preserve precision in JSON."""

    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


@dataclass
class RuntimeContext:
    settings: MarketMakerSettings
    trading_client: PerpetualTradingClient
    market_info: Any
    tick_size: Decimal
    min_order_size: Decimal
    min_order_size_change: Decimal
    order_size: Decimal
    ob_mgr: OrderbookManager
    order_mgr: OrderManager
    risk_mgr: RiskManager
    account_stream: AccountStreamManager
    journal: TradeJournal
    metrics: MetricsCollector
    strategy: Any
    fee_resolver: FeeResolver | None = None


def _configure_logging(settings: MarketMakerSettings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def _log_startup(settings: MarketMakerSettings) -> None:
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
        logger.info("  Fixed offset: %.4f%% per level", settings.price_offset_per_level_percent)
    else:
        logger.info(
            "  Dynamic offset: spread_mult=%.2f min=%sbps max=%sbps",
            settings.spread_multiplier,
            settings.min_offset_bps,
            settings.max_offset_bps,
        )


def _validate_startup(settings: MarketMakerSettings) -> bool:
    if not settings.enabled:
        logger.warning("MM_ENABLED is false — exiting")
        return False
    if not settings.is_configured:
        logger.error(
            "Missing credentials (MM_VAULT_ID, MM_STARK_PRIVATE_KEY, "
            "MM_STARK_PUBLIC_KEY, MM_API_KEY). Exiting."
        )
        return False
    return True


def _build_trading_client(settings: MarketMakerSettings) -> PerpetualTradingClient:
    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    return PerpetualTradingClient(settings.endpoint_config, account)


async def _load_market_params(
    settings: MarketMakerSettings,
    trading_client: PerpetualTradingClient,
) -> tuple[Any, Decimal, Decimal, Decimal, Decimal]:
    markets = await trading_client.markets_info.get_markets_dict()
    market_info = markets.get(settings.market_name)
    if market_info is None:
        raise LookupError(f"Market {settings.market_name} not found on exchange")

    tick_size = Decimal(str(market_info.trading_config.min_price_change))
    min_order_size = Decimal(str(market_info.trading_config.min_order_size))
    min_order_size_change = Decimal(str(market_info.trading_config.min_order_size_change))
    order_size = (min_order_size * settings.order_size_multiplier).quantize(
        min_order_size_change, rounding=ROUND_DOWN
    )
    if order_size < min_order_size:
        order_size = min_order_size
    return market_info, tick_size, min_order_size, min_order_size_change, order_size


def _log_market_params(
    settings: MarketMakerSettings,
    *,
    tick_size: Decimal,
    min_order_size: Decimal,
    min_order_size_change: Decimal,
    order_size: Decimal,
) -> None:
    logger.info(
        "Market %s: tick_size=%s min_order_size=%s step=%s",
        settings.market_name,
        tick_size,
        min_order_size,
        min_order_size_change,
    )
    logger.info("Order size per level: %s", order_size)


def _build_runtime_context(
    settings: MarketMakerSettings,
    strategy_cls: Type,
    trading_client: PerpetualTradingClient,
    market_info: Any,
    *,
    tick_size: Decimal,
    min_order_size: Decimal,
    min_order_size_change: Decimal,
    order_size: Decimal,
) -> RuntimeContext:
    fee_resolver = FeeResolver(
        trading_client=trading_client,
        market_name=settings.market_name,
        refresh_interval_s=settings.fee_refresh_interval_s,
        builder_program_enabled=settings.builder_program_enabled,
        builder_id=settings.builder_id if settings.builder_id > 0 else None,
        configured_builder_fee_rate=settings.builder_fee_rate,
    )
    ob_mgr = OrderbookManager(
        settings.endpoint_config,
        settings.market_name,
        staleness_threshold_s=settings.orderbook_staleness_threshold_s,
    )
    order_mgr = OrderManager(
        trading_client,
        settings.market_name,
        max_orders_per_second=settings.max_orders_per_second,
        maintenance_pause_s=settings.maintenance_pause_s,
        fee_resolver=fee_resolver,
        rate_limit_degraded_s=settings.rate_limit_degraded_s,
        rate_limit_halt_window_s=settings.rate_limit_halt_window_s,
        rate_limit_halt_hits=settings.rate_limit_halt_hits,
        rate_limit_halt_s=settings.rate_limit_halt_s,
        rate_limit_extra_offset_bps=settings.rate_limit_extra_offset_bps,
        rate_limit_reprice_multiplier=settings.rate_limit_reprice_multiplier,
    )
    risk_mgr = RiskManager(
        trading_client,
        settings.market_name,
        settings.max_position_size,
        max_order_notional_usd=settings.max_order_notional_usd,
        max_position_notional_usd=settings.max_position_notional_usd,
        gross_exposure_limit_usd=settings.gross_exposure_limit_usd,
        max_long_position_size=settings.max_long_position_size,
        max_short_position_size=settings.max_short_position_size,
        balance_aware_sizing_enabled=settings.balance_aware_sizing_enabled,
        balance_usage_factor=settings.balance_usage_factor,
        balance_notional_multiplier=settings.balance_notional_multiplier,
        balance_min_available_usd=settings.balance_min_available_usd,
        balance_staleness_max_s=settings.balance_staleness_max_s,
        balance_stale_action=settings.balance_stale_action,
        orderbook_mgr=ob_mgr,
    )
    account_stream = AccountStreamManager(
        settings.endpoint_config, settings.api_key, settings.market_name
    )
    journal = TradeJournal(
        settings.market_name,
        run_id=uuid.uuid4().hex,
        schema_version=2,
        max_size_mb=settings.journal_max_size_mb,
    )
    metrics = MetricsCollector(
        orderbook_mgr=ob_mgr,
        order_mgr=order_mgr,
        risk_mgr=risk_mgr,
        account_stream=account_stream,
        journal=journal,
    )
    strategy = strategy_cls(
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
    try:
        strategy._set_funding_rate(Decimal(str(market_info.market_stats.funding_rate)))
    except Exception:
        logger.debug("Funding rate unavailable at startup", exc_info=True)

    return RuntimeContext(
        settings=settings,
        trading_client=trading_client,
        market_info=market_info,
        tick_size=tick_size,
        min_order_size=min_order_size,
        min_order_size_change=min_order_size_change,
        order_size=order_size,
        fee_resolver=fee_resolver,
        ob_mgr=ob_mgr,
        order_mgr=order_mgr,
        risk_mgr=risk_mgr,
        account_stream=account_stream,
        journal=journal,
        metrics=metrics,
        strategy=strategy,
    )


def _record_run_start(ctx: RuntimeContext) -> None:
    ctx.journal.record_run_start(
        environment=ctx.settings.environment.value,
        config=ctx.strategy._sanitized_run_config(ctx.settings),
        market_static={
            "tick_size": ctx.tick_size,
            "min_order_size": ctx.min_order_size,
            "min_order_size_change": ctx.min_order_size_change,
        },
        provenance=ctx.strategy._run_provenance(),
    )


def _register_callbacks(ctx: RuntimeContext) -> None:
    ctx.account_stream.on_order_update(ctx.order_mgr.handle_order_update)
    ctx.account_stream.on_position_update(ctx.risk_mgr.handle_position_update)
    ctx.account_stream.on_balance_update(ctx.risk_mgr.handle_balance_update)
    ctx.order_mgr.on_level_freed(ctx.strategy._on_level_freed)
    ctx.account_stream.on_fill(ctx.strategy._on_fill)
    # Wire cross-references for health monitoring and event logging.
    ctx.account_stream.set_order_manager(ctx.order_mgr)
    ctx.account_stream.set_journal(ctx.journal)
    ctx.account_stream.set_fail_safe_handler(ctx.strategy._on_stream_desync)
    ctx.ob_mgr.set_fail_safe_handler(ctx.strategy._on_stream_desync)
    ctx.order_mgr.set_journal(ctx.journal)


async def _start_services(ctx: RuntimeContext) -> None:
    await ctx.ob_mgr.start()
    await ctx.account_stream.start()
    await ctx.metrics.start()
    ctx.order_mgr.start_rate_limiter()


async def _stop_services(ctx: RuntimeContext) -> None:
    await ctx.metrics.stop()
    await ctx.order_mgr.stop_rate_limiter()
    await ctx.account_stream.stop()
    await ctx.ob_mgr.stop()
    # Wait for in-flight order operations to complete before closing.
    await ctx.order_mgr.wait_for_inflight(timeout_s=5.0)
    await ctx.trading_client.close()


async def _wait_for_orderbook(ctx: RuntimeContext) -> bool:
    logger.info("Waiting for orderbook data...")
    for _ in range(100):
        if ctx.ob_mgr.has_data():
            return True
        await asyncio.sleep(0.2)
    logger.error("Timed out waiting for orderbook data — exiting")
    return False


def _log_market_diagnostics(ctx: RuntimeContext) -> None:
    bid_lvl = ctx.ob_mgr.best_bid()
    ask_lvl = ctx.ob_mgr.best_ask()
    logger.info("Orderbook ready: bid=%s ask=%s", bid_lvl, ask_lvl)
    if not bid_lvl or not ask_lvl or bid_lvl.price <= 0:
        return

    raw_spread = ask_lvl.price - bid_lvl.price
    mid = (bid_lvl.price + ask_lvl.price) / 2
    spread_bps_val = raw_spread / mid * Decimal("10000")
    logger.info(
        "MARKET DIAGNOSTICS for %s:\n"
        "  mid_price  = %s\n"
        "  spread     = %s (%.1f bps)\n"
        "  tick_size  = %s (%.2f bps of mid)\n"
        "  L0 offset  = %s (%.1f bps)  [%s mode]",
        ctx.settings.market_name,
        mid,
        raw_spread,
        spread_bps_val,
        ctx.tick_size,
        ctx.tick_size / mid * Decimal("10000"),
        (
            f"{ctx.settings.price_offset_per_level_percent}%"
            if ctx.settings.offset_mode == OffsetMode.FIXED
            else f"~{ctx.settings.spread_multiplier}x spread"
        ),
        (
            ctx.settings.price_offset_per_level_percent * Decimal("100")
            if ctx.settings.offset_mode == OffsetMode.FIXED
            else spread_bps_val * ctx.settings.spread_multiplier
        ),
        ctx.settings.offset_mode.value,
    )
    if (
        ctx.settings.offset_mode == OffsetMode.FIXED
        and ctx.settings.price_offset_per_level_percent * Decimal("100")
        > spread_bps_val * Decimal("10")
    ):
        logger.warning(
            "L0 offset (%.0f bps) is >10x the current spread (%.1f bps). "
            "Orders will likely never fill. Consider using "
            "MM_OFFSET_MODE=dynamic or lowering MM_PRICE_OFFSET_PER_LEVEL_PERCENT.",
            ctx.settings.price_offset_per_level_percent * Decimal("100"),
            spread_bps_val,
        )


async def _refresh_risk_state(ctx: RuntimeContext) -> None:
    await ctx.risk_mgr.refresh_position()
    await ctx.risk_mgr.refresh_balance()
    logger.info("Current position: %s", ctx.risk_mgr.get_current_position())
    logger.info("Available for trade: %s", ctx.risk_mgr.get_available_for_trade())


def _install_signal_handlers(strategy: Any) -> None:
    global _shutdown_in_progress, _force_exit_event
    loop = asyncio.get_running_loop()
    _force_exit_event = asyncio.Event()

    def _double_signal_handler() -> None:
        global _shutdown_in_progress
        if _shutdown_in_progress:
            # Second signal: force-exit immediately
            logger.critical(
                "Second signal received during shutdown — skipping flatten, force-exiting"
            )
            if _force_exit_event is not None:
                _force_exit_event.set()
            return
        # First signal: normal shutdown
        strategy._handle_signal()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _double_signal_handler)
    loop.add_signal_handler(signal.SIGHUP, strategy._handle_reload)


def _create_tasks(ctx: RuntimeContext) -> list[asyncio.Task]:
    tasks: list[asyncio.Task] = []
    for level in range(ctx.settings.num_price_levels):
        tasks.append(
            asyncio.create_task(ctx.strategy._level_task(OrderSide.BUY, level), name=f"mm-buy-L{level}")
        )
        tasks.append(
            asyncio.create_task(ctx.strategy._level_task(OrderSide.SELL, level), name=f"mm-sell-L{level}")
        )
    tasks.append(asyncio.create_task(ctx.strategy._position_refresh_task(), name="mm-pos-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._balance_refresh_task(), name="mm-balance-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._margin_guard_task(), name="mm-margin-guard"))
    tasks.append(asyncio.create_task(ctx.strategy._circuit_breaker_task(), name="mm-circuit-breaker"))
    tasks.append(asyncio.create_task(ctx.strategy._funding_refresh_task(), name="mm-funding-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._drawdown_watchdog_task(), name="mm-drawdown-watchdog"))
    tasks.append(asyncio.create_task(ctx.strategy._kpi_watchdog_task(), name="mm-kpi-watchdog"))
    tasks.append(asyncio.create_task(ctx.strategy._pnl_attribution_task(), name="mm-pnl-attribution"))
    tasks.append(asyncio.create_task(ctx.strategy._qtr_monitor_task(), name="mm-qtr-monitor"))
    tasks.append(asyncio.create_task(ctx.strategy._latency_sla_task(), name="mm-latency-sla"))
    tasks.append(asyncio.create_task(ctx.strategy._config_rollback_task(), name="mm-config-rollback"))
    tasks.append(asyncio.create_task(_heartbeat_task(ctx), name="mm-heartbeat"))
    if ctx.settings.deadman_enabled:
        tasks.append(asyncio.create_task(_deadman_heartbeat_task(ctx), name="mm-deadman"))
    logger.info("Market maker running with %d tasks", len(tasks))
    return tasks


async def _heartbeat_task(ctx: RuntimeContext) -> None:
    """Emit a heartbeat journal event every 30 seconds."""
    start_ts = time.monotonic()
    while True:
        try:
            await asyncio.sleep(30.0)
            ctx.journal.record_heartbeat(
                position=ctx.risk_mgr.get_current_position(),
                event_count=ctx.journal.event_count,
                active_orders=ctx.order_mgr.active_order_count(),
                uptime_s=time.monotonic() - start_ts,
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Heartbeat error: %s", exc)


async def _deadman_heartbeat_task(ctx: RuntimeContext) -> None:
    """Periodically arm the exchange dead-man switch."""
    countdown_s = int(max(0, ctx.settings.deadman_countdown_s))
    interval_s = max(1.0, float(ctx.settings.deadman_heartbeat_s))
    while True:
        try:
            await ctx.trading_client.account.set_deadman_switch(countdown_s)
            logger.info(
                "Dead-man heartbeat armed: market=%s countdown=%ss interval=%.1fs",
                ctx.settings.market_name,
                countdown_s,
                interval_s,
            )
            ctx.journal.record_exchange_event(
                event_type="deadman_heartbeat",
                details={
                    "countdown_s": countdown_s,
                    "interval_s": interval_s,
                },
            )
            ctx.metrics.set_deadman_status(
                armed=True,
                countdown_s=countdown_s,
                last_ok_ts=time.time(),
            )
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Dead-man heartbeat error: %s", exc)
            ctx.metrics.set_deadman_status(
                armed=False,
                countdown_s=countdown_s,
                last_ok_ts=None,
            )
            await asyncio.sleep(min(interval_s, 5.0))


async def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------
# State persistence helpers
# ------------------------------------------------------------------

def _write_json_state(directory: str, market_name: str, data: dict) -> Path | None:
    """Write a JSON state file to a directory, creating it if needed.

    Returns the written path, or None if writing fails.
    """
    try:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        safe_market = market_name.replace("/", "-")
        file_path = dir_path / f"{safe_market}_{ts}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, cls=_DecimalEncoder, indent=2)
        return file_path
    except Exception as exc:
        logger.error("Failed to write state file to %s: %s", directory, exc)
        return None


def _collect_orderbook_state(ctx: RuntimeContext) -> dict:
    """Collect current orderbook state for persistence."""
    bid = ctx.ob_mgr.best_bid()
    ask = ctx.ob_mgr.best_ask()
    return {
        "best_bid": str(bid.price) if bid else None,
        "best_bid_size": str(bid.size) if bid else None,
        "best_ask": str(ask.price) if ask else None,
        "best_ask_size": str(ask.size) if ask else None,
        "is_stale": ctx.ob_mgr.is_stale(),
        "spread_bps": str(ctx.ob_mgr.spread_bps()) if bid and ask else None,
    }


def _collect_active_orders(ctx: RuntimeContext) -> list[dict]:
    """Collect active order info for persistence."""
    result = []
    for ext_id, info in ctx.order_mgr.get_active_orders().items():
        result.append({
            "external_id": ext_id,
            "side": str(info.side),
            "price": str(info.price),
            "size": str(info.size),
            "level": info.level,
            "exchange_order_id": info.exchange_order_id,
        })
    return result


def _write_pre_shutdown_state(ctx: RuntimeContext, shutdown_reason: str) -> None:
    """Write a snapshot before starting the shutdown sequence."""
    data = {
        "type": "pre_shutdown_state",
        "timestamp": time.time(),
        "market": ctx.settings.market_name,
        "shutdown_reason": shutdown_reason,
        "position": ctx.risk_mgr.get_current_position(),
        "position_pnl": {
            "realized": ctx.risk_mgr.get_position_realized_pnl(),
            "unrealized": ctx.risk_mgr.get_position_unrealized_pnl(),
            "total": ctx.risk_mgr.get_position_total_pnl(),
        },
        "available_for_trade": ctx.risk_mgr.get_available_for_trade(),
        "active_orders": _collect_active_orders(ctx),
        "orderbook": _collect_orderbook_state(ctx),
        "config": {
            "max_position_size": ctx.settings.max_position_size,
            "max_position_notional_usd": ctx.settings.max_position_notional_usd,
            "flatten_position_on_shutdown": ctx.settings.flatten_position_on_shutdown,
            "shutdown_flatten_retries": ctx.settings.shutdown_flatten_retries,
            "shutdown_flatten_slippage_bps": ctx.settings.shutdown_flatten_slippage_bps,
            "shutdown_flatten_slippage_step_bps": ctx.settings.shutdown_flatten_slippage_step_bps,
            "shutdown_flatten_max_slippage_bps": ctx.settings.shutdown_flatten_max_slippage_bps,
            "shutdown_timeout_s": ctx.settings.shutdown_timeout_s,
        },
    }
    path = _write_json_state("data/mm_shutdown_state", ctx.settings.market_name, data)
    if path:
        logger.info("Pre-shutdown state written to %s", path)


def _write_emergency_state(
    ctx: RuntimeContext,
    reason: str,
) -> None:
    """Write emergency state file when shutdown timeout is exceeded."""
    data = {
        "type": "emergency_state",
        "timestamp": time.time(),
        "market": ctx.settings.market_name,
        "reason": reason,
        "position": ctx.risk_mgr.get_current_position(),
        "active_orders": _collect_active_orders(ctx),
        "orderbook": _collect_orderbook_state(ctx),
    }
    path = _write_json_state("data/mm_emergency", ctx.settings.market_name, data)
    if path:
        logger.critical("Emergency state written to %s", path)


def _get_last_known_mid(ctx: RuntimeContext) -> Decimal | None:
    """Get the last known mid price from orderbook, even if stale."""
    bid = ctx.ob_mgr.best_bid()
    ask = ctx.ob_mgr.best_ask()
    if bid is not None and ask is not None and bid.price > 0 and ask.price > 0:
        return (bid.price + ask.price) / 2
    # Try individual sides
    if bid is not None and bid.price > 0:
        return bid.price
    if ask is not None and ask.price > 0:
        return ask.price
    return None


# ------------------------------------------------------------------
# Progressive slippage
# ------------------------------------------------------------------

def _compute_progressive_slippage(
    attempt: int,
    base_bps: Decimal,
    step_bps: Decimal,
    max_bps: Decimal,
) -> Decimal:
    """Compute slippage for a given attempt (1-indexed).

    attempt 1 → base_bps
    attempt 2 → base_bps + step_bps
    attempt N → min(base_bps + (N-1) * step_bps, max_bps)
    """
    slippage = base_bps + step_bps * Decimal(str(attempt - 1))
    return min(slippage, max_bps)


# ------------------------------------------------------------------
# Shutdown flatten with progressive slippage & one-sided book
# ------------------------------------------------------------------

async def _attempt_shutdown_flatten(
    ctx: RuntimeContext,
    shutdown_reason: str,
) -> dict[str, Any]:
    flatten_enabled = ctx.settings.flatten_position_on_shutdown or shutdown_reason == "drawdown_stop"
    flatten_attempted = False
    flatten_submitted = False
    flatten_reason = "disabled"
    flatten_attempts = 0
    if not flatten_enabled:
        return {
            "enabled": flatten_enabled,
            "attempted": flatten_attempted,
            "submitted": flatten_submitted,
            "reason": flatten_reason,
            "attempts": flatten_attempts,
        }

    flatten_reason = "already_flat"
    for attempt in range(1, ctx.settings.shutdown_flatten_retries + 1):
        # Check for force-exit signal
        if _force_exit_event is not None and _force_exit_event.is_set():
            flatten_reason = "force_exit"
            logger.warning("Flatten aborted by force-exit signal on attempt %d", attempt)
            break

        current_position = ctx.risk_mgr.get_current_position()
        if current_position == 0:
            break

        bid_lvl = ctx.ob_mgr.best_bid()
        ask_lvl = ctx.ob_mgr.best_ask()
        last_mid = _get_last_known_mid(ctx)
        flatten_attempts = attempt

        slippage = _compute_progressive_slippage(
            attempt=attempt,
            base_bps=ctx.settings.shutdown_flatten_slippage_bps,
            step_bps=ctx.settings.shutdown_flatten_slippage_step_bps,
            max_bps=ctx.settings.shutdown_flatten_max_slippage_bps,
        )

        flatten_result = await ctx.order_mgr.flatten_position(
            signed_position=current_position,
            best_bid=bid_lvl.price if bid_lvl else None,
            best_ask=ask_lvl.price if ask_lvl else None,
            tick_size=ctx.tick_size,
            min_order_size=ctx.min_order_size,
            size_step=ctx.min_order_size_change,
            slippage_bps=slippage,
            last_known_mid=last_mid,
        )
        flatten_attempted = flatten_attempted or flatten_result.attempted
        flatten_submitted = flatten_submitted or flatten_result.success
        flatten_reason = flatten_result.reason

        await asyncio.sleep(ctx.settings.shutdown_flatten_retry_delay_s)
        await ctx.risk_mgr.refresh_position()
        remaining_position = ctx.risk_mgr.get_current_position()
        if remaining_position == 0:
            flatten_reason = "flattened"
            logger.warning(
                "Shutdown flatten completed on attempt %d/%d for market=%s (slippage=%sbps)",
                attempt,
                ctx.settings.shutdown_flatten_retries,
                ctx.settings.market_name,
                slippage,
            )
            break

        logger.warning(
            "Shutdown flatten attempt %d/%d incomplete for market=%s: "
            "before=%s remaining=%s reason=%s slippage=%sbps",
            attempt,
            ctx.settings.shutdown_flatten_retries,
            ctx.settings.market_name,
            current_position,
            remaining_position,
            flatten_result.reason,
            slippage,
        )
        if flatten_result.reason in {"below_min_order_size", "missing_orderbook_price"}:
            break

    # --- Feature 6: Fresh REST verification after flatten loop ---
    try:
        await ctx.risk_mgr.refresh_position()
    except Exception as exc:
        logger.error("Failed to refresh position after flatten loop: %s", exc)
    final_position = ctx.risk_mgr.get_current_position()
    if final_position != 0:
        bid_lvl = ctx.ob_mgr.best_bid()
        ask_lvl = ctx.ob_mgr.best_ask()
        ref_price = Decimal("0")
        if bid_lvl and ask_lvl:
            ref_price = (bid_lvl.price + ask_lvl.price) / 2
        elif bid_lvl:
            ref_price = bid_lvl.price
        elif ask_lvl:
            ref_price = ask_lvl.price
        remaining_notional = abs(final_position) * ref_price if ref_price > 0 else Decimal("0")
        logger.critical(
            "POSITION NOT FLAT after all flatten retries for market=%s: "
            "remaining_position=%s remaining_notional_usd=%s",
            ctx.settings.market_name,
            final_position,
            remaining_notional,
        )

    return {
        "enabled": flatten_enabled,
        "attempted": flatten_attempted,
        "submitted": flatten_submitted,
        "reason": flatten_reason,
        "attempts": flatten_attempts,
    }


# ------------------------------------------------------------------
# Core shutdown sequence
# ------------------------------------------------------------------

async def _shutdown_core(ctx: RuntimeContext, tasks: list[asyncio.Task]) -> None:
    """Core shutdown logic: cancel tasks/orders, flatten, record stats."""
    global _shutdown_in_progress
    _shutdown_in_progress = True

    logger.info("Shutting down — cancelling tasks, orders, and flattening position...")
    ctx.strategy._set_runtime_mode("shutdown")

    # --- Feature 4: Pre-shutdown state persistence ---
    shutdown_reason = ctx.strategy.shutdown_reason
    _write_pre_shutdown_state(ctx, shutdown_reason)

    await _cancel_tasks(tasks)
    await ctx.order_mgr.cancel_all_orders()

    await ctx.risk_mgr.refresh_position()
    shutdown_position_before_flatten = ctx.risk_mgr.get_current_position()

    # Check force-exit before flatten
    if _force_exit_event is not None and _force_exit_event.is_set():
        flatten = {
            "enabled": False,
            "attempted": False,
            "submitted": False,
            "reason": "force_exit",
            "attempts": 0,
        }
        logger.warning("Skipping flatten due to force-exit signal")
    else:
        ctx.strategy._set_runtime_mode("flatten")
        flatten = await _attempt_shutdown_flatten(ctx, shutdown_reason)
        ctx.strategy._set_runtime_mode("shutdown")

    deadman_enabled = bool(getattr(ctx.settings, "deadman_enabled", False))
    if deadman_enabled:
        try:
            await ctx.trading_client.account.set_deadman_switch(0)
            logger.info("Dead-man switch disarmed for market=%s", ctx.settings.market_name)
            ctx.journal.record_exchange_event(
                event_type="deadman_disarmed",
                details={"countdown_s": 0},
            )
            if hasattr(ctx.metrics, "set_deadman_status"):
                ctx.metrics.set_deadman_status(
                    armed=False,
                    countdown_s=0,
                    last_ok_ts=time.time(),
                )
        except Exception as exc:
            logger.error("Failed to disarm dead-man switch: %s", exc)

    await ctx.risk_mgr.refresh_position()
    shutdown_position_after_flatten = ctx.risk_mgr.get_current_position()
    final_snap = ctx.metrics.snapshot()
    await _stop_services(ctx)
    ctx.journal.record_run_end(
        reason=shutdown_reason,
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
            "shutdown_position_before_flatten": shutdown_position_before_flatten,
            "shutdown_position_after_flatten": shutdown_position_after_flatten,
            "shutdown_flatten_enabled": flatten["enabled"],
            "shutdown_flatten_attempted": flatten["attempted"],
            "shutdown_flatten_submitted": flatten["submitted"],
            "shutdown_flatten_attempts": flatten["attempts"],
            "shutdown_flatten_reason": flatten["reason"],
        },
    )
    ctx.journal.close()
    logger.info("Market maker stopped")


async def _shutdown_and_record(ctx: RuntimeContext, tasks: list[asyncio.Task]) -> None:
    """Wrap shutdown in a hard timeout. On timeout, write emergency state and force exit."""
    timeout_s = ctx.settings.shutdown_timeout_s
    try:
        await asyncio.wait_for(_shutdown_core(ctx, tasks), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.critical(
            "SHUTDOWN TIMEOUT: shutdown sequence exceeded %ss for market=%s — force exiting",
            timeout_s,
            ctx.settings.market_name,
        )
        _write_emergency_state(ctx, reason="shutdown_timeout")
        # Best-effort cleanup
        try:
            ctx.journal.close()
        except Exception:
            pass
        try:
            await _stop_services(ctx)
        except Exception:
            pass
        os._exit(1)


async def run_strategy(strategy_cls: Type) -> None:
    """Load config, initialise components, and run the strategy."""
    global _shutdown_in_progress, _force_exit_event
    _shutdown_in_progress = False
    _force_exit_event = None

    settings = MarketMakerSettings()
    _configure_logging(settings)
    _log_startup(settings)
    if not _validate_startup(settings):
        return

    trading_client = _build_trading_client(settings)
    try:
        market_info, tick_size, min_order_size, min_order_size_change, order_size = await _load_market_params(
            settings, trading_client
        )
    except LookupError:
        logger.error("Market %s not found on exchange", settings.market_name)
        await trading_client.close()
        return

    _log_market_params(
        settings,
        tick_size=tick_size,
        min_order_size=min_order_size,
        min_order_size_change=min_order_size_change,
        order_size=order_size,
    )
    try:
        ctx = _build_runtime_context(
            settings,
            strategy_cls,
            trading_client,
            market_info,
            tick_size=tick_size,
            min_order_size=min_order_size,
            min_order_size_change=min_order_size_change,
            order_size=order_size,
        )
    except Exception as exc:
        logger.error("Failed to initialize runtime context: %s", exc)
        await trading_client.close()
        return
    try:
        await ctx.fee_resolver.refresh(force=True)
        await ctx.fee_resolver.validate_builder_config()
    except Exception as exc:
        logger.error("Fee resolver startup validation failed: %s", exc)
        await trading_client.close()
        return
    _record_run_start(ctx)
    _register_callbacks(ctx)
    await _start_services(ctx)

    if not await _wait_for_orderbook(ctx):
        await _stop_services(ctx)
        ctx.journal.record_run_end(reason="startup_timeout")
        ctx.journal.close()
        return

    _log_market_diagnostics(ctx)
    await _refresh_risk_state(ctx)
    _install_signal_handlers(ctx.strategy)
    tasks = _create_tasks(ctx)

    try:
        await ctx.strategy._shutdown_event.wait()
    finally:
        await _shutdown_and_record(ctx, tasks)
