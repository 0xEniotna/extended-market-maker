from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Type

from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager
from .config import MarketMakerSettings, OffsetMode
from .fee_resolver import FeeResolver
from .metrics import MetricsCollector
from .monitoring_tasks import (
    balance_refresh_task,
    config_rollback_task,
    kpi_watchdog_task,
    latency_sla_task,
    pnl_attribution_task,
    position_refresh_task,
    qtr_monitor_task,
)
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .shutdown_manager import (
    install_signal_handlers,
    reset_module_state,
    shutdown_and_record,
    stop_services,
)
from .strategy_components import run_provenance, sanitized_run_config
from .strategy_factory import StrategyFactory
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)


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
    return StrategyFactory(settings).build_trading_client()


async def _load_market_params(
    settings: MarketMakerSettings,
    trading_client: PerpetualTradingClient,
) -> tuple[Any, Decimal, Decimal, Decimal, Decimal]:
    return await StrategyFactory(settings).load_market_params(trading_client)


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
    factory = StrategyFactory(settings)
    fee_resolver = factory.build_fee_resolver(trading_client)
    ob_mgr = factory.build_orderbook_manager()
    order_mgr = factory.build_order_manager(trading_client, fee_resolver)
    risk_mgr = factory.build_risk_manager(trading_client, ob_mgr)
    account_stream = factory.build_account_stream()
    journal = factory.build_journal()
    metrics = factory.build_metrics(
        ob_mgr=ob_mgr,
        order_mgr=order_mgr,
        risk_mgr=risk_mgr,
        account_stream=account_stream,
        journal=journal,
    )
    strategy = factory.build_strategy(
        strategy_cls,
        trading_client=trading_client,
        market_info=market_info,
        ob_mgr=ob_mgr,
        order_mgr=order_mgr,
        risk_mgr=risk_mgr,
        account_stream=account_stream,
        metrics=metrics,
        journal=journal,
        tick_size=tick_size,
        order_size=order_size,
        min_order_size=min_order_size,
        min_order_size_change=min_order_size_change,
    )
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
        config=sanitized_run_config(ctx.settings),
        market_static={
            "tick_size": ctx.tick_size,
            "min_order_size": ctx.min_order_size,
            "min_order_size_change": ctx.min_order_size_change,
        },
        provenance=run_provenance(),
    )


def _register_callbacks(ctx: RuntimeContext) -> None:
    ctx.account_stream.on_order_update(ctx.order_mgr.handle_order_update)
    ctx.account_stream.on_position_update(ctx.risk_mgr.handle_position_update)
    ctx.account_stream.on_balance_update(ctx.risk_mgr.handle_balance_update)
    ctx.order_mgr.on_level_freed(ctx.strategy._on_level_freed)
    ctx.account_stream.on_fill(ctx.strategy._on_fill)
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


def _create_tasks(ctx: RuntimeContext) -> list[asyncio.Task]:
    tasks: list[asyncio.Task] = []
    for level in range(ctx.settings.num_price_levels):
        tasks.append(
            asyncio.create_task(ctx.strategy._level_task(OrderSide.BUY, level), name=f"mm-buy-L{level}")
        )
        tasks.append(
            asyncio.create_task(ctx.strategy._level_task(OrderSide.SELL, level), name=f"mm-sell-L{level}")
        )
    tasks.append(asyncio.create_task(position_refresh_task(ctx.strategy), name="mm-pos-refresh"))
    tasks.append(asyncio.create_task(balance_refresh_task(ctx.strategy), name="mm-balance-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._margin_guard_task(), name="mm-margin-guard"))
    tasks.append(asyncio.create_task(ctx.strategy._circuit_breaker_task(), name="mm-circuit-breaker"))
    tasks.append(asyncio.create_task(ctx.strategy._funding_refresh_task(), name="mm-funding-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._drawdown_watchdog_task(), name="mm-drawdown-watchdog"))
    tasks.append(asyncio.create_task(kpi_watchdog_task(ctx.strategy), name="mm-kpi-watchdog"))
    tasks.append(asyncio.create_task(pnl_attribution_task(ctx.strategy), name="mm-pnl-attribution"))
    tasks.append(asyncio.create_task(qtr_monitor_task(ctx.strategy), name="mm-qtr-monitor"))
    tasks.append(asyncio.create_task(latency_sla_task(ctx.strategy), name="mm-latency-sla"))
    tasks.append(asyncio.create_task(config_rollback_task(ctx.strategy), name="mm-config-rollback"))
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


async def run_strategy(strategy_cls: Type) -> None:
    """Load config, initialise components, and run the strategy."""
    reset_module_state()

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
        await stop_services(ctx)
        ctx.journal.record_run_end(reason="startup_timeout")
        ctx.journal.close()
        return

    _log_market_diagnostics(ctx)
    await _refresh_risk_state(ctx)
    install_signal_handlers(ctx.strategy)
    tasks = _create_tasks(ctx)

    try:
        await ctx.strategy._shutdown_event.wait()
    finally:
        await shutdown_and_record(ctx, tasks)
