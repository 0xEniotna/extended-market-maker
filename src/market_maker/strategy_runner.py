from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any, Type

from x10.perpetual.accounts import StarkPerpetualAccount
from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager
from .config import MarketMakerSettings, OffsetMode
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
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
    ob_mgr = OrderbookManager(settings.endpoint_config, settings.market_name)
    order_mgr = OrderManager(trading_client, settings.market_name)
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
        orderbook_mgr=ob_mgr,
    )
    account_stream = AccountStreamManager(
        settings.endpoint_config, settings.api_key, settings.market_name
    )
    journal = TradeJournal(settings.market_name, run_id=uuid.uuid4().hex, schema_version=2)
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


async def _start_services(ctx: RuntimeContext) -> None:
    await ctx.ob_mgr.start()
    await ctx.account_stream.start()
    await ctx.metrics.start()


async def _stop_services(ctx: RuntimeContext) -> None:
    await ctx.metrics.stop()
    await ctx.account_stream.stop()
    await ctx.ob_mgr.stop()
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
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, strategy._handle_signal)
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
    tasks.append(asyncio.create_task(ctx.strategy._circuit_breaker_task(), name="mm-circuit-breaker"))
    tasks.append(asyncio.create_task(ctx.strategy._funding_refresh_task(), name="mm-funding-refresh"))
    tasks.append(asyncio.create_task(ctx.strategy._drawdown_watchdog_task(), name="mm-drawdown-watchdog"))
    logger.info("Market maker running with %d tasks", len(tasks))
    return tasks


async def _cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


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
        current_position = ctx.risk_mgr.get_current_position()
        if current_position == 0:
            break

        bid_lvl = ctx.ob_mgr.best_bid()
        ask_lvl = ctx.ob_mgr.best_ask()
        flatten_attempts = attempt
        flatten_result = await ctx.order_mgr.flatten_position(
            signed_position=current_position,
            best_bid=bid_lvl.price if bid_lvl else None,
            best_ask=ask_lvl.price if ask_lvl else None,
            tick_size=ctx.tick_size,
            min_order_size=ctx.min_order_size,
            size_step=ctx.min_order_size_change,
            slippage_bps=ctx.settings.shutdown_flatten_slippage_bps,
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
                "Shutdown flatten completed on attempt %d/%d for market=%s",
                attempt,
                ctx.settings.shutdown_flatten_retries,
                ctx.settings.market_name,
            )
            break

        logger.warning(
            "Shutdown flatten attempt %d/%d incomplete for market=%s: "
            "before=%s remaining=%s reason=%s",
            attempt,
            ctx.settings.shutdown_flatten_retries,
            ctx.settings.market_name,
            current_position,
            remaining_position,
            flatten_result.reason,
        )
        if flatten_result.reason in {"below_min_order_size", "missing_orderbook_price"}:
            break

    return {
        "enabled": flatten_enabled,
        "attempted": flatten_attempted,
        "submitted": flatten_submitted,
        "reason": flatten_reason,
        "attempts": flatten_attempts,
    }


async def _shutdown_and_record(ctx: RuntimeContext, tasks: list[asyncio.Task]) -> None:
    logger.info("Shutting down — cancelling tasks, orders, and flattening position...")
    await _cancel_tasks(tasks)
    await ctx.order_mgr.cancel_all_orders()

    await ctx.risk_mgr.refresh_position()
    shutdown_position_before_flatten = ctx.risk_mgr.get_current_position()
    shutdown_reason = ctx.strategy.shutdown_reason
    flatten = await _attempt_shutdown_flatten(ctx, shutdown_reason)

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


async def run_strategy(strategy_cls: Type) -> None:
    """Load config, initialise components, and run the strategy."""
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
