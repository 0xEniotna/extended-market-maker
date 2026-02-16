from __future__ import annotations

import asyncio
import logging
import signal
import sys
import uuid
from decimal import ROUND_DOWN, Decimal
from typing import Type

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


async def run_strategy(strategy_cls: Type) -> None:
    """Load config, initialise components, and run the strategy."""
    settings = MarketMakerSettings()
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
        logger.info("  Fixed offset: %.4f%% per level", settings.price_offset_per_level_percent)
    else:
        logger.info(
            "  Dynamic offset: spread_mult=%.2f min=%sbps max=%sbps",
            settings.spread_multiplier,
            settings.min_offset_bps,
            settings.max_offset_bps,
        )

    if not settings.enabled:
        logger.warning("MM_ENABLED is false — exiting")
        return
    if not settings.is_configured:
        logger.error(
            "Missing credentials (MM_VAULT_ID, MM_STARK_PRIVATE_KEY, "
            "MM_STARK_PUBLIC_KEY, MM_API_KEY). Exiting."
        )
        return

    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    trading_client = PerpetualTradingClient(settings.endpoint_config, account)

    markets = await trading_client.markets_info.get_markets_dict()
    market_info = markets.get(settings.market_name)
    if market_info is None:
        logger.error("Market %s not found on exchange", settings.market_name)
        await trading_client.close()
        return

    tick_size = Decimal(str(market_info.trading_config.min_price_change))
    min_order_size = Decimal(str(market_info.trading_config.min_order_size))
    min_order_size_change = Decimal(str(market_info.trading_config.min_order_size_change))
    logger.info(
        "Market %s: tick_size=%s min_order_size=%s step=%s",
        settings.market_name,
        tick_size,
        min_order_size,
        min_order_size_change,
    )

    order_size = (min_order_size * settings.order_size_multiplier).quantize(
        min_order_size_change, rounding=ROUND_DOWN
    )
    if order_size < min_order_size:
        order_size = min_order_size
    logger.info("Order size per level: %s", order_size)

    ob_mgr = OrderbookManager(settings.endpoint_config, settings.market_name)
    order_mgr = OrderManager(trading_client, settings.market_name)
    risk_mgr = RiskManager(
        trading_client,
        settings.market_name,
        settings.max_position_size,
        max_order_notional_usd=settings.max_order_notional_usd,
        max_position_notional_usd=settings.max_position_notional_usd,
        balance_aware_sizing_enabled=settings.balance_aware_sizing_enabled,
        balance_usage_factor=settings.balance_usage_factor,
        balance_notional_multiplier=settings.balance_notional_multiplier,
        balance_min_available_usd=settings.balance_min_available_usd,
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

    account_stream.on_order_update(order_mgr.handle_order_update)
    account_stream.on_position_update(risk_mgr.handle_position_update)
    account_stream.on_balance_update(risk_mgr.handle_balance_update)
    order_mgr.on_level_freed(strategy._on_level_freed)
    account_stream.on_fill(strategy._on_fill)

    await ob_mgr.start()
    await account_stream.start()
    await metrics.start()

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
    logger.info("Orderbook ready: bid=%s ask=%s", bid_lvl, ask_lvl)
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
                "MM_OFFSET_MODE=dynamic or lowering MM_PRICE_OFFSET_PER_LEVEL_PERCENT.",
                settings.price_offset_per_level_percent * Decimal("100"),
                spread_bps_val,
            )

    await risk_mgr.refresh_position()
    await risk_mgr.refresh_balance()
    logger.info("Current position: %s", risk_mgr.get_current_position())
    logger.info("Available for trade: %s", risk_mgr.get_available_for_trade())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, strategy._handle_signal)
    loop.add_signal_handler(signal.SIGHUP, strategy._handle_reload)

    tasks: list[asyncio.Task] = []
    for level in range(settings.num_price_levels):
        tasks.append(asyncio.create_task(strategy._level_task(OrderSide.BUY, level), name=f"mm-buy-L{level}"))
        tasks.append(asyncio.create_task(strategy._level_task(OrderSide.SELL, level), name=f"mm-sell-L{level}"))
    tasks.append(asyncio.create_task(strategy._position_refresh_task(), name="mm-pos-refresh"))
    tasks.append(asyncio.create_task(strategy._balance_refresh_task(), name="mm-balance-refresh"))
    tasks.append(asyncio.create_task(strategy._circuit_breaker_task(), name="mm-circuit-breaker"))
    tasks.append(asyncio.create_task(strategy._funding_refresh_task(), name="mm-funding-refresh"))
    logger.info("Market maker running with %d tasks", len(tasks))

    try:
        await strategy._shutdown_event.wait()
    finally:
        logger.info("Shutting down — cancelling tasks, orders, and flattening position...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await order_mgr.cancel_all_orders()

        await risk_mgr.refresh_position()
        shutdown_position_before_flatten = risk_mgr.get_current_position()
        flatten_enabled = settings.flatten_position_on_shutdown
        flatten_attempted = False
        flatten_submitted = False
        flatten_reason = "disabled"
        flatten_attempts = 0

        if flatten_enabled:
            flatten_reason = "already_flat"
            for attempt in range(1, settings.shutdown_flatten_retries + 1):
                current_position = risk_mgr.get_current_position()
                if current_position == 0:
                    flatten_reason = "already_flat"
                    break

                bid_lvl = ob_mgr.best_bid()
                ask_lvl = ob_mgr.best_ask()
                flatten_attempts = attempt
                flatten_result = await order_mgr.flatten_position(
                    signed_position=current_position,
                    best_bid=bid_lvl.price if bid_lvl else None,
                    best_ask=ask_lvl.price if ask_lvl else None,
                    tick_size=tick_size,
                    min_order_size=min_order_size,
                    size_step=min_order_size_change,
                    slippage_bps=settings.shutdown_flatten_slippage_bps,
                )
                flatten_attempted = flatten_attempted or flatten_result.attempted
                flatten_submitted = flatten_submitted or flatten_result.success
                flatten_reason = flatten_result.reason

                await asyncio.sleep(settings.shutdown_flatten_retry_delay_s)
                await risk_mgr.refresh_position()
                remaining_position = risk_mgr.get_current_position()
                if remaining_position == 0:
                    flatten_reason = "flattened"
                    logger.warning(
                        "Shutdown flatten completed on attempt %d/%d for market=%s",
                        attempt,
                        settings.shutdown_flatten_retries,
                        settings.market_name,
                    )
                    break

                logger.warning(
                    "Shutdown flatten attempt %d/%d incomplete for market=%s: "
                    "before=%s remaining=%s reason=%s",
                    attempt,
                    settings.shutdown_flatten_retries,
                    settings.market_name,
                    current_position,
                    remaining_position,
                    flatten_result.reason,
                )

                if flatten_result.reason in {"below_min_order_size", "missing_orderbook_price"}:
                    # Non-retryable without a position size or book update change.
                    break

        await risk_mgr.refresh_position()
        shutdown_position_after_flatten = risk_mgr.get_current_position()
        final_snap = metrics.snapshot()
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
                "shutdown_position_before_flatten": shutdown_position_before_flatten,
                "shutdown_position_after_flatten": shutdown_position_after_flatten,
                "shutdown_flatten_enabled": flatten_enabled,
                "shutdown_flatten_attempted": flatten_attempted,
                "shutdown_flatten_submitted": flatten_submitted,
                "shutdown_flatten_attempts": flatten_attempts,
                "shutdown_flatten_reason": flatten_reason,
            },
        )
        journal.close()
        logger.info("Market maker stopped")
