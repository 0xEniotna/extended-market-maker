"""
Shutdown Manager

Encapsulates the full shutdown lifecycle extracted from ``strategy_runner``:

- Pre-shutdown state persistence
- Progressive-slippage flatten loop
- Deadman switch disarming
- Emergency state writing on timeout
- Signal handling (double-signal detection)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .strategy_runner import RuntimeContext

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


# ------------------------------------------------------------------
# Signal handling
# ------------------------------------------------------------------

def install_signal_handlers(strategy: Any) -> None:
    global _shutdown_in_progress, _force_exit_event
    loop = asyncio.get_running_loop()
    _force_exit_event = asyncio.Event()

    def _double_signal_handler() -> None:
        global _shutdown_in_progress
        if _shutdown_in_progress:
            logger.critical(
                "Second signal received during shutdown — skipping flatten, force-exiting"
            )
            if _force_exit_event is not None:
                _force_exit_event.set()
            return
        strategy._handle_signal()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _double_signal_handler)
    loop.add_signal_handler(signal.SIGHUP, strategy._handle_reload)


# ------------------------------------------------------------------
# Task cancellation
# ------------------------------------------------------------------

async def cancel_tasks(tasks: list[asyncio.Task]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ------------------------------------------------------------------
# Service lifecycle
# ------------------------------------------------------------------

async def stop_services(ctx: RuntimeContext) -> None:
    await ctx.metrics.stop()
    await ctx.order_mgr.stop_rate_limiter()
    await ctx.account_stream.stop()
    await ctx.ob_mgr.stop()
    await ctx.order_mgr.wait_for_inflight(timeout_s=5.0)
    await ctx.trading_client.close()


# ------------------------------------------------------------------
# State persistence helpers
# ------------------------------------------------------------------

def _write_json_state(directory: str, market_name: str, data: dict) -> Path | None:
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


def _write_emergency_state(ctx: RuntimeContext, reason: str) -> None:
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
    bid = ctx.ob_mgr.best_bid()
    ask = ctx.ob_mgr.best_ask()
    if bid is not None and ask is not None and bid.price > 0 and ask.price > 0:
        return (bid.price + ask.price) / 2
    if bid is not None and bid.price > 0:
        return bid.price
    if ask is not None and ask.price > 0:
        return ask.price
    return None


# ------------------------------------------------------------------
# Progressive slippage
# ------------------------------------------------------------------

def compute_progressive_slippage(
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

async def attempt_shutdown_flatten(
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

        slippage = compute_progressive_slippage(
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

    # Fresh REST verification after flatten loop
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

async def shutdown_core(ctx: RuntimeContext, tasks: list[asyncio.Task]) -> None:
    """Core shutdown logic: cancel tasks/orders, flatten, record stats."""
    global _shutdown_in_progress
    _shutdown_in_progress = True

    logger.info("Shutting down — cancelling tasks, orders, and flattening position...")
    ctx.strategy._set_runtime_mode("shutdown")

    shutdown_reason = ctx.strategy.shutdown_reason
    _write_pre_shutdown_state(ctx, shutdown_reason)

    await cancel_tasks(tasks)
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
        flatten = await attempt_shutdown_flatten(ctx, shutdown_reason)
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
    await stop_services(ctx)
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


async def shutdown_and_record(ctx: RuntimeContext, tasks: list[asyncio.Task]) -> None:
    """Wrap shutdown in a hard timeout. On timeout, write emergency state and force exit."""
    timeout_s = ctx.settings.shutdown_timeout_s
    try:
        await asyncio.wait_for(shutdown_core(ctx, tasks), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.critical(
            "SHUTDOWN TIMEOUT: shutdown sequence exceeded %ss for market=%s — force exiting",
            timeout_s,
            ctx.settings.market_name,
        )
        _write_emergency_state(ctx, reason="shutdown_timeout")
        try:
            ctx.journal.close()
        except Exception:
            pass
        try:
            await stop_services(ctx)
        except Exception:
            pass
        os._exit(1)


def reset_module_state() -> None:
    """Reset module-level state (for test isolation)."""
    global _shutdown_in_progress, _force_exit_event
    _shutdown_in_progress = False
    _force_exit_event = None
