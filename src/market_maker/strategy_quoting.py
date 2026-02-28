"""
Strategy Quoting Logic

Per-level quoting lifecycle extracted from ``MarketMakerStrategy``:
level_task loop, reprice orchestration, cancel-with-barrier,
reprice decision journaling, and halt/age/markout helpers.
"""
from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Optional

from x10.perpetual.orders import OrderSide

from .decision_models import RegimeState, RepriceMarketContext, TrendState
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)

# Exception messages/types that indicate an irrecoverable error.
_FATAL_EXCEPTION_PATTERNS = frozenset({
    "authentication",
    "unauthorized",
    "forbidden",
    "api key",
    "invalid key",
    "permission denied",
    "market not found",
    "market delisted",
    "account suspended",
    "account disabled",
})


def normalise_side(side_value: str) -> str:
    side_upper = side_value.upper()
    if "BUY" in side_upper:
        return "BUY"
    if "SELL" in side_upper:
        return "SELL"
    return side_value


def is_fatal_exception(exc: BaseException) -> bool:
    """Return True if the exception indicates an irrecoverable error."""
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    for pattern in _FATAL_EXCEPTION_PATTERNS:
        if pattern in msg or pattern in exc_type:
            return True
    return False


async def level_task(s: Any, side: OrderSide, level: int) -> None:
    """Continuously quote on one (side, level) slot."""
    key = (str(side), level)
    s._clear_level_slot(key)

    condition = (
        s._ob.best_bid_condition
        if side == OrderSide.BUY
        else s._ob.best_ask_condition
    )

    while not s._shutdown_event.is_set():
        sync_quote_halt_state(s)
        if s._circuit_open:
            await asyncio.sleep(1.0)
            continue
        if s._quote_halt_reasons:
            await asyncio.sleep(0.2)
            continue
        if s._level_cancel_pending_ext_id.get(key) is not None:
            await asyncio.sleep(0.1)
            continue

        try:
            async with condition:
                await asyncio.wait_for(condition.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            return

        try:
            await maybe_reprice(s, side, level)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(
                "Error in level task %s L%d: %s", side, level, exc,
                exc_info=True,
            )
            s._journal.record_error(
                component=f"level_task_{side}_L{level}",
                exception_type=type(exc).__name__,
                message=str(exc),
                stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                stack_trace=TradeJournal.format_stack_trace(exc),
            )
            if is_fatal_exception(exc):
                logger.critical(
                    "FATAL error in level task %s L%d — initiating shutdown: %s",
                    side, level, exc,
                )
                s._shutdown_event.set()
                return
            await asyncio.sleep(1.0)


async def maybe_reprice(s: Any, side: OrderSide, level: int) -> None:
    sync_quote_halt_state(s)
    if s._quote_halt_reasons:
        return
    market_ctx = build_reprice_market_context(s)
    await s._reprice.evaluate(s, side, level, market_ctx=market_ctx)


def build_reprice_market_context(s: Any) -> RepriceMarketContext:
    if s._settings.market_profile == "crypto":
        regime = s._volatility.evaluate()
        trend = s._trend_signal.evaluate()
    else:
        regime = RegimeState(regime="NORMAL")
        trend = TrendState()
    min_interval, max_order_age_s = s._volatility.cadence(regime)
    rate_limit_multiplier = getattr(s._orders, "rate_limit_reprice_multiplier", Decimal("1"))
    if not isinstance(rate_limit_multiplier, Decimal):
        rate_limit_multiplier = Decimal("1")
    min_interval *= float(rate_limit_multiplier)
    return RepriceMarketContext(
        regime=regime,
        trend=trend,
        min_reprice_interval_s=min_interval,
        max_order_age_s=max_order_age_s,
        funding_bias_bps=s._funding_bias_bps(),
        inventory_band=s._pricing.inventory_band(),
    )


def record_reprice_decision(s: Any, **kwargs: Any) -> None:
    if not s._settings.journal_reprice_decisions:
        return
    side = kwargs.get("side")
    if side is not None:
        kwargs["side"] = normalise_side(str(side))
    s._journal.record_reprice_decision(**kwargs)


async def cancel_level_order(
    s: Any,
    *,
    key: tuple[str, int],
    external_id: str,
    side: OrderSide,
    level: int,
    reason: str,
) -> bool:
    """Request cancel for a level order and store a structured reason.

    Returns True when the level slot can be safely freed.
    """
    _ = (side, level)
    pending_ext = s._level_cancel_pending_ext_id.get(key)
    if pending_ext == external_id:
        return False
    s._pending_cancel_reasons[external_id] = reason
    ok = await s._orders.cancel_order(external_id)
    if ok:
        s._level_cancel_pending_ext_id[key] = external_id
        return False

    if s._orders.find_order_by_external_id(external_id) is not None:
        if s._orders.get_active_order(external_id) is None:
            s._clear_level_slot(key)
            return True

    s._pending_cancel_reasons.pop(external_id, None)
    return False


def on_adverse_markout_widen(s: Any, key: tuple[str, int], reason: str) -> None:
    """Callback from FillQualityTracker when a level has adverse markout."""
    base_ticks = max(1, int(s._settings.post_only_safety_ticks))
    max_ticks = max(base_ticks, int(s._settings.pof_max_safety_ticks))
    current = s._post_only.dynamic_safety_ticks.get(key, base_ticks)
    new_ticks = min(max_ticks, current + 1)
    s._post_only.dynamic_safety_ticks[key] = new_ticks
    logger.warning(
        "Adverse markout widen for %s: safety_ticks %d -> %d (reason=%s)",
        key, current, new_ticks, reason,
    )


async def on_stream_desync(s: Any, reason: str) -> None:
    s._halt_mgr.set_halt("stream_desync")
    s._journal.record_exchange_event(
        event_type="stream_desync",
        details={"reason": reason},
    )
    if s._orders.active_order_count() > 0:
        try:
            await s._orders.cancel_all_orders()
        except Exception:
            logger.debug("stream desync cancel-all failed", exc_info=True)


def sync_quote_halt_state(s: Any) -> None:
    rate_limit_halt = getattr(s._orders, "in_rate_limit_halt", False)
    if not isinstance(rate_limit_halt, bool):
        rate_limit_halt = False
    s._halt_mgr.sync_state(
        rate_limit_halt=rate_limit_halt,
        streams_healthy=s._streams_healthy(),
    )


def order_age_exceeded(
    s: Any,
    key: tuple[str, int],
    *,
    max_age_s: Optional[float] = None,
) -> bool:
    """Return True if the tracked order at *key* exceeded max_order_age_s."""
    if max_age_s is None:
        max_age_s = s._settings.max_order_age_s
    if max_age_s <= 0:
        return False
    placed_ts = s._level_order_created_at.get(key)
    if placed_ts is None:
        return False
    return bool((time.monotonic() - placed_ts) > max_age_s)
