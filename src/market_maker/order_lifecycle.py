"""
Order Lifecycle Operations

Extracted from ``OrderManager``: place_order, cancel_order, cancel_all_orders,
and sweep_pending_cancels.  These are the core order lifecycle operations
that interact with the exchange API.
"""
from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional

from x10.perpetual.orders import OrderSide, OrderType, TimeInForce

if TYPE_CHECKING:
    from .order_manager import OrderManager

logger = logging.getLogger(__name__)

try:
    from x10.utils.http import RateLimitException as _RateLimitException
    if not isinstance(_RateLimitException, type) or not issubclass(_RateLimitException, BaseException):
        raise TypeError
    RateLimitException = _RateLimitException
except Exception:
    class RateLimitException(Exception):  # type: ignore[no-redef]
        pass

# Default timeout for pending-cancel orders before force-removal.
_PENDING_CANCEL_TIMEOUT_S = 10.0


async def place_order(
    mgr: OrderManager,
    side: OrderSide,
    price: Decimal,
    size: Decimal,
    level: int,
) -> Optional[str]:
    """Place a post-only limit order via the manager's trading client.

    Returns the external_id on success, None on failure.
    """
    if mgr.in_maintenance:
        logger.debug(
            "Skipping order placement during maintenance (%.0fs remaining)",
            mgr.maintenance_remaining_s,
        )
        return None
    if mgr.in_rate_limit_halt:
        logger.warning(
            "Skipping order placement during rate-limit halt (%.0fs remaining)",
            max(0.0, mgr._rate_state._rate_limit_halt_until - time.monotonic()),
        )
        return None

    fee_cfg = None
    if mgr._fee_resolver is not None:
        fee_cfg = await mgr._fee_resolver.resolve_order_fees(
            post_only=True, fail_closed=True,
        )
        if fee_cfg is None:
            logger.error(
                "Skipping post-only placement for %s: unable to resolve maker max-fee rate",
                mgr._market_name,
            )
            return None

    if not await mgr._rate_state.acquire_rate_token(timeout=2.0):
        logger.warning(
            "Rate limiter timeout: order placement throttled for market=%s",
            mgr._market_name,
        )
        return None

    external_id = mgr._generate_external_id()

    # Lazy import to avoid circular dependency.
    from .order_manager import OrderInfo

    pending_info = OrderInfo(
        external_id=external_id, side=side, price=price, size=size, level=level,
    )
    mgr._pending_placements[external_id] = pending_info
    mgr._latency.record_send(external_id)

    mgr._record_attempt()
    mgr._begin_inflight()
    try:
        resp = await mgr._client.place_order(
            market_name=mgr._market_name,
            amount_of_synthetic=size,
            price=price,
            side=side,
            order_type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTT,
            post_only=True,
            external_id=external_id,
            max_fee_rate=fee_cfg.max_fee_rate if fee_cfg is not None else None,
            builder_fee_rate=fee_cfg.builder_fee_rate if fee_cfg is not None else None,
            builder_id=fee_cfg.builder_id if fee_cfg is not None else None,
        )

        if mgr._check_maintenance_response(resp=resp):
            mgr._pending_placements.pop(external_id, None)
            mgr._latency.discard(external_id)
            asyncio.create_task(mgr._maintenance_cancel_orders())
            return None

        if hasattr(resp, "status") and hasattr(resp, "error"):
            if resp.status != "OK" or resp.error is not None:
                mgr.consecutive_failures += 1
                mgr._record_failure()
                mgr._pending_placements.pop(external_id, None)
                mgr._latency.discard(external_id)
                side_name = getattr(side, "value", str(side))
                logger.warning(
                    "Order rejected: side=%s price=%s error=%s failures=%d",
                    side_name, price, resp.error, mgr.consecutive_failures,
                )
                return None

        exchange_id = mgr._extract_exchange_id(resp)

        if external_id in mgr._active_orders:
            info = mgr._active_orders[external_id]
            if exchange_id is not None and info.exchange_order_id is None:
                info.exchange_order_id = exchange_id
                mgr._orders_by_exchange_id[exchange_id] = info
        else:
            info = mgr._pending_placements.pop(external_id, pending_info)
            if exchange_id is not None:
                info.exchange_order_id = exchange_id
            mgr._active_orders[external_id] = info
            if exchange_id is not None:
                mgr._orders_by_exchange_id[exchange_id] = info

        mgr._recent_orders_by_external_id[external_id] = info
        mgr._pending_placements.pop(external_id, None)
        mgr.consecutive_failures = 0

        logger.info(
            "Order placed: side=%s price=%s size=%s level=%d ext_id=%s exch_id=%s",
            side, price, size, level, external_id, exchange_id,
        )
        return external_id

    except RateLimitException as exc:
        should_halt = mgr._record_rate_limit_hit()
        mgr._pending_placements.pop(external_id, None)
        mgr._latency.discard(external_id)
        if should_halt:
            asyncio.create_task(mgr.cancel_all_orders())
        logger.error(
            "Rate-limited placing order: side=%s price=%s size=%s level=%d error=%s",
            side, price, size, level, exc,
        )
        return None
    except Exception as exc:
        if mgr._check_maintenance_response(exc=exc):
            mgr._pending_placements.pop(external_id, None)
            mgr._latency.discard(external_id)
            asyncio.create_task(mgr._maintenance_cancel_orders())
            return None

        mgr.consecutive_failures += 1
        mgr._record_failure()
        mgr._pending_placements.pop(external_id, None)
        mgr._latency.discard(external_id)
        logger.error(
            "Failed to place order: side=%s price=%s size=%s level=%d error=%s",
            side, price, size, level, exc,
        )
        return None
    finally:
        mgr._end_inflight()


async def cancel_order(mgr: Any, external_id: str) -> bool:
    """Cancel a single order by its external_id."""
    info = mgr._active_orders.get(external_id)
    if info is None:
        return False

    mgr._begin_inflight()
    try:
        await mgr._client.orders.cancel_order_by_external_id(
            order_external_id=external_id,
        )
        logger.info(
            "Order cancel requested: ext_id=%s exchange_id=%s",
            external_id, info.exchange_order_id,
        )
        return True
    except RateLimitException as exc:
        mgr._record_rate_limit_hit()
        logger.warning(
            "Cancel rate-limited for ext_id=%s exchange_id=%s: %s",
            external_id, info.exchange_order_id, exc,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Cancel failed for ext_id=%s exchange_id=%s: %s",
            external_id, info.exchange_order_id, exc,
        )
        return False
    finally:
        mgr._end_inflight()


async def cancel_all_orders(mgr: Any) -> None:
    """Cancel every order for this market using mass-cancel."""
    mgr._begin_inflight()
    try:
        now = time.monotonic()
        for ext_id in list(mgr._active_orders):
            mgr._pending_cancel[ext_id] = now

        await mgr._client.orders.mass_cancel(markets=[mgr._market_name])
        logger.info(
            "Mass cancel issued for market=%s (%d orders marked pending_cancel)",
            mgr._market_name, len(mgr._pending_cancel),
        )
    except RateLimitException as exc:
        should_halt = mgr._record_rate_limit_hit()
        logger.error(
            "Mass cancel rate-limited for market=%s: %s", mgr._market_name, exc,
        )
        if should_halt:
            mgr._rate_state._rate_limit_halt_until = max(
                mgr._rate_state._rate_limit_halt_until,
                time.monotonic() + mgr._rate_state._rate_limit_halt_s,
            )
    except Exception as exc:
        logger.error(
            "Mass cancel failed for market=%s: %s — falling back to per-order cancel",
            mgr._market_name, exc,
        )
        mgr._pending_cancel.clear()
        ext_ids = list(mgr._active_orders.keys())
        for ext_id in ext_ids:
            await cancel_order(mgr, ext_id)
    finally:
        mgr._end_inflight()


def sweep_pending_cancels(mgr: Any, timeout_s: float = _PENDING_CANCEL_TIMEOUT_S) -> int:
    """Force-remove orders stuck in pending_cancel beyond *timeout_s*."""
    now = time.monotonic()
    removed = 0
    for ext_id in list(mgr._pending_cancel):
        cancel_ts = mgr._pending_cancel[ext_id]
        if (now - cancel_ts) < timeout_s:
            continue
        info = mgr._active_orders.pop(ext_id, None)
        mgr._pending_cancel.pop(ext_id, None)
        if info is not None:
            mgr._recent_orders_by_external_id[ext_id] = info
            logger.warning(
                "Force-removed pending_cancel order after %.1fs: "
                "ext_id=%s exchange_id=%s level=%d",
                now - cancel_ts, ext_id, info.exchange_order_id, info.level,
            )
            for cb in mgr._level_freed_callbacks:
                try:
                    cb(
                        str(info.side), info.level, info.external_id,
                        rejected=False, status="CANCELLED",
                        reason="sweep_pending_cancel_timeout", price=info.price,
                    )
                except Exception as exc:
                    logger.error("level_freed callback error during sweep: %s", exc)
            removed += 1
    return removed
