from __future__ import annotations

import logging
import time
from collections import deque
from decimal import Decimal
from typing import Optional

from .account_stream import FillEvent

logger = logging.getLogger(__name__)

# Maximum number of trade IDs to retain for deduplication.
_SEEN_TRADE_IDS_MAX = 10_000


def on_fill(strategy, fill: FillEvent) -> None:
    """Record fill telemetry and reset adaptive POF state for matched levels.

    Skips duplicate fills identified by ``trade_id`` to prevent double-counting
    when the account stream delivers the same trade event more than once.
    """
    # --- Deduplication ---
    seen: deque = strategy._seen_trade_ids
    if fill.trade_id in strategy._seen_trade_ids_set:
        logger.warning(
            "Duplicate fill ignored: trade_id=%s side=%s qty=%s",
            fill.trade_id,
            fill.side,
            fill.qty,
        )
        return

    seen.append(fill.trade_id)
    strategy._seen_trade_ids_set.add(fill.trade_id)
    # Cap set size by evicting oldest entries
    while len(seen) > _SEEN_TRADE_IDS_MAX:
        evicted = seen.popleft()
        strategy._seen_trade_ids_set.discard(evicted)

    # --- Normal fill processing ---
    bid = strategy._ob.best_bid()
    ask = strategy._ob.best_ask()
    market_snapshot = strategy._ob.market_snapshot(
        depth=strategy._settings.fill_snapshot_depth,
        micro_vol_window_s=strategy._settings.micro_vol_window_s,
        micro_drift_window_s=strategy._settings.micro_drift_window_s,
        imbalance_window_s=strategy._settings.imbalance_window_s,
    )

    order_info = strategy._orders.find_order_by_exchange_id(str(fill.order_id))
    level = order_info.level if order_info is not None else None
    if order_info is not None:
        key = (str(order_info.side), order_info.level)
        strategy._reset_pof_state(key)

    strategy._journal.record_fill(
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
        position=strategy._risk.get_current_position(),
        market_snapshot=market_snapshot,
    )


def on_level_freed(
    strategy,
    side_value: str,
    level: int,
    external_id: str,
    *,
    rejected: bool = False,
    status: Optional[str] = None,
    reason: Optional[str] = None,
    price: Optional[Decimal] = None,
) -> None:
    """Handle terminal order states and cooldown bookkeeping."""
    key = (side_value, level)
    current = strategy._level_ext_ids.get(key)
    status_upper = str(status or "").upper()
    side_for_journal = strategy._normalise_side(side_value)
    order_info = strategy._orders.find_order_by_external_id(external_id)
    exchange_id = order_info.exchange_order_id if order_info is not None else None
    if current == external_id:
        strategy._clear_level_slot(key)
    cancel_reason = strategy._pending_cancel_reasons.pop(
        external_id,
        reason or "terminal",
    )

    if rejected:
        strategy._journal.record_rejection(
            external_id=external_id,
            exchange_id=exchange_id,
            side=side_for_journal,
            price=price if price is not None else Decimal("0"),
            reason=reason or "REJECTED",
        )
    elif status_upper in {"CANCELLED", "EXPIRED"}:
        strategy._journal.record_order_cancelled(
            external_id=external_id,
            exchange_id=exchange_id,
            side=side_for_journal,
            level=level,
            reason=cancel_reason,
        )

    # Apply POF cooldown to prevent immediate retry storms.
    if (
        rejected
        and strategy._settings.pof_cooldown_s > 0
        and (reason is None or "POST_ONLY_FAILED" in str(reason).upper())
    ):
        if strategy._settings.adaptive_pof_enabled:
            strategy._apply_adaptive_pof_reject(key)
        else:
            strategy._level_pof_until[key] = time.monotonic() + strategy._settings.pof_cooldown_s
