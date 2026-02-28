"""
Per-Session P&L Attribution

Decomposes real-time P&L into actionable components:
- **Spread capture**: half-spread earned on each maker fill pair
- **Inventory P&L**: mark-to-market on held inventory
- **Fee P&L**: maker rebates minus taker fees
- **Funding P&L**: funding payments received/paid

Without attribution, you cannot determine whether losses come from
adverse selection (widen spreads), inventory risk (reduce limits),
or fees (change tier).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, Optional

# Max fill records to keep for attribution (rolling window).
_MAX_FILL_RECORDS = 5000


@dataclass(frozen=True)
class PnLSnapshot:
    """Point-in-time P&L attribution breakdown."""

    spread_capture_usd: Decimal = Decimal("0")
    inventory_pnl_usd: Decimal = Decimal("0")
    fee_pnl_usd: Decimal = Decimal("0")
    funding_pnl_usd: Decimal = Decimal("0")
    total_usd: Decimal = Decimal("0")
    fill_count: int = 0
    buy_fill_count: int = 0
    sell_fill_count: int = 0
    maker_fill_count: int = 0
    taker_fill_count: int = 0
    avg_spread_capture_bps: Decimal = Decimal("0")
    total_volume_usd: Decimal = Decimal("0")


@dataclass
class _FillRecord:
    """Internal fill record for attribution."""

    ts: float
    side: str  # "BUY" or "SELL"
    price: Decimal
    qty: Decimal
    fee: Decimal
    is_taker: bool
    mid_at_fill: Optional[Decimal]


class PnLAttributionTracker:
    """Real-time per-session P&L attribution.

    Tracks spread capture, inventory mark-to-market, fees, and funding
    across the session lifetime.

    Thread-safety: not thread-safe. All calls should come from the
    asyncio event loop (single-threaded).
    """

    def __init__(self) -> None:
        self._fills: deque[_FillRecord] = deque(maxlen=_MAX_FILL_RECORDS)

        # Cumulative counters.
        self._spread_capture_usd = Decimal("0")
        self._fee_pnl_usd = Decimal("0")
        self._funding_pnl_usd = Decimal("0")
        self._total_volume_usd = Decimal("0")
        self._buy_fill_count = 0
        self._sell_fill_count = 0
        self._maker_fill_count = 0
        self._taker_fill_count = 0

        # Inventory tracking for mark-to-market.
        self._net_position = Decimal("0")
        self._position_cost_basis = Decimal("0")  # weighted avg entry price × qty

        # Spread capture tracking: per-fill edge from mid.
        self._total_edge_bps_weighted = Decimal("0")  # sum(edge_bps × notional)

    def record_fill(
        self,
        *,
        side: str,
        price: Decimal,
        qty: Decimal,
        fee: Decimal,
        is_taker: bool,
        mid_price: Optional[Decimal] = None,
    ) -> None:
        """Record a fill and update attribution components."""
        notional = price * qty
        self._total_volume_usd += notional

        record = _FillRecord(
            ts=time.time(),
            side=side,
            price=price,
            qty=qty,
            fee=fee,
            is_taker=is_taker,
            mid_at_fill=mid_price,
        )
        self._fills.append(record)

        # --- Count fills ---
        if side == "BUY":
            self._buy_fill_count += 1
        else:
            self._sell_fill_count += 1
        if is_taker:
            self._taker_fill_count += 1
        else:
            self._maker_fill_count += 1

        # --- Fee P&L: negative = cost, positive = rebate ---
        # Convention: fee is always positive in the fill event; maker rebates
        # are negative fees. We negate here so that rebates add to P&L.
        self._fee_pnl_usd -= fee

        # --- Spread capture: edge from mid at fill time ---
        if mid_price is not None and mid_price > 0:
            if side == "BUY":
                edge_bps = (mid_price - price) / mid_price * Decimal("10000")
            else:
                edge_bps = (price - mid_price) / mid_price * Decimal("10000")
            edge_usd = edge_bps * notional / Decimal("10000")
            self._spread_capture_usd += edge_usd
            self._total_edge_bps_weighted += edge_bps * notional

        # --- Inventory cost basis (FIFO-like weighted average) ---
        signed_qty = qty if side == "BUY" else -qty
        old_position = self._net_position

        if old_position == 0 or (old_position > 0 and side == "BUY") or (old_position < 0 and side == "SELL"):
            # Increasing position: update cost basis
            self._position_cost_basis += price * qty
            self._net_position += signed_qty
        else:
            # Reducing position: realize P&L against cost basis
            if abs(old_position) > 0:
                avg_entry = abs(self._position_cost_basis / old_position)
            else:
                avg_entry = price
            reduce_qty = min(qty, abs(old_position))

            # Reduce cost basis proportionally
            if abs(old_position) > 0:
                self._position_cost_basis -= self._position_cost_basis * reduce_qty / abs(old_position)

            self._net_position += signed_qty

            # If we crossed zero, the remainder opens a new position
            if abs(signed_qty) > reduce_qty:
                opening_qty = abs(signed_qty) - reduce_qty
                self._position_cost_basis += price * opening_qty

    def record_funding_payment(self, amount_usd: Decimal) -> None:
        """Record a funding payment (positive = received, negative = paid)."""
        self._funding_pnl_usd += amount_usd

    def update_mark_price(self, mark_price: Decimal) -> None:
        """Update the mark price for inventory mark-to-market.

        The inventory P&L is computed on-demand in snapshot() using the
        last known mark price. This avoids storing a price field.
        """
        self._last_mark_price = mark_price

    def snapshot(self, current_mid: Optional[Decimal] = None) -> PnLSnapshot:
        """Compute the current P&L attribution snapshot."""
        # Inventory P&L: mark-to-market unrealized.
        inventory_pnl = Decimal("0")
        mid = current_mid or getattr(self, "_last_mark_price", None)
        if mid is not None and mid > 0 and self._net_position != 0:
            market_value = self._net_position * mid
            inventory_pnl = market_value - self._position_cost_basis

        total = self._spread_capture_usd + inventory_pnl + self._fee_pnl_usd + self._funding_pnl_usd

        # Average spread capture in bps.
        avg_spread_bps = Decimal("0")
        if self._total_volume_usd > 0:
            avg_spread_bps = self._total_edge_bps_weighted / self._total_volume_usd

        fill_count = self._buy_fill_count + self._sell_fill_count

        return PnLSnapshot(
            spread_capture_usd=self._spread_capture_usd,
            inventory_pnl_usd=inventory_pnl,
            fee_pnl_usd=self._fee_pnl_usd,
            funding_pnl_usd=self._funding_pnl_usd,
            total_usd=total,
            fill_count=fill_count,
            buy_fill_count=self._buy_fill_count,
            sell_fill_count=self._sell_fill_count,
            maker_fill_count=self._maker_fill_count,
            taker_fill_count=self._taker_fill_count,
            avg_spread_capture_bps=avg_spread_bps,
            total_volume_usd=self._total_volume_usd,
        )
