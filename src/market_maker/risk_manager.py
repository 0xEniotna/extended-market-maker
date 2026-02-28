"""
Risk Manager

Standalone position/risk tracking for the market making strategy.
Enforces MAX_POSITION_SIZE before every order.

Supports both periodic REST-based refresh and instant updates from the
account WebSocket stream via ``handle_position_update()``.
"""
from __future__ import annotations

import logging
import time
from decimal import ROUND_DOWN, Decimal
from typing import Dict, Optional

from x10.perpetual.orders import OrderSide
from x10.perpetual.positions import PositionModel, PositionSide, PositionStatus
from x10.perpetual.trading_client import PerpetualTradingClient

from .risk_sizing import allowed_order_size as _allowed_order_size

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Queries the current position for the configured market and enforces
    the maximum position size limit.
    """

    def __init__(
        self,
        trading_client: PerpetualTradingClient,
        market_name: str,
        max_position_size: Decimal,
        max_order_notional_usd: Decimal = Decimal("0"),
        max_position_notional_usd: Decimal = Decimal("0"),
        *,
        gross_exposure_limit_usd: Decimal = Decimal("0"),
        max_long_position_size: Decimal = Decimal("0"),
        max_short_position_size: Decimal = Decimal("0"),
        balance_aware_sizing_enabled: bool = True,
        balance_usage_factor: Decimal = Decimal("0.95"),
        balance_notional_multiplier: Decimal = Decimal("1.0"),
        balance_min_available_usd: Decimal = Decimal("0"),
        balance_staleness_max_s: float = 30.0,
        balance_stale_action: str = "reduce",
        orderbook_mgr=None,
    ) -> None:
        self._client = trading_client
        self._market_name = market_name
        self._max_position_size = max_position_size
        self._max_long_position_size = max_long_position_size
        self._max_short_position_size = max_short_position_size
        self._max_order_notional_usd = max_order_notional_usd
        self._max_position_notional_usd = max_position_notional_usd
        self._gross_exposure_limit_usd = gross_exposure_limit_usd
        self._balance_aware_sizing_enabled = balance_aware_sizing_enabled
        self._balance_usage_factor = max(Decimal("0"), balance_usage_factor)
        self._balance_notional_multiplier = max(Decimal("0"), balance_notional_multiplier)
        self._balance_min_available_usd = max(Decimal("0"), balance_min_available_usd)
        self._balance_staleness_max_s = max(0.0, float(balance_staleness_max_s))
        self._balance_stale_action = balance_stale_action if balance_stale_action in ("skip", "reduce", "halt") else "reduce"
        self._orderbook_mgr = orderbook_mgr
        self._cached_position: Decimal = Decimal("0")
        self._cached_realized_pnl: Decimal = Decimal("0")
        self._cached_unrealized_pnl: Decimal = Decimal("0")
        self._cached_total_position_pnl: Decimal = Decimal("0")
        # Cumulative session P&L: survives position close/reset cycles.
        self._session_realized_pnl: Decimal = Decimal("0")
        self._cached_available_for_trade: Decimal | None = None
        self._cached_equity: Decimal | None = None
        self._cached_initial_margin: Decimal | None = None
        self._cached_balance_updated_at: float | None = None
        self._cached_mark_price: Decimal | None = None
        self._cached_liquidation_price: Decimal | None = None
        self._cached_position_updated_at: float | None = None

        # In-flight order notional tracker.
        # Keyed by external_id → notional (size * price) reserved at submission.
        self._inflight_notional: Dict[str, Decimal] = {}

    # ------------------------------------------------------------------
    # In-flight order tracking
    # ------------------------------------------------------------------

    def reserve_inflight(self, external_id: str, notional: Decimal) -> None:
        """Reserve notional for an in-flight order at submission time."""
        self._inflight_notional[external_id] = max(Decimal("0"), notional)

    def release_inflight(self, external_id: str) -> None:
        """Release notional reservation on terminal status."""
        self._inflight_notional.pop(external_id, None)

    def total_inflight_notional(self) -> Decimal:
        """Return sum of all in-flight reserved notionals."""
        return sum(self._inflight_notional.values(), Decimal("0"))

    # ------------------------------------------------------------------
    # Account-stream integration
    # ------------------------------------------------------------------

    def handle_position_update(self, pos: PositionModel) -> None:
        """Called by AccountStreamManager on every position update.

        Instantly updates the cached position so risk checks are always
        based on the latest data, not the 10-second poll.
        """
        if pos.market != self._market_name:
            return

        if pos.status == PositionStatus.CLOSED:
            self._cached_position = Decimal("0")
            self._reset_position_pnl()
            self._cached_mark_price = None
            self._cached_liquidation_price = None
        else:
            sign = Decimal("-1") if pos.side == PositionSide.SHORT else Decimal("1")
            self._cached_position = pos.size * sign
            self._update_position_pnl(
                realized=getattr(pos, "realised_pnl", Decimal("0")),
                unrealized=getattr(pos, "unrealised_pnl", Decimal("0")),
            )
            mark_price = getattr(pos, "mark_price", None)
            self._cached_mark_price = (
                Decimal(str(mark_price))
                if mark_price is not None
                else self._cached_mark_price
            )
            liq_price = getattr(pos, "liquidation_price", None)
            self._cached_liquidation_price = (
                Decimal(str(liq_price))
                if liq_price is not None
                else None
            )
        self._cached_position_updated_at = time.monotonic()

        logger.debug(
            "Position updated (stream): %s %s (raw side=%s size=%s pnl=%s)",
            self._market_name,
            self._cached_position,
            pos.side,
            pos.size,
            self._cached_total_position_pnl,
        )

    def handle_balance_update(self, balance) -> None:
        """Called by AccountStreamManager when balance/equity updates arrive."""
        self._update_balance_cache(balance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def refresh_position(self) -> Decimal:
        """Fetch current position size from exchange and cache it.

        Returns signed size (positive = long, negative = short).
        """
        try:
            resp = await self._client.account.get_positions()
            data = resp.data if hasattr(resp, "data") else resp

            for pos in data or []:
                market = getattr(pos, "market", None) or (
                    pos.get("market") if isinstance(pos, dict) else None
                )
                if market != self._market_name:
                    continue

                size_abs = Decimal(str(getattr(pos, "size", "0")))
                side = str(getattr(pos, "side", "")).upper()
                sign = -1 if "SHORT" in side else 1
                self._cached_position = size_abs * sign
                realized = getattr(pos, "realised_pnl", None)
                unrealized = getattr(pos, "unrealised_pnl", None)
                if realized is None and isinstance(pos, dict):
                    realized = pos.get("realised_pnl", "0")
                if unrealized is None and isinstance(pos, dict):
                    unrealized = pos.get("unrealised_pnl", "0")
                self._update_position_pnl(
                    realized=realized if realized is not None else Decimal("0"),
                    unrealized=unrealized if unrealized is not None else Decimal("0"),
                )
                mark_price = getattr(pos, "mark_price", None)
                if mark_price is None and isinstance(pos, dict):
                    mark_price = pos.get("mark_price")
                self._cached_mark_price = (
                    Decimal(str(mark_price))
                    if mark_price is not None
                    else self._cached_mark_price
                )
                liq_price = getattr(pos, "liquidation_price", None)
                if liq_price is None and isinstance(pos, dict):
                    liq_price = pos.get("liquidation_price")
                self._cached_liquidation_price = (
                    Decimal(str(liq_price))
                    if liq_price is not None
                    else None
                )
                self._cached_position_updated_at = time.monotonic()
                return self._cached_position

            # No position found for this market
            self._cached_position = Decimal("0")
            self._reset_position_pnl()
            self._cached_mark_price = None
            self._cached_liquidation_price = None
            self._cached_position_updated_at = time.monotonic()
            return self._cached_position
        except Exception as exc:
            logger.error("Failed to fetch position: %s", exc)
            return self._cached_position

    async def refresh_balance(self) -> Decimal | None:
        """Fetch account collateral balance and cache available_for_trade."""
        try:
            resp = await self._client.account.get_balance()
            data = resp.data if hasattr(resp, "data") else resp
            if data is not None:
                self._update_balance_cache(data)
            return self._cached_available_for_trade
        except Exception as exc:
            logger.debug("Failed to fetch balance: %s", exc)
            return self._cached_available_for_trade

    def get_current_position(self) -> Decimal:
        """Return the last cached position."""
        return self._cached_position

    def get_position_realized_pnl(self) -> Decimal:
        """Return last cached realised PnL for this market position."""
        return self._cached_realized_pnl

    def get_position_unrealized_pnl(self) -> Decimal:
        """Return last cached unrealised PnL for this market position."""
        return self._cached_unrealized_pnl

    def get_position_total_pnl(self) -> Decimal:
        """Return last cached realised+unrealised PnL for this market position."""
        return self._cached_total_position_pnl

    def get_session_pnl(self) -> Decimal:
        """Return cumulative session P&L (survives position close/reset cycles).

        This is the sum of all realized P&L accumulated across every position
        lifecycle in this process, plus the current position's total
        (realized + unrealized) P&L.
        """
        return self._session_realized_pnl + self._cached_total_position_pnl

    def get_available_for_trade(self) -> Decimal | None:
        """Return the last cached available-for-trade collateral."""
        return self._cached_available_for_trade

    def get_equity(self) -> Decimal | None:
        """Return cached account equity."""
        return self._cached_equity

    def get_initial_margin(self) -> Decimal | None:
        """Return cached account initial margin."""
        return self._cached_initial_margin

    def available_balance_ratio(self) -> Decimal | None:
        """Return available_for_trade / equity if both are known and equity > 0."""
        if (
            self._cached_available_for_trade is None
            or self._cached_equity is None
            or self._cached_equity <= 0
        ):
            return None
        return self._cached_available_for_trade / self._cached_equity

    def margin_utilization(self) -> Decimal | None:
        """Return initial_margin / equity if both are known and equity > 0."""
        if (
            self._cached_initial_margin is None
            or self._cached_equity is None
            or self._cached_equity <= 0
        ):
            return None
        return self._cached_initial_margin / self._cached_equity

    def get_mark_price(self) -> Decimal | None:
        """Return cached mark price for the tracked market position."""
        return self._cached_mark_price

    def get_liquidation_price(self) -> Decimal | None:
        """Return cached liquidation price for the tracked market position."""
        return self._cached_liquidation_price

    def liquidation_distance_bps(self) -> Decimal | None:
        """Return absolute distance (bps) between mark and liquidation prices."""
        if self._cached_position == 0:
            return None
        if (
            self._cached_mark_price is None
            or self._cached_mark_price <= 0
            or self._cached_liquidation_price is None
            or self._cached_liquidation_price <= 0
        ):
            return None
        return (
            abs(self._cached_mark_price - self._cached_liquidation_price)
            / self._cached_mark_price
            * Decimal("10000")
        )

    def margin_snapshot(self) -> Dict[str, Decimal | None]:
        """Return a lightweight margin/liquidation state snapshot."""
        return {
            "available_for_trade": self._cached_available_for_trade,
            "equity": self._cached_equity,
            "initial_margin": self._cached_initial_margin,
            "available_ratio": self.available_balance_ratio(),
            "margin_utilization": self.margin_utilization(),
            "mark_price": self._cached_mark_price,
            "liquidation_price": self._cached_liquidation_price,
            "liq_distance_bps": self.liquidation_distance_bps(),
        }

    def allowed_order_size(
        self,
        side: OrderSide,
        requested_size: Decimal,
        reference_price: Decimal,
        reserved_same_side_qty: Decimal = Decimal("0"),
        reserved_open_notional_usd: Decimal = Decimal("0"),
    ) -> Decimal:
        return _allowed_order_size(
            self, side, requested_size, reference_price,
            reserved_same_side_qty=reserved_same_side_qty,
            reserved_open_notional_usd=reserved_open_notional_usd,
        )

    def can_place_order(self, side: OrderSide, size: Decimal) -> bool:
        """Check whether placing *size* on *side* would breach the position limit.

        A BUY increases position; a SELL decreases it.
        """
        allowed = self.allowed_order_size(
            side=side,
            requested_size=size,
            reference_price=Decimal("0"),
        )
        within_limit = allowed >= size
        if not within_limit:
            if side == OrderSide.BUY:
                projected = self._cached_position + size
            else:
                projected = self._cached_position - size
            logger.warning(
                "Position limit breach: current=%s side=%s size=%s projected=%s max=%s",
                self._cached_position,
                side,
                size,
                projected,
                self._max_position_size,
            )
        return within_limit

    def _update_balance_cache(self, balance) -> None:
        """Normalize and cache account balance/equity fields."""
        available_for_trade = getattr(balance, "available_for_trade", None)
        equity = getattr(balance, "equity", None)
        initial_margin = getattr(balance, "initial_margin", None)

        if available_for_trade is not None:
            self._cached_available_for_trade = Decimal(str(available_for_trade))
            self._cached_balance_updated_at = time.monotonic()
        if equity is not None:
            self._cached_equity = Decimal(str(equity))
        if initial_margin is not None:
            self._cached_initial_margin = Decimal(str(initial_margin))

    def _update_position_pnl(self, *, realized, unrealized) -> None:
        realized_dec = Decimal(str(realized))
        unrealized_dec = Decimal(str(unrealized))
        self._cached_realized_pnl = realized_dec
        self._cached_unrealized_pnl = unrealized_dec
        self._cached_total_position_pnl = realized_dec + unrealized_dec

    def _reset_position_pnl(self) -> None:
        # Accumulate realized P&L into session total before zeroing.
        self._session_realized_pnl += self._cached_realized_pnl
        self._cached_realized_pnl = Decimal("0")
        self._cached_unrealized_pnl = Decimal("0")
        self._cached_total_position_pnl = Decimal("0")

    @staticmethod
    def _is_buy_side(side: OrderSide) -> bool:
        side_name = str(side).upper()
        return side_name == "BUY" or side_name.endswith("BUY")

    @classmethod
    def _split_reducing_and_opening_qty(
        cls,
        *,
        side: OrderSide,
        current_position: Decimal,
        size: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Split order size into reducing and position-increasing components."""
        size = max(Decimal("0"), size)
        if size == 0 or current_position == 0:
            return Decimal("0"), size

        is_buy = cls._is_buy_side(side)
        is_reducing = (
            (current_position < 0 and is_buy)
            or (current_position > 0 and not is_buy)
        )
        if not is_reducing:
            return Decimal("0"), size

        reducing_qty = min(abs(current_position), size)
        opening_qty = max(Decimal("0"), size - reducing_qty)
        return reducing_qty, opening_qty
