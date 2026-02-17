"""
Risk Manager

Standalone position/risk tracking for the market making strategy.
Enforces MAX_POSITION_SIZE before every order.

Supports both periodic REST-based refresh and instant updates from the
account WebSocket stream via ``handle_position_update()``.
"""
from __future__ import annotations

import logging
from decimal import Decimal

from x10.perpetual.orders import OrderSide
from x10.perpetual.positions import PositionModel, PositionSide, PositionStatus
from x10.perpetual.trading_client import PerpetualTradingClient

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
        balance_aware_sizing_enabled: bool = True,
        balance_usage_factor: Decimal = Decimal("0.95"),
        balance_notional_multiplier: Decimal = Decimal("1.0"),
        balance_min_available_usd: Decimal = Decimal("0"),
    ) -> None:
        self._client = trading_client
        self._market_name = market_name
        self._max_position_size = max_position_size
        self._max_order_notional_usd = max_order_notional_usd
        self._max_position_notional_usd = max_position_notional_usd
        self._balance_aware_sizing_enabled = balance_aware_sizing_enabled
        self._balance_usage_factor = max(Decimal("0"), balance_usage_factor)
        self._balance_notional_multiplier = max(Decimal("0"), balance_notional_multiplier)
        self._balance_min_available_usd = max(Decimal("0"), balance_min_available_usd)
        self._cached_position: Decimal = Decimal("0")
        self._cached_realized_pnl: Decimal = Decimal("0")
        self._cached_unrealized_pnl: Decimal = Decimal("0")
        self._cached_total_position_pnl: Decimal = Decimal("0")
        self._cached_available_for_trade: Decimal | None = None
        self._cached_equity: Decimal | None = None
        self._cached_initial_margin: Decimal | None = None

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
        else:
            sign = Decimal("-1") if pos.side == PositionSide.SHORT else Decimal("1")
            self._cached_position = pos.size * sign
            self._update_position_pnl(
                realized=getattr(pos, "realised_pnl", Decimal("0")),
                unrealized=getattr(pos, "unrealised_pnl", Decimal("0")),
            )

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
                return self._cached_position

            # No position found for this market
            self._cached_position = Decimal("0")
            self._reset_position_pnl()
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

    def get_available_for_trade(self) -> Decimal | None:
        """Return the last cached available-for-trade collateral."""
        return self._cached_available_for_trade

    def allowed_order_size(
        self,
        side: OrderSide,
        requested_size: Decimal,
        reference_price: Decimal,
        reserved_same_side_qty: Decimal = Decimal("0"),
        reserved_open_notional_usd: Decimal = Decimal("0"),
    ) -> Decimal:
        """Return the maximum safe size that can be placed for this order.

        Clips by:
        - contract position size limit (``max_position_size``)
        - per-order notional cap (``max_order_notional_usd``)
        - total position notional cap (``max_position_notional_usd``)
        - reserved same-side resting quantity headroom
        - account available_for_trade headroom (when enabled)
        """
        if requested_size <= 0:
            return Decimal("0")

        clipped = requested_size
        current = self._cached_position
        reserved_same_side_qty = max(Decimal("0"), reserved_same_side_qty)
        reserved_open_notional_usd = max(Decimal("0"), reserved_open_notional_usd)

        # Quantity headroom from max position size.
        if self._max_position_size > 0:
            if side == OrderSide.BUY:
                qty_headroom = self._max_position_size - current - reserved_same_side_qty
            else:
                qty_headroom = self._max_position_size + current - reserved_same_side_qty
            clipped = min(clipped, max(Decimal("0"), qty_headroom))

        if reference_price > 0:
            # Per-order notional cap.
            if self._max_order_notional_usd > 0:
                per_order_max_size = self._max_order_notional_usd / reference_price
                clipped = min(clipped, max(Decimal("0"), per_order_max_size))

            # Absolute position notional cap.
            if self._max_position_notional_usd > 0:
                current_notional = abs(current) * reference_price
                reserved_notional = reserved_same_side_qty * reference_price
                remaining_notional = (
                    self._max_position_notional_usd
                    - current_notional
                    - reserved_notional
                )
                if remaining_notional <= 0:
                    clipped = Decimal("0")
                else:
                    max_size_from_pos_notional = remaining_notional / reference_price
                    clipped = min(
                        clipped,
                        max(Decimal("0"), max_size_from_pos_notional),
                    )

            reducing_qty, opening_qty = self._split_reducing_and_opening_qty(
                side=side,
                current_position=current,
                size=clipped,
            )
            if (
                self._balance_aware_sizing_enabled
                and self._cached_available_for_trade is not None
                and opening_qty > 0
            ):
                balance_headroom = (
                    self._cached_available_for_trade * self._balance_usage_factor
                ) - self._balance_min_available_usd
                notional_headroom = (
                    balance_headroom * self._balance_notional_multiplier
                ) - reserved_open_notional_usd
                if notional_headroom <= 0:
                    opening_qty = Decimal("0")
                else:
                    max_size_from_balance = notional_headroom / reference_price
                    opening_qty = min(
                        opening_qty,
                        max(Decimal("0"), max_size_from_balance),
                    )
                clipped = reducing_qty + opening_qty

        clipped = max(Decimal("0"), clipped)
        if clipped < requested_size:
            reducing_qty, opening_qty = self._split_reducing_and_opening_qty(
                side=side,
                current_position=current,
                size=clipped,
            )
            logger.info(
                "Order size clipped: side=%s requested=%s allowed=%s reducing=%s opening=%s "
                "reserved_qty=%s reserved_notional=%s avail_for_trade=%s ref_price=%s",
                side,
                requested_size,
                clipped,
                reducing_qty,
                opening_qty,
                reserved_same_side_qty,
                reserved_open_notional_usd,
                self._cached_available_for_trade,
                reference_price,
            )
        return clipped

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
