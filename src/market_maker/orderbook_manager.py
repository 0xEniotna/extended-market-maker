"""
Orderbook Manager

Wraps the SDK's OrderBook WebSocket subscription and exposes best bid/ask
with asyncio.Condition notifications for the strategy to react to price
changes efficiently.

Uses native SDK callbacks for instant price-change notification (no polling).
Includes staleness detection — if no update arrives within a configurable
window the cached prices are treated as stale and return None.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from x10.perpetual.configuration import EndpointConfig
from x10.perpetual.orderbook import OrderBook

logger = logging.getLogger(__name__)

# If no orderbook update arrives within this many seconds, data is stale.
_DEFAULT_STALENESS_THRESHOLD_S = 15.0
_DEFAULT_STALE_LOG_INTERVAL_S = 5.0
_MID_HISTORY_MAX_AGE_S = 120.0
# Auto-reconnect after this many seconds of staleness.
_DEFAULT_RECONNECT_AFTER_S = 60.0
_RECONNECT_BACKOFF_BASE_S = 5.0
_RECONNECT_BACKOFF_MAX_S = 120.0


@dataclass(frozen=True)
class PriceLevel:
    """Snapshot of a single price level."""

    price: Decimal
    size: Decimal


class OrderbookManager:
    """
    Manages a single-market orderbook subscription and notifies consumers
    when the best bid or ask changes.

    Uses the SDK's native ``best_bid_change_callback`` /
    ``best_ask_change_callback`` for zero-latency notification instead of
    a polling loop.
    """

    def __init__(
        self,
        endpoint_config: EndpointConfig,
        market_name: str,
        *,
        depth: Optional[int] = None,
        staleness_threshold_s: float = _DEFAULT_STALENESS_THRESHOLD_S,
    ) -> None:
        self._config = endpoint_config
        self._market_name = market_name
        self._depth = depth
        self._staleness_threshold_s = staleness_threshold_s

        self._orderbook: Optional[OrderBook] = None

        # Conditions the strategy waits on
        self.best_bid_condition: asyncio.Condition = asyncio.Condition()
        self.best_ask_condition: asyncio.Condition = asyncio.Condition()

        # Cached latest values
        self._last_bid: Optional[PriceLevel] = None
        self._last_ask: Optional[PriceLevel] = None

        # Timestamp of most recent orderbook update (monotonic clock)
        self._last_update_ts: float = 0.0

        # EMA-smoothed spread in basis points (avoids noisy offset computation)
        self._spread_ema_bps: Optional[Decimal] = None
        self._spread_ema_alpha: Decimal = Decimal("0.15")  # ~7-sample half-life

        # Mid-price history for short-horizon toxicity metrics.
        self._mid_history = deque()
        self._mid_history_max_age_s = _MID_HISTORY_MAX_AGE_S
        self._imbalance_history = deque()
        self._imbalance_history_max_age_s = _MID_HISTORY_MAX_AGE_S

        # Staleness warning throttling.
        self._last_stale_log_ts: float = 0.0
        self._stale_log_interval_s: float = _DEFAULT_STALE_LOG_INTERVAL_S
        self._was_stale: bool = False

        # Auto-reconnect state.
        self._reconnect_after_s: float = _DEFAULT_RECONNECT_AFTER_S
        self._reconnect_task: Optional[asyncio.Task] = None
        self._reconnect_attempts: int = 0
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create and start the orderbook subscription with native callbacks."""
        self._stopped = False
        await self._create_orderbook()
        # Start the watchdog that auto-reconnects on prolonged staleness.
        self._reconnect_task = asyncio.create_task(
            self._reconnect_watchdog(),
            name=f"ob-reconnect-{self._market_name}",
        )

    async def _create_orderbook(self) -> None:
        """Internal: create (or recreate) the orderbook WS subscription."""
        self._orderbook = await OrderBook.create(
            endpoint_config=self._config,
            market_name=self._market_name,
            best_bid_change_callback=self._on_bid_change,
            best_ask_change_callback=self._on_ask_change,
            start=True,
            depth=self._depth,
        )
        logger.info(
            "Orderbook manager started for %s (depth=%s)",
            self._market_name,
            self._depth or "full",
        )

    async def stop(self) -> None:
        """Close the orderbook subscription."""
        self._stopped = True
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reconnect_task = None
        if self._orderbook is not None:
            try:
                await self._orderbook.close()
            except Exception as exc:
                logger.warning(
                    "Error closing orderbook for %s: %s", self._market_name, exc
                )
            self._orderbook = None

        logger.info("Orderbook manager stopped for %s", self._market_name)

    def best_bid(self) -> Optional[PriceLevel]:
        """Return the cached best bid, or None if data is stale."""
        if self._is_stale():
            return None
        return self._last_bid

    def best_ask(self) -> Optional[PriceLevel]:
        """Return the cached best ask, or None if data is stale."""
        if self._is_stale():
            return None
        return self._last_ask

    def has_data(self) -> bool:
        """Return True if at least one bid and one ask have been received.

        Unlike ``best_bid()``/``best_ask()`` this does **not** apply the
        staleness check — it is meant for the initial startup wait where
        we just need to know the first snapshot has arrived.
        """
        return self._last_bid is not None and self._last_ask is not None

    def is_stale(self) -> bool:
        """Return True if orderbook data is stale (without warning logs)."""
        return self._is_stale(log_warning=False)

    def seconds_since_update(self) -> Optional[float]:
        """Return seconds since last update, or None if no data was received."""
        if self._last_update_ts == 0.0:
            return None
        return max(0.0, time.monotonic() - self._last_update_ts)

    def spread_bps(self) -> Optional[Decimal]:
        """Return the current (raw) spread in basis points, or None if unavailable."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is None or ask is None or bid.price <= 0:
            return None
        mid = (bid.price + ask.price) / Decimal("2")
        if mid == 0:
            return None
        return (ask.price - bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self) -> Optional[Decimal]:
        """Return the EMA-smoothed spread in basis points.

        Smoothing avoids noisy offset computation from transient spread spikes.
        Falls back to the raw spread if no EMA has been computed yet.
        """
        if self._spread_ema_bps is not None:
            return self._spread_ema_bps
        return self.spread_bps()

    def micro_volatility_bps(self, window_s: float) -> Optional[Decimal]:
        """Return high-low range in bps over *window_s* seconds."""
        samples = self._mid_window(window_s)
        if len(samples) < 2:
            return None
        mids = [mid for _, mid in samples]
        latest_mid = mids[-1]
        if latest_mid <= 0:
            return None
        high = max(mids)
        low = min(mids)
        return (high - low) / latest_mid * Decimal("10000")

    def micro_drift_bps(self, window_s: float) -> Optional[Decimal]:
        """Return signed drift in bps over *window_s* seconds."""
        samples = self._mid_window(window_s)
        if len(samples) < 2:
            return None
        first_mid = samples[0][1]
        last_mid = samples[-1][1]
        if first_mid <= 0:
            return None
        return (last_mid - first_mid) / first_mid * Decimal("10000")

    def orderbook_imbalance(self, window_s: float) -> Optional[Decimal]:
        """Return smoothed top-of-book imbalance over *window_s* seconds.

        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        in [-1, +1]. Positive means bid-dominant.
        """
        samples = self._imbalance_window(window_s)
        if not samples:
            return None
        values = [imb for _, imb in samples]
        return sum(values) / Decimal(len(values))

    def market_snapshot(
        self,
        depth: int = 5,
        *,
        micro_vol_window_s: float = 5.0,
        micro_drift_window_s: float = 3.0,
        imbalance_window_s: float = 2.0,
    ) -> Dict[str, Any]:
        """Return a point-in-time market snapshot for journaling.

        Includes BBO/mid/spread, top-of-book depth, microstructure metrics,
        and staleness/update-lag state. Never raises.
        """
        depth = max(1, int(depth))
        bid = self._last_bid
        ask = self._last_ask
        is_stale = self.is_stale()

        mid: Optional[Decimal] = None
        spread_bps: Optional[Decimal] = None
        if bid is not None and ask is not None and bid.price > 0 and ask.price > 0:
            mid = (bid.price + ask.price) / Decimal("2")
            if mid > 0:
                spread_bps = (ask.price - bid.price) / mid * Decimal("10000")

        return {
            "best_bid": bid.price if bid else None,
            "best_ask": ask.price if ask else None,
            "mid": mid,
            "spread_bps": spread_bps,
            "bids_top": self._book_levels("bid", depth),
            "asks_top": self._book_levels("ask", depth),
            "micro_vol_bps": self.micro_volatility_bps(window_s=micro_vol_window_s),
            "micro_drift_bps": self.micro_drift_bps(window_s=micro_drift_window_s),
            "imbalance": self.orderbook_imbalance(window_s=imbalance_window_s),
            "is_stale": is_stale,
            "seconds_since_update": self.seconds_since_update(),
            "depth": depth,
        }

    # ------------------------------------------------------------------
    # Auto-reconnect watchdog
    # ------------------------------------------------------------------

    async def _reconnect_watchdog(self) -> None:
        """Periodically check staleness and reconnect the WS if needed."""
        while not self._stopped:
            try:
                await asyncio.sleep(5.0)
                if self._stopped:
                    return

                elapsed = self.seconds_since_update()
                if elapsed is None or elapsed < self._reconnect_after_s:
                    # Data is fresh (or never received yet during startup).
                    if elapsed is not None and elapsed < self._staleness_threshold_s:
                        self._reconnect_attempts = 0
                    continue

                # Prolonged staleness — attempt reconnect.
                self._reconnect_attempts += 1
                backoff = min(
                    _RECONNECT_BACKOFF_BASE_S * (2 ** (self._reconnect_attempts - 1)),
                    _RECONNECT_BACKOFF_MAX_S,
                )
                logger.warning(
                    "Orderbook stale for %.0fs — reconnecting %s (attempt %d, backoff %.0fs)",
                    elapsed,
                    self._market_name,
                    self._reconnect_attempts,
                    backoff,
                )

                # Close old subscription.
                if self._orderbook is not None:
                    try:
                        await self._orderbook.close()
                    except Exception as exc:
                        logger.warning("Error closing stale orderbook: %s", exc)
                    self._orderbook = None

                # Wait before reconnecting (exponential backoff).
                await asyncio.sleep(backoff)
                if self._stopped:
                    return

                # Recreate the subscription.
                try:
                    await self._create_orderbook()
                    logger.info(
                        "Orderbook reconnected for %s (attempt %d)",
                        self._market_name,
                        self._reconnect_attempts,
                    )
                except Exception as exc:
                    logger.error(
                        "Orderbook reconnect failed for %s: %s",
                        self._market_name,
                        exc,
                    )

            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Reconnect watchdog error: %s", exc, exc_info=True)
                await asyncio.sleep(10.0)

    # ------------------------------------------------------------------
    # SDK callbacks (called from the OrderBook WS task)
    # ------------------------------------------------------------------

    async def _on_bid_change(self, raw_bid) -> None:
        """Called by the SDK when the best bid changes.

        Wrapped in a try/except so an error here never kills the SDK's
        background orderbook task.
        """
        try:
            new_bid = self._to_price_level(raw_bid)
            if new_bid is None:
                return
            self._last_bid = new_bid
            self._last_update_ts = time.monotonic()
            self._update_spread_ema()
            self._record_mid()
            self._record_imbalance()
            if self._was_stale:
                self._was_stale = False
                logger.info("Orderbook data fresh again for %s", self._market_name)
            async with self.best_bid_condition:
                self.best_bid_condition.notify_all()
        except Exception as exc:
            logger.error("Error in bid callback: %s", exc, exc_info=True)

    async def _on_ask_change(self, raw_ask) -> None:
        """Called by the SDK when the best ask changes.

        Wrapped in a try/except so an error here never kills the SDK's
        background orderbook task.
        """
        try:
            new_ask = self._to_price_level(raw_ask)
            if new_ask is None:
                return
            self._last_ask = new_ask
            self._last_update_ts = time.monotonic()
            self._update_spread_ema()
            self._record_mid()
            self._record_imbalance()
            if self._was_stale:
                self._was_stale = False
                logger.info("Orderbook data fresh again for %s", self._market_name)
            async with self.best_ask_condition:
                self.best_ask_condition.notify_all()
        except Exception as exc:
            logger.error("Error in ask callback: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_spread_ema(self) -> None:
        """Recompute the EMA-smoothed spread after a bid/ask change."""
        if self._last_bid is None or self._last_ask is None:
            return
        if self._last_bid.price <= 0 or self._last_ask.price <= 0:
            return
        mid = (self._last_bid.price + self._last_ask.price) / Decimal("2")
        if mid == 0:
            return
        raw_spread = (self._last_ask.price - self._last_bid.price) / mid * Decimal("10000")
        if raw_spread < 0:
            return  # Crossed book, ignore
        alpha = self._spread_ema_alpha
        if self._spread_ema_bps is None:
            self._spread_ema_bps = raw_spread
        else:
            self._spread_ema_bps = alpha * raw_spread + (Decimal("1") - alpha) * self._spread_ema_bps

    def _record_mid(self) -> None:
        """Append the current mid price to history and prune old points."""
        if self._last_bid is None or self._last_ask is None:
            return
        if self._last_bid.price <= 0 or self._last_ask.price <= 0:
            return
        now = time.monotonic()
        mid = (self._last_bid.price + self._last_ask.price) / Decimal("2")
        self._mid_history.append((now, mid))
        self._prune_mid_history(now)

    def _prune_mid_history(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._mid_history_max_age_s
        while self._mid_history and self._mid_history[0][0] < cutoff:
            self._mid_history.popleft()

    def _mid_window(self, window_s: float):
        if window_s <= 0:
            return []
        now = time.monotonic()
        self._prune_mid_history(now)
        cutoff = now - float(window_s)
        return [(ts, mid) for ts, mid in self._mid_history if ts >= cutoff]

    def _record_imbalance(self) -> None:
        """Append current top-of-book imbalance and prune old points."""
        if self._last_bid is None or self._last_ask is None:
            return
        bid_size = self._last_bid.size
        ask_size = self._last_ask.size
        denom = bid_size + ask_size
        if denom <= 0:
            return
        now = time.monotonic()
        imbalance = (bid_size - ask_size) / denom
        self._imbalance_history.append((now, imbalance))
        self._prune_imbalance_history(now)

    def _prune_imbalance_history(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.monotonic()
        cutoff = now - self._imbalance_history_max_age_s
        while self._imbalance_history and self._imbalance_history[0][0] < cutoff:
            self._imbalance_history.popleft()

    def _imbalance_window(self, window_s: float):
        if window_s <= 0:
            return []
        now = time.monotonic()
        self._prune_imbalance_history(now)
        cutoff = now - float(window_s)
        return [(ts, imb) for ts, imb in self._imbalance_history if ts >= cutoff]

    def _book_levels(self, side: str, depth: int) -> List[Dict[str, Decimal]]:
        """Return top *depth* levels from the in-memory orderbook cache."""
        if self._orderbook is None:
            return []
        if side == "bid":
            raw = getattr(self._orderbook, "_bid_prices", None)
            if raw is None:
                return []
            items = list(raw.items())[-depth:]
            items = list(reversed(items))
        else:
            raw = getattr(self._orderbook, "_ask_prices", None)
            if raw is None:
                return []
            items = list(raw.items())[:depth]

        levels: List[Dict[str, Decimal]] = []
        for price, entry in items:
            amount = getattr(entry, "amount", getattr(entry, "size", Decimal("0")))
            levels.append({
                "price": Decimal(str(price)),
                "size": Decimal(str(amount)),
            })
        return levels

    def _is_stale(self, *, log_warning: bool = True) -> bool:
        """Return True if data is older than the staleness threshold."""
        if self._last_update_ts == 0.0:
            return True  # Never received an update
        elapsed = time.monotonic() - self._last_update_ts
        if elapsed > self._staleness_threshold_s:
            self._was_stale = True
            if log_warning:
                now = time.monotonic()
                if (now - self._last_stale_log_ts) >= self._stale_log_interval_s:
                    logger.warning(
                        "Orderbook data stale for %s (%.1fs since last update)",
                        self._market_name,
                        elapsed,
                    )
                    self._last_stale_log_ts = now
            return True
        return False

    @staticmethod
    def _to_price_level(raw) -> Optional[PriceLevel]:
        """Convert whatever the SDK returns into our PriceLevel."""
        if raw is None:
            return None
        # The SDK may return an object with .price and .size attributes,
        # or just a Decimal price.  Handle both.
        if isinstance(raw, (int, float, Decimal)):
            return PriceLevel(price=Decimal(str(raw)), size=Decimal("0"))
        price = getattr(raw, "price", None)
        size = getattr(raw, "size", getattr(raw, "amount", Decimal("0")))
        if price is None:
            return None
        return PriceLevel(price=Decimal(str(price)), size=Decimal(str(size)))
