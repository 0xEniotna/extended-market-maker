"""
Funding Manager

Encapsulates funding-rate fetching and funding-bias computation that was
previously inlined in MarketMakerStrategy.  The strategy now delegates
to this class for all funding-related state.

The funding bias adjusts quote prices to capture positive funding when
long/short positioning aligns with the rate direction.
"""
from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)

_FUNDING_REFRESH_INTERVAL_S = 300.0


class FundingManager:
    """Manages funding rate state and computes funding bias for pricing."""

    def __init__(
        self,
        market_profile: str,
        funding_bias_enabled: bool,
        funding_inventory_weight: Decimal,
        funding_bias_cap_bps: Decimal,
    ) -> None:
        self._market_profile = market_profile
        self._funding_bias_enabled = funding_bias_enabled
        self._funding_inventory_weight = funding_inventory_weight
        self._funding_bias_cap_bps = funding_bias_cap_bps
        self._funding_rate = Decimal("0")

    @property
    def funding_rate(self) -> Decimal:
        return self._funding_rate

    def set_funding_rate(self, funding_rate: Decimal) -> None:
        """Safely set the current funding rate, ignoring invalid values."""
        try:
            value = Decimal(str(funding_rate))
        except Exception:
            return
        if not value.is_finite():
            return
        self._funding_rate = value

    def funding_bias_bps(self) -> Decimal:
        """Compute funding bias offset in basis points for pricing skew."""
        if self._market_profile != "crypto":
            return Decimal("0")
        if not self._funding_bias_enabled:
            return Decimal("0")
        raw_bps = (
            self._funding_rate
            * Decimal("10000")
            * self._funding_inventory_weight
        )
        cap = max(Decimal("0"), self._funding_bias_cap_bps)
        if cap <= 0:
            return raw_bps
        return max(-cap, min(cap, raw_bps))

    def update_settings(
        self,
        *,
        market_profile: str,
        funding_bias_enabled: bool,
        funding_inventory_weight: Decimal,
        funding_bias_cap_bps: Decimal,
    ) -> None:
        """Update settings after a hot-reload."""
        self._market_profile = market_profile
        self._funding_bias_enabled = funding_bias_enabled
        self._funding_inventory_weight = funding_inventory_weight
        self._funding_bias_cap_bps = funding_bias_cap_bps

    async def refresh_task(
        self,
        client: object,
        market_name: str,
        shutdown_event: asyncio.Event,
    ) -> None:
        """Async task that periodically refreshes the funding rate from the exchange."""
        while not shutdown_event.is_set():
            try:
                if (
                    self._market_profile == "crypto"
                    and self._funding_bias_enabled
                ):
                    markets = await client.markets_info.get_markets_dict()  # type: ignore[attr-defined]
                    market_info = markets.get(market_name)
                    if market_info is not None:
                        self.set_funding_rate(
                            Decimal(str(market_info.market_stats.funding_rate))
                        )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Funding refresh failed", exc_info=True)
            await asyncio.sleep(_FUNDING_REFRESH_INTERVAL_S)
