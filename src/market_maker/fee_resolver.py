from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

try:
    from x10.perpetual.fees import DEFAULT_FEES, TradingFeeModel
except Exception:  # pragma: no cover - tests may stub x10 modules partially
    @dataclass(frozen=True)
    class TradingFeeModel:  # type: ignore[no-redef]
        market: str = "UNKNOWN"
        maker_fee_rate: Decimal = Decimal("0")
        taker_fee_rate: Decimal = Decimal("0")
        builder_fee_rate: Decimal = Decimal("0")

    DEFAULT_FEES = TradingFeeModel()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedFeeConfig:
    """Resolved order fee parameters for one placement."""

    max_fee_rate: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    builder_fee_rate: Decimal
    builder_id: Optional[int]


class FeeResolver:
    """Fetch and cache account fee rates and derive per-order fee caps."""

    def __init__(
        self,
        trading_client,
        market_name: str,
        *,
        refresh_interval_s: float = 60.0,
        builder_program_enabled: bool = False,
        builder_id: Optional[int] = None,
        configured_builder_fee_rate: Decimal = Decimal("0"),
    ) -> None:
        self._client = trading_client
        self._market_name = market_name
        self._refresh_interval_s = max(1.0, float(refresh_interval_s))
        self._builder_program_enabled = bool(builder_program_enabled)
        self._builder_id = builder_id if self._builder_program_enabled else None
        self._configured_builder_fee_rate = Decimal(str(configured_builder_fee_rate))

        self._lock = asyncio.Lock()
        self._cached: Optional[TradingFeeModel] = None
        self._updated_at_mono: float = 0.0

    @property
    def has_cache(self) -> bool:
        return self._cached is not None

    def cache_age_s(self) -> Optional[float]:
        if self._updated_at_mono <= 0:
            return None
        return max(0.0, time.monotonic() - self._updated_at_mono)

    def _is_stale(self) -> bool:
        age = self.cache_age_s()
        if age is None:
            return True
        return age > self._refresh_interval_s

    async def refresh(self, *, force: bool = False) -> Optional[TradingFeeModel]:
        if not force and self._cached is not None and not self._is_stale():
            return self._cached

        async with self._lock:
            if not force and self._cached is not None and not self._is_stale():
                return self._cached

            query_builder_id = self._builder_id if self._builder_program_enabled else None
            try:
                response = await self._client.account.get_fees(
                    market_names=[self._market_name],
                    builder_id=query_builder_id,
                )
                rows = response.data if hasattr(response, "data") else response
                selected: Optional[TradingFeeModel] = None
                for row in rows or []:
                    if getattr(row, "market", None) == self._market_name:
                        selected = row
                        break
                if selected is None and rows:
                    selected = rows[0]

                if selected is None:
                    logger.warning(
                        "FeeResolver: no fee rows returned for market=%s",
                        self._market_name,
                    )
                    return self._cached

                self._cached = selected
                self._updated_at_mono = time.monotonic()
                logger.debug(
                    "FeeResolver updated for %s: maker=%s taker=%s builder=%s",
                    self._market_name,
                    selected.maker_fee_rate,
                    selected.taker_fee_rate,
                    selected.builder_fee_rate,
                )
                return self._cached
            except Exception as exc:
                logger.warning("FeeResolver refresh failed for %s: %s", self._market_name, exc)
                return self._cached

    async def validate_builder_config(self) -> None:
        if not self._builder_program_enabled:
            return
        fee_model = await self.refresh(force=True)
        if fee_model is None:
            raise ValueError(
                "builder program enabled but fee schedule unavailable; cannot validate builder fee"
            )
        if self._configured_builder_fee_rate > fee_model.builder_fee_rate:
            raise ValueError(
                f"configured builder fee rate {self._configured_builder_fee_rate} exceeds allowed {fee_model.builder_fee_rate}"
            )
        if self._builder_id is None:
            raise ValueError("builder program enabled but builder_id is not set")

    async def resolve_order_fees(
        self,
        *,
        post_only: bool,
        fail_closed: bool = True,
    ) -> Optional[ResolvedFeeConfig]:
        fee_model = await self.refresh(force=False)
        if fee_model is None:
            if post_only and fail_closed:
                return None
            fee_model = DEFAULT_FEES

        max_fee_rate = fee_model.maker_fee_rate if post_only else fee_model.taker_fee_rate

        if self._builder_program_enabled:
            builder_fee_rate = self._configured_builder_fee_rate
            builder_id = self._builder_id
            if builder_id is None:
                if post_only and fail_closed:
                    return None
                builder_fee_rate = Decimal("0")
        else:
            builder_fee_rate = Decimal("0")
            builder_id = None

        return ResolvedFeeConfig(
            max_fee_rate=max_fee_rate,
            maker_fee_rate=fee_model.maker_fee_rate,
            taker_fee_rate=fee_model.taker_fee_rate,
            builder_fee_rate=builder_fee_rate,
            builder_id=builder_id,
        )
