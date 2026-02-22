from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional


@dataclass(frozen=True)
class RegimeState:
    regime: str = "NORMAL"
    vol_short_bps: Optional[Decimal] = None
    vol_medium_bps: Optional[Decimal] = None
    drift_short_bps: Optional[Decimal] = None
    offset_scale: Decimal = Decimal("1")
    pause: bool = False


@dataclass(frozen=True)
class TrendState:
    direction: str = "NEUTRAL"
    strength: Decimal = Decimal("0")


@dataclass(frozen=True)
class GuardDecision:
    allow: bool
    reason: str
    extra_offset_bps: Decimal = Decimal("0")
    pause_until_ts: Optional[float] = None


@dataclass(frozen=True)
class PricingContext:
    side: Any
    level: int
    best_price: Decimal
    regime: RegimeState
    trend: TrendState
    inventory_norm: Decimal
    funding_bias_bps: Decimal = Decimal("0")


@dataclass(frozen=True)
class RepriceMarketContext:
    regime: RegimeState
    trend: TrendState
    min_reprice_interval_s: float
    max_order_age_s: float
    funding_bias_bps: Decimal
    inventory_band: str
