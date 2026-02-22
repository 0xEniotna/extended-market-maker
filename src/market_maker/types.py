from __future__ import annotations

from decimal import Decimal
from typing import Optional, Protocol, runtime_checkable

LevelKey = tuple[str, int]


@runtime_checkable
class OffsetModeLike(Protocol):
    value: str


@runtime_checkable
class PriceLevelLike(Protocol):
    price: Decimal
    size: Decimal


@runtime_checkable
class OrderbookLike(Protocol):
    def spread_bps_ema(self) -> Optional[Decimal]: ...
    def best_bid(self) -> Optional[object]: ...
    def best_ask(self) -> Optional[object]: ...
    def micro_volatility_bps(self, window_s: float) -> Optional[Decimal]: ...
    def micro_drift_bps(self, window_s: float) -> Optional[Decimal]: ...
    def mid_prices(self, window_s: float) -> list[Decimal]: ...


@runtime_checkable
class RiskManagerLike(Protocol):
    def get_current_position(self) -> Decimal: ...


@runtime_checkable
class OrderInfoLike(Protocol):
    side: object
    price: Decimal
    size: Decimal
    level: int
    exchange_order_id: Optional[str]


@runtime_checkable
class PricingSettingsLike(Protocol):
    offset_mode: object
    spread_multiplier: Decimal
    min_offset_bps: Decimal
    max_offset_bps: Decimal
    price_offset_per_level_percent: Decimal
    max_position_size: Decimal
    inventory_hard_pct: Decimal
    inventory_critical_pct: Decimal
    inventory_warn_pct: Decimal
    inventory_deadband_pct: Decimal
    skew_shape_k: Decimal
    skew_max_bps: Decimal
    inventory_skew_factor: Decimal
    trend_skew_boost: Decimal
    market_profile: object
    size_scale_per_level: Decimal


@runtime_checkable
class PostOnlySettingsLike(Protocol):
    post_only_safety_ticks: int
    adaptive_pof_enabled: bool
    pof_cooldown_s: float
    pof_streak_reset_s: float
    pof_max_safety_ticks: int
    pof_backoff_multiplier: Decimal
