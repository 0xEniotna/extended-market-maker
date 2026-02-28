from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, Optional, Protocol, runtime_checkable

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


# ---------------------------------------------------------------------------
# StrategyContext — interface that RepricePipeline depends on instead of
# the concrete MarketMakerStrategy class.  This decouples the pipeline
# from the god-object strategy and makes the dependency explicit.
# ---------------------------------------------------------------------------

class _RepricePipelineSettingsLike(Protocol):
    """Settings subset needed by RepricePipeline."""
    market_profile: object
    trend_cancel_counter_on_strong: bool
    trend_strong_threshold: Decimal
    trend_counter_side_size_cut: Decimal
    cancel_on_stale_book: bool
    stale_cancel_grace_s: float
    imbalance_window_s: float


class _OrderManagerLike(Protocol):
    """Order manager subset needed by RepricePipeline."""
    rate_limit_extra_offset_bps: Decimal
    rate_limit_reprice_multiplier: Decimal
    in_rate_limit_degraded: bool

    def get_active_order(self, ext_id: Optional[str]) -> Optional[OrderInfoLike]: ...
    def reserved_exposure(
        self, *, side: object, exclude_external_id: Optional[str],
    ) -> tuple[Decimal, Decimal]: ...
    async def place_order(
        self, *, side: object, price: Decimal, size: Decimal, level: int,
    ) -> Optional[str]: ...


class _JournalLike(Protocol):
    """Trade journal subset needed by RepricePipeline."""
    def record_order_placed(self, **kwargs: Any) -> None: ...


class StrategyContext(Protocol):
    """Structural interface for the strategy object as consumed by
    ``RepricePipeline``.

    ``MarketMakerStrategy`` satisfies this protocol without explicit
    subclassing.  New consumers (back-test harnesses, strategy variants)
    only need to implement this contract to reuse the pipeline.
    """

    # --- Composed objects ---
    _settings: _RepricePipelineSettingsLike
    _ob: OrderbookLike
    _orders: _OrderManagerLike
    _risk: RiskManagerLike
    _guards: Any  # GuardPolicy
    _volatility: Any  # VolatilityRegime
    _trend_signal: Any  # TrendSignal
    _pricing: Any  # PricingEngine
    _journal: _JournalLike
    _market_min_order_size: Decimal

    # --- Per-slot state dicts ---
    _level_ext_ids: Dict[LevelKey, Optional[str]]
    _level_order_created_at: Dict[LevelKey, Optional[float]]
    _level_last_reprice_at: Dict[LevelKey, float]
    _level_stale_since: Dict[LevelKey, Optional[float]]
    _level_cancel_pending_ext_id: Dict[LevelKey, Optional[str]]
    _level_pof_until: Dict[LevelKey, float]

    # --- Methods the pipeline calls on the strategy ---
    def _normalise_side(self, raw: str) -> str: ...
    def _increases_inventory(self, side: object) -> bool: ...
    def _is_strong_counter_trend_side(self, side_name: str, trend: Any) -> bool: ...
    def _counter_trend_side(self, trend: Any) -> Optional[str]: ...
    def _order_age_exceeded(self, key: LevelKey, *, max_age_s: Optional[float] = None) -> bool: ...
    def _compute_target_price(self, side: object, level: int, best_price: Decimal, **kw: Any) -> Decimal: ...
    def _level_size(self, level: int) -> Decimal: ...
    def _needs_reprice(self, side: object, prev_price: Decimal, current_best: Decimal, level: int, **kw: Any) -> tuple[bool, str]: ...
    def _quantize_size(self, size: Decimal) -> Decimal: ...
    def _apply_post_only_safety(self, *, side: object, target_price: Decimal, bid_price: Decimal, ask_price: Decimal, safety_ticks: int) -> Optional[Decimal]: ...
    def _effective_safety_ticks(self, key: LevelKey) -> int: ...
    def _on_successful_quote(self, key: LevelKey) -> None: ...
    def _funding_bias_bps(self) -> Decimal: ...
    def _record_reprice_decision(self, **kwargs: Any) -> None: ...
    async def _cancel_level_order(self, *, key: LevelKey, external_id: str, side: object, level: int, reason: str) -> bool: ...
