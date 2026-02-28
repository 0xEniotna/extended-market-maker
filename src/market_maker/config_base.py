"""Base settings class with core market maker fields.

Credentials, strategy parameters, pricing, sizing, and microstructure
fields live here.  MarketMakerSettings inherits from this class and
adds risk/operational/regime fields plus validators.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .config_env import ENV_FILE, MMEnvironment, OffsetMode, QuoteAnchor


class MarketMakerSettingsBase(BaseSettings):
    """First half of market-maker configuration fields."""

    model_config = SettingsConfigDict(
        env_prefix="MM_",
        env_file=ENV_FILE,
        extra="ignore",
    )

    # --- Credentials ---
    vault_id: str = Field(default="", description="Vault ID for the market maker account")
    stark_private_key: str = Field(default="", description="Stark private key")
    stark_public_key: str = Field(default="", description="Stark public key")
    api_key: str = Field(default="", description="API key")

    # --- Environment ---
    environment: MMEnvironment = Field(
        default=MMEnvironment.TESTNET,
        description="Network environment (testnet or mainnet)",
    )

    # --- Strategy Parameters ---
    market_name: str = Field(
        default="ETH-USD",
        description="Market to make on (e.g. ETH-USD)",
    )
    market_profile: Literal["legacy", "crypto"] = Field(
        default="legacy",
        description=(
            "Behavior profile. 'legacy' keeps prior behavior, "
            "'crypto' enables regime/trend/funding/inventory-band logic."
        ),
    )
    quote_anchor: QuoteAnchor = Field(
        default=QuoteAnchor.MID,
        description=(
            "Reference price anchor used for quote construction "
            "(mid, mark, or index)."
        ),
    )
    markout_anchor: QuoteAnchor = Field(
        default=QuoteAnchor.MID,
        description=(
            "Reference price anchor used for markout diagnostics."
        ),
    )
    num_price_levels: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Number of price levels per side",
    )
    # --- Offset Mode ---
    offset_mode: OffsetMode = Field(
        default=OffsetMode.FIXED,
        description=(
            "Offset computation mode. "
            "'fixed' uses price_offset_per_level_percent directly. "
            "'dynamic' computes offset from the live spread."
        ),
    )

    price_offset_per_level_percent: Decimal = Field(
        default=Decimal("0.3"),
        gt=0,
        description="Price offset per level as % of best price (used in FIXED mode)",
    )

    # Dynamic offset parameters (only used when offset_mode=dynamic)
    spread_multiplier: Decimal = Field(
        default=Decimal("1.5"),
        gt=0,
        description=(
            "In DYNAMIC mode: offset = spread * spread_multiplier * (level + 1). "
            "1.0 = quote at the edge of the spread. "
            "1.5 = quote 50% wider than the spread per level."
        ),
    )
    min_offset_bps: Decimal = Field(
        default=Decimal("3"),
        ge=0,
        description=(
            "In DYNAMIC mode: floor for the per-level offset in basis points. "
            "Prevents quoting inside fees even when spread is near-zero. "
            "Should be >= your maker fee in bps."
        ),
    )
    max_offset_bps: Decimal = Field(
        default=Decimal("100"),
        gt=0,
        description=(
            "In DYNAMIC mode: ceiling for the per-level offset in basis points. "
            "Prevents unreasonably wide quotes during spread spikes."
        ),
    )
    order_size_multiplier: Decimal = Field(
        default=Decimal("1.0"),
        gt=0,
        description="Multiplier applied to the market's minimum order size",
    )
    max_position_size: Decimal = Field(
        default=Decimal("100"),
        gt=0,
        description="Maximum absolute position size (in contracts)",
    )
    max_long_position_size: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description=(
            "Maximum long position size (contracts). "
            "0 falls back to the symmetric max_position_size."
        ),
    )
    max_short_position_size: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description=(
            "Maximum short position size (contracts). "
            "0 falls back to the symmetric max_position_size."
        ),
    )
    max_order_notional_usd: Decimal = Field(
        default=Decimal("250"),
        ge=0,
        description=(
            "Maximum notional (USD) per order. "
            "0 disables this cap."
        ),
    )
    max_position_notional_usd: Decimal = Field(
        default=Decimal("2500"),
        ge=0,
        description=(
            "Maximum absolute position notional (USD). "
            "0 disables this cap."
        ),
    )
    gross_exposure_limit_usd: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description=(
            "Maximum gross exposure = abs(position * ref_price) + sum(active order notionals). "
            "0 disables this cap."
        ),
    )
    balance_aware_sizing_enabled: bool = Field(
        default=True,
        description=(
            "When true, clip order size using account available_for_trade "
            "to account for exchange-side margin reservation of open orders."
        ),
    )
    balance_usage_factor: Decimal = Field(
        default=Decimal("0.95"),
        ge=Decimal("0"),
        description=(
            "Fraction of available_for_trade to treat as usable headroom. "
            "Values below 1 keep a safety margin."
        ),
    )
    balance_notional_multiplier: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        description=(
            "Converts available collateral headroom to notional headroom. "
            "Set above 1 when trading with leverage."
        ),
    )
    balance_min_available_usd: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description=(
            "Absolute collateral buffer to keep untouched before sizing new orders."
        ),
    )
    balance_staleness_max_s: float = Field(
        default=30.0,
        ge=0,
        description=(
            "Maximum age (seconds) of cached balance before it is treated as stale. "
            "When stale, balance-aware sizing is skipped rather than using outdated data. "
            "0 disables staleness checking."
        ),
    )
    balance_stale_action: Literal["skip", "reduce", "halt"] = Field(
        default="reduce",
        description=(
            "Action when cached balance is stale. "
            "'skip' bypasses balance-aware sizing entirely (fail-open, prior default). "
            "'reduce' halves order size for opening orders. "
            "'halt' zeroes order size until fresh balance arrives."
        ),
    )
    reprice_tolerance_percent: Decimal = Field(
        default=Decimal("0.1"),
        gt=0,
        description="Tolerance factor for repricing band (as fraction of target offset)",
    )
    min_reprice_move_ticks: int = Field(
        default=2,
        ge=0,
        description=(
            "Minimum price movement in ticks required to cancel/replace an order. "
            "0 disables this gate."
        ),
    )
    min_reprice_edge_delta_bps: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        description=(
            "Minimum change in theoretical edge (bps vs same-side best) "
            "required before repricing."
        ),
    )
    post_only_safety_ticks: int = Field(
        default=2,
        ge=0,
        description=(
            "Extra tick buffer from the opposite BBO before placing a post-only order. "
            "0 still enforces a minimum 1-tick safety clamp."
        ),
    )
    adaptive_pof_enabled: bool = Field(
        default=True,
        description=(
            "Enable adaptive POST_ONLY_FAILED handling per (side, level)."
        ),
    )
    pof_max_safety_ticks: int = Field(
        default=8,
        ge=1,
        description=(
            "Maximum post-only safety ticks when adaptive POF handling is enabled."
        ),
    )
    pof_backoff_multiplier: Decimal = Field(
        default=Decimal("1.7"),
        ge=Decimal("1"),
        description=(
            "Cooldown multiplier applied per consecutive POST_ONLY_FAILED rejection."
        ),
    )
    pof_streak_reset_s: float = Field(
        default=45.0,
        ge=0,
        description=(
            "Reset adaptive POF streak if no POF rejection occurs within this window."
        ),
    )

    min_acceptable_markout_bps: Decimal = Field(
        default=Decimal("-2"),
        description=(
            "Minimum acceptable 5-second markout (bps) per level. "
            "If a level's rolling average drops below this, its offset "
            "is automatically widened by 1 tick."
        ),
    )

    # --- Reprice Rate Limiting ---
    min_reprice_interval_s: float = Field(
        default=0.5,
        ge=0,
        description=(
            "Minimum seconds between reprices on the same (side, level) slot. "
            "Prevents cancel/place churn on noisy books. 0 = disabled."
        ),
    )
    pof_cooldown_s: float = Field(
        default=2.0,
        ge=0,
        description=(
            "Seconds to pause a level after a POST_ONLY_FAILED rejection. "
            "Prevents rapid retry storms when quoting near the BBO."
        ),
    )

    # --- Order Age ---
    max_order_age_s: float = Field(
        default=15.0,
        ge=0,
        description=(
            "Maximum age (seconds) of an order before forcing a reprice. "
            "0 = disabled. Prevents stale orders from being adversely selected."
        ),
    )
    cancel_on_stale_book: bool = Field(
        default=True,
        description=(
            "Cancel resting orders when the orderbook is stale for too long."
        ),
    )
    stale_cancel_grace_s: float = Field(
        default=3.0,
        ge=0,
        description=(
            "Grace period in seconds before cancelling resting orders on stale orderbook data."
        ),
    )
    orderbook_staleness_threshold_s: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Seconds since last orderbook stream event before data is considered stale. "
            "Set per-market via env (MM_ORDERBOOK_STALENESS_THRESHOLD_S)."
        ),
    )

    # --- Per-Level Size Scaling ---
    size_scale_per_level: Decimal = Field(
        default=Decimal("1.0"),
        ge=1,
        description=(
            "Size multiplier growth per level. 1.0 = flat (all levels same size). "
            "1.5 = each deeper level is 1.5x the previous. "
            "E.g. with 3 levels and scale=1.5: L0=1x, L1=1.5x, L2=2.25x."
        ),
    )

    # --- Inventory Skew ---
    inventory_skew_factor: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        description=(
            "Inventory skew intensity (Avellaneda-Stoikov style). "
            "0 = no skew (symmetric quoting). Higher values shift quotes "
            "more aggressively to reduce inventory."
        ),
    )
    inventory_deadband_pct: Decimal = Field(
        default=Decimal("0.10"),
        ge=0,
        le=1,
        description=(
            "Deadband as a fraction of max position where inventory skew is zero."
        ),
    )
    skew_shape_k: Decimal = Field(
        default=Decimal("2.0"),
        ge=0,
        description=(
            "Shape coefficient for nonlinear inventory skew (tanh curve)."
        ),
    )
    skew_max_bps: Decimal = Field(
        default=Decimal("20"),
        ge=0,
        description=(
            "Maximum inventory skew contribution in basis points before "
            "inventory_skew_factor scaling."
        ),
    )

    # --- Inventory Guard Bands ---
    inventory_warn_pct: Decimal = Field(
        default=Decimal("0.5"),
        ge=0,
        le=1,
        description="Warn threshold as fraction of max position.",
    )
    inventory_critical_pct: Decimal = Field(
        default=Decimal("0.8"),
        ge=0,
        le=1,
        description="Critical threshold as fraction of max position.",
    )
    inventory_hard_pct: Decimal = Field(
        default=Decimal("0.95"),
        ge=0,
        le=1,
        description="Hard threshold as fraction of max position.",
    )

    # --- Minimum Spread ---
    min_spread_bps: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description=(
            "Minimum market spread in basis points to quote. "
            "If the live spread is thinner than this, quoting is paused "
            "to avoid placing orders inside the fee. 0 = disabled."
        ),
    )
    micro_vol_window_s: float = Field(
        default=5.0,
        ge=0,
        description=(
            "Window for micro-volatility computation in seconds. "
            "0 disables volatility guard."
        ),
    )
    micro_vol_max_bps: Decimal = Field(
        default=Decimal("8"),
        ge=0,
        description=(
            "Soft micro-volatility threshold in bps for toxicity guard."
        ),
    )
    micro_drift_window_s: float = Field(
        default=3.0,
        ge=0,
        description=(
            "Window for micro-drift computation in seconds. "
            "0 disables drift guard."
        ),
    )
    micro_drift_max_bps: Decimal = Field(
        default=Decimal("6"),
        ge=0,
        description=(
            "Soft micro-drift threshold in bps for toxicity guard."
        ),
    )
    imbalance_window_s: float = Field(
        default=2.0,
        ge=0,
        description=(
            "Window in seconds for top-of-book imbalance smoothing."
        ),
    )
    imbalance_pause_threshold: Decimal = Field(
        default=Decimal("0.70"),
        ge=0,
        le=1,
        description=(
            "Pause repricing on one side when signed book imbalance exceeds this threshold."
        ),
    )
    volatility_offset_multiplier: Decimal = Field(
        default=Decimal("0.35"),
        ge=0,
        description=(
            "Additional offset widening applied when micro-volatility or drift "
            "exceeds soft thresholds."
        ),
    )
