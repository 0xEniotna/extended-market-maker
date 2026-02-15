"""
Market Maker Configuration

Loads MM_ prefixed environment variables using pydantic-settings.
Defaults to testnet—requires explicit MM_ENVIRONMENT=mainnet for production.
"""
from __future__ import annotations

import os
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Env file resolution (mirrors existing pattern in settings.py)
# ---------------------------------------------------------------------------

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return current.parents[2]


PROJECT_ROOT = _find_project_root()


def _resolve_env_file() -> Path:
    env_file = os.getenv("ENV", ".env")
    candidates = []
    if env_file:
        if not env_file.startswith("."):
            candidates.append(f".{env_file}")
        candidates.append(env_file)
    else:
        candidates.append(".env")

    for candidate in candidates:
        path = Path(candidate)
        if not path.is_absolute():
            path = PROJECT_ROOT / candidate
        if path.exists():
            return path

    return PROJECT_ROOT / ".env"


ENV_FILE = _resolve_env_file()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MMEnvironment(str, Enum):
    TESTNET = "testnet"
    MAINNET = "mainnet"


class OffsetMode(str, Enum):
    """How to compute the per-level price offset.

    FIXED   — use ``price_offset_per_level_percent`` directly (absolute % of price).
    DYNAMIC — compute offset from the live spread:
              offset = max(spread * spread_multiplier, min_offset_bps / 10_000)
              Automatically adapts to any market's liquidity.
    """

    FIXED = "fixed"
    DYNAMIC = "dynamic"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class MarketMakerSettings(BaseSettings):
    """Configuration for the market making strategy."""

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

    # --- Circuit Breaker ---
    circuit_breaker_max_failures: int = Field(
        default=5,
        ge=1,
        description="Pause quoting after this many consecutive order placement failures",
    )
    circuit_breaker_cooldown_s: float = Field(
        default=30.0,
        gt=0,
        description="Seconds to pause when the circuit breaker trips",
    )
    failure_window_s: float = Field(
        default=60.0,
        gt=0,
        description="Rolling window in seconds for failure-rate circuit breaker checks.",
    )
    failure_rate_trip: Decimal = Field(
        default=Decimal("0.35"),
        ge=0,
        le=1,
        description="Trip failure-rate breaker when failures/attempts exceeds this threshold.",
    )
    min_attempts_for_breaker: int = Field(
        default=10,
        ge=1,
        description="Minimum placement attempts in failure window before rate breaker can trip.",
    )

    # --- Safety ---
    enabled: bool = Field(
        default=True,
        description="Kill switch — set to false to disable the market maker",
    )

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Log level")
    journal_reprice_decisions: bool = Field(
        default=True,
        description="Emit reprice decision telemetry events in the trade journal.",
    )
    fill_snapshot_depth: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Depth of orderbook levels captured in per-fill market snapshots."
        ),
    )

    # --- Helpers ---

    @property
    def is_configured(self) -> bool:
        return bool(
            self.vault_id
            and self.stark_private_key
            and self.stark_public_key
            and self.api_key
        )

    @property
    def endpoint_config(self) -> Any:
        from x10.perpetual.configuration import MAINNET_CONFIG, TESTNET_CONFIG

        if self.environment == MMEnvironment.MAINNET:
            return MAINNET_CONFIG
        return TESTNET_CONFIG

    @field_validator("environment", mode="before")
    @classmethod
    def _normalise_environment(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("offset_mode", mode="before")
    @classmethod
    def _normalise_offset_mode(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v
