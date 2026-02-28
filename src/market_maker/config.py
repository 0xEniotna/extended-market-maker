"""
Market Maker Configuration

Loads MM_ prefixed environment variables using pydantic-settings.
Defaults to testnet—requires explicit MM_ENVIRONMENT=mainnet for production.

Fields are logically grouped into sub-config categories via
``field_groups()`` for policy file generation and advisor-agent
validation, while keeping a flat env-var namespace (``MM_*``) for
backward compatibility.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, FrozenSet, List

from pydantic import Field, field_validator, model_validator

from .config_base import MarketMakerSettingsBase
from .config_env import (  # noqa: F401 — re-export for backwards compat
    ENV_FILE,
    MMEnvironment,
    OffsetMode,
    PROJECT_ROOT,
    QuoteAnchor,
)
from .config_metadata import SETTINGS_FIELD_GROUPS  # noqa: F401 — re-export


class MarketMakerSettings(MarketMakerSettingsBase):
    """Configuration for the market making strategy."""

    # --- Volatility Regime ---
    vol_regime_enabled: bool = Field(
        default=True,
        description="Enable volatility regime classification.",
    )
    vol_regime_short_window_s: float = Field(
        default=15.0,
        ge=0,
        description="Short horizon window (seconds) for volatility regime.",
    )
    vol_regime_medium_window_s: float = Field(
        default=60.0,
        ge=0,
        description="Medium horizon window (seconds) for volatility regime.",
    )
    vol_regime_long_window_s: float = Field(
        default=120.0,
        ge=0,
        description="Long horizon window (seconds) for volatility regime.",
    )
    vol_regime_calm_bps: Decimal = Field(
        default=Decimal("8"),
        ge=0,
        description="Volatility threshold (bps) below which regime is CALM.",
    )
    vol_regime_elevated_bps: Decimal = Field(
        default=Decimal("20"),
        ge=0,
        description="Volatility threshold (bps) above which regime is ELEVATED.",
    )
    vol_regime_extreme_bps: Decimal = Field(
        default=Decimal("45"),
        ge=0,
        description="Volatility threshold (bps) above which regime is EXTREME.",
    )
    vol_offset_scale_calm: Decimal = Field(
        default=Decimal("0.8"),
        ge=0,
        description="Offset scale applied in CALM regime.",
    )
    vol_offset_scale_elevated: Decimal = Field(
        default=Decimal("1.5"),
        ge=0,
        description="Offset scale applied in ELEVATED regime.",
    )
    vol_offset_scale_extreme: Decimal = Field(
        default=Decimal("2.2"),
        ge=0,
        description="Offset scale applied in EXTREME regime.",
    )

    # --- Trend Signal ---
    trend_enabled: bool = Field(
        default=True,
        description="Enable trend estimation and directional quoting bias.",
    )
    trend_fast_ema_s: float = Field(
        default=15.0,
        ge=0,
        description="Fast EMA horizon in seconds for trend estimation.",
    )
    trend_slow_ema_s: float = Field(
        default=60.0,
        ge=0,
        description="Slow EMA horizon in seconds for trend estimation.",
    )
    trend_strong_threshold: Decimal = Field(
        default=Decimal("0.7"),
        ge=0,
        le=1,
        description="Trend strength threshold above which trend is considered strong.",
    )
    trend_counter_side_size_cut: Decimal = Field(
        default=Decimal("0.6"),
        ge=0,
        le=1,
        description="Max size reduction applied to counter-trend side.",
    )
    trend_skew_boost: Decimal = Field(
        default=Decimal("1.5"),
        ge=1,
        description="Inventory skew multiplier under stronger trends.",
    )
    trend_one_way_enabled: bool = Field(
        default=False,
        description=(
            "When enabled (crypto profile only), disable all counter-trend "
            "quoting while trend strength is above trend_strong_threshold."
        ),
    )
    trend_cancel_counter_on_strong: bool = Field(
        default=False,
        description=(
            "When trend_one_way_enabled is true, immediately cancel resting "
            "counter-trend orders under strong trend."
        ),
    )

    # --- Funding Bias ---
    funding_bias_enabled: bool = Field(
        default=True,
        description="Enable funding-rate-based carry bias for inventory handling.",
    )
    funding_inventory_weight: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        description="Weight applied to funding-rate inventory bias.",
    )
    funding_bias_cap_bps: Decimal = Field(
        default=Decimal("5"),
        ge=0,
        description="Absolute cap (bps) for funding carry bias.",
    )

    # --- Per-Asset Drawdown Stop ---
    drawdown_stop_enabled: bool = Field(
        default=False,
        description=(
            "Enable per-asset drawdown stop based on market position PnL "
            "(realised + unrealised)."
        ),
    )
    drawdown_stop_pct_of_max_notional: Decimal = Field(
        default=Decimal("1.5"),
        ge=0,
        description=(
            "Drawdown threshold as percent of max_position_notional_usd. "
            "Example: 1.5 means threshold = 1.5% of max notional."
        ),
    )
    drawdown_use_high_watermark: bool = Field(
        default=True,
        description=(
            "Use high-water-mark drawdown (peak PnL minus current PnL). "
            "If false, drawdown is measured from run-start PnL."
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
    flatten_position_on_shutdown: bool = Field(
        default=True,
        description=(
            "When true, on shutdown submit a reduce-only MARKET+IOC order "
            "to flatten any open position after cancelling resting orders."
        ),
    )
    shutdown_flatten_slippage_bps: Decimal = Field(
        default=Decimal("20"),
        ge=0,
        description=(
            "Initial price aggressiveness (bps) used for shutdown flatten orders. "
            "Higher values increase fill probability."
        ),
    )
    shutdown_flatten_slippage_step_bps: Decimal = Field(
        default=Decimal("10"),
        ge=0,
        description=(
            "Additional slippage added per flatten retry (progressive slippage). "
            "Attempt 1 uses shutdown_flatten_slippage_bps, "
            "attempt 2 uses +step, etc."
        ),
    )
    shutdown_flatten_max_slippage_bps: Decimal = Field(
        default=Decimal("100"),
        ge=0,
        description=(
            "Maximum slippage (bps) for shutdown flatten orders, "
            "capping the progressive slippage escalation."
        ),
    )
    shutdown_flatten_retries: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Maximum flatten attempts on shutdown before giving up."
        ),
    )
    shutdown_flatten_retry_delay_s: float = Field(
        default=1.0,
        ge=0,
        le=60,
        description=(
            "Delay between shutdown flatten retries."
        ),
    )
    shutdown_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Hard timeout for the entire shutdown sequence. "
            "If exceeded, writes an emergency state file and force-exits."
        ),
    )

    # --- Exchange fee / builder config ---
    fee_refresh_interval_s: float = Field(
        default=60.0,
        gt=0,
        description="Fee cache refresh interval in seconds.",
    )
    builder_program_enabled: bool = Field(
        default=False,
        description=(
            "Enable explicit builder fee/id on order placement. "
            "When false, builderFee is forced to 0 and builderId is omitted."
        ),
    )
    builder_id: int = Field(
        default=0,
        ge=0,
        description="Builder ID used when builder_program_enabled is true.",
    )
    builder_fee_rate: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Configured builder fee rate cap when builder program is enabled.",
    )

    # --- Dead-man switch ---
    deadman_enabled: bool = Field(
        default=True,
        description="Enable exchange dead-man switch heartbeat.",
    )
    deadman_countdown_s: int = Field(
        default=30,
        ge=0,
        description="Dead-man switch countdown (seconds). 0 disables remote auto-cancel.",
    )
    deadman_heartbeat_s: float = Field(
        default=10.0,
        gt=0,
        description="Dead-man switch heartbeat interval in seconds.",
    )

    # --- Margin guard ---
    margin_guard_enabled: bool = Field(
        default=True,
        description="Enable proactive margin/liquidation-distance quote guard.",
    )
    min_available_balance_for_trading: Decimal = Field(
        default=Decimal("100"),
        ge=0,
        description="Minimum required available_for_trade collateral before quoting.",
    )
    min_available_balance_ratio: Decimal = Field(
        default=Decimal("0.15"),
        ge=0,
        le=1,
        description="Minimum available_for_trade/equity ratio before quoting.",
    )
    max_margin_utilization: Decimal = Field(
        default=Decimal("0.85"),
        ge=0,
        le=1,
        description="Maximum initial_margin/equity utilization allowed while quoting.",
    )
    min_liq_distance_bps: Decimal = Field(
        default=Decimal("500"),
        ge=0,
        description=(
            "Minimum liquidation distance in bps from mark price. "
            "If below this threshold, quoting is halted."
        ),
    )
    margin_guard_shutdown_breach_s: float = Field(
        default=15.0,
        ge=0,
        description=(
            "Persisted breach duration before escalating from quote halt "
            "to shutdown flatten."
        ),
    )

    # --- Rate-limit hybrid mode ---
    rate_limit_degraded_s: float = Field(
        default=20.0,
        ge=0,
        description="Duration of degraded quoting mode after a 429 event.",
    )
    rate_limit_halt_window_s: float = Field(
        default=60.0,
        gt=0,
        description="Window (seconds) for counting 429 bursts before halt.",
    )
    rate_limit_halt_hits: int = Field(
        default=5,
        ge=1,
        description="Number of 429 hits within window required to trigger quote halt.",
    )
    rate_limit_halt_s: float = Field(
        default=30.0,
        gt=0,
        description="Quote-halt duration after a 429 burst trip.",
    )
    rate_limit_extra_offset_bps: Decimal = Field(
        default=Decimal("5"),
        ge=0,
        description="Additional quote widening (bps) during 429 degraded mode.",
    )
    rate_limit_reprice_multiplier: Decimal = Field(
        default=Decimal("2"),
        ge=1,
        description="Reprice cadence multiplier during 429 degraded mode.",
    )

    # --- Network resilience ---
    max_orders_per_second: float = Field(
        default=10.0,
        gt=0,
        le=100,
        description=(
            "Maximum order placements per second (token-bucket rate limiter). "
            "Prevents burst placement after circuit breaker reset or startup."
        ),
    )
    maintenance_pause_s: float = Field(
        default=60.0,
        ge=0,
        le=600,
        description=(
            "Seconds to pause all quoting after detecting exchange maintenance "
            "(HTTP 503). Cancels resting orders on entry."
        ),
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
    journal_max_size_mb: float = Field(
        default=50.0,
        ge=0,
        le=1000,
        description=(
            "Maximum journal file size in MB before rotation. "
            "When exceeded, the current file is closed and a new one opened "
            "with an incremented suffix. A 'latest' symlink is maintained."
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

    @field_validator("market_profile", mode="before")
    @classmethod
    def _normalise_market_profile(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("quote_anchor", "markout_anchor", mode="before")
    @classmethod
    def _normalise_anchor(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @model_validator(mode="after")
    def _validate_anchor_and_builder(self) -> "MarketMakerSettings":
        if self.quote_anchor != self.markout_anchor:
            raise ValueError("quote_anchor and markout_anchor must match for coherent diagnostics")
        if self.quote_anchor != QuoteAnchor.MID:
            raise ValueError("this rollout currently supports MM_QUOTE_ANCHOR=mid only")

        if not self.builder_program_enabled:
            return self

        if self.builder_id <= 0:
            raise ValueError("builder_program_enabled=true requires MM_BUILDER_ID > 0")
        if self.builder_fee_rate < 0:
            raise ValueError("builder_fee_rate must be >= 0")
        return self

    # ------------------------------------------------------------------
    # Sub-config grouping (delegates to config_metadata module)
    # ------------------------------------------------------------------

    @classmethod
    def field_groups(cls) -> Dict[str, FrozenSet[str]]:
        """Return the canonical mapping of group name -> field names."""
        from .config_metadata import field_groups
        return field_groups()

    @classmethod
    def fields_for_group(cls, group: str) -> FrozenSet[str]:
        """Return the set of field names belonging to *group*."""
        from .config_metadata import fields_for_group
        return fields_for_group(group)

    @classmethod
    def field_metadata(cls) -> List[Dict[str, Any]]:
        """Return structured metadata for every field."""
        from .config_metadata import field_metadata
        return field_metadata(cls)
