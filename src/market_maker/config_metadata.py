"""Field group definitions and metadata helpers for MarketMakerSettings."""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Literal, Optional

from decimal import Decimal


# ---------------------------------------------------------------------------
# Field group definitions
# ---------------------------------------------------------------------------

SETTINGS_FIELD_GROUPS: Dict[str, FrozenSet[str]] = {
    "credentials": frozenset({
        "vault_id", "stark_private_key", "stark_public_key", "api_key",
    }),
    "risk": frozenset({
        "max_position_size", "max_long_position_size",
        "max_short_position_size", "max_order_notional_usd",
        "max_position_notional_usd", "gross_exposure_limit_usd",
        "balance_aware_sizing_enabled", "balance_usage_factor",
        "balance_notional_multiplier", "balance_min_available_usd",
        "balance_staleness_max_s", "balance_stale_action",
        "inventory_skew_factor", "inventory_deadband_pct",
        "skew_shape_k", "skew_max_bps",
        "inventory_warn_pct", "inventory_critical_pct",
        "inventory_hard_pct",
        "drawdown_stop_enabled",
        "drawdown_stop_pct_of_max_notional",
        "drawdown_use_high_watermark",
        "min_liq_distance_bps",
    }),
    "pricing": frozenset({
        "offset_mode", "price_offset_per_level_percent",
        "spread_multiplier", "min_offset_bps", "max_offset_bps",
        "order_size_multiplier", "num_price_levels",
        "reprice_tolerance_percent", "min_reprice_move_ticks",
        "min_reprice_edge_delta_bps",
        "post_only_safety_ticks", "adaptive_pof_enabled",
        "pof_max_safety_ticks", "pof_backoff_multiplier",
        "pof_streak_reset_s", "pof_cooldown_s",
        "min_acceptable_markout_bps",
        "min_reprice_interval_s", "max_order_age_s",
        "size_scale_per_level", "min_spread_bps",
        "quote_anchor", "markout_anchor",
    }),
    "trend": frozenset({
        "trend_enabled", "trend_fast_ema_s", "trend_slow_ema_s",
        "trend_strong_threshold", "trend_counter_side_size_cut",
        "trend_skew_boost", "trend_one_way_enabled",
        "trend_cancel_counter_on_strong",
        "vol_regime_enabled",
        "vol_regime_short_window_s", "vol_regime_medium_window_s",
        "vol_regime_long_window_s",
        "vol_regime_calm_bps", "vol_regime_elevated_bps",
        "vol_regime_extreme_bps",
        "vol_offset_scale_calm", "vol_offset_scale_elevated",
        "vol_offset_scale_extreme",
        "micro_vol_window_s", "micro_vol_max_bps",
        "micro_drift_window_s", "micro_drift_max_bps",
        "imbalance_window_s", "imbalance_pause_threshold",
        "volatility_offset_multiplier",
        "funding_bias_enabled", "funding_inventory_weight",
        "funding_bias_cap_bps",
    }),
    "operational": frozenset({
        "enabled", "environment", "market_name", "market_profile",
        "flatten_position_on_shutdown",
        "shutdown_flatten_slippage_bps",
        "shutdown_flatten_slippage_step_bps",
        "shutdown_flatten_max_slippage_bps",
        "shutdown_flatten_retries", "shutdown_flatten_retry_delay_s",
        "shutdown_timeout_s",
        "circuit_breaker_max_failures", "circuit_breaker_cooldown_s",
        "failure_window_s", "failure_rate_trip",
        "min_attempts_for_breaker",
        "cancel_on_stale_book", "stale_cancel_grace_s",
        "orderbook_staleness_threshold_s",
        "fee_refresh_interval_s",
        "builder_program_enabled", "builder_id", "builder_fee_rate",
        "deadman_enabled", "deadman_countdown_s", "deadman_heartbeat_s",
        "margin_guard_enabled",
        "min_available_balance_for_trading",
        "min_available_balance_ratio",
        "max_margin_utilization",
        "margin_guard_shutdown_breach_s",
        "rate_limit_degraded_s", "rate_limit_halt_window_s",
        "rate_limit_halt_hits", "rate_limit_halt_s",
        "rate_limit_extra_offset_bps", "rate_limit_reprice_multiplier",
        "max_orders_per_second", "maintenance_pause_s",
        "log_level", "journal_reprice_decisions",
        "fill_snapshot_depth", "journal_max_size_mb",
    }),
}


def _resolve_type_name(annotation: Any) -> str:
    """Map a type annotation to a simple policy-file type string."""
    if annotation is None:
        return "string"
    origin = getattr(annotation, "__origin__", None)
    if origin is Literal:
        return "string"
    if annotation is bool or annotation is Optional[bool]:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is Decimal:
        return "float"
    if annotation is str:
        return "string"
    type_str = str(annotation)
    if "Decimal" in type_str:
        return "float"
    if "bool" in type_str:
        return "bool"
    if "int" in type_str:
        return "int"
    if "float" in type_str:
        return "float"
    return "string"


def field_groups() -> Dict[str, FrozenSet[str]]:
    """Return the canonical mapping of group name -> field names."""
    return dict(SETTINGS_FIELD_GROUPS)


def fields_for_group(group: str) -> FrozenSet[str]:
    """Return the set of field names belonging to *group*."""
    return SETTINGS_FIELD_GROUPS.get(group, frozenset())


def field_metadata(cls: Any) -> List[Dict[str, Any]]:
    """Return structured metadata for every field of *cls*.

    Each entry includes: name, env_var, type, default, description,
    group, and any ge/le/gt/lt constraints from the Field.
    """
    prefix = cls.model_config.get("env_prefix", "MM_")
    field_to_group: Dict[str, str] = {}
    for group_name, fields in SETTINGS_FIELD_GROUPS.items():
        for fname in fields:
            field_to_group[fname] = group_name

    result: List[Dict[str, Any]] = []
    for name, field_info in cls.model_fields.items():
        annotation = field_info.annotation
        type_name = _resolve_type_name(annotation)

        entry: Dict[str, Any] = {
            "name": name,
            "env_var": f"{prefix}{name.upper()}",
            "type": type_name,
            "default": field_info.default,
            "description": field_info.description or "",
            "group": field_to_group.get(name, "ungrouped"),
        }
        for meta_obj in field_info.metadata:
            cls_name = type(meta_obj).__name__
            if cls_name == "Ge" and hasattr(meta_obj, "ge"):
                entry["ge"] = meta_obj.ge
            elif cls_name == "Gt" and hasattr(meta_obj, "gt"):
                entry["gt"] = meta_obj.gt
            elif cls_name == "Le" and hasattr(meta_obj, "le"):
                entry["le"] = meta_obj.le
            elif cls_name == "Lt" and hasattr(meta_obj, "lt"):
                entry["lt"] = meta_obj.lt
        result.append(entry)
    return result
