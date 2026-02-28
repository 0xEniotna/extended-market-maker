"""
Strategy Component Builder

Extracted from ``MarketMakerStrategy``: creates and wires the internal
components on startup and config reload, and provides the toxicity
adjustment and tick-time estimation logic.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Optional

from .config import ENV_FILE, MarketMakerSettings
from .config_rollback import ConfigRollbackWatchdog, PerformanceBaseline
from .drawdown_stop import DrawdownStop
from .fill_quality import FillQualityTracker
from .guard_policy import GuardPolicy
from .latency_monitor import LatencyMonitor
from .pnl_attribution import PnLAttributionTracker
from .post_only_safety import PostOnlySafety
from .pricing_engine import PricingEngine
from .quote_trade_ratio import QuoteTradeRatioTracker
from .reprice_pipeline import RepricePipeline
from .trade_journal import TradeJournal
from .trend_signal import TrendSignal
from .volatility_regime import VolatilityRegime

logger = logging.getLogger(__name__)


def sanitized_run_config(settings: MarketMakerSettings) -> Dict[str, Any]:
    """Return a config snapshot safe to persist in journals."""
    data: Dict[str, Any] = settings.model_dump(mode="python")
    redact_keys = {
        "vault_id",
        "stark_private_key",
        "stark_public_key",
        "api_key",
    }
    for key in redact_keys:
        if key in data:
            data[key] = "***redacted***"
    return data


def run_provenance() -> Dict[str, Any]:
    """Best-effort runtime provenance for reproducibility."""
    provenance: Dict[str, Any] = {
        "env_file": str(ENV_FILE),
        "cwd": os.getcwd(),
    }
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        sha = res.stdout.strip()
        if res.returncode == 0 and sha:
            provenance["git_sha"] = sha
    except Exception:
        pass
    return provenance


def rebuild_components(s: Any) -> None:
    """Build (or rebuild after config reload) all composed components."""
    settings = s._settings
    quote_anchor = str(getattr(settings.quote_anchor, "value", settings.quote_anchor)).lower()
    markout_anchor = str(
        getattr(settings.markout_anchor, "value", settings.markout_anchor)
    ).lower()
    if quote_anchor != "mid" or markout_anchor != "mid":
        raise ValueError(
            "This rollout locks quote_anchor and markout_anchor to 'mid' for coherence."
        )

    s._pricing = PricingEngine(
        settings=settings,
        orderbook_mgr=s._ob,
        risk_mgr=s._risk,
        tick_size=s._tick_size,
        base_order_size=s._base_order_size,
        min_order_size_step=s._min_order_size_step,
    )
    s._post_only = PostOnlySafety(
        settings=settings,
        tick_size=s._tick_size,
        round_to_tick=s._pricing.round_to_tick,
    )
    s._volatility = VolatilityRegime(settings, s._ob)
    s._trend_signal = TrendSignal(settings, s._ob)
    s._guards = GuardPolicy(settings)
    s._reprice = RepricePipeline(settings, s._tick_size, s._pricing)
    s._drawdown_stop = DrawdownStop(
        enabled=settings.drawdown_stop_enabled,
        max_position_notional_usd=settings.max_position_notional_usd,
        drawdown_pct_of_max_notional=settings.drawdown_stop_pct_of_max_notional,
        use_high_watermark=settings.drawdown_use_high_watermark,
    )
    # Sync mutable refs into risk watchdog if it exists.
    if hasattr(s, "_risk_watchdog"):
        s._risk_watchdog._drawdown_stop = s._drawdown_stop
        s._risk_watchdog._post_only = s._post_only

    # Fill quality tracker for markout analysis and auto-widening.
    s._fill_quality = FillQualityTracker(s._ob)
    s._fill_quality.set_min_acceptable_markout_bps(
        settings.min_acceptable_markout_bps,
    )
    s._fill_quality.set_offset_widen_callback(s._on_adverse_markout_widen)

    # Wire optional trackers into metrics.
    s._metrics.set_fill_quality_tracker(s._fill_quality)
    s._metrics.set_post_only_safety(s._post_only)
    s._metrics.set_quote_halt_state(s._quote_halt_reasons)
    s._metrics.set_margin_guard_breached(s._margin_breach_since is not None)

    # --- Phase 4 institutional features ---
    if not hasattr(s, "_pnl_attribution"):
        s._pnl_attribution = PnLAttributionTracker()
    s._qtr = QuoteTradeRatioTracker(
        warn_threshold=50.0,
        critical_threshold=100.0,
    )
    s._latency_monitor = LatencyMonitor(
        warn_ms=200.0,
        critical_ms=1000.0,
    )
    if not hasattr(s, "_config_rollback"):
        s._config_rollback = ConfigRollbackWatchdog()

    # Keep legacy attribute names as aliases for one release cycle.
    s._level_pof_until = s._post_only.pof_until
    s._level_pof_streak = s._post_only.pof_streak
    s._level_pof_last_ts = s._post_only.pof_last_ts
    s._level_dynamic_safety_ticks = s._post_only.dynamic_safety_ticks
    s._level_imbalance_pause_until = s._guards._level_imbalance_pause_until


def handle_reload(s: Any) -> None:
    """SIGHUP handler — reload config from environment / .env file."""
    try:
        old_config = sanitized_run_config(s._settings)
        new_settings = MarketMakerSettings()
        new_config = sanitized_run_config(new_settings)

        # Guard: reject reload if immutable keys changed.
        for key in s._RELOAD_IMMUTABLE_KEYS:
            old_val = getattr(s._settings, key, None)
            new_val = getattr(new_settings, key, None)
            if str(old_val) != str(new_val):
                logger.error(
                    "SIGHUP reload REJECTED: immutable key '%s' changed "
                    "(%s -> %s) — restart required to apply this change",
                    key, old_val, new_val,
                )
                return

        # Compute diff: keys where values changed.
        diff: Dict[str, Any] = {}
        all_keys = set(old_config) | set(new_config)
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            if str(old_val) != str(new_val):
                diff[key] = {"before": old_val, "after": new_val}

        s._settings = new_settings
        rebuild_components(s)

        # Journal the config change event.
        if hasattr(s, "_journal") and s._journal is not None:
            s._journal.record_config_change(
                before=old_config,
                after=new_config,
                diff=diff,
            )

        # Notify config rollback watchdog.
        if diff and hasattr(s, "_config_rollback"):
            fq = getattr(s, "_fill_quality", None)
            avg_markout_1s = Decimal("0")
            if fq is not None:
                summary = fq.segmented_markout_summary("1s")
                avg_markout_1s = summary.avg_all
            pnl_snap = s._pnl_attribution.snapshot()
            baseline = PerformanceBaseline(
                captured_at=time.monotonic(),
                avg_markout_1s_bps=avg_markout_1s,
                fill_count=pnl_snap.fill_count,
                session_pnl_usd=s._risk.get_session_pnl(),
                avg_spread_capture_bps=pnl_snap.avg_spread_capture_bps,
            )
            s._config_rollback.on_config_change(old_config, new_config, baseline)

        logger.info(
            "Config reloaded: offset_mode=%s skew=%.2f spread_min=%s levels=%d max_age=%ss diff_keys=%s",
            new_settings.offset_mode.value,
            new_settings.inventory_skew_factor,
            new_settings.min_spread_bps,
            new_settings.num_price_levels,
            new_settings.max_order_age_s,
            list(diff.keys()) if diff else "none",
        )
    except Exception as exc:
        logger.error("Config reload failed: %s", exc)
        if hasattr(s, "_journal") and s._journal is not None:
            s._journal.record_error(
                component="config_reload",
                exception_type=type(exc).__name__,
                message=str(exc),
                stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                stack_trace=TradeJournal.format_stack_trace(exc),
            )


def toxicity_adjustment(s: Any) -> tuple[Decimal, Optional[str]]:
    """Return (extra_offset_bps, pause_reason) from microstructure stress."""
    if s._settings.market_profile == "crypto":
        regime = s._volatility.evaluate()
        if regime.pause:
            return Decimal("0"), "volatility_spike"
        extra_bps = max(
            Decimal("0"),
            (regime.offset_scale - Decimal("1")) * s._settings.min_offset_bps,
        )
        return extra_bps, None

    vol_bps = s._ob.micro_volatility_bps(s._settings.micro_vol_window_s)
    drift_bps = s._ob.micro_drift_bps(s._settings.micro_drift_window_s)
    drift_abs = abs(drift_bps) if drift_bps is not None else None

    vol_limit = s._settings.micro_vol_max_bps
    drift_limit = s._settings.micro_drift_max_bps

    # Hard pause on severe micro-regime stress.
    if (
        vol_bps is not None
        and vol_limit > 0
        and vol_bps > (vol_limit * Decimal("1.25"))
    ):
        return Decimal("0"), "volatility_spike"
    if (
        drift_abs is not None
        and drift_limit > 0
        and drift_abs > (drift_limit * Decimal("1.25"))
    ):
        return Decimal("0"), "drift_spike"

    # Moderate widening above soft thresholds.
    extra_bps = Decimal("0")
    if vol_bps is not None and vol_limit > 0 and vol_bps > vol_limit:
        extra_bps += (vol_bps - vol_limit) * s._settings.volatility_offset_multiplier
    if drift_abs is not None and drift_limit > 0 and drift_abs > drift_limit:
        extra_bps += (
            drift_abs - drift_limit
        ) * s._settings.volatility_offset_multiplier
    return max(Decimal("0"), extra_bps), None


def estimate_tick_time_ms(s: Any) -> float:
    """Estimate how many ms it takes for the market to move one tick."""
    window_s = s._settings.micro_vol_window_s
    if window_s <= 0:
        return 0.0
    vol_bps = s._ob.micro_volatility_bps(window_s)
    if vol_bps is None or vol_bps <= 0:
        return 0.0
    bid = s._ob.best_bid()
    ask = s._ob.best_ask()
    if bid is None or ask is None:
        return 0.0
    bp = getattr(bid, "price", None)
    ap = getattr(ask, "price", None)
    if bp is None or ap is None or bp <= 0 or ap <= 0:
        return 0.0
    mid = (bp + ap) / 2
    tick_bps = float(s._tick_size / mid * Decimal("10000"))
    if tick_bps <= 0:
        return 0.0
    vol_per_ms = float(vol_bps) / (window_s * 1000.0)
    if vol_per_ms <= 0:
        return 0.0
    return tick_bps / vol_per_ms
