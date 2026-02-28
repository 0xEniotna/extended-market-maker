"""
Automated Config Rollback Watchdog

Monitors key performance metrics (markout, fill rate, P&L) after config
changes and automatically reverts to the last known-good config when
performance degrades.

With 3 advisor agents modifying config, a bad parameter change could go
unnoticed for hours.  This watchdog catches degradation within minutes
and reverts to prevent prolonged losses.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How long to wait after a config change before evaluating its impact.
_DEFAULT_EVALUATION_WINDOW_S = 300.0  # 5 minutes
# Minimum fills required for a meaningful comparison.
_MIN_FILLS_FOR_EVALUATION = 5


@dataclass(frozen=True)
class RollbackDecision:
    """Result of the config rollback evaluation."""

    should_rollback: bool = False
    reason: str = ""
    degraded_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Baseline performance metrics captured before a config change."""

    captured_at: float = 0.0
    avg_markout_1s_bps: Decimal = Decimal("0")
    avg_markout_5s_bps: Decimal = Decimal("0")
    fill_count: int = 0
    maker_fill_pct: Decimal = Decimal("0")
    session_pnl_usd: Decimal = Decimal("0")
    avg_spread_capture_bps: Decimal = Decimal("0")


class ConfigRollbackWatchdog:
    """Monitors performance after config changes and triggers rollback.

    Usage:
        1. Call ``on_config_change(old_config, new_config)`` when config changes.
        2. Periodically call ``evaluate(current_metrics)`` to check if
           performance has degraded.
        3. If ``evaluate()`` returns ``should_rollback=True``, apply the
           stored ``last_known_good_config``.
        4. Call ``mark_config_good()`` to save current config as known-good
           (e.g., after a stable period).

    Parameters
    ----------
    evaluation_window_s : float
        Seconds to wait after a config change before evaluating.
    markout_degradation_bps : Decimal
        Minimum degradation in average 1s markout (bps) to trigger rollback.
    pnl_degradation_usd : Decimal
        Session P&L degradation threshold in USD.
    min_fills : int
        Minimum fills in evaluation window for meaningful comparison.
    """

    def __init__(
        self,
        *,
        evaluation_window_s: float = _DEFAULT_EVALUATION_WINDOW_S,
        markout_degradation_bps: Decimal = Decimal("2"),
        pnl_degradation_usd: Decimal = Decimal("50"),
        min_fills: int = _MIN_FILLS_FOR_EVALUATION,
    ) -> None:
        self._evaluation_window_s = evaluation_window_s
        self._markout_degradation_bps = markout_degradation_bps
        self._pnl_degradation_usd = pnl_degradation_usd
        self._min_fills = min_fills

        self._last_known_good_config: Optional[Dict[str, Any]] = None
        self._baseline: Optional[PerformanceBaseline] = None
        self._pending_config: Optional[Dict[str, Any]] = None
        self._change_ts: Optional[float] = None
        self._rollback_count = 0

    def mark_config_good(self, config: Dict[str, Any]) -> None:
        """Save the current config as the last known-good state."""
        self._last_known_good_config = copy.deepcopy(config)
        logger.info("Config marked as known-good (rollback baseline updated)")

    def on_config_change(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any],
        baseline: PerformanceBaseline,
    ) -> None:
        """Record a config change and capture the performance baseline.

        Call this from the SIGHUP reload handler after applying new config.
        """
        if self._last_known_good_config is None:
            self._last_known_good_config = copy.deepcopy(old_config)

        self._pending_config = copy.deepcopy(new_config)
        self._baseline = baseline
        self._change_ts = time.monotonic()
        logger.info(
            "Config change recorded — evaluating impact in %.0fs",
            self._evaluation_window_s,
        )

    def evaluate(
        self,
        *,
        avg_markout_1s_bps: Decimal = Decimal("0"),
        avg_markout_5s_bps: Decimal = Decimal("0"),
        fill_count: int = 0,
        maker_fill_pct: Decimal = Decimal("0"),
        session_pnl_usd: Decimal = Decimal("0"),
        avg_spread_capture_bps: Decimal = Decimal("0"),
    ) -> RollbackDecision:
        """Evaluate whether the recent config change caused degradation.

        Returns a RollbackDecision indicating whether to revert config.
        """
        if self._change_ts is None or self._baseline is None:
            return RollbackDecision()

        elapsed = time.monotonic() - self._change_ts
        if elapsed < self._evaluation_window_s:
            return RollbackDecision()  # Too early to evaluate

        if self._last_known_good_config is None:
            return RollbackDecision()

        # Check if we have enough data.
        new_fills = fill_count - self._baseline.fill_count
        if new_fills < self._min_fills:
            # Not enough fills — extend the window (don't decide yet).
            return RollbackDecision()

        degraded: Dict[str, Any] = {}

        # --- Markout degradation ---
        markout_delta = avg_markout_1s_bps - self._baseline.avg_markout_1s_bps
        if markout_delta < -self._markout_degradation_bps:
            degraded["markout_1s_delta_bps"] = markout_delta

        # --- P&L degradation ---
        pnl_delta = session_pnl_usd - self._baseline.session_pnl_usd
        if pnl_delta < -self._pnl_degradation_usd:
            degraded["pnl_delta_usd"] = pnl_delta

        # --- Spread capture degradation ---
        if self._baseline.avg_spread_capture_bps > 0:
            capture_delta = avg_spread_capture_bps - self._baseline.avg_spread_capture_bps
            if capture_delta < -self._markout_degradation_bps:
                degraded["spread_capture_delta_bps"] = capture_delta

        if degraded:
            self._rollback_count += 1
            reason = "; ".join(f"{k}={v}" for k, v in degraded.items())
            logger.warning(
                "CONFIG ROLLBACK TRIGGERED (rollback #%d): %s",
                self._rollback_count,
                reason,
            )
            # Clear pending state so we don't re-trigger.
            self._change_ts = None
            self._baseline = None

            return RollbackDecision(
                should_rollback=True,
                reason=reason,
                degraded_metrics=degraded,
            )

        # Config change looks good — promote it to known-good.
        if self._pending_config is not None:
            self._last_known_good_config = copy.deepcopy(self._pending_config)
            logger.info(
                "Config change validated — promoted to known-good after %.0fs "
                "(%d new fills, markout_1s=%.2fbps, pnl=$%.2f)",
                elapsed, new_fills, avg_markout_1s_bps, session_pnl_usd,
            )
        self._change_ts = None
        self._baseline = None
        self._pending_config = None
        return RollbackDecision()

    @property
    def last_known_good_config(self) -> Optional[Dict[str, Any]]:
        """Return the last known-good config, or None if not set."""
        return copy.deepcopy(self._last_known_good_config) if self._last_known_good_config else None

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def has_pending_evaluation(self) -> bool:
        return self._change_ts is not None
