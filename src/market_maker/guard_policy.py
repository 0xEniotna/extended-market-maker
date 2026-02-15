from __future__ import annotations

import time
from decimal import Decimal

from .decision_models import GuardDecision, RegimeState


class GuardPolicy:
    """Apply spread/imbalance/toxicity guards and return a unified decision."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._level_imbalance_pause_until: dict[tuple[str, int], float] = {}
        self._level_regime_pause_until: dict[tuple[str, int], float] = {}

    def check(
        self,
        *,
        side,
        level: int,
        spread_bps,
        imbalance,
        regime: RegimeState,
    ) -> GuardDecision:
        key = (str(side), level)
        side_name = str(side).upper()
        is_sell = side_name == "SELL" or side_name.endswith("SELL")
        now = time.monotonic()

        if self._settings.min_spread_bps > 0:
            if spread_bps is not None and spread_bps < self._settings.min_spread_bps:
                return GuardDecision(allow=False, reason="skip_min_spread")

        pause_threshold = self._settings.imbalance_pause_threshold
        pause_until = self._level_imbalance_pause_until.get(key, 0.0)
        if now < pause_until:
            return GuardDecision(
                allow=False,
                reason="skip_imbalance_pause",
                pause_until_ts=pause_until,
            )

        regime_pause_until = self._level_regime_pause_until.get(key, 0.0)
        if now < regime_pause_until:
            return GuardDecision(
                allow=False,
                reason="skip_toxicity",
                pause_until_ts=regime_pause_until,
            )

        if (
            imbalance is not None
            and pause_threshold > 0
            and (
                (is_sell and imbalance > pause_threshold)
                or ((not is_sell) and imbalance < -pause_threshold)
            )
        ):
            pause_for = max(1.0, float(self._settings.imbalance_window_s))
            until = now + pause_for
            self._level_imbalance_pause_until[key] = until
            return GuardDecision(
                allow=False,
                reason="skip_imbalance",
                pause_until_ts=until,
            )

        if regime.pause:
            cooldown_s = max(
                1.0,
                float(getattr(self._settings, "vol_regime_short_window_s", 15.0)) / 10.0,
            )
            until = now + cooldown_s
            self._level_regime_pause_until[key] = until
            return GuardDecision(
                allow=False,
                reason="skip_toxicity",
                pause_until_ts=until,
            )

        extra_bps = Decimal("0")
        if regime.offset_scale > Decimal("1"):
            # Convert regime widening into explicit bps to preserve journal semantics.
            extra_bps = (regime.offset_scale - Decimal("1")) * self._settings.min_offset_bps
        return GuardDecision(
            allow=True,
            reason="allow",
            extra_offset_bps=max(Decimal("0"), extra_bps),
        )
