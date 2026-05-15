"""
Markout-Feedback Overlay

A reactive per-side widening policy. On each resting fill, the policy
records the fill price and (after a configurable horizon) computes the
realized markout from the MM's perspective:

    BUY fill:  markout_bps = (mid(t+H) - fill_price) / fill_price * 1e4
    SELL fill: markout_bps = (fill_price - mid(t+H)) / fill_price * 1e4

Positive markout = good for MM (we were on the right side of post-fill
mid drift). Negative markout = adverse selection biting.

The policy maintains a **per-side EWMA** of these markouts. When the
EWMA on side X drops below ``-threshold_bps`` (i.e., side X is bleeding
on average), the policy widens that side's offset by:

    widen_bps = min(gain * (-ewma - threshold_bps), cap_bps)

The widening is applied to the side's raw price in
``PricingEngine.compute_target_price``:

    BUY:  raw_f -= widen  (push bid further below BBO, less aggressive)
    SELL: raw_f += widen  (push ask further above BBO, less aggressive)

When the EWMA recovers (markouts turn neutral or positive), the widening
decays to zero naturally via the half-life.

Why this works (verified by Phase 1 calibration on ETH journals, n=1,751):
post-fill markouts on the same side are temporally autocorrelated
(lag-1 ≈ +0.45). A bad fill predicts a bad next fill. The reactive
feedback policy exploits this clustering to widen at exactly the
windows that turn out to be toxic.

Caveats:
- The first bad fill in a streak is never prevented; only the next ones.
- The widening applies at quote PLACEMENT, not to orders already
  resting. BBO drift can still erode the protective gap before a fill.
- Sample-level effects are noisy; reliable signal needs ≥30 fills per side.

Integration:
- Strategy constructs the policy in ``strategy_components.rebuild_components``,
  same pattern as ``funding_aware`` (factory returns None when flag off).
- ``strategy_callbacks.on_fill`` calls ``policy.on_fill(ts, side, price)``
  on every resting fill.
- ``PricingEngine.compute_target_price`` calls ``policy.tick(now_ts, mid)``
  to process matured fills, then ``policy.extra_widening_bps(side)`` to
  get the current widening to apply.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

# Convenient typed aliases
_SideStr = str  # "BUY" or "SELL"


@dataclass(frozen=True)
class MarkoutFeedbackConfig:
    """Static configuration for :class:`MarkoutFeedbackPolicy`."""

    enabled: bool
    half_life_s: Decimal
    threshold_bps: Decimal
    gain: Decimal
    cap_bps: Decimal
    horizon_s: int  # delay before markout is computed


class MarkoutFeedbackPolicy:
    """Per-side EWMA markout-feedback widening policy.

    Stateful (holds the EWMAs and a queue of pending fills awaiting
    markout horizon). Single-threaded use only — designed for the MM's
    asyncio event loop.

    The ``mid_source`` callable returns the current mid price (Decimal),
    or None if mid is unavailable (e.g., stale orderbook). When mid is
    None at markout-computation time, the pending fill is deferred to
    the next tick.
    """

    BUY = "BUY"
    SELL = "SELL"
    _SIDES = (BUY, SELL)

    def __init__(
        self,
        cfg: MarkoutFeedbackConfig,
        mid_source: Callable[[], Optional[Decimal]],
    ) -> None:
        self._cfg = cfg
        self._mid_source = mid_source
        # Per-second decay factor; clamped to 0 if half_life is 0.
        if cfg.half_life_s > 0:
            self._decay_per_sec = math.exp(
                -math.log(2) / float(cfg.half_life_s)
            )
        else:
            self._decay_per_sec = 0.0
        # Per-side EWMA of realized markouts (float bps for hot-path speed).
        self._ewma: dict[str, float] = {self.BUY: 0.0, self.SELL: 0.0}
        # Last decay timestamp per side (None until first update).
        self._last_decay_ts: dict[str, Optional[float]] = {
            self.BUY: None, self.SELL: None,
        }
        # Queue of pending fills awaiting markout computation.
        # Each entry: (deadline_ts, side, fill_price_float).
        self._pending: list[tuple[float, str, float]] = []

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def config(self) -> MarkoutFeedbackConfig:
        return self._cfg

    # --- Hot-path read state for diagnostics / journaling ---

    def ewma(self, side: str) -> float:
        """Current EWMA value for side (no decay applied — read-only inspection)."""
        return self._ewma.get(side, 0.0)

    def pending_count(self) -> int:
        return len(self._pending)

    # --- Fill ingestion ---

    def on_fill(self, ts: float, side: str, price: Decimal) -> None:
        """Observe a new resting fill. Markout is computed at ts + horizon_s."""
        if not self._cfg.enabled:
            return
        side_u = (side or "").upper()
        if side_u not in self._SIDES:
            return
        if self._cfg.horizon_s < 0:
            return
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return
        if price_f <= 0:
            return
        deadline = ts + float(self._cfg.horizon_s)
        self._pending.append((deadline, side_u, price_f))

    # --- Tick: process matured fills, update EWMA ---

    def tick(self, now_ts: Optional[float] = None) -> int:
        """Process any pending fills whose horizon has elapsed.

        Returns the number of fills processed in this tick (for telemetry).
        Safe to call frequently; cheap when ``_pending`` is empty.
        """
        if not self._cfg.enabled:
            return 0
        if now_ts is None:
            now_ts = time.time()
        if not self._pending:
            return 0
        # Process in arrival order, in-place rebuild for items not yet due.
        n_processed = 0
        unprocessed: list[tuple[float, str, float]] = []
        mid_dec: Optional[Decimal] = None
        for deadline, side, fill_price_f in self._pending:
            if deadline > now_ts:
                unprocessed.append((deadline, side, fill_price_f))
                continue
            # Need mid at the (approx) deadline; fetch once per tick.
            if mid_dec is None:
                mid_dec = self._mid_source()
            if mid_dec is None:
                # Can't compute markout — keep fill pending for next tick.
                unprocessed.append((deadline, side, fill_price_f))
                continue
            mid_f = float(mid_dec)
            if mid_f <= 0:
                unprocessed.append((deadline, side, fill_price_f))
                continue
            # MM-perspective markout in bps of fill price.
            if side == self.BUY:
                markout_bps = (mid_f - fill_price_f) / fill_price_f * 1e4
            else:
                markout_bps = (fill_price_f - mid_f) / fill_price_f * 1e4
            # Decay existing EWMA on this side to deadline timestamp, then add.
            self._decay_to(side, deadline)
            self._ewma[side] += markout_bps
            n_processed += 1
        self._pending = unprocessed
        return n_processed

    def _decay_to(self, side: str, ts: float) -> None:
        """Decay the EWMA on ``side`` from its last update to ``ts``."""
        if self._decay_per_sec <= 0:
            self._ewma[side] = 0.0
            self._last_decay_ts[side] = ts
            return
        last = self._last_decay_ts.get(side)
        if last is None:
            self._last_decay_ts[side] = ts
            return
        dt = ts - last
        if dt > 0:
            # decay_per_sec ∈ (0,1); raised to dt seconds.
            self._ewma[side] *= self._decay_per_sec ** dt
        self._last_decay_ts[side] = ts

    # --- Widening query ---

    def extra_widening_bps(
        self,
        side: str,
        now_ts: Optional[float] = None,
    ) -> Decimal:
        """Return the additional widening (bps, non-negative) to apply to
        the given side at the current moment.

        Convention: callers ADD this to the side's offset from BBO. The
        sign is always non-negative — widening only ever moves the quote
        away from BBO.
        """
        if not self._cfg.enabled:
            return Decimal("0")
        side_u = (side or "").upper()
        if side_u not in self._SIDES:
            return Decimal("0")
        if now_ts is None:
            now_ts = time.time()
        # Apply pending decay to the queried side (no EWMA update — just
        # a read with proper decay). Note: when ``_decay_per_sec == 0``
        # (half_life_s = 0), ``0 ** dt == 0`` for any dt > 0 produces the
        # expected "instant decay" behavior; ``0 ** 0 == 1`` leaves the
        # value untouched when dt = 0 (same-instant query).
        ewma_decayed = self._ewma[side_u]
        last = self._last_decay_ts.get(side_u)
        if last is not None:
            dt = now_ts - last
            if dt > 0:
                ewma_decayed *= self._decay_per_sec ** dt
        threshold = float(self._cfg.threshold_bps)
        if ewma_decayed >= -threshold:
            return Decimal("0")
        excess = -ewma_decayed - threshold
        gain = float(self._cfg.gain)
        cap = float(self._cfg.cap_bps)
        widen = min(gain * excess, cap)
        if widen <= 0:
            return Decimal("0")
        # Convert back to Decimal for caller. Use string conv to avoid
        # binary-float artifacts in journaled values.
        return Decimal(str(widen))


def make_policy_if_enabled(
    *,
    enabled: bool,
    half_life_s: Decimal,
    threshold_bps: Decimal,
    gain: Decimal,
    cap_bps: Decimal,
    horizon_s: int,
    mid_source: Callable[[], Optional[Decimal]],
) -> Optional[MarkoutFeedbackPolicy]:
    """Factory that returns ``None`` when the feature flag is off.

    Used by ``strategy_components`` so the ``pricing_engine`` short-circuits
    on ``self._markout_feedback is None`` with zero runtime overhead when
    the overlay is disabled.
    """
    if not enabled:
        return None
    cfg = MarkoutFeedbackConfig(
        enabled=True,
        half_life_s=half_life_s,
        threshold_bps=threshold_bps,
        gain=gain,
        cap_bps=cap_bps,
        horizon_s=int(horizon_s),
    )
    return MarkoutFeedbackPolicy(cfg, mid_source)
