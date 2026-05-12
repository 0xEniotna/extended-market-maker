"""
Funding-Aware Quoting Overlay

Heuristic carry overlay inspired by the linear-quadratic limit of the HJB
in Le, *Funding-Aware Optimal Market Making for Perpetual DEXs*
(arXiv:2605.06405v1, 2026). When enabled, this overlay replaces the
legacy ``FundingManager.funding_bias_bps()`` contribution with a
horizon-aware, cash-scaled bid/ask asymmetric perturbation.

Math (LQ closed form):

    f_cash         = S · F                       # cash funding per contract per period
    funding_dollar = f_cash · H_eff              # signed dollars over hold horizon
    signal_bps     = clip(funding_dollar / S · 1e4, ±cap_bps)
    bid_delta_bps  = +signal_bps                 # F>0 widens bid (avoid growing long)
    ask_delta_bps  = −signal_bps                 # F>0 tightens ask (shed long if any)

The asymmetry depends on ``sign(F)`` only. Inventory-dependent skew is
already produced by ``PricingEngine._skew_component_f``; adding a second
``sign(q)`` term in this overlay would double-count.

The signal is non-zero at ``q = 0`` (correct: any fill creates future
funding exposure over ``H_eff``). When ``F = 0`` the signal is zero by
construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional


@dataclass(frozen=True)
class FundingAwareConfig:
    """Static configuration for :class:`FundingAwarePolicy`."""

    enabled: bool
    coupling_bps_max: Decimal
    hold_horizon_periods: Decimal
    # Saturate the dollar signal at this fraction of (max_position_size × mid)
    # so the knob is scale-invariant across markets (a $5k-notional bot and
    # a $50k-notional bot get proportionally calibrated caps).
    dollar_cap_pct_of_notional: Decimal


class FundingAwarePolicy:
    """Compute the bid/ask perturbation from the LQ carry overlay.

    Stateless except for a reference to a funding-rate source. Safe to share
    across slot tasks. The source is a callable returning the current
    fractional funding rate per period (e.g.,
    ``lambda: funding_mgr.funding_rate``); decoupling via callable avoids a
    hard dependency on ``FundingManager`` in this module.
    """

    def __init__(
        self,
        cfg: FundingAwareConfig,
        funding_rate_source: Callable[[], Decimal],
    ) -> None:
        self._cfg = cfg
        self._funding_rate_source = funding_rate_source

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def config(self) -> FundingAwareConfig:
        return self._cfg

    def compute_funding_signal_bps(
        self,
        *,
        mid_price: Decimal,
        max_position_size: Decimal,
    ) -> Decimal:
        """Return the signed funding signal in bps.

        Caller derives ``bid_delta_bps = +signal`` and ``ask_delta_bps =
        −signal``. Returns ``Decimal("0")`` when disabled or when inputs
        cannot produce a meaningful signal (zero funding, non-positive
        mid, zero horizon, zero position cap).
        """
        if not self._cfg.enabled:
            return Decimal("0")
        if mid_price <= 0:
            return Decimal("0")
        if self._cfg.hold_horizon_periods <= 0:
            return Decimal("0")
        if max_position_size <= 0:
            return Decimal("0")

        funding_rate = self._funding_rate_source()
        if funding_rate == 0:
            return Decimal("0")

        # Cash funding per contract over the hold horizon (signed dollars).
        funding_dollar = mid_price * funding_rate * self._cfg.hold_horizon_periods

        # Soft saturation: clamp the dollar magnitude before converting to bps.
        max_notional = max_position_size * mid_price
        dollar_cap = self._cfg.dollar_cap_pct_of_notional * max_notional
        if dollar_cap <= 0:
            return Decimal("0")

        if funding_dollar > dollar_cap:
            funding_dollar = dollar_cap
        elif funding_dollar < -dollar_cap:
            funding_dollar = -dollar_cap

        # Convert to bps of mid.
        signal_bps = funding_dollar / mid_price * Decimal("10000")

        # Hard bps cap (operator-facing knob).
        cap = self._cfg.coupling_bps_max
        if signal_bps > cap:
            signal_bps = cap
        elif signal_bps < -cap:
            signal_bps = -cap

        return signal_bps


def make_policy_if_enabled(
    *,
    enabled: bool,
    coupling_bps_max: Decimal,
    hold_horizon_periods: Decimal,
    dollar_cap_pct_of_notional: Decimal,
    funding_rate_source: Callable[[], Decimal],
) -> Optional[FundingAwarePolicy]:
    """Factory that returns ``None`` when the feature flag is off.

    Used by ``strategy.py`` so the ``pricing_engine`` short-circuits on
    ``self._funding_aware is None`` with zero runtime overhead.
    """
    if not enabled:
        return None
    cfg = FundingAwareConfig(
        enabled=True,
        coupling_bps_max=coupling_bps_max,
        hold_horizon_periods=hold_horizon_periods,
        dollar_cap_pct_of_notional=dollar_cap_pct_of_notional,
    )
    return FundingAwarePolicy(cfg, funding_rate_source)
