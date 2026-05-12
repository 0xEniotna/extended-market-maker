#!/usr/bin/env python3
"""Counterfactual backtest of the funding-aware LQ overlay.

Replays a recorded MM journal and, for each ``order_placed`` event,
reconstructs the bot's quote-computation inputs and runs
``PricingEngine.compute_target_price`` twice:
  1. With ``funding_aware=None`` — baseline (current production behavior).
  2. With ``funding_aware=FundingAwarePolicy(...)`` and a funding rate
     looked up from a downloaded funding-history file.

Outputs per-event (timestamp, side, level, baseline_offset_bps,
overlay_offset_bps, perturbation_bps) and aggregates into a report
covering:
  * distribution of perturbation magnitude
  * monotonicity vs |F|
  * per-regime tabulation (calm: spread<5bps, wide: spread>=20bps)
  * max/p99 perturbation (sanity-bound against coupling_bps_max)

This is a Phase 4 gate per the funding-aware-mm rollout plan.
It does NOT place orders, hit the exchange, or modify any state.

Usage:
    PYTHONPATH=src python scripts/backtest_funding_aware_counterfactual.py \
        --market ETH-USD \
        --journal data/mm_journal/mm_ETH-USD_latest.jsonl \
        --funding data/funding_history/ETH-USD.json \
        --max-events 50000 \
        --out docs/funding_aware_mm_backtest_ETH.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from bisect import bisect_right
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Stub SDK modules so PricingEngine imports work without the live SDK,
# mirroring tests/conftest.py.
from unittest.mock import MagicMock  # noqa: E402

_SDK_MODULES = [
    "x10", "x10.perpetual", "x10.perpetual.orders",
    "x10.perpetual.trading_client", "x10.perpetual.positions",
    "x10.perpetual.accounts", "x10.perpetual.configuration",
    "x10.perpetual.orderbook", "x10.perpetual.trades",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.utils", "x10.utils.http",
]
for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()
_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED", CANCELLED="CANCELLED", EXPIRED="EXPIRED",
    REJECTED="REJECTED", OPEN="OPEN",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT", MARKET="MARKET")
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT", IOC="IOC")
_orders_mod.OpenOrderModel = object
_orders_mod.OrderStatusReason = SimpleNamespace(POST_ONLY_FAILED="POST_ONLY_FAILED")
_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")

from market_maker.funding_aware import (  # noqa: E402
    FundingAwarePolicy,
    make_policy_if_enabled,
)
from market_maker.orderbook_manager import PriceLevel  # noqa: E402
from market_maker.pricing_engine import PricingEngine  # noqa: E402

# ---------------------------------------------------------------------------
# Funding history lookup (step function, sorted)
# ---------------------------------------------------------------------------


class FundingLookup:
    """Step-function lookup: funding rate active *at or before* a given ts."""

    def __init__(self, entries: list[dict]) -> None:
        entries = sorted(entries, key=lambda e: e["timestamp"])
        self._ts_ms = [int(e["timestamp"]) for e in entries]
        self._rates = [Decimal(e["funding_rate"]) for e in entries]

    def rate_at(self, ts_seconds: float) -> Decimal:
        if not self._ts_ms:
            return Decimal("0")
        ts_ms = int(ts_seconds * 1000)
        idx = bisect_right(self._ts_ms, ts_ms) - 1
        if idx < 0:
            return self._rates[0]
        return self._rates[idx]

    def __len__(self) -> int:
        return len(self._ts_ms)


# ---------------------------------------------------------------------------
# Minimal fixtures used by PricingEngine
# ---------------------------------------------------------------------------


class _ReplayOB:
    """Static orderbook view exposing the recorded BBO for one tick."""

    def __init__(self, bid: Decimal, ask: Decimal) -> None:
        self._bid = PriceLevel(price=bid, size=Decimal("0"))
        self._ask = PriceLevel(price=ask, size=Decimal("0"))

    def best_bid(self) -> PriceLevel:
        return self._bid

    def best_ask(self) -> PriceLevel:
        return self._ask

    def spread_bps(self) -> Decimal:
        mid = (self._bid.price + self._ask.price) / 2
        if mid <= 0:
            return Decimal("0")
        return (self._ask.price - self._bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self) -> Decimal:
        return self.spread_bps()

    def is_stale(self) -> bool:
        return False


class _ReplayRisk:
    def __init__(self, position: Decimal) -> None:
        self._pos = position

    def get_current_position(self) -> Decimal:
        return self._pos


# ---------------------------------------------------------------------------
# Settings reconstruction
# ---------------------------------------------------------------------------


def _load_settings_for_market(market: str, env_file: Path | None) -> SimpleNamespace:
    """Load the live env file for this market and build a minimal settings ns.

    We don't go through pydantic to avoid pulling SDK validation; we just
    read the env values needed by PricingEngine.compute_target_price.
    """
    if env_file is None:
        # Derive from convention: ETH-USD -> .env.eth
        token = market.split("-")[0].lower().replace("_24_5", "_24_5")
        env_file = Path(f".env.{token}")
    env_values: dict[str, str] = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env_values[k.strip()] = v.strip()

    def _d(key: str, default: str) -> Decimal:
        return Decimal(env_values.get(key, default))

    def _s(key: str, default: str) -> str:
        return env_values.get(key, default)

    return SimpleNamespace(
        offset_mode=_s("MM_OFFSET_MODE", "dynamic"),
        spread_multiplier=_d("MM_SPREAD_MULTIPLIER", "1.0"),
        min_offset_bps=_d("MM_MIN_OFFSET_BPS", "4"),
        max_offset_bps=_d("MM_MAX_OFFSET_BPS", "90"),
        price_offset_per_level_percent=_d("MM_PRICE_OFFSET_PER_LEVEL_PERCENT", "0.05"),
        max_position_size=_d("MM_MAX_POSITION_SIZE", "1"),
        inventory_hard_pct=_d("MM_INVENTORY_HARD_PCT", "0.95"),
        inventory_critical_pct=_d("MM_INVENTORY_CRITICAL_PCT", "0.80"),
        inventory_warn_pct=_d("MM_INVENTORY_WARN_PCT", "0.50"),
        inventory_deadband_pct=_d("MM_INVENTORY_DEADBAND_PCT", "0.10"),
        skew_shape_k=_d("MM_SKEW_SHAPE_K", "2.0"),
        skew_max_bps=_d("MM_SKEW_MAX_BPS", "25"),
        inventory_skew_factor=_d("MM_INVENTORY_SKEW_FACTOR", "0.6"),
        trend_skew_boost=_d("MM_TREND_SKEW_BOOST", "1.5"),
        market_profile=_s("MM_MARKET_PROFILE", "crypto"),
        size_scale_per_level=_d("MM_SIZE_SCALE_PER_LEVEL", "1.0"),
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _classify_regime(spread_bps: Decimal) -> str:
    if spread_bps < Decimal("5"):
        return "calm"
    if spread_bps < Decimal("20"):
        return "normal"
    return "wide"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = int(p / 100.0 * (len(sv) - 1))
    return sv[idx]


def replay(
    *,
    journal_path: Path,
    funding_lookup: FundingLookup,
    settings: SimpleNamespace,
    policy: FundingAwarePolicy,
    tick_size: Decimal,
    max_events: int,
) -> dict:
    base_engine = PricingEngine(
        settings=settings,
        orderbook_mgr=_ReplayOB(Decimal("1"), Decimal("2")),  # rebuilt per tick
        risk_mgr=_ReplayRisk(Decimal("0")),
        tick_size=tick_size,
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.01"),
        funding_aware=None,
    )
    overlay_engine = PricingEngine(
        settings=settings,
        orderbook_mgr=_ReplayOB(Decimal("1"), Decimal("2")),
        risk_mgr=_ReplayRisk(Decimal("0")),
        tick_size=tick_size,
        base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.01"),
        funding_aware=policy,
    )

    perturbations: list[dict] = []
    n_processed = 0
    n_skipped = 0
    with journal_path.open() as f:
        for line in f:
            if n_processed >= max_events:
                break
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "order_placed":
                continue
            try:
                bid = Decimal(str(r["best_bid"]))
                ask = Decimal(str(r["best_ask"]))
                pos = Decimal(str(r["position"]))
                side = r["side"]
                level = int(r["level"])
                ts = float(r["ts"])
            except (KeyError, TypeError, ValueError):
                n_skipped += 1
                continue
            if bid <= 0 or ask <= 0 or ask <= bid:
                n_skipped += 1
                continue

            # Inject per-tick state into both engines via attribute mutation.
            base_engine._ob = _ReplayOB(bid, ask)  # type: ignore[attr-defined]
            overlay_engine._ob = _ReplayOB(bid, ask)  # type: ignore[attr-defined]
            base_engine._risk = _ReplayRisk(pos)  # type: ignore[attr-defined]
            overlay_engine._risk = _ReplayRisk(pos)  # type: ignore[attr-defined]

            funding_rate = funding_lookup.rate_at(ts)
            # Same-side BBO is what compute_target_price wants as best_price.
            same_side_best = bid if side == "BUY" else ask

            # Re-bind the overlay's funding source to this tick's rate.
            overlay_engine._funding_aware = FundingAwarePolicy(  # type: ignore[attr-defined]
                policy.config,
                funding_rate_source=lambda fr=funding_rate: fr,
            )

            base_target = base_engine.compute_target_price(
                side, level, same_side_best,
                funding_bias_bps=Decimal("0"),
            )
            overlay_target = overlay_engine.compute_target_price(
                side, level, same_side_best,
                funding_bias_bps=Decimal("0"),
            )

            mid = (bid + ask) / Decimal("2")
            spread_bps = (ask - bid) / mid * Decimal("10000")
            perturb_quote_bps = (overlay_target - base_target) / mid * Decimal("10000")

            perturbations.append({
                "ts": ts,
                "side": side,
                "level": level,
                "spread_bps": float(spread_bps),
                "funding_rate": float(funding_rate),
                "base_price": float(base_target),
                "overlay_price": float(overlay_target),
                "perturb_quote_bps": float(perturb_quote_bps),
                "regime": _classify_regime(spread_bps),
            })
            n_processed += 1

    return {
        "n_processed": n_processed,
        "n_skipped": n_skipped,
        "perturbations": perturbations,
    }


def _aggregate(perturbations: list[dict], coupling_bps_max: float) -> dict:
    if not perturbations:
        return {"empty": True}
    by_side: dict[str, list[float]] = {"BUY": [], "SELL": []}
    by_regime: dict[str, list[float]] = {"calm": [], "normal": [], "wide": []}
    by_funding_sign: dict[str, list[float]] = {"+": [], "0": [], "-": []}
    all_abs: list[float] = []
    funding_vs_perturb: list[tuple[float, float]] = []

    for p in perturbations:
        v = p["perturb_quote_bps"]
        by_side[p["side"]].append(v)
        by_regime[p["regime"]].append(v)
        f = p["funding_rate"]
        if f > 0:
            by_funding_sign["+"].append(v)
        elif f < 0:
            by_funding_sign["-"].append(v)
        else:
            by_funding_sign["0"].append(v)
        all_abs.append(abs(v))
        funding_vs_perturb.append((abs(f), abs(v)))

    def _summary(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        n = len(values)
        s = sorted(values)
        return {
            "count": n,
            "mean": sum(values) / n,
            "median": s[n // 2],
            "min": s[0],
            "max": s[-1],
            "p99": s[int(0.99 * (n - 1))],
            "p95": s[int(0.95 * (n - 1))],
            "abs_max": max(abs(x) for x in values),
        }

    # Monotonicity check: correlation between |F| and |perturbation|.
    if funding_vs_perturb:
        n = len(funding_vs_perturb)
        sum_f = sum(p[0] for p in funding_vs_perturb)
        sum_v = sum(p[1] for p in funding_vs_perturb)
        sum_fv = sum(p[0] * p[1] for p in funding_vs_perturb)
        sum_ff = sum(p[0] ** 2 for p in funding_vs_perturb)
        sum_vv = sum(p[1] ** 2 for p in funding_vs_perturb)
        denom = ((n * sum_ff - sum_f ** 2) * (n * sum_vv - sum_v ** 2)) ** 0.5
        pearson = (n * sum_fv - sum_f * sum_v) / denom if denom > 0 else 0.0
    else:
        pearson = 0.0

    sanity_violations = sum(1 for v in all_abs if v > coupling_bps_max + 1e-6)

    return {
        "empty": False,
        "n": len(perturbations),
        "overall": _summary([p["perturb_quote_bps"] for p in perturbations]),
        "by_side": {k: _summary(v) for k, v in by_side.items()},
        "by_regime": {k: _summary(v) for k, v in by_regime.items()},
        "by_funding_sign": {k: _summary(v) for k, v in by_funding_sign.items()},
        "pearson_abs_funding_vs_abs_perturb": pearson,
        "sanity_violations_vs_cap": sanity_violations,
        "coupling_bps_max": coupling_bps_max,
    }


def _render_report(market: str, journal_path: Path, funding_path: Path,
                   result: dict, policy_cfg, n_skipped: int) -> str:
    if result.get("empty"):
        return f"# {market} backtest\n\nNo data.\n"

    overall = result["overall"]
    lines = [
        f"# Funding-Aware Counterfactual Backtest — {market}",
        "",
        f"- Journal: `{journal_path}`",
        f"- Funding history: `{funding_path}`",
        f"- Policy: coupling_bps_max={policy_cfg.coupling_bps_max}, "
        f"hold_horizon_periods={policy_cfg.hold_horizon_periods}, "
        f"dollar_cap_pct_of_notional={policy_cfg.dollar_cap_pct_of_notional}",
        f"- Events replayed: {result['n']}",
        f"- Events skipped (bad data): {n_skipped}",
        f"- Sanity-cap violations (|perturb| > coupling_bps_max): "
        f"**{result['sanity_violations_vs_cap']}**",
        f"- Pearson(|F|, |perturb|): "
        f"**{result['pearson_abs_funding_vs_abs_perturb']:+.4f}**  "
        "(positive ⇒ monotonic, expected)",
        "",
        "## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)",
        "",
        "| metric | value |",
        "|---|---|",
        f"| count | {overall['count']} |",
        f"| mean | {overall['mean']:+.4f} |",
        f"| median | {overall['median']:+.4f} |",
        f"| min | {overall['min']:+.4f} |",
        f"| max | {overall['max']:+.4f} |",
        f"| p95 | {overall['p95']:+.4f} |",
        f"| p99 | {overall['p99']:+.4f} |",
        f"| abs max | {overall['abs_max']:.4f} |",
        "",
    ]
    for label, by in (
        ("By side", "by_side"),
        ("By regime", "by_regime"),
        ("By funding sign", "by_funding_sign"),
    ):
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| bucket | count | mean | median | abs_max |")
        lines.append("|---|---|---|---|---|")
        for k, s in result[by].items():
            if s.get("count", 0) == 0:
                lines.append(f"| {k} | 0 | – | – | – |")
                continue
            lines.append(
                f"| {k} | {s['count']} | {s['mean']:+.4f} | "
                f"{s['median']:+.4f} | {s['abs_max']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market", required=True)
    p.add_argument("--journal", type=Path, required=True)
    p.add_argument("--funding", type=Path, required=True)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--tick-size", type=Decimal, default=Decimal("0.0001"))
    p.add_argument("--max-events", type=int, default=50000)
    p.add_argument("--coupling-bps-max", type=Decimal, default=Decimal("8"))
    p.add_argument("--hold-horizon-periods", type=Decimal, default=Decimal("4"))
    p.add_argument("--dollar-cap-pct", type=Decimal, default=Decimal("0.001"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    funding_data = json.loads(args.funding.read_text())
    funding_lookup = FundingLookup(funding_data["entries"])

    settings = _load_settings_for_market(args.market, args.env_file)
    policy = make_policy_if_enabled(
        enabled=True,
        coupling_bps_max=args.coupling_bps_max,
        hold_horizon_periods=args.hold_horizon_periods,
        dollar_cap_pct_of_notional=args.dollar_cap_pct,
        funding_rate_source=lambda: Decimal("0"),  # rebound per-tick during replay
    )
    if policy is None:
        print("policy=None — abort", file=sys.stderr)
        return 1

    print(f"== {args.market} ==")
    print(f"  journal: {args.journal}  ({os.path.getsize(args.journal)} bytes)")
    print(f"  funding: {args.funding}  ({len(funding_lookup)} entries)")
    print(f"  policy: coupling_bps_max={args.coupling_bps_max} "
          f"H={args.hold_horizon_periods}  dollar_cap_pct={args.dollar_cap_pct}")

    res = replay(
        journal_path=args.journal,
        funding_lookup=funding_lookup,
        settings=settings,
        policy=policy,
        tick_size=args.tick_size,
        max_events=args.max_events,
    )
    print(f"  processed={res['n_processed']}  skipped={res['n_skipped']}")
    agg = _aggregate(res["perturbations"], float(args.coupling_bps_max))
    md = _render_report(
        args.market, args.journal, args.funding, agg, policy.config,
        n_skipped=res["n_skipped"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    print(f"  report: {args.out}")
    print("\n--- summary ---")
    if not agg.get("empty"):
        o = agg["overall"]
        print(f"  abs_max perturb: {o['abs_max']:.3f} bps  (cap={args.coupling_bps_max})")
        print(f"  p99 perturb: {o['p99']:+.3f} bps")
        print(f"  Pearson(|F|, |perturb|): "
              f"{agg['pearson_abs_funding_vs_abs_perturb']:+.4f}")
        print(f"  sanity violations: {agg['sanity_violations_vs_cap']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
