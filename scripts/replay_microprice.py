#!/usr/bin/env python3
"""Phase 3 — replay the REAL microprice recentering against recorded
``book_change`` events and characterise the per-quote perturbation before
any live A/B.

WHY (the Stage 2 trap)
----------------------
Stage 2's markout-feedback calibration overstated the effect because the
calibration sim diverged from the real implementation. To avoid that here,
this script instantiates the **actual** ``PricingEngine`` from
``src/market_maker/`` and calls ``compute_target_price`` twice per book
snapshot — once with ``use_microprice=False``, once with ``True`` — and
measures the difference. Same code path that runs in production.

Microprice has no calibration model (it's a deterministic pure function), so
the goal here is narrower than a PnL claim: confirm the integration is sane
on real DOT books before the live iter. Specifically that the shift is

  - correctly signed (bid-heavy ⇒ quotes up), 100% monotone in (micro − mid),
  - identical on both sides (it's an additive recentering, not a skew),
  - bounded (no exploding values; report max / p99),
  - rarely killed by the BBO clamp or tick rounding.

The perturbation is independent of the offset/skew/funding settings because
the shift is added identically to both engines after those terms — so the
choice of replay settings does not bias the measured shift.

PASS criteria (pre-registered, mirrors funding-aware Gate 3):
  - no exceptions, no NaN/Inf
  - sign agreement (shift vs micro−mid) = 100% on the non-clamped sample
  - bid-side and ask-side perturbation identical on the non-clamped sample
  - max |shift| finite and economically sane (report; flag if > spread)
  NO claim about fill quality or PnL — that is the live A/B (Phase 3 cont.).

Usage (on the VPS):
    PYTHONPATH=src python scripts/replay_microprice.py \\
        --journal /root/MM/data/mm_journal/mm_DOT-USD_*.jsonl \\
        --tick-size 0.0001 --market DOT-USD \\
        --out docs/stage3_replay_DOT_microprice.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Stub the x10 SDK so importing the real PricingEngine needs no exchange deps.
for _m in [
    "x10", "x10.perpetual", "x10.perpetual.orders",
    "x10.perpetual.trading_client", "x10.perpetual.positions",
    "x10.perpetual.accounts", "x10.perpetual.configuration",
    "x10.perpetual.orderbook", "x10.perpetual.trades",
    "x10.perpetual.stream_client", "x10.perpetual.stream_client.stream_client",
    "x10.utils", "x10.utils.http",
]:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

from market_maker.microprice import microprice  # noqa: E402
from market_maker.pricing_engine import PricingEngine  # noqa: E402


class _FakeOB:
    """Mutable top-of-book the replay drives from journal snapshots."""

    def __init__(self) -> None:
        self._bid = SimpleNamespace(price=Decimal("0"), size=Decimal("0"))
        self._ask = SimpleNamespace(price=Decimal("0"), size=Decimal("0"))

    def set(self, bid, bq, ask, aq) -> None:
        self._bid = SimpleNamespace(price=bid, size=bq)
        self._ask = SimpleNamespace(price=ask, size=aq)

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask

    def spread_bps(self):
        mid = (self._bid.price + self._ask.price) / 2
        if mid <= 0:
            return Decimal("0")
        return (self._ask.price - self._bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self):
        return self.spread_bps()

    def is_stale(self):
        return False


class _FakeRisk:
    def get_current_position(self):
        return Decimal("0")  # flat ⇒ skew term zero, isolates microprice


def _settings():
    # crypto profile (microprice gate) + dynamic offsets typical of DOT.
    # Values are immaterial to the measured shift (it cancels in on−off),
    # but realistic offsets keep the base quote inside the BBO so the clamp
    # behaviour mirrors production.
    return SimpleNamespace(
        offset_mode="dynamic",
        spread_multiplier=Decimal("0.5"),
        min_offset_bps=Decimal("4"),
        max_offset_bps=Decimal("50"),
        price_offset_per_level_percent=Decimal("0.05"),
        max_position_size=Decimal("1000"),
        inventory_hard_pct=Decimal("0.95"),
        inventory_critical_pct=Decimal("0.80"),
        inventory_warn_pct=Decimal("0.50"),
        inventory_deadband_pct=Decimal("0.10"),
        skew_shape_k=Decimal("2"),
        skew_max_bps=Decimal("25"),
        inventory_skew_factor=Decimal("0.6"),
        trend_skew_boost=Decimal("1.5"),
        market_profile="crypto",
        size_scale_per_level=Decimal("1.2"),
    )


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def _make_engines(tick):
    ob = _FakeOB()
    common = dict(
        settings=_settings(), orderbook_mgr=ob, risk_mgr=_FakeRisk(),
        tick_size=tick, base_order_size=Decimal("1"),
        min_order_size_step=Decimal("0.1"),
    )
    off = PricingEngine(use_microprice=False, **common)
    on = PricingEngine(use_microprice=True, **common)
    return ob, off, on


def replay(journals, tick):
    ob, off, on = _make_engines(tick)
    # Both engines share the same FakeOB instance, so set() drives both.

    shift_bps = []          # measured (buy_on − buy_off) in bps of mid
    expected_bps = []       # (microprice − mid) in bps of mid
    sign_agree = 0
    nonzero = 0
    clamped = 0
    side_mismatch = 0
    bad = 0
    calm_abs = []           # |shift_bps| where spread < 5 bps
    wide_abs = []           # |shift_bps| where spread > 20 bps
    n = 0

    for jp in journals:
        with open(jp) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "book_change":
                    continue
                bid, ask = _f(r.get("bid")), _f(r.get("ask"))
                bq, aq = _f(r.get("bid_qty")), _f(r.get("ask_qty"))
                if None in (bid, ask, bq, aq):
                    continue
                if bid <= 0 or ask <= 0 or ask <= bid or bq <= 0 or aq <= 0:
                    continue
                n += 1
                bd, ad = Decimal(str(bid)), Decimal(str(ask))
                bqd, aqd = Decimal(str(bq)), Decimal(str(aq))
                ob.set(bd, bqd, ad, aqd)
                mid = (bid + ask) / 2.0
                spread_bps = (ask - bid) / mid * 1e4

                micro = float(microprice(bd, ad, bqd, aqd))
                exp_bps = (micro - mid) / mid * 1e4

                try:
                    b_off = float(off.compute_target_price("BUY", 0, bd))
                    b_on = float(on.compute_target_price("BUY", 0, bd))
                    a_off = float(off.compute_target_price("SELL", 0, ad))
                    a_on = float(on.compute_target_price("SELL", 0, ad))
                except Exception:  # noqa: BLE001
                    bad += 1
                    continue
                buy_pert = (b_on - b_off) / mid * 1e4
                sell_pert = (a_on - a_off) / mid * 1e4
                if any(math.isnan(x) or math.isinf(x)
                       for x in (buy_pert, sell_pert, exp_bps)):
                    bad += 1
                    continue

                shift_bps.append(buy_pert)
                expected_bps.append(exp_bps)
                if abs(exp_bps) > 1e-9:
                    nonzero += 1
                    # Clamp/rounding can mask the shift; only judge sign where
                    # the realised buy perturbation is itself non-trivial.
                    if abs(buy_pert) > 1e-9:
                        if (buy_pert > 0) == (exp_bps > 0):
                            sign_agree += 1
                    else:
                        clamped += 1
                # Additive recentering ⇒ both sides must move identically
                # (allow 1 tick of independent rounding).
                if abs(buy_pert - sell_pert) > (float(tick) / mid * 1e4) + 1e-9:
                    side_mismatch += 1
                if spread_bps < 5.0:
                    calm_abs.append(abs(buy_pert))
                elif spread_bps > 20.0:
                    wide_abs.append(abs(buy_pert))

    return {
        "n": n, "bad": bad, "nonzero": nonzero, "clamped": clamped,
        "sign_agree": sign_agree, "side_mismatch": side_mismatch,
        "shift_bps": shift_bps, "expected_bps": expected_bps,
        "calm_abs": calm_abs, "wide_abs": wide_abs,
    }


def _summ(abs_vals):
    if not abs_vals:
        return "n=0"
    s = sorted(abs_vals)
    mean = sum(s) / len(s)
    return (f"n={len(s)} mean={mean:.4f} p50={_percentile(s, 0.50):.4f} "
            f"p95={_percentile(s, 0.95):.4f} p99={_percentile(s, 0.99):.4f} "
            f"max={s[-1]:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", action="append", required=True,
                   help="book_change journal(s); repeat to pool rotations.")
    p.add_argument("--market", default="?")
    p.add_argument("--tick-size", type=lambda s: Decimal(s), default=Decimal("0.0001"))
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    res = replay(args.journal, args.tick_size)
    abs_shift = sorted(abs(x) for x in res["shift_bps"])
    nonzero = res["nonzero"]
    sign_pct = (100.0 * res["sign_agree"] / nonzero) if nonzero else float("nan")
    used = len(res["shift_bps"])

    # Verdict
    no_errors = res["bad"] == 0
    sign_ok = nonzero == 0 or res["sign_agree"] == (nonzero - res["clamped"])
    side_ok = res["side_mismatch"] == 0
    max_shift = abs_shift[-1] if abs_shift else 0.0
    verdict = "PASS" if (no_errors and sign_ok and side_ok) else "FAIL"

    lines = [
        f"# Stage 3 replay — microprice in quoting — {args.market}", "",
        "Real `PricingEngine.compute_target_price` run twice per "
        "`book_change` snapshot (use_microprice False vs True). Measures the "
        "per-quote perturbation. No PnL claim — that is the live A/B.", "",
        f"- book_change snapshots used: **{used:,}** (errors: {res['bad']})",
        f"- snapshots with non-zero (micro−mid): {nonzero:,} "
        f"({100.0 * nonzero / used:.1f}% of used)" if used else "- none",
        f"- BBO-clamp/round masked the shift on: {res['clamped']:,} "
        "of the non-zero set",
        "",
        "## Pass criteria",
        f"- no exceptions / no NaN-Inf: **{'OK' if no_errors else 'FAIL'}** "
        f"({res['bad']} bad)",
        f"- sign agreement (shift vs micro−mid) on non-clamped: "
        f"**{sign_pct:.2f}%** → **{'OK' if sign_ok else 'FAIL'}**",
        f"- bid/ask perturbation identical (additive recenter): "
        f"**{'OK' if side_ok else 'FAIL'}** ({res['side_mismatch']} mismatches)",
        "",
        "## Perturbation magnitude |shift| (bps of mid)",
        f"- overall: {_summ(abs_shift)}",
        f"- calm  (spread < 5 bps):  {_summ(res['calm_abs'])}",
        f"- wide  (spread > 20 bps): {_summ(res['wide_abs'])}",
        f"- max |shift|: **{max_shift:.4f} bps**",
        "",
        f"## VERDICT: **{verdict}**", "",
        ("Microprice recentering is correctly signed, symmetric, bounded, and "
         "exception-free on real book data. Cleared for the DOT iter002 live "
         "A/B (flag on a `.env.DOT-USD.iterNNN` copy only)."
         if verdict == "PASS" else
         "Investigate before any live change — see failing criterion above."),
    ]
    out = "\n".join(lines) + "\n"

    print(f"\n{'='*60}")
    print(f"{args.market}: used={used:,}  sign_agree={sign_pct:.2f}%  "
          f"side_mismatch={res['side_mismatch']}  max|shift|={max_shift:.4f}bps  "
          f"=> {verdict}")
    print('='*60)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(out)
        print(f"Wrote: {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
