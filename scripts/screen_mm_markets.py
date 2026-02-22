#!/usr/bin/env python
"""
Screen Extended Exchange markets for market-making suitability.

Samples active markets over a time window, computes spread/tick/volume
metrics, and ranks them by a composite score.

Usage:
    PYTHONPATH=src python scripts/screen_mm_markets.py --duration-s 180 --interval-s 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dotenv import load_dotenv  # noqa: E402

from market_maker.public_markets import PublicMarketsClient  # noqa: E402


def _d(v, default: str = "0") -> Decimal:
    try:
        return Decimal(str(v))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _get(d: dict, *keys, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _percentile(values: list[Decimal], pct: Decimal) -> Decimal:
    if not values:
        return Decimal("0")
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = Decimal(str(rank - lo))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * weight


def _to_jsonable(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def screen_markets(
    duration_s: float = 180.0,
    interval_s: float = 2.0,
    api_base: str | None = None,
) -> tuple[list[dict], int, float]:
    client = (
        PublicMarketsClient.default()
        if not api_base
        else PublicMarketsClient(api_base=api_base.rstrip("/"))
    )

    duration_s = max(0.0, float(duration_s))
    interval_s = max(0.1, float(interval_s))
    t0 = time.monotonic()
    deadline = t0 + duration_s
    rounds = 0

    by_market: dict[str, dict] = {}

    while True:
        try:
            raw_markets = client.fetch_all_markets()
            rounds += 1
        except Exception as exc:
            print(f"Warning: snapshot fetch failed: {exc}", file=sys.stderr)
            raw_markets = []

        for m in raw_markets:
            if m.get("status") != "ACTIVE" or not m.get("active"):
                continue

            name = m.get("name", "")
            if not name:
                continue

            stats = m.get("marketStats") or {}
            tc = m.get("tradingConfig") or {}

            bid = _d(_get(stats, "bidPrice", "bid_price"))
            ask = _d(_get(stats, "askPrice", "ask_price"))
            mark = _d(_get(stats, "markPrice", "mark_price"))
            daily_vol = _d(_get(stats, "dailyVolume"))
            oi = _d(_get(stats, "openInterest"))
            tick = _d(_get(tc, "minPriceChange", "min_price_change"), "0.01")
            min_size = _d(_get(tc, "minOrderSize", "min_order_size"), "0.001")
            min_size_change = _d(
                _get(tc, "minOrderSizeChange", "min_order_size_change"),
                "0.001",
            )

            if bid <= 0 or ask <= 0 or mark <= 0:
                continue

            mid = (bid + ask) / 2
            spread = ask - bid
            spread_bps = spread / mid * Decimal("10000")
            tick_bps = tick / mid * Decimal("10000")
            ticks_in_spread = spread / tick if tick > 0 else Decimal("0")
            min_notional = min_size * mid

            rec = by_market.setdefault(
                name,
                {
                    "name": name,
                    "spread_samples_bps": [],
                    "ticks_samples": [],
                    "mid_samples": [],
                    "bid": bid,
                    "ask": ask,
                    "tick_bps": tick_bps,
                    "tick": tick,
                    "min_size": min_size,
                    "min_size_change": min_size_change,
                    "min_notional": min_notional,
                    "daily_vol": daily_vol,
                    "oi": oi,
                },
            )

            rec["spread_samples_bps"].append(spread_bps)
            rec["ticks_samples"].append(ticks_in_spread)
            rec["mid_samples"].append(mid)
            # Keep latest values for display fields.
            rec["bid"] = bid
            rec["ask"] = ask
            rec["tick_bps"] = tick_bps
            rec["tick"] = tick
            rec["min_size"] = min_size
            rec["min_size_change"] = min_size_change
            rec["min_notional"] = min_notional
            rec["daily_vol"] = daily_vol
            rec["oi"] = oi

        if duration_s <= 0:
            break
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(interval_s, deadline - now))

    results: list[dict] = []
    for rec in by_market.values():
        spreads: list[Decimal] = rec["spread_samples_bps"]
        ticks: list[Decimal] = rec["ticks_samples"]
        mids: list[Decimal] = rec["mid_samples"]
        if not spreads or not ticks or not mids:
            continue

        n = len(spreads)
        spread_median = _percentile(spreads, Decimal("50"))
        spread_p90 = _percentile(spreads, Decimal("90"))
        spread_mean = sum(spreads, Decimal("0")) / Decimal(n)
        ticks_median = _percentile(ticks, Decimal("50"))
        mid = _percentile(mids, Decimal("50"))
        cov_3bps_pct = (
            Decimal(sum(1 for s in spreads if s >= Decimal("3"))) * Decimal("100")
        ) / Decimal(n)

        results.append(
            {
                "name": rec["name"],
                "n_samples": n,
                "mid": mid,
                "bid": rec["bid"],
                "ask": rec["ask"],
                "spread_bps": spread_median,
                "spread_bps_p90": spread_p90,
                "spread_bps_mean": spread_mean,
                "coverage_3bps_pct": cov_3bps_pct,
                "tick_bps": rec["tick_bps"],
                "ticks_in_spread": ticks_median,
                "tick": rec["tick"],
                "min_size": rec["min_size"],
                "min_size_change": rec["min_size_change"],
                "min_notional": rec["min_notional"],
                "daily_vol": rec["daily_vol"],
                "oi": rec["oi"],
            }
        )

    elapsed_s = time.monotonic() - t0
    return results, rounds, elapsed_s


def score_market(m: dict) -> Decimal:
    """
    Composite score for MM suitability. Higher = better.

    Uses aggregated spread stats (median + p90) to reduce single-snapshot noise.
    """
    spread = float(m["spread_bps"])
    spread_p90 = float(m["spread_bps_p90"])
    ticks = float(m["ticks_in_spread"])
    vol = float(m["daily_vol"])
    oi = float(m["oi"])
    cov_3bps = float(m["coverage_3bps_pct"])

    if vol < 1000:
        return Decimal("-1")  # Too illiquid

    # Spread score: peaks around 5-30 bps, penalize <1 and >200
    if spread < 0.5:
        spread_score = 0.0
    elif spread <= 50:
        spread_score = min(spread / 5, 5.0)  # Up to 5 points
    else:
        spread_score = max(0.0, 5.0 - (spread - 50) / 50)  # Decay

    # Penalize unstable regimes with very spiky p90.
    if spread > 0 and spread_p90 > spread * 2.5:
        spread_score *= 0.8

    # Granularity score: more ticks in spread = better
    tick_score = min(ticks / 2, 3.0)  # Up to 3 points

    # Volume and OI score (log scale)
    vol_score = min(math.log10(max(vol, 1)) - 3, 4.0)  # 0 at $1k, 4 at $10M
    oi_score = min(math.log10(max(oi, 1)) - 3, 3.0)  # 0 at $1k, 3 at $1M

    # Reward sustained spread availability.
    cov_score = max(0.0, min(cov_3bps / 25.0, 4.0))  # 0..4

    total = spread_score + tick_score + vol_score + oi_score + cov_score
    return Decimal(str(round(total, 2)))


def _build_json_payload(markets: list[dict], args, rounds: int, elapsed_s: float) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sampling": {
            "duration_s": args.duration_s,
            "interval_s": args.interval_s,
            "rounds": rounds,
            "elapsed_s": elapsed_s,
        },
        "count": len(markets),
        "markets": markets,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Screen MM markets using rolling spread sampling.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=180.0,
        help="Sampling duration in seconds (default: 180)",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=2.0,
        help="Delay between snapshots in seconds (default: 2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=80,
        help="Maximum rows to print (default: 80)",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write structured JSON output.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Print structured JSON payload to stdout.",
    )
    args = parser.parse_args()

    markets, rounds, elapsed_s = screen_markets(
        duration_s=args.duration_s,
        interval_s=args.interval_s,
        api_base=args.api_base,
    )
    if not markets:
        print("No active markets found.")
        if args.json_out or args.json_stdout:
            payload = _build_json_payload([], args, rounds, elapsed_s)
            json_text = json.dumps(_to_jsonable(payload), indent=2) + "\n"
            if args.json_out:
                out_path = Path(args.json_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json_text)
            if args.json_stdout:
                print(json_text, end="")
        return

    for m in markets:
        m["score"] = score_market(m)
    markets.sort(key=lambda x: x["score"], reverse=True)

    header = (
        f"{'#':>2}  {'Market':<12} {'Mid':>10} {'SprdMed':>9} {'SprdP90':>9} {'Cov3%':>7} "
        f"{'Tick':>8} {'Tks/Sprd':>8} {'N':>4} {'MinSize':>8} {'MinNotl':>8} "
        f"{'24hVol':>12} {'OI':>12} {'Score':>6}"
    )
    print("Market Making Suitability Screen — Extended Exchange")
    print(
        f"Sampling: duration={args.duration_s:.0f}s interval={args.interval_s:.1f}s "
        f"rounds={rounds} elapsed={elapsed_s:.1f}s"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    shown = 0
    for i, m in enumerate(markets, 1):
        if m["score"] < 0:
            continue
        shown += 1
        if shown > args.limit:
            break
        print(
            f"{i:>2}  {m['name']:<12} {m['mid']:>10.2f} "
            f"{m['spread_bps']:>8.1f}bp "
            f"{m['spread_bps_p90']:>8.1f}bp "
            f"{m['coverage_3bps_pct']:>6.0f}% "
            f"{m['tick_bps']:>7.2f}bp "
            f"{m['ticks_in_spread']:>8.1f} "
            f"{m['n_samples']:>4} "
            f"{m['min_size']:>8} "
            f"${m['min_notional']:>7.0f} "
            f"${m['daily_vol']:>11,.0f} "
            f"${m['oi']:>11,.0f} "
            f"{m['score']:>6}"
        )

    print()
    print("Legend:")
    print("  SprdMed   — median spread in bps over sampling window")
    print("  SprdP90   — 90th percentile spread in bps (spike indicator)")
    print("  Cov3%     — % of samples with spread >= 3 bps")
    print("  Tick      — tick size in bps (smaller = finer price granularity)")
    print("  Tks/Sprd  — median ticks fitting in the spread")
    print("  MinNotl   — minimum order notional in USD")
    print("  Score     — composite MM suitability (higher = better)")

    payload_markets = []
    for m in markets:
        payload_markets.append({
            "name": m["name"],
            "n_samples": m["n_samples"],
            "mid": m["mid"],
            "bid": m["bid"],
            "ask": m["ask"],
            "spread_bps": m["spread_bps"],
            "spread_bps_p90": m["spread_bps_p90"],
            "spread_bps_mean": m["spread_bps_mean"],
            "coverage_3bps_pct": m["coverage_3bps_pct"],
            "tick_bps": m["tick_bps"],
            "ticks_in_spread": m["ticks_in_spread"],
            "tick": m["tick"],
            "min_size": m["min_size"],
            "min_size_change": m["min_size_change"],
            "min_notional": m["min_notional"],
            "daily_vol": m["daily_vol"],
            "oi": m["oi"],
            "score": m["score"],
        })
    payload = _build_json_payload(payload_markets, args, rounds, elapsed_s)
    json_text = json.dumps(_to_jsonable(payload), indent=2) + "\n"
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_text)
    if args.json_stdout:
        print(json_text, end="")

    out = PROJECT_ROOT / "data" / "mm_market_screen.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for m in markets:
            if m["score"] >= 0:
                f.write(
                    f"{m['name']}\t"
                    f"spread_med={m['spread_bps']:.1f}bps\t"
                    f"spread_p90={m['spread_bps_p90']:.1f}bps\t"
                    f"cov3={m['coverage_3bps_pct']:.0f}%\t"
                    f"ticks_med={m['ticks_in_spread']:.0f}\t"
                    f"vol=${m['daily_vol']:,.0f}\t"
                    f"score={m['score']}\n"
                )
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
