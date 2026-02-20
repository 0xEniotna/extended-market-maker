#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_maker.public_markets import PublicMarketsClient  # noqa: E402


@dataclass
class MarketSnapshot:
    name: str
    asset_name: str
    spreads_bps: List[Decimal] = field(default_factory=list)
    vol_pcts: List[Decimal] = field(default_factory=list)
    open_interest: Decimal = Decimal("0")
    daily_volume: Decimal = Decimal("0")
    bid: Decimal = Decimal("0")
    ask: Decimal = Decimal("0")
    mid: Optional[Decimal] = None


def _to_decimal(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _fmt(value: Optional[Decimal], places: int = 2, commas: bool = True) -> str:
    if value is None:
        return "-"
    quant = Decimal("1").scaleb(-places)
    try:
        value = value.quantize(quant)
    except Exception:
        pass
    if commas:
        return f"{value:,.{places}f}"
    return f"{value:.{places}f}"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _vol_pct(stats: Dict[str, Any], mid: Optional[Decimal]) -> Optional[Decimal]:
    high = _to_decimal(stats.get("dailyHigh"))
    low = _to_decimal(stats.get("dailyLow"))
    if mid and mid > 0 and high > 0 and low > 0 and high >= low:
        return (high - low) / mid * Decimal("100")

    pct_raw = stats.get("dailyPriceChangePercentage")
    if pct_raw is None:
        return None
    pct = _to_decimal(pct_raw).copy_abs()
    if pct <= 1:
        return pct * Decimal("100")
    return pct


def _spread_bps(bid: Decimal, ask: Decimal, mid: Optional[Decimal]) -> Optional[Decimal]:
    if mid is None or mid <= 0:
        return None
    if bid <= 0 or ask <= 0 or ask <= bid:
        return None
    return (ask - bid) / mid * Decimal("10000")


def _percentile(values: List[Decimal], pct: Decimal) -> Optional[Decimal]:
    if not values:
        return None
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]

    rank = (len(ordered) - 1) * (float(pct) / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]

    weight = Decimal(str(rank - lo))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * weight


def _mean(values: List[Decimal]) -> Optional[Decimal]:
    if not values:
        return None
    return sum(values, Decimal("0")) / Decimal(len(values))


def _collect_samples(
    client: PublicMarketsClient,
    duration_s: float,
    interval_s: float,
) -> Tuple[Dict[str, MarketSnapshot], int, float]:
    samples_by_market: Dict[str, MarketSnapshot] = {}
    rounds = 0
    start = time.monotonic()
    duration_s = max(0.0, duration_s)
    interval_s = max(0.1, interval_s)
    deadline = start + duration_s

    while True:
        try:
            markets = client.fetch_all_markets()
            rounds += 1
        except Exception as exc:
            print(f"Warning: failed to fetch markets sample: {exc}", file=sys.stderr)
            markets = []

        for market in markets:
            if not market.get("active"):
                continue
            if str(market.get("status", "")).upper() != "ACTIVE":
                continue

            name = str(market.get("name", "")).strip()
            if not name:
                continue

            stats = market.get("marketStats") or {}
            bid = _to_decimal(stats.get("bidPrice"))
            ask = _to_decimal(stats.get("askPrice"))
            mark = _to_decimal(stats.get("markPrice"))
            mid = (bid + ask) / Decimal("2") if bid > 0 and ask > 0 else (mark if mark > 0 else None)
            spread_bps = _spread_bps(bid, ask, mid)
            if spread_bps is None:
                continue

            vol_pct = _vol_pct(stats, mid)
            daily_volume = _to_decimal(stats.get("dailyVolume"))
            open_interest = _to_decimal(stats.get("openInterest"))

            snapshot = samples_by_market.get(name)
            if snapshot is None:
                snapshot = MarketSnapshot(
                    name=name,
                    asset_name=str(market.get("assetName", "")),
                )
                samples_by_market[name] = snapshot

            snapshot.spreads_bps.append(spread_bps)
            if vol_pct is not None:
                snapshot.vol_pcts.append(vol_pct)
            snapshot.open_interest = open_interest
            snapshot.daily_volume = daily_volume
            snapshot.bid = bid
            snapshot.ask = ask
            snapshot.mid = mid

        if duration_s <= 0:
            break
        now = time.monotonic()
        if now >= deadline:
            break
        sleep_s = min(interval_s, max(0.0, deadline - now))
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            break

    elapsed_s = time.monotonic() - start
    return samples_by_market, rounds, elapsed_s


def _select_markets(
    sampled: Dict[str, MarketSnapshot],
    min_spread_bps: Decimal,
    max_spread_bps: Decimal,
    max_vol_pct: Decimal,
    min_daily_volume: Decimal,
    min_coverage_pct: Decimal,
    min_samples: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for snapshot in sampled.values():
        spreads = snapshot.spreads_bps
        sample_count = len(spreads)
        if sample_count < min_samples:
            continue

        spread_median = _percentile(spreads, Decimal("50"))
        spread_p90 = _percentile(spreads, Decimal("90"))
        spread_mean = _mean(spreads)
        if spread_median is None or spread_p90 is None or spread_mean is None:
            continue

        coverage_pct = Decimal("100")
        if min_spread_bps > 0:
            covered = sum(1 for s in spreads if s >= min_spread_bps)
            coverage_pct = (Decimal(covered) * Decimal("100")) / Decimal(sample_count)
            if coverage_pct < min_coverage_pct:
                continue

        if max_spread_bps > 0 and spread_median > max_spread_bps:
            continue

        vol_pct = _percentile(snapshot.vol_pcts, Decimal("50")) if snapshot.vol_pcts else None
        if max_vol_pct > 0 and vol_pct is not None and vol_pct > max_vol_pct:
            continue

        if min_daily_volume > 0 and snapshot.daily_volume < min_daily_volume:
            continue

        selected.append(
            {
                "name": snapshot.name,
                "asset_name": snapshot.asset_name,
                "_samples": sample_count,
                "_coverage_pct": coverage_pct,
                "_spread_median_bps": spread_median,
                "_spread_p90_bps": spread_p90,
                "_spread_mean_bps": spread_mean,
                "_vol_pct": vol_pct,
                "_open_interest": snapshot.open_interest,
                "_daily_volume": snapshot.daily_volume,
                "_bid": snapshot.bid,
                "_ask": snapshot.ask,
            }
        )

    return selected


def _sort_markets(markets: List[Dict[str, Any]], sort_by: str) -> None:
    if sort_by == "coverage":
        markets.sort(
            key=lambda m: (
                m.get("_coverage_pct", Decimal("0")),
                m.get("_open_interest", Decimal("0")),
            ),
            reverse=True,
        )
    elif sort_by == "spread_median":
        markets.sort(
            key=lambda m: (
                m.get("_spread_median_bps", Decimal("0")),
                m.get("_open_interest", Decimal("0")),
            ),
            reverse=True,
        )
    elif sort_by == "daily_volume":
        markets.sort(
            key=lambda m: (
                m.get("_daily_volume", Decimal("0")),
                m.get("_open_interest", Decimal("0")),
            ),
            reverse=True,
        )
    else:
        markets.sort(
            key=lambda m: (
                m.get("_open_interest", Decimal("0")),
                m.get("_coverage_pct", Decimal("0")),
                m.get("_spread_median_bps", Decimal("0")),
            ),
            reverse=True,
        )


def _print_table(markets: List[Dict[str, Any]], limit: int, min_spread_bps: Decimal) -> None:
    header = (
        f"{'#':>2}  {'Market':<12} {'Cov%':>7} {'Smed':>7} {'Sp90':>7} {'Savg':>7} {'N':>4} {'Vol%':>8} "
        f"{'OpenInt':>14} {'DailyVol':>14} {'Bid':>10} {'Ask':>10}"
    )
    lines = [header, "-" * len(header)]

    for i, market in enumerate(markets[:limit], start=1):
        bid = market.get("_bid")
        ask = market.get("_ask")
        coverage_pct = market.get("_coverage_pct")
        spread_median = market.get("_spread_median_bps")
        spread_p90 = market.get("_spread_p90_bps")
        spread_mean = market.get("_spread_mean_bps")
        samples = market.get("_samples")
        vol_pct = market.get("_vol_pct")
        open_interest = market.get("_open_interest")
        daily_volume = market.get("_daily_volume")

        lines.append(
            f"{i:>2}  {market.get('name', ''):<12}"
            f" {_fmt(coverage_pct, 1, commas=False):>7}"
            f" {_fmt(spread_median, 1, commas=False):>7}"
            f" {_fmt(spread_p90, 1, commas=False):>7}"
            f" {_fmt(spread_mean, 1, commas=False):>7}"
            f" {samples:>4}"
            f" {_fmt(vol_pct, 2, commas=False):>8}"
            f" {_fmt(open_interest, 2):>14}"
            f" {_fmt(daily_volume, 2):>14}"
            f" {_fmt(bid, 3, commas=False):>10}"
            f" {_fmt(ask, 3, commas=False):>10}"
        )

    lines.insert(0, f"Coverage = % samples where spread >= {min_spread_bps} bps")
    print("\n".join(lines))


def _build_json_payload(
    *,
    markets: List[Dict[str, Any]],
    sampled_count: int,
    rounds: int,
    elapsed_s: float,
    min_samples: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload_markets: List[Dict[str, Any]] = []
    for market in markets:
        payload_markets.append({
            "name": market.get("name"),
            "asset_name": market.get("asset_name"),
            "samples": market.get("_samples"),
            "coverage_pct": market.get("_coverage_pct"),
            "spread_median_bps": market.get("_spread_median_bps"),
            "spread_p90_bps": market.get("_spread_p90_bps"),
            "spread_mean_bps": market.get("_spread_mean_bps"),
            "vol_pct": market.get("_vol_pct"),
            "open_interest": market.get("_open_interest"),
            "daily_volume": market.get("_daily_volume"),
            "bid": market.get("_bid"),
            "ask": market.get("_ask"),
        })

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sampling": {
            "duration_s": args.duration_s,
            "interval_s": args.interval_s,
            "rounds": rounds,
            "elapsed_s": elapsed_s,
            "sampled_markets": sampled_count,
        },
        "filters": {
            "min_spread_bps": args.min_spread_bps,
            "max_spread_bps": args.max_spread_bps,
            "min_coverage_pct": args.min_coverage_pct,
            "max_vol_pct": args.max_vol_pct,
            "min_daily_volume": args.min_daily_volume,
            "min_samples": min_samples,
            "sort_by": args.sort_by,
        },
        "matched_markets": len(payload_markets),
        "markets": payload_markets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find Extended markets suitable for fee-based market making using rolling "
            "spread sampling (sustained spread, not a single snapshot)."
        )
    )
    parser.add_argument("--limit", type=int, default=40, help="Max rows to show")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=120.0,
        help="Sampling window in seconds (0 = single snapshot)",
    )
    parser.add_argument(
        "--interval-s",
        type=float,
        default=2.0,
        help="Delay between market snapshots in seconds",
    )
    parser.add_argument(
        "--min-spread-bps",
        type=Decimal,
        default=Decimal("8"),
        help="Spread threshold in bps used for coverage scoring",
    )
    parser.add_argument(
        "--max-spread-bps",
        type=Decimal,
        default=Decimal("50"),
        help="Maximum accepted median spread in bps (0 disables)",
    )
    parser.add_argument(
        "--min-coverage-pct",
        type=Decimal,
        default=Decimal("70"),
        help="Minimum %% of samples with spread >= min-spread-bps",
    )
    parser.add_argument(
        "--max-vol-pct",
        type=Decimal,
        default=Decimal("8"),
        help="Maximum median daily high/low range as percent of mid (0 disables)",
    )
    parser.add_argument(
        "--min-daily-volume",
        type=Decimal,
        default=Decimal("0"),
        help="Minimum daily volume (collateral) (0 disables)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=0,
        help="Minimum spread samples per market (0 = auto, 60%% of successful rounds)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["open_interest", "coverage", "spread_median", "daily_volume"],
        default="open_interest",
        help="Sort key for matched markets",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (e.g. https://api.starknet.extended.exchange/api/v1)",
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

    load_dotenv()

    if args.api_base:
        client = PublicMarketsClient(api_base=str(args.api_base).rstrip("/"))
    else:
        client = PublicMarketsClient.default()

    sampled, rounds, elapsed_s = _collect_samples(
        client,
        duration_s=args.duration_s,
        interval_s=args.interval_s,
    )
    if rounds == 0:
        print("No successful market snapshots collected.")
        if args.json_out or args.json_stdout:
            payload = _build_json_payload(
                markets=[],
                sampled_count=0,
                rounds=0,
                elapsed_s=elapsed_s,
                min_samples=max(1, args.min_samples),
                args=args,
            )
            json_text = json.dumps(_to_jsonable(payload), indent=2) + "\n"
            if args.json_out:
                out_path = Path(args.json_out)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json_text)
            if args.json_stdout:
                print(json_text, end="")
        return 1

    min_samples = args.min_samples if args.min_samples > 0 else max(1, int(rounds * 0.6))

    filtered = _select_markets(
        sampled,
        min_spread_bps=args.min_spread_bps,
        max_spread_bps=args.max_spread_bps,
        max_vol_pct=args.max_vol_pct,
        min_daily_volume=args.min_daily_volume,
        min_coverage_pct=args.min_coverage_pct,
        min_samples=min_samples,
    )
    _sort_markets(filtered, sort_by=args.sort_by)

    print(
        "Sampling: duration_s={} interval_s={} rounds={} elapsed_s={:.1f}"
        .format(
            args.duration_s,
            args.interval_s,
            rounds,
            elapsed_s,
        )
    )
    print(
        "Filters: min_spread_bps={} min_coverage_pct={} max_spread_bps={} max_vol_pct={} "
        "min_daily_volume={} min_samples={} sort_by={}"
        .format(
            args.min_spread_bps,
            args.min_coverage_pct,
            args.max_spread_bps,
            args.max_vol_pct,
            args.min_daily_volume,
            min_samples,
            args.sort_by,
        )
    )
    print(f"Matched markets: {len(filtered)} / sampled_markets={len(sampled)}")
    _print_table(filtered, args.limit, args.min_spread_bps)

    if args.json_out or args.json_stdout:
        payload = _build_json_payload(
            markets=filtered,
            sampled_count=len(sampled),
            rounds=rounds,
            elapsed_s=elapsed_s,
            min_samples=min_samples,
            args=args,
        )
        json_text = json.dumps(_to_jsonable(payload), indent=2) + "\n"
        if args.json_out:
            out_path = Path(args.json_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_text)
        if args.json_stdout:
            print(json_text, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
