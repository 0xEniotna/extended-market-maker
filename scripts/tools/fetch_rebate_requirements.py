#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_maker.public_markets import (  # noqa: E402
    MAINNET_API_BASE,
    TESTNET_API_BASE,
    PublicMarketsClient,
)

_PAYOUT_FLOOR_USD = Decimal("10")
_DEFAULT_DAYS = Decimal("30")
_MARKET_STATS_PATH_TEMPLATE = "/info/markets/{market}/stats"
_REQUEST_TIMEOUT_S = 10
_USER_AGENT = "extended-market-maker/0.1"
_REBATE_TIERS = (
    {"share_pct": Decimal("0.5"), "rebate_pct": Decimal("0.002")},
    {"share_pct": Decimal("1"), "rebate_pct": Decimal("0.004")},
    {"share_pct": Decimal("2.5"), "rebate_pct": Decimal("0.008")},
    {"share_pct": Decimal("5"), "rebate_pct": Decimal("0.013")},
)


def _to_decimal(value: Any, *, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal(default)


def _parse_decimal_arg(raw: str, *, label: str) -> Decimal:
    value = raw.strip()
    if not value:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid {label}: {raw}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"{label} must be > 0")
    return parsed


def _resolve_api_base(api_base: Optional[str]) -> str:
    if api_base:
        return api_base.rstrip("/")

    explicit = os.getenv("EXTENDED_API_BASE", "").strip()
    if explicit:
        return explicit.rstrip("/")

    env = (os.getenv("MM_ENVIRONMENT") or os.getenv("EXTENDED_ENV") or "testnet").strip().lower()
    if env == "mainnet":
        return MAINNET_API_BASE
    return TESTNET_API_BASE


def _parse_market_filters(markets_arg: Optional[str], market_flags: Sequence[str]) -> List[str]:
    requested: List[str] = []
    seen: set[str] = set()

    candidates: List[str] = []
    if markets_arg:
        candidates.extend(markets_arg.split(","))
    candidates.extend(market_flags)

    for value in candidates:
        name = value.strip().upper()
        if not name or name in seen:
            continue
        seen.add(name)
        requested.append(name)
    return requested


def _parse_retry_after_seconds(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _request_json_with_retry(
    *,
    url: str,
    max_retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> Dict[str, Any]:
    attempt = 0
    while True:
        resp = requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=_REQUEST_TIMEOUT_S,
        )
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError:
            if resp.status_code != 429 or attempt >= max_retries:
                raise

            retry_after_s = _parse_retry_after_seconds(resp.headers.get("Retry-After"))
            if retry_after_s is None:
                retry_after_s = min(max_backoff_s, initial_backoff_s * (2 ** attempt))
            time.sleep(max(0.0, retry_after_s))
            attempt += 1
            continue

        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected payload (not a dict): {payload!r}")
        return payload


def _fetch_markets_with_retry(
    client: PublicMarketsClient,
    *,
    max_retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> List[Dict[str, Any]]:
    attempt = 0
    while True:
        try:
            return client.fetch_all_markets()
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 429 or attempt >= max_retries:
                raise

            retry_after_s = _parse_retry_after_seconds(
                exc.response.headers.get("Retry-After") if exc.response is not None else None
            )
            if retry_after_s is None:
                retry_after_s = min(max_backoff_s, initial_backoff_s * (2 ** attempt))

            time.sleep(max(0.0, retry_after_s))
            attempt += 1


def _fetch_market_stats(
    *,
    api_base: str,
    market: str,
    max_retries: int,
    initial_backoff_s: float,
    max_backoff_s: float,
) -> Dict[str, Any]:
    market_name = market.strip().upper()
    encoded_market = quote(market_name, safe="")
    url = f"{api_base}{_MARKET_STATS_PATH_TEMPLATE.format(market=encoded_market)}"
    payload = _request_json_with_retry(
        url=url,
        max_retries=max_retries,
        initial_backoff_s=initial_backoff_s,
        max_backoff_s=max_backoff_s,
    )

    status = str(payload.get("status", "")).lower()
    if status != "ok":
        raise RuntimeError(f"Extended API error for {market_name}: {payload!r}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected `data` for {market_name}: {data!r}")
    return data


def _tier_label(share_pct: Decimal) -> str:
    text = format(share_pct.normalize(), "f")
    text = text.rstrip("0").rstrip(".") if "." in text else text
    return f"{text}%"


def _build_market_tier_requirements(
    *,
    market_30d_volume_usd: Decimal,
    days: Decimal,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tier in _REBATE_TIERS:
        share_pct = tier["share_pct"]
        rebate_pct = tier["rebate_pct"]
        share_ratio = share_pct / Decimal("100")
        rebate_ratio = rebate_pct / Decimal("100")
        required_30d = market_30d_volume_usd * share_ratio
        required_daily = required_30d / days
        rows.append(
            {
                "tier": _tier_label(share_pct),
                "rebate_pct": rebate_pct,
                "required_30d_maker_volume_usd": required_30d,
                "required_avg_daily_maker_volume_usd": required_daily,
                "min_daily_volume_for_10usd_payout_usd": _PAYOUT_FLOOR_USD / rebate_ratio,
            }
        )
    return rows


def _build_market_result(
    *,
    market: str,
    daily_volume_usd: Decimal,
    days: Decimal,
) -> Dict[str, Any]:
    market_30d_volume_usd = daily_volume_usd * days
    return {
        "market": market,
        "daily_volume_usd": daily_volume_usd,
        "estimated_30d_volume_usd": market_30d_volume_usd,
        "tiers": _build_market_tier_requirements(
            market_30d_volume_usd=market_30d_volume_usd,
            days=days,
        ),
    }


def _fmt_usd(value: Decimal) -> str:
    return f"${value:,.2f}"


def _fmt_pct(value: Decimal, places: int = 3) -> str:
    return f"{value:.{places}f}%"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _build_payload(
    *,
    api_base: str,
    days: Decimal,
    requested_markets: Sequence[str],
    missing_markets: Sequence[str],
    market_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "api_base": api_base,
        "assumptions": {
            "window_days": days,
            "volume_source": "/info/markets/{market}/stats dailyVolume",
            "payout_floor_usd": _PAYOUT_FLOOR_USD,
        },
        "selection": {
            "requested_markets": list(requested_markets),
            "missing_markets": list(missing_markets),
            "selected_markets": len(market_results),
        },
        "markets": list(market_results),
    }


def _print_report(payload: Dict[str, Any]) -> None:
    assumptions = payload["assumptions"]
    selection = payload["selection"]
    markets = payload["markets"]

    print("Extended Maker Rebate Requirements (Per Market, USD)")
    print(f"Window: {assumptions['window_days']} days")
    print(f"Volume source: {assumptions['volume_source']}")
    print(f"Selected markets: {selection['selected_markets']}")

    if selection["requested_markets"]:
        print(f"Requested markets: {', '.join(selection['requested_markets'])}")
    if selection["missing_markets"]:
        print(f"Missing markets: {', '.join(selection['missing_markets'])}")

    for market in markets:
        print(
            "\n{}  24hVol={}  30dVol={}".format(
                market["market"],
                _fmt_usd(market["daily_volume_usd"]),
                _fmt_usd(market["estimated_30d_volume_usd"]),
            )
        )
        header = f"{'Tier':>6} {'Rebate':>8} {'Req 30d':>14} {'Req/day':>14}"
        print(header)
        print("-" * len(header))
        for row in market["tiers"]:
            print(
                f"{row['tier']:>6} "
                f"{_fmt_pct(row['rebate_pct'], 3):>8} "
                f"{_fmt_usd(row['required_30d_maker_volume_usd']):>14} "
                f"{_fmt_usd(row['required_avg_daily_maker_volume_usd']):>14}"
            )

    print("\n$10 payout floor (daily maker volume) by tier:")
    for tier in markets[0]["tiers"] if markets else []:
        print(
            "  rebate {} -> {}".format(
                _fmt_pct(tier["rebate_pct"], 3),
                _fmt_usd(tier["min_daily_volume_for_10usd_payout_usd"]),
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-market rebate eligibility volume requirements from "
            "GET /api/v1/info/markets/{market}/stats."
        ),
    )
    parser.add_argument(
        "--market",
        action="append",
        default=[],
        help="Market to include. Repeatable (e.g. --market ETH-USD --market MON-USD).",
    )
    parser.add_argument(
        "--markets",
        default=None,
        help="Comma-separated list of markets to include.",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="When no market list is given, include inactive markets too.",
    )
    parser.add_argument(
        "--days",
        type=lambda raw: _parse_decimal_arg(raw, label="days"),
        default=_DEFAULT_DAYS,
        help="Window length in days for 30d estimate from dailyVolume (default: 30).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override API base URL (e.g. https://api.starknet.extended.exchange/api/v1).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retry count for HTTP 429 responses (default: 4).",
    )
    parser.add_argument(
        "--initial-backoff-s",
        type=float,
        default=1.0,
        help="Initial retry delay in seconds for 429 retries (default: 1.0).",
    )
    parser.add_argument(
        "--max-backoff-s",
        type=float,
        default=16.0,
        help="Max retry delay in seconds for 429 retries (default: 16.0).",
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

    api_base = _resolve_api_base(args.api_base)
    max_retries = max(0, int(args.max_retries))
    initial_backoff_s = max(0.0, float(args.initial_backoff_s))
    max_backoff_s = max(initial_backoff_s, float(args.max_backoff_s))
    days = args.days

    requested_markets = _parse_market_filters(args.markets, args.market)
    markets_to_query: List[str]
    if requested_markets:
        markets_to_query = list(requested_markets)
    else:
        client = PublicMarketsClient(api_base=api_base)
        try:
            all_markets = _fetch_markets_with_retry(
                client,
                max_retries=max_retries,
                initial_backoff_s=initial_backoff_s,
                max_backoff_s=max_backoff_s,
            )
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 429:
                raise SystemExit(
                    "Extended API rate-limited this request (HTTP 429). "
                    "Retry shortly or increase --max-retries/--initial-backoff-s."
                ) from exc
            raise

        markets_to_query = []
        seen: set[str] = set()
        for market in all_markets:
            name = str(market.get("name", "")).strip().upper()
            if not name or name in seen:
                continue
            seen.add(name)
            status = str(market.get("status", ""))
            is_active = bool(market.get("active")) and status.upper() == "ACTIVE"
            if args.include_inactive or is_active:
                markets_to_query.append(name)

    if not markets_to_query:
        raise SystemExit("No markets to query.")

    missing_markets: List[str] = []
    market_results: List[Dict[str, Any]] = []
    for market in markets_to_query:
        try:
            stats = _fetch_market_stats(
                api_base=api_base,
                market=market,
                max_retries=max_retries,
                initial_backoff_s=initial_backoff_s,
                max_backoff_s=max_backoff_s,
            )
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code == 404:
                missing_markets.append(market)
                continue
            if status_code == 429:
                raise SystemExit(
                    "Extended API rate-limited this request (HTTP 429). "
                    "Retry shortly or increase --max-retries/--initial-backoff-s."
                ) from exc
            raise
        except RuntimeError:
            missing_markets.append(market)
            continue

        daily_volume_usd = _to_decimal(stats.get("dailyVolume"))
        market_results.append(
            _build_market_result(
                market=market,
                daily_volume_usd=daily_volume_usd,
                days=days,
            )
        )

    market_results.sort(key=lambda row: row["daily_volume_usd"], reverse=True)

    if not market_results:
        requested = ", ".join(markets_to_query)
        raise SystemExit(f"No market stats returned for requested markets: {requested}")

    payload = _build_payload(
        api_base=api_base,
        days=days,
        requested_markets=requested_markets if requested_markets else markets_to_query,
        missing_markets=missing_markets,
        market_results=market_results,
    )
    _print_report(payload)

    if args.json_out or args.json_stdout:
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
