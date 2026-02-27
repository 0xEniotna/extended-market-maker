#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_report_common import (  # noqa: E402
    compute_time_above_util,
    discover_market_journals,
    iso_utc,
    load_do_not_restart,
    load_market_env,
    load_recent_entries,
    normalize_market_name,
    parse_env_map,
    pct,
    quantile_average,
    rolling_price_change,
    round_or_none,
    to_float,
    truncate_markdown,
)
from mm_audit_common import write_json  # noqa: E402


def _env_value(env_map: Dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in env_map and env_map[key]:
            return env_map[key]
    return None


def _main_flag(flags: List[str]) -> str:
    if "accumulating_position" in flags:
        return "🔴 Accumulating"
    if "one_sided_fills" in flags:
        return "🔴 Fill bias"
    if "stuck_position" in flags:
        return "🔴 Stuck"
    if "high_exposure" in flags:
        return "🔴 Exposure"
    if "elevated_inventory" in flags:
        return "⚠️ Inventory"
    return "✅"


def _trend_label(delta: float) -> str:
    if delta > 0:
        return "↑ long"
    if delta < 0:
        return "↓ short"
    return "→ flat"


def _render_report(rows: List[Dict[str, Any]], generated_at: str) -> str:
    lines: List[str] = []
    lines.append(f"## Position Risk — {generated_at}")
    lines.append("| Market | Position | Util% | Trend | Fill Bal | Exposure | Flag |")
    lines.append("|--------|----------|-------|-------|----------|----------|------|")
    for row in rows:
        lines.append(
            f"| {row['market']} | {row['current_position']:.2f} | {row['utilization_pct']:.0f}% | "
            f"{row['trend_label']} | {row['fill_balance_pct']:.0f}% BUY | "
            f"${row['directional_exposure_usd']:+.0f} | {row['main_flag']} |"
        )

    recs = [rec for row in rows for rec in row["recommendations"]]
    if recs:
        lines.append("")
        lines.append("### Recommendations")
        for rec in recs[:8]:
            lines.append(f"- {rec}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    now_ts = time.time()
    min_ts = now_ts - (args.lookback_hours * 3600.0)
    journal_dir = Path(args.journal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_markets = load_do_not_restart(Path(args.do_not_restart_file))
    env_map = parse_env_map(args.env_map)
    journals = discover_market_journals(journal_dir)

    results: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []

    for market, journal_path in journals.items():
        norm_market = normalize_market_name(market)
        if norm_market in skip_markets:
            continue

        events = load_recent_entries(journal_path, min_ts)
        if not events:
            continue

        position_samples: List[Tuple[float, float]] = []
        fills: List[Dict[str, Any]] = []
        for event in events:
            ts = to_float(event.get("ts"))
            if ts is None:
                continue
            position = to_float(event.get("position"))
            if position is not None:
                position_samples.append((ts, position))
            if str(event.get("type")) == "fill":
                fills.append(event)

        if not position_samples and not fills:
            continue

        _, env_values = load_market_env(norm_market, env_map)
        max_position_value = _env_value(
            env_values,
            "MM_MAX_POSITION_SIZE",
            "MAX_POSITION_SIZE",
            "max_position_size",
        )
        max_position_size = abs(float(max_position_value)) if max_position_value is not None else 0.0

        current_position = position_samples[-1][1] if position_samples else 0.0
        min_position = min((sample[1] for sample in position_samples), default=current_position)
        max_position_seen = max((sample[1] for sample in position_samples), default=current_position)

        ordered_positions = [sample[1] for sample in position_samples]
        if len(ordered_positions) >= 4:
            first_avg = quantile_average(ordered_positions, 0.25)
            last_avg = quantile_average(list(reversed(ordered_positions)), 0.25)
            trend_delta = last_avg - first_avg
        elif ordered_positions:
            trend_delta = ordered_positions[-1] - ordered_positions[0]
        else:
            trend_delta = 0.0
        trend_label = _trend_label(trend_delta)

        total_fills = len(fills)
        buy_fills = 0
        buy_qty = 0.0
        total_qty = 0.0
        for fill in fills:
            side = str(fill.get("side") or "").upper()
            qty = abs(to_float(fill.get("qty")) or 0.0)
            total_qty += qty
            if "BUY" in side:
                buy_fills += 1
                buy_qty += qty
        fill_balance = (buy_fills / total_fills) if total_fills > 0 else 0.5
        fill_volume_balance = (buy_qty / total_qty) if total_qty > 0 else 0.5

        utilization_pct = (abs(current_position) / max_position_size * 100.0) if max_position_size > 0 else 0.0
        time_above_50_s = compute_time_above_util(
            position_samples,
            max_position=max_position_size if max_position_size > 0 else 0.0,
            threshold=0.50,
            window_end_ts=now_ts,
        )

        price_change = rolling_price_change(events)
        directional_exposure_usd = current_position * price_change

        trending_away = False
        if max_position_size > 0 and ordered_positions:
            start_abs = abs(ordered_positions[0])
            end_abs = abs(ordered_positions[-1])
            trending_away = end_abs > (start_abs + max_position_size * 0.02)
        elif ordered_positions:
            trending_away = abs(ordered_positions[-1]) > abs(ordered_positions[0])

        position_span = (max_position_seen - min_position)
        stuck_position = (
            max_position_size > 0
            and utilization_pct > 30.0
            and position_span <= (max_position_size * 0.05)
        )

        flags: List[str] = []
        if utilization_pct > 70.0 and trending_away:
            flags.append("accumulating_position")
        if fill_balance < 0.30 or fill_balance > 0.70:
            flags.append("one_sided_fills")
        if stuck_position:
            flags.append("stuck_position")
        if utilization_pct > 50.0:
            flags.append("elevated_inventory")
        if abs(directional_exposure_usd) > 200.0:
            flags.append("high_exposure")

        recommendations: List[str] = []
        if "accumulating_position" in flags:
            recommendations.append(
                f"{norm_market}: Position accumulating at {utilization_pct:.1f}% — increase "
                "INVENTORY_SKEW_FACTOR or inspect side blocking with audit_reprice_quality.py."
            )
        if "stuck_position" in flags:
            recommendations.append(
                f"{norm_market}: Position appears stuck. Check market activity and consider manual close via close_mm_position.py."
            )
        if "one_sided_fills" in flags:
            dominant_side = "BUY" if fill_balance > 0.5 else "SELL"
            skew = _env_value(env_values, "MM_INVENTORY_SKEW_FACTOR", "INVENTORY_SKEW_FACTOR", "inventory_skew_factor")
            if skew is None:
                recommendations.append(
                    f"{norm_market}: {fill_balance * 100:.0f}% of fills are {dominant_side}. Skew system may be insufficient."
                )
            else:
                recommendations.append(
                    f"{norm_market}: {fill_balance * 100:.0f}% of fills are {dominant_side}. "
                    f"Current INVENTORY_SKEW_FACTOR={skew}."
                )
        if "high_exposure" in flags:
            recommendations.append(
                f"{norm_market}: Open directional exposure is ${directional_exposure_usd:+.0f}. "
                "Reduce inventory or tighten risk controls."
            )
        if "elevated_inventory" in flags and "accumulating_position" not in flags:
            recommendations.append(f"{norm_market}: Inventory utilization elevated at {utilization_pct:.1f}%.")
        if not recommendations:
            recommendations.append(f"{norm_market}: Position risk metrics are stable.")
        recommendations = list(dict.fromkeys(recommendations))

        main_flag = _main_flag(flags)
        results[norm_market] = {
            "current_position": round_or_none(current_position, 6),
            "position_range": {
                "min": round_or_none(min_position, 6),
                "max": round_or_none(max_position_seen, 6),
            },
            "position_trend_delta": round_or_none(trend_delta, 6),
            "fill_balance": round_or_none(fill_balance, 6),
            "fill_volume_balance": round_or_none(fill_volume_balance, 6),
            "fill_count": total_fills,
            "inventory_utilization_pct": round_or_none(utilization_pct, 3),
            "time_above_50pct_s": round_or_none(time_above_50_s, 3),
            "directional_exposure_usd": round_or_none(directional_exposure_usd, 3),
            "max_position_size": round_or_none(max_position_size, 6) if max_position_size > 0 else None,
            "flags": flags,
            "recommendations": recommendations,
        }

        rows.append({
            "market": norm_market,
            "current_position": current_position,
            "utilization_pct": utilization_pct,
            "trend_label": trend_label,
            "fill_balance_pct": fill_balance * 100.0,
            "directional_exposure_usd": directional_exposure_usd,
            "main_flag": main_flag,
            "recommendations": recommendations,
        })

    rows.sort(
        key=lambda row: (
            0 if row["main_flag"].startswith("🔴") else 1 if row["main_flag"].startswith("⚠️") else 2,
            row["market"],
        )
    )

    payload = {
        "timestamp": iso_utc(now_ts),
        "lookback_hours": args.lookback_hours,
        "markets": results,
    }
    write_json(output_dir / "position_risk.json", payload)

    report = _render_report(rows, iso_utc(now_ts))
    report = truncate_markdown(report, args.char_limit)
    print(report, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit inventory and position risk per market.")
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument("--output-dir", default="data/mm_audit", help="Output directory for JSON artifacts.")
    parser.add_argument("--lookback-hours", type=float, default=2.0, help="Lookback window in hours.")
    parser.add_argument("--do-not-restart-file", default="data/do_not_restart.txt", help="Markets to skip.")
    parser.add_argument(
        "--env-map",
        default=None,
        help=(
            "Optional env mapping. JSON object, file path, or CSV entries "
            "(e.g. 'ETH-USD=.env.eth,MON-USD=.env.mon')."
        ),
    )
    parser.add_argument("--char-limit", type=int, default=1500, help="Maximum markdown output length.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"audit_position_risk error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
