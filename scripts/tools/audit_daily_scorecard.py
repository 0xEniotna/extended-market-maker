#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_report_common import (  # noqa: E402
    collect_mid_series,
    compute_time_above_util,
    discover_market_journals,
    iso_utc,
    load_do_not_restart,
    load_market_env,
    load_recent_entries,
    median,
    mid_at_or_after,
    normalize_market_name,
    parse_env_map,
    pct,
    round_or_none,
    to_float,
    truncate_markdown,
    zero_crossings,
)
from mm_audit_common import append_jsonl, write_json  # noqa: E402

PNL_MARKET_RE = re.compile(
    r"^\s*\d+\.\s+([A-Z0-9_-]+)\s+total=([-0-9.]+)\s+closed=([-0-9.]+)\s+open_realized=([-0-9.]+)\s+open_unrealized=([-0-9.]+)",
    re.MULTILINE,
)


def _env_value(env_map: Dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in env_map and env_map[key]:
            return env_map[key]
    return None


def _fetch_total_pnl_snapshot(now_ts: float, lookback_hours: float) -> Dict[str, Any]:
    since_ts = datetime.fromtimestamp(now_ts - lookback_hours * 3600.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "scripts" / "tools" / "fetch_total_pnl.py"),
        "--env",
        ".env.eth",
        "--since",
        since_ts,
        "--include-preexisting-open",
    ]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "fetch_total_pnl failed")

    text = proc.stdout
    total_match = re.search(r"total_pnl=([-0-9.]+)\s+USD", text)
    if total_match is None:
        raise RuntimeError("Unable to parse total_pnl from fetch_total_pnl output")

    start_match = re.search(r"starting_equity:\s+([-0-9.]+)\s+USD", text)
    return_match = re.search(r"total_return:\s+([-0-9.]+)%", text)

    per_market: Dict[str, float] = {}
    for market, total, _closed, _open_realized, _open_unrealized in PNL_MARKET_RE.findall(text):
        per_market[normalize_market_name(market)] = float(total)

    fleet_pnl = float(total_match.group(1))
    start_equity = float(start_match.group(1)) if start_match else None
    total_return_pct = float(return_match.group(1)) if return_match else None
    current_equity = (start_equity + fleet_pnl) if start_equity is not None else None
    return {
        "source": "fetch_total_pnl",
        "fleet_pnl": fleet_pnl,
        "equity": current_equity,
        "delta_pct": total_return_pct,
        "per_market_pnl": per_market,
    }


def _journal_fallback_snapshot(markets_payload: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_market: Dict[str, float] = {}
    for market, payload in markets_payload.items():
        per_market[market] = payload.get("edge_pnl_estimate_usd", 0.0)
    fleet = sum(per_market.values())
    return {
        "source": "journal_edge_estimate",
        "fleet_pnl": fleet,
        "equity": None,
        "delta_pct": None,
        "per_market_pnl": per_market,
    }


def _summarize_verdict(
    *,
    market_pnl: float,
    adverse_pct: float,
    uptime_pct: float,
    peak_util_pct: float,
    negative_edge_pct: float,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    severe = (
        market_pnl < 0
        or adverse_pct >= 65.0
        or uptime_pct < 25.0
        or peak_util_pct >= 70.0
        or negative_edge_pct >= 40.0
    )
    marginal = (
        adverse_pct >= 55.0
        or uptime_pct < 40.0
        or peak_util_pct >= 50.0
        or negative_edge_pct >= 25.0
        or market_pnl < 10.0
    )

    if market_pnl < 0:
        reasons.append("losing money")
    if adverse_pct >= 65.0:
        reasons.append("high adverse selection")
    if uptime_pct < 25.0:
        reasons.append("very low quoting uptime")
    if peak_util_pct >= 70.0:
        reasons.append("high inventory utilization")

    if severe:
        return ("🔴", reasons)
    if marginal:
        return ("⚠️", reasons or ["marginal quality metrics"])
    return ("🟢", reasons or ["healthy"])


def _render_report(
    generated_at: str,
    fleet_snapshot: Dict[str, Any],
    rows: List[Dict[str, Any]],
    winners: List[str],
    problems: List[str],
    recommendations: List[str],
) -> str:
    fleet_pnl = fleet_snapshot.get("fleet_pnl")
    equity = fleet_snapshot.get("equity")
    delta_pct = fleet_snapshot.get("delta_pct")
    source = fleet_snapshot.get("source", "unknown")

    lines: List[str] = []
    lines.append(f"## Daily Scorecard — {generated_at[:10]}")
    headline = f"**Fleet PnL: ${fleet_pnl:+.0f}**"
    if equity is not None:
        headline += f" | Equity: ${equity:,.0f}"
    if delta_pct is not None:
        headline += f" | Δ: {delta_pct:+.2f}%"
    headline += f" ({source})"
    lines.append(headline)
    lines.append("")
    lines.append("| Market | PnL | Fills | Edge | Adv% | Uptime% | Peak Util | Verdict |")
    lines.append("|--------|-----|-------|------|------|---------|-----------|---------|")
    for row in rows:
        lines.append(
            f"| {row['market']} | ${row['pnl']:+.0f} | {row['fills']} | {row['edge_bps']:.2f}bps | "
            f"{row['adverse_pct']:.0f}% | {row['uptime_pct']:.0f}% | {row['peak_util_pct']:.0f}% | {row['verdict']} |"
        )

    if winners:
        lines.append("")
        lines.append("### Winners")
        for winner in winners[:3]:
            lines.append(f"- {winner}")

    if problems:
        lines.append("")
        lines.append("### Problems")
        for issue in problems[:4]:
            lines.append(f"- {issue}")

    if recommendations:
        lines.append("")
        lines.append("### Recommendations")
        for idx, recommendation in enumerate(recommendations[:5], start=1):
            lines.append(f"{idx}. {recommendation}")

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

    per_market_payload: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    winners: List[str] = []
    problems: List[str] = []
    recommendations: List[str] = []

    for market, journal_path in journals.items():
        norm_market = normalize_market_name(market)
        if norm_market in skip_markets:
            continue

        events = load_recent_entries(journal_path, min_ts)
        if not events:
            continue

        fills = [event for event in events if str(event.get("type")) == "fill"]
        reprice = [event for event in events if str(event.get("type")) == "reprice_decision"]

        fill_count = len(fills)
        fills_per_hour = fill_count / max(args.lookback_hours, 1e-6)
        buy_count = 0
        sell_count = 0
        buy_qty = 0.0
        sell_qty = 0.0
        edge_values: List[float] = []
        negative_edge_count = 0
        edge_pnl_estimate_usd = 0.0

        for fill in fills:
            side = str(fill.get("side") or "").upper()
            qty = abs(to_float(fill.get("qty")) or 0.0)
            price = to_float(fill.get("price")) or 0.0
            edge_bps = to_float(fill.get("edge_bps")) or 0.0
            if "BUY" in side:
                buy_count += 1
                buy_qty += qty
            else:
                sell_count += 1
                sell_qty += qty
            edge_values.append(edge_bps)
            if edge_bps < 0:
                negative_edge_count += 1
            edge_pnl_estimate_usd += (qty * price) * (edge_bps / 10000.0)

        median_edge_bps = median(edge_values) if edge_values else 0.0
        negative_edge_pct = pct(float(negative_edge_count), float(fill_count))

        ts_values, mids = collect_mid_series(events)
        adverse_hits = 0
        adverse_total = 0
        for fill in fills:
            fill_ts = to_float(fill.get("ts"))
            fill_price = to_float(fill.get("price"))
            if fill_ts is None or fill_price is None or fill_price <= 0:
                continue
            mid_after = mid_at_or_after(ts_values, mids, fill_ts + 60.0, max_wait_s=180.0)
            if mid_after is None:
                continue
            side = str(fill.get("side") or "").upper()
            adverse_total += 1
            if ("BUY" in side and mid_after < fill_price) or ("SELL" in side and mid_after > fill_price):
                adverse_hits += 1
        adverse_pct = pct(float(adverse_hits), float(adverse_total))

        reason_counts: Dict[str, int] = {}
        for row in reprice:
            reason = str(row.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        uptime_count = sum(reason_counts.get(reason, 0) for reason in ("replace_target_shift", "replace_max_age"))
        uptime_pct = pct(float(uptime_count), float(len(reprice)))

        positions: List[Tuple[float, float]] = []
        for row in events:
            ts = to_float(row.get("ts"))
            position = to_float(row.get("position"))
            if ts is None or position is None:
                continue
            positions.append((ts, position))

        _, env_values = load_market_env(norm_market, env_map)
        max_position_raw = _env_value(env_values, "MM_MAX_POSITION_SIZE", "MAX_POSITION_SIZE", "max_position_size")
        max_position_size = abs(float(max_position_raw)) if max_position_raw is not None else 0.0
        peak_util_pct = 0.0
        time_above_50_s = 0.0
        crossing_count = 0
        if positions and max_position_size > 0:
            peak_util_pct = max(abs(position) / max_position_size * 100.0 for _ts, position in positions)
            time_above_50_s = compute_time_above_util(
                positions,
                max_position=max_position_size,
                threshold=0.5,
                window_end_ts=now_ts,
            )
            crossing_count = zero_crossings([position for _ts, position in positions])

        per_market_payload[norm_market] = {
            "fills": {
                "count": fill_count,
                "fills_per_hour": round_or_none(fills_per_hour, 3),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "buy_qty": round_or_none(buy_qty, 6),
                "sell_qty": round_or_none(sell_qty, 6),
            },
            "edge_quality": {
                "median_edge_bps": round_or_none(median_edge_bps, 4),
                "negative_edge_pct": round_or_none(negative_edge_pct, 4),
            },
            "adverse_selection": {
                "adverse_1m_pct": round_or_none(adverse_pct, 4),
                "samples": adverse_total,
            },
            "quoting_efficiency": {
                "uptime_pct": round_or_none(uptime_pct, 4),
                "reprice_decisions": len(reprice),
            },
            "position_management": {
                "peak_inventory_utilization_pct": round_or_none(peak_util_pct, 4),
                "time_above_50pct_s": round_or_none(time_above_50_s, 4),
                "zero_crossings": crossing_count,
            },
            "edge_pnl_estimate_usd": round_or_none(edge_pnl_estimate_usd, 6),
        }

    try:
        fleet_snapshot = _fetch_total_pnl_snapshot(now_ts, args.lookback_hours)
    except Exception as exc:
        fleet_snapshot = _journal_fallback_snapshot(per_market_payload)
        fleet_snapshot["fetch_error"] = str(exc)

    per_market_pnl = fleet_snapshot.get("per_market_pnl", {})

    for market, payload in sorted(per_market_payload.items()):
        fills_meta = payload["fills"]
        edge_meta = payload["edge_quality"]
        adverse_meta = payload["adverse_selection"]
        quote_meta = payload["quoting_efficiency"]
        pos_meta = payload["position_management"]

        market_pnl = float(per_market_pnl.get(market, payload.get("edge_pnl_estimate_usd", 0.0)))
        adverse_pct = float(adverse_meta.get("adverse_1m_pct") or 0.0)
        uptime_pct = float(quote_meta.get("uptime_pct") or 0.0)
        peak_util_pct = float(pos_meta.get("peak_inventory_utilization_pct") or 0.0)
        negative_edge_pct = float(edge_meta.get("negative_edge_pct") or 0.0)

        verdict, verdict_reasons = _summarize_verdict(
            market_pnl=market_pnl,
            adverse_pct=adverse_pct,
            uptime_pct=uptime_pct,
            peak_util_pct=peak_util_pct,
            negative_edge_pct=negative_edge_pct,
        )

        if verdict == "🟢":
            winners.append(
                f"{market}: ${market_pnl:+.0f}, edge {float(edge_meta.get('median_edge_bps') or 0.0):.2f}bps, "
                f"{int(pos_meta.get('zero_crossings') or 0)} zero-crossings."
            )
        if verdict == "🔴":
            problems.append(
                f"{market}: ${market_pnl:+.0f}, adverse {adverse_pct:.0f}%, uptime {uptime_pct:.0f}% "
                f"({'; '.join(verdict_reasons[:2])})."
            )
            if market_pnl < 0:
                recommendations.append(f"{market}: tighten risk controls or pause until edge quality recovers.")
            if adverse_pct >= 65.0:
                recommendations.append(f"{market}: adverse selection is high; widen quotes or reduce size in toxic windows.")
            if uptime_pct < 25.0:
                recommendations.append(f"{market}: quoting uptime too low; tune blocking filters and stale-book handling.")
            if peak_util_pct >= 70.0:
                recommendations.append(f"{market}: inventory peaks near limits; increase skew response and de-risk faster.")

        rows.append({
            "market": market,
            "pnl": market_pnl,
            "fills": int(fills_meta.get("count") or 0),
            "edge_bps": float(edge_meta.get("median_edge_bps") or 0.0),
            "adverse_pct": adverse_pct,
            "uptime_pct": uptime_pct,
            "peak_util_pct": peak_util_pct,
            "verdict": verdict,
        })
        payload["pnl_usd"] = round_or_none(market_pnl, 6)
        payload["verdict"] = verdict
        payload["verdict_reasons"] = verdict_reasons

    recommendations = list(dict.fromkeys(recommendations))
    rows.sort(key=lambda row: (0 if row["verdict"] == "🔴" else 1 if row["verdict"] == "⚠️" else 2, -abs(row["pnl"])))

    report = _render_report(
        generated_at=iso_utc(now_ts),
        fleet_snapshot=fleet_snapshot,
        rows=rows,
        winners=winners,
        problems=problems,
        recommendations=recommendations,
    )
    report = truncate_markdown(report, args.char_limit)

    payload = {
        "timestamp": iso_utc(now_ts),
        "lookback_hours": args.lookback_hours,
        "fleet": {
            "source": fleet_snapshot.get("source"),
            "fleet_pnl": round_or_none(fleet_snapshot.get("fleet_pnl"), 6),
            "equity": round_or_none(fleet_snapshot.get("equity"), 6),
            "delta_pct": round_or_none(fleet_snapshot.get("delta_pct"), 6),
            "fetch_error": fleet_snapshot.get("fetch_error"),
        },
        "markets": per_market_payload,
        "winners": winners,
        "problems": problems,
        "recommendations": recommendations,
    }
    write_json(output_dir / "daily_scorecard.json", payload)
    append_jsonl(
        output_dir / "scorecard_history.jsonl",
        {
            "timestamp": payload["timestamp"],
            "lookback_hours": args.lookback_hours,
            "fleet_pnl": payload["fleet"]["fleet_pnl"],
            "source": payload["fleet"]["source"],
            "red_markets": sum(1 for row in rows if row["verdict"] == "🔴"),
            "warn_markets": sum(1 for row in rows if row["verdict"] == "⚠️"),
            "green_markets": sum(1 for row in rows if row["verdict"] == "🟢"),
        },
    )

    print(report, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily MM fleet scorecard from journals and account PnL.")
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument("--output-dir", default="data/mm_audit", help="Output directory for JSON artifacts.")
    parser.add_argument("--lookback-hours", type=float, default=24.0, help="Lookback window in hours.")
    parser.add_argument("--do-not-restart-file", default="data/do_not_restart.txt", help="Markets to skip.")
    parser.add_argument(
        "--env-map",
        default=None,
        help=(
            "Optional env mapping. JSON object, file path, or CSV entries "
            "(e.g. 'ETH-USD=.env.eth,MON-USD=.env.mon')."
        ),
    )
    parser.add_argument("--char-limit", type=int, default=3000, help="Maximum markdown output length.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"audit_daily_scorecard error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
