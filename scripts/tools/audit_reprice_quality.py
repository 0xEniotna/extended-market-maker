#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_report_common import (  # noqa: E402
    discover_market_journals,
    is_block_reason,
    iso_utc,
    load_do_not_restart,
    load_market_env,
    load_recent_entries,
    normalize_market_name,
    parse_env_map,
    pct,
    round_or_none,
    to_float,
    truncate_markdown,
)
from mm_audit_common import write_json  # noqa: E402

UPTIME_REASONS = {"replace_target_shift", "replace_max_age"}
RECOMMENDATION_HINTS = {
    "skip_imbalance_pause": "Raise MM_IMBALANCE_PAUSE_THRESHOLD",
    "hold_within_tolerance": "Lower MM_REPRICE_TOLERANCE_PERCENT",
    "skip_toxicity": "Review toxicity thresholds",
    "skip_stale": "Check websocket connectivity — orderbook data going stale",
    "skip_trend_counter_strong": "Lower MM_TREND_STRONG_THRESHOLD",
}


def _env_value(env_map: Dict[str, str], *keys: str) -> str | None:
    for key in keys:
        if key in env_map and env_map[key]:
            return env_map[key]
    return None


def _recommendation_for_reason(reason: str, env_map: Dict[str, str]) -> str:
    if reason == "skip_imbalance_pause":
        current = _env_value(env_map, "MM_IMBALANCE_PAUSE_THRESHOLD", "IMBALANCE_PAUSE_THRESHOLD")
        if current is None:
            return "Raise MM_IMBALANCE_PAUSE_THRESHOLD"
        return f"Raise MM_IMBALANCE_PAUSE_THRESHOLD (current: {current})"
    if reason == "hold_within_tolerance":
        current = _env_value(env_map, "MM_REPRICE_TOLERANCE_PERCENT", "REPRICE_TOLERANCE_PERCENT")
        if current is None:
            return "Lower MM_REPRICE_TOLERANCE_PERCENT"
        return f"Lower MM_REPRICE_TOLERANCE_PERCENT (current: {current})"
    if reason == "skip_trend_counter_strong":
        current = _env_value(env_map, "MM_TREND_STRONG_THRESHOLD", "TREND_STRONG_THRESHOLD")
        if current is None:
            return "Lower MM_TREND_STRONG_THRESHOLD"
        return f"Lower MM_TREND_STRONG_THRESHOLD (current: {current})"
    return RECOMMENDATION_HINTS.get(reason, f"Review {reason} behavior")


def _compute_filter_stacking(decisions: List[Dict[str, Any]]) -> int:
    if len(decisions) < 2:
        return 0
    stacked = 0
    prev = decisions[0]
    prev_ts = to_float(prev.get("ts"))
    prev_reason = str(prev.get("reason") or "")
    for cur in decisions[1:]:
        cur_ts = to_float(cur.get("ts"))
        cur_reason = str(cur.get("reason") or "")
        if (
            prev_ts is not None
            and cur_ts is not None
            and is_block_reason(prev_reason)
            and is_block_reason(cur_reason)
            and (cur_ts - prev_ts) <= 0.1
        ):
            stacked += 1
        prev = cur
        prev_ts = cur_ts
        prev_reason = cur_reason
    return stacked


def _main_flag(flags: List[str], top_blocker: str) -> str:
    if "side_bias" in flags:
        return "🔴 Side bias"
    if "low_uptime" in flags:
        return "🔴 Low uptime"
    if top_blocker == "skip_stale":
        return "⚠️ Stale OB"
    if "warn_low_uptime" in flags:
        return "⚠️ Low uptime"
    if "single_filter_dominance" in flags:
        return "⚠️ Dominance"
    return "✅"


def _render_report(rows: List[Dict[str, Any]], generated_at: str) -> str:
    lines: List[str] = []
    lines.append(f"## Reprice Quality — {generated_at}")
    lines.append("| Market | Uptime% | Bias | Top Blocker | Flag |")
    lines.append("|--------|---------|------|-------------|------|")
    for row in rows:
        top = row["top_blockers"][0] if row["top_blockers"] else {"reason": "-", "pct_blocks": 0.0}
        lines.append(
            f"| {row['market']} | {row['quoting_uptime_pct']:.0f}% | {row['bias_text']} | "
            f"{top['reason']} ({top['pct_blocks']:.0f}%) | {row['main_flag']} |"
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
        decisions = [row for row in events if str(row.get("type")) == "reprice_decision"]
        if not decisions:
            continue

        side_reason_counts: Dict[str, Counter[str]] = {
            "BUY": Counter(),
            "SELL": Counter(),
        }
        reason_counts: Counter[str] = Counter()
        buy_blocks = 0
        sell_blocks = 0

        for decision in decisions:
            side = str(decision.get("side") or "").upper()
            if side not in ("BUY", "SELL"):
                side = "BUY" if "BUY" in side else "SELL" if "SELL" in side else "BUY"
            reason = str(decision.get("reason") or "unknown")
            reason_counts[reason] += 1
            side_reason_counts[side][reason] += 1
            if is_block_reason(reason):
                if side == "BUY":
                    buy_blocks += 1
                else:
                    sell_blocks += 1

        total_decisions = len(decisions)
        uptime_count = sum(reason_counts.get(reason, 0) for reason in UPTIME_REASONS)
        uptime_pct = pct(float(uptime_count), float(total_decisions))

        max_blocks = max(buy_blocks, sell_blocks)
        min_blocks = min(buy_blocks, sell_blocks)
        side_bias_ratio = float(max_blocks) / float(min_blocks if min_blocks > 0 else 1)
        if buy_blocks > sell_blocks:
            bias_direction = "BUY_blocked_more"
            bias_text = f"BUY {side_bias_ratio:.1f}x"
        elif sell_blocks > buy_blocks:
            bias_direction = "SELL_blocked_more"
            bias_text = f"SELL {side_bias_ratio:.1f}x"
        else:
            bias_direction = "balanced"
            bias_text = "1.0x"

        block_reasons = [(reason, count) for reason, count in reason_counts.items() if is_block_reason(reason)]
        block_reasons.sort(key=lambda item: (-item[1], item[0]))
        total_blocks = sum(count for _, count in block_reasons)

        top_blockers: List[Dict[str, Any]] = []
        for reason, count in block_reasons[:3]:
            top_blockers.append({
                "reason": reason,
                "count": count,
                "pct_total": round(pct(float(count), float(total_decisions)), 2),
                "pct_blocks": round(pct(float(count), float(total_blocks)), 2) if total_blocks > 0 else 0.0,
            })

        dominant_reason = top_blockers[0]["reason"] if top_blockers else "none"
        dominant_share = (top_blockers[0]["count"] / total_blocks) if total_blocks > 0 and top_blockers else 0.0

        flags: List[str] = []
        if max_blocks > 0 and side_bias_ratio > 3.0:
            flags.append("side_bias")
        if uptime_pct < 20.0:
            flags.append("low_uptime")
        elif uptime_pct < 40.0:
            flags.append("warn_low_uptime")
        if dominant_share > 0.50 and dominant_reason != "none":
            flags.append("single_filter_dominance")

        env_path, env_values = load_market_env(norm_market, env_map)
        recommendations: List[str] = []
        if dominant_reason != "none":
            if dominant_reason in RECOMMENDATION_HINTS:
                recommendations.append(f"{norm_market}: {_recommendation_for_reason(dominant_reason, env_values)}")
        if "side_bias" in flags:
            recommendations.append(
                f"{norm_market}: BUY/SELL side blocked {side_bias_ratio:.1f}x more — inventory bias risk."
            )
        if "low_uptime" in flags:
            recommendations.append(
                f"{norm_market}: Bot mostly not quoting (uptime {uptime_pct:.1f}%). "
                f"Dominant blocker: {dominant_reason}."
            )
        if "single_filter_dominance" in flags and dominant_reason not in RECOMMENDATION_HINTS:
            recommendations.append(
                f"{norm_market}: Single filter dominance detected ({dominant_reason}). Review this guardrail."
            )
        if not recommendations:
            recommendations.append(f"{norm_market}: No urgent tuning change required.")

        deduped_recs = list(dict.fromkeys(recommendations))
        main_flag = _main_flag(flags, dominant_reason)
        filter_stacking_hits = _compute_filter_stacking(decisions)

        results[norm_market] = {
            "total_decisions": total_decisions,
            "quoting_uptime_pct": round_or_none(uptime_pct, 3),
            "side_bias_ratio": round_or_none(side_bias_ratio, 3),
            "bias_direction": bias_direction,
            "buy_blocks": buy_blocks,
            "sell_blocks": sell_blocks,
            "buy_decisions": dict(sorted(side_reason_counts["BUY"].items(), key=lambda item: (-item[1], item[0]))),
            "sell_decisions": dict(sorted(side_reason_counts["SELL"].items(), key=lambda item: (-item[1], item[0]))),
            "total_block_decisions": total_blocks,
            "top_blockers": top_blockers,
            "filter_stacking_hits": filter_stacking_hits,
            "flags": flags,
            "recommendations": deduped_recs,
            "env_path": str(env_path) if env_path is not None else None,
        }

        rows.append({
            "market": norm_market,
            "quoting_uptime_pct": uptime_pct,
            "side_bias_ratio": side_bias_ratio,
            "bias_text": bias_text,
            "top_blockers": top_blockers,
            "flags": flags,
            "main_flag": main_flag,
            "recommendations": deduped_recs,
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
    write_json(output_dir / "reprice_quality.json", payload)

    report = _render_report(rows, iso_utc(now_ts))
    report = truncate_markdown(report, args.char_limit)
    print(report, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit reprice quality and side bias per market.")
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument("--output-dir", default="data/mm_audit", help="Output directory for JSON artifacts.")
    parser.add_argument("--lookback-hours", type=float, default=1.0, help="Lookback window in hours.")
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
        print(f"audit_reprice_quality error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
