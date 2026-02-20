#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mm_audit_common import (  # noqa: E402
    append_jsonl,
    discover_recent_markets_from_journals,
    extract_last_json_object,
    find_latest_journal,
    iso_utc,
    load_market_jobs,
    load_policy,
    now_ts,
    read_json,
    read_jsonl,
    safe_decimal,
    write_json,
)


def _d(value: Any, default: str = "0") -> Decimal:
    parsed = safe_decimal(value)
    if parsed is None:
        return Decimal(default)
    return parsed


def _json_stdout(cmd: List[str], label: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return None, f"{label}_timeout"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or f"{label}_failed"
    try:
        payload = extract_last_json_object(proc.stdout)
    except Exception as exc:
        return None, f"{label}_json_parse_failed: {exc}"
    return payload, None


def _run_inventory_summary(
    *,
    repo_root: Path,
    market: str,
    journal_dir: Path,
    output_dir: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "tools" / "export_inventory_timeseries.py"),
        str(journal_dir),
        "--market",
        market,
        "--output-dir",
        str(output_dir),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return None, "inventory_export_timeout"
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip() or "inventory_export_failed"

    summary_path = output_dir / f"{market.replace('-', '_')}_summary.json"
    if not summary_path.exists():
        # fallback if slug normalization differs
        candidates = sorted(output_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            return None, "inventory_summary_missing"
        summary_path = candidates[-1]

    payload = read_json(summary_path, default={})
    if not isinstance(payload, dict):
        return None, "inventory_summary_invalid"
    return payload, None


def _run_pnl(
    *,
    repo_root: Path,
    market: str,
    env_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "tools" / "fetch_pnl.py"),
        market,
        "--days",
        "1",
        "--max-pages",
        "30",
        "--json-stdout",
    ]
    if env_path:
        cmd.extend(["--env", env_path])
    return _json_stdout(cmd, "fetch_pnl")


def _is_scout_fresh(report_path: Path, action_path: Path, max_age_min: int) -> bool:
    if not report_path.exists() or not action_path.exists():
        return False
    newest_mtime = max(report_path.stat().st_mtime, action_path.stat().st_mtime)
    age_s = now_ts() - newest_mtime
    return age_s <= (max_age_min * 60)


def _validate_gate_flags(action: Dict[str, Any]) -> bool:
    evidence = action.get("evidence")
    if not isinstance(evidence, dict):
        return False
    flags = evidence.get("gate_flags")
    if not isinstance(flags, dict):
        return False
    return all(v is True for v in flags.values())


def _build_decision(
    *,
    action: Dict[str, Any],
    active_markets: set[str],
    stale: bool,
    hard_conflict: bool,
    required_cycles: int,
    min_score_delta: Decimal,
) -> Dict[str, Any]:
    action_id = str(action.get("action_id", "unknown"))
    action_type = str(action.get("action_type", "unknown"))
    market = str(action.get("market", "unknown"))

    decision = "HOLD"
    reason = "missing_data"

    if stale:
        return {
            "action_id": action_id,
            "action_type": action_type,
            "market": market,
            "decision": "HOLD",
            "reason": "scout_artifact_stale",
        }

    if action_type == "launch":
        if market in active_markets:
            decision, reason = "REJECT", "market_already_live"
        elif hard_conflict:
            decision, reason = "HOLD", "hard_inventory_risk_conflict"
        elif not _validate_gate_flags(action):
            decision, reason = "REJECT", "candidate_failed_gates"
        else:
            decision, reason = "APPROVE", "launch_policy_pass"

    elif action_type == "rotate":
        evidence = action.get("evidence") if isinstance(action.get("evidence"), dict) else {}
        streak = int(evidence.get("underperformance_streak") or 0)
        score_delta = _d(evidence.get("score_delta"))
        if market in active_markets:
            decision, reason = "REJECT", "replacement_already_live"
        elif streak < required_cycles:
            decision, reason = "REJECT", "underperformance_not_sustained"
        elif score_delta < min_score_delta:
            decision, reason = "REJECT", "insufficient_score_delta"
        elif hard_conflict:
            decision, reason = "HOLD", "hard_inventory_risk_conflict"
        else:
            decision, reason = "APPROVE", "rotation_policy_pass"

    return {
        "action_id": action_id,
        "action_type": action_type,
        "market": market,
        "decision": decision,
        "reason": reason,
    }


def _render_auditor_message(
    *,
    generated_at: str,
    stale: bool,
    config_change_rows: List[Dict[str, Any]],
    active_context: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append("# Auditor Decision Report")
    lines.append(f"Generated: {generated_at}")
    lines.append(f"Scout freshness_ok: {not stale}")
    lines.append("")

    param_counts = Counter(str(r.get("param")) for r in config_change_rows if r.get("param"))
    lines.append("## Config Changes (last 24h)")
    lines.append(f"- total_rows: {len(config_change_rows)}")
    if param_counts:
        top = ", ".join(f"{k}:{v}" for k, v in param_counts.most_common(8))
        lines.append(f"- top_params: {top}")
    lines.append("")

    lines.append("## Active Market Context")
    if not active_context:
        lines.append("- none")
    else:
        lines.append("| Market | PnL 24h | Max Util % | Time > Hard(s) |")
        lines.append("|---|---:|---:|---:|")
        for row in active_context:
            lines.append(
                "| {market} | {pnl} | {util} | {hard} |".format(
                    market=row.get("market"),
                    pnl=row.get("pnl_24h_usd", "n/a"),
                    util=row.get("utilization_max_pct", "n/a"),
                    hard=row.get("time_above_hard_s", "n/a"),
                )
            )
    lines.append("")

    lines.append("## Decisions")
    if not decisions:
        lines.append("- no actions")
    else:
        lines.append("| action_id | type | market | decision | reason |")
        lines.append("|---|---|---|---|---|")
        for d in decisions:
            lines.append(
                f"| {d.get('action_id')} | {d.get('action_type')} | {d.get('market')} | {d.get('decision')} | {d.get('reason')} |"
            )
    lines.append("")

    return "\n".join(lines)


def _render_mm_message(decisions: List[Dict[str, Any]], actions_by_id: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# MM Playbook")
    approvals = [d for d in decisions if d.get("decision") == "APPROVE"]
    if not approvals:
        lines.append("No approved actions in this cycle.")
        lines.append("")
        return "\n".join(lines)

    for d in approvals:
        action = actions_by_id.get(str(d.get("action_id")), {})
        lines.append(f"## {d.get('action_id')} ({d.get('action_type')} {d.get('market')})")
        lines.append("```bash")
        for cmd in action.get("commands", []):
            lines.append(str(cmd))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply auditor decisioning over scout outputs.")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument("--policy", default="config/market_scout_policy.yaml", help="Policy path.")
    parser.add_argument("--scout-dir", default="data/mm_audit/scout", help="Scout artifact directory.")
    parser.add_argument("--auditor-dir", default="data/mm_audit/auditor", help="Auditor artifact directory.")
    parser.add_argument(
        "--jobs-json",
        default="/home/flexouille/.openclaw/cron/jobs.json",
        help="OpenClaw jobs.json path.",
    )
    parser.add_argument(
        "--config-changelog",
        default="data/mm_audit/config_changelog.jsonl",
        help="Config changelog JSONL path.",
    )
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument("--inventory-dir", default="data/mm_audit/inventory", help="Inventory output dir.")
    parser.add_argument(
        "--max-scout-age-min",
        type=int,
        default=None,
        help="Override max allowed scout artifact age in minutes.",
    )
    parser.add_argument(
        "--print-target",
        choices=["auditor", "mm", "both"],
        default="both",
        help="Which message to print to stdout.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    policy = load_policy((repo_root / args.policy).resolve())
    rotation_cfg = policy.get("rotation", {}) if isinstance(policy, dict) else {}
    freshness_cfg = policy.get("freshness", {}) if isinstance(policy, dict) else {}
    followup_cfg = policy.get("followup", {}) if isinstance(policy, dict) else {}

    required_cycles = int(rotation_cfg.get("underperformance_cycles", 2))
    min_score_delta = _d(rotation_cfg.get("min_score_delta", 1.5))
    max_age_min = int(
        args.max_scout_age_min
        if args.max_scout_age_min is not None
        else freshness_cfg.get("scout_max_age_minutes", 15)
    )
    escalation_minutes = int(followup_cfg.get("escalation_minutes", 30))

    scout_dir = (repo_root / args.scout_dir).resolve()
    auditor_dir = (repo_root / args.auditor_dir).resolve()
    journal_dir = (repo_root / args.journal_dir).resolve()
    inventory_dir = (repo_root / args.inventory_dir).resolve()
    config_changelog_path = (repo_root / args.config_changelog).resolve()
    jobs_path = Path(args.jobs_json).resolve()

    report_path = scout_dir / "market_scout_report.json"
    action_pack_path = scout_dir / "action_pack.json"

    report = read_json(report_path, default={})
    actions = read_json(action_pack_path, default=[])
    if not isinstance(report, dict):
        report = {}
    if not isinstance(actions, list):
        actions = []

    stale = not _is_scout_fresh(report_path, action_pack_path, max_age_min)

    active_jobs = load_market_jobs(jobs_path, repo_root)
    if not active_jobs:
        fallback_markets = discover_recent_markets_from_journals(journal_dir, lookback_s=86400.0)
        active_jobs = [
            {
                "job_id": f"journal-fallback-{idx}",
                "job_name": "journal_fallback",
                "market": market,
                "env_path": None,
            }
            for idx, market in enumerate(fallback_markets)
        ]
    active_markets = {row["market"] for row in active_jobs}

    now = now_ts()
    cutoff = now - 86400
    config_changes = [row for row in read_jsonl(config_changelog_path) if safe_decimal(row.get("ts")) and float(row.get("ts")) >= cutoff]

    active_context: List[Dict[str, Any]] = []
    hard_conflict = False
    for job in active_jobs:
        market = str(job["market"])

        inventory_summary, inv_err = _run_inventory_summary(
            repo_root=repo_root,
            market=market,
            journal_dir=journal_dir,
            output_dir=inventory_dir,
        )
        pnl_summary, pnl_err = _run_pnl(
            repo_root=repo_root,
            market=market,
            env_path=job.get("env_path"),
        )

        pnl_24h = None
        if isinstance(pnl_summary, dict):
            totals = pnl_summary.get("totals")
            if isinstance(totals, dict):
                pnl_24h = safe_decimal(totals.get("total_pnl_including_open_usd"))

        util_max = None
        time_above_hard_s = None
        if isinstance(inventory_summary, dict):
            util_max = safe_decimal(inventory_summary.get("p95_utilization_pct"))
            hard = safe_decimal(inventory_summary.get("time_above_hard_s"))
            if hard is not None:
                time_above_hard_s = hard
                if hard > 0:
                    hard_conflict = True

        active_context.append({
            "market": market,
            "pnl_24h_usd": pnl_24h,
            "utilization_max_pct": util_max,
            "time_above_hard_s": time_above_hard_s,
            "inventory_error": inv_err,
            "pnl_error": pnl_err,
            "latest_journal": str(find_latest_journal(journal_dir, market)) if find_latest_journal(journal_dir, market) else None,
        })

    decisions: List[Dict[str, Any]] = []
    actions_by_id: Dict[str, Dict[str, Any]] = {}
    for action in actions:
        if not isinstance(action, dict):
            continue
        action_id = str(action.get("action_id", ""))
        if action_id:
            actions_by_id[action_id] = action
        decision = _build_decision(
            action=action,
            active_markets=active_markets,
            stale=stale,
            hard_conflict=hard_conflict,
            required_cycles=required_cycles,
            min_score_delta=min_score_delta,
        )
        decisions.append(decision)

    decisions_path = auditor_dir / "auditor_decisions.jsonl"
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.touch(exist_ok=True)
    for decision in decisions:
        append_jsonl(
            decisions_path,
            {
                "ts": now,
                "generated_at": iso_utc(now),
                "decision": decision,
                "stale": stale,
                "hard_conflict": hard_conflict,
                "active_markets": sorted(active_markets),
            },
        )

    pending_path = auditor_dir / "pending_actions.json"
    existing_pending = read_json(pending_path, default=[])
    if not isinstance(existing_pending, list):
        existing_pending = []

    existing_map = {
        str(row.get("action", {}).get("action_id")): row
        for row in existing_pending
        if isinstance(row, dict) and isinstance(row.get("action"), dict)
    }

    for decision in decisions:
        if decision.get("decision") != "APPROVE":
            continue
        action = actions_by_id.get(str(decision.get("action_id")))
        if not action:
            continue
        action_id = str(action.get("action_id"))
        if action_id in existing_map:
            continue
        existing_map[action_id] = {
            "action": action,
            "approved_at_ts": now,
            "approved_at": iso_utc(now),
            "deadline_ts": now + escalation_minutes * 60,
            "deadline_at": iso_utc(now + escalation_minutes * 60),
            "status": "pending",
            "last_check_ts": None,
        }

    pending_rows = sorted(existing_map.values(), key=lambda row: row.get("approved_at_ts", now))
    write_json(pending_path, pending_rows)

    auditor_message = _render_auditor_message(
        generated_at=iso_utc(now),
        stale=stale,
        config_change_rows=config_changes,
        active_context=active_context,
        decisions=decisions,
    )
    mm_message = _render_mm_message(decisions, actions_by_id)

    auditor_md_path = auditor_dir / "auditor_message.md"
    mm_md_path = auditor_dir / "mm_playbook.md"
    auditor_md_path.parent.mkdir(parents=True, exist_ok=True)
    auditor_md_path.write_text(auditor_message + "\n")
    mm_md_path.write_text(mm_message + "\n")

    print(f"auditor_message={auditor_md_path}")
    print(f"mm_playbook={mm_md_path}")
    print(f"decisions={len(decisions)}")
    print(f"pending_actions={len(pending_rows)}")

    if args.print_target == "auditor":
        print("\n" + auditor_message)
    elif args.print_target == "mm":
        print("\n" + mm_message)
    else:
        print("\n" + auditor_message)
        print("\n---\n")
        print(mm_message)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
