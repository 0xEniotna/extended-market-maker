#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mm_audit_common import (  # noqa: E402
    append_jsonl,
    find_latest_journal,
    iso_utc,
    load_market_jobs,
    load_policy,
    now_ts,
    read_json,
    safe_decimal,
    write_json,
)


def _journal_started_after(path: Path, min_ts: float) -> bool:
    if not path.exists():
        return False
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "run_start":
                continue
            ts = safe_decimal(event.get("ts"))
            if ts is not None and float(ts) >= min_ts:
                return True
    return False


def _evaluate_evidence(
    *,
    action: Dict[str, Any],
    active_markets: set[str],
    journal_dir: Path,
    approved_at_ts: float,
) -> Tuple[bool, List[str], Dict[str, bool]]:
    expected = action.get("expected_evidence") if isinstance(action.get("expected_evidence"), dict) else {}

    checks: Dict[str, bool] = {}
    blockers: List[str] = []

    env_file = expected.get("env_file_exists")
    if isinstance(env_file, str) and env_file:
        ok = Path(env_file).exists()
        checks["env_file_exists"] = ok
        if not ok:
            blockers.append(f"missing_env:{env_file}")

    cron_market = expected.get("cron_market")
    if isinstance(cron_market, str) and cron_market:
        ok = cron_market in active_markets
        checks["cron_market"] = ok
        if not ok:
            blockers.append(f"missing_cron_market:{cron_market}")

    market = str(action.get("market") or cron_market or "")
    if market:
        latest = find_latest_journal(journal_dir, market)
        ok = bool(latest and _journal_started_after(latest, approved_at_ts))
        checks["journal_run_start"] = ok
        if not ok:
            blockers.append(f"missing_recent_run_start:{market}")

    if not checks:
        return False, ["no_expected_evidence_checks"], checks

    return all(checks.values()), blockers, checks


def _render_auditor_message(generated_at: str, resolved: List[Dict[str, Any]], escalated: List[Dict[str, Any]], pending: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Auditor Follow-up")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append(f"- resolved: {len(resolved)}")
    lines.append(f"- escalated: {len(escalated)}")
    lines.append(f"- still_pending: {len(pending)}")
    lines.append("")

    if escalated:
        lines.append("## Escalations")
        lines.append("| action_id | type | market | blockers |")
        lines.append("|---|---|---|---|")
        for row in escalated:
            lines.append(
                f"| {row.get('action_id')} | {row.get('action_type')} | {row.get('market')} | {', '.join(row.get('blockers', []))} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_mm_message(escalated: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# MM Escalation Follow-up")
    if not escalated:
        lines.append("No escalations in this follow-up cycle.")
        lines.append("")
        return "\n".join(lines)

    for row in escalated:
        lines.append(f"## {row.get('action_id')} ({row.get('action_type')} {row.get('market')})")
        lines.append(f"Blockers: {', '.join(row.get('blockers', []))}")
        lines.append("```bash")
        for cmd in row.get("commands", []):
            lines.append(str(cmd))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Follow-up checker for pending auditor actions.")
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument("--policy", default="config/market_scout_policy.yaml", help="Policy path.")
    parser.add_argument("--auditor-dir", default="data/mm_audit/auditor", help="Auditor artifacts directory.")
    parser.add_argument("--journal-dir", default="data/mm_journal", help="Journal directory.")
    parser.add_argument(
        "--jobs-json",
        default="/home/flexouille/.openclaw/cron/jobs.json",
        help="OpenClaw jobs.json path.",
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
    followup_cfg = policy.get("followup", {}) if isinstance(policy, dict) else {}
    escalation_minutes = int(followup_cfg.get("escalation_minutes", 30))

    auditor_dir = (repo_root / args.auditor_dir).resolve()
    journal_dir = (repo_root / args.journal_dir).resolve()
    jobs_path = Path(args.jobs_json).resolve()

    pending_path = auditor_dir / "pending_actions.json"
    followup_log_path = auditor_dir / "auditor_followup_log.jsonl"
    auditor_md_path = auditor_dir / "followup_auditor.md"
    mm_md_path = auditor_dir / "followup_mm.md"
    followup_log_path.parent.mkdir(parents=True, exist_ok=True)
    followup_log_path.touch(exist_ok=True)

    pending_rows = read_json(pending_path, default=[])
    if not isinstance(pending_rows, list):
        pending_rows = []

    active_jobs = load_market_jobs(jobs_path, repo_root)
    active_markets = {row["market"] for row in active_jobs}

    now = now_ts()
    resolved: List[Dict[str, Any]] = []
    escalated: List[Dict[str, Any]] = []
    next_pending: List[Dict[str, Any]] = []

    for row in pending_rows:
        if not isinstance(row, dict):
            continue
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        action_id = str(action.get("action_id", "unknown"))
        action_type = str(action.get("action_type", "unknown"))
        market = str(action.get("market", "unknown"))

        approved_at_ts = float(safe_decimal(row.get("approved_at_ts")) or now)
        deadline_ts = float(safe_decimal(row.get("deadline_ts")) or (approved_at_ts + escalation_minutes * 60))

        ok, blockers, checks = _evaluate_evidence(
            action=action,
            active_markets=active_markets,
            journal_dir=journal_dir,
            approved_at_ts=approved_at_ts,
        )

        status = "pending"
        if ok:
            status = "resolved"
            resolved.append({
                "action_id": action_id,
                "action_type": action_type,
                "market": market,
            })
        else:
            already_escalated = row.get("status") == "escalated"
            if now >= deadline_ts and not already_escalated:
                status = "escalated"
                escalated.append({
                    "action_id": action_id,
                    "action_type": action_type,
                    "market": market,
                    "blockers": blockers,
                    "commands": action.get("commands", []),
                })
            else:
                status = str(row.get("status") or "pending")

            row["status"] = status
            row["last_check_ts"] = now
            row["last_check_at"] = iso_utc(now)
            row["last_checks"] = checks
            row["last_blockers"] = blockers
            next_pending.append(row)

        append_jsonl(
            followup_log_path,
            {
                "ts": now,
                "generated_at": iso_utc(now),
                "action_id": action_id,
                "action_type": action_type,
                "market": market,
                "status": status,
                "checks": checks,
                "blockers": blockers,
            },
        )

    write_json(pending_path, next_pending)

    auditor_message = _render_auditor_message(
        generated_at=iso_utc(now),
        resolved=resolved,
        escalated=escalated,
        pending=next_pending,
    )
    mm_message = _render_mm_message(escalated)

    auditor_md_path.parent.mkdir(parents=True, exist_ok=True)
    auditor_md_path.write_text(auditor_message + "\n")
    mm_md_path.write_text(mm_message + "\n")

    print(f"auditor_followup={auditor_md_path}")
    print(f"mm_followup={mm_md_path}")
    print(f"resolved={len(resolved)}")
    print(f"escalated={len(escalated)}")
    print(f"pending={len(next_pending)}")

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
