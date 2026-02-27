#!/usr/bin/env python3
"""Apply advisor proposals with an approval-gated contract.

This script is the only component allowed to mutate `.env*` files from advisor
proposal rows. It appends apply receipts and changelog apply rows for every
attempt (applied/failed/skipped).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "scripts" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from mm_audit_common import (  # noqa: E402
    append_jsonl,
    iso_utc,
    now_ts,
    parse_env,
    read_env_lines,
    read_jsonl,
    safe_decimal,
    update_env_lines,
)

ALLOWED_APPLIED_BY = {"human", "warren_auto"}
ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SENSITIVE_KEY_MARKERS = (
    "PRIVATE_KEY",
    "_SECRET",
    "PASSPHRASE",
    "MNEMONIC",
    "_API_KEY",
)
DEFAULT_PROTECTED_KEYS = {
    "MM_API_KEY",
    "MM_STARK_PRIVATE_KEY",
    "MM_STARK_PUBLIC_KEY",
    "MM_VAULT_ID",
    "MM_BUILDER_ID",
}


def _load_protected_keys(repo_root: Path) -> set[str]:
    path = (repo_root / "mm_config" / "policy" / "protected_keys.json").resolve()
    keys: set[str] = set(DEFAULT_PROTECTED_KEYS)
    if not path.exists():
        return keys
    try:
        raw = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"invalid protected_keys.json at {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise RuntimeError(f"invalid protected_keys.json at {path}: expected array")
    for item in raw:
        key = str(item).strip()
        if not ENV_KEY_RE.match(key):
            raise RuntimeError(f"invalid protected key in {path}: {item!r}")
        keys.add(key)
    return keys


def _looks_sensitive_key(param: str) -> bool:
    upper = param.upper()
    return any(marker in upper for marker in SENSITIVE_KEY_MARKERS)


def _env_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _ts(row: Dict[str, Any]) -> float:
    parsed = safe_decimal(row.get("ts"))
    if parsed is None:
        return 0.0
    return float(parsed)


def _is_pending(proposal_id: str, receipts: List[Dict[str, Any]]) -> bool:
    for row in receipts:
        if not isinstance(row, dict):
            continue
        if str(row.get("proposal_id") or "") != proposal_id:
            continue
        result = str(row.get("result") or "")
        if result in {"applied", "failed", "skipped"}:
            return False
    return True


def _is_applyable_row(row: Dict[str, Any], protected_keys: set[str]) -> Tuple[bool, str]:
    if row.get("guardrail_status") != "passed":
        return False, "guardrail_not_passed"
    if row.get("rejected") is True:
        return False, "proposal_rejected"
    param = str(row.get("param") or "")
    if not param or param == "*":
        return False, "proposal_not_param_level"
    if param in protected_keys or _looks_sensitive_key(param):
        return False, "protected_key"
    proposed = row.get("proposed")
    if proposed is None:
        return False, "missing_proposed_value"
    env_path = str(row.get("env_path") or "")
    if not env_path:
        return False, "missing_env_path"
    return True, "ok"


def _norm_bool_text(value: Any) -> Optional[bool]:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _values_match(current_value: Any, old_value: Any) -> bool:
    if current_value is None and old_value is None:
        return True
    if current_value is None or old_value is None:
        return False
    current_text = str(current_value).strip()
    old_text = str(old_value).strip()
    if current_text == old_text:
        return True

    current_num = safe_decimal(current_text)
    old_num = safe_decimal(old_text)
    if current_num is not None and old_num is not None:
        return current_num == old_num

    current_bool = _norm_bool_text(current_text)
    old_bool = _norm_bool_text(old_text)
    if current_bool is not None and old_bool is not None:
        return current_bool == old_bool

    return False


def _is_warren_auto_eligible(row: Dict[str, Any]) -> bool:
    return (
        row.get("deadman") is True
        and row.get("guardrail_status") == "passed"
        and row.get("confidence") == "high"
        and row.get("escalation_target") == "warren"
        and row.get("rejected") is not True
    )


def _proposal_apply_row(
    proposal: Dict[str, Any],
    *,
    apply_ts: float,
    applied_by: str,
    result: str,
    env_before_hash: Optional[str],
    env_after_hash: Optional[str],
    failure_reason: Optional[str],
) -> Dict[str, Any]:
    out = dict(proposal)
    out.update(
        {
            "ts": apply_ts,
            "created_at": iso_utc(apply_ts),
            "proposal_only": False,
            "applied": result == "applied",
            "rejected": result != "applied",
            "applied_by": applied_by,
            "applied_ts": apply_ts,
            "apply_result": result,
            "failure_reason": failure_reason,
            "env_before_hash": env_before_hash,
            "env_after_hash": env_after_hash,
            "source": "mm_advisor_apply",
        }
    )
    return out


def _receipt_row(
    *,
    proposal_id: str,
    applied_by: str,
    apply_ts: float,
    result: str,
    env_before_hash: Optional[str],
    env_after_hash: Optional[str],
    failure_reason: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "proposal_id": proposal_id,
        "applied_by": applied_by,
        "applied_ts": apply_ts,
        "result": result,
        "env_before_hash": env_before_hash,
        "env_after_hash": env_after_hash,
    }
    if failure_reason:
        row["failure_reason"] = failure_reason
    return row


def _select_proposals(
    *,
    proposals: List[Dict[str, Any]],
    receipts: List[Dict[str, Any]],
    mode: str,
    proposal_id: Optional[str],
    market_filter: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    filtered = [row for row in proposals if isinstance(row, dict)]
    if market_filter:
        filtered = [row for row in filtered if str(row.get("market") or "") == market_filter]
    filtered.sort(key=_ts)

    if proposal_id:
        matches = [row for row in filtered if str(row.get("proposal_id") or "") == proposal_id]
        if not matches:
            return []
        latest = max(matches, key=_ts)
        if not _is_pending(str(latest.get("proposal_id")), receipts):
            return []
        return [latest]

    if mode != "warren-auto":
        return []

    latest_by_param: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in filtered:
        pid = str(row.get("proposal_id") or "")
        if not pid or not _is_pending(pid, receipts):
            continue
        if not _is_warren_auto_eligible(row):
            continue
        key = (str(row.get("market") or ""), str(row.get("param") or ""))
        previous = latest_by_param.get(key)
        if previous is None or _ts(row) > _ts(previous):
            latest_by_param[key] = row

    selected = sorted(latest_by_param.values(), key=_ts)
    if limit > 0:
        return selected[:limit]
    return selected


def _proposal_id_status(
    proposals: List[Dict[str, Any]],
    receipts: List[Dict[str, Any]],
    proposal_id: str,
) -> str:
    matches = [
        row
        for row in proposals
        if isinstance(row, dict) and str(row.get("proposal_id") or "") == proposal_id
    ]
    if not matches:
        return "proposal_id_not_found"
    latest = max(matches, key=_ts)
    if not _is_pending(proposal_id, receipts):
        return "proposal_not_pending"
    if latest.get("guardrail_status") != "passed":
        return "guardrail_not_passed"
    if latest.get("rejected") is True:
        return "proposal_rejected"
    if not str(latest.get("param") or "") or str(latest.get("param") or "") == "*":
        return "proposal_not_param_level"
    if latest.get("proposed") is None:
        return "missing_proposed_value"
    if not str(latest.get("env_path") or ""):
        return "missing_env_path"
    return "pending"


def _list_pending(
    *,
    proposals: List[Dict[str, Any]],
    receipts: List[Dict[str, Any]],
    market_filter: Optional[str],
    limit: int,
) -> List[Dict[str, Any]]:
    filtered = [row for row in proposals if isinstance(row, dict)]
    if market_filter:
        filtered = [row for row in filtered if str(row.get("market") or "") == market_filter]
    filtered.sort(key=_ts, reverse=True)

    pending: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in filtered:
        pid = str(row.get("proposal_id") or "")
        if not pid or pid in seen_ids:
            continue
        seen_ids.add(pid)
        if not _is_pending(pid, receipts):
            continue
        pending.append(
            {
                "proposal_id": pid,
                "market": row.get("market"),
                "param": row.get("param"),
                "old": row.get("old"),
                "proposed": row.get("proposed"),
                "deadman": row.get("deadman"),
                "confidence": row.get("confidence"),
                "escalation_target": row.get("escalation_target"),
                "guardrail_status": row.get("guardrail_status"),
                "ts": row.get("ts"),
            }
        )
        if limit > 0 and len(pending) >= limit:
            break
    return pending


def _apply_one(
    proposal: Dict[str, Any],
    *,
    approve: bool,
    dry_run: bool,
    allow_old_mismatch: bool,
    applied_by: str,
    protected_keys: set[str],
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    apply_ts = now_ts()
    proposal_id = str(proposal.get("proposal_id") or "")
    env_before_hash: Optional[str] = None
    env_after_hash: Optional[str] = None

    def finish(result: str, reason: Optional[str]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        receipt = _receipt_row(
            proposal_id=proposal_id,
            applied_by=applied_by,
            apply_ts=apply_ts,
            result=result,
            env_before_hash=env_before_hash,
            env_after_hash=env_after_hash,
            failure_reason=reason,
        )
        apply_row = _proposal_apply_row(
            proposal,
            apply_ts=apply_ts,
            applied_by=applied_by,
            result=result,
            env_before_hash=env_before_hash,
            env_after_hash=env_after_hash,
            failure_reason=reason,
        )
        return receipt, apply_row

    if not approve:
        return finish("skipped", "approval_required")

    ok, reason = _is_applyable_row(proposal, protected_keys)
    if not ok:
        return finish("failed", reason)

    env_path = Path(str(proposal.get("env_path"))).resolve()
    if not env_path.exists():
        return finish("failed", "env_file_not_found")

    try:
        original_lines = read_env_lines(env_path)
        original_map = parse_env(original_lines)
    except Exception:
        return finish("failed", "env_read_failed")

    env_before_hash = _env_hash(env_path)
    param = str(proposal.get("param") or "")
    proposed_value = str(proposal.get("proposed"))
    old_value = proposal.get("old")
    current_value = original_map.get(param)

    # Idempotency first: if target is already live, do not fail on stale `old`.
    if _values_match(current_value, proposed_value):
        env_after_hash = env_before_hash
        return finish("skipped", "already_at_proposed")

    if old_value is not None and not allow_old_mismatch:
        if not _values_match(current_value, old_value):
            env_after_hash = env_before_hash
            return finish("failed", "current_value_mismatch")

    updated_lines = update_env_lines(original_lines, {param: proposed_value})
    new_text = "\n".join(updated_lines).rstrip("\n") + "\n"
    old_text = "\n".join(original_lines).rstrip("\n") + "\n"

    if new_text == old_text:
        env_after_hash = env_before_hash
        return finish("skipped", "already_at_proposed")

    if dry_run:
        env_after_hash = env_before_hash
        return finish("skipped", "dry_run")

    try:
        env_path.write_text(new_text)
    except Exception:
        env_after_hash = env_before_hash
        return finish("failed", "env_write_failed")

    env_after_hash = _env_hash(env_path)
    return finish("applied", None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply advisor proposals with explicit approval.")
    parser.add_argument("--repo", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument(
        "--proposals",
        default="data/mm_audit/advisor/proposals.jsonl",
        help="Proposal stream path.",
    )
    parser.add_argument(
        "--apply-receipts",
        default="data/mm_audit/advisor/apply_receipts.jsonl",
        help="Apply receipts stream path.",
    )
    parser.add_argument(
        "--config-changelog",
        default="data/mm_audit/config_changelog.jsonl",
        help="Config changelog JSONL path.",
    )
    parser.add_argument(
        "--mode",
        choices=["human", "warren-auto"],
        default="human",
        help="Apply mode: human (single proposal-id) or warren-auto (deadman-only).",
    )
    parser.add_argument("--proposal-id", default=None, help="Explicit proposal id to apply.")
    parser.add_argument(
        "--list-pending",
        action="store_true",
        help="List pending proposal IDs and exit.",
    )
    parser.add_argument(
        "--list-limit",
        type=int,
        default=50,
        help="Max number of pending proposals to list.",
    )
    parser.add_argument("--market", default=None, help="Optional market filter.")
    parser.add_argument("--approve", action="store_true", help="Required to mutate env files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate + log without writing env.")
    parser.add_argument(
        "--allow-old-mismatch",
        action="store_true",
        help="Allow apply even when current env value differs from proposal.old.",
    )
    parser.add_argument(
        "--applied-by",
        choices=sorted(ALLOWED_APPLIED_BY),
        default=None,
        help="Apply actor label. Defaults by mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max proposals to apply in warren-auto mode.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    args = parser.parse_args(argv)

    if args.mode == "human" and not args.proposal_id and not args.list_pending:
        raise SystemExit("--proposal-id is required in human mode.")

    applied_by = args.applied_by or ("warren_auto" if args.mode == "warren-auto" else "human")
    if args.mode == "warren-auto" and applied_by != "warren_auto":
        raise SystemExit("warren-auto mode requires --applied-by warren_auto.")
    if args.mode == "human" and applied_by != "human":
        raise SystemExit("human mode requires --applied-by human.")

    repo_root = Path(args.repo).resolve()
    proposals_path = (repo_root / args.proposals).resolve()
    receipts_path = (repo_root / args.apply_receipts).resolve()
    changelog_path = (repo_root / args.config_changelog).resolve()

    receipts_path.parent.mkdir(parents=True, exist_ok=True)
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    if not receipts_path.exists():
        receipts_path.touch()
    if not changelog_path.exists():
        changelog_path.touch()

    proposals = read_jsonl(proposals_path)
    receipts = read_jsonl(receipts_path)
    if args.list_pending:
        pending_rows = _list_pending(
            proposals=proposals,
            receipts=receipts,
            market_filter=args.market,
            limit=args.list_limit,
        )
        summary = {
            "pending": len(pending_rows),
            "mode": args.mode,
            "proposals_path": str(proposals_path),
            "results": pending_rows,
        }
        if args.json:
            print(json.dumps(summary))
        else:
            if not pending_rows:
                print(f"pending=0 proposals_path={proposals_path}")
            else:
                print(f"pending={len(pending_rows)} proposals_path={proposals_path}")
                for row in pending_rows:
                    print(row["proposal_id"])
        return 0

    selected = _select_proposals(
        proposals=proposals,
        receipts=receipts,
        mode=args.mode,
        proposal_id=args.proposal_id,
        market_filter=args.market,
        limit=args.limit,
    )
    protected_keys = _load_protected_keys(repo_root)

    summary = {
        "selected": len(selected),
        "applied": 0,
        "failed": 0,
        "skipped": 0,
        "mode": args.mode,
        "applied_by": applied_by,
        "dry_run": bool(args.dry_run),
        "results": [],
    }

    if args.proposal_id and summary["selected"] == 0:
        summary["proposal_id"] = args.proposal_id
        summary["proposals_path"] = str(proposals_path)
        summary["reason"] = _proposal_id_status(proposals, receipts, args.proposal_id)
        if args.json:
            print(json.dumps(summary))
        else:
            print(
                "selected=0 applied=0 failed=0 skipped=0 "
                f"reason={summary['reason']} proposals_path={proposals_path}"
            )
        return 1

    for proposal in selected:
        receipt, apply_row = _apply_one(
            proposal,
            approve=bool(args.approve),
            dry_run=bool(args.dry_run),
            allow_old_mismatch=bool(args.allow_old_mismatch),
            applied_by=applied_by,
            protected_keys=protected_keys,
        )
        append_jsonl(receipts_path, receipt)
        if apply_row is not None:
            append_jsonl(changelog_path, apply_row)

        result = str(receipt.get("result") or "failed")
        if result == "applied":
            summary["applied"] += 1
        elif result == "skipped":
            summary["skipped"] += 1
        else:
            summary["failed"] += 1
        summary["results"].append(receipt)

    if args.json:
        print(json.dumps(summary))
    else:
        print(
            "selected={selected} applied={applied} failed={failed} skipped={skipped} "
            "mode={mode} applied_by={applied_by}".format(**summary)
        )

    if summary["failed"] > 0:
        return 1
    if summary["applied"] == 0 and summary["skipped"] > 0 and args.proposal_id:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
