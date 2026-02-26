#!/usr/bin/env python3
"""Deterministic submission of manual advisor proposals.

This script converts manual/analyst proposal intents into canonical
`proposals.jsonl` rows that `mm_advisor_apply.py` can consume safely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    slugify,
)

ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
MARKET_RE = re.compile(r"^[A-Z0-9_.-]+$")
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


class SubmissionError(RuntimeError):
    pass


def _load_policy(repo_root: Path) -> Tuple[set[str], Dict[str, Dict[str, Any]], set[str]]:
    policy_dir = repo_root / "mm_config" / "policy"
    whitelist_path = policy_dir / "whitelist.json"
    bounds_path = policy_dir / "bounds.json"
    protected_path = policy_dir / "protected_keys.json"

    whitelist_raw = json.loads(whitelist_path.read_text())
    bounds_raw = json.loads(bounds_path.read_text())
    protected_raw: Any = []
    if protected_path.exists():
        protected_raw = json.loads(protected_path.read_text())

    if not isinstance(whitelist_raw, list):
        raise SubmissionError(f"invalid whitelist file: {whitelist_path}")
    if not isinstance(bounds_raw, dict):
        raise SubmissionError(f"invalid bounds file: {bounds_path}")
    if not isinstance(protected_raw, list):
        raise SubmissionError(f"invalid protected file: {protected_path}")

    whitelist = {str(x).strip() for x in whitelist_raw}
    bounds = {str(k).strip(): dict(v) for k, v in bounds_raw.items() if isinstance(v, dict)}
    protected = set(DEFAULT_PROTECTED_KEYS)
    for x in protected_raw:
        protected.add(str(x).strip())

    return whitelist, bounds, protected


def _normalize_bool(value: Any, *, key: str) -> str:
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return "true"
    if text in {"false", "0"}:
        return "false"
    raise SubmissionError(f"invalid bool for {key}: {value!r}")


def _normalize_value(param: str, raw: Any, rule: Dict[str, Any]) -> str:
    value_type = str(rule.get("type") or "").strip().lower()
    text = str(raw).strip()
    if value_type == "int":
        if not re.fullmatch(r"[+-]?\d+", text):
            raise SubmissionError(f"invalid int for {param}: {raw!r}")
        parsed = int(text)
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and parsed < int(min_v):
            raise SubmissionError(f"{param} below min bound ({min_v})")
        if max_v is not None and parsed > int(max_v):
            raise SubmissionError(f"{param} above max bound ({max_v})")
        return str(parsed)
    if value_type == "float":
        try:
            parsed = Decimal(text)
        except InvalidOperation as exc:
            raise SubmissionError(f"invalid float for {param}: {raw!r}") from exc
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and parsed < Decimal(str(min_v)):
            raise SubmissionError(f"{param} below min bound ({min_v})")
        if max_v is not None and parsed > Decimal(str(max_v)):
            raise SubmissionError(f"{param} above max bound ({max_v})")
        normalized = format(parsed.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized
    if value_type == "bool":
        return _normalize_bool(text, key=param)
    if value_type == "string":
        allowed = rule.get("allowed")
        if isinstance(allowed, list):
            allowed_set = {str(x) for x in allowed}
            if text not in allowed_set:
                raise SubmissionError(f"{param} not in allowed set")
        return text
    raise SubmissionError(f"unsupported bounds type for {param}: {value_type!r}")


def _looks_sensitive_key(param: str) -> bool:
    upper = param.upper()
    return any(marker in upper for marker in SENSITIVE_KEY_MARKERS)


def _env_index(repo_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for env_path in sorted(repo_root.glob(".env*")):
        if not env_path.is_file():
            continue
        if env_path.name.endswith(".candidate"):
            continue
        if env_path.name in {".env.example", ".env.sample", ".env.template"}:
            continue
        try:
            env_map = parse_env(read_env_lines(env_path))
        except Exception:
            continue
        market = str(env_map.get("MM_MARKET_NAME") or "").strip()
        if not market:
            continue
        current = out.get(market)
        if current is None:
            out[market] = env_path
        elif current.name == ".env" and env_path.name != ".env":
            out[market] = env_path
    return out


def _resolve_env_path(repo_root: Path, market: str, env_file_raw: Optional[Any]) -> Path:
    if env_file_raw:
        path = Path(str(env_file_raw)).expanduser()
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if not path.exists():
            raise SubmissionError(f"env file not found: {path}")
        env_map = parse_env(read_env_lines(path))
        actual_market = str(env_map.get("MM_MARKET_NAME") or "").strip()
        if actual_market and actual_market != market:
            raise SubmissionError(
                f"env file market mismatch: expected {market}, got {actual_market} ({path})"
            )
        return path

    idx = _env_index(repo_root)
    path = idx.get(market)
    if path is None:
        market_slug = slugify(market).lower()
        guesses = [f".env.{market_slug}"]
        if "_" in market_slug:
            guesses.append(f".env.{market_slug.split('_', 1)[0]}")
        primary = market.lower().split("-", 1)[0]
        guesses.append(f".env.{primary}")
        for name in guesses:
            candidate = (repo_root / name).resolve()
            if candidate.exists():
                path = candidate
                break
    if path is None:
        raise SubmissionError(f"could not resolve env file for market: {market}")
    return path


def _build_proposal_row(
    *,
    repo_root: Path,
    existing_ids: set[str],
    whitelist: set[str],
    bounds: Dict[str, Dict[str, Any]],
    protected: set[str],
    market: str,
    param: str,
    proposed: Any,
    reason: Optional[Any],
    confidence: str,
    escalation_target: str,
    deadman: bool,
    env_file_raw: Optional[Any],
    proposal_id: Optional[Any],
    source: str,
) -> Dict[str, Any]:
    market_text = str(market or "").strip()
    if not MARKET_RE.fullmatch(market_text):
        raise SubmissionError(f"invalid market: {market!r}")
    param_text = str(param or "").strip()
    if not ENV_KEY_RE.fullmatch(param_text):
        raise SubmissionError(f"invalid param: {param!r}")
    if param_text not in whitelist:
        raise SubmissionError(f"param not whitelisted: {param_text}")
    if param_text in protected or _looks_sensitive_key(param_text):
        raise SubmissionError(f"param is protected: {param_text}")
    rule = bounds.get(param_text)
    if rule is None:
        raise SubmissionError(f"missing bounds rule for param: {param_text}")

    proposed_text = _normalize_value(param_text, proposed, rule)
    env_path = _resolve_env_path(repo_root, market_text, env_file_raw)
    env_map = parse_env(read_env_lines(env_path))
    old_value = env_map.get(param_text)

    proposal_id_text = str(proposal_id or "").strip()
    if not proposal_id_text:
        base = f"{slugify(market_text)}-{int(now_ts())}-manual-{slugify(param_text)}"
        proposal_id_text = base
        suffix = 1
        while proposal_id_text in existing_ids:
            suffix += 1
            proposal_id_text = f"{base}-{suffix}"
    elif proposal_id_text in existing_ids:
        raise SubmissionError(f"proposal_id already exists: {proposal_id_text}")

    confidence_text = str(confidence or "medium").strip().lower()
    if confidence_text not in {"low", "medium", "high"}:
        raise SubmissionError(f"invalid confidence: {confidence!r}")
    escalation_text = str(escalation_target or "human").strip().lower()
    if escalation_text not in {"human", "warren"}:
        raise SubmissionError(f"invalid escalation_target: {escalation_target!r}")

    now_value = now_ts()
    reason_note = str(reason or "").strip()
    row = {
        "proposal_id": proposal_id_text,
        "ts": now_value,
        "created_at": iso_utc(now_value),
        "market": market_text,
        "env_path": str(env_path),
        "iteration": 0,
        "param": param_text,
        "old": old_value,
        "proposed": proposed_text,
        "new": proposed_text,
        "baseline_value": old_value,
        "reason_codes": ["manual_submit"],
        "confidence": confidence_text,
        "guardrail_status": "passed",
        "guardrail_reason": "ok",
        "proposal_only": True,
        "applied": False,
        "rejected": False,
        "deadman": bool(deadman),
        "escalation_target": escalation_text,
        "cooldown_until_ts": None,
        "source": source,
    }
    if reason_note:
        row["reason_note"] = reason_note
    return row


def _iter_import_rows(path: Path) -> Iterable[Tuple[int, Optional[Dict[str, Any]], Optional[str]]]:
    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        text = raw.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            yield line_no, None, f"invalid JSON at {path}:{line_no}: {exc}"
            continue
        if not isinstance(row, dict):
            yield line_no, None, f"invalid row at {path}:{line_no}: expected object"
            continue
        yield line_no, row, None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create canonical advisor proposals safely.")
    parser.add_argument("--repo", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument(
        "--proposals",
        default="data/mm_audit/advisor/proposals.jsonl",
        help="Proposal stream path (relative to --repo).",
    )
    parser.add_argument("--market", default=None, help="Market (e.g. ETH-USD).")
    parser.add_argument("--param", default=None, help="Env key to set.")
    parser.add_argument("--proposed", default=None, help="Proposed value.")
    parser.add_argument("--reason", default=None, help="Short rationale note.")
    parser.add_argument(
        "--confidence",
        default="medium",
        choices=["low", "medium", "high"],
        help="Proposal confidence.",
    )
    parser.add_argument(
        "--escalation-target",
        default="human",
        choices=["human", "warren"],
        help="Escalation target.",
    )
    parser.add_argument("--deadman", action="store_true", help="Mark proposal as deadman.")
    parser.add_argument("--env-file", default=None, help="Optional explicit env file path.")
    parser.add_argument("--proposal-id", default=None, help="Optional explicit proposal_id.")
    parser.add_argument(
        "--source",
        default="mm_analyst_submit",
        help="Source label recorded in proposal rows.",
    )
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Optional import file (one JSON object per line).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no writes.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve()
    proposals_path = (repo_root / args.proposals).resolve()
    proposals_path.parent.mkdir(parents=True, exist_ok=True)
    if not proposals_path.exists():
        proposals_path.touch()

    whitelist, bounds, protected = _load_policy(repo_root)
    existing_ids = {str(x.get("proposal_id") or "") for x in read_jsonl(proposals_path)}

    created: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    def submit_one(payload: Dict[str, Any], line_no: Optional[int]) -> None:
        market = payload.get("market", args.market)
        param = payload.get("param", args.param)
        proposed = payload.get("proposed", args.proposed)
        if market is None or param is None or proposed is None:
            where = f" line={line_no}" if line_no is not None else ""
            raise SubmissionError(f"missing market/param/proposed{where}")
        row = _build_proposal_row(
            repo_root=repo_root,
            existing_ids=existing_ids,
            whitelist=whitelist,
            bounds=bounds,
            protected=protected,
            market=str(market),
            param=str(param),
            proposed=proposed,
            reason=payload.get("reason", args.reason),
            confidence=str(payload.get("confidence", args.confidence)),
            escalation_target=str(payload.get("escalation_target", args.escalation_target)),
            deadman=bool(payload.get("deadman", args.deadman)),
            env_file_raw=payload.get("env_file", args.env_file),
            proposal_id=payload.get("proposal_id", args.proposal_id),
            source=str(payload.get("source", args.source)),
        )
        existing_ids.add(row["proposal_id"])
        created.append(row)

    try:
        if args.input_jsonl:
            import_path = Path(args.input_jsonl).expanduser()
            if not import_path.is_absolute():
                import_path = (Path.cwd() / import_path).resolve()
            if not import_path.exists():
                raise SubmissionError(f"input file not found: {import_path}")
            for line_no, payload, parse_error in _iter_import_rows(import_path):
                if parse_error:
                    failures.append({"line": line_no, "error": parse_error})
                    continue
                assert payload is not None
                try:
                    submit_one(payload, line_no)
                except SubmissionError as exc:
                    failures.append({"line": line_no, "error": str(exc)})
        else:
            submit_one({}, None)
    except SubmissionError as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(f"error={exc}")
        return 1

    all_valid = len(created) > 0 and len(failures) == 0
    if not args.dry_run and all_valid:
        for row in created:
            append_jsonl(proposals_path, row)

    summary = {
        "ok": all_valid,
        "created": len(created),
        "written": len(created) if (not args.dry_run and all_valid) else 0,
        "failed": len(failures),
        "proposals_path": str(proposals_path),
        "proposal_ids": [row["proposal_id"] for row in created],
        "failures": failures,
    }
    if args.json:
        print(json.dumps(summary))
    else:
        print(
            " ".join(
                [
                    f"created={summary['created']}",
                    f"written={summary['written']}",
                    f"failed={summary['failed']}",
                    f"proposals_path={summary['proposals_path']}",
                ]
            )
        )
        for pid in summary["proposal_ids"]:
            print(pid)
        for item in failures:
            print(f"failure line={item['line']} error={item['error']}")

    if not all_valid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
