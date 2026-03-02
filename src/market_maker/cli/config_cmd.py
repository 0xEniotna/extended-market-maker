"""mmctl config — Config proposal management (legacy commands from original mmctl.py)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _handle_apply(args) -> int:
    from market_maker.mm_config_pipeline import (
        ConfigProposalError,
        ProposalApplyError,
        ProposalValidationError,
        load_manager,
    )

    repo_root = Path(args.repo_root).resolve()
    config_root = Path(args.config_root).resolve() if args.config_root else None
    manager = load_manager(repo_root=repo_root, config_root=config_root)

    try:
        result = manager.apply_proposal(args.proposal, lock_timeout_s=args.lock_timeout_s)
        payload = asdict(result)
        if args.json:
            _print_json(payload)
        else:
            print(
                f"ok={payload['ok']} proposal_id={payload['proposal_id']} "
                f"market={payload['market']} env_file={payload['env_file']}"
            )
        return 0 if payload["ok"] else 1
    except (ConfigProposalError, ProposalValidationError, ProposalApplyError) as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


def _handle_rollback(args) -> int:
    from market_maker.mm_config_pipeline import (
        ConfigProposalError,
        ProposalApplyError,
        ProposalValidationError,
        load_manager,
    )

    repo_root = Path(args.repo_root).resolve()
    config_root = Path(args.config_root).resolve() if args.config_root else None
    manager = load_manager(repo_root=repo_root, config_root=config_root)

    try:
        rb_result = manager.rollback(args.market, args.to, lock_timeout_s=args.lock_timeout_s)
        payload = asdict(rb_result)
        if args.json:
            _print_json(payload)
        else:
            print(
                f"ok={payload['ok']} market={payload['market']} "
                f"env_file={payload['env_file']} restored_from={payload['restored_from']}"
            )
        return 0 if payload["ok"] else 1
    except (ConfigProposalError, ProposalValidationError, ProposalApplyError) as exc:
        if args.json:
            _print_json({"ok": False, "error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


def _handle_diff(args) -> int:
    from market_maker.mm_config_pipeline import (
        ConfigProposalError,
        ProposalValidationError,
        load_manager,
    )

    repo_root = Path(args.repo_root).resolve()
    config_root = Path(args.config_root).resolve() if args.config_root else None
    manager = load_manager(repo_root=repo_root, config_root=config_root)

    try:
        payload = manager.diff_proposal(args.proposal)
        print(f"Proposal {payload['proposal_id']} market={payload['market']}")
        print(f"Env file: {payload['env_file']}")
        for row in payload["changes"]:
            marker = "=" if not row["changed"] else "->"
            print(f"  {row['key']}: {row['old']} {marker} {row['new']}")
        return 0
    except (ConfigProposalError, ProposalValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def register(subparsers) -> None:
    config_parser = subparsers.add_parser("config", help="Config proposal management")
    config_parser.add_argument(
        "--config-root",
        default=None,
        help="Config root directory (default: <repo-root>/mm_config).",
    )
    config_sub = config_parser.add_subparsers(dest="config_command")

    # apply
    apply_p = config_sub.add_parser("apply", help="Validate + apply one proposal")
    apply_p.add_argument("proposal", help="Proposal id or JSON file path")
    apply_p.add_argument("--json", action="store_true", help="Emit JSON output")
    apply_p.add_argument(
        "--lock-timeout-s", type=float, default=10.0, help="File lock timeout."
    )
    apply_p.set_defaults(func=_handle_apply)

    # rollback
    rollback_p = config_sub.add_parser("rollback", help="Rollback one market env file")
    rollback_p.add_argument("market", help="Market symbol (e.g., MON)")
    rollback_p.add_argument("--to", required=True, help="Snapshot path or snapshot id")
    rollback_p.add_argument("--json", action="store_true", help="Emit JSON output")
    rollback_p.add_argument(
        "--lock-timeout-s", type=float, default=10.0, help="File lock timeout."
    )
    rollback_p.set_defaults(func=_handle_rollback)

    # diff
    diff_p = config_sub.add_parser("diff", help="Show proposal diff without applying")
    diff_p.add_argument("proposal", help="Proposal id or JSON file path")
    diff_p.set_defaults(func=_handle_diff)

    config_parser.set_defaults(func=lambda args: _config_help(config_parser, args))


def _config_help(parser, args) -> int:
    if not getattr(args, "config_command", None):
        parser.print_help()
        return 1
    return 0
