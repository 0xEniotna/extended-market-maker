from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from market_maker.mm_config_pipeline import (
    ConfigProposalError,
    ProposalApplyError,
    ProposalValidationError,
    load_manager,
)


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MM config proposal control plane")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--config-root",
        default=None,
        help="Config root directory (default: <repo-root>/mm_config).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    apply_p = sub.add_parser("apply-proposal", help="Validate + apply one proposal")
    apply_p.add_argument("proposal", help="Proposal id or JSON file path")
    apply_p.add_argument("--json", action="store_true", help="Emit JSON output")
    apply_p.add_argument(
        "--lock-timeout-s",
        type=float,
        default=10.0,
        help="File lock timeout in seconds.",
    )

    rollback_p = sub.add_parser("rollback", help="Rollback one market env file")
    rollback_p.add_argument("market", help="Market symbol (e.g., MON)")
    rollback_p.add_argument("--to", required=True, help="Snapshot path or snapshot id")
    rollback_p.add_argument("--json", action="store_true", help="Emit JSON output")
    rollback_p.add_argument(
        "--lock-timeout-s",
        type=float,
        default=10.0,
        help="File lock timeout in seconds.",
    )

    diff_p = sub.add_parser("diff-proposal", help="Show proposal diff without applying")
    diff_p.add_argument("proposal", help="Proposal id or JSON file path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    config_root = Path(args.config_root).resolve() if args.config_root else None
    manager = load_manager(repo_root=repo_root, config_root=config_root)

    try:
        if args.command == "apply-proposal":
            result = manager.apply_proposal(args.proposal, lock_timeout_s=args.lock_timeout_s)
            payload = asdict(result)
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"ok={payload['ok']} proposal_id={payload['proposal_id']} market={payload['market']} "
                    f"env_file={payload['env_file']}"
                )
            return 0 if payload["ok"] else 1

        if args.command == "rollback":
            result = manager.rollback(
                args.market,
                args.to,
                lock_timeout_s=args.lock_timeout_s,
            )
            payload = asdict(result)
            if args.json:
                _print_json(payload)
            else:
                print(
                    f"ok={payload['ok']} market={payload['market']} env_file={payload['env_file']} "
                    f"restored_from={payload['restored_from']}"
                )
            return 0 if payload["ok"] else 1

        if args.command == "diff-proposal":
            payload = manager.diff_proposal(args.proposal)
            print(f"Proposal {payload['proposal_id']} market={payload['market']}")
            print(f"Env file: {payload['env_file']}")
            for row in payload["changes"]:
                marker = "=" if not row["changed"] else "->"
                print(f"  {row['key']}: {row['old']} {marker} {row['new']}")
            return 0

        raise SystemExit(f"Unsupported command: {args.command}")
    except (ConfigProposalError, ProposalValidationError, ProposalApplyError) as exc:
        if getattr(args, "json", False):
            _print_json({"ok": False, "error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
