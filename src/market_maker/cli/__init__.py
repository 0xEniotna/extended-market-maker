"""mmctl — Market Maker Control Plane CLI."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mmctl",
        description="Market Maker Control Plane",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    sub = parser.add_subparsers(dest="command")

    # --- Instance management ---
    from market_maker.cli.instance import register as register_instance

    register_instance(sub)

    # --- PnL ---
    from market_maker.cli.pnl import register as register_pnl

    register_pnl(sub)

    # --- Positions ---
    from market_maker.cli.positions import register as register_positions

    register_positions(sub)

    # --- Markets ---
    from market_maker.cli.markets import register as register_markets

    register_markets(sub)

    # --- Journal ---
    from market_maker.cli.journal import register as register_journal

    register_journal(sub)

    # --- Legacy config commands (from original mmctl.py) ---
    from market_maker.cli.config_cmd import register as register_config

    register_config(sub)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
