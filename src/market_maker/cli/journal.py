"""mmctl journal — Journal analysis, export, and reprice quality."""

from __future__ import annotations

import sys
from pathlib import Path

from market_maker.cli.common import PROJECT_ROOT


# ---------------------------------------------------------------------------
# mmctl journal analyze (wraps scripts/analyse_mm_journal.py)
# ---------------------------------------------------------------------------


def _run_analyze(args) -> int:
    """Delegate to the standalone journal analyzer (62KB, kept as importable module)."""
    scripts_dir = PROJECT_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # The analyzer uses sys.argv for argparse, so we pass through
    argv = list(getattr(args, "journal_files", None) or [])
    if not argv:
        argv = [str(PROJECT_ROOT / "data" / "mm_journal")]

    old_argv = sys.argv
    sys.argv = ["analyse_mm_journal"] + argv
    try:
        from analyse_mm_journal import main as journal_main

        return journal_main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# mmctl journal export (wraps scripts/tools/export_inventory_timeseries.py)
# ---------------------------------------------------------------------------


def _run_export(args) -> int:
    scripts_tools = PROJECT_ROOT / "scripts" / "tools"
    if str(scripts_tools) not in sys.path:
        sys.path.insert(0, str(scripts_tools))

    from export_inventory_timeseries import main as export_main

    argv = []
    if args.export_target:
        argv.append(args.export_target)
    if getattr(args, "market", None):
        argv.extend(["--market", args.market])
    if getattr(args, "bucket_seconds", None):
        argv.extend(["--bucket-seconds", str(args.bucket_seconds)])

    old_argv = sys.argv
    sys.argv = ["export_inventory_timeseries"] + argv
    try:
        export_main()
        return 0
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# mmctl journal reprice-quality (wraps scripts/tools/audit_reprice_quality.py)
# ---------------------------------------------------------------------------


def _run_reprice_quality(args) -> int:
    scripts_tools = PROJECT_ROOT / "scripts" / "tools"
    if str(scripts_tools) not in sys.path:
        sys.path.insert(0, str(scripts_tools))

    from audit_reprice_quality import build_parser as rq_build_parser, run as rq_run

    rq_parser = rq_build_parser()
    rq_argv = []
    if hasattr(args, "lookback_hours") and args.lookback_hours is not None:
        rq_argv.extend(["--lookback-hours", str(args.lookback_hours)])
    if hasattr(args, "env_map") and args.env_map:
        rq_argv.extend(["--env-map", args.env_map])

    rq_args = rq_parser.parse_args(rq_argv)
    return rq_run(rq_args)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    journal_parser = subparsers.add_parser("journal", help="Journal analysis and export")
    journal_sub = journal_parser.add_subparsers(dest="journal_command")

    # analyze
    analyze_p = journal_sub.add_parser("analyze", help="Analyze MM journal files")
    analyze_p.add_argument(
        "journal_files", nargs="*", default=None,
        help="Journal files or directory (default: data/mm_journal)",
    )
    analyze_p.set_defaults(func=_run_analyze)

    # export
    export_p = journal_sub.add_parser("export", help="Export inventory time-series as CSV")
    export_p.add_argument(
        "export_target", nargs="?", default=None,
        help="Journal file or directory (default: data/mm_journal)",
    )
    export_p.add_argument("--market", default=None, help="Market filter")
    export_p.add_argument("--bucket-seconds", type=int, default=60, help="Bucket size in seconds")
    export_p.set_defaults(func=_run_export)

    # reprice-quality
    rq_p = journal_sub.add_parser("reprice-quality", help="Audit reprice quality and side bias")
    rq_p.add_argument("--lookback-hours", type=float, default=1.0, help="Lookback window")
    rq_p.add_argument("--env-map", default=None, help="Market-to-env mapping")
    rq_p.set_defaults(func=_run_reprice_quality)

    journal_parser.set_defaults(func=lambda args: _journal_help(journal_parser, args))


def _journal_help(parser, args) -> int:
    if not getattr(args, "journal_command", None):
        parser.print_help()
        return 1
    return 0
