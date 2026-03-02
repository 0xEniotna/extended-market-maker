"""mmctl start/stop/restart/status — MM instance management via PID files."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from market_maker.cli.common import PROJECT_ROOT, resolve_env_file, to_jsonable
from market_maker.audit_common import parse_env, read_env_lines

PID_DIR = PROJECT_ROOT / "data" / "pids"
LOG_DIR = PROJECT_ROOT / "data" / "mm_journal"
GRACE_PERIOD_S = 30


def _find_python() -> str:
    """Find the best Python binary."""
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _normalize_env_label(raw: str) -> str:
    """Normalize market/env label to an env filename.

    Accepts: 'eth', 'ETH-USD', '.env.eth', 'env.eth'
    Returns: '.env.eth'
    """
    raw = raw.strip()
    if not raw:
        return ""

    # Already a path or .env* format
    if raw.startswith("/") or raw.startswith(".env") or "/" in raw:
        return raw

    lowered = raw.lower()
    if lowered.endswith("-usd"):
        lowered = lowered[:-4]
    # Handle underscored market names like AMZN_24_5
    lowered = lowered.replace("-", "_")

    return f".env.{lowered}"


def _controller_id(env_label: str) -> str:
    """Generate a clean controller ID from an env label."""
    base = os.path.basename(env_label)
    base = re.sub(r"[^A-Za-z0-9]", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "default"


def _pid_file(controller_id: str) -> Path:
    return PID_DIR / f"{controller_id}.pid"


def _log_file(controller_id: str) -> Path:
    return LOG_DIR / f"mm_{controller_id}.log"


def _is_running(pid_file: Path) -> tuple[bool, Optional[int]]:
    """Check if a process is running from its PID file."""
    if not pid_file.exists():
        return False, None
    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return False, None
    try:
        os.kill(pid, 0)
        return True, pid
    except (ProcessLookupError, PermissionError):
        return False, pid


def _read_market_from_env(env_path: Path) -> Optional[str]:
    """Read MM_MARKET_NAME from an env file."""
    if not env_path.exists():
        return None
    try:
        env_map = parse_env(read_env_lines(env_path))
        return env_map.get("MM_MARKET_NAME")
    except Exception:
        return None


def _start_instance(env_label: str) -> Dict[str, Any]:
    """Start a market maker instance."""
    env_file = resolve_env_file(env_label)
    if not env_file.exists():
        return {"status": "error", "error": f"Env file not found: {env_file}"}

    controller_id = _controller_id(env_label)
    pid_file = _pid_file(controller_id)
    log_file = _log_file(controller_id)

    running, existing_pid = _is_running(pid_file)
    if running:
        market = _read_market_from_env(env_file)
        return {
            "status": "already_running",
            "pid": existing_pid,
            "market": market,
            "env": str(env_file),
            "log": str(log_file),
        }

    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build environment: merge current env with .env file values
    proc_env = dict(os.environ)
    try:
        env_values = parse_env(read_env_lines(env_file))
        proc_env.update(env_values)
    except Exception as exc:
        return {"status": "error", "error": f"Failed to parse env file: {exc}"}

    proc_env["ENV"] = str(env_file)
    proc_env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    python_bin = _find_python()
    market = env_values.get("MM_MARKET_NAME", "unknown")

    with open(log_file, "a") as log_fh:
        proc = subprocess.Popen(
            [python_bin, "-m", "market_maker.strategy"],
            env=proc_env,
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    pid_file.write_text(str(proc.pid))

    return {
        "status": "started",
        "pid": proc.pid,
        "market": market,
        "env": str(env_file),
        "log": str(log_file),
    }


def _stop_instance(env_label: str) -> Dict[str, Any]:
    """Stop a market maker instance with graceful SIGINT then SIGTERM."""
    controller_id = _controller_id(env_label)
    pid_file = _pid_file(controller_id)
    log_file = _log_file(controller_id)

    running, pid = _is_running(pid_file)
    if not running:
        # Clean up stale PID file
        if pid_file.exists():
            pid_file.unlink(missing_ok=True)
        return {"status": "not_running", "env": env_label}

    assert pid is not None

    # Graceful shutdown: SIGINT first
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return {"status": "stopped", "pid": pid}

    # Wait for graceful shutdown
    waited = 0
    while waited < GRACE_PERIOD_S:
        time.sleep(1)
        waited += 1
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        # Force kill
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    pid_file.unlink(missing_ok=True)
    return {"status": "stopped", "pid": pid, "log": str(log_file)}


def _status_all() -> List[Dict[str, Any]]:
    """Scan PID directory and check each instance."""
    results = []
    if not PID_DIR.exists():
        return results

    for pid_file in sorted(PID_DIR.glob("*.pid")):
        controller_id = pid_file.stem
        running, pid = _is_running(pid_file)
        log_file = _log_file(controller_id)

        # Try to find the env file and market
        env_label = controller_id.replace("env_", ".env.")
        env_file = resolve_env_file(env_label)
        market = _read_market_from_env(env_file) if env_file.exists() else None

        results.append({
            "controller_id": controller_id,
            "status": "running" if running else "stopped",
            "pid": pid,
            "market": market,
            "env": str(env_file) if env_file.exists() else None,
            "log": str(log_file),
        })

    return results


def _status_one(env_label: str) -> Dict[str, Any]:
    """Check status of a specific instance."""
    controller_id = _controller_id(env_label)
    pid_file = _pid_file(controller_id)
    log_file = _log_file(controller_id)
    running, pid = _is_running(pid_file)

    env_file = resolve_env_file(env_label)
    market = _read_market_from_env(env_file) if env_file.exists() else None

    return {
        "controller_id": controller_id,
        "status": "running" if running else "stopped",
        "pid": pid,
        "market": market,
        "env": str(env_file) if env_file.exists() else None,
        "log": str(log_file),
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_start(args) -> int:
    env_label = _normalize_env_label(args.market)
    result = _start_instance(env_label)

    if getattr(args, "json", False):
        print(json.dumps(to_jsonable(result), indent=2))
    else:
        status = result.get("status")
        market = result.get("market", "-")
        if status == "started":
            print(f"Started {market} (pid={result['pid']}, log={result['log']})")
        elif status == "already_running":
            print(f"Already running {market} (pid={result['pid']})")
        else:
            print(f"Error: {result.get('error', 'unknown')}", file=sys.stderr)
            return 1

    return 0


def _handle_stop(args) -> int:
    env_label = _normalize_env_label(args.market)
    result = _stop_instance(env_label)

    if getattr(args, "json", False):
        print(json.dumps(to_jsonable(result), indent=2))
    else:
        status = result.get("status")
        if status == "stopped":
            print(f"Stopped (pid={result.get('pid')})")
        elif status == "not_running":
            print("Not running.")
        else:
            print(f"Error: {result.get('error', 'unknown')}", file=sys.stderr)
            return 1

    return 0


def _handle_restart(args) -> int:
    env_label = _normalize_env_label(args.market)
    stop_result = _stop_instance(env_label)
    if stop_result.get("status") == "stopped":
        # Brief pause for port/resource release
        time.sleep(1)

    start_result = _start_instance(env_label)

    if getattr(args, "json", False):
        print(json.dumps(to_jsonable({"stop": stop_result, "start": start_result}), indent=2))
    else:
        market = start_result.get("market", "-")
        if start_result.get("status") == "started":
            print(f"Restarted {market} (pid={start_result['pid']})")
        else:
            print(f"Restart failed: {start_result.get('error', 'unknown')}", file=sys.stderr)
            return 1

    return 0


def _handle_status(args) -> int:
    market = getattr(args, "market", None)

    if market:
        env_label = _normalize_env_label(market)
        result = _status_one(env_label)
        results = [result]
    else:
        results = _status_all()

    if getattr(args, "json", False):
        payload = {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "instances": results,
        }
        print(json.dumps(to_jsonable(payload), indent=2))
    else:
        if not results:
            print("No instances found.")
        else:
            for r in results:
                status_icon = "RUNNING" if r["status"] == "running" else "STOPPED"
                market_name = r.get("market") or r.get("controller_id") or "-"
                pid_str = f" pid={r['pid']}" if r.get("pid") else ""
                print(f"  {status_icon}  {market_name}{pid_str}")

    return 0


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def register(subparsers) -> None:
    # start
    start_p = subparsers.add_parser("start", help="Start a market maker instance")
    start_p.add_argument("market", help="Market or env label (e.g. eth, .env.eth, ETH-USD)")
    start_p.add_argument("--json", action="store_true", help="JSON output")
    start_p.set_defaults(func=_handle_start)

    # stop
    stop_p = subparsers.add_parser("stop", help="Stop a market maker instance")
    stop_p.add_argument("market", help="Market or env label")
    stop_p.add_argument("--json", action="store_true", help="JSON output")
    stop_p.set_defaults(func=_handle_stop)

    # restart
    restart_p = subparsers.add_parser("restart", help="Restart a market maker instance")
    restart_p.add_argument("market", help="Market or env label")
    restart_p.add_argument("--json", action="store_true", help="JSON output")
    restart_p.set_defaults(func=_handle_restart)

    # status
    status_p = subparsers.add_parser("status", help="Show running instances")
    status_p.add_argument("market", nargs="?", help="Optional specific market")
    status_p.add_argument("--json", action="store_true", help="JSON output")
    status_p.set_defaults(func=_handle_status)
