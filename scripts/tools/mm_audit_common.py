#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_ts() -> float:
    return time.time()


def iso_utc(ts: Optional[float] = None) -> str:
    if ts is None:
        ts = now_ts()
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def safe_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "unknown"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    text = path.read_text().strip()
    if not text:
        return default
    return json.loads(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(to_jsonable(row)) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_policy(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    text = path.read_text().strip()
    if not text:
        return {}

    # YAML is a superset of JSON, so this works for JSON-style YAML config.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Policy parsing failed. Use JSON-compatible YAML or install PyYAML."
            ) from exc
        payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise RuntimeError("Policy file must parse to a dictionary/object.")
    return payload


def read_env_lines(path: Path) -> List[str]:
    return path.read_text().splitlines()


def parse_env(lines: Iterable[str]) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value_clean = re.split(r"\s+#", value, maxsplit=1)[0].strip()
        env[key.strip()] = value_clean
    return env


def update_env_lines(lines: List[str], updates: Dict[str, str]) -> List[str]:
    remaining = dict(updates)
    out: List[str] = []
    for line in lines:
        if "=" not in line or line.strip().startswith("#"):
            out.append(line)
            continue
        key, rest = line.split("=", 1)
        key_clean = key.strip()
        if key_clean in updates:
            match = re.match(r"([^#]*)(#.*)?", rest)
            comment = match.group(2) if match else ""
            new_value = updates[key_clean]
            out.append(f"{key_clean}={new_value}{(' ' + comment) if comment else ''}")
            remaining.pop(key_clean, None)
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    return out


def _extract_env_path_from_text(text: str) -> Optional[str]:
    # Common patterns in command/prompt payloads.
    matches = re.findall(r"(\.?/[^\s]+\.env[^\s]*)", text)
    if matches:
        return matches[-1]
    matches = re.findall(r"(\.env[\w._/-]*)", text)
    if matches:
        return matches[-1]
    return None


def resolve_env_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _find_job_market(job: Dict[str, Any]) -> Optional[str]:
    for key in ("market", "market_name", "mm_market_name", "MM_MARKET_NAME"):
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    args = job.get("args")
    if isinstance(args, dict):
        for key in ("market", "market_name", "mm_market_name", "MM_MARKET_NAME"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _find_job_env_raw(job: Dict[str, Any]) -> Optional[str]:
    for key in ("env", "env_file", "base_env", "baseEnv", "ENV", "mm_env"):
        value = job.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    args = job.get("args")
    if isinstance(args, dict):
        for key in ("env", "env_file", "base_env", "baseEnv"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("command", "prompt", "input"):
        value = job.get(key)
        if isinstance(value, str):
            found = _extract_env_path_from_text(value)
            if found:
                return found
    return None


def load_market_jobs(jobs_path: Path, repo_root: Path) -> List[Dict[str, Any]]:
    payload = read_json(jobs_path, default={})
    jobs: List[Dict[str, Any]] = []

    if isinstance(payload, list):
        jobs = [j for j in payload if isinstance(j, dict)]
    elif isinstance(payload, dict):
        for key in ("jobs", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                jobs = [j for j in value if isinstance(j, dict)]
                break

    out: List[Dict[str, Any]] = []
    for idx, job in enumerate(jobs):
        enabled = job.get("enabled", True)
        if isinstance(enabled, bool) and not enabled:
            continue

        market = _find_job_market(job)
        env_path: Optional[Path] = None
        env_raw = _find_job_env_raw(job)
        if env_raw:
            env_path = resolve_env_path(env_raw, repo_root)
            if not env_path.exists():
                env_path = None

        if market is None and env_path is not None:
            try:
                env_map = parse_env(read_env_lines(env_path))
                market = env_map.get("MM_MARKET_NAME")
            except Exception:
                market = None

        if not market:
            continue

        out.append({
            "job_id": str(job.get("id") or job.get("jobId") or job.get("name") or f"job-{idx}"),
            "job_name": str(job.get("name") or job.get("title") or f"job-{idx}"),
            "market": str(market),
            "env_path": str(env_path) if env_path else None,
        })

    out.sort(key=lambda row: row["market"])
    return out


def find_latest_journal(journal_dir: Path, market: str) -> Optional[Path]:
    pattern = f"mm_{market}_*.jsonl"
    files = [
        p for p in journal_dir.glob(pattern)
        if "mm_tuning_log_" not in p.name
    ]
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]


def discover_recent_markets_from_journals(
    journal_dir: Path,
    *,
    lookback_s: float = 86400.0,
) -> List[str]:
    now = now_ts()
    markets: set[str] = set()
    if not journal_dir.exists():
        return []
    for path in journal_dir.glob("mm_*.jsonl"):
        if "mm_tuning_log_" in path.name:
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if now - mtime > lookback_s:
            continue
        match = re.match(r"^mm_(.+)_\d{8}_\d{6}\.jsonl$", path.name)
        if match:
            markets.add(match.group(1))
    return sorted(markets)


def extract_last_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    positions = [i for i, ch in enumerate(text) if ch == "{"]
    for pos in reversed(positions):
        candidate = text[pos:].lstrip()
        try:
            payload, _ = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise RuntimeError("Unable to parse JSON object from command output.")
