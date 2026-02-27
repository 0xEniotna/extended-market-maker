from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import fcntl

DEFAULT_CONFIG_ROOT = Path("mm_config")
DEFAULT_ENV_DIR = Path("./mm-env")

POLICY_DIR = "policy"
PROPOSALS_DIR = "proposals"
APPLIED_DIR = "applied"
SNAPSHOTS_DIR = "snapshots"

PROPOSAL_TOP_LEVEL_KEYS = {
    "proposal_id",
    "market",
    "env_file",
    "changes",
    "reason",
    "canary",
    "created_at",
}
PROPOSAL_REQUIRED_KEYS = {
    "proposal_id",
    "market",
    "changes",
    "reason",
    "canary",
    "created_at",
}
CHANGE_KEYS = {"key", "op", "value"}
ALLOWED_OPS = {"set"}
MARKET_RE = re.compile(r"^[A-Z0-9]+$")
ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
ENV_ASSIGN_RE = re.compile(r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$")
SENSITIVE_KEY_MARKERS = (
    "PRIVATE_KEY",
    "_SECRET",
    "PASSPHRASE",
    "MNEMONIC",
    "_API_KEY",
)


class ConfigProposalError(RuntimeError):
    pass


class ProposalValidationError(ConfigProposalError):
    pass


class ProposalApplyError(ConfigProposalError):
    pass


@dataclass(frozen=True)
class Policy:
    whitelist: set[str]
    bounds: Dict[str, Dict[str, Any]]
    protected: set[str]


@dataclass(frozen=True)
class ValidatedChange:
    key: str
    op: str
    value: str


@dataclass(frozen=True)
class ValidatedProposal:
    proposal_id: str
    market: str
    env_file: Path
    changes: Tuple[ValidatedChange, ...]
    reason: Dict[str, Any]
    canary: Dict[str, Any]
    created_at: str
    source_path: Optional[Path]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(ts: Optional[datetime] = None) -> str:
    value = ts or _now_utc()
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigProposalError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigProposalError(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _canonical_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _ensure_runtime_dirs(config_root: Path) -> Dict[str, Path]:
    paths = {
        "policy": config_root / POLICY_DIR,
        "proposals": config_root / PROPOSALS_DIR,
        "applied": config_root / APPLIED_DIR,
        "snapshots": config_root / SNAPSHOTS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _resolve_env_dir(repo_root: Path) -> Path:
    raw = os.getenv("MM_ENV_DIR", str(DEFAULT_ENV_DIR))
    env_dir = Path(raw).expanduser()
    if not env_dir.is_absolute():
        env_dir = (repo_root / env_dir).resolve()
    return env_dir


def _normalize_market(raw: Any) -> str:
    value = str(raw or "").strip().upper()
    if not MARKET_RE.match(value):
        raise ProposalValidationError(f"Invalid market: {raw!r}")
    return value


def _is_sensitive_key_name(key: str) -> bool:
    upper = key.upper()
    return any(marker in upper for marker in SENSITIVE_KEY_MARKERS)


def _parse_iso8601(raw: Any) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ProposalValidationError("created_at is required")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ProposalValidationError(f"Invalid created_at timestamp: {value}") from exc
    return value


def _parse_int(raw: Any, *, key: str) -> int:
    text = str(raw).strip()
    if not text:
        raise ProposalValidationError(f"Empty value for int key {key}")
    if not re.fullmatch(r"[+-]?\d+", text):
        raise ProposalValidationError(f"Invalid int for {key}: {raw!r}")
    return int(text)


def _parse_float(raw: Any, *, key: str) -> Decimal:
    text = str(raw).strip()
    if not text:
        raise ProposalValidationError(f"Empty value for float key {key}")
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise ProposalValidationError(f"Invalid float for {key}: {raw!r}") from exc


def _normalize_bool(raw: Any, *, key: str) -> str:
    text = str(raw).strip().lower()
    if text in {"true", "1"}:
        return "true"
    if text in {"false", "0"}:
        return "false"
    raise ProposalValidationError(f"Invalid bool for {key}: {raw!r}")


def _value_with_rule(key: str, raw_value: Any, rule: Dict[str, Any]) -> str:
    value_type = str(rule.get("type") or "").strip().lower()
    if value_type == "int":
        parsed = _parse_int(raw_value, key=key)
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and parsed < int(min_v):
            raise ProposalValidationError(f"{key} below min bound ({min_v})")
        if max_v is not None and parsed > int(max_v):
            raise ProposalValidationError(f"{key} above max bound ({max_v})")
        return str(parsed)
    if value_type == "float":
        parsed = _parse_float(raw_value, key=key)
        min_v = rule.get("min")
        max_v = rule.get("max")
        if min_v is not None and parsed < Decimal(str(min_v)):
            raise ProposalValidationError(f"{key} below min bound ({min_v})")
        if max_v is not None and parsed > Decimal(str(max_v)):
            raise ProposalValidationError(f"{key} above max bound ({max_v})")
        return _canonical_decimal(parsed)
    if value_type == "bool":
        return _normalize_bool(raw_value, key=key)
    if value_type == "string":
        text = str(raw_value)
        allowed = rule.get("allowed")
        if isinstance(allowed, list) and text not in {str(v) for v in allowed}:
            raise ProposalValidationError(f"{key} not in allowed set")
        return text
    raise ProposalValidationError(f"Unsupported type for {key}: {value_type!r}")


def _split_value_comment(value_part: str) -> Tuple[str, str]:
    match = re.match(r"^(.*?)(\s+#.*)$", value_part)
    if not match:
        return value_part, ""
    return match.group(1), match.group(2)


def update_env_lines_preserving_format(lines: List[str], updates: Dict[str, str]) -> List[str]:
    remaining = dict(updates)
    out: List[str] = []
    for raw_line in lines:
        newline = "\n" if raw_line.endswith("\n") else ""
        body = raw_line[:-1] if newline else raw_line
        match = ENV_ASSIGN_RE.match(body)
        if not match:
            out.append(raw_line)
            continue

        prefix, key, eq, value_part = match.groups()
        if key not in updates:
            out.append(raw_line)
            continue

        _, comment = _split_value_comment(value_part)
        new_value = updates[key]
        out.append(f"{prefix}{key}{eq}{new_value}{comment}{newline}")
        remaining.pop(key, None)

    if remaining:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        for key in sorted(remaining):
            out.append(f"{key}={remaining[key]}\n")
    return out


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@contextmanager
def env_file_lock(env_file: Path, timeout_s: float = 10.0, poll_s: float = 0.05) -> Iterator[None]:
    lock_path = env_file.with_name(env_file.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_fh:
        start = time.monotonic()
        while True:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start >= timeout_s:
                    raise ProposalApplyError(f"Timeout acquiring lock: {lock_path}")
                time.sleep(poll_s)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


class ProposalValidator:
    def __init__(self, *, policy: Policy, repo_root: Path, env_dir: Path):
        self._policy = policy
        self._repo_root = repo_root
        self._env_dir = env_dir

    def _resolve_env_file(self, *, market: str, env_file_raw: Optional[Any]) -> Path:
        expected_name = f".env.{market.lower()}"
        if env_file_raw is None:
            return (self._env_dir / expected_name).resolve()
        env_file = Path(str(env_file_raw)).expanduser()
        if not env_file.is_absolute():
            env_file = (self._repo_root / env_file).resolve()
        if env_file.name != expected_name:
            raise ProposalValidationError(
                f"env_file must match market naming ({expected_name}), got {env_file.name}"
            )
        return env_file

    def _validate_change(self, raw: Dict[str, Any]) -> ValidatedChange:
        unknown = set(raw.keys()) - CHANGE_KEYS
        if unknown:
            raise ProposalValidationError(f"Unknown change fields: {sorted(unknown)}")
        missing = CHANGE_KEYS - set(raw.keys())
        if missing:
            raise ProposalValidationError(f"Missing change fields: {sorted(missing)}")

        key = str(raw.get("key") or "").strip()
        if not ENV_KEY_RE.match(key):
            raise ProposalValidationError(f"Invalid env key: {key!r}")
        if key not in self._policy.whitelist:
            raise ProposalValidationError(f"Key {key} is not whitelisted")
        if key in self._policy.protected or _is_sensitive_key_name(key):
            raise ProposalValidationError(f"Key {key} is protected and cannot be changed")
        if key not in self._policy.bounds:
            raise ProposalValidationError(f"Missing bounds rule for key {key}")

        op = str(raw.get("op") or "").strip().lower()
        if op not in ALLOWED_OPS:
            raise ProposalValidationError(f"Unsupported op {op!r} for key {key}")

        normalized_value = _value_with_rule(key, raw.get("value"), self._policy.bounds[key])
        return ValidatedChange(key=key, op=op, value=normalized_value)

    def validate(self, payload: Dict[str, Any], *, source_path: Optional[Path]) -> ValidatedProposal:
        unknown = set(payload.keys()) - PROPOSAL_TOP_LEVEL_KEYS
        if unknown:
            raise ProposalValidationError(f"Unknown proposal fields: {sorted(unknown)}")
        missing = PROPOSAL_REQUIRED_KEYS - set(payload.keys())
        if missing:
            raise ProposalValidationError(f"Missing proposal fields: {sorted(missing)}")

        proposal_id = str(payload.get("proposal_id") or "").strip()
        if not proposal_id:
            raise ProposalValidationError("proposal_id is required")

        market = _normalize_market(payload.get("market"))
        created_at = _parse_iso8601(payload.get("created_at"))
        env_file = self._resolve_env_file(market=market, env_file_raw=payload.get("env_file"))

        reason = payload.get("reason")
        canary = payload.get("canary")
        if not isinstance(reason, dict):
            raise ProposalValidationError("reason must be an object")
        if not isinstance(canary, dict):
            raise ProposalValidationError("canary must be an object")

        raw_changes = payload.get("changes")
        if not isinstance(raw_changes, list) or not raw_changes:
            raise ProposalValidationError("changes must be a non-empty list")
        changes: List[ValidatedChange] = []
        seen_keys: set[str] = set()
        for raw_change in raw_changes:
            if not isinstance(raw_change, dict):
                raise ProposalValidationError("each change must be an object")
            change = self._validate_change(raw_change)
            if change.key in seen_keys:
                raise ProposalValidationError(f"duplicate change for key {change.key}")
            seen_keys.add(change.key)
            changes.append(change)

        return ValidatedProposal(
            proposal_id=proposal_id,
            market=market,
            env_file=env_file,
            changes=tuple(changes),
            reason=reason,
            canary=canary,
            created_at=created_at,
            source_path=source_path,
        )


def _load_policy(config_root: Path) -> Policy:
    whitelist_path = config_root / POLICY_DIR / "whitelist.json"
    bounds_path = config_root / POLICY_DIR / "bounds.json"
    protected_path = config_root / POLICY_DIR / "protected_keys.json"
    whitelist_raw = _load_json(whitelist_path)
    bounds_raw = _load_json(bounds_path)
    protected_raw: Any = []
    if protected_path.exists():
        protected_raw = _load_json(protected_path)

    if not isinstance(whitelist_raw, list):
        raise ConfigProposalError(f"Whitelist must be an array: {whitelist_path}")
    if not isinstance(bounds_raw, dict):
        raise ConfigProposalError(f"Bounds must be an object: {bounds_path}")
    if not isinstance(protected_raw, list):
        raise ConfigProposalError(f"Protected keys must be an array: {protected_path}")

    whitelist: set[str] = set()
    for raw_key in whitelist_raw:
        key = str(raw_key).strip()
        if not ENV_KEY_RE.match(key):
            raise ConfigProposalError(f"Invalid whitelist key: {raw_key!r}")
        whitelist.add(key)

    bounds: Dict[str, Dict[str, Any]] = {}
    for key, rule in bounds_raw.items():
        key_text = str(key).strip()
        if not ENV_KEY_RE.match(key_text):
            raise ConfigProposalError(f"Invalid bounds key: {key!r}")
        if not isinstance(rule, dict):
            raise ConfigProposalError(f"Bounds rule for {key_text} must be an object")
        if "type" not in rule:
            raise ConfigProposalError(f"Bounds rule for {key_text} missing type")
        bounds[key_text] = dict(rule)
    protected: set[str] = set()
    for raw_key in protected_raw:
        key = str(raw_key).strip()
        if not ENV_KEY_RE.match(key):
            raise ConfigProposalError(f"Invalid protected key: {raw_key!r}")
        protected.add(key)
    return Policy(whitelist=whitelist, bounds=bounds, protected=protected)


def _hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def _snapshot_name(env_file: Path, content_hash: str) -> str:
    ts = _now_utc().strftime("%Y%m%dT%H%M%SZ")
    return f"{env_file.name}.{ts}.{content_hash[:12]}.bak"


@dataclass(frozen=True)
class ApplyResult:
    ok: bool
    proposal_id: str
    market: str
    env_file: str
    snapshot_before: str
    snapshot_after: str
    archived_proposal: str
    reload: Dict[str, Any]


@dataclass(frozen=True)
class RollbackResult:
    ok: bool
    market: str
    env_file: str
    restored_from: str
    snapshot_before: str
    reload: Dict[str, Any]


class ProposalManager:
    def __init__(self, *, repo_root: Path, config_root: Optional[Path] = None):
        self.repo_root = repo_root.resolve()
        self.config_root = (config_root or (self.repo_root / DEFAULT_CONFIG_ROOT)).resolve()
        self.paths = _ensure_runtime_dirs(self.config_root)
        self.env_dir = _resolve_env_dir(self.repo_root)
        self.policy = _load_policy(self.config_root)
        self.validator = ProposalValidator(
            policy=self.policy,
            repo_root=self.repo_root,
            env_dir=self.env_dir,
        )

    def _load_proposal_payload(self, proposal_id_or_path: str) -> Tuple[Path, Dict[str, Any]]:
        candidate = Path(proposal_id_or_path).expanduser()
        if candidate.exists() and candidate.is_file():
            path = candidate.resolve()
        else:
            path = (self.paths["proposals"] / f"{proposal_id_or_path}.json").resolve()
            if not path.exists():
                raise ConfigProposalError(f"Proposal not found: {proposal_id_or_path}")
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ProposalValidationError(f"Proposal must be a JSON object: {path}")
        return path, payload

    def load_and_validate_proposal(self, proposal_id_or_path: str) -> ValidatedProposal:
        proposal_path, payload = self._load_proposal_payload(proposal_id_or_path)
        return self.validator.validate(payload, source_path=proposal_path)

    def _snapshot(self, *, env_file: Path, proposal_id: str, stage: str) -> Path:
        if not env_file.exists():
            raise ProposalApplyError(f"Env file does not exist: {env_file}")
        content = env_file.read_bytes()
        content_hash = _hash_bytes(content)
        snapshot_name = _snapshot_name(env_file, content_hash)
        snapshot_path = (self.paths["snapshots"] / snapshot_name).resolve()
        _atomic_write_bytes(snapshot_path, content)
        metadata = {
            "proposal_id": proposal_id,
            "market": env_file.name.replace(".env.", "").upper(),
            "snapshot_path": str(snapshot_path),
            "snapshot_id": snapshot_path.name,
            "created_at": _iso_utc(),
            "stage": stage,
            "env_file": str(env_file),
            "sha256": content_hash,
        }
        _write_json(snapshot_path.with_suffix(snapshot_path.suffix + ".json"), metadata)
        return snapshot_path

    def _archive_proposal(
        self,
        *,
        proposal: ValidatedProposal,
        source_payload: Dict[str, Any],
        snapshot_before: Path,
        snapshot_after: Path,
        reload_result: Dict[str, Any],
    ) -> Path:
        ts = _now_utc().strftime("%Y%m%dT%H%M%SZ")
        archive_base = f"{proposal.proposal_id}.{ts}"
        archive_path = (self.paths["applied"] / f"{archive_base}.proposal.json").resolve()
        _write_json(archive_path, source_payload)

        metadata = {
            "proposal_id": proposal.proposal_id,
            "market": proposal.market,
            "env_file": str(proposal.env_file),
            "snapshot_before": str(snapshot_before),
            "snapshot_after": str(snapshot_after),
            "applied_at": _iso_utc(),
            "reload": reload_result,
        }
        metadata_path = (self.paths["applied"] / f"{archive_base}.meta.json").resolve()
        _write_json(metadata_path, metadata)

        if proposal.source_path and proposal.source_path.resolve().parent == self.paths["proposals"]:
            proposal.source_path.unlink(missing_ok=True)
        return archive_path

    def _run_reload(self, market: str) -> Dict[str, Any]:
        template = os.getenv("MM_RELOAD_CMD_TEMPLATE", "").strip()
        if not template:
            return {
                "mode": "noop",
                "ok": True,
                "message": "Reload is NOOP; process must be restarted to pick changes.",
            }

        command = template.replace("{market}", market.lower()).replace("{MARKET}", market.upper())
        timeout_s_raw = os.getenv("MM_RELOAD_TIMEOUT_S", "30")
        try:
            timeout_s = float(timeout_s_raw)
        except ValueError:
            timeout_s = 30.0

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "mode": "command",
                "ok": False,
                "command": command,
                "timeout_s": timeout_s,
                "error": "timeout",
            }
        return {
            "mode": "command",
            "ok": proc.returncode == 0,
            "command": command,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }

    def _read_env_lines(self, env_file: Path) -> List[str]:
        if not env_file.exists():
            raise ProposalApplyError(f"Env file not found: {env_file}")
        return env_file.read_text().splitlines(keepends=True)

    def _resolve_snapshot(self, *, market: str, target: str) -> Path:
        raw = Path(target).expanduser()
        if raw.exists():
            return raw.resolve()

        candidates: List[Path] = []
        direct = (self.paths["snapshots"] / target).resolve()
        if direct.exists():
            candidates.append(direct)
        if not target.endswith(".bak"):
            alt = (self.paths["snapshots"] / f"{target}.bak").resolve()
            if alt.exists():
                candidates.append(alt)

        market_lower = market.lower()
        pattern = f".env.{market_lower}.*.bak"
        for path in self.paths["snapshots"].glob(pattern):
            if target in path.name:
                candidates.append(path.resolve())

        unique = sorted({str(p): p for p in candidates}.values(), key=lambda p: str(p))
        if not unique:
            raise ProposalApplyError(f"Snapshot not found: {target}")
        if len(unique) > 1:
            raise ProposalApplyError(
                f"Snapshot id is ambiguous ({target}); matches: {[p.name for p in unique]}"
            )
        return unique[0]

    def apply_proposal(self, proposal_id_or_path: str, *, lock_timeout_s: float = 10.0) -> ApplyResult:
        proposal_path, source_payload = self._load_proposal_payload(proposal_id_or_path)
        proposal = self.validator.validate(source_payload, source_path=proposal_path)
        updates = {change.key: change.value for change in proposal.changes}

        with env_file_lock(proposal.env_file, timeout_s=lock_timeout_s):
            before_snapshot = self._snapshot(
                env_file=proposal.env_file,
                proposal_id=proposal.proposal_id,
                stage="before_apply",
            )
            lines = self._read_env_lines(proposal.env_file)
            new_lines = update_env_lines_preserving_format(lines, updates)
            new_text = "".join(new_lines)
            _atomic_write_text(proposal.env_file, new_text)
            after_snapshot = self._snapshot(
                env_file=proposal.env_file,
                proposal_id=proposal.proposal_id,
                stage="after_apply",
            )

        reload_result = self._run_reload(proposal.market)
        archived_proposal = self._archive_proposal(
            proposal=proposal,
            source_payload=source_payload,
            snapshot_before=before_snapshot,
            snapshot_after=after_snapshot,
            reload_result=reload_result,
        )

        return ApplyResult(
            ok=reload_result.get("ok", False) or reload_result.get("mode") == "noop",
            proposal_id=proposal.proposal_id,
            market=proposal.market,
            env_file=str(proposal.env_file),
            snapshot_before=str(before_snapshot),
            snapshot_after=str(after_snapshot),
            archived_proposal=str(archived_proposal),
            reload=reload_result,
        )

    def rollback(
        self,
        market: str,
        target: str,
        *,
        lock_timeout_s: float = 10.0,
    ) -> RollbackResult:
        normalized_market = _normalize_market(market)
        env_file = (self.env_dir / f".env.{normalized_market.lower()}").resolve()
        snapshot_path = self._resolve_snapshot(market=normalized_market, target=target)
        if not snapshot_path.exists():
            raise ProposalApplyError(f"Snapshot does not exist: {snapshot_path}")

        with env_file_lock(env_file, timeout_s=lock_timeout_s):
            before_snapshot = self._snapshot(
                env_file=env_file,
                proposal_id=f"rollback-{normalized_market}",
                stage="before_rollback",
            )
            snapshot_data = snapshot_path.read_bytes()
            _atomic_write_bytes(env_file, snapshot_data)

        reload_result = self._run_reload(normalized_market)
        return RollbackResult(
            ok=reload_result.get("ok", False) or reload_result.get("mode") == "noop",
            market=normalized_market,
            env_file=str(env_file),
            restored_from=str(snapshot_path),
            snapshot_before=str(before_snapshot),
            reload=reload_result,
        )

    def diff_proposal(self, proposal_id_or_path: str) -> Dict[str, Any]:
        proposal = self.load_and_validate_proposal(proposal_id_or_path)
        updates = {change.key: change.value for change in proposal.changes}

        current_map: Dict[str, str] = {}
        if proposal.env_file.exists():
            lines = proposal.env_file.read_text().splitlines()
            for raw in lines:
                stripped = raw.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.replace("export", "", 1).strip() if key.strip().startswith("export ") else key
                key = key.strip()
                if not key:
                    continue
                current_map[key] = value.split(" #", 1)[0].strip()

        changes = []
        for key in sorted(updates):
            old = current_map.get(key)
            new = updates[key]
            changes.append({"key": key, "old": old, "new": new, "changed": old != new})
        return {
            "proposal_id": proposal.proposal_id,
            "market": proposal.market,
            "env_file": str(proposal.env_file),
            "changes": changes,
        }


def load_manager(
    *,
    repo_root: Optional[Path] = None,
    config_root: Optional[Path] = None,
) -> ProposalManager:
    root = (repo_root or Path.cwd()).resolve()
    cfg_root = config_root.resolve() if config_root is not None else None
    return ProposalManager(repo_root=root, config_root=cfg_root)
