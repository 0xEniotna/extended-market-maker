from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from market_maker.mm_config_pipeline import (
    ProposalApplyError,
    ProposalValidationError,
    ProposalValidator,
    Policy,
    ProposalManager,
    update_env_lines_preserving_format,
)
from market_maker.mmctl import main as mmctl_main


def _write_policy(config_root: Path) -> None:
    policy_dir = config_root / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    whitelist = [
        "REPRICE_MIN_INTERVAL_MS",
        "QUOTE_OFFSET_BPS",
        "QUOTE_SIZE_USD",
        "MAX_INVENTORY_USD",
        "SKEW_STRENGTH",
        "VOL_SPREAD_MULTIPLIER",
        "TOXICITY_GUARD_ENABLED",
        "TOXICITY_EXTRA_OFFSET_BPS",
    ]
    bounds = {
        "REPRICE_MIN_INTERVAL_MS": {"type": "int", "min": 100, "max": 5000},
        "QUOTE_OFFSET_BPS": {"type": "float", "min": 0.5, "max": 50.0},
        "QUOTE_SIZE_USD": {"type": "float", "min": 5.0, "max": 500.0},
        "MAX_INVENTORY_USD": {"type": "float", "min": 0.0, "max": 20000.0},
        "SKEW_STRENGTH": {"type": "float", "min": 0.0, "max": 5.0},
        "VOL_SPREAD_MULTIPLIER": {"type": "float", "min": 0.5, "max": 10.0},
        "TOXICITY_GUARD_ENABLED": {"type": "bool"},
        "TOXICITY_EXTRA_OFFSET_BPS": {"type": "float", "min": 0.0, "max": 100.0},
    }
    protected = ["TOXICITY_EXTRA_OFFSET_BPS"]
    (policy_dir / "whitelist.json").write_text(json.dumps(whitelist, indent=2) + "\n")
    (policy_dir / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
    (policy_dir / "protected_keys.json").write_text(json.dumps(protected, indent=2) + "\n")


def _write_env(path: Path) -> str:
    content = (
        "# base config\n"
        "\n"
        "export QUOTE_OFFSET_BPS=4.5 # keep-comment\n"
        "QUOTE_SIZE_USD=20\n"
        "REPRICE_MIN_INTERVAL_MS=300\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return content


def _proposal_payload(*, proposal_id: str, market: str = "MON", changes: list[dict]) -> dict:
    return {
        "proposal_id": proposal_id,
        "market": market,
        "changes": changes,
        "reason": {"hypothesis": "unit-test"},
        "canary": {"duration_minutes": 30, "success_metrics": {"pnl": ">=0"}},
        "created_at": "2026-02-26T14:30:00Z",
    }


def test_dotenv_editor_preserves_comments_and_appends_missing_key():
    original = [
        "# header\n",
        "\n",
        "export QUOTE_OFFSET_BPS=4.5 # keep\n",
        "QUOTE_SIZE_USD=20\n",
    ]
    updated = update_env_lines_preserving_format(
        original,
        {"QUOTE_OFFSET_BPS": "9.0", "MAX_INVENTORY_USD": "1000"},
    )
    assert updated[0] == "# header\n"
    assert updated[1] == "\n"
    assert updated[2] == "export QUOTE_OFFSET_BPS=9.0 # keep\n"
    assert updated[3] == "QUOTE_SIZE_USD=20\n"
    assert updated[-1] == "MAX_INVENTORY_USD=1000\n"


def test_validator_rejects_unknown_key_and_out_of_bounds_and_normalizes_bool(tmp_path: Path):
    repo_root = tmp_path / "repo"
    config_root = repo_root / "mm_config"
    _write_policy(config_root)
    env_dir = repo_root / "mm-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy(
        whitelist=set(json.loads((config_root / "policy/whitelist.json").read_text())),
        bounds=json.loads((config_root / "policy/bounds.json").read_text()),
        protected=set(json.loads((config_root / "policy/protected_keys.json").read_text())),
    )
    validator = ProposalValidator(policy=policy, repo_root=repo_root, env_dir=env_dir)

    with pytest.raises(ProposalValidationError, match="not whitelisted"):
        validator.validate(
            _proposal_payload(
                proposal_id="p1",
                changes=[{"key": "UNKNOWN_PARAM", "op": "set", "value": "1"}],
            ),
            source_path=None,
        )

    with pytest.raises(ProposalValidationError, match="below min bound"):
        validator.validate(
            _proposal_payload(
                proposal_id="p2",
                changes=[{"key": "QUOTE_OFFSET_BPS", "op": "set", "value": "0.1"}],
            ),
            source_path=None,
        )

    validated = validator.validate(
        _proposal_payload(
            proposal_id="p3",
            changes=[{"key": "TOXICITY_GUARD_ENABLED", "op": "set", "value": "1"}],
        ),
        source_path=None,
    )
    assert validated.changes[0].value == "true"

    with pytest.raises(ProposalValidationError, match="protected and cannot be changed"):
        validator.validate(
            _proposal_payload(
                proposal_id="p4",
                changes=[{"key": "TOXICITY_EXTRA_OFFSET_BPS", "op": "set", "value": "20"}],
            ),
            source_path=None,
        )


def test_validator_rejects_sensitive_key_pattern_even_if_whitelisted(tmp_path: Path):
    repo_root = tmp_path / "repo"
    env_dir = repo_root / "mm-env"
    env_dir.mkdir(parents=True, exist_ok=True)
    policy = Policy(
        whitelist={"MM_FUTURE_PRIVATE_KEY"},
        bounds={"MM_FUTURE_PRIVATE_KEY": {"type": "string"}},
        protected=set(),
    )
    validator = ProposalValidator(policy=policy, repo_root=repo_root, env_dir=env_dir)

    with pytest.raises(ProposalValidationError, match="protected and cannot be changed"):
        validator.validate(
            _proposal_payload(
                proposal_id="p-sensitive",
                changes=[{"key": "MM_FUTURE_PRIVATE_KEY", "op": "set", "value": "new"}],
            ),
            source_path=None,
        )


def test_apply_and_rollback_roundtrip(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config_root = repo_root / "mm_config"
    _write_policy(config_root)

    env_dir = repo_root / "mm-env"
    env_file = env_dir / ".env.mon"
    original = _write_env(env_file)

    proposal_path = config_root / "proposals" / "prop-1.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            _proposal_payload(
                proposal_id="prop-1",
                changes=[
                    {"key": "QUOTE_OFFSET_BPS", "op": "set", "value": "9.0"},
                    {"key": "TOXICITY_GUARD_ENABLED", "op": "set", "value": "false"},
                ],
            ),
            indent=2,
        )
        + "\n"
    )

    manager = ProposalManager(repo_root=repo_root, config_root=config_root)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MM_ENV_DIR", str(env_dir))
        manager = ProposalManager(repo_root=repo_root, config_root=config_root)
        apply_result = manager.apply_proposal("prop-1")

    assert apply_result.ok is True
    updated_text = env_file.read_text()
    assert "QUOTE_OFFSET_BPS=9" in updated_text
    assert "TOXICITY_GUARD_ENABLED=false" in updated_text
    assert not proposal_path.exists()
    assert Path(apply_result.snapshot_before).exists()
    assert Path(apply_result.snapshot_after).exists()
    assert Path(apply_result.archived_proposal).exists()

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MM_ENV_DIR", str(env_dir))
        manager = ProposalManager(repo_root=repo_root, config_root=config_root)
        rollback = manager.rollback("MON", apply_result.snapshot_before)
    assert rollback.ok is True
    assert env_file.read_text() == original


def test_lock_prevents_concurrent_apply(tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config_root = repo_root / "mm_config"
    _write_policy(config_root)
    env_dir = repo_root / "mm-env"
    env_file = env_dir / ".env.mon"
    _write_env(env_file)

    proposal_path = config_root / "proposals" / "prop-lock.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            _proposal_payload(
                proposal_id="prop-lock",
                changes=[{"key": "QUOTE_OFFSET_BPS", "op": "set", "value": "7.0"}],
            ),
            indent=2,
        )
        + "\n"
    )

    lock_path = env_file.with_name(env_file.name + ".lock")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import fcntl,sys,time;"
                "p=sys.argv[1];"
                "f=open(p,'a+');"
                "fcntl.flock(f.fileno(), fcntl.LOCK_EX);"
                "time.sleep(2)"
            ),
            str(lock_path),
        ]
    )
    try:
        time.sleep(0.2)
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("MM_ENV_DIR", str(env_dir))
            manager = ProposalManager(repo_root=repo_root, config_root=config_root)
            with pytest.raises(ProposalApplyError, match="Timeout acquiring lock"):
                manager.apply_proposal("prop-lock", lock_timeout_s=0.2)
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_mmctl_diff_proposal_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    config_root = repo_root / "mm_config"
    _write_policy(config_root)
    env_dir = repo_root / "mm-env"
    env_file = env_dir / ".env.mon"
    _write_env(env_file)

    proposal_path = config_root / "proposals" / "prop-diff.json"
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_text(
        json.dumps(
            _proposal_payload(
                proposal_id="prop-diff",
                changes=[{"key": "QUOTE_OFFSET_BPS", "op": "set", "value": "10.5"}],
            ),
            indent=2,
        )
        + "\n"
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MM_ENV_DIR", str(env_dir))
        rc = mmctl_main(
            [
                "--repo-root",
                str(repo_root),
                "--config-root",
                str(config_root),
                "diff-proposal",
                "prop-diff",
            ]
        )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Proposal prop-diff market=MON" in out
    assert "QUOTE_OFFSET_BPS" in out
