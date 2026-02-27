from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_module():
    path = Path("scripts/mm_advisor_apply.py")
    spec = importlib.util.spec_from_file_location("mm_advisor_apply_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_env(path: Path, *, market: str, spread: str = "1.0", min_offset: str = "4") -> None:
    path.write_text(
        "\n".join(
            [
                "MM_ENVIRONMENT=mainnet",
                f"MM_MARKET_NAME={market}",
                f"MM_SPREAD_MULTIPLIER={spread}",
                f"MM_MIN_OFFSET_BPS={min_offset}",
            ]
        )
        + "\n"
    )


def _proposal_row(
    *,
    proposal_id: str,
    ts: float,
    market: str,
    env_path: Path,
    param: str,
    old: str,
    proposed: str,
    deadman: bool = False,
    confidence: str = "medium",
    escalation_target: str = "human",
    guardrail_status: str = "passed",
    rejected: bool = False,
) -> dict:
    return {
        "proposal_id": proposal_id,
        "ts": ts,
        "created_at": "2026-02-26T00:00:00Z",
        "market": market,
        "env_path": str(env_path),
        "iteration": 1,
        "param": param,
        "old": old,
        "proposed": proposed,
        "new": proposed,
        "baseline_value": old,
        "reason_codes": ["unit_test"],
        "confidence": confidence,
        "guardrail_status": guardrail_status,
        "guardrail_reason": "ok" if guardrail_status == "passed" else "rejected",
        "proposal_only": True,
        "applied": False,
        "rejected": rejected,
        "deadman": deadman,
        "escalation_target": escalation_target,
        "cooldown_until_ts": None,
        "source": "mm_advisor_loop",
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_human_apply_updates_env_and_writes_receipt(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.eth"
    _write_env(env, market="ETH-USD", spread="1.0")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposal = _proposal_row(
        proposal_id="eth-1",
        ts=1000.0,
        market="ETH-USD",
        env_path=env,
        param="MM_SPREAD_MULTIPLIER",
        old="1.0",
        proposed="1.2",
    )
    proposals.write_text(json.dumps(proposal) + "\n")

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--proposal-id",
            "eth-1",
            "--approve",
            "--json",
        ]
    )
    assert rc == 0
    assert "MM_SPREAD_MULTIPLIER=1.2" in env.read_text()

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["proposal_id"] == "eth-1"
    assert receipts[0]["applied_by"] == "human"
    assert receipts[0]["result"] == "applied"

    changelog = _read_jsonl(repo / "data/mm_audit/config_changelog.jsonl")
    assert len(changelog) == 1
    assert changelog[0]["proposal_id"] == "eth-1"
    assert changelog[0]["proposal_only"] is False
    assert changelog[0]["applied"] is True
    assert changelog[0]["apply_result"] == "applied"


def test_missing_approval_skips_and_preserves_env(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.sol"
    _write_env(env, market="SOL-USD", spread="1.0")
    before = env.read_text()

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="sol-1",
                ts=1000.0,
                market="SOL-USD",
                env_path=env,
                param="MM_SPREAD_MULTIPLIER",
                old="1.0",
                proposed="1.1",
            )
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--proposal-id", "sol-1", "--json"])
    assert rc == 1
    assert env.read_text() == before

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "skipped"
    assert receipts[0]["failure_reason"] == "approval_required"


def test_warren_auto_mode_applies_deadman_only(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    env_deadman = repo / ".env.eth"
    env_human = repo / ".env.aapl"
    _write_env(env_deadman, market="ETH-USD", spread="1.4", min_offset="4")
    _write_env(env_human, market="AAPL_24_5-USD", spread="1.1", min_offset="6")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _proposal_row(
            proposal_id="eth-deadman-1",
            ts=1000.0,
            market="ETH-USD",
            env_path=env_deadman,
            param="MM_SPREAD_MULTIPLIER",
            old="1.4",
            proposed="1.0",
            deadman=True,
            confidence="high",
            escalation_target="warren",
        ),
        _proposal_row(
            proposal_id="aapl-human-1",
            ts=1001.0,
            market="AAPL_24_5-USD",
            env_path=env_human,
            param="MM_MIN_OFFSET_BPS",
            old="6",
            proposed="5.5",
            deadman=False,
            confidence="high",
            escalation_target="human",
        ),
    ]
    proposals.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    rc = mod.main(["--repo", str(repo), "--mode", "warren-auto", "--approve", "--json"])
    assert rc == 0

    assert "MM_SPREAD_MULTIPLIER=1.0" in env_deadman.read_text()
    assert "MM_MIN_OFFSET_BPS=6" in env_human.read_text()

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["proposal_id"] == "eth-deadman-1"
    assert receipts[0]["applied_by"] == "warren_auto"
    assert receipts[0]["result"] == "applied"


def test_current_value_mismatch_fails_without_override(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.xng"
    _write_env(env, market="XNG-USD", spread="1.3")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="xng-1",
                ts=1000.0,
                market="XNG-USD",
                env_path=env,
                param="MM_SPREAD_MULTIPLIER",
                old="1.1",
                proposed="1.0",
                deadman=True,
                confidence="high",
                escalation_target="warren",
            )
        )
        + "\n"
    )

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--proposal-id",
            "xng-1",
            "--approve",
            "--json",
        ]
    )
    assert rc == 1
    assert "MM_SPREAD_MULTIPLIER=1.3" in env.read_text()

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "failed"
    assert receipts[0]["failure_reason"] == "current_value_mismatch"


def test_numeric_old_value_equivalence_does_not_fail(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.xpt"
    _write_env(env, market="XPT-USD", min_offset="2.0")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="xpt-1",
                ts=1000.0,
                market="XPT-USD",
                env_path=env,
                param="MM_MIN_OFFSET_BPS",
                old="2",
                proposed="2.5",
            )
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--proposal-id", "xpt-1", "--approve", "--json"])
    assert rc == 0
    assert "MM_MIN_OFFSET_BPS=2.5" in env.read_text()

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "applied"


def test_stale_old_but_already_at_target_is_skipped(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.xpt"
    _write_env(env, market="XPT-USD", min_offset="2.5")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="xpt-stale-1",
                ts=1000.0,
                market="XPT-USD",
                env_path=env,
                param="MM_MIN_OFFSET_BPS",
                old="2",
                proposed="2.5",
            )
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--proposal-id", "xpt-stale-1", "--approve", "--json"])
    assert rc == 1
    assert "MM_MIN_OFFSET_BPS=2.5" in env.read_text()

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "skipped"
    assert receipts[0]["failure_reason"] == "already_at_proposed"


def test_unknown_proposal_id_returns_nonzero(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text("")
    rc = mod.main(["--repo", str(repo), "--proposal-id", "missing-id", "--approve", "--json"])
    assert rc == 1


def test_unknown_proposal_id_reports_not_found_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text("")

    rc = mod.main(["--repo", str(repo), "--proposal-id", "missing-id", "--approve", "--json"])
    assert rc == 1

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["reason"] == "proposal_id_not_found"
    assert payload["proposal_id"] == "missing-id"
    assert payload["proposals_path"].endswith("data/mm_audit/advisor/proposals.jsonl")


def test_list_pending_only_shows_pending(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    env_eth = repo / ".env.eth"
    env_near = repo / ".env.near"
    _write_env(env_eth, market="ETH-USD")
    _write_env(env_near, market="NEAR-USD")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        "\n".join(
            [
                json.dumps(
                    _proposal_row(
                        proposal_id="eth-1",
                        ts=1000.0,
                        market="ETH-USD",
                        env_path=env_eth,
                        param="MM_SPREAD_MULTIPLIER",
                        old="1.0",
                        proposed="1.1",
                    )
                ),
                json.dumps(
                    _proposal_row(
                        proposal_id="near-1",
                        ts=1001.0,
                        market="NEAR-USD",
                        env_path=env_near,
                        param="MM_MIN_OFFSET_BPS",
                        old="4",
                        proposed="5",
                    )
                ),
            ]
        )
        + "\n"
    )

    receipts = repo / "data/mm_audit/advisor/apply_receipts.jsonl"
    receipts.parent.mkdir(parents=True, exist_ok=True)
    receipts.write_text(
        json.dumps(
            {
                "proposal_id": "eth-1",
                "applied_by": "human",
                "applied_ts": 1002.0,
                "result": "applied",
                "env_before_hash": "a",
                "env_after_hash": "b",
            }
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--list-pending", "--json"])
    assert rc == 0

    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["pending"] == 1
    assert payload["results"][0]["proposal_id"] == "near-1"


def test_proposal_id_reports_not_pending_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.eth"
    _write_env(env, market="ETH-USD")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="eth-1",
                ts=1000.0,
                market="ETH-USD",
                env_path=env,
                param="MM_SPREAD_MULTIPLIER",
                old="1.0",
                proposed="1.1",
            )
        )
        + "\n"
    )
    receipts = repo / "data/mm_audit/advisor/apply_receipts.jsonl"
    receipts.parent.mkdir(parents=True, exist_ok=True)
    receipts.write_text(
        json.dumps(
            {
                "proposal_id": "eth-1",
                "applied_by": "human",
                "applied_ts": 1002.0,
                "result": "applied",
                "env_before_hash": "a",
                "env_after_hash": "b",
            }
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--proposal-id", "eth-1", "--approve", "--json"])
    assert rc == 1
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["reason"] == "proposal_not_pending"


def test_protected_key_is_rejected(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.eth"
    _write_env(env, market="ETH-USD", spread="1.0")

    policy_dir = repo / "mm_config" / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "protected_keys.json").write_text(
        json.dumps(["MM_STARK_PRIVATE_KEY"], indent=2) + "\n"
    )

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="eth-protected-1",
                ts=1000.0,
                market="ETH-USD",
                env_path=env,
                param="MM_STARK_PRIVATE_KEY",
                old="old",
                proposed="new",
            )
        )
        + "\n"
    )

    rc = mod.main(
        ["--repo", str(repo), "--proposal-id", "eth-protected-1", "--approve", "--json"]
    )
    assert rc == 1

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "failed"
    assert receipts[0]["failure_reason"] == "protected_key"


def test_sensitive_key_pattern_is_rejected_without_policy_file(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    env = repo / ".env.eth"
    _write_env(env, market="ETH-USD", spread="1.0")

    proposals = repo / "data/mm_audit/advisor/proposals.jsonl"
    proposals.parent.mkdir(parents=True, exist_ok=True)
    proposals.write_text(
        json.dumps(
            _proposal_row(
                proposal_id="eth-sensitive-1",
                ts=1000.0,
                market="ETH-USD",
                env_path=env,
                param="MM_FUTURE_PRIVATE_KEY",
                old="old",
                proposed="new",
            )
        )
        + "\n"
    )

    rc = mod.main(
        ["--repo", str(repo), "--proposal-id", "eth-sensitive-1", "--approve", "--json"]
    )
    assert rc == 1

    receipts = _read_jsonl(repo / "data/mm_audit/advisor/apply_receipts.jsonl")
    assert len(receipts) == 1
    assert receipts[0]["result"] == "failed"
    assert receipts[0]["failure_reason"] == "protected_key"
