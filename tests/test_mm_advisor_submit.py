from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("scripts/mm_advisor_submit.py")
    spec = importlib.util.spec_from_file_location("mm_advisor_submit_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_policy(repo: Path) -> None:
    policy_dir = repo / "mm_config" / "policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    whitelist = ["MM_MIN_OFFSET_BPS", "MM_SPREAD_MULTIPLIER", "MM_STARK_PRIVATE_KEY"]
    bounds = {
        "MM_MIN_OFFSET_BPS": {"type": "float", "min": 0.0, "max": 100.0},
        "MM_SPREAD_MULTIPLIER": {"type": "float", "min": 0.05, "max": 8.0},
        "MM_STARK_PRIVATE_KEY": {"type": "string"},
    }
    protected = ["MM_STARK_PRIVATE_KEY"]
    (policy_dir / "whitelist.json").write_text(json.dumps(whitelist, indent=2) + "\n")
    (policy_dir / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n")
    (policy_dir / "protected_keys.json").write_text(json.dumps(protected, indent=2) + "\n")


def _write_env(repo: Path, *, market: str = "ETH-USD") -> Path:
    env = repo / ".env.eth"
    env.write_text(
        "\n".join(
            [
                "MM_ENVIRONMENT=mainnet",
                f"MM_MARKET_NAME={market}",
                "MM_MIN_OFFSET_BPS=4.0",
                "MM_SPREAD_MULTIPLIER=1.20",
            ]
        )
        + "\n"
    )
    return env


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out


def test_single_submit_writes_canonical_row(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    _write_env(repo)

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--market",
            "ETH-USD",
            "--param",
            "MM_MIN_OFFSET_BPS",
            "--proposed",
            "3",
            "--reason",
            "tighten for unwind",
            "--json",
        ]
    )
    assert rc == 0

    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["market"] == "ETH-USD"
    assert row["param"] == "MM_MIN_OFFSET_BPS"
    assert row["old"] == "4.0"
    assert row["proposed"] == "3"
    assert row["guardrail_status"] == "passed"
    assert row["proposal_only"] is True
    assert row["source"] == "mm_analyst_submit"
    assert row["reason_note"] == "tighten for unwind"


def test_submit_rejects_protected_key(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    _write_env(repo)

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--market",
            "ETH-USD",
            "--param",
            "MM_STARK_PRIVATE_KEY",
            "--proposed",
            "abc",
            "--json",
        ]
    )
    assert rc == 1
    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert rows == []


def test_import_jsonl_converts_michel_style_to_live_old_value(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    _write_env(repo)

    import_path = repo / "michel.jsonl"
    import_path.write_text(
        json.dumps(
            {
                "proposal_id": "ETH_USD-michel-20260226-2-MM_MIN_OFFSET_BPS",
                "market": "ETH-USD",
                "env_file": ".env.eth",
                "param": "MM_MIN_OFFSET_BPS",
                "current": "999",
                "proposed": "3",
                "op": "set",
                "reason": "manual analyst idea",
                "guardrail_status": "passed",
                "deadman": False,
                "escalation_target": "human",
            }
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--input-jsonl", str(import_path), "--json"])
    assert rc == 0
    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["proposal_id"] == "ETH_USD-michel-20260226-2-MM_MIN_OFFSET_BPS"
    # Must come from live env value, not pasted "current" field.
    assert row["old"] == "4.0"
    assert row["proposed"] == "3"
    assert row["env_path"].endswith(".env.eth")


def test_submit_resolves_env_by_filename_when_market_name_missing(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    (repo / ".env.eth").write_text("MM_MIN_OFFSET_BPS=4\n")

    rc = mod.main(
        [
            "--repo",
            str(repo),
            "--market",
            "ETH-USD",
            "--param",
            "MM_MIN_OFFSET_BPS",
            "--proposed",
            "3",
            "--json",
        ]
    )
    assert rc == 0
    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert len(rows) == 1
    assert rows[0]["env_path"].endswith(".env.eth")


def test_import_rejects_malformed_jsonl_without_partial_writes(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    _write_env(repo)

    import_path = repo / "mixed.jsonl"
    import_path.write_text(
        "\n".join(
            [
                '{"proposal_id":"good-1","market":"ETH-USD","env_file":".env.eth","param":"MM_MIN_OFFSET_BPS","proposed":"3"}',
                '{"proposal_id":"bad-1","market":"ETH-USD"',  # malformed
            ]
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--input-jsonl", str(import_path), "--json"])
    assert rc == 1
    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert rows == []


def test_import_rejects_invalid_row_without_partial_writes(tmp_path: Path):
    mod = _load_module()
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_policy(repo)
    _write_env(repo)

    import_path = repo / "mixed_invalid.jsonl"
    import_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "proposal_id": "good-1",
                        "market": "ETH-USD",
                        "env_file": ".env.eth",
                        "param": "MM_MIN_OFFSET_BPS",
                        "proposed": "3",
                    }
                ),
                json.dumps(
                    {
                        "proposal_id": "bad-1",
                        "market": "ETH-USD",
                        "env_file": ".env.eth",
                        "param": "MM_MIN_OFFSET_BPS",
                        "proposed": "-1",  # below min bound
                    }
                ),
            ]
        )
        + "\n"
    )

    rc = mod.main(["--repo", str(repo), "--input-jsonl", str(import_path), "--json"])
    assert rc == 1
    rows = _read_jsonl(repo / "data/mm_audit/advisor/proposals.jsonl")
    assert rows == []
