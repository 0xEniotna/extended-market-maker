from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_evidence_detects_missing_components(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/auditor_followup.py"), "auditor_followup_mod")

    action = {
        "market": "AMZN_24_5-USD",
        "expected_evidence": {
            "env_file_exists": str(tmp_path / ".env.amzn"),
            "cron_market": "AMZN_24_5-USD",
        },
    }

    ok, blockers, checks = mod._evaluate_evidence(
        action=action,
        active_markets=set(),
        journal_dir=tmp_path,
        approved_at_ts=1000.0,
    )

    assert ok is False
    assert checks["env_file_exists"] is False
    assert checks["cron_market"] is False
    assert any("missing_env" in b for b in blockers)
    assert any("missing_cron_market" in b for b in blockers)


def test_journal_started_after_detects_run_start(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/auditor_followup.py"), "auditor_followup_mod2")

    journal = tmp_path / "mm_LIT-USD_20260219_120000.jsonl"
    journal.write_text(
        "\n".join(
            [
                json.dumps({"type": "snapshot", "ts": 1000.0}),
                json.dumps({"type": "run_start", "ts": 1010.0}),
            ]
        )
        + "\n"
    )

    assert mod._journal_started_after(journal, 1005.0) is True
    assert mod._journal_started_after(journal, 1020.0) is False
