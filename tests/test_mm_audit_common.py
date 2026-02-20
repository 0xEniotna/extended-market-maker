from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_market_jobs_uses_env_for_market(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/mm_audit_common.py"), "mm_audit_common_mod")

    env_path = tmp_path / ".env.amzn"
    env_path.write_text("MM_MARKET_NAME=AMZN_24_5-USD\n")

    jobs = {
        "jobs": [
            {
                "id": "job-1",
                "name": "mm-amzn",
                "enabled": True,
                "env": str(env_path),
            }
        ]
    }
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(jobs) + "\n")

    rows = mod.load_market_jobs(jobs_path, tmp_path)
    assert len(rows) == 1
    assert rows[0]["market"] == "AMZN_24_5-USD"
    assert rows[0]["env_path"] == str(env_path)


def test_load_policy_json_compatible_yaml(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/mm_audit_common.py"), "mm_audit_common_mod2")

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text('{"launch": {"max_new_per_day": 4}}\n')
    payload = mod.load_policy(policy_path)
    assert payload["launch"]["max_new_per_day"] == 4


def test_discover_recent_markets_from_journals(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/mm_audit_common.py"), "mm_audit_common_mod3")

    (tmp_path / "mm_AMZN_24_5-USD_20260219_120000.jsonl").write_text("{}\n")
    (tmp_path / "mm_ETH-USD_20260219_120000.jsonl").write_text("{}\n")
    (tmp_path / "mm_tuning_log_AMZN_foo.jsonl").write_text("{}\n")

    markets = mod.discover_recent_markets_from_journals(tmp_path, lookback_s=3600 * 48)
    assert markets == ["AMZN_24_5-USD", "ETH-USD"]
