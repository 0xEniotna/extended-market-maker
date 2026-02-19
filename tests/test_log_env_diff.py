from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("scripts/tools/log_env_diff.py")
    spec = importlib.util.spec_from_file_location("log_env_diff_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_rows_only_mm_keys_and_old_new_values(tmp_path: Path):
    mod = _load_module()

    before = {
        "MM_MARKET_NAME": "AMZN_24_5-USD",
        "MM_SPREAD_MULTIPLIER": "0.35",
        "MM_MIN_OFFSET_BPS": "5.0",
        "API_KEY": "secret-old",
    }
    after = {
        "MM_MARKET_NAME": "AMZN_24_5-USD",
        "MM_SPREAD_MULTIPLIER": "0.40",
        "MM_MIN_OFFSET_BPS": "5.0",
        "API_KEY": "secret-new",
    }
    rows = mod.build_rows(
        before_env=before,
        after_env=after,
        env_before_path=tmp_path / ".env.before",
        env_after_path=tmp_path / ".env.after",
        market="AMZN_24_5-USD",
        agent="manual",
        source="unit_test",
        mm_only=True,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["param"] == "MM_SPREAD_MULTIPLIER"
    assert row["old"] == "0.35"
    assert row["new"] == "0.40"
    assert row["agent"] == "manual"
    assert row["source"] == "unit_test"


def test_parse_env_handles_inline_comments():
    mod = _load_module()
    lines = [
        "# comment",
        "MM_SPREAD_MULTIPLIER=0.35 # keep conservative",
        "MM_MIN_OFFSET_BPS=5.0",
        "BROKEN",
    ]
    parsed = mod.parse_env(lines)
    assert parsed["MM_SPREAD_MULTIPLIER"] == "0.35"
    assert parsed["MM_MIN_OFFSET_BPS"] == "5.0"
    assert "BROKEN" not in parsed
