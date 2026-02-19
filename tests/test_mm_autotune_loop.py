from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path


def _load_module():
    path = Path("scripts/mm_autotune_loop.py")
    spec = importlib.util.spec_from_file_location("mm_autotune_loop_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _touch_with_mtime(path: Path, mtime: float) -> None:
    path.write_text("{}\n")
    os.utime(path, (mtime, mtime))


def test_find_latest_journal_filters_by_market_and_since(tmp_path: Path):
    mod = _load_module()
    now = time.time()

    amzn_old = tmp_path / "mm_AMZN_24_5-USD_20260216_000000.jsonl"
    amzn_new = tmp_path / "mm_AMZN_24_5-USD_20260216_001000.jsonl"
    pump_newer = tmp_path / "mm_PUMP-USD_20260216_001500.jsonl"

    _touch_with_mtime(amzn_old, now - 120)
    _touch_with_mtime(amzn_new, now - 30)
    _touch_with_mtime(pump_newer, now - 10)

    selected = mod.find_latest_journal(
        tmp_path,
        since=now - 60,
        market_name="AMZN_24_5-USD",
    )
    assert selected == amzn_new


def test_find_latest_journal_returns_none_when_no_recent_match(tmp_path: Path):
    mod = _load_module()
    now = time.time()

    amzn_old = tmp_path / "mm_AMZN_24_5-USD_20260216_000000.jsonl"
    _touch_with_mtime(amzn_old, now - 120)

    selected = mod.find_latest_journal(
        tmp_path,
        since=now - 30,
        market_name="AMZN_24_5-USD",
    )
    assert selected is None


def test_append_config_changelog_rows_writes_old_new(tmp_path: Path):
    mod = _load_module()
    changelog = tmp_path / "config_changelog.jsonl"
    metrics = mod.Metrics(
        fills=10,
        fill_rate_pct=Decimal("2.5"),
        markout_5s_bps=Decimal("-1.2"),
        last10_realized_pnl=Decimal("-0.5"),
    )
    mod.append_config_changelog_rows(
        changelog_path=changelog,
        market_name="AMZN_24_5-USD",
        iteration=3,
        env_before=tmp_path / ".env.amzn.iter003",
        env_after=tmp_path / ".env.amzn.iter004",
        current_env_map={"MM_SPREAD_MULTIPLIER": "0.35"},
        updates={"MM_SPREAD_MULTIPLIER": "0.40"},
        reasons=["unit test"],
        stop_reason="last10_realized_pnl_negative",
        run_started=1234.5,
        metrics_payload=mod._metrics_payload(metrics),
        analysis_file=tmp_path / "journal.analysis.txt",
        tuning_log_path=tmp_path / "mm_tuning_log.jsonl",
    )
    lines = changelog.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["param"] == "MM_SPREAD_MULTIPLIER"
    assert row["old"] == "0.35"
    assert row["new"] == "0.40"
    assert row["market"] == "AMZN_24_5-USD"
    assert row["trigger"]["markout_5s_bps"] == "-1.2"
