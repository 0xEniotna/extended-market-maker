from __future__ import annotations

import importlib.util
import os
import sys
import time
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
