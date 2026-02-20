from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_summary_exposes_key_metrics(tmp_path: Path):
    mod = _load_module(Path("scripts/analyse_mm_journal.py"), "analyse_mm_journal_mod")

    events = [
        {
            "type": "run_start",
            "ts": 1000.0,
            "market": "LIT-USD",
            "position": "0",
            "best_bid": "100",
            "best_ask": "101",
        },
        {
            "type": "order_placed",
            "ts": 1001.0,
            "market": "LIT-USD",
        },
        {
            "type": "fill",
            "ts": 1002.0,
            "market": "LIT-USD",
            "side": "BUY",
            "price": "100",
            "qty": "1",
            "edge_bps": "1.0",
            "position": "1",
            "best_bid": "100",
            "best_ask": "101",
        },
        {
            "type": "snapshot",
            "ts": 1007.0,
            "market": "LIT-USD",
            "position": "1",
            "best_bid": "101",
            "best_ask": "102",
        },
    ]

    summary = mod.build_summary(events, tmp_path / "mm_LIT-USD.jsonl", assumed_fee_bps=None)
    metrics = summary["metrics"]
    assert summary["market"] == "LIT-USD"
    assert metrics["fill_rate_pct"] is not None
    assert metrics["avg_edge_bps"] is not None
    assert metrics["final_position"] is not None
    assert metrics["markout_5s_bps"] is not None
