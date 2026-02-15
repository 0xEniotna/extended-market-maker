from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path


def _load_analyser_module():
    path = Path("scripts/analyse_mm_journal.py")
    spec = importlib.util.spec_from_file_location("analyse_mm_journal_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analysis_uses_observed_start_position_and_single_markout_section():
    mod = _load_analyser_module()

    events = [
        {
            "ts": 1000.0,
            "type": "snapshot",
            "market": "TEST-USD",
            "position": "100",
            "best_bid": "10.00",
            "best_ask": "10.20",
            "spread_bps": "198.02",
        },
        {
            "ts": 1001.0,
            "type": "order_placed",
            "market": "TEST-USD",
            "external_id": "o1",
            "side": "BUY",
            "price": "10.00",
            "size": "1",
            "level": 0,
            "best_bid": "10.00",
            "best_ask": "10.20",
            "spread_bps": "198.02",
        },
        {
            "ts": 1002.0,
            "type": "fill",
            "market": "TEST-USD",
            "trade_id": 1,
            "order_id": 11,
            "side": "BUY",
            "price": "10.00",
            "qty": "1",
            "fee": "0",
            "is_taker": False,
            "level": 0,
            "best_bid": "10.00",
            "best_ask": "10.20",
            "mid": "10.10",
            "spread_bps": "198.02",
            "edge_bps": "99.01",
            "position": "101",
        },
        {
            "ts": 1007.0,
            "type": "snapshot",
            "market": "TEST-USD",
            "position": "101",
            "best_bid": "10.10",
            "best_ask": "10.30",
            "spread_bps": "196.08",
        },
        {
            "ts": 1008.0,
            "type": "rejection",
            "market": "TEST-USD",
            "external_id": "o2",
            "side": "BUY",
            "price": "10.20",
            "reason": "POST_ONLY_FAILED",
        },
        {
            "ts": 1009.0,
            "type": "order_cancelled",
            "market": "TEST-USD",
            "external_id": "o1",
            "side": "BUY",
            "level": 0,
            "reason": "reprice",
        },
        {
            "ts": 1020.0,
            "type": "snapshot",
            "market": "TEST-USD",
            "position": "101",
            "best_bid": "10.40",
            "best_ask": "10.60",
            "spread_bps": "190.48",
        },
    ]

    report = mod.analyse(events, Path("mm_TEST-USD.jsonl"), Decimal("0"))

    assert report.count("  Markout (bps):") == 1
    assert "Post-only rejects: 1" in report
    assert "Level toxicity (+5s markout):" in report
    assert "## Context Regime Analysis" in report
    assert "fill_snapshot_present:" in report
    assert "(start pos 100" in report
