from __future__ import annotations

import json
from decimal import Decimal

from market_maker.trade_journal import TradeJournal


def _read_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_journal_v2_records_run_metadata_and_monotonic_seq(tmp_path):
    journal = TradeJournal(
        "TEST-USD",
        journal_dir=tmp_path,
        run_id="run-abc",
        schema_version=2,
    )
    journal.record_run_start(
        environment="mainnet",
        config={"market_name": "TEST-USD", "api_key": "***redacted***"},
        market_static={"tick_size": Decimal("0.1")},
        provenance={"env_file": ".env.test"},
    )
    journal.record_order_cancelled(
        external_id="ext-1",
        side="BUY",
        level=0,
        reason="reprice",
    )
    journal.record_run_end(reason="shutdown")
    journal.close()

    records = _read_records(journal.path)
    assert records[0]["type"] == "run_start"
    assert records[-1]["type"] == "run_end"
    assert [r["seq"] for r in records] == [1, 2, 3]
    assert all(r["run_id"] == "run-abc" for r in records)
    assert all(r["schema_version"] == 2 for r in records)


def test_fill_record_keeps_market_snapshot_payload(tmp_path):
    journal = TradeJournal("TEST-USD", journal_dir=tmp_path, run_id="run-snap")
    market_snapshot = {
        "best_bid": Decimal("1.0"),
        "best_ask": Decimal("1.1"),
        "spread_bps": Decimal("952.38"),
        "bids_top": [{"price": Decimal("1.0"), "size": Decimal("100")}],
        "asks_top": [{"price": Decimal("1.1"), "size": Decimal("90")}],
        "micro_vol_bps": Decimal("3.2"),
        "micro_drift_bps": Decimal("-0.4"),
        "imbalance": Decimal("0.1"),
        "is_stale": False,
        "seconds_since_update": 0.25,
        "depth": 5,
    }
    journal.record_fill(
        trade_id=1,
        order_id=10,
        side="BUY",
        price=Decimal("1.0"),
        qty=Decimal("10"),
        fee=Decimal("0"),
        is_taker=False,
        level=0,
        best_bid=Decimal("1.0"),
        best_ask=Decimal("1.1"),
        position=Decimal("10"),
        market_snapshot=market_snapshot,
    )
    journal.close()

    records = _read_records(journal.path)
    assert len(records) == 1
    assert records[0]["type"] == "fill"
    assert records[0]["market_snapshot"]["depth"] == 5
    assert len(records[0]["market_snapshot"]["bids_top"]) == 1
