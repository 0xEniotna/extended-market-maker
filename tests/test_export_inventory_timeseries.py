from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path


def _load_module():
    path = Path("scripts/tools/export_inventory_timeseries.py")
    spec = importlib.util.spec_from_file_location("export_inventory_timeseries_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_rows_infers_limits_and_positions(tmp_path: Path):
    mod = _load_module()
    journal = tmp_path / "mm_AMZN_24_5-USD_20260219_120000.jsonl"
    rows = [
        {
            "ts": 1000.0,
            "type": "run_start",
            "market": "AMZN_24_5-USD",
            "config": {
                "max_position_size": "100",
                "inventory_warn_pct": "0.5",
                "inventory_critical_pct": "0.8",
                "inventory_hard_pct": "0.95",
            },
        },
        {
            "ts": 1001.0,
            "type": "snapshot",
            "market": "AMZN_24_5-USD",
            "position": "10",
            "best_bid": "100",
            "best_ask": "101",
            "spread_bps": "99.5",
            "circuit_open": False,
        },
        {
            "ts": 1005.0,
            "type": "fill",
            "market": "AMZN_24_5-USD",
            "position": "20",
        },
    ]
    journal.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    pos_rows, inferred = mod.load_position_rows(
        target=tmp_path,
        market="AMZN_24_5-USD",
        start_ts=None,
        end_ts=None,
    )
    assert len(pos_rows) == 2
    assert inferred["max_position_size"] == Decimal("100")
    assert inferred["inventory_warn_pct"] == Decimal("0.5")
    assert pos_rows[0]["position"] == Decimal("10")
    assert pos_rows[1]["position"] == Decimal("20")


def test_summary_and_bucket_metrics():
    mod = _load_module()
    rows = [
        {"ts": 0.0, "position": Decimal("60"), "abs_position": Decimal("60"), "utilization_pct": Decimal("60"), "market": "AMZN_24_5-USD", "source_event": "snapshot"},
        {"ts": 10.0, "position": Decimal("60"), "abs_position": Decimal("60"), "utilization_pct": Decimal("60"), "market": "AMZN_24_5-USD", "source_event": "snapshot"},
        {"ts": 20.0, "position": Decimal("20"), "abs_position": Decimal("20"), "utilization_pct": Decimal("20"), "market": "AMZN_24_5-USD", "source_event": "fill"},
        {"ts": 30.0, "position": Decimal("0"), "abs_position": Decimal("0"), "utilization_pct": Decimal("0"), "market": "AMZN_24_5-USD", "source_event": "snapshot"},
    ]
    buckets = mod.bucket_rows(rows, 20)
    assert len(buckets) == 2
    assert buckets[0]["position_open"] == Decimal("60")
    assert buckets[0]["position_close"] == Decimal("60")
    assert buckets[1]["position_close"] == Decimal("0")

    summary = mod.summarize_rows(
        rows,
        max_position_size=Decimal("100"),
        warn_pct=Decimal("0.5"),
        critical_pct=Decimal("0.8"),
        hard_pct=Decimal("0.95"),
    )
    assert summary["max_abs_position"] == Decimal("60")
    assert summary["time_above_warn_s"] == 20.0
    assert summary["time_above_critical_s"] == 0.0
    assert summary["longest_drift_s"] == 30.0


def test_load_rows_skips_malformed_json_lines(tmp_path: Path):
    mod = _load_module()
    journal = tmp_path / "mm_AMZN_24_5-USD_20260219_130000.jsonl"
    journal.write_text(
        "\n".join([
            "not-json",
            json.dumps({"ts": 1000.0, "type": "snapshot", "market": "AMZN_24_5-USD", "position": "1"}),
        ]) + "\n"
    )
    pos_rows, _ = mod.load_position_rows(
        target=tmp_path,
        market="AMZN_24_5-USD",
        start_ts=None,
        end_ts=None,
    )
    assert len(pos_rows) == 1
    assert pos_rows[0]["position"] == Decimal("1")
