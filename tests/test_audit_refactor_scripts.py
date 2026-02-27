from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_audit_reprice_quality_flags_side_bias_and_low_uptime(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/audit_reprice_quality.py"), "audit_reprice_quality_mod")

    now = time.time()
    journal_dir = tmp_path / "journals"
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.eth"
    env_file.write_text("MM_IMBALANCE_PAUSE_THRESHOLD=0.78\n")
    do_not_restart = tmp_path / "do_not_restart.txt"
    do_not_restart.write_text("")

    rows = []
    ts = now - 60
    for _ in range(4):
        rows.append({"ts": ts, "type": "reprice_decision", "side": "BUY", "reason": "skip_imbalance_pause"})
        ts += 0.02
    rows.append({"ts": ts, "type": "reprice_decision", "side": "SELL", "reason": "skip_imbalance_pause"})
    ts += 0.02
    rows.append({"ts": ts, "type": "reprice_decision", "side": "SELL", "reason": "replace_target_shift"})
    ts += 0.02
    rows.append({"ts": ts, "type": "reprice_decision", "side": "BUY", "reason": "hold_within_tolerance"})
    _write_jsonl(journal_dir / "mm_ETH-USD_latest.jsonl", rows)

    args = argparse.Namespace(
        journal_dir=str(journal_dir),
        output_dir=str(output_dir),
        lookback_hours=1.0,
        do_not_restart_file=str(do_not_restart),
        env_map=f"ETH-USD={env_file}",
        char_limit=1500,
    )
    rc = mod.run(args)
    assert rc == 0

    payload = json.loads((output_dir / "reprice_quality.json").read_text())
    market = payload["markets"]["ETH-USD"]
    assert "side_bias" in market["flags"]
    assert "low_uptime" in market["flags"]
    assert any("MM_IMBALANCE_PAUSE_THRESHOLD" in rec for rec in market["recommendations"])


def test_audit_position_risk_flags_accumulation_fill_bias_and_exposure(tmp_path: Path):
    mod = _load_module(Path("scripts/tools/audit_position_risk.py"), "audit_position_risk_mod")

    now = time.time()
    journal_dir = tmp_path / "journals"
    output_dir = tmp_path / "out"
    env_file = tmp_path / ".env.mon"
    env_file.write_text("MM_MAX_POSITION_SIZE=100\nMM_INVENTORY_SKEW_FACTOR=0.25\n")
    do_not_restart = tmp_path / "do_not_restart.txt"
    do_not_restart.write_text("")

    rows = []
    ts = now - 600
    position = -20.0
    for _ in range(6):
        rows.append({"ts": ts, "type": "position_tick", "position": position, "mid": 100.0 + (ts - (now - 600)) / 120.0})
        ts += 60
        position -= 15.0

    fill_ts = now - 540
    for idx in range(10):
        side = "SELL" if idx < 8 else "BUY"
        rows.append({
            "ts": fill_ts + idx * 12,
            "type": "fill",
            "side": side,
            "qty": "1.0",
            "price": str(100.0 + idx),
            "position": str(-30 - idx * 5),
        })

    _write_jsonl(journal_dir / "mm_MON-USD_latest.jsonl", rows)

    args = argparse.Namespace(
        journal_dir=str(journal_dir),
        output_dir=str(output_dir),
        lookback_hours=2.0,
        do_not_restart_file=str(do_not_restart),
        env_map=f"MON-USD={env_file}",
        char_limit=1500,
    )
    rc = mod.run(args)
    assert rc == 0

    payload = json.loads((output_dir / "position_risk.json").read_text())
    market = payload["markets"]["MON-USD"]
    assert "accumulating_position" in market["flags"]
    assert "one_sided_fills" in market["flags"]
    assert "high_exposure" in market["flags"]


def test_audit_daily_scorecard_writes_history_and_verdicts(tmp_path: Path, monkeypatch):
    mod = _load_module(Path("scripts/tools/audit_daily_scorecard.py"), "audit_daily_scorecard_mod")

    now = time.time()
    journal_dir = tmp_path / "journals"
    output_dir = tmp_path / "out"
    do_not_restart = tmp_path / "do_not_restart.txt"
    do_not_restart.write_text("")

    env_eth = tmp_path / ".env.eth"
    env_eth.write_text("MM_MAX_POSITION_SIZE=100\n")
    env_mon = tmp_path / ".env.mon"
    env_mon.write_text("MM_MAX_POSITION_SIZE=100\n")

    # ETH healthy
    eth_rows = [
        {"ts": now - 500, "type": "reprice_decision", "side": "BUY", "reason": "replace_target_shift", "position": 5, "mid": 100},
        {"ts": now - 480, "type": "reprice_decision", "side": "SELL", "reason": "replace_max_age", "position": -3, "mid": 100.2},
        {"ts": now - 460, "type": "fill", "side": "BUY", "price": "100", "qty": "1", "edge_bps": "1.2", "position": "4"},
        {"ts": now - 390, "type": "book", "mid": "101.0", "position": "4"},
        {"ts": now - 350, "type": "fill", "side": "SELL", "price": "102", "qty": "1", "edge_bps": "1.1", "position": "-2"},
        {"ts": now - 270, "type": "book", "mid": "101.0", "position": "-2"},
    ]
    _write_jsonl(journal_dir / "mm_ETH-USD_latest.jsonl", eth_rows)

    # MON problematic
    mon_rows = [
        {"ts": now - 500, "type": "reprice_decision", "side": "BUY", "reason": "skip_stale", "position": 90, "mid": 100},
        {"ts": now - 470, "type": "reprice_decision", "side": "SELL", "reason": "skip_stale", "position": 92, "mid": 100.4},
        {"ts": now - 440, "type": "fill", "side": "BUY", "price": "100", "qty": "1", "edge_bps": "-1.0", "position": "93"},
        {"ts": now - 360, "type": "book", "mid": "99.0", "position": "93"},
    ]
    _write_jsonl(journal_dir / "mm_MON-USD_latest.jsonl", mon_rows)

    monkeypatch.setattr(
        mod,
        "_fetch_total_pnl_snapshot",
        lambda now_ts, lookback_hours: {
            "source": "fetch_total_pnl",
            "fleet_pnl": 10.0,
            "equity": 1010.0,
            "delta_pct": 1.0,
            "per_market_pnl": {"ETH-USD": 50.0, "MON-USD": -40.0},
        },
    )

    args = argparse.Namespace(
        journal_dir=str(journal_dir),
        output_dir=str(output_dir),
        lookback_hours=24.0,
        do_not_restart_file=str(do_not_restart),
        env_map=f"ETH-USD={env_eth},MON-USD={env_mon}",
        char_limit=3000,
    )
    rc = mod.run(args)
    assert rc == 0

    payload = json.loads((output_dir / "daily_scorecard.json").read_text())
    assert payload["fleet"]["source"] == "fetch_total_pnl"
    assert payload["markets"]["MON-USD"]["verdict"] == "🔴"
    assert payload["markets"]["ETH-USD"]["verdict"] in {"🟢", "⚠️"}

    history_lines = (output_dir / "scorecard_history.jsonl").read_text().strip().splitlines()
    assert len(history_lines) == 1
