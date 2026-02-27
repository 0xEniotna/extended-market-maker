from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path


def _load_module():
    path = Path("scripts/mm_advisor_loop.py")
    spec = importlib.util.spec_from_file_location("mm_advisor_loop_mod", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_env(path: Path, *, market: str, spread_multiplier: str = "1.2") -> None:
    path.write_text(
        "\n".join(
            [
                "MM_ENVIRONMENT=mainnet",
                f"MM_MARKET_NAME={market}",
                f"MM_SPREAD_MULTIPLIER={spread_multiplier}",
                "MM_MIN_OFFSET_BPS=4",
                "MM_MAX_OFFSET_BPS=80",
                "MM_ORDER_SIZE_MULTIPLIER=1.0",
                "MM_INVENTORY_SKEW_FACTOR=0.35",
                "MM_MIN_REPRICE_INTERVAL_S=0.6",
                "MM_MIN_REPRICE_MOVE_TICKS=2",
                "MM_MIN_REPRICE_EDGE_DELTA_BPS=0.5",
                "MM_REPRICE_TOLERANCE_PERCENT=0.2",
            ]
        )
        + "\n"
    )


def _write_journal(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _baseline_payload(
    *,
    market: str,
    env_path: Path,
    env_hash: str,
    captured_ts: float,
    baseline_params: dict[str, str],
    refresh_reason: str = "initial",
) -> dict:
    return {
        "market": market,
        "env_path": str(env_path),
        "captured_at": "2026-02-22T00:00:00Z",
        "captured_ts": captured_ts,
        "env_hash": env_hash,
        "env_mtime": 0.0,
        "refresh_reason": refresh_reason,
        "baseline_params": baseline_params,
    }


def _process_market_common(tmp_path: Path, mod, *, market: str, now: float):
    baselines_dir = tmp_path / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    proposals_path = tmp_path / "proposals.jsonl"
    changelog_path = tmp_path / "changelog.jsonl"
    apply_receipts_path = tmp_path / "apply_receipts.jsonl"
    baseline_refresh_log_path = tmp_path / "baseline_refresh_log.jsonl"
    state_row = {
        "iteration": 0,
        "proposal_timestamps": [],
        "deadman_cooldown_until_ts": 0.0,
    }
    return {
        "baselines_dir": baselines_dir,
        "proposals_path": proposals_path,
        "changelog_path": changelog_path,
        "apply_receipts_path": apply_receipts_path,
        "baseline_refresh_log_path": baseline_refresh_log_path,
        "state_row": state_row,
        "now": now,
    }


def test_cleanup_integrity_no_autotune_reference():
    assert "mm_autotune_loop.py" not in Path("README.md").read_text()
    assert "mm_autotune_loop.py" not in Path("scripts/mm_openclaw_controller.sh").read_text()


def test_deadman_proposal_routes_to_warren(tmp_path: Path):
    mod = _load_module()
    now = 10_000.0
    market = "ETH-USD"
    env_path = tmp_path / ".env.eth"
    _write_env(env_path, market=market, spread_multiplier="1.2")

    events = [
        {
            "ts": now - 3590,
            "type": "run_start",
            "market": market,
            "config": {
                "max_position_size": "100",
                "inventory_hard_pct": "0.95",
            },
        },
        {"ts": now - 1800, "type": "reprice_decision", "market": market, "reason": "allow"},
        {"ts": now - 1200, "type": "reprice_decision", "market": market, "reason": "allow"},
        {"ts": now - 600, "type": "reprice_decision", "market": market, "reason": "allow"},
    ]
    journal_path = tmp_path / "mm_ETH-USD_20260222_000000.jsonl"
    _write_journal(journal_path, events)

    ctx = _process_market_common(tmp_path, mod, market=market, now=now)
    baseline_path = ctx["baselines_dir"] / f"{mod.slugify(market)}.json"
    env_hash = mod._env_file_hash(env_path)
    baseline = _baseline_payload(
        market=market,
        env_path=env_path,
        env_hash=env_hash,
        captured_ts=now - 5000,
        baseline_params={
            "MM_SPREAD_MULTIPLIER": "1.0",
            "MM_MIN_OFFSET_BPS": "4",
            "MM_MAX_OFFSET_BPS": "80",
            "MM_ORDER_SIZE_MULTIPLIER": "1.0",
            "MM_INVENTORY_SKEW_FACTOR": "0.35",
            "MM_MIN_REPRICE_INTERVAL_S": "0.6",
            "MM_MIN_REPRICE_MOVE_TICKS": "2",
            "MM_MIN_REPRICE_EDGE_DELTA_BPS": "0.5",
            "MM_REPRICE_TOLERANCE_PERCENT": "0.2",
        },
    )
    baseline_path.write_text(json.dumps(baseline) + "\n")

    result = mod._process_market(
        market=market,
        env_path=env_path,
        journal_path=journal_path,
        baselines_dir=ctx["baselines_dir"],
        proposals_path=ctx["proposals_path"],
        changelog_path=ctx["changelog_path"],
        apply_receipts_path=ctx["apply_receipts_path"],
        baseline_refresh_log_path=ctx["baseline_refresh_log_path"],
        state_row=ctx["state_row"],
        now_value=now,
        deadman_window_s=3600,
        deadman_cooldown_s=1800,
        inventory_window_s=7200,
        max_changes_per_hour=10,
    )
    rows = result["rows"]
    assert any(row["deadman"] is True for row in rows)
    assert any(row["escalation_target"] == "warren" for row in rows)
    assert any(row["confidence"] == "high" for row in rows)
    assert any(row["guardrail_status"] == "passed" for row in rows)
    assert float(ctx["state_row"]["deadman_cooldown_until_ts"]) > now


def test_over_drift_is_rejected(monkeypatch, tmp_path: Path):
    mod = _load_module()
    now = 20_000.0
    market = "SOL-USD"
    env_path = tmp_path / ".env.sol"
    _write_env(env_path, market=market, spread_multiplier="1.1")

    events = [
        {
            "ts": now - 100,
            "type": "run_start",
            "market": market,
            "config": {"max_position_size": "100", "inventory_hard_pct": "0.95"},
        },
        {"ts": now - 50, "type": "snapshot", "market": market, "position": "0", "best_bid": "100", "best_ask": "101"},
    ]
    journal_path = tmp_path / "mm_SOL-USD_20260222_000000.jsonl"
    _write_journal(journal_path, events)

    ctx = _process_market_common(tmp_path, mod, market=market, now=now)
    baseline_path = ctx["baselines_dir"] / f"{mod.slugify(market)}.json"
    baseline_path.write_text(
        json.dumps(
            _baseline_payload(
                market=market,
                env_path=env_path,
                env_hash=mod._env_file_hash(env_path),
                captured_ts=now - 4000,
                baseline_params={
                    "MM_SPREAD_MULTIPLIER": "1.0",
                },
            )
        )
        + "\n"
    )

    original = mod._compute_candidate_updates

    def _fake_compute(**kwargs):
        _ = kwargs
        return (
            {"MM_SPREAD_MULTIPLIER": Decimal("5.0")},
            ["unit_test"],
            "medium",
            False,
            "human",
        )

    monkeypatch.setattr(mod, "_compute_candidate_updates", _fake_compute)
    try:
        result = mod._process_market(
            market=market,
            env_path=env_path,
            journal_path=journal_path,
            baselines_dir=ctx["baselines_dir"],
            proposals_path=ctx["proposals_path"],
            changelog_path=ctx["changelog_path"],
            apply_receipts_path=ctx["apply_receipts_path"],
            baseline_refresh_log_path=ctx["baseline_refresh_log_path"],
            state_row=ctx["state_row"],
            now_value=now,
            deadman_window_s=3600,
            deadman_cooldown_s=1800,
            inventory_window_s=7200,
            max_changes_per_hour=10,
        )
    finally:
        monkeypatch.setattr(mod, "_compute_candidate_updates", original)

    row = result["rows"][0]
    assert row["guardrail_status"] == "rejected"
    assert row["guardrail_reason"] == "max_drift_exceeded"


def test_hourly_cap_blocks_eleventh_change(monkeypatch, tmp_path: Path):
    mod = _load_module()
    now = 30_000.0
    market = "ARB-USD"
    env_path = tmp_path / ".env.arb"
    _write_env(env_path, market=market, spread_multiplier="1.0")
    journal_path = tmp_path / "mm_ARB-USD_20260222_000000.jsonl"
    _write_journal(
        journal_path,
        [
            {"ts": now - 100, "type": "run_start", "market": market, "config": {"max_position_size": "100"}},
            {"ts": now - 10, "type": "snapshot", "market": market, "position": "0", "best_bid": "50", "best_ask": "51"},
        ],
    )

    ctx = _process_market_common(tmp_path, mod, market=market, now=now)
    ctx["state_row"]["proposal_timestamps"] = [now - 100 + i for i in range(10)]
    baseline_path = ctx["baselines_dir"] / f"{mod.slugify(market)}.json"
    baseline_path.write_text(
        json.dumps(
            _baseline_payload(
                market=market,
                env_path=env_path,
                env_hash=mod._env_file_hash(env_path),
                captured_ts=now - 4000,
                baseline_params={"MM_SPREAD_MULTIPLIER": "1.0"},
            )
        )
        + "\n"
    )

    original = mod._compute_candidate_updates
    monkeypatch.setattr(
        mod,
        "_compute_candidate_updates",
        lambda **kwargs: ({"MM_SPREAD_MULTIPLIER": Decimal("1.1")}, ["unit_test"], "medium", False, "human"),
    )
    try:
        result = mod._process_market(
            market=market,
            env_path=env_path,
            journal_path=journal_path,
            baselines_dir=ctx["baselines_dir"],
            proposals_path=ctx["proposals_path"],
            changelog_path=ctx["changelog_path"],
            apply_receipts_path=ctx["apply_receipts_path"],
            baseline_refresh_log_path=ctx["baseline_refresh_log_path"],
            state_row=ctx["state_row"],
            now_value=now,
            deadman_window_s=3600,
            deadman_cooldown_s=1800,
            inventory_window_s=7200,
            max_changes_per_hour=10,
        )
    finally:
        monkeypatch.setattr(mod, "_compute_candidate_updates", original)

    row = result["rows"][0]
    assert row["guardrail_status"] == "rejected"
    assert row["guardrail_reason"] == "hourly_cap_exceeded"


def test_baseline_refresh_reasons(tmp_path: Path):
    mod = _load_module()
    market = "LTC-USD"
    env_path = tmp_path / ".env.ltc"
    _write_env(env_path, market=market)
    refresh_log_path = tmp_path / "refresh.jsonl"
    receipts_path = tmp_path / "apply_receipts.jsonl"
    baseline_path = tmp_path / f"{mod.slugify(market)}.json"

    old_baseline = _baseline_payload(
        market=market,
        env_path=env_path,
        env_hash="stale-hash",
        captured_ts=1000.0,
        baseline_params={"MM_SPREAD_MULTIPLIER": "1.0"},
    )
    baseline_path.write_text(json.dumps(old_baseline) + "\n")
    env_map = mod.parse_env(mod.read_env_lines(env_path))

    refreshed = mod._refresh_baseline_if_needed(
        market=market,
        env_path=env_path,
        baseline_path=baseline_path,
        env_map=env_map,
        receipts=[],
        refresh_log_path=refresh_log_path,
        now_value=2000.0,
    )
    assert refreshed["refresh_reason"] == "external_env_change"

    baseline_path.write_text(json.dumps(old_baseline) + "\n")
    env_hash = mod._env_file_hash(env_path)
    receipts = [
        {
            "proposal_id": "p1",
            "applied_by": "human",
            "applied_ts": 1500.0,
            "result": "applied",
            "env_before_hash": "x",
            "env_after_hash": env_hash,
        }
    ]
    receipts_path.write_text(json.dumps(receipts[0]) + "\n")
    refreshed2 = mod._refresh_baseline_if_needed(
        market=market,
        env_path=env_path,
        baseline_path=baseline_path,
        env_map=env_map,
        receipts=mod.read_jsonl(receipts_path),
        refresh_log_path=refresh_log_path,
        now_value=2500.0,
    )
    assert refreshed2["refresh_reason"] == "applied_change"


def test_warren_auto_apply_filter_and_no_env_mutation(tmp_path: Path):
    mod = _load_module()
    assert mod._warren_auto_apply_candidate(
        {"deadman": True, "guardrail_status": "passed", "confidence": "high"}
    )
    assert not mod._warren_auto_apply_candidate(
        {"deadman": False, "guardrail_status": "passed", "confidence": "high"}
    )

    now = 40_000.0
    market = "AVAX-USD"
    env_path = tmp_path / ".env.avax"
    _write_env(env_path, market=market, spread_multiplier="1.2")
    env_before = env_path.read_text()

    journal_path = tmp_path / "mm_AVAX-USD_20260222_000000.jsonl"
    _write_journal(
        journal_path,
        [
            {
                "ts": now - 200,
                "type": "run_start",
                "market": market,
                "config": {"max_position_size": "100", "inventory_hard_pct": "0.95"},
            },
            {"ts": now - 180, "type": "order_placed", "market": market},
            {
                "ts": now - 170,
                "type": "fill",
                "market": market,
                "side": "BUY",
                "price": "100",
                "qty": "1",
            },
            {"ts": now - 165, "type": "snapshot", "market": market, "position": "1", "best_bid": "100", "best_ask": "101"},
        ],
    )

    ctx = _process_market_common(tmp_path, mod, market=market, now=now)
    baseline_path = ctx["baselines_dir"] / f"{mod.slugify(market)}.json"
    baseline_path.write_text(
        json.dumps(
            _baseline_payload(
                market=market,
                env_path=env_path,
                env_hash=mod._env_file_hash(env_path),
                captured_ts=now - 1000,
                baseline_params={
                    "MM_SPREAD_MULTIPLIER": "1.2",
                    "MM_MIN_OFFSET_BPS": "4",
                    "MM_MAX_OFFSET_BPS": "80",
                    "MM_ORDER_SIZE_MULTIPLIER": "1.0",
                    "MM_INVENTORY_SKEW_FACTOR": "0.35",
                    "MM_MIN_REPRICE_INTERVAL_S": "0.6",
                    "MM_MIN_REPRICE_MOVE_TICKS": "2",
                    "MM_MIN_REPRICE_EDGE_DELTA_BPS": "0.5",
                    "MM_REPRICE_TOLERANCE_PERCENT": "0.2",
                },
            )
        )
        + "\n"
    )

    _ = mod._process_market(
        market=market,
        env_path=env_path,
        journal_path=journal_path,
        baselines_dir=ctx["baselines_dir"],
        proposals_path=ctx["proposals_path"],
        changelog_path=ctx["changelog_path"],
        apply_receipts_path=ctx["apply_receipts_path"],
        baseline_refresh_log_path=ctx["baseline_refresh_log_path"],
        state_row=ctx["state_row"],
        now_value=now,
        deadman_window_s=3600,
        deadman_cooldown_s=1800,
        inventory_window_s=7200,
        max_changes_per_hour=10,
    )
    assert env_path.read_text() == env_before
