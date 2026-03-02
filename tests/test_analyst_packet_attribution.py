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


def test_spread_capture_sign_correctness_buy_vs_sell():
    mod = _load_analyser_module()

    fills = [
        {
            "ts": 1.0,
            "side": "BUY",
            "price": "99",
            "qty": "1",
            "mid": "100",
        },
        {
            "ts": 2.0,
            "side": "SELL",
            "price": "101",
            "qty": "2",
            "mid": "100",
        },
    ]

    spread_capture = mod.compute_spread_capture_usd(fills, [], [])
    assert spread_capture == Decimal("3")


def test_inventory_drift_integral_known_path():
    mod = _load_analyser_module()

    fills = [
        {
            "ts": 0.5,
            "side": "BUY",
            "qty": "1",
            "price": "100",
        }
    ]
    mid_points = [
        (0.0, Decimal("100")),
        (1.0, Decimal("101")),
        (2.0, Decimal("99")),
    ]

    drift = mod.compute_inventory_drift_usd(
        fills,
        mid_points,
        start_position=Decimal("0"),
    )
    assert drift == Decimal("-2")


def test_markout_aggregation_and_toxicity_share():
    mod = _load_analyser_module()

    fills = [
        {
            "ts": 0.0,
            "side": "BUY",
            "price": "100",
            "qty": "1",
            "is_taker": False,
        },
        {
            "ts": 0.0,
            "side": "SELL",
            "price": "100",
            "qty": "1",
            "is_taker": False,
        },
    ]
    ts_values = [0.0, 0.25, 1.0, 1.25]
    mid_values = [
        Decimal("100"),
        Decimal("99"),
        Decimal("100"),
        Decimal("101"),
    ]

    markout = mod.aggregate_markouts(fills, ts_values, mid_values, [0.25, 1.0])

    assert markout["250ms"]["mean_bps"] == Decimal("0")
    assert markout["250ms"]["toxicity_share"] == Decimal("0.5")
    assert markout["250ms"]["maker_buy"]["mean_bps"] == Decimal("-100")
    assert markout["250ms"]["maker_sell"]["mean_bps"] == Decimal("100")


def test_diagnosis_rules_trigger_expected_actions():
    mod = _load_analyser_module()

    zero_fill_diag = mod.diagnose_market(
        {
            "fills_count": 0,
            "orders_sent": 80,
            "zero_fill_flag": True,
            "markout_250ms_mean": None,
            "toxicity_share_250ms": None,
            "lifetime_p10_ms": None,
            "spread_capture_usd": Decimal("0"),
            "inventory_drift_usd": None,
            "funding_usd": None,
            "churn_ratio": Decimal("0"),
        }
    )
    assert any("Too wide / not competitive" in action for action in zero_fill_diag["actions"])

    toxic_diag = mod.diagnose_market(
        {
            "fills_count": 12,
            "orders_sent": 30,
            "zero_fill_flag": False,
            "markout_250ms_mean": Decimal("-1.2"),
            "toxicity_share_250ms": Decimal("0.70"),
            "lifetime_p10_ms": Decimal("100"),
            "spread_capture_usd": Decimal("3"),
            "inventory_drift_usd": Decimal("-20"),
            "funding_usd": Decimal("-10"),
            "churn_ratio": Decimal("20"),
        }
    )

    assert toxic_diag["severity"] == "SEV1"
    actions_text = " ".join(toxic_diag["actions"])
    assert "Toxic/picked off" in actions_text
    assert "Inventory risk" in actions_text
    assert "Funding drag" in actions_text
    assert "Over-churning" in actions_text
