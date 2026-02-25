from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from market_maker.decision_models import RegimeState
from market_maker.guard_policy import GuardPolicy


def _settings(**overrides):
    base = {
        "min_spread_bps": Decimal("0"),
        "imbalance_pause_threshold": Decimal("0.70"),
        "imbalance_window_s": 2.0,
        "vol_regime_short_window_s": 15.0,
        "min_offset_bps": Decimal("3"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_blocks_on_min_spread():
    policy = GuardPolicy(_settings(min_spread_bps=Decimal("2")))
    decision = policy.check(
        side="BUY",
        level=0,
        spread_bps=Decimal("1"),
        imbalance=Decimal("0"),
        regime=RegimeState(regime="NORMAL"),
    )
    assert not decision.allow
    assert decision.reason == "skip_min_spread"


def test_sets_and_honors_imbalance_pause():
    policy = GuardPolicy(_settings())
    first = policy.check(
        side="SELL",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0.9"),
        regime=RegimeState(regime="NORMAL"),
    )
    assert not first.allow
    assert first.reason == "skip_imbalance"

    second = policy.check(
        side="SELL",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0.0"),
        regime=RegimeState(regime="NORMAL"),
    )
    assert not second.allow
    assert second.reason == "skip_imbalance_pause"


def test_inventory_override_bypasses_imbalance_block():
    policy = GuardPolicy(_settings())
    decision = policy.check(
        side="SELL",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0.9"),
        regime=RegimeState(regime="NORMAL"),
        inventory_override=True,
    )
    assert decision.allow
    assert decision.reason == "allow_imbalance_inventory_override"


def test_inventory_override_bypasses_existing_imbalance_pause():
    policy = GuardPolicy(_settings())
    first = policy.check(
        side="SELL",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0.9"),
        regime=RegimeState(regime="NORMAL"),
    )
    assert not first.allow
    assert first.reason == "skip_imbalance"

    decision = policy.check(
        side="SELL",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0"),
        regime=RegimeState(regime="NORMAL"),
        inventory_override=True,
    )
    assert decision.allow
    assert decision.reason == "allow_imbalance_inventory_override"


def test_extreme_regime_pauses_with_cooldown():
    policy = GuardPolicy(_settings())
    first = policy.check(
        side="BUY",
        level=1,
        spread_bps=Decimal("4"),
        imbalance=Decimal("0"),
        regime=RegimeState(regime="EXTREME", pause=True),
    )
    assert not first.allow
    assert first.reason == "skip_toxicity"
    assert first.pause_until_ts is not None

    second = policy.check(
        side="BUY",
        level=1,
        spread_bps=Decimal("4"),
        imbalance=Decimal("0"),
        regime=RegimeState(regime="NORMAL"),
    )
    assert not second.allow
    assert second.reason == "skip_toxicity"


def test_elevated_regime_adds_extra_offset():
    policy = GuardPolicy(_settings(min_offset_bps=Decimal("4")))
    decision = policy.check(
        side="BUY",
        level=0,
        spread_bps=Decimal("5"),
        imbalance=Decimal("0"),
        regime=RegimeState(regime="ELEVATED", offset_scale=Decimal("1.5")),
    )
    assert decision.allow
    assert decision.extra_offset_bps == Decimal("2")
