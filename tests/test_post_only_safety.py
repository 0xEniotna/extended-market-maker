from __future__ import annotations

import time
from decimal import Decimal
from types import SimpleNamespace

from market_maker.post_only_safety import PostOnlySafety


def _settings(**overrides):
    base = {
        "post_only_safety_ticks": 2,
        "adaptive_pof_enabled": True,
        "pof_cooldown_s": 1.0,
        "pof_streak_reset_s": 45.0,
        "pof_max_safety_ticks": 8,
        "pof_backoff_multiplier": Decimal("2.0"),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _round_to_tick(price: Decimal, side) -> Decimal:
    _ = side
    return price.quantize(Decimal("0.01"))


def test_buy_clamp_stays_away_from_ask():
    safety = PostOnlySafety(_settings(), Decimal("0.01"), _round_to_tick)
    clamped = safety.clamp_price(
        side="BUY",
        target_price=Decimal("1.01"),
        bid_price=Decimal("1.00"),
        ask_price=Decimal("1.01"),
    )
    assert clamped == Decimal("0.99")


def test_rejection_increases_streak_and_ticks():
    safety = PostOnlySafety(_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    safety.on_rejection(key)
    safety.on_rejection(key)

    assert safety.pof_streak[key] == 2
    assert safety.effective_ticks(key) == 4
    assert safety.pof_until[key] > time.monotonic()


def test_success_decays_streak():
    safety = PostOnlySafety(_settings(), Decimal("0.01"), _round_to_tick)
    key = ("SELL", 1)
    safety.on_rejection(key)
    safety.on_rejection(key)
    safety.on_success(key)
    assert safety.pof_streak[key] == 1
