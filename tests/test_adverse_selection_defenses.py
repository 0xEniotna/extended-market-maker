"""Tests for adverse selection defenses (Prompt 3).

Covers:
- Latency tracking in OrderManager
- Latency-aware safety ticks in PostOnlySafety
- Fill quality / markout tracking in FillQualityTracker
- POF streak decay (on_success) in PostOnlySafety
- POF-spread correlation in PostOnlySafety
"""
from __future__ import annotations

import sys
import time
from collections import deque
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# SDK stubs (must be set before importing market_maker modules)
# ---------------------------------------------------------------------------

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.trading_client",
]

for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED",
    CANCELLED="CANCELLED",
    EXPIRED="EXPIRED",
    REJECTED="REJECTED",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT")
_orders_mod.OrderType.MARKET = "MARKET"
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT", IOC="IOC")
_orders_mod.OpenOrderModel = object

OrderSide = _orders_mod.OrderSide
OrderStatus = _orders_mod.OrderStatus

from market_maker.order_manager import OrderManager  # noqa: E402
from market_maker.post_only_safety import PostOnlySafety  # noqa: E402
from market_maker.fill_quality import FillQualityTracker, LevelFillQuality  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================


def _pof_settings(**overrides):
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


def _make_order_update(ext_id, status, exchange_id=None, status_reason=None):
    """Minimal mock of an OpenOrderModel for handle_order_update."""
    return SimpleNamespace(
        external_id=ext_id,
        status=status,
        id=exchange_id,
        status_reason=status_reason,
    )


def _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101")):
    """Minimal mock orderbook manager for FillQualityTracker."""
    ob = SimpleNamespace()
    ob.best_bid = lambda: SimpleNamespace(price=bid_price) if bid_price else None
    ob.best_ask = lambda: SimpleNamespace(price=ask_price) if ask_price else None
    return ob


# ===========================================================================
# A. Latency Tracking (OrderManager)
# ===========================================================================


@pytest.mark.asyncio
async def test_latency_tracking_records_samples():
    """place_order records send time, handle_order_update records latency."""
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(
            status="OK", error=None, data=SimpleNamespace(id=1)
        )
    )
    mgr = OrderManager(client, "TEST-USD")

    ext_id = await mgr.place_order(
        side=OrderSide.BUY, price=Decimal("10"), size=Decimal("1"), level=0
    )
    assert ext_id is not None

    # Simulate stream ack arriving
    mgr.handle_order_update(_make_order_update(ext_id, "OPEN", exchange_id=1))

    assert mgr.latency_sample_count() == 1
    assert mgr.avg_placement_latency_ms() >= 0


@pytest.mark.asyncio
async def test_latency_tracking_no_double_count():
    """A second stream update should NOT record a second latency sample."""
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(
            status="OK", error=None, data=SimpleNamespace(id=2)
        )
    )
    mgr = OrderManager(client, "TEST-USD")

    ext_id = await mgr.place_order(
        side=OrderSide.SELL, price=Decimal("20"), size=Decimal("1"), level=0
    )

    mgr.handle_order_update(_make_order_update(ext_id, "OPEN", exchange_id=2))
    mgr.handle_order_update(_make_order_update(ext_id, "OPEN", exchange_id=2))

    assert mgr.latency_sample_count() == 1


@pytest.mark.asyncio
async def test_latency_tracking_cleared_on_rejection():
    """A rejected order should not leave dangling send timestamps."""
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(status="ERR", error="rejected", data=None)
    )
    mgr = OrderManager(client, "TEST-USD")

    result = await mgr.place_order(
        side=OrderSide.BUY, price=Decimal("10"), size=Decimal("1"), level=0
    )
    assert result is None
    assert mgr.latency_sample_count() == 0
    assert len(mgr._placement_send_ts) == 0


@pytest.mark.asyncio
async def test_latency_tracking_cleared_on_exception():
    """A failed order should clean up send timestamps."""
    client = MagicMock()
    client.place_order = AsyncMock(side_effect=Exception("network error"))
    mgr = OrderManager(client, "TEST-USD")

    result = await mgr.place_order(
        side=OrderSide.BUY, price=Decimal("10"), size=Decimal("1"), level=0
    )
    assert result is None
    assert len(mgr._placement_send_ts) == 0


@pytest.mark.asyncio
async def test_latency_rolling_average():
    """Average latency correctly reflects multiple samples."""
    client = MagicMock()
    client.place_order = AsyncMock(
        return_value=SimpleNamespace(
            status="OK", error=None, data=SimpleNamespace(id=None)
        )
    )
    mgr = OrderManager(client, "TEST-USD")

    # Inject synthetic latency samples directly.
    mgr._latency_samples.extend([10.0, 20.0, 30.0])
    avg = mgr.avg_placement_latency_ms()
    assert abs(avg - 20.0) < 0.01


def test_latency_empty_returns_zero():
    """avg_placement_latency_ms returns 0 when no samples exist."""
    client = MagicMock()
    mgr = OrderManager(client, "TEST-USD")
    assert mgr.avg_placement_latency_ms() == 0.0
    assert mgr.latency_sample_count() == 0


# ===========================================================================
# B. Latency-Aware Safety Ticks (PostOnlySafety)
# ===========================================================================


def test_latency_ticks_adds_extra_ticks():
    """effective_ticks includes latency buffer when avg_latency > 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # No POF escalation, just latency: ceil(100 / 50) = 2 extra ticks
    ticks = safety.effective_ticks(key, avg_latency_ms=100.0, tick_time_ms=50.0)
    assert ticks == 2 + 2  # base_ticks(2) + latency_ticks(2)


def test_latency_ticks_zero_when_no_latency():
    """No extra ticks when avg_latency_ms is 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("SELL", 1)
    ticks = safety.effective_ticks(key, avg_latency_ms=0.0, tick_time_ms=50.0)
    assert ticks == 2  # just base_ticks


def test_latency_ticks_zero_when_no_tick_time():
    """No extra ticks when tick_time_ms is 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    ticks = safety.effective_ticks(key, avg_latency_ms=100.0, tick_time_ms=0.0)
    assert ticks == 2


def test_latency_ticks_fractional_rounds_up():
    """ceil(75/50) = 2 extra ticks."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    ticks = safety.effective_ticks(key, avg_latency_ms=75.0, tick_time_ms=50.0)
    assert ticks == 2 + 2  # base(2) + ceil(75/50)=2


def test_latency_ticks_stacks_with_pof_escalation():
    """Latency ticks add on top of adaptive POF ticks."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # Build up 3 POF rejections
    safety.on_rejection(key)
    safety.on_rejection(key)
    safety.on_rejection(key)
    # pof_streak=3, dynamic_ticks = base(2)+3 = 5
    assert safety.pof_streak[key] == 3
    assert safety.dynamic_safety_ticks[key] == 5

    # Add latency: ceil(100/50) = 2
    ticks = safety.effective_ticks(key, avg_latency_ms=100.0, tick_time_ms=50.0)
    assert ticks == 5 + 2  # dynamic(5) + latency(2)


def test_latency_ticks_static_method():
    """Directly test _latency_ticks static method."""
    assert PostOnlySafety._latency_ticks(0, 0) == 0
    assert PostOnlySafety._latency_ticks(100, 0) == 0
    assert PostOnlySafety._latency_ticks(0, 50) == 0
    assert PostOnlySafety._latency_ticks(100, 50) == 2
    assert PostOnlySafety._latency_ticks(51, 50) == 2  # ceil
    assert PostOnlySafety._latency_ticks(50, 50) == 1
    assert PostOnlySafety._latency_ticks(10, 100) == 1  # ceil(0.1) = 1


# ===========================================================================
# C. Fill Quality / Markout Tracking (FillQualityTracker)
# ===========================================================================


def test_fill_quality_edge_bps_buy():
    """Edge at fill for a BUY: mid=100.5, fill=100 -> edge = (100.5-100)/100.5 * 10000."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)

    key = ("BUY", 0)
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")

    q = tracker.level_quality(key)
    assert q.sample_count == 1
    # mid = 100.5, edge = (100.5 - 100) / 100.5 * 10000 ≈ 49.75 bps
    assert q.avg_edge_bps > Decimal("49") and q.avg_edge_bps < Decimal("50")


def test_fill_quality_edge_bps_sell():
    """Edge at fill for a SELL: mid=100.5, fill=101 -> edge = (101-100.5)/100.5 * 10000."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)

    key = ("SELL", 0)
    tracker.record_fill(key=key, fill_price=Decimal("101"), side_name="SELL")

    q = tracker.level_quality(key)
    assert q.sample_count == 1
    assert q.avg_edge_bps > Decimal("49") and q.avg_edge_bps < Decimal("50")


def test_fill_quality_no_mid():
    """No edge recorded when bid or ask is missing."""
    ob = _make_ob_manager(bid_price=None, ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    key = ("BUY", 0)
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")
    q = tracker.level_quality(key)
    assert q.sample_count == 0


def test_fill_quality_markout_sync_buy():
    """Synchronous markout recording for BUY: price moved up → positive markout."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    key = ("BUY", 0)

    # Fill at 100, mid later at (102 + 103) / 2 = 102.5
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")

    # Simulate mid price moving up for markout
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("102"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("103"))

    tracker.record_markout_sync(key, Decimal("100"), "BUY", "1s")
    q = tracker.level_quality(key)
    # markout_1s = (102.5 - 100) / 100 * 10000 = 250 bps
    assert q.avg_markout_1s > Decimal("200")


def test_fill_quality_markout_sync_sell():
    """Synchronous markout recording for SELL: price moved down → positive markout."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    key = ("SELL", 0)
    tracker.record_fill(key=key, fill_price=Decimal("101"), side_name="SELL")

    # Mid drops: (98 + 99) / 2 = 98.5
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("98"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("99"))

    tracker.record_markout_sync(key, Decimal("101"), "SELL", "5s")
    q = tracker.level_quality(key)
    # markout_5s = (101 - 98.5) / 101 * 10000 ≈ 247.5 bps
    assert q.avg_markout_5s > Decimal("200")


def test_fill_quality_adverse_fill_pct():
    """Adverse fill % reflects negative markouts."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    key = ("BUY", 0)

    # Record 2 fills for samples
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")

    # 5s markout: price dropped → adverse for BUY
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("98"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("99"))
    tracker.record_markout_sync(key, Decimal("100"), "BUY", "5s")

    # 5s markout: price rose → good for BUY
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("102"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("103"))
    tracker.record_markout_sync(key, Decimal("100"), "BUY", "5s")

    q = tracker.level_quality(key)
    # 1 adverse out of 2 = 50%
    assert q.adverse_fill_pct == Decimal("50")


def test_fill_quality_all_level_qualities():
    """all_level_qualities returns data for all tracked levels."""
    ob = _make_ob_manager()
    tracker = FillQualityTracker(ob)

    tracker.record_fill(key=("BUY", 0), fill_price=Decimal("100"), side_name="BUY")
    tracker.record_fill(key=("SELL", 1), fill_price=Decimal("101"), side_name="SELL")

    all_q = tracker.all_level_qualities()
    assert ("BUY", 0) in all_q
    assert ("SELL", 1) in all_q
    assert all_q[("BUY", 0)].sample_count == 1
    assert all_q[("SELL", 1)].sample_count == 1


def test_fill_quality_auto_widen_callback():
    """Adverse 5s markout triggers offset widen callback."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    tracker.set_min_acceptable_markout_bps(Decimal("-2"))

    callback_calls = []
    tracker.set_offset_widen_callback(lambda key, reason: callback_calls.append((key, reason)))

    key = ("BUY", 0)
    # Record enough adverse markouts to drop average below threshold
    for _ in range(5):
        tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")

    # Simulate price drop after fill (adverse for BUY)
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("99.5"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("99.7"))

    # Record adverse 5s markouts
    for _ in range(5):
        tracker.record_markout_sync(key, Decimal("100"), "BUY", "5s")

    assert len(callback_calls) > 0
    assert callback_calls[0][0] == key
    assert callback_calls[0][1] == "adverse_markout"


def test_fill_quality_no_widen_when_markout_ok():
    """No callback when markout is above threshold."""
    ob = _make_ob_manager(bid_price=Decimal("100"), ask_price=Decimal("101"))
    tracker = FillQualityTracker(ob)
    tracker.set_min_acceptable_markout_bps(Decimal("-2"))

    callback_calls = []
    tracker.set_offset_widen_callback(lambda key, reason: callback_calls.append((key, reason)))

    key = ("BUY", 0)
    tracker.record_fill(key=key, fill_price=Decimal("100"), side_name="BUY")

    # Price goes UP (good markout for BUY)
    ob.best_bid = lambda: SimpleNamespace(price=Decimal("102"))
    ob.best_ask = lambda: SimpleNamespace(price=Decimal("103"))
    tracker.record_markout_sync(key, Decimal("100"), "BUY", "5s")

    assert len(callback_calls) == 0


# ===========================================================================
# D. POF Streak Decay (PostOnlySafety.on_success)
# ===========================================================================


def test_on_success_halves_streak():
    """on_success halves streak (rounded down)."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # Build streak=6
    for _ in range(6):
        safety.on_rejection(key)
    assert safety.pof_streak[key] == 6

    # First success: min(6//2, 6-2) = min(3, 4) = 3
    safety.on_success(key)
    assert safety.pof_streak[key] == 3


def test_on_success_min_decay_of_2():
    """For small streaks, decay reduces by at least 2."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # streak=3 -> min(3//2, 3-2) = min(1, 1) = 1
    for _ in range(3):
        safety.on_rejection(key)
    safety.on_success(key)
    assert safety.pof_streak[key] == 1


def test_on_success_full_reset_after_3_consecutive():
    """After 3 consecutive successes, streak resets to 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # Build streak=10
    for _ in range(10):
        safety.on_rejection(key)
    assert safety.pof_streak[key] == 10

    # 3 consecutive successes
    safety.on_success(key)  # streak: min(10//2, 10-2) = 5
    assert safety.pof_streak[key] == 5
    safety.on_success(key)  # streak: min(5//2, 5-2) = 2
    assert safety.pof_streak[key] == 2
    safety.on_success(key)  # 3rd consecutive → full reset
    assert safety.pof_streak[key] == 0
    assert safety.pof_until[key] == 0.0


def test_on_success_consecutive_resets_on_rejection():
    """A rejection resets the consecutive success counter."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    # Build streak to 10 (high enough to survive several successes)
    for _ in range(10):
        safety.on_rejection(key)
    assert safety.pof_streak[key] == 10

    safety.on_success(key)  # consec=1, streak: min(10//2,10-2)=5
    assert safety.pof_streak[key] == 5
    safety.on_success(key)  # consec=2, streak: min(5//2,5-2)=2
    assert safety.pof_streak[key] == 2
    safety.on_rejection(key)  # consec reset to 0, streak increments to 3
    assert safety._level_consecutive_successes[key] == 0
    assert safety.pof_streak[key] == 3

    # Now need 3 consecutive successes for full reset
    safety.on_success(key)  # consec=1, streak: min(3//2,3-2)=1
    assert safety.pof_streak[key] == 1
    safety.on_success(key)  # consec=2, streak: min(1//2,1-2)=max(0,min(0,-1))=0
    assert safety.pof_streak[key] == 0  # already 0 via decay
    # on_success sees streak=0 and returns early, so consec counter is not incremented
    safety.on_success(key)  # streak is 0, noop
    assert safety.pof_streak[key] == 0


def test_on_success_noop_when_streak_zero():
    """on_success does nothing when streak is already 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    safety.on_success(key)  # should be noop
    assert safety.pof_streak.get(key, 0) == 0


def test_on_success_updates_dynamic_ticks():
    """Dynamic safety ticks are reduced along with the streak."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)
    for _ in range(6):
        safety.on_rejection(key)
    # dynamic_ticks = min(8, 2+6) = 8
    assert safety.dynamic_safety_ticks[key] == 8

    safety.on_success(key)  # streak: 3, ticks = 2+3 = 5
    assert safety.dynamic_safety_ticks[key] == 5


def test_on_success_disabled_noop():
    """on_success does nothing when adaptive_pof_enabled is False."""
    safety = PostOnlySafety(
        _pof_settings(adaptive_pof_enabled=False), Decimal("0.01"), _round_to_tick
    )
    key = ("BUY", 0)
    safety._level_pof_streak[key] = 5
    safety.on_success(key)
    assert safety.pof_streak[key] == 5  # unchanged


# ===========================================================================
# E. POF-Spread Correlation (PostOnlySafety)
# ===========================================================================


def test_pof_rate_60s_counts_recent_rejections():
    """pof_rate_60s returns count of POF rejections in last 60s."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    key = ("BUY", 0)

    safety.on_rejection(key)
    safety.on_rejection(key)
    safety.on_rejection(key)

    assert safety.pof_rate_60s() == 3


def test_pof_rate_60s_expires_old():
    """Old timestamps are pruned from pof_rate_60s."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    # Inject an old timestamp
    safety._pof_timestamps.append(time.monotonic() - 120.0)
    safety._pof_timestamps.append(time.monotonic())
    assert safety.pof_rate_60s() == 1  # old one pruned


def test_update_pof_offset_boost_increases():
    """Boost increases when POF rate > 10%."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    assert safety.pof_offset_boost_bps == Decimal("0")

    # Simulate: 5 POFs out of 20 placements = 25% > 10%
    for _ in range(5):
        safety._pof_timestamps.append(time.monotonic())

    safety.update_pof_offset_boost(20)
    assert safety.pof_offset_boost_bps == Decimal("1")


def test_update_pof_offset_boost_decreases():
    """Boost decreases when POF rate < 5%."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    safety._pof_offset_boost_bps = Decimal("2")

    # Simulate: 1 POF out of 100 placements = 1% < 5%
    safety._pof_timestamps.append(time.monotonic())

    safety.update_pof_offset_boost(100)
    assert safety.pof_offset_boost_bps == Decimal("1")


def test_update_pof_offset_boost_capped_at_3():
    """Boost cannot exceed 3 bps."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    safety._pof_offset_boost_bps = Decimal("3")

    for _ in range(10):
        safety._pof_timestamps.append(time.monotonic())

    safety.update_pof_offset_boost(20)  # 50% > 10%
    assert safety.pof_offset_boost_bps == Decimal("3")  # still capped


def test_update_pof_offset_boost_floored_at_0():
    """Boost cannot go below 0 bps."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    safety._pof_offset_boost_bps = Decimal("0")

    # 0 POFs out of 100 placements = 0% < 5%
    safety.update_pof_offset_boost(100)
    assert safety.pof_offset_boost_bps == Decimal("0")


def test_update_pof_offset_boost_noop_zero_placements():
    """Boost unchanged when total_placements_60s is 0."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    safety._pof_offset_boost_bps = Decimal("2")

    safety.update_pof_offset_boost(0)
    assert safety.pof_offset_boost_bps == Decimal("2")


def test_update_pof_offset_boost_stays_in_deadband():
    """Boost unchanged when 5% <= POF rate <= 10%."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    safety._pof_offset_boost_bps = Decimal("1")

    # 7 POFs out of 100 = 7%, in 5-10% deadband
    for _ in range(7):
        safety._pof_timestamps.append(time.monotonic())

    safety.update_pof_offset_boost(100)
    assert safety.pof_offset_boost_bps == Decimal("1")  # unchanged


def test_pof_boost_progressive_increase():
    """Multiple calls with high POF rate increase boost step by step."""
    safety = PostOnlySafety(_pof_settings(), Decimal("0.01"), _round_to_tick)
    assert safety.pof_offset_boost_bps == Decimal("0")

    for _ in range(15):
        safety._pof_timestamps.append(time.monotonic())

    safety.update_pof_offset_boost(20)  # 75% > 10% → +1
    assert safety.pof_offset_boost_bps == Decimal("1")
    safety.update_pof_offset_boost(20)  # still 75% → +1
    assert safety.pof_offset_boost_bps == Decimal("2")
    safety.update_pof_offset_boost(20)  # still 75% → +1
    assert safety.pof_offset_boost_bps == Decimal("3")
    safety.update_pof_offset_boost(20)  # capped at 3
    assert safety.pof_offset_boost_bps == Decimal("3")
