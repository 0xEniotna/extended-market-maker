"""
Tests for market maker strategy: tick rounding, target price, and repricing logic.

Exercises the pure computation methods in isolation by building a minimal
strategy instance with mocked dependencies.
"""
from __future__ import annotations

import asyncio
import sys
import time
from decimal import Decimal
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the x10 SDK modules so tests run without it installed
# ---------------------------------------------------------------------------
_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.accounts",
    "x10.perpetual.configuration",
    "x10.perpetual.orderbook",
    "x10.perpetual.orders",
    "x10.perpetual.positions",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.perpetual.trades",
    "x10.perpetual.trading_client",
    "x10.utils",
    "x10.utils.http",
]

for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

# Provide concrete enum-like objects that the strategy imports
_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
_orders_mod.OrderStatus = SimpleNamespace(
    FILLED="FILLED", CANCELLED="CANCELLED", EXPIRED="EXPIRED", REJECTED="REJECTED",
)
_orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT")
_orders_mod.TimeInForce = SimpleNamespace(GTT="GTT")

# Now safe to import
from market_maker.config import MarketMakerSettings  # noqa: E402
from market_maker.decision_models import TrendState  # noqa: E402
from market_maker.orderbook_manager import OrderbookManager, PriceLevel  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeOrderbook:
    """Minimal stand-in for OrderbookManager used in strategy tests."""

    def __init__(self, bid: Decimal, ask: Decimal):
        self._bid = PriceLevel(price=bid, size=Decimal("100"))
        self._ask = PriceLevel(price=ask, size=Decimal("100"))
        self.best_bid_condition = asyncio.Condition()
        self.best_ask_condition = asyncio.Condition()
        self._stale = False
        self._micro_vol_bps: Optional[Decimal] = None
        self._micro_drift_bps: Optional[Decimal] = None
        self._imbalance: Optional[Decimal] = None

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask

    def spread_bps(self):
        mid = (self._bid.price + self._ask.price) / 2
        return (self._ask.price - self._bid.price) / mid * Decimal("10000")

    def spread_bps_ema(self):
        return self.spread_bps()

    def is_stale(self):
        return self._stale

    def micro_volatility_bps(self, window_s: float):
        _ = window_s
        return self._micro_vol_bps

    def micro_drift_bps(self, window_s: float):
        _ = window_s
        return self._micro_drift_bps

    def orderbook_imbalance(self, window_s: float):
        _ = window_s
        return self._imbalance

    def market_snapshot(
        self,
        depth: int = 5,
        *,
        micro_vol_window_s: float = 5.0,
        micro_drift_window_s: float = 3.0,
        imbalance_window_s: float = 2.0,
    ):
        _ = (micro_vol_window_s, micro_drift_window_s, imbalance_window_s)
        bid = self._bid
        ask = self._ask
        mid = (bid.price + ask.price) / 2
        spread_bps = (ask.price - bid.price) / mid * Decimal("10000")
        return {
            "best_bid": bid.price,
            "best_ask": ask.price,
            "mid": mid,
            "spread_bps": spread_bps,
            "bids_top": [{"price": bid.price, "size": bid.size}] * min(depth, 5),
            "asks_top": [{"price": ask.price, "size": ask.size}] * min(depth, 5),
            "micro_vol_bps": self._micro_vol_bps,
            "micro_drift_bps": self._micro_drift_bps,
            "imbalance": self._imbalance,
            "is_stale": self._stale,
            "seconds_since_update": 0.0,
            "depth": depth,
        }


class FakeRisk:
    def __init__(self, position: Decimal = Decimal("0"), total_pnl: Decimal = Decimal("0")):
        self._pos = position
        self._total_pnl = total_pnl

    def get_current_position(self):
        return self._pos

    def get_position_total_pnl(self):
        return self._total_pnl

    def can_place_order(self, side, size):
        return True

    def allowed_order_size(
        self,
        side,
        requested_size,
        reference_price,
        reserved_same_side_qty=Decimal("0"),
        reserved_open_notional_usd=Decimal("0"),
    ):
        _ = (
            side,
            reference_price,
            reserved_same_side_qty,
            reserved_open_notional_usd,
        )
        return requested_size


def _make_strategy(
    *,
    tick_size: Decimal = Decimal("0.0001"),
    min_order_size: Decimal = Decimal("10"),
    min_order_size_step: Decimal = Decimal("0.1"),
    market_min_order_size: Decimal = Decimal("1"),
    bid: Decimal = Decimal("1.6100"),
    ask: Decimal = Decimal("1.6110"),
    position: Decimal = Decimal("0"),
    offset_mode: str = "dynamic",
    spread_multiplier: Decimal = Decimal("1.0"),
    min_offset_bps: Decimal = Decimal("2"),
    max_offset_bps: Decimal = Decimal("50"),
    inventory_skew_factor: Decimal = Decimal("0.05"),
    max_position_size: Decimal = Decimal("300"),
    max_position_notional_usd: Decimal = Decimal("2500"),
    market_profile: str = "legacy",
):
    """Build a strategy instance with all dependencies mocked."""
    from market_maker.strategy import MarketMakerStrategy

    settings = MarketMakerSettings(
        vault_id="1",
        stark_private_key="0x1",
        stark_public_key="0x2",
        api_key="key",
        market_name="TEST-USD",
        market_profile=market_profile,
        offset_mode=offset_mode,
        spread_multiplier=spread_multiplier,
        min_offset_bps=min_offset_bps,
        max_offset_bps=max_offset_bps,
        inventory_skew_factor=inventory_skew_factor,
        max_position_size=max_position_size,
        max_position_notional_usd=max_position_notional_usd,
        reprice_tolerance_percent=Decimal("0.5"),
    )

    ob = FakeOrderbook(bid, ask)
    risk = FakeRisk(position)
    orders = MagicMock()
    orders.get_active_orders.return_value = {}
    orders.get_active_order.side_effect = (
        lambda ext_id: orders.get_active_orders.return_value.get(ext_id)
        if ext_id is not None
        else None
    )
    orders.find_order_by_external_id.side_effect = (
        lambda ext_id: orders.get_active_orders.return_value.get(ext_id)
    )
    orders.reserved_exposure.return_value = (Decimal("0"), Decimal("0"))
    orders.active_order_count.return_value = 0
    orders.consecutive_failures = 0
    orders.avg_placement_latency_ms.return_value = 0.0
    orders.latency_sample_count.return_value = 0

    strategy = MarketMakerStrategy(
        settings=settings,
        trading_client=MagicMock(),
        orderbook_mgr=ob,
        order_mgr=orders,
        risk_mgr=risk,
        account_stream=MagicMock(),
        metrics=MagicMock(),
        journal=MagicMock(),
        tick_size=tick_size,
        base_order_size=min_order_size,
        market_min_order_size=market_min_order_size,
        min_order_size_step=min_order_size_step,
    )
    return strategy


# ---------------------------------------------------------------------------
# Tests: tick rounding
# ---------------------------------------------------------------------------

OrderSide = _orders_mod.OrderSide


class TestTickRounding:
    def test_buy_rounds_down(self):
        s = _make_strategy()
        # 1.61379 → should floor to 1.6137
        result = s._round_to_tick(Decimal("1.61379"), OrderSide.BUY)
        assert result == Decimal("1.6137")

    def test_sell_rounds_up(self):
        s = _make_strategy()
        # 1.61371 → should ceil to 1.6138
        result = s._round_to_tick(Decimal("1.61371"), OrderSide.SELL)
        assert result == Decimal("1.6138")

    def test_sell_exact_tick_no_change(self):
        s = _make_strategy()
        result = s._round_to_tick(Decimal("1.6137"), OrderSide.SELL)
        assert result == Decimal("1.6137")

    def test_buy_exact_tick_no_change(self):
        s = _make_strategy()
        result = s._round_to_tick(Decimal("1.6137"), OrderSide.BUY)
        assert result == Decimal("1.6137")

    def test_none_side_rounds_down(self):
        """Default (no side) should round down for safety."""
        s = _make_strategy()
        result = s._round_to_tick(Decimal("1.61379"), None)
        assert result == Decimal("1.6137")


# ---------------------------------------------------------------------------
# Tests: target price computation
# ---------------------------------------------------------------------------

class TestTargetPrice:
    def test_buy_below_bid(self):
        """BUY target should be below the best bid."""
        s = _make_strategy(bid=Decimal("100.00"), ask=Decimal("100.10"))
        target = s._compute_target_price(OrderSide.BUY, 0, Decimal("100.00"))
        assert target < Decimal("100.00")

    def test_sell_above_ask(self):
        """SELL target should be above the best ask."""
        s = _make_strategy(bid=Decimal("100.00"), ask=Decimal("100.10"))
        target = s._compute_target_price(OrderSide.SELL, 0, Decimal("100.10"))
        assert target > Decimal("100.10")

    def test_inventory_skew_shifts_sell_down_when_long(self):
        """When long, sell target should be more aggressive (lower) than flat."""
        s_flat = _make_strategy(position=Decimal("0"))
        s_long = _make_strategy(position=Decimal("150"))

        sell_flat = s_flat._compute_target_price(OrderSide.SELL, 0, Decimal("1.6110"))
        sell_long = s_long._compute_target_price(OrderSide.SELL, 0, Decimal("1.6110"))

        assert sell_long < sell_flat

    def test_inventory_skew_shifts_buy_down_when_long(self):
        """When long, buy target should be less aggressive (lower) than flat."""
        s_flat = _make_strategy(position=Decimal("0"))
        s_long = _make_strategy(position=Decimal("150"))

        buy_flat = s_flat._compute_target_price(OrderSide.BUY, 0, Decimal("1.6100"))
        buy_long = s_long._compute_target_price(OrderSide.BUY, 0, Decimal("1.6100"))

        assert buy_long < buy_flat

    def test_safety_clamp_sell_never_below_bid(self):
        """Even with extreme skew, sell should never be at or below the bid."""
        s = _make_strategy(
            bid=Decimal("1.6100"),
            ask=Decimal("1.6102"),
            position=Decimal("299"),  # near max
            inventory_skew_factor=Decimal("1.0"),  # extreme skew
        )
        target = s._compute_target_price(OrderSide.SELL, 0, Decimal("1.6102"))
        assert target > Decimal("1.6100"), f"Sell target {target} should be > bid 1.6100"

    def test_safety_clamp_buy_never_above_ask(self):
        """Even with extreme skew, buy should never be at or above the ask."""
        s = _make_strategy(
            bid=Decimal("1.6100"),
            ask=Decimal("1.6102"),
            position=Decimal("-299"),  # near max short
            inventory_skew_factor=Decimal("1.0"),  # extreme skew
        )
        target = s._compute_target_price(OrderSide.BUY, 0, Decimal("1.6100"))
        assert target < Decimal("1.6102"), f"Buy target {target} should be < ask 1.6102"


# ---------------------------------------------------------------------------
# Tests: repricing tolerance
# ---------------------------------------------------------------------------

class TestNeedsReprice:
    def test_order_at_target_no_reprice(self):
        """Order at the target price should not trigger a reprice."""
        s = _make_strategy()
        target = s._compute_target_price(OrderSide.BUY, 0, Decimal("1.6100"))
        should, reason = s._needs_reprice(OrderSide.BUY, target, Decimal("1.6100"), 0)
        assert not should
        assert reason == "hold_within_tolerance"

    def test_order_far_from_target_triggers_reprice(self):
        """Order far from target should trigger a reprice."""
        s = _make_strategy()
        # Place the order way off
        should, reason = s._needs_reprice(
            OrderSide.BUY, Decimal("1.5000"), Decimal("1.6100"), 0
        )
        assert should
        assert reason == "replace_target_shift"

    def test_position_change_triggers_reprice(self):
        """After a fill changes position, the skew shifts the target enough
        to trigger a reprice even though BBO hasn't moved."""
        s = _make_strategy(
            position=Decimal("0"),
            inventory_skew_factor=Decimal("0.5"),  # strong skew
        )
        best = Decimal("1.6100")
        target_flat = s._compute_target_price(OrderSide.BUY, 0, best)

        # Simulate a fill: position changes to 200
        s._risk._pos = Decimal("200")

        # The old order is at target_flat, but now the target has shifted
        needs, reason = s._needs_reprice(OrderSide.BUY, target_flat, best, 0)
        assert needs, "Position change should trigger reprice via skew shift"
        assert reason == "replace_target_shift"

    def test_small_move_holds_below_tick_gate(self):
        s = _make_strategy(
            tick_size=Decimal("0.01"),
            bid=Decimal("100.00"),
            ask=Decimal("100.02"),
        )
        s._settings.reprice_tolerance_percent = Decimal("0.1")
        s._settings.min_reprice_move_ticks = 3
        s._settings.min_reprice_edge_delta_bps = Decimal("0")
        should, reason = s._needs_reprice(
            OrderSide.BUY,
            prev_price=Decimal("99.99"),
            current_best=Decimal("100.00"),
            level=0,
        )
        assert not should
        assert reason == "hold_below_tick_gate"

    def test_small_edge_delta_holds_below_edge_gate(self):
        s = _make_strategy(
            tick_size=Decimal("0.001"),
            bid=Decimal("100.000"),
            ask=Decimal("100.020"),
        )
        s._settings.reprice_tolerance_percent = Decimal("0.01")
        s._settings.min_reprice_move_ticks = 0
        s._settings.min_reprice_edge_delta_bps = Decimal("2.0")
        should, reason = s._needs_reprice(
            OrderSide.SELL,
            prev_price=Decimal("100.021"),
            current_best=Decimal("100.020"),
            level=0,
        )
        assert not should
        assert reason == "hold_below_edge_gate"


# ---------------------------------------------------------------------------
# Tests: post-only safety and toxicity guard
# ---------------------------------------------------------------------------

class TestPostOnlySafety:
    def test_buy_price_clamped_away_from_ask(self):
        s = _make_strategy(tick_size=Decimal("0.01"))
        safe = s._apply_post_only_safety(
            side=OrderSide.BUY,
            target_price=Decimal("1.01"),
            bid_price=Decimal("1.00"),
            ask_price=Decimal("1.01"),
        )
        assert safe == Decimal("0.99")

    def test_sell_price_clamped_away_from_bid(self):
        s = _make_strategy(tick_size=Decimal("0.01"))
        safe = s._apply_post_only_safety(
            side=OrderSide.SELL,
            target_price=Decimal("1.00"),
            bid_price=Decimal("1.00"),
            ask_price=Decimal("1.01"),
        )
        assert safe == Decimal("1.02")


class TestToxicityGuard:
    def test_soft_stress_widens_offset(self):
        s = _make_strategy()
        s._ob._micro_vol_bps = Decimal("9")
        s._ob._micro_drift_bps = Decimal("0")

        extra, pause = s._toxicity_adjustment()
        assert pause is None
        assert extra > 0

    def test_hard_stress_pauses_quoting(self):
        s = _make_strategy()
        s._ob._micro_vol_bps = Decimal("11")
        s._ob._micro_drift_bps = Decimal("0")

        extra, pause = s._toxicity_adjustment()
        assert extra == Decimal("0")
        assert pause == "volatility_spike"


class TestSizing:
    def test_level_size_uses_step_quantization(self):
        s = _make_strategy(
            min_order_size=Decimal("4"),
            min_order_size_step=Decimal("0.001"),
        )
        s._settings.size_scale_per_level = Decimal("1.2")
        assert s._level_size(1) == Decimal("4.800")


class TestStaleCancel:
    @pytest.mark.asyncio
    async def test_stale_book_cancels_after_grace(self):
        s = _make_strategy()
        key = (str(OrderSide.BUY), 0)
        s._level_ext_ids[key] = "ext-1"
        s._level_order_created_at[key] = time.monotonic()
        s._ob._stale = True
        s._settings.cancel_on_stale_book = True
        s._settings.stale_cancel_grace_s = 0
        s._orders.cancel_order = AsyncMock(return_value=True)
        s._orders.get_active_orders.return_value = {
            "ext-1": SimpleNamespace(price=Decimal("1.60"))
        }

        await s._maybe_reprice(OrderSide.BUY, 0)
        s._orders.cancel_order.assert_awaited_once_with("ext-1")


class TestCancelSafety:
    @pytest.mark.asyncio
    async def test_cancel_failure_does_not_replace_or_clear_slot(self):
        s = _make_strategy()
        key = (str(OrderSide.BUY), 0)
        s._level_ext_ids[key] = "ext-1"
        s._level_order_created_at[key] = time.monotonic()
        s._orders.cancel_order = AsyncMock(return_value=False)
        s._orders.place_order = AsyncMock(return_value="ext-2")
        s._orders.get_active_orders.return_value = {
            "ext-1": SimpleNamespace(
                side=OrderSide.BUY,
                price=Decimal("1.5000"),
                size=Decimal("10"),
                level=0,
            ),
        }

        await s._maybe_reprice(OrderSide.BUY, 0)

        s._orders.cancel_order.assert_awaited_once_with("ext-1")
        s._orders.place_order.assert_not_awaited()
        assert s._level_ext_ids[key] == "ext-1"


# ---------------------------------------------------------------------------
# Tests: spread EMA
# ---------------------------------------------------------------------------

class TestSpreadEma:
    def test_ema_initializes_to_first_value(self):
        ob = OrderbookManager.__new__(OrderbookManager)
        ob._last_bid = PriceLevel(price=Decimal("100"), size=Decimal("10"))
        ob._last_ask = PriceLevel(price=Decimal("101"), size=Decimal("10"))
        ob._spread_ema_bps = None
        ob._spread_ema_alpha = Decimal("0.15")
        ob._last_update_ts = time.monotonic()
        ob._staleness_threshold_s = 15.0

        ob._update_spread_ema()
        # First value: raw spread is 1/100.5 * 10000 ≈ 99.5 bps
        assert ob._spread_ema_bps is not None
        assert ob._spread_ema_bps > Decimal("90")

    def test_ema_smooths_spike(self):
        ob = OrderbookManager.__new__(OrderbookManager)
        ob._spread_ema_alpha = Decimal("0.15")
        ob._staleness_threshold_s = 15.0
        ob._last_update_ts = time.monotonic()

        # Establish baseline: 10 bps spread
        ob._last_bid = PriceLevel(price=Decimal("100.00"), size=Decimal("10"))
        ob._last_ask = PriceLevel(price=Decimal("100.10"), size=Decimal("10"))
        ob._spread_ema_bps = None
        ob._update_spread_ema()
        baseline = ob._spread_ema_bps

        # Spike to 50 bps
        ob._last_ask = PriceLevel(price=Decimal("100.50"), size=Decimal("10"))
        ob._update_spread_ema()
        after_spike = ob._spread_ema_bps

        # EMA should move toward spike but not jump all the way
        assert after_spike > baseline
        assert after_spike < Decimal("50")  # way less than the raw spike


class TestAdaptivePof:
    def test_pof_reject_increases_dynamic_ticks(self):
        s = _make_strategy()
        s._settings.adaptive_pof_enabled = True
        s._settings.post_only_safety_ticks = 2
        s._settings.pof_max_safety_ticks = 8
        s._settings.pof_backoff_multiplier = Decimal("2")
        s._settings.pof_cooldown_s = 1.0
        key = (str(OrderSide.BUY), 0)

        s._apply_adaptive_pof_reject(key)
        s._apply_adaptive_pof_reject(key)

        assert s._level_pof_streak[key] == 2
        assert s._effective_safety_ticks(key) == 4
        assert s._level_pof_until[key] > time.monotonic()


class TestImbalancePause:
    @pytest.mark.asyncio
    async def test_imbalance_pauses_sell_side_reprice(self):
        s = _make_strategy()
        s._ob._imbalance = Decimal("0.9")
        s._settings.imbalance_pause_threshold = Decimal("0.7")
        s._orders.place_order = AsyncMock(return_value="ext-1")

        await s._maybe_reprice(OrderSide.SELL, 0)
        s._orders.place_order.assert_not_awaited()


class TestInventoryBands:
    @pytest.mark.asyncio
    async def test_critical_band_blocks_inventory_increasing_side(self):
        s = _make_strategy(position=Decimal("240"), max_position_size=Decimal("300"))
        s._orders.place_order = AsyncMock(return_value="ext-1")

        await s._maybe_reprice(OrderSide.BUY, 0)
        s._orders.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_critical_band_allows_inventory_reducing_side(self):
        s = _make_strategy(position=Decimal("240"), max_position_size=Decimal("300"))
        s._orders.place_order = AsyncMock(return_value="ext-1")

        await s._maybe_reprice(OrderSide.SELL, 0)
        s._orders.place_order.assert_awaited_once()


class TestTrendOneWay:
    @pytest.mark.asyncio
    async def test_one_way_trend_blocks_counter_side_all_levels(self):
        s = _make_strategy(market_profile="crypto")
        s._settings.trend_one_way_enabled = True
        s._settings.trend_cancel_counter_on_strong = False
        s._settings.trend_strong_threshold = Decimal("0.7")
        s._orders.place_order = AsyncMock(return_value="ext-1")
        s._trend_signal = SimpleNamespace(
            evaluate=lambda: TrendState(direction="BULLISH", strength=Decimal("0.9"))
        )

        await s._maybe_reprice(OrderSide.SELL, 1)
        s._orders.place_order.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_way_trend_can_cancel_resting_counter_side(self):
        s = _make_strategy(market_profile="crypto")
        s._settings.trend_one_way_enabled = True
        s._settings.trend_cancel_counter_on_strong = True
        s._settings.trend_strong_threshold = Decimal("0.7")
        s._orders.cancel_order = AsyncMock(return_value=True)
        s._trend_signal = SimpleNamespace(
            evaluate=lambda: TrendState(direction="BULLISH", strength=Decimal("1.0"))
        )
        key = (str(OrderSide.SELL), 0)
        s._level_ext_ids[key] = "ext-1"
        s._level_order_created_at[key] = time.monotonic()
        s._orders.get_active_orders.return_value = {
            "ext-1": SimpleNamespace(
                side=OrderSide.SELL,
                price=Decimal("1.62"),
                size=Decimal("10"),
                level=0,
            )
        }

        await s._maybe_reprice(OrderSide.SELL, 0)
        s._orders.cancel_order.assert_awaited_once_with("ext-1")

    @pytest.mark.asyncio
    async def test_default_trend_behavior_still_allows_reduced_counter_side_on_l1(self):
        s = _make_strategy(market_profile="crypto")
        s._settings.trend_one_way_enabled = False
        s._settings.trend_strong_threshold = Decimal("0.7")
        s._orders.place_order = AsyncMock(return_value="ext-1")
        s._trend_signal = SimpleNamespace(
            evaluate=lambda: TrendState(direction="BULLISH", strength=Decimal("1.0"))
        )

        await s._maybe_reprice(OrderSide.SELL, 1)
        s._orders.place_order.assert_awaited_once()


class TestDrawdownStop:
    def test_drawdown_stop_triggers_shutdown_and_journal_event(self):
        s = _make_strategy(
            max_position_notional_usd=Decimal("1000"),
            market_profile="crypto",
        )
        s._settings.drawdown_stop_enabled = True
        s._settings.drawdown_stop_pct_of_max_notional = Decimal("1.5")
        s._settings.drawdown_use_high_watermark = True
        from market_maker.drawdown_stop import DrawdownStop

        s._drawdown_stop = DrawdownStop(
            enabled=s._settings.drawdown_stop_enabled,
            max_position_notional_usd=s._settings.max_position_notional_usd,
            drawdown_pct_of_max_notional=s._settings.drawdown_stop_pct_of_max_notional,
            use_high_watermark=s._settings.drawdown_use_high_watermark,
        )

        s._risk._total_pnl = Decimal("40")
        assert s._evaluate_drawdown_stop() is False
        s._risk._total_pnl = Decimal("25")
        assert s._evaluate_drawdown_stop() is True
        assert s._shutdown_event.is_set()
        assert s.shutdown_reason == "drawdown_stop"
        s._journal.record_drawdown_stop.assert_called_once()


class TestRunMetadata:
    def test_sanitized_config_redacts_secrets(self):
        from market_maker.strategy import MarketMakerStrategy

        settings = MarketMakerSettings(
            vault_id="123",
            stark_private_key="0xabc",
            stark_public_key="0xdef",
            api_key="secret",
            market_name="TEST-USD",
        )
        sanitized = MarketMakerStrategy._sanitized_run_config(settings)

        assert sanitized["vault_id"] == "***redacted***"
        assert sanitized["stark_private_key"] == "***redacted***"
        assert sanitized["stark_public_key"] == "***redacted***"
        assert sanitized["api_key"] == "***redacted***"
        assert sanitized["market_name"] == "TEST-USD"


class TestFillSnapshot:
    def test_fill_records_market_snapshot(self):
        s = _make_strategy()
        s._orders.find_order_by_exchange_id.return_value = SimpleNamespace(
            level=0, side=OrderSide.BUY
        )

        fill = SimpleNamespace(
            trade_id=1,
            order_id=11,
            side=OrderSide.BUY,
            price=Decimal("1.61"),
            qty=Decimal("10"),
            fee=Decimal("0"),
            is_taker=False,
            timestamp=0,
        )

        s._on_fill(fill)

        kwargs = s._journal.record_fill.call_args.kwargs
        assert "market_snapshot" in kwargs
        snap = kwargs["market_snapshot"]
        assert isinstance(snap, dict)
        assert snap["depth"] == s._settings.fill_snapshot_depth
        assert "bids_top" in snap and "asks_top" in snap
