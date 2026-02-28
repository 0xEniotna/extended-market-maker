"""Tests for Phase 4 institutional features."""
from __future__ import annotations

import time
from decimal import Decimal

import pytest

from market_maker.pnl_attribution import PnLAttributionTracker
from market_maker.quote_trade_ratio import QuoteTradeRatioTracker
from market_maker.latency_monitor import LatencyMonitor
from market_maker.config_rollback import ConfigRollbackWatchdog, PerformanceBaseline


# ===================================================================
# 1. P&L Attribution
# ===================================================================

class TestPnLAttribution:

    def test_empty_snapshot(self):
        t = PnLAttributionTracker()
        snap = t.snapshot()
        assert snap.fill_count == 0
        assert snap.total_usd == Decimal("0")

    def test_single_buy_fill_with_mid(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="BUY",
            price=Decimal("100"),
            qty=Decimal("10"),
            fee=Decimal("0.5"),
            is_taker=False,
            mid_price=Decimal("100.50"),
        )
        snap = t.snapshot(current_mid=Decimal("100.50"))
        assert snap.fill_count == 1
        assert snap.buy_fill_count == 1
        assert snap.maker_fill_count == 1
        # Spread capture: bought at 100 when mid=100.50, edge = 0.50/100.50 * 10000 ≈ 49.75 bps
        assert snap.spread_capture_usd > 0
        # Fee P&L: -0.5
        assert snap.fee_pnl_usd == Decimal("-0.5")

    def test_sell_fill_spread_capture(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="SELL",
            price=Decimal("101"),
            qty=Decimal("10"),
            fee=Decimal("0.5"),
            is_taker=False,
            mid_price=Decimal("100.50"),
        )
        snap = t.snapshot(current_mid=Decimal("100.50"))
        assert snap.sell_fill_count == 1
        # Sold at 101 when mid=100.50, edge = 0.50/100.50 * 10000 ≈ 49.75 bps
        assert snap.spread_capture_usd > 0

    def test_inventory_pnl_mark_to_market(self):
        t = PnLAttributionTracker()
        # Buy 10 at 100
        t.record_fill(
            side="BUY",
            price=Decimal("100"),
            qty=Decimal("10"),
            fee=Decimal("0"),
            is_taker=False,
            mid_price=Decimal("100"),
        )
        # Price rises to 105 — inventory P&L should be positive
        snap = t.snapshot(current_mid=Decimal("105"))
        assert snap.inventory_pnl_usd == Decimal("50")  # 10 * (105-100)

    def test_inventory_pnl_loss(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="BUY",
            price=Decimal("100"),
            qty=Decimal("10"),
            fee=Decimal("0"),
            is_taker=False,
            mid_price=Decimal("100"),
        )
        snap = t.snapshot(current_mid=Decimal("95"))
        assert snap.inventory_pnl_usd == Decimal("-50")  # 10 * (95-100)

    def test_funding_pnl(self):
        t = PnLAttributionTracker()
        t.record_funding_payment(Decimal("5"))
        t.record_funding_payment(Decimal("-2"))
        snap = t.snapshot()
        assert snap.funding_pnl_usd == Decimal("3")

    def test_total_combines_all_components(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="BUY",
            price=Decimal("100"),
            qty=Decimal("10"),
            fee=Decimal("1"),
            is_taker=False,
            mid_price=Decimal("100.50"),
        )
        t.record_funding_payment(Decimal("2"))
        snap = t.snapshot(current_mid=Decimal("101"))
        assert snap.total_usd == (
            snap.spread_capture_usd
            + snap.inventory_pnl_usd
            + snap.fee_pnl_usd
            + snap.funding_pnl_usd
        )

    def test_taker_fill_counted(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="BUY",
            price=Decimal("100"),
            qty=Decimal("1"),
            fee=Decimal("0.1"),
            is_taker=True,
            mid_price=Decimal("100"),
        )
        snap = t.snapshot()
        assert snap.taker_fill_count == 1
        assert snap.maker_fill_count == 0

    def test_volume_tracking(self):
        t = PnLAttributionTracker()
        t.record_fill(
            side="BUY",
            price=Decimal("50"),
            qty=Decimal("20"),
            fee=Decimal("0"),
            is_taker=False,
            mid_price=Decimal("50"),
        )
        snap = t.snapshot()
        assert snap.total_volume_usd == Decimal("1000")


# ===================================================================
# 2. Quote-to-Trade Ratio
# ===================================================================

class TestQuoteTradeRatio:

    def test_empty_state(self):
        qtr = QuoteTradeRatioTracker()
        snap = qtr.evaluate()
        assert snap.ratio == Decimal("0")
        assert not snap.warn_active
        assert not snap.critical_active

    def test_quotes_only_infinity(self):
        qtr = QuoteTradeRatioTracker()
        qtr.record_quote()
        qtr.record_quote()
        snap = qtr.evaluate()
        assert snap.quotes_in_window == 2
        assert snap.fills_in_window == 0
        assert snap.ratio == Decimal("Infinity")

    def test_normal_ratio(self):
        qtr = QuoteTradeRatioTracker(warn_threshold=10.0)
        for _ in range(20):
            qtr.record_quote()
        for _ in range(5):
            qtr.record_fill()
        snap = qtr.evaluate()
        assert snap.ratio == Decimal("4")
        assert not snap.warn_active

    def test_warn_threshold(self):
        qtr = QuoteTradeRatioTracker(warn_threshold=5.0, critical_threshold=20.0)
        for _ in range(30):
            qtr.record_quote()
        for _ in range(5):
            qtr.record_fill()
        snap = qtr.evaluate()
        assert snap.ratio == Decimal("6")
        assert snap.warn_active
        assert not snap.critical_active

    def test_critical_threshold(self):
        qtr = QuoteTradeRatioTracker(warn_threshold=5.0, critical_threshold=10.0)
        for _ in range(50):
            qtr.record_quote()
        for _ in range(3):
            qtr.record_fill()
        snap = qtr.evaluate()
        assert snap.critical_active

    def test_is_critical_property(self):
        qtr = QuoteTradeRatioTracker(warn_threshold=2.0, critical_threshold=5.0)
        for _ in range(30):
            qtr.record_quote()
        qtr.record_fill()
        qtr.evaluate()
        assert qtr.is_critical


# ===================================================================
# 3. Latency Monitor
# ===================================================================

class TestLatencyMonitor:

    def test_empty_state(self):
        m = LatencyMonitor()
        snap = m.evaluate()
        assert snap.avg_latency_ms == 0.0
        assert not snap.degraded
        assert not snap.halt_quoting

    def test_normal_latency(self):
        m = LatencyMonitor(warn_ms=200.0, critical_ms=1000.0)
        for _ in range(10):
            m.record_latency(50.0)
        snap = m.evaluate()
        assert snap.avg_latency_ms == 50.0
        assert not snap.degraded

    def test_degraded_state(self):
        m = LatencyMonitor(warn_ms=100.0, critical_ms=500.0)
        for _ in range(20):
            m.record_latency(200.0)
        snap = m.evaluate()
        assert snap.degraded
        assert not snap.halt_quoting
        assert snap.extra_offset_bps > Decimal("0")

    def test_halt_state(self):
        m = LatencyMonitor(warn_ms=100.0, critical_ms=500.0)
        for _ in range(20):
            m.record_latency(600.0)
        snap = m.evaluate()
        assert snap.halt_quoting
        assert m.should_halt

    def test_extra_offset_scales_with_latency(self):
        m = LatencyMonitor(
            warn_ms=100.0,
            critical_ms=1000.0,
            extra_offset_bps_per_ms=Decimal("0.01"),
        )
        # p95 at 200ms → excess = 100ms → offset = 1.0 bps
        for _ in range(20):
            m.record_latency(200.0)
        snap = m.evaluate()
        assert snap.extra_offset_bps == Decimal("1.00") or snap.extra_offset_bps == Decimal("1.0")

    def test_max_extra_offset_capped(self):
        m = LatencyMonitor(
            warn_ms=100.0,
            critical_ms=2000.0,
            extra_offset_bps_per_ms=Decimal("0.1"),
            max_extra_offset_bps=Decimal("3"),
        )
        # p95 at 500ms → excess = 400ms → offset would be 40 but capped at 3
        for _ in range(20):
            m.record_latency(500.0)
        snap = m.evaluate()
        assert snap.extra_offset_bps == Decimal("3")


# ===================================================================
# 4. Config Rollback Watchdog
# ===================================================================

class TestConfigRollback:

    def test_no_pending_returns_empty(self):
        w = ConfigRollbackWatchdog()
        decision = w.evaluate()
        assert not decision.should_rollback

    def test_mark_config_good(self):
        w = ConfigRollbackWatchdog()
        w.mark_config_good({"key": "value"})
        assert w.last_known_good_config == {"key": "value"}

    def test_rollback_on_markout_degradation(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,  # immediate evaluation
            markout_degradation_bps=Decimal("1"),
            min_fills=2,
        )
        w.mark_config_good({"spread": "1.0"})
        baseline = PerformanceBaseline(
            captured_at=time.monotonic(),
            avg_markout_1s_bps=Decimal("5"),
            fill_count=0,
        )
        w.on_config_change({"spread": "1.0"}, {"spread": "0.5"}, baseline)

        decision = w.evaluate(
            avg_markout_1s_bps=Decimal("2"),  # degraded by 3 bps
            fill_count=10,  # enough fills
        )
        assert decision.should_rollback
        assert "markout_1s_delta_bps" in decision.degraded_metrics

    def test_no_rollback_when_improving(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,
            markout_degradation_bps=Decimal("1"),
            min_fills=2,
        )
        w.mark_config_good({"spread": "1.0"})
        baseline = PerformanceBaseline(
            captured_at=time.monotonic(),
            avg_markout_1s_bps=Decimal("5"),
            fill_count=0,
        )
        w.on_config_change({"spread": "1.0"}, {"spread": "0.8"}, baseline)

        decision = w.evaluate(
            avg_markout_1s_bps=Decimal("6"),  # improved
            fill_count=10,
        )
        assert not decision.should_rollback

    def test_not_enough_fills_defers(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,
            min_fills=10,
        )
        w.mark_config_good({"a": "1"})
        baseline = PerformanceBaseline(
            captured_at=time.monotonic(),
            avg_markout_1s_bps=Decimal("5"),
            fill_count=0,
        )
        w.on_config_change({"a": "1"}, {"a": "2"}, baseline)

        decision = w.evaluate(
            avg_markout_1s_bps=Decimal("-10"),  # terrible, but not enough fills
            fill_count=3,
        )
        assert not decision.should_rollback
        assert w.has_pending_evaluation

    def test_rollback_on_pnl_degradation(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,
            markout_degradation_bps=Decimal("100"),  # won't trigger
            pnl_degradation_usd=Decimal("10"),
            min_fills=1,
        )
        w.mark_config_good({"x": "1"})
        baseline = PerformanceBaseline(
            captured_at=time.monotonic(),
            session_pnl_usd=Decimal("100"),
            fill_count=0,
        )
        w.on_config_change({"x": "1"}, {"x": "2"}, baseline)

        decision = w.evaluate(
            session_pnl_usd=Decimal("80"),  # lost $20
            fill_count=5,
        )
        assert decision.should_rollback
        assert "pnl_delta_usd" in decision.degraded_metrics

    def test_rollback_count_increments(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,
            pnl_degradation_usd=Decimal("1"),
            min_fills=1,
        )
        w.mark_config_good({"a": "1"})

        for i in range(3):
            baseline = PerformanceBaseline(
                captured_at=time.monotonic(),
                session_pnl_usd=Decimal("100"),
                fill_count=i * 10,
            )
            w.on_config_change({"a": "1"}, {"a": str(i)}, baseline)
            w.evaluate(
                session_pnl_usd=Decimal("90"),
                fill_count=(i + 1) * 10,
            )

        assert w.rollback_count == 3

    def test_good_config_promoted_after_evaluation(self):
        w = ConfigRollbackWatchdog(
            evaluation_window_s=0.0,
            markout_degradation_bps=Decimal("1"),
            min_fills=1,
        )
        w.mark_config_good({"version": "1"})
        baseline = PerformanceBaseline(
            captured_at=time.monotonic(),
            avg_markout_1s_bps=Decimal("5"),
            fill_count=0,
        )
        w.on_config_change({"version": "1"}, {"version": "2"}, baseline)
        w.evaluate(
            avg_markout_1s_bps=Decimal("6"),  # improved
            fill_count=10,
        )
        # The new config should now be known-good
        assert w.last_known_good_config == {"version": "2"}
        assert not w.has_pending_evaluation
