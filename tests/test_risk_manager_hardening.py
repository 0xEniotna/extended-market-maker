"""Tests for risk manager hardening (prompt 1).

Covers:
1. In-flight order tracker (reserve/release notional, integrated in allowed_order_size)
2. gross_exposure_limit_usd enforcement
3. Per-side max position (max_long_position_size, max_short_position_size)
4. Orderbook mid price lookup + staleness check in allowed_order_size
5. balance_staleness_max_s guard
"""
from __future__ import annotations

import sys
import time
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# SDK module stubs
# ---------------------------------------------------------------------------

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.positions",
    "x10.perpetual.trading_client",
]

for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

_orders_mod = sys.modules["x10.perpetual.orders"]
_orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")

_positions_mod = sys.modules["x10.perpetual.positions"]
_positions_mod.PositionModel = object
_positions_mod.PositionSide = SimpleNamespace(SHORT="SHORT", LONG="LONG")
_positions_mod.PositionStatus = SimpleNamespace(CLOSED="CLOSED", OPENED="OPENED")

from market_maker.risk_manager import RiskManager  # noqa: E402

OrderSide = _orders_mod.OrderSide


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rm(
    *,
    max_position_size: Decimal = Decimal("100"),
    max_order_notional_usd: Decimal = Decimal("0"),
    max_position_notional_usd: Decimal = Decimal("0"),
    gross_exposure_limit_usd: Decimal = Decimal("0"),
    max_long_position_size: Decimal = Decimal("0"),
    max_short_position_size: Decimal = Decimal("0"),
    balance_aware_sizing_enabled: bool = False,
    balance_usage_factor: Decimal = Decimal("0.95"),
    balance_notional_multiplier: Decimal = Decimal("1.0"),
    balance_min_available_usd: Decimal = Decimal("0"),
    balance_staleness_max_s: float = 30.0,
    orderbook_mgr=None,
) -> RiskManager:
    return RiskManager(
        trading_client=MagicMock(),
        market_name="TEST-USD",
        max_position_size=max_position_size,
        max_order_notional_usd=max_order_notional_usd,
        max_position_notional_usd=max_position_notional_usd,
        gross_exposure_limit_usd=gross_exposure_limit_usd,
        max_long_position_size=max_long_position_size,
        max_short_position_size=max_short_position_size,
        balance_aware_sizing_enabled=balance_aware_sizing_enabled,
        balance_usage_factor=balance_usage_factor,
        balance_notional_multiplier=balance_notional_multiplier,
        balance_min_available_usd=balance_min_available_usd,
        balance_staleness_max_s=balance_staleness_max_s,
        orderbook_mgr=orderbook_mgr,
    )


def _make_ob_mgr(
    *,
    bid_price: Decimal = Decimal("100"),
    ask_price: Decimal = Decimal("100.10"),
    stale: bool = False,
):
    """Create a mock OrderbookManager."""
    ob = MagicMock()
    ob.is_stale = MagicMock(return_value=stale)
    if stale:
        ob.best_bid = MagicMock(return_value=None)
        ob.best_ask = MagicMock(return_value=None)
    else:
        ob.best_bid = MagicMock(
            return_value=SimpleNamespace(price=bid_price, size=Decimal("10"))
        )
        ob.best_ask = MagicMock(
            return_value=SimpleNamespace(price=ask_price, size=Decimal("10"))
        )
    return ob


# ===================================================================
# 1. In-flight order tracker
# ===================================================================

class TestInflightOrderTracker:

    def test_reserve_and_release(self):
        rm = _make_rm()
        rm.reserve_inflight("ext-1", Decimal("500"))
        rm.reserve_inflight("ext-2", Decimal("300"))
        assert rm.total_inflight_notional() == Decimal("800")

        rm.release_inflight("ext-1")
        assert rm.total_inflight_notional() == Decimal("300")

        rm.release_inflight("ext-2")
        assert rm.total_inflight_notional() == Decimal("0")

    def test_release_nonexistent_is_noop(self):
        rm = _make_rm()
        rm.release_inflight("nonexistent")  # should not raise
        assert rm.total_inflight_notional() == Decimal("0")

    def test_inflight_notional_included_in_balance_sizing(self):
        """In-flight notional should reduce balance headroom for opening orders."""
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=0,  # disable staleness
        )
        rm._cached_available_for_trade = Decimal("1000")
        rm._cached_balance_updated_at = time.monotonic()

        # Reserve 600 USD of in-flight notional
        rm.reserve_inflight("ext-1", Decimal("600"))

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("200"),
        )
        # headroom = (1000*1 - 0)*1 - 200 - 600 = 200 => max size = 200/100 = 2
        assert allowed == Decimal("2")

    def test_inflight_notional_included_in_gross_exposure(self):
        """In-flight notional should count toward gross exposure."""
        rm = _make_rm(gross_exposure_limit_usd=Decimal("5000"))
        rm._cached_position = Decimal("10")
        rm.reserve_inflight("ext-1", Decimal("2000"))

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("30"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("1000"),
        )
        # gross = |10|*100 + 1000 + 2000 = 4000
        # headroom = 5000 - 4000 = 1000 => max size = 1000/100 = 10
        assert allowed == Decimal("10")


# ===================================================================
# 2. gross_exposure_limit_usd
# ===================================================================

class TestGrossExposureLimit:

    def test_gross_exposure_clips_order_size(self):
        rm = _make_rm(gross_exposure_limit_usd=Decimal("5000"))
        rm._cached_position = Decimal("20")

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("50"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("1000"),
        )
        # gross_current = |20|*100 + 1000 = 3000
        # headroom = 5000 - 3000 = 2000 => max = 2000/100 = 20
        assert allowed == Decimal("20")

    def test_gross_exposure_zeroes_when_limit_already_breached(self):
        rm = _make_rm(gross_exposure_limit_usd=Decimal("3000"))
        rm._cached_position = Decimal("20")

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("2000"),
        )
        # gross_current = 2000 + 2000 = 4000 > 3000 => 0
        assert allowed == Decimal("0")

    def test_gross_exposure_disabled_when_zero(self):
        """gross_exposure_limit_usd=0 should not apply any limit."""
        rm = _make_rm(gross_exposure_limit_usd=Decimal("0"))
        rm._cached_position = Decimal("20")

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("80"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("80")

    def test_gross_exposure_with_short_position(self):
        rm = _make_rm(gross_exposure_limit_usd=Decimal("10000"))
        rm._cached_position = Decimal("-50")

        allowed = rm.allowed_order_size(
            side=OrderSide.SELL,
            requested_size=Decimal("100"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("0"),
        )
        # gross = |-50|*100 + 0 = 5000
        # headroom = 10000 - 5000 = 5000 => max = 50
        assert allowed == Decimal("50")


# ===================================================================
# 3. Per-side max position
# ===================================================================

class TestPerSideMaxPosition:

    def test_max_long_position_size_clips_buy(self):
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_long_position_size=Decimal("30"),
        )
        rm._cached_position = Decimal("20")

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("50"),
            reference_price=Decimal("100"),
        )
        # headroom = 30 - 20 = 10
        assert allowed == Decimal("10")

    def test_max_short_position_size_clips_sell(self):
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_short_position_size=Decimal("40"),
        )
        rm._cached_position = Decimal("-30")

        allowed = rm.allowed_order_size(
            side=OrderSide.SELL,
            requested_size=Decimal("50"),
            reference_price=Decimal("100"),
        )
        # headroom = 40 + (-30) = 10
        assert allowed == Decimal("10")

    def test_per_side_fallback_to_symmetric(self):
        """When per-side limit is 0, fall back to symmetric max_position_size."""
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_long_position_size=Decimal("0"),
            max_short_position_size=Decimal("0"),
        )
        rm._cached_position = Decimal("95")

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("5")

    def test_asymmetric_limits(self):
        """Can set different limits for long and short."""
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_long_position_size=Decimal("50"),
            max_short_position_size=Decimal("200"),
        )

        # Buy side: limited to 50
        rm._cached_position = Decimal("0")
        buy_allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("100"),
            reference_price=Decimal("10"),
        )
        assert buy_allowed == Decimal("50")

        # Sell side: limited to 200
        sell_allowed = rm.allowed_order_size(
            side=OrderSide.SELL,
            requested_size=Decimal("300"),
            reference_price=Decimal("10"),
        )
        assert sell_allowed == Decimal("200")

    def test_per_side_does_not_affect_other_side(self):
        """max_long_position_size should not affect sell orders."""
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_long_position_size=Decimal("10"),
            max_short_position_size=Decimal("0"),
        )
        rm._cached_position = Decimal("0")

        # Sell side falls back to symmetric 100
        allowed = rm.allowed_order_size(
            side=OrderSide.SELL,
            requested_size=Decimal("80"),
            reference_price=Decimal("10"),
        )
        assert allowed == Decimal("80")


# ===================================================================
# 4. Orderbook mid price + staleness check
# ===================================================================

class TestOrderbookMidPrice:

    def test_uses_orderbook_mid_instead_of_caller_price(self):
        """allowed_order_size should use orderbook mid, not caller-provided price."""
        ob = _make_ob_mgr(bid_price=Decimal("200"), ask_price=Decimal("202"))
        rm = _make_rm(
            max_order_notional_usd=Decimal("1000"),
            orderbook_mgr=ob,
        )

        # Caller passes reference_price=100 but orderbook mid = 201
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),  # should be ignored
        )
        # max_size = 1000 / 201 ≈ 4.975...
        expected_mid = (Decimal("200") + Decimal("202")) / 2
        expected_max = Decimal("1000") / expected_mid
        assert allowed == expected_max

    def test_stale_orderbook_returns_zero(self):
        """If orderbook is stale, return 0 size regardless."""
        ob = _make_ob_mgr(stale=True)
        rm = _make_rm(orderbook_mgr=ob)

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("0")

    def test_no_orderbook_mgr_uses_caller_price(self):
        """Without orderbook_mgr, fall back to caller-provided reference_price."""
        rm = _make_rm(
            max_order_notional_usd=Decimal("1000"),
            orderbook_mgr=None,
        )

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("50"),
            reference_price=Decimal("100"),
        )
        # max_size = 1000 / 100 = 10
        assert allowed == Decimal("10")

    def test_orderbook_with_no_data_falls_back(self):
        """If best_bid or best_ask is None (but not stale), use caller price."""
        ob = MagicMock()
        ob.is_stale = MagicMock(return_value=False)
        ob.best_bid = MagicMock(return_value=None)
        ob.best_ask = MagicMock(return_value=SimpleNamespace(price=Decimal("100"), size=Decimal("10")))

        rm = _make_rm(
            max_order_notional_usd=Decimal("500"),
            orderbook_mgr=ob,
        )

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("50"),
            reference_price=Decimal("50"),
        )
        # Mid not available, fallback to reference_price=50, max_size = 500/50 = 10
        assert allowed == Decimal("10")


# ===================================================================
# 5. balance_staleness_max_s
# ===================================================================

class TestBalanceStaleness:

    def test_stale_balance_skips_balance_sizing(self):
        """When balance is stale, balance-aware sizing should be skipped."""
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=30.0,
        )
        # Set available_for_trade very low — would clip to 0 if used
        rm._cached_available_for_trade = Decimal("1")
        # But make it stale (updated 60 seconds ago)
        rm._cached_balance_updated_at = time.monotonic() - 60.0

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        # Balance should be ignored because it's stale, so size is not clipped
        assert allowed == Decimal("10")

    def test_fresh_balance_is_used_for_sizing(self):
        """When balance is fresh, it should clip as usual."""
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=30.0,
        )
        rm._cached_available_for_trade = Decimal("500")
        rm._cached_balance_updated_at = time.monotonic()  # fresh

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        # headroom = 500/100 = 5
        assert allowed == Decimal("5")

    def test_no_balance_update_ever_is_treated_as_stale(self):
        """If balance was never updated (None timestamp), skip balance sizing."""
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=30.0,
        )
        rm._cached_available_for_trade = Decimal("1")
        rm._cached_balance_updated_at = None  # never updated

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        # Stale => balance sizing skipped => full size allowed
        assert allowed == Decimal("10")

    def test_staleness_disabled_when_zero(self):
        """balance_staleness_max_s=0 disables staleness checking."""
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=0,  # disabled
        )
        rm._cached_available_for_trade = Decimal("500")
        rm._cached_balance_updated_at = time.monotonic() - 9999  # very old

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        # staleness disabled, balance IS used => headroom = 500/100 = 5
        assert allowed == Decimal("5")

    def test_handle_balance_update_refreshes_timestamp(self):
        rm = _make_rm(balance_staleness_max_s=30.0)
        assert rm._cached_balance_updated_at is None

        before = time.monotonic()
        rm.handle_balance_update(
            SimpleNamespace(
                available_for_trade=Decimal("123.45"),
                equity=Decimal("150"),
                initial_margin=Decimal("12"),
            )
        )
        after = time.monotonic()

        assert rm._cached_balance_updated_at is not None
        assert before <= rm._cached_balance_updated_at <= after
        assert rm.get_available_for_trade() == Decimal("123.45")


# ===================================================================
# Backward compatibility
# ===================================================================

class TestBackwardCompatibility:
    """Existing behavior must not change when new params use defaults."""

    def test_position_size_clipping_unchanged(self):
        rm = _make_rm(max_position_size=Decimal("100"))
        rm._cached_position = Decimal("95")
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("10"),
        )
        assert allowed == Decimal("5")

    def test_order_notional_clipping_unchanged(self):
        rm = _make_rm(max_order_notional_usd=Decimal("250"))
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("5"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("2.5")

    def test_position_notional_clipping_unchanged(self):
        rm = _make_rm(max_position_notional_usd=Decimal("2500"))
        rm._cached_position = Decimal("20")
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("5")

    def test_balance_aware_sizing_unchanged(self):
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("0.9"),
            balance_notional_multiplier=Decimal("2.0"),
            balance_min_available_usd=Decimal("10"),
            balance_staleness_max_s=0,  # disabled to match old behavior
        )
        rm._cached_available_for_trade = Decimal("100")
        rm._cached_balance_updated_at = time.monotonic()
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("20"),
            reserved_open_notional_usd=Decimal("100"),
        )
        # usable_notional = (100*0.9 - 10) * 2 - 100 = 60 => size 3
        assert allowed == Decimal("3")

    def test_reserved_same_side_qty_still_works(self):
        rm = _make_rm(max_position_size=Decimal("100"))
        rm._cached_position = Decimal("70")
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("40"),
            reference_price=Decimal("10"),
            reserved_same_side_qty=Decimal("20"),
        )
        assert allowed == Decimal("10")

    def test_reducing_order_bypasses_balance_clip(self):
        rm = _make_rm(
            balance_aware_sizing_enabled=True,
            balance_usage_factor=Decimal("1"),
            balance_notional_multiplier=Decimal("1"),
            balance_min_available_usd=Decimal("0"),
            balance_staleness_max_s=0,
        )
        rm._cached_position = Decimal("-68")
        rm._cached_available_for_trade = Decimal("1500")
        rm._cached_balance_updated_at = time.monotonic()
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("20"),
            reference_price=Decimal("100"),
            reserved_open_notional_usd=Decimal("14000"),
        )
        assert allowed == Decimal("20")

    def test_default_constructor_still_works(self):
        """RiskManager with only original params should construct fine."""
        rm = RiskManager(
            trading_client=MagicMock(),
            market_name="TEST-USD",
            max_position_size=Decimal("50"),
        )
        assert rm._max_position_size == Decimal("50")
        assert rm._gross_exposure_limit_usd == Decimal("0")
        assert rm._max_long_position_size == Decimal("0")
        assert rm._max_short_position_size == Decimal("0")
        assert rm._balance_staleness_max_s == 30.0
        assert rm._orderbook_mgr is None


# ===================================================================
# Combined scenarios
# ===================================================================

class TestCombinedScenarios:

    def test_all_limits_interact_correctly(self):
        """When multiple limits are active, the tightest one wins."""
        ob = _make_ob_mgr(bid_price=Decimal("99"), ask_price=Decimal("101"))
        rm = _make_rm(
            max_position_size=Decimal("100"),
            max_long_position_size=Decimal("50"),
            max_order_notional_usd=Decimal("2000"),
            max_position_notional_usd=Decimal("4000"),
            gross_exposure_limit_usd=Decimal("6000"),
            orderbook_mgr=ob,
        )
        rm._cached_position = Decimal("30")
        rm.reserve_inflight("ext-1", Decimal("1000"))

        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("100"),
            reference_price=Decimal("999"),  # should be ignored; mid=100
            reserved_open_notional_usd=Decimal("500"),
        )
        # mid = 100
        # per-side headroom: 50 - 30 = 20
        # per-order notional: 2000/100 = 20
        # position notional: (4000 - 3000 - 0) / 100 = 10
        # gross: (3000 + 500 + 1000) = 4500, headroom 1500 => 15
        # tightest = 10
        assert allowed == Decimal("10")

    def test_orderbook_stale_overrides_everything(self):
        """Stale orderbook should return 0 even if other limits are generous."""
        ob = _make_ob_mgr(stale=True)
        rm = _make_rm(
            max_position_size=Decimal("1000"),
            orderbook_mgr=ob,
        )
        allowed = rm.allowed_order_size(
            side=OrderSide.BUY,
            requested_size=Decimal("10"),
            reference_price=Decimal("100"),
        )
        assert allowed == Decimal("0")


# ===================================================================
# 6. Session P&L tracking (survives position close/reset)
# ===================================================================

PositionStatus = _positions_mod.PositionStatus
PositionSide = _positions_mod.PositionSide


class TestSessionPnL:

    def test_session_pnl_starts_at_zero(self):
        rm = _make_rm()
        assert rm.get_session_pnl() == Decimal("0")

    def test_session_pnl_reflects_current_position_pnl(self):
        rm = _make_rm()
        rm._update_position_pnl(realized=Decimal("10"), unrealized=Decimal("5"))
        assert rm.get_session_pnl() == Decimal("15")

    def test_session_pnl_accumulates_across_position_resets(self):
        """Realized P&L must survive _reset_position_pnl (position close)."""
        rm = _make_rm()
        # Position 1: realized +25
        rm._update_position_pnl(realized=Decimal("25"), unrealized=Decimal("0"))
        rm._reset_position_pnl()
        # After reset: session_realized = 25, position = 0
        assert rm.get_session_pnl() == Decimal("25")

        # Position 2: realized +10, unrealized +3
        rm._update_position_pnl(realized=Decimal("10"), unrealized=Decimal("3"))
        assert rm.get_session_pnl() == Decimal("38")  # 25 + 10 + 3

        # Position 2 closes
        rm._reset_position_pnl()
        assert rm.get_session_pnl() == Decimal("35")  # 25 + 10

    def test_session_pnl_accumulates_losses(self):
        """Negative realized P&L accumulates correctly."""
        rm = _make_rm()
        # Position 1: loss of -15
        rm._update_position_pnl(realized=Decimal("-15"), unrealized=Decimal("0"))
        rm._reset_position_pnl()
        assert rm.get_session_pnl() == Decimal("-15")

        # Position 2: loss of -8
        rm._update_position_pnl(realized=Decimal("-8"), unrealized=Decimal("0"))
        rm._reset_position_pnl()
        assert rm.get_session_pnl() == Decimal("-23")

    def test_session_pnl_mixed_wins_and_losses(self):
        """Multiple position lifecycles with mixed outcomes."""
        rm = _make_rm()
        # Win: +50
        rm._update_position_pnl(realized=Decimal("50"), unrealized=Decimal("0"))
        rm._reset_position_pnl()
        # Loss: -30
        rm._update_position_pnl(realized=Decimal("-30"), unrealized=Decimal("0"))
        rm._reset_position_pnl()
        # Current unrealized: -5
        rm._update_position_pnl(realized=Decimal("0"), unrealized=Decimal("-5"))

        # Session = 50 + (-30) + 0 + (-5) = 15
        assert rm.get_session_pnl() == Decimal("15")

    def test_position_total_pnl_still_resets(self):
        """get_position_total_pnl must still zero on reset (backward compat)."""
        rm = _make_rm()
        rm._update_position_pnl(realized=Decimal("50"), unrealized=Decimal("10"))
        assert rm.get_position_total_pnl() == Decimal("60")
        rm._reset_position_pnl()
        assert rm.get_position_total_pnl() == Decimal("0")
