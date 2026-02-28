"""Tests for Phase 3 code quality refactors."""
from __future__ import annotations

from decimal import Decimal
from typing import Set
from unittest.mock import MagicMock

import pytest

from market_maker.quote_halt_manager import QuoteHaltManager

# ===================================================================
# 1. QuoteHaltManager
# ===================================================================

class _FakeJournal:
    def __init__(self):
        self.events = []

    def record_exchange_event(self, *, event_type: str, details: dict):
        self.events.append({"event_type": event_type, "details": details})


class _FakeMetrics:
    def __init__(self):
        self.halt_state: Set[str] = set()
        self.margin_breached = False

    def set_quote_halt_state(self, reasons: Set[str]):
        self.halt_state = set(reasons)

    def set_margin_guard_breached(self, breached: bool):
        self.margin_breached = breached


def _make_halt_mgr():
    journal = _FakeJournal()
    metrics = _FakeMetrics()
    mgr = QuoteHaltManager(market_name="ETH-USD", journal=journal, metrics=metrics)
    return mgr, journal, metrics


class TestQuoteHaltManager:

    def test_initial_state_not_halted(self):
        mgr, _, _ = _make_halt_mgr()
        assert not mgr.is_halted
        assert len(mgr.reasons) == 0

    def test_set_halt_adds_reason(self):
        mgr, journal, metrics = _make_halt_mgr()
        mgr.set_halt("margin_guard")
        assert mgr.is_halted
        assert "margin_guard" in mgr.reasons
        assert len(journal.events) == 1
        assert journal.events[0]["event_type"] == "quote_halt"

    def test_set_halt_duplicate_no_op(self):
        mgr, journal, _ = _make_halt_mgr()
        mgr.set_halt("margin_guard")
        mgr.set_halt("margin_guard")
        assert len(journal.events) == 1  # Only one event

    def test_clear_halt_removes_reason(self):
        mgr, journal, _ = _make_halt_mgr()
        mgr.set_halt("margin_guard")
        mgr.clear_halt("margin_guard")
        assert not mgr.is_halted
        assert len(journal.events) == 2
        assert journal.events[1]["event_type"] == "quote_halt_cleared"

    def test_clear_halt_nonexistent_no_op(self):
        mgr, journal, _ = _make_halt_mgr()
        mgr.clear_halt("nonexistent")
        assert len(journal.events) == 0

    def test_multiple_reasons_require_all_cleared(self):
        mgr, _, _ = _make_halt_mgr()
        mgr.set_halt("margin_guard")
        mgr.set_halt("stream_desync")
        assert mgr.is_halted
        mgr.clear_halt("margin_guard")
        assert mgr.is_halted  # still halted — stream_desync remains
        mgr.clear_halt("stream_desync")
        assert not mgr.is_halted

    def test_sync_state_rate_limit(self):
        mgr, _, _ = _make_halt_mgr()
        mgr.sync_state(rate_limit_halt=True, streams_healthy=True)
        assert "rate_limit_halt" in mgr.reasons
        mgr.sync_state(rate_limit_halt=False, streams_healthy=True)
        assert "rate_limit_halt" not in mgr.reasons

    def test_sync_state_stream_desync(self):
        mgr, _, _ = _make_halt_mgr()
        mgr.sync_state(rate_limit_halt=False, streams_healthy=False)
        assert "stream_desync" in mgr.reasons
        mgr.sync_state(rate_limit_halt=False, streams_healthy=True)
        assert "stream_desync" not in mgr.reasons

    def test_sync_state_margin_breach_tracking(self):
        mgr, _, metrics = _make_halt_mgr()
        mgr.margin_breach_since = 12345.0
        mgr.sync_state(rate_limit_halt=False, streams_healthy=True)
        assert metrics.margin_breached is True

        mgr.margin_breach_since = None
        mgr.sync_state(rate_limit_halt=False, streams_healthy=True)
        assert metrics.margin_breached is False

    def test_check_streams_healthy_both_ok(self):
        account = MagicMock()
        account.is_sequence_healthy.return_value = True
        ob = MagicMock()
        ob.is_sequence_healthy.return_value = True
        ob.has_data.return_value = True
        assert QuoteHaltManager.check_streams_healthy(account, ob) is True

    def test_check_streams_healthy_account_unhealthy(self):
        account = MagicMock()
        account.is_sequence_healthy.return_value = False
        ob = MagicMock()
        ob.is_sequence_healthy.return_value = True
        ob.has_data.return_value = True
        assert QuoteHaltManager.check_streams_healthy(account, ob) is False

    def test_check_streams_healthy_no_data(self):
        account = MagicMock()
        account.is_sequence_healthy.return_value = True
        ob = MagicMock()
        ob.is_sequence_healthy.return_value = True
        ob.has_data.return_value = False
        assert QuoteHaltManager.check_streams_healthy(account, ob) is False

    def test_check_streams_healthy_no_attrs(self):
        """Objects without health-check methods are assumed healthy."""
        account = object()
        ob = object()
        assert QuoteHaltManager.check_streams_healthy(account, ob) is True


# ===================================================================
# 2. Pricing engine float conversion
# ===================================================================

from market_maker.pricing_engine import PricingEngine  # noqa: E402


class _FakeRisk:
    def __init__(self, position=Decimal("0")):
        self._position = position

    def get_current_position(self):
        return self._position


class _FakeOb:
    def __init__(self, spread_bps=None, bid=None, ask=None):
        self._spread = spread_bps
        self._bid = bid
        self._ask = ask

    def spread_bps_ema(self):
        return self._spread

    def best_bid(self):
        return self._bid

    def best_ask(self):
        return self._ask

    def micro_volatility_bps(self, window_s):
        return None

    def micro_drift_bps(self, window_s):
        return None

    def mid_prices(self, window_s):
        return []


class _FakeLevel:
    def __init__(self, price, size=Decimal("1")):
        self.price = Decimal(str(price))
        self.size = size


class _FakeSettings:
    offset_mode = "dynamic"
    spread_multiplier = Decimal("1.5")
    min_offset_bps = Decimal("3")
    max_offset_bps = Decimal("100")
    price_offset_per_level_percent = Decimal("0.3")
    max_position_size = Decimal("100")
    inventory_hard_pct = Decimal("0.95")
    inventory_critical_pct = Decimal("0.8")
    inventory_warn_pct = Decimal("0.5")
    inventory_deadband_pct = Decimal("0.10")
    skew_shape_k = Decimal("2.0")
    skew_max_bps = Decimal("20")
    inventory_skew_factor = Decimal("0.5")
    trend_skew_boost = Decimal("1.5")
    market_profile = "crypto"
    size_scale_per_level = Decimal("1.0")


class TestPricingEngineFloat:
    """Verify pricing engine produces correct results after float conversion."""

    def test_compute_offset_dynamic_mode(self):
        ob = _FakeOb(spread_bps=Decimal("10"))
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=ob,
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        offset = pe.compute_offset(0, Decimal("1000"))
        # level 0: spread_bps(10) * multiplier(1.5) * (0+1) = 15 bps
        # floor: 3 * 1 = 3, ceiling: 100 * 1 = 100 → 15 bps
        # offset = 1000 * 15 / 10000 = 1.50
        assert offset == Decimal("1.5")

    def test_compute_target_price_buy(self):
        bid = _FakeLevel(999)
        ask = _FakeLevel(1001)
        ob = _FakeOb(spread_bps=Decimal("10"), bid=bid, ask=ask)
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=ob,
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        price = pe.compute_target_price("BUY", 0, Decimal("1000"))
        # Buy: raw = best_price - offset - skew - funding
        # With zero position → skew ≈ 0, no funding → raw = 1000 - 1.5 = 998.5
        assert price < Decimal("1000")
        assert price > Decimal("990")

    def test_compute_target_price_sell(self):
        bid = _FakeLevel(999)
        ask = _FakeLevel(1001)
        ob = _FakeOb(spread_bps=Decimal("10"), bid=bid, ask=ask)
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=ob,
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        price = pe.compute_target_price("SELL", 0, Decimal("1000"))
        assert price > Decimal("1000")

    def test_skew_component_long_inventory(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(position=Decimal("50")),  # 50% of max
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        skew = pe._skew_component()
        # Long 50% → positive skew → shift quotes down
        assert skew > 0

    def test_skew_component_short_inventory(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(position=Decimal("-50")),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        skew = pe._skew_component()
        assert skew < 0

    def test_skew_within_deadband_is_zero(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(position=Decimal("5")),  # 5% of max → within 10% deadband
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        skew = pe._skew_component()
        assert skew == Decimal("0")

    def test_round_to_tick_buy_rounds_down(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        result = pe.round_to_tick(Decimal("100.555"), "BUY")
        assert result == Decimal("100.55")

    def test_round_to_tick_sell_rounds_up(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        result = pe.round_to_tick(Decimal("100.551"), "SELL")
        assert result == Decimal("100.56")

    def test_theoretical_edge_bps_buy(self):
        pe = PricingEngine(
            settings=_FakeSettings(),
            orderbook_mgr=_FakeOb(),
            risk_mgr=_FakeRisk(),
            tick_size=Decimal("0.01"),
            base_order_size=Decimal("1"),
            min_order_size_step=Decimal("0.01"),
        )
        # Buy at 999, best = 1000 → edge = 10 bps
        edge = pe.theoretical_edge_bps("BUY", Decimal("999"), Decimal("1000"))
        assert edge == Decimal("10")


# ===================================================================
# 3. Policy generation (generate_policy.py)
# ===================================================================

from market_maker.config import MarketMakerSettings  # noqa: E402


class TestConfigGrouping:
    """Test config sub-grouping and field metadata."""

    def test_all_fields_have_descriptions(self):
        for name, field_info in MarketMakerSettings.model_fields.items():
            assert field_info.description, f"Field {name} missing description"

    def test_env_prefix_is_mm(self):
        assert MarketMakerSettings.model_config.get("env_prefix") == "MM_"

    def test_all_fields_are_grouped(self):
        groups = MarketMakerSettings.field_groups()
        grouped = set()
        for fields in groups.values():
            grouped |= fields
        model_fields = set(MarketMakerSettings.model_fields)
        ungrouped = model_fields - grouped
        assert not ungrouped, f"Ungrouped fields: {ungrouped}"

    def test_no_field_in_multiple_groups(self):
        groups = MarketMakerSettings.field_groups()
        seen: dict = {}
        for group, fields in groups.items():
            for f in fields:
                assert f not in seen, f"Field {f} in both {seen[f]} and {group}"
                seen[f] = group

    def test_field_metadata_has_all_fields(self):
        meta = MarketMakerSettings.field_metadata()
        meta_names = {m["name"] for m in meta}
        model_names = set(MarketMakerSettings.model_fields)
        assert meta_names == model_names

    def test_field_metadata_includes_env_var(self):
        meta = MarketMakerSettings.field_metadata()
        for entry in meta:
            assert entry["env_var"].startswith("MM_")

    def test_field_metadata_includes_group(self):
        meta = MarketMakerSettings.field_metadata()
        valid_groups = set(MarketMakerSettings.field_groups()) | {"ungrouped"}
        for entry in meta:
            assert entry["group"] in valid_groups, (
                f"Field {entry['name']} has invalid group {entry['group']}"
            )

    def test_bounds_json_fields_are_subset_of_model(self):
        import json
        from pathlib import Path
        bounds_path = Path(__file__).resolve().parents[1] / "mm_config" / "policy" / "bounds.json"
        if not bounds_path.exists():
            pytest.skip("bounds.json not found")
        with open(bounds_path) as f:
            bounds = json.load(f)
        prefix = "MM_"
        model_env_names = set()
        for name in MarketMakerSettings.model_fields:
            model_env_names.add(f"{prefix}{name.upper()}")
        # Keys not in the MM pydantic model (external or deprecated).
        non_model_keys = {"EXTENDED_ENV", "MM_IMBALANCE_PAUSE_SIDE"}
        for key in bounds:
            if key in non_model_keys:
                continue
            assert key in model_env_names, (
                f"bounds.json key {key} not found in MarketMakerSettings fields"
            )


# ===================================================================
# 4. FundingManager
# ===================================================================

from market_maker.funding_manager import FundingManager  # noqa: E402


class TestFundingManager:

    def test_initial_rate_is_zero(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        assert fm.funding_rate == Decimal("0")

    def test_set_funding_rate_valid(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.0005"))
        assert fm.funding_rate == Decimal("0.0005")

    def test_set_funding_rate_invalid_ignored(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        fm.set_funding_rate("not-a-number")
        assert fm.funding_rate == Decimal("0.001")  # Unchanged

    def test_set_funding_rate_nan_ignored(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        fm.set_funding_rate(Decimal("NaN"))
        assert fm.funding_rate == Decimal("0.001")

    def test_funding_bias_non_crypto_returns_zero(self):
        fm = FundingManager(
            market_profile="equity",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        assert fm.funding_bias_bps() == Decimal("0")

    def test_funding_bias_disabled_returns_zero(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=False,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        assert fm.funding_bias_bps() == Decimal("0")

    def test_funding_bias_computation(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        # 0.001 * 10000 * 1 = 10 bps
        assert fm.funding_bias_bps() == Decimal("10")

    def test_funding_bias_capped(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("2"),
            funding_bias_cap_bps=Decimal("5"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        # 0.001 * 10000 * 2 = 20 bps, cap = 5
        assert fm.funding_bias_bps() == Decimal("5")

    def test_update_settings(self):
        fm = FundingManager(
            market_profile="crypto",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        fm.set_funding_rate(Decimal("0.001"))
        fm.update_settings(
            market_profile="equity",
            funding_bias_enabled=True,
            funding_inventory_weight=Decimal("1"),
            funding_bias_cap_bps=Decimal("50"),
        )
        assert fm.funding_bias_bps() == Decimal("0")  # Equity → 0


# ===================================================================
# 5. RiskWatchdog
# ===================================================================

from market_maker.risk_watchdog import RiskWatchdog  # noqa: E402


class _FakeDrawdownState:
    triggered = False
    current_pnl = Decimal("0")
    peak_pnl = Decimal("0")
    drawdown = Decimal("0")
    threshold_usd = Decimal("0")


class _FakeDrawdownStop:
    def __init__(self, should_trigger=False):
        self._trigger = should_trigger

    def evaluate(self, pnl):
        state = _FakeDrawdownState()
        state.triggered = self._trigger
        return state


class _MockRiskWatchdogSettings:
    market_name = "ETH-USD"
    margin_guard_enabled = True
    min_available_balance_for_trading = Decimal("10")
    min_available_balance_ratio = Decimal("0")
    max_margin_utilization = Decimal("0")
    min_liq_distance_bps = Decimal("0")
    margin_guard_shutdown_breach_s = Decimal("0")
    circuit_breaker_max_failures = 5
    circuit_breaker_cooldown_s = Decimal("30")
    failure_window_s = Decimal("60")
    failure_rate_trip = Decimal("0.5")
    min_attempts_for_breaker = 3
    max_order_age_s = Decimal("120")


class _MockRiskMgr:
    def __init__(self, position=Decimal("0")):
        self._position = position

    def get_current_position(self):
        return self._position

    def get_session_pnl(self):
        return Decimal("0")

    def margin_snapshot(self):
        return {"available_for_trade": Decimal("100")}


class TestRiskWatchdog:

    def _make_watchdog(self, *, trigger_drawdown=False):
        settings = _MockRiskWatchdogSettings()
        risk = _MockRiskMgr()
        orders = MagicMock()
        halt_mgr = MagicMock()
        halt_mgr.margin_breach_since = None
        metrics = MagicMock()
        journal = MagicMock()
        journal.make_stack_trace_hash = MagicMock(return_value="abc")
        post_only = MagicMock()
        drawdown = _FakeDrawdownStop(should_trigger=trigger_drawdown)
        shutdown_fn = MagicMock()
        wd = RiskWatchdog(
            settings=settings,
            risk_mgr=risk,
            orders=orders,
            halt_mgr=halt_mgr,
            metrics=metrics,
            journal=journal,
            post_only=post_only,
            drawdown_stop=drawdown,
            request_shutdown_fn=shutdown_fn,
        )
        return wd, shutdown_fn, halt_mgr

    def test_circuit_open_initially_false(self):
        wd, _, _ = self._make_watchdog()
        assert wd.circuit_open is False

    def test_evaluate_drawdown_stop_no_trigger(self):
        wd, shutdown_fn, _ = self._make_watchdog(trigger_drawdown=False)
        assert wd.evaluate_drawdown_stop() is False
        shutdown_fn.assert_not_called()

    def test_evaluate_drawdown_stop_triggers(self):
        wd, shutdown_fn, _ = self._make_watchdog(trigger_drawdown=True)
        assert wd.evaluate_drawdown_stop() is True
        shutdown_fn.assert_called_once_with("drawdown_stop")

    def test_margin_guard_breach_no_breach(self):
        wd, _, _ = self._make_watchdog()
        breached, reasons, snap = wd.margin_guard_breach()
        assert breached is False
        assert len(reasons) == 0

    def test_margin_guard_breach_low_balance(self):
        wd, _, _ = self._make_watchdog()
        wd._risk.margin_snapshot = lambda: {"available_for_trade": Decimal("5")}
        breached, reasons, snap = wd.margin_guard_breach()
        assert breached is True
        assert "available_for_trade" in reasons

    def test_margin_guard_breach_missing_balance(self):
        wd, _, _ = self._make_watchdog()
        wd._risk.margin_snapshot = lambda: {"available_for_trade": None}
        breached, reasons, snap = wd.margin_guard_breach()
        assert breached is True
        assert "available_for_trade_missing" in reasons


# ===================================================================
# 6. Shared utils
# ===================================================================

from market_maker.utils import safe_decimal, safe_float  # noqa: E402


class TestSharedUtils:

    def test_safe_decimal_normal(self):
        assert safe_decimal("123.45") == Decimal("123.45")

    def test_safe_decimal_from_int(self):
        assert safe_decimal(42) == Decimal("42")

    def test_safe_decimal_from_float(self):
        result = safe_decimal(1.5)
        assert result == Decimal("1.5")

    def test_safe_decimal_invalid_returns_default(self):
        assert safe_decimal("not-a-number") == Decimal("0")

    def test_safe_decimal_none_returns_default(self):
        assert safe_decimal(None) == Decimal("0")

    def test_safe_decimal_custom_default(self):
        assert safe_decimal("bad", default="99") == Decimal("99")

    def test_safe_float_normal(self):
        assert safe_float("42.5") == 42.5

    def test_safe_float_invalid_returns_default(self):
        assert safe_float("bad") == 0.0

    def test_safe_float_none_returns_default(self):
        assert safe_float(None) == 0.0

    def test_safe_float_custom_default(self):
        assert safe_float("bad", default=-1.0) == -1.0

    def test_pricing_engine_uses_shared_helpers(self):
        """Verify PricingEngine._to_decimal delegates to safe_decimal."""
        result = PricingEngine._to_decimal("42.5")
        assert result == Decimal("42.5")
        assert PricingEngine._to_decimal("bad") == Decimal("0")

    def test_pricing_engine_uses_shared_float_helper(self):
        """Verify PricingEngine._to_float delegates to safe_float."""
        result = PricingEngine._to_float("42.5")
        assert result == 42.5
        assert PricingEngine._to_float("bad") == 0.0


# ===================================================================
# 7. StrategyFactory (requires x10 SDK stubs)
# ===================================================================

import sys  # noqa: E402

# StrategyFactory transitively imports x10 SDK types. If the SDK is not
# installed (CI / lightweight test env), skip the factory tests.
_x10_available = "x10.perpetual.trading_client" in sys.modules or True
try:
    from market_maker.strategy_factory import StrategyFactory
except ImportError:
    _x10_available = False
    StrategyFactory = None  # type: ignore[assignment, misc]


@pytest.mark.skipif(not _x10_available, reason="x10 SDK not available")
class TestStrategyFactory:

    def test_factory_creates_with_settings(self):
        settings = MagicMock()
        factory = StrategyFactory(settings)
        assert factory._settings is settings
