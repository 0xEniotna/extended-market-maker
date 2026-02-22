"""
Market Maker Strategy

Main orchestration for the market making bot.  Spawns one async task per
(level, side) pair.  Each task waits for a price change on the relevant
side of the book, computes the target price, and reprices its order when
the current order drifts outside a tolerance band.

Features:
- Inventory-skewed pricing (Avellaneda-Stoikov style)
- Real-time fill/order tracking via account WebSocket stream
- Atomic cancel-and-replace via ``previous_order_id``
- Minimum spread check to avoid quoting inside the fee
- Circuit breaker on consecutive order-placement failures
- Staleness detection on orderbook data

Entry point: ``MarketMakerStrategy.run()``
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from collections import deque
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Optional, Set

from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager, FillEvent
from .config import ENV_FILE, MarketMakerSettings
from .decision_models import RegimeState, RepriceMarketContext, TrendState
from .drawdown_stop import DrawdownStop
from .fill_quality import FillQualityTracker
from .guard_policy import GuardPolicy
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .post_only_safety import PostOnlySafety
from .pricing_engine import PricingEngine
from .reprice_pipeline import RepricePipeline
from .risk_manager import RiskManager
from .strategy_callbacks import on_fill, on_level_freed
from .strategy_runner import run_strategy
from .trade_journal import TradeJournal
from .trend_signal import TrendSignal
from .volatility_regime import VolatilityRegime

logger = logging.getLogger(__name__)

# Refresh the exchange position every N seconds as a safety net.
# The account stream handles real-time updates; this is a fallback.
_POSITION_REFRESH_INTERVAL_S = 30.0
_FUNDING_REFRESH_INTERVAL_S = 300.0
_BALANCE_REFRESH_INTERVAL_S = 10.0
_DRAWDOWN_CHECK_INTERVAL_S = 1.0
_MARGIN_GUARD_CHECK_INTERVAL_S = 1.0
_KPI_WATCHDOG_INTERVAL_S = 5.0

# Default circuit-breaker settings
_CIRCUIT_BREAKER_MAX_FAILURES = 5
_CIRCUIT_BREAKER_COOLDOWN_S = 30.0


class MarketMakerStrategy:
    """
    Market making strategy that quotes on multiple price levels per side.
    """

    def __init__(
        self,
        settings: MarketMakerSettings,
        trading_client: PerpetualTradingClient,
        orderbook_mgr: OrderbookManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
        account_stream: AccountStreamManager,
        metrics: MetricsCollector,
        journal: TradeJournal,
        tick_size: Decimal,
        base_order_size: Decimal,
        market_min_order_size: Decimal,
        min_order_size_step: Decimal,
    ) -> None:
        self._settings = settings
        self._client = trading_client
        self._ob = orderbook_mgr
        self._orders = order_mgr
        self._risk = risk_mgr
        self._account_stream = account_stream
        self._metrics = metrics
        self._journal = journal
        self._tick_size = tick_size
        self._base_order_size = base_order_size
        self._market_min_order_size = market_min_order_size
        self._min_order_size_step = min_order_size_step

        # Keyed by (side_name, level)
        self._level_ext_ids: Dict[tuple[str, int], Optional[str]] = {}
        self._level_order_created_at: Dict[tuple[str, int], Optional[float]] = {}
        # Reprice rate limiting: monotonic timestamp of last reprice per slot
        self._level_last_reprice_at: Dict[tuple[str, int], float] = {}
        # Stale-book grace tracking by slot
        self._level_stale_since: Dict[tuple[str, int], Optional[float]] = {}
        # Cancel barrier by slot: blocks re-placement until WS terminal update arrives.
        self._level_cancel_pending_ext_id: Dict[tuple[str, int], Optional[str]] = {}
        # Cancel reason tracking for journaling once cancellation is confirmed
        self._pending_cancel_reasons: Dict[str, str] = {}

        # Fill deduplication: ordered deque + fast lookup set, capped at 10k entries.
        self._seen_trade_ids: deque = deque()
        self._seen_trade_ids_set: Set[int] = set()

        # Circuit-breaker / halt state
        self._circuit_open = False
        self._quote_halt_reasons: Set[str] = set()
        self._margin_breach_since: Optional[float] = None
        self._runtime_mode: str = "normal"
        self._last_taker_leakage_warn_at: float = 0.0

        self._rebuild_components()

        self._funding_rate = Decimal("0")
        self._shutdown_reason = "shutdown"

        self._shutdown_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Class-level entry point
    # ------------------------------------------------------------------

    @classmethod
    async def run(cls) -> None:
        await run_strategy(cls)

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_signal(self) -> None:
        logger.info("Signal received, initiating shutdown")
        self._request_shutdown("shutdown")

    def _handle_reload(self) -> None:
        """SIGHUP handler — reload config from environment / .env file."""
        try:
            old_config = self._sanitized_run_config(self._settings)
            new_settings = MarketMakerSettings()
            new_config = self._sanitized_run_config(new_settings)

            # Compute diff: keys where values changed.
            diff: Dict[str, Any] = {}
            all_keys = set(old_config) | set(new_config)
            for key in all_keys:
                old_val = old_config.get(key)
                new_val = new_config.get(key)
                if str(old_val) != str(new_val):
                    diff[key] = {"before": old_val, "after": new_val}

            self._settings = new_settings
            self._rebuild_components()

            # Journal the config change event.
            if hasattr(self, "_journal") and self._journal is not None:
                self._journal.record_config_change(
                    before=old_config,
                    after=new_config,
                    diff=diff,
                )

            logger.info(
                "Config reloaded: offset_mode=%s skew=%.2f spread_min=%s levels=%d max_age=%ss diff_keys=%s",
                new_settings.offset_mode.value,
                new_settings.inventory_skew_factor,
                new_settings.min_spread_bps,
                new_settings.num_price_levels,
                new_settings.max_order_age_s,
                list(diff.keys()) if diff else "none",
            )
        except Exception as exc:
            logger.error("Config reload failed: %s", exc)
            if hasattr(self, "_journal") and self._journal is not None:
                self._journal.record_error(
                    component="config_reload",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                    stack_trace=TradeJournal.format_stack_trace(exc),
                )

    @property
    def shutdown_reason(self) -> str:
        return self._shutdown_reason

    @staticmethod
    def _sanitized_run_config(settings: MarketMakerSettings) -> Dict[str, Any]:
        """Return a config snapshot safe to persist in journals."""
        data: Dict[str, Any] = settings.model_dump(mode="python")
        redact_keys = {
            "vault_id",
            "stark_private_key",
            "stark_public_key",
            "api_key",
        }
        for key in redact_keys:
            if key in data:
                data[key] = "***redacted***"
        return data

    @staticmethod
    def _run_provenance() -> Dict[str, Any]:
        """Best-effort runtime provenance for reproducibility."""
        provenance: Dict[str, Any] = {
            "env_file": str(ENV_FILE),
            "cwd": os.getcwd(),
        }
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
            sha = res.stdout.strip()
            if res.returncode == 0 and sha:
                provenance["git_sha"] = sha
        except Exception:
            # Keep provenance optional and never break strategy startup.
            pass
        return provenance

    # ------------------------------------------------------------------
    # Fill callback (from account stream → trade journal)
    # ------------------------------------------------------------------

    def _on_fill(self, fill: FillEvent) -> None:
        on_fill(self, fill)

    # ------------------------------------------------------------------
    # Level-freed callback (from account stream → order manager)
    # ------------------------------------------------------------------

    def _on_level_freed(
        self,
        side_value: str,
        level: int,
        external_id: str,
        *,
        rejected: bool = False,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        price: Optional[Decimal] = None,
    ) -> None:
        on_level_freed(
            self,
            side_value=side_value,
            level=level,
            external_id=external_id,
            rejected=rejected,
            status=status,
            reason=reason,
            price=price,
        )

    # ------------------------------------------------------------------
    # Per-level task
    # ------------------------------------------------------------------

    async def _level_task(self, side: OrderSide, level: int) -> None:
        """Continuously quote on one (side, level) slot."""
        key = (str(side), level)
        self._clear_level_slot(key)

        condition = (
            self._ob.best_bid_condition
            if side == OrderSide.BUY
            else self._ob.best_ask_condition
        )

        while not self._shutdown_event.is_set():
            self._sync_quote_halt_state()
            # Pause while circuit breaker is open
            if self._circuit_open:
                await asyncio.sleep(1.0)
                continue
            if self._quote_halt_reasons:
                await asyncio.sleep(0.2)
                continue
            if self._level_cancel_pending_ext_id.get(key) is not None:
                await asyncio.sleep(0.1)
                continue

            try:
                # Wait for a price change on this side of the book
                async with condition:
                    await asyncio.wait_for(condition.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass  # Re-evaluate anyway every few seconds
            except asyncio.CancelledError:
                return

            try:
                await self._maybe_reprice(side, level)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error(
                    "Error in level task %s L%d: %s", side, level, exc,
                    exc_info=True,
                )
                self._journal.record_error(
                    component=f"level_task_{side}_L{level}",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                    stack_trace=TradeJournal.format_stack_trace(exc),
                )
                await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Repricing logic
    # ------------------------------------------------------------------

    async def _maybe_reprice(self, side: OrderSide, level: int) -> None:
        self._sync_quote_halt_state()
        if self._quote_halt_reasons:
            return
        market_ctx = self._build_reprice_market_context()
        await self._reprice.evaluate(self, side, level, market_ctx=market_ctx)

    def _build_reprice_market_context(self) -> RepriceMarketContext:
        if self._settings.market_profile == "crypto":
            regime = self._volatility.evaluate()
            trend = self._trend_signal.evaluate()
        else:
            regime = RegimeState(regime="NORMAL")
            trend = TrendState()
        min_interval, max_order_age_s = self._volatility.cadence(regime)
        rate_limit_multiplier = getattr(self._orders, "rate_limit_reprice_multiplier", Decimal("1"))
        if not isinstance(rate_limit_multiplier, Decimal):
            rate_limit_multiplier = Decimal("1")
        min_interval *= float(rate_limit_multiplier)
        return RepriceMarketContext(
            regime=regime,
            trend=trend,
            min_reprice_interval_s=min_interval,
            max_order_age_s=max_order_age_s,
            funding_bias_bps=self._funding_bias_bps(),
            inventory_band=self._pricing.inventory_band(),
        )

    def _clear_level_slot(self, key: tuple[str, int]) -> None:
        """Clear tracking for one (side, level) slot."""
        self._level_ext_ids[key] = None
        self._level_order_created_at[key] = None
        self._level_stale_since[key] = None
        self._level_cancel_pending_ext_id[key] = None

    def _record_reprice_decision(
        self,
        *,
        side: OrderSide,
        level: int,
        reason: str,
        current_best: Optional[Decimal] = None,
        prev_price: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
        spread_bps: Optional[Decimal] = None,
        extra_offset_bps: Optional[Decimal] = None,
        regime: Optional[str] = None,
        trend_direction: Optional[str] = None,
        trend_strength: Optional[Decimal] = None,
        inventory_band: Optional[str] = None,
        funding_bias_bps: Optional[Decimal] = None,
    ) -> None:
        if not self._settings.journal_reprice_decisions:
            return
        self._journal.record_reprice_decision(
            side=self._normalise_side(str(side)),
            level=level,
            reason=reason,
            current_best=current_best,
            prev_price=prev_price,
            target_price=target_price,
            spread_bps=spread_bps,
            extra_offset_bps=extra_offset_bps,
            regime=regime,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            inventory_band=inventory_band,
            funding_bias_bps=funding_bias_bps,
        )

    async def _cancel_level_order(
        self,
        *,
        key: tuple[str, int],
        external_id: str,
        side: OrderSide,
        level: int,
        reason: str,
    ) -> bool:
        """Request cancel for a level order and store a structured reason.

        Returns True when the level slot can be safely freed:
        - order already reached a terminal state.
        A successful cancel request does not free the slot immediately;
        replacement waits for WS terminal confirmation.
        """
        _ = (side, level)
        pending_ext = self._level_cancel_pending_ext_id.get(key)
        if pending_ext == external_id:
            return False
        self._pending_cancel_reasons[external_id] = reason
        ok = await self._orders.cancel_order(external_id)
        if ok:
            self._level_cancel_pending_ext_id[key] = external_id
            return False

        # If the order is no longer active, treat it as terminal and free the slot.
        if self._orders.find_order_by_external_id(external_id) is not None:
            if self._orders.get_active_order(external_id) is None:
                self._clear_level_slot(key)
                return True

        self._pending_cancel_reasons.pop(external_id, None)
        return False

    def _quantize_size(self, size: Decimal) -> Decimal:
        """Quantize order size to market step size."""
        if size <= 0:
            return Decimal("0")
        if self._min_order_size_step <= 0:
            return size
        return (size / self._min_order_size_step).quantize(
            Decimal("1"), rounding=ROUND_DOWN
        ) * self._min_order_size_step

    def _effective_safety_ticks(self, key: tuple[str, int]) -> int:
        avg_latency_ms = self._orders.avg_placement_latency_ms()
        tick_time_ms = self._estimate_tick_time_ms()
        return self._post_only.effective_ticks(
            key, avg_latency_ms=avg_latency_ms, tick_time_ms=tick_time_ms,
        )

    def _estimate_tick_time_ms(self) -> float:
        """Estimate how many ms it takes for the market to move one tick.

        Uses recent micro-volatility to derive price speed, then converts
        tick_size into a time estimate.  Returns 0 if data is insufficient
        (which disables the latency-tick buffer).
        """
        window_s = self._settings.micro_vol_window_s
        if window_s <= 0:
            return 0.0
        vol_bps = self._ob.micro_volatility_bps(window_s)
        if vol_bps is None or vol_bps <= 0:
            return 0.0
        bid = self._ob.best_bid()
        ask = self._ob.best_ask()
        if bid is None or ask is None:
            return 0.0
        bp = getattr(bid, "price", None)
        ap = getattr(ask, "price", None)
        if bp is None or ap is None or bp <= 0 or ap <= 0:
            return 0.0
        mid = (bp + ap) / 2
        tick_bps = float(self._tick_size / mid * Decimal("10000"))
        if tick_bps <= 0:
            return 0.0
        # vol_bps is measured over window_s seconds
        vol_per_ms = float(vol_bps) / (window_s * 1000.0)
        if vol_per_ms <= 0:
            return 0.0
        return tick_bps / vol_per_ms

    def _on_adverse_markout_widen(self, key: tuple[str, int], reason: str) -> None:
        """Callback from FillQualityTracker when a level has adverse markout."""
        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        max_ticks = max(base_ticks, int(self._settings.pof_max_safety_ticks))
        current = self._post_only.dynamic_safety_ticks.get(key, base_ticks)
        new_ticks = min(max_ticks, current + 1)
        self._post_only.dynamic_safety_ticks[key] = new_ticks
        logger.warning(
            "Adverse markout widen for %s: safety_ticks %d -> %d (reason=%s)",
            key, current, new_ticks, reason,
        )

    def _apply_adaptive_pof_reject(self, key: tuple[str, int]) -> None:
        self._post_only.on_rejection(key)

    def _on_successful_quote(self, key: tuple[str, int]) -> None:
        self._post_only.on_success(key)

    def _reset_pof_state(self, key: tuple[str, int]) -> None:
        self._post_only.reset(key)

    def _apply_post_only_safety(
        self,
        *,
        side: OrderSide,
        target_price: Decimal,
        bid_price: Decimal,
        ask_price: Decimal,
        safety_ticks: Optional[int] = None,
    ) -> Optional[Decimal]:
        return self._post_only.clamp_price(
            side=self._normalise_side(str(side)),
            target_price=target_price,
            bid_price=bid_price,
            ask_price=ask_price,
            safety_ticks=safety_ticks,
        )

    def _toxicity_adjustment(self) -> tuple[Decimal, Optional[str]]:
        """Return (extra_offset_bps, pause_reason) from microstructure stress."""
        if self._settings.market_profile == "crypto":
            regime = self._volatility.evaluate()
            if regime.pause:
                return Decimal("0"), "volatility_spike"
            extra_bps = max(
                Decimal("0"),
                (regime.offset_scale - Decimal("1")) * self._settings.min_offset_bps,
            )
            return extra_bps, None

        vol_bps = self._ob.micro_volatility_bps(self._settings.micro_vol_window_s)
        drift_bps = self._ob.micro_drift_bps(self._settings.micro_drift_window_s)
        drift_abs = abs(drift_bps) if drift_bps is not None else None

        vol_limit = self._settings.micro_vol_max_bps
        drift_limit = self._settings.micro_drift_max_bps

        # Hard pause on severe micro-regime stress.
        if (
            vol_bps is not None
            and vol_limit > 0
            and vol_bps > (vol_limit * Decimal("1.25"))
        ):
            return Decimal("0"), "volatility_spike"
        if (
            drift_abs is not None
            and drift_limit > 0
            and drift_abs > (drift_limit * Decimal("1.25"))
        ):
            return Decimal("0"), "drift_spike"

        # Moderate widening above soft thresholds.
        extra_bps = Decimal("0")
        if vol_bps is not None and vol_limit > 0 and vol_bps > vol_limit:
            extra_bps += (vol_bps - vol_limit) * self._settings.volatility_offset_multiplier
        if drift_abs is not None and drift_limit > 0 and drift_abs > drift_limit:
            extra_bps += (
                drift_abs - drift_limit
            ) * self._settings.volatility_offset_multiplier
        return max(Decimal("0"), extra_bps), None

    @staticmethod
    def _normalise_side(side_value: str) -> str:
        side_upper = side_value.upper()
        if "BUY" in side_upper:
            return "BUY"
        if "SELL" in side_upper:
            return "SELL"
        return side_value

    def _increases_inventory(self, side: OrderSide) -> bool:
        side_name = self._normalise_side(str(side))
        pos = self._risk.get_current_position()
        if side_name == "BUY":
            return pos >= 0
        if side_name == "SELL":
            return pos <= 0
        return False

    def _set_funding_rate(self, funding_rate: Decimal) -> None:
        try:
            value = Decimal(str(funding_rate))
        except Exception:
            return
        if not value.is_finite():
            return
        self._funding_rate = value

    def _funding_bias_bps(self) -> Decimal:
        if self._settings.market_profile != "crypto":
            return Decimal("0")
        if not self._settings.funding_bias_enabled:
            return Decimal("0")
        raw_bps = (
            self._funding_rate
            * Decimal("10000")
            * self._settings.funding_inventory_weight
        )
        cap = max(Decimal("0"), self._settings.funding_bias_cap_bps)
        if cap <= 0:
            return raw_bps
        return max(-cap, min(cap, raw_bps))

    @staticmethod
    def _counter_trend_side(trend: TrendState) -> Optional[str]:
        if trend.direction == "BULLISH":
            return "SELL"
        if trend.direction == "BEARISH":
            return "BUY"
        return None

    def _is_strong_counter_trend_side(self, side_name: str, trend: TrendState) -> bool:
        if self._settings.market_profile != "crypto":
            return False
        if not self._settings.trend_one_way_enabled:
            return False
        if trend.strength < self._settings.trend_strong_threshold:
            return False
        counter_side = self._counter_trend_side(trend)
        return counter_side is not None and side_name == counter_side

    def _request_shutdown(self, reason: str) -> None:
        if self._shutdown_event.is_set():
            return
        self._set_runtime_mode("shutdown")
        self._shutdown_reason = reason
        self._shutdown_event.set()

    def _set_runtime_mode(self, mode: str) -> None:
        self._runtime_mode = str(mode)

    def _is_normal_quoting_mode(self) -> bool:
        return (
            self._runtime_mode == "normal"
            and not self._shutdown_event.is_set()
            and not self._quote_halt_reasons
        )

    def _set_quote_halt(self, reason: str) -> None:
        if reason in self._quote_halt_reasons:
            return
        self._quote_halt_reasons.add(reason)
        logger.warning(
            "Quote halt engaged for %s: reasons=%s",
            self._settings.market_name,
            sorted(self._quote_halt_reasons),
        )
        self._journal.record_exchange_event(
            event_type="quote_halt",
            details={"reason": reason, "reasons": sorted(self._quote_halt_reasons)},
        )
        self._metrics.set_quote_halt_state(self._quote_halt_reasons)

    def _clear_quote_halt(self, reason: str) -> None:
        if reason not in self._quote_halt_reasons:
            return
        self._quote_halt_reasons.discard(reason)
        logger.info(
            "Quote halt reason cleared for %s: %s (remaining=%s)",
            self._settings.market_name,
            reason,
            sorted(self._quote_halt_reasons),
        )
        self._journal.record_exchange_event(
            event_type="quote_halt_cleared",
            details={"reason": reason, "remaining": sorted(self._quote_halt_reasons)},
        )
        self._metrics.set_quote_halt_state(self._quote_halt_reasons)

    def _streams_healthy(self) -> bool:
        account_ok = True
        if hasattr(self._account_stream, "is_sequence_healthy"):
            account_state = self._account_stream.is_sequence_healthy()
            account_ok = account_state if isinstance(account_state, bool) else True
        book_ok = True
        if hasattr(self._ob, "is_sequence_healthy"):
            book_state = self._ob.is_sequence_healthy()
            book_ok = book_state if isinstance(book_state, bool) else True
        has_data_fn = getattr(self._ob, "has_data", None)
        has_data = has_data_fn() if callable(has_data_fn) else True
        return account_ok and book_ok and has_data

    async def _on_stream_desync(self, reason: str) -> None:
        self._set_quote_halt("stream_desync")
        self._journal.record_exchange_event(
            event_type="stream_desync",
            details={"reason": reason},
        )
        if self._orders.active_order_count() > 0:
            try:
                await self._orders.cancel_all_orders()
            except Exception:
                logger.debug("stream desync cancel-all failed", exc_info=True)

    def _sync_quote_halt_state(self) -> None:
        rate_limit_halt = getattr(self._orders, "in_rate_limit_halt", False)
        if not isinstance(rate_limit_halt, bool):
            rate_limit_halt = False
        if rate_limit_halt:
            self._set_quote_halt("rate_limit_halt")
        else:
            self._clear_quote_halt("rate_limit_halt")

        if self._streams_healthy():
            self._clear_quote_halt("stream_desync")
        else:
            self._set_quote_halt("stream_desync")
        self._metrics.set_quote_halt_state(self._quote_halt_reasons)
        self._metrics.set_margin_guard_breached(self._margin_breach_since is not None)

    def _rebuild_components(self) -> None:
        quote_anchor = str(getattr(self._settings.quote_anchor, "value", self._settings.quote_anchor)).lower()
        markout_anchor = str(
            getattr(self._settings.markout_anchor, "value", self._settings.markout_anchor)
        ).lower()
        if quote_anchor != "mid" or markout_anchor != "mid":
            raise ValueError(
                "This rollout locks quote_anchor and markout_anchor to 'mid' for coherence."
            )

        self._pricing = PricingEngine(
            settings=self._settings,
            orderbook_mgr=self._ob,
            risk_mgr=self._risk,
            tick_size=self._tick_size,
            base_order_size=self._base_order_size,
            min_order_size_step=self._min_order_size_step,
        )
        self._post_only = PostOnlySafety(
            settings=self._settings,
            tick_size=self._tick_size,
            round_to_tick=self._pricing.round_to_tick,
        )
        self._volatility = VolatilityRegime(self._settings, self._ob)
        self._trend_signal = TrendSignal(self._settings, self._ob)
        self._guards = GuardPolicy(self._settings)
        self._reprice = RepricePipeline(self._settings, self._tick_size, self._pricing)
        self._drawdown_stop = DrawdownStop(
            enabled=self._settings.drawdown_stop_enabled,
            max_position_notional_usd=self._settings.max_position_notional_usd,
            drawdown_pct_of_max_notional=self._settings.drawdown_stop_pct_of_max_notional,
            use_high_watermark=self._settings.drawdown_use_high_watermark,
        )

        # Fill quality tracker for markout analysis and auto-widening.
        self._fill_quality = FillQualityTracker(self._ob)
        self._fill_quality.set_min_acceptable_markout_bps(
            self._settings.min_acceptable_markout_bps,
        )
        self._fill_quality.set_offset_widen_callback(self._on_adverse_markout_widen)

        # Wire optional trackers into metrics.
        self._metrics.set_fill_quality_tracker(self._fill_quality)
        self._metrics.set_post_only_safety(self._post_only)
        self._metrics.set_quote_halt_state(self._quote_halt_reasons)
        self._metrics.set_margin_guard_breached(self._margin_breach_since is not None)

        # Keep legacy attribute names as aliases for one release cycle.
        self._level_pof_until = self._post_only.pof_until
        self._level_pof_streak = self._post_only.pof_streak
        self._level_pof_last_ts = self._post_only.pof_last_ts
        self._level_dynamic_safety_ticks = self._post_only.dynamic_safety_ticks
        self._level_imbalance_pause_until = self._guards._level_imbalance_pause_until

    def _evaluate_drawdown_stop(self) -> bool:
        state = self._drawdown_stop.evaluate(self._risk.get_position_total_pnl())
        if not state.triggered:
            return False

        action = "cancel_all_flatten_terminate"
        logger.error(
            "DRAWDOWN STOP TRIGGERED: market=%s current_pnl=%s peak_pnl=%s drawdown=%s "
            "threshold=%s action=%s",
            self._settings.market_name,
            state.current_pnl,
            state.peak_pnl,
            state.drawdown,
            state.threshold_usd,
            action,
        )
        self._journal.record_drawdown_stop(
            current_pnl=state.current_pnl,
            peak_pnl=state.peak_pnl,
            drawdown=state.drawdown,
            threshold_usd=state.threshold_usd,
            action=action,
        )
        self._request_shutdown("drawdown_stop")
        return True

    async def _drawdown_watchdog_task(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                if self._evaluate_drawdown_stop():
                    return
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Drawdown watchdog error", exc_info=True)
                self._journal.record_error(
                    component="drawdown_watchdog",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                )
            await asyncio.sleep(_DRAWDOWN_CHECK_INTERVAL_S)

    async def _funding_refresh_task(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                if (
                    self._settings.market_profile == "crypto"
                    and self._settings.funding_bias_enabled
                ):
                    markets = await self._client.markets_info.get_markets_dict()
                    market_info = markets.get(self._settings.market_name)
                    if market_info is not None:
                        self._set_funding_rate(
                            Decimal(str(market_info.market_stats.funding_rate))
                        )
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Funding refresh failed", exc_info=True)
            await asyncio.sleep(_FUNDING_REFRESH_INTERVAL_S)

    async def _balance_refresh_task(self) -> None:
        """Periodically refresh available-for-trade collateral headroom."""
        while not self._shutdown_event.is_set():
            try:
                await self._risk.refresh_balance()
            except asyncio.CancelledError:
                return
            except Exception:
                logger.debug("Balance refresh failed", exc_info=True)
            await asyncio.sleep(_BALANCE_REFRESH_INTERVAL_S)

    def _margin_guard_breach(self) -> tuple[bool, list[str], dict[str, Optional[Decimal]]]:
        snapshot = self._risk.margin_snapshot()
        reasons: list[str] = []

        available = snapshot.get("available_for_trade")
        equity = snapshot.get("equity")
        initial_margin = snapshot.get("initial_margin")
        available_ratio = snapshot.get("available_ratio")
        margin_utilization = snapshot.get("margin_utilization")
        liq_distance_bps = snapshot.get("liq_distance_bps")
        current_position = self._risk.get_current_position()

        if available is None:
            reasons.append("available_for_trade_missing")
        elif available < self._settings.min_available_balance_for_trading:
            reasons.append("available_for_trade")

        if self._settings.min_available_balance_ratio > 0:
            if available_ratio is None or equity is None or equity <= 0:
                reasons.append("available_ratio_missing")
            elif available_ratio < self._settings.min_available_balance_ratio:
                reasons.append("available_ratio")

        if self._settings.max_margin_utilization > 0:
            if margin_utilization is None or initial_margin is None or equity is None or equity <= 0:
                reasons.append("margin_utilization_missing")
            elif margin_utilization > self._settings.max_margin_utilization:
                reasons.append("margin_utilization")

        # Liquidation distance is only relevant while we hold risk.
        if current_position != 0 and self._settings.min_liq_distance_bps > 0:
            if liq_distance_bps is None:
                reasons.append("liq_distance_missing")
            elif liq_distance_bps < self._settings.min_liq_distance_bps:
                reasons.append("liq_distance")

        return bool(reasons), reasons, snapshot

    async def _margin_guard_task(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                if not self._settings.margin_guard_enabled:
                    await asyncio.sleep(_MARGIN_GUARD_CHECK_INTERVAL_S)
                    continue

                breached, reasons, snapshot = self._margin_guard_breach()
                now = time.monotonic()
                if breached:
                    self._metrics.set_margin_guard_breached(True)
                    self._set_quote_halt("margin_guard")
                    if self._margin_breach_since is None:
                        self._margin_breach_since = now
                        self._journal.record_exchange_event(
                            event_type="margin_guard_breach",
                            details={
                                "reasons": reasons,
                                "snapshot": snapshot,
                            },
                        )
                        if self._orders.active_order_count() > 0:
                            await self._orders.cancel_all_orders()
                    breach_elapsed = now - self._margin_breach_since
                    if (
                        self._settings.margin_guard_shutdown_breach_s > 0
                        and breach_elapsed >= self._settings.margin_guard_shutdown_breach_s
                    ):
                        self._journal.record_exchange_event(
                            event_type="margin_guard_shutdown",
                            details={
                                "reasons": reasons,
                                "breach_elapsed_s": breach_elapsed,
                                "snapshot": snapshot,
                            },
                        )
                        self._request_shutdown("margin_guard")
                        return
                else:
                    if self._margin_breach_since is not None:
                        self._journal.record_exchange_event(
                            event_type="margin_guard_cleared",
                            details={"snapshot": snapshot},
                        )
                    self._margin_breach_since = None
                    self._metrics.set_margin_guard_breached(False)
                    self._clear_quote_halt("margin_guard")
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Margin guard task error: %s", exc, exc_info=True)
                self._journal.record_error(
                    component="margin_guard",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                )
            await asyncio.sleep(_MARGIN_GUARD_CHECK_INTERVAL_S)

    async def _kpi_watchdog_task(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                self._sync_quote_halt_state()

                taker_count = (
                    int(self._account_stream.taker_fill_count_1m())
                    if hasattr(self._account_stream, "taker_fill_count_1m")
                    else 0
                )
                taker_ratio = (
                    Decimal(str(self._account_stream.taker_fill_notional_ratio_1m()))
                    if hasattr(self._account_stream, "taker_fill_notional_ratio_1m")
                    else Decimal("0")
                )

                now = time.monotonic()
                if (
                    taker_count > 0
                    and self._is_normal_quoting_mode()
                    and (now - self._last_taker_leakage_warn_at) >= 30.0
                ):
                    self._last_taker_leakage_warn_at = now
                    logger.warning(
                        "Taker leakage detected for %s: count_1m=%d notional_ratio_1m=%.2f%%",
                        self._settings.market_name,
                        taker_count,
                        float(taker_ratio * Decimal("100")),
                    )
                    self._journal.record_exchange_event(
                        event_type="taker_leakage_warning",
                        details={
                            "count_1m": taker_count,
                            "notional_ratio_1m": taker_ratio,
                            "runtime_mode": self._runtime_mode,
                        },
                    )
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("KPI watchdog task error: %s", exc, exc_info=True)
            await asyncio.sleep(_KPI_WATCHDOG_INTERVAL_S)

    def _order_age_exceeded(
        self,
        key: tuple[str, int],
        *,
        max_age_s: Optional[float] = None,
    ) -> bool:
        """Return True if the tracked order at *key* exceeded max_order_age_s."""
        if max_age_s is None:
            max_age_s = self._settings.max_order_age_s
        if max_age_s <= 0:
            return False
        placed_ts = self._level_order_created_at.get(key)
        if placed_ts is None:
            return False
        return (time.monotonic() - placed_ts) > max_age_s

    def _compute_offset(
        self,
        level: int,
        best_price: Decimal,
        *,
        regime_scale: Decimal = Decimal("1"),
    ) -> Decimal:
        return self._pricing.compute_offset(
            level,
            best_price,
            regime_scale=regime_scale,
        )

    def _compute_target_price(
        self,
        side: OrderSide,
        level: int,
        best_price: Decimal,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend: Optional[TrendState] = None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> Decimal:
        return self._pricing.compute_target_price(
            side,
            level,
            best_price,
            extra_offset_bps=extra_offset_bps,
            regime_scale=regime_scale,
            trend=trend,
            funding_bias_bps=funding_bias_bps,
        )

    def _round_to_tick(
        self, price: Decimal, side: Optional[OrderSide] = None
    ) -> Decimal:
        return self._pricing.round_to_tick(price, side)

    def _level_size(self, level: int) -> Decimal:
        return self._pricing.level_size(level)

    def _needs_reprice(
        self,
        side: OrderSide,
        prev_price: Decimal,
        current_best: Decimal,
        level: int,
        *,
        extra_offset_bps: Decimal = Decimal("0"),
        regime_scale: Decimal = Decimal("1"),
        trend: Optional[TrendState] = None,
        funding_bias_bps: Decimal = Decimal("0"),
    ) -> tuple[bool, str]:
        return self._reprice.needs_reprice(
            side,
            prev_price,
            current_best,
            level,
            extra_offset_bps=extra_offset_bps,
            regime_scale=regime_scale,
            trend=trend,
            funding_bias_bps=funding_bias_bps,
        )

    def _theoretical_edge_bps(
        self,
        side: OrderSide,
        quote_price: Decimal,
        current_best: Decimal,
    ) -> Decimal:
        return self._pricing.theoretical_edge_bps(side, quote_price, current_best)

    # ------------------------------------------------------------------
    # Background position refresh (safety-net)
    # ------------------------------------------------------------------

    async def _position_refresh_task(self) -> None:
        """Periodically refresh the cached position from the exchange.

        This is a safety net — the account stream provides real-time updates.
        """
        while not self._shutdown_event.is_set():
            try:
                await self._risk.refresh_position()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Position refresh error: %s", exc)
                self._journal.record_error(
                    component="position_refresh",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                )
            await asyncio.sleep(_POSITION_REFRESH_INTERVAL_S)

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    async def _circuit_breaker_task(self) -> None:
        """Monitor consecutive failures, sweep pending cancels, and detect zombies."""
        max_failures = self._settings.circuit_breaker_max_failures
        cooldown = self._settings.circuit_breaker_cooldown_s
        failure_window_s = self._settings.failure_window_s
        failure_rate_trip = float(self._settings.failure_rate_trip)
        min_attempts = self._settings.min_attempts_for_breaker

        while not self._shutdown_event.is_set():
            try:
                # --- Sweep pending-cancel orders that timed out ---
                self._orders.sweep_pending_cancels()

                # --- Zombie order detection ---
                zombie_threshold_s = self._settings.max_order_age_s * 2
                if zombie_threshold_s > 0:
                    zombies = self._orders.find_zombie_orders(zombie_threshold_s)
                    for zombie in zombies:
                        logger.warning(
                            "Zombie order detected (no stream update in %.0fs): "
                            "ext_id=%s exchange_id=%s side=%s level=%d — cancelling",
                            time.monotonic() - zombie.placed_at,
                            zombie.external_id,
                            zombie.exchange_order_id,
                            zombie.side,
                            zombie.level,
                        )
                        await self._orders.cancel_order(zombie.external_id)

                # --- POF-spread correlation update ---
                placement_stats_60s = self._orders.failure_window_stats(60.0)
                total_placements_60s = int(placement_stats_60s["attempts"])
                self._post_only.update_pof_offset_boost(total_placements_60s)

                # --- Failure rate circuit breaker ---
                window_stats = self._orders.failure_window_stats(failure_window_s)
                attempts = int(window_stats["attempts"])
                failure_rate = float(window_stats["failure_rate"])
                window_trip = (
                    attempts >= min_attempts and failure_rate >= failure_rate_trip
                )
                if (
                    not self._circuit_open
                    and (
                        self._orders.consecutive_failures >= max_failures
                        or window_trip
                    )
                ):
                    self._circuit_open = True
                    self._metrics.circuit_open = True
                    logger.warning(
                        "CIRCUIT BREAKER OPEN: consecutive_failures=%d attempts=%d "
                        "failure_rate=%.2f — pausing for %.0fs",
                        self._orders.consecutive_failures,
                        attempts,
                        failure_rate,
                        cooldown,
                    )
                    await asyncio.sleep(cooldown)
                    self._orders.reset_failure_tracking()
                    self._circuit_open = False
                    self._metrics.circuit_open = False
                    logger.info("Circuit breaker reset — resuming")
                else:
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.error("Circuit breaker task error: %s", exc)
                self._journal.record_error(
                    component="circuit_breaker",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                    stack_trace_hash=TradeJournal.make_stack_trace_hash(exc),
                )
                await asyncio.sleep(1.0)
