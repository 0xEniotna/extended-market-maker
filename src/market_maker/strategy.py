"""
Market Maker Strategy

Main orchestration for the market making bot.  Spawns one async task per
(level, side) pair.  Each task waits for a price change on the relevant
side of the book, computes the target price, and reprices its order when
the current order drifts outside a tolerance band.

Entry point: ``MarketMakerStrategy.run()``
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, Optional, Set

from x10.perpetual.orders import OrderSide
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager, FillEvent
from .config import MarketMakerSettings
from .decision_models import TrendState
from .drawdown_stop import DrawdownStop
from .funding_manager import FundingManager
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .post_only_safety import PostOnlySafety
from .pricing_engine import PricingEngine
from .quote_halt_manager import QuoteHaltManager
from .reprice_pipeline import RepricePipeline
from .risk_manager import RiskManager
from .risk_watchdog import RiskWatchdog
from .strategy_callbacks import on_fill, on_level_freed
from .strategy_components import (
    estimate_tick_time_ms,
    handle_reload,
    rebuild_components,
    sanitized_run_config,
    toxicity_adjustment,
)
from .strategy_quoting import (
    build_reprice_market_context,
    cancel_level_order,
    is_fatal_exception,
    level_task,
    normalise_side,
    on_adverse_markout_widen,
    on_stream_desync,
    order_age_exceeded,
    record_reprice_decision,
    sync_quote_halt_state,
)
from .strategy_runner import run_strategy
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class MarketMakerStrategy:
    """Market making strategy that quotes on multiple price levels per side."""

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
        self._level_last_reprice_at: Dict[tuple[str, int], float] = {}
        self._level_stale_since: Dict[tuple[str, int], Optional[float]] = {}
        self._level_cancel_pending_ext_id: Dict[tuple[str, int], Optional[str]] = {}
        self._pending_cancel_reasons: Dict[str, str] = {}

        # Fill deduplication
        self._seen_trade_ids: deque = deque()
        self._seen_trade_ids_set: Set[int] = set()

        # Halt state
        self._halt_mgr = QuoteHaltManager(
            market_name=settings.market_name,
            journal=journal,
            metrics=metrics,
        )
        self._runtime_mode: str = "normal"
        self._last_taker_leakage_warn_at: float = 0.0

        # Declared for mypy; populated by _rebuild_components().
        self._pricing: PricingEngine
        self._post_only: PostOnlySafety
        self._reprice: RepricePipeline
        self._drawdown_stop: DrawdownStop
        self._level_pof_until: Dict[tuple[str, int], float]

        self._rebuild_components()

        self._risk_watchdog = RiskWatchdog(
            settings=settings,
            risk_mgr=risk_mgr,
            orders=order_mgr,
            halt_mgr=self._halt_mgr,
            metrics=metrics,
            journal=journal,
            post_only=self._post_only,
            drawdown_stop=self._drawdown_stop,
            request_shutdown_fn=self._request_shutdown,
        )
        self._funding_mgr = FundingManager(
            market_profile=str(settings.market_profile),
            funding_bias_enabled=bool(settings.funding_bias_enabled),
            funding_inventory_weight=settings.funding_inventory_weight,
            funding_bias_cap_bps=settings.funding_bias_cap_bps,
        )
        self._shutdown_reason = "shutdown"
        self._shutdown_event = asyncio.Event()

    @classmethod
    async def run(cls) -> None:
        await run_strategy(cls)

    def _handle_signal(self) -> None:
        logger.info("Signal received, initiating shutdown")
        self._request_shutdown("shutdown")

    _RELOAD_IMMUTABLE_KEYS = frozenset({
        "market_name", "environment", "vault_id", "api_key",
        "stark_private_key", "stark_public_key",
    })

    def _handle_reload(self) -> None:
        handle_reload(self)

    @property
    def shutdown_reason(self) -> str:
        return self._shutdown_reason

    @staticmethod
    def _sanitized_run_config(settings: MarketMakerSettings) -> Dict[str, Any]:
        return sanitized_run_config(settings)

    def _on_fill(self, fill: FillEvent) -> None:
        on_fill(self, fill)

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

    async def _level_task(self, side: OrderSide, level: int) -> None:
        await level_task(self, side, level)

    async def _maybe_reprice(self, side: OrderSide, level: int) -> None:
        sync_quote_halt_state(self)
        if self._quote_halt_reasons:
            return
        market_ctx = build_reprice_market_context(self)
        await self._reprice.evaluate(self, side, level, market_ctx=market_ctx)  # type: ignore[arg-type]

    def _clear_level_slot(self, key: tuple[str, int]) -> None:
        self._level_ext_ids[key] = None
        self._level_order_created_at[key] = None
        self._level_stale_since[key] = None
        self._level_cancel_pending_ext_id[key] = None

    def _record_reprice_decision(self, **kwargs: Any) -> None:
        record_reprice_decision(self, **kwargs)

    async def _cancel_level_order(
        self,
        *,
        key: tuple[str, int],
        external_id: str,
        side: OrderSide,
        level: int,
        reason: str,
    ) -> bool:
        return await cancel_level_order(
            self, key=key, external_id=external_id,
            side=side, level=level, reason=reason,
        )

    def _quantize_size(self, size: Decimal) -> Decimal:
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
        return estimate_tick_time_ms(self)

    def _on_adverse_markout_widen(self, key: tuple[str, int], reason: str) -> None:
        on_adverse_markout_widen(self, key, reason)

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
        return toxicity_adjustment(self)

    @staticmethod
    def _normalise_side(side_value: str) -> str:
        return normalise_side(side_value)

    @staticmethod
    def _is_fatal_exception(exc: BaseException) -> bool:
        return is_fatal_exception(exc)

    def _increases_inventory(self, side: OrderSide) -> bool:
        side_name = self._normalise_side(str(side))
        pos = self._risk.get_current_position()
        if side_name == "BUY":
            return pos >= 0
        if side_name == "SELL":
            return pos <= 0
        return False

    def _set_funding_rate(self, funding_rate: Decimal) -> None:
        self._funding_mgr.set_funding_rate(funding_rate)

    def _funding_bias_bps(self) -> Decimal:
        return self._funding_mgr.funding_bias_bps()

    @property
    def _funding_rate(self) -> Decimal:
        return self._funding_mgr.funding_rate

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

    @property
    def _quote_halt_reasons(self) -> Set[str]:
        return self._halt_mgr.reasons

    @property
    def _margin_breach_since(self) -> Optional[float]:
        return self._halt_mgr.margin_breach_since

    @_margin_breach_since.setter
    def _margin_breach_since(self, value: Optional[float]) -> None:
        self._halt_mgr.margin_breach_since = value

    def _is_normal_quoting_mode(self) -> bool:
        return (
            self._runtime_mode == "normal"
            and not self._shutdown_event.is_set()
            and not self._halt_mgr.is_halted
        )

    def _set_quote_halt(self, reason: str) -> None:
        self._halt_mgr.set_halt(reason)

    def _clear_quote_halt(self, reason: str) -> None:
        self._halt_mgr.clear_halt(reason)

    def _streams_healthy(self) -> bool:
        return QuoteHaltManager.check_streams_healthy(
            self._account_stream, self._ob,
        )

    async def _on_stream_desync(self, reason: str) -> None:
        await on_stream_desync(self, reason)

    def _sync_quote_halt_state(self) -> None:
        sync_quote_halt_state(self)

    def _rebuild_components(self) -> None:
        rebuild_components(self)

    @property
    def _circuit_open(self) -> bool:
        return self._risk_watchdog.circuit_open

    def _evaluate_drawdown_stop(self) -> bool:
        return self._risk_watchdog.evaluate_drawdown_stop()

    async def _drawdown_watchdog_task(self) -> None:
        await self._risk_watchdog.drawdown_watchdog_task(self._shutdown_event)

    async def _funding_refresh_task(self) -> None:
        await self._funding_mgr.refresh_task(
            self._client, self._settings.market_name, self._shutdown_event,
        )

    def _margin_guard_breach(self) -> tuple[bool, list[str], dict[str, Optional[Decimal]]]:
        return self._risk_watchdog.margin_guard_breach()

    async def _margin_guard_task(self) -> None:
        await self._risk_watchdog.margin_guard_task(self._shutdown_event)

    def _order_age_exceeded(
        self,
        key: tuple[str, int],
        *,
        max_age_s: Optional[float] = None,
    ) -> bool:
        return order_age_exceeded(self, key, max_age_s=max_age_s)

    def _compute_target_price(self, side, level, best_price, **kw) -> Decimal:
        return self._pricing.compute_target_price(side, level, best_price, **kw)

    def _level_size(self, level: int) -> Decimal:
        return self._pricing.level_size(level)

    def _needs_reprice(self, side, prev_price, current_best, level, **kw):
        return self._reprice.needs_reprice(side, prev_price, current_best, level, **kw)

    def _compute_offset(
        self,
        level: int,
        best_price: Decimal,
        *,
        regime_scale: Decimal = Decimal("1"),
    ) -> Decimal:
        return self._pricing.compute_offset(
            level, best_price, regime_scale=regime_scale,
        )

    def _round_to_tick(
        self, price: Decimal, side: Optional[OrderSide] = None
    ) -> Decimal:
        return self._pricing.round_to_tick(price, side)

    def _theoretical_edge_bps(
        self, side: OrderSide, quote_price: Decimal, current_best: Decimal,
    ) -> Decimal:
        return self._pricing.theoretical_edge_bps(side, quote_price, current_best)

    async def _circuit_breaker_task(self) -> None:
        await self._risk_watchdog.circuit_breaker_task(self._shutdown_event)
