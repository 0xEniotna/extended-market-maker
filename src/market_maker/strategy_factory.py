"""
Strategy Factory

Groups the component creation steps for building a RuntimeContext.
Previously these were inlined in ``strategy_runner._build_runtime_context``
— extracting them here makes each step independently testable and keeps
the runner module focused on lifecycle orchestration.
"""
from __future__ import annotations

import logging
import uuid
from decimal import ROUND_DOWN, Decimal
from typing import Any, Type

from x10.perpetual.accounts import StarkPerpetualAccount
from x10.perpetual.trading_client import PerpetualTradingClient

from .account_stream import AccountStreamManager
from .config import MarketMakerSettings
from .fee_resolver import FeeResolver
from .metrics import MetricsCollector
from .order_manager import OrderManager
from .orderbook_manager import OrderbookManager
from .risk_manager import RiskManager
from .trade_journal import TradeJournal

logger = logging.getLogger(__name__)


class StrategyFactory:
    """Creates and wires all managers for a market-making strategy."""

    def __init__(self, settings: MarketMakerSettings) -> None:
        self._settings = settings

    def build_trading_client(self) -> PerpetualTradingClient:
        account = StarkPerpetualAccount(
            vault=int(self._settings.vault_id),
            private_key=self._settings.stark_private_key,
            public_key=self._settings.stark_public_key,
            api_key=self._settings.api_key,
        )
        return PerpetualTradingClient(self._settings.endpoint_config, account)

    async def load_market_params(
        self,
        trading_client: PerpetualTradingClient,
    ) -> tuple[Any, Decimal, Decimal, Decimal, Decimal]:
        """Fetch market info and derive tick/order size parameters."""
        markets = await trading_client.markets_info.get_markets_dict()
        market_info = markets.get(self._settings.market_name)
        if market_info is None:
            raise LookupError(
                f"Market {self._settings.market_name} not found on exchange"
            )

        tick_size = Decimal(str(market_info.trading_config.min_price_change))
        min_order_size = Decimal(str(market_info.trading_config.min_order_size))
        min_order_size_change = Decimal(
            str(market_info.trading_config.min_order_size_change)
        )
        order_size = (
            min_order_size * self._settings.order_size_multiplier
        ).quantize(min_order_size_change, rounding=ROUND_DOWN)
        if order_size < min_order_size:
            order_size = min_order_size
        return market_info, tick_size, min_order_size, min_order_size_change, order_size

    def build_fee_resolver(
        self,
        trading_client: PerpetualTradingClient,
    ) -> FeeResolver:
        return FeeResolver(
            trading_client=trading_client,
            market_name=self._settings.market_name,
            refresh_interval_s=self._settings.fee_refresh_interval_s,
            builder_program_enabled=self._settings.builder_program_enabled,
            builder_id=(
                self._settings.builder_id
                if self._settings.builder_id > 0
                else None
            ),
            configured_builder_fee_rate=self._settings.builder_fee_rate,
        )

    def build_orderbook_manager(self) -> OrderbookManager:
        return OrderbookManager(
            self._settings.endpoint_config,
            self._settings.market_name,
            staleness_threshold_s=self._settings.orderbook_staleness_threshold_s,
        )

    def build_order_manager(
        self,
        trading_client: PerpetualTradingClient,
        fee_resolver: FeeResolver,
    ) -> OrderManager:
        return OrderManager(
            trading_client,
            self._settings.market_name,
            max_orders_per_second=self._settings.max_orders_per_second,
            maintenance_pause_s=self._settings.maintenance_pause_s,
            fee_resolver=fee_resolver,
            rate_limit_degraded_s=self._settings.rate_limit_degraded_s,
            rate_limit_halt_window_s=self._settings.rate_limit_halt_window_s,
            rate_limit_halt_hits=self._settings.rate_limit_halt_hits,
            rate_limit_halt_s=self._settings.rate_limit_halt_s,
            rate_limit_extra_offset_bps=self._settings.rate_limit_extra_offset_bps,
            rate_limit_reprice_multiplier=self._settings.rate_limit_reprice_multiplier,
        )

    def build_risk_manager(
        self,
        trading_client: PerpetualTradingClient,
        ob_mgr: OrderbookManager,
    ) -> RiskManager:
        return RiskManager(
            trading_client,
            self._settings.market_name,
            self._settings.max_position_size,
            max_order_notional_usd=self._settings.max_order_notional_usd,
            max_position_notional_usd=self._settings.max_position_notional_usd,
            gross_exposure_limit_usd=self._settings.gross_exposure_limit_usd,
            max_long_position_size=self._settings.max_long_position_size,
            max_short_position_size=self._settings.max_short_position_size,
            balance_aware_sizing_enabled=self._settings.balance_aware_sizing_enabled,
            balance_usage_factor=self._settings.balance_usage_factor,
            balance_notional_multiplier=self._settings.balance_notional_multiplier,
            balance_min_available_usd=self._settings.balance_min_available_usd,
            balance_staleness_max_s=self._settings.balance_staleness_max_s,
            balance_stale_action=self._settings.balance_stale_action,
            orderbook_mgr=ob_mgr,
        )

    def build_account_stream(self) -> AccountStreamManager:
        return AccountStreamManager(
            self._settings.endpoint_config,
            self._settings.api_key,
            self._settings.market_name,
        )

    def build_journal(self) -> TradeJournal:
        return TradeJournal(
            self._settings.market_name,
            run_id=uuid.uuid4().hex,
            schema_version=2,
            max_size_mb=self._settings.journal_max_size_mb,
        )

    def build_metrics(
        self,
        *,
        ob_mgr: OrderbookManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
        account_stream: AccountStreamManager,
        journal: TradeJournal,
    ) -> MetricsCollector:
        return MetricsCollector(
            orderbook_mgr=ob_mgr,
            order_mgr=order_mgr,
            risk_mgr=risk_mgr,
            account_stream=account_stream,
            journal=journal,
        )

    def build_strategy(
        self,
        strategy_cls: Type,
        *,
        trading_client: PerpetualTradingClient,
        market_info: Any,
        ob_mgr: OrderbookManager,
        order_mgr: OrderManager,
        risk_mgr: RiskManager,
        account_stream: AccountStreamManager,
        metrics: MetricsCollector,
        journal: TradeJournal,
        tick_size: Decimal,
        order_size: Decimal,
        min_order_size: Decimal,
        min_order_size_change: Decimal,
    ) -> Any:
        """Create the strategy instance and initialise funding rate."""
        strategy = strategy_cls(
            settings=self._settings,
            trading_client=trading_client,
            orderbook_mgr=ob_mgr,
            order_mgr=order_mgr,
            risk_mgr=risk_mgr,
            account_stream=account_stream,
            metrics=metrics,
            journal=journal,
            tick_size=tick_size,
            base_order_size=order_size,
            market_min_order_size=min_order_size,
            min_order_size_step=min_order_size_change,
        )
        try:
            strategy._set_funding_rate(
                Decimal(str(market_info.market_stats.funding_rate))
            )
        except Exception:
            logger.debug("Funding rate unavailable at startup", exc_info=True)
        return strategy
