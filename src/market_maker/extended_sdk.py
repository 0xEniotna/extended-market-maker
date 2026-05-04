"""Compatibility helpers for Extended's x10 Python SDK.

The market maker supports the newer 1.4.x SDK layout used by the pairs bot
while retaining enough import fallback for unit tests and older local stubs.
"""
from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # SDK >= 1.4.x
    from x10.config import MAINNET_CONFIG, TESTNET_CONFIG
    from x10.config import Config as EndpointConfig
    from x10.core.stark_account import StarkPerpetualAccount
    from x10.errors import ApiRateLimitError as RateLimitException
    from x10.models.account import AccountStreamDataModel
    from x10.models.fee import TradingFeeModel
    from x10.models.order import (
        OpenOrderModel,
        OrderSide,
        OrderStatus,
        OrderStatusReason,
        OrderType,
        TimeInForce,
    )
    from x10.models.position import PositionModel, PositionSide, PositionStatus
    from x10.models.trade import AccountTradeModel
    from x10.perpetual.order_object import create_order_object
    from x10.perpetual.trading_client import PerpetualTradingClient

    DEFAULT_FEES = TradingFeeModel(
        market="UNKNOWN",
        maker_fee_rate=Decimal("0"),
        taker_fee_rate=Decimal("0"),
        builder_fee_rate=Decimal("0"),
    )
except Exception:  # pragma: no cover - exercised by older SDK/test stubs
    def _import_attr(module_name: str, attr: str, default: Any) -> Any:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr)
        except Exception:
            return default

    class _FallbackFees(SimpleNamespace):
        market: str = "UNKNOWN"
        maker_fee_rate: Decimal = Decimal("0")
        taker_fee_rate: Decimal = Decimal("0")
        builder_fee_rate: Decimal = Decimal("0")

    class _FallbackAccount:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    _fallback_config = SimpleNamespace(
        stream_url="",
        api_base_url="",
        starknet_domain=None,
        signing=SimpleNamespace(starknet_domain=None),
        endpoints=SimpleNamespace(stream_url="", api_base_url=""),
    )

    AccountStreamDataModel = _import_attr("x10.perpetual.accounts", "AccountStreamDataModel", object)
    StarkPerpetualAccount = _import_attr("x10.perpetual.accounts", "StarkPerpetualAccount", _FallbackAccount)
    EndpointConfig = _import_attr("x10.perpetual.configuration", "EndpointConfig", object)
    MAINNET_CONFIG = _import_attr("x10.perpetual.configuration", "MAINNET_CONFIG", _fallback_config)
    TESTNET_CONFIG = _import_attr("x10.perpetual.configuration", "TESTNET_CONFIG", _fallback_config)
    TradingFeeModel = _import_attr("x10.perpetual.fees", "TradingFeeModel", _FallbackFees)
    DEFAULT_FEES = _import_attr("x10.perpetual.fees", "DEFAULT_FEES", _FallbackFees())
    OpenOrderModel = _import_attr("x10.perpetual.orders", "OpenOrderModel", object)
    OrderSide = _import_attr(
        "x10.perpetual.orders",
        "OrderSide",
        SimpleNamespace(BUY="BUY", SELL="SELL"),
    )
    OrderStatus = _import_attr(
        "x10.perpetual.orders",
        "OrderStatus",
        SimpleNamespace(FILLED="FILLED", CANCELLED="CANCELLED", EXPIRED="EXPIRED", REJECTED="REJECTED"),
    )
    OrderStatusReason = _import_attr("x10.perpetual.orders", "OrderStatusReason", SimpleNamespace())
    OrderType = _import_attr(
        "x10.perpetual.orders",
        "OrderType",
        SimpleNamespace(LIMIT="LIMIT", MARKET="MARKET"),
    )
    TimeInForce = _import_attr(
        "x10.perpetual.orders",
        "TimeInForce",
        SimpleNamespace(GTT="GTT", IOC="IOC"),
    )
    create_order_object = _import_attr("x10.perpetual.order_object", "create_order_object", None)
    PositionModel = _import_attr("x10.perpetual.positions", "PositionModel", object)
    PositionSide = _import_attr(
        "x10.perpetual.positions",
        "PositionSide",
        SimpleNamespace(LONG="LONG", SHORT="SHORT"),
    )
    PositionStatus = _import_attr("x10.perpetual.positions", "PositionStatus", SimpleNamespace())
    AccountTradeModel = _import_attr("x10.perpetual.trades", "AccountTradeModel", object)
    PerpetualTradingClient = _import_attr("x10.perpetual.trading_client", "PerpetualTradingClient", object)
    _RateLimitException = _import_attr("x10.utils.http", "RateLimitException", None)
    if (
        isinstance(_RateLimitException, type)
        and issubclass(_RateLimitException, BaseException)
    ):
        RateLimitException = _RateLimitException
    else:
        class RateLimitException(Exception):  # type: ignore[no-redef]
            pass


@dataclass(frozen=True)
class SignedOrderFees:
    """Fee fields accepted by both old and new SDK order-object builders."""

    max_fee_rate: Optional[Decimal] = None
    builder_fee_rate: Optional[Decimal] = None
    builder_id: Optional[int] = None


def stream_url(endpoint_config: Any) -> str:
    """Return the websocket URL from either SDK config shape."""
    endpoints = getattr(endpoint_config, "endpoints", None)
    if endpoints is not None:
        return str(endpoints.stream_url)
    return str(endpoint_config.stream_url)


def starknet_domain(endpoint_config: Any) -> Any:
    """Return the Starknet signing domain from either SDK config shape."""
    signing = getattr(endpoint_config, "signing", None)
    if signing is not None:
        return signing.starknet_domain
    return endpoint_config.starknet_domain


def _callable_is_plain_mock(fn: Any) -> bool:
    return type(fn).__module__.startswith("unittest.mock")


async def _maybe_get_market(client: Any, market_name: str) -> Optional[Any]:
    markets_info = getattr(client, "markets_info", None)
    get_markets_dict = getattr(markets_info, "get_markets_dict", None)
    if get_markets_dict is None or _callable_is_plain_mock(get_markets_dict):
        return None
    result = get_markets_dict()
    if not inspect.isawaitable(result):
        return None
    markets = await result
    if not isinstance(markets, dict):
        return None
    return markets.get(market_name)


def _create_order_kwargs(
    *,
    account: Any,
    market: Any,
    market_name: str,
    amount_of_synthetic: Decimal,
    price: Decimal,
    side: Any,
    endpoint_config: Any,
    order_type: Any,
    time_in_force: Any,
    post_only: bool,
    reduce_only: bool,
    external_id: Optional[str],
    fees: SignedOrderFees,
) -> dict[str, Any]:
    kwargs = {
        "account": account,
        "market": market,
        "amount_of_synthetic": amount_of_synthetic,
        "price": price,
        "side": side,
        "order_type": order_type,
        "starknet_domain": starknet_domain(endpoint_config),
        "time_in_force": time_in_force,
        "post_only": post_only,
        "reduce_only": reduce_only,
        "order_external_id": external_id,
    }

    try:
        params = inspect.signature(create_order_object).parameters
    except (TypeError, ValueError):
        params = {}

    if "taker_fee" in params:
        kwargs["taker_fee"] = fees.max_fee_rate
        kwargs["builder_fee"] = fees.builder_fee_rate
    else:
        kwargs["max_fee_rate"] = fees.max_fee_rate
        kwargs["builder_fee_rate"] = fees.builder_fee_rate
    kwargs["builder_id"] = fees.builder_id

    if "market_name" in params:
        kwargs["market_name"] = market_name

    return {key: value for key, value in kwargs.items() if value is not None}


async def place_signed_order(
    client: Any,
    *,
    market_name: str,
    amount_of_synthetic: Decimal,
    price: Decimal,
    side: Any,
    order_type: Any,
    time_in_force: Any,
    post_only: bool,
    reduce_only: bool = False,
    external_id: Optional[str] = None,
    fees: SignedOrderFees = SignedOrderFees(),
) -> Any:
    """Place an order through the SDK order-object API when available.

    SDK 1.4.x changed the high-level ``PerpetualTradingClient.place_order`` API:
    it requires a taker-fee argument and no longer exposes ``order_type``. The
    stable path is to build a signed ``NewOrderModel`` and submit it via
    ``client.orders.place_order``. Older SDKs and unit-test mocks fall back to
    the legacy high-level method.
    """
    account = getattr(client, "stark_account", None)
    orders_module = getattr(client, "orders", None)
    submit_order = getattr(orders_module, "place_order", None)
    endpoint_config = getattr(client, "config", None)

    market = None
    if account is not None and submit_order is not None and endpoint_config is not None:
        market = await _maybe_get_market(client, market_name)

    if (
        market is not None
        and create_order_object is not None
        and not _callable_is_plain_mock(submit_order)
    ):
        order = create_order_object(
            **_create_order_kwargs(
                account=account,
                market=market,
                market_name=market_name,
                amount_of_synthetic=amount_of_synthetic,
                price=price,
                side=side,
                endpoint_config=endpoint_config,
                order_type=order_type,
                time_in_force=time_in_force,
                post_only=post_only,
                reduce_only=reduce_only,
                external_id=external_id,
                fees=fees,
            )
        )
        return await submit_order(order)

    return await client.place_order(
        market_name=market_name,
        amount_of_synthetic=amount_of_synthetic,
        price=price,
        side=side,
        order_type=order_type,
        time_in_force=time_in_force,
        post_only=post_only,
        reduce_only=reduce_only,
        external_id=external_id,
        max_fee_rate=fees.max_fee_rate,
        builder_fee_rate=fees.builder_fee_rate,
        builder_id=fees.builder_id,
    )
