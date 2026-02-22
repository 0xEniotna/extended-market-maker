#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _stub_sdk_modules() -> None:
    modules = [
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
    for mod_name in modules:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    orders_mod = sys.modules["x10.perpetual.orders"]
    orders_mod.OrderSide = SimpleNamespace(BUY="BUY", SELL="SELL")
    orders_mod.OrderStatus = SimpleNamespace(
        FILLED="FILLED",
        CANCELLED="CANCELLED",
        EXPIRED="EXPIRED",
        REJECTED="REJECTED",
    )
    orders_mod.OrderType = SimpleNamespace(LIMIT="LIMIT")
    orders_mod.TimeInForce = SimpleNamespace(GTT="GTT")


def _make_strategy(levels: int):
    _stub_sdk_modules()
    from market_maker.config import MarketMakerSettings
    from market_maker.orderbook_manager import PriceLevel
    from market_maker.strategy import MarketMakerStrategy

    class FakeOrderbook:
        def __init__(self) -> None:
            self._bid = PriceLevel(price=Decimal("100.00"), size=Decimal("10"))
            self._ask = PriceLevel(price=Decimal("100.10"), size=Decimal("10"))
            self.best_bid_condition = asyncio.Condition()
            self.best_ask_condition = asyncio.Condition()
            self._imbalance = Decimal("0")

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
            return False

        def micro_volatility_bps(self, window_s: float):
            _ = window_s
            return Decimal("0.5")

        def micro_drift_bps(self, window_s: float):
            _ = window_s
            return Decimal("0")

        def mid_prices(self, window_s: float):
            _ = window_s
            return [Decimal("100.00"), Decimal("100.02"), Decimal("100.01"), Decimal("100.03")]

        def orderbook_imbalance(self, window_s: float):
            _ = window_s
            return self._imbalance

        def market_snapshot(self, depth: int = 5, **kwargs):
            _ = kwargs
            return {
                "depth": depth,
                "best_bid": self._bid.price,
                "best_ask": self._ask.price,
                "bids_top": [],
                "asks_top": [],
            }

    class FakeRisk:
        def get_current_position(self):
            return Decimal("0")

        def get_position_total_pnl(self):
            return Decimal("0")

        def allowed_order_size(self, side, requested_size, reference_price, **kwargs):
            _ = (side, reference_price, kwargs)
            return requested_size

    settings = MarketMakerSettings(
        vault_id="1",
        stark_private_key="0x1",
        stark_public_key="0x2",
        api_key="k",
        market_name="TEST-USD",
        market_profile="crypto",
        num_price_levels=levels,
        offset_mode="dynamic",
        spread_multiplier=Decimal("1.0"),
        min_offset_bps=Decimal("2"),
        max_offset_bps=Decimal("50"),
        reprice_tolerance_percent=Decimal("0.5"),
    )
    ob = FakeOrderbook()
    risk = FakeRisk()
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
    orders.cancel_order = AsyncMock(return_value=True)
    orders.place_order = AsyncMock(return_value="ext")
    orders.consecutive_failures = 0

    strategy = MarketMakerStrategy(
        settings=settings,
        trading_client=MagicMock(),
        orderbook_mgr=ob,
        order_mgr=orders,
        risk_mgr=risk,
        account_stream=MagicMock(),
        metrics=MagicMock(),
        journal=MagicMock(),
        tick_size=Decimal("0.01"),
        base_order_size=Decimal("1"),
        market_min_order_size=Decimal("0.1"),
        min_order_size_step=Decimal("0.1"),
    )
    return strategy


async def _run_benchmark(levels: int, iterations: int) -> None:
    strategy = _make_strategy(levels)
    side_buy = "BUY"
    side_sell = "SELL"

    for _ in range(20):
        for level in range(levels):
            await strategy._maybe_reprice(side_buy, level)
            await strategy._maybe_reprice(side_sell, level)

    samples_us: list[float] = []
    for _ in range(iterations):
        for level in range(levels):
            t0 = time.perf_counter()
            await strategy._maybe_reprice(side_buy, level)
            samples_us.append((time.perf_counter() - t0) * 1_000_000)
            t0 = time.perf_counter()
            await strategy._maybe_reprice(side_sell, level)
            samples_us.append((time.perf_counter() - t0) * 1_000_000)

    p95 = statistics.quantiles(samples_us, n=20)[18]
    print(f"samples={len(samples_us)} levels={levels} iterations={iterations}")
    print(f"mean_us={statistics.mean(samples_us):.2f}")
    print(f"median_us={statistics.median(samples_us):.2f}")
    print(f"p95_us={p95:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic reprice pipeline micro-benchmark.")
    parser.add_argument("--levels", type=int, default=4, help="Number of quote levels per side.")
    parser.add_argument("--iterations", type=int, default=500, help="Evaluation loops per level/side.")
    args = parser.parse_args()

    asyncio.run(_run_benchmark(levels=max(1, args.levels), iterations=max(1, args.iterations)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
