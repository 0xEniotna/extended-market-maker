"""Orderbook adapter with callbacks required by the market maker.

The upstream 1.4.x Extended SDK orderbook exposes best bid/ask callbacks, but
not the same-BBO heartbeat and sequence-gap hooks this strategy uses for stale
quote protection. This adapter keeps that behavior in the MM repo instead of
carrying a dirty SDK submodule patch.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Optional

from sortedcontainers import SortedDict
from x10.perpetual.stream_client.stream_client import PerpetualStreamClient
from x10.utils.http import StreamDataType

from .extended_sdk import stream_url

logger = logging.getLogger(__name__)


@dataclass
class OrderBookEntry:
    price: Decimal
    amount: Decimal


class OrderBook:
    """Small orderbook implementation compatible with the SDK's public shape."""

    @staticmethod
    async def create(
        config,
        market_name: str,
        best_ask_change_callback: Optional[
            Callable[[Optional[OrderBookEntry]], Awaitable[None]]
        ] = None,
        best_bid_change_callback: Optional[
            Callable[[Optional[OrderBookEntry]], Awaitable[None]]
        ] = None,
        orderbook_update_callback: Optional[Callable[[], Awaitable[None]]] = None,
        sequence_gap_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
        snapshot_callback: Optional[Callable[[int], Awaitable[None]]] = None,
        start: bool = False,
        depth: Optional[int] = None,
    ) -> "OrderBook":
        book = OrderBook(
            config,
            market_name,
            best_ask_change_callback=best_ask_change_callback,
            best_bid_change_callback=best_bid_change_callback,
            orderbook_update_callback=orderbook_update_callback,
            sequence_gap_callback=sequence_gap_callback,
            snapshot_callback=snapshot_callback,
            depth=depth,
        )
        if start:
            await book.start_orderbook()
        return book

    def __init__(
        self,
        config,
        market_name: str,
        *,
        best_ask_change_callback: Optional[
            Callable[[Optional[OrderBookEntry]], Awaitable[None]]
        ] = None,
        best_bid_change_callback: Optional[
            Callable[[Optional[OrderBookEntry]], Awaitable[None]]
        ] = None,
        orderbook_update_callback: Optional[Callable[[], Awaitable[None]]] = None,
        sequence_gap_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
        snapshot_callback: Optional[Callable[[int], Awaitable[None]]] = None,
        depth: Optional[int] = None,
    ) -> None:
        self._stream_client = PerpetualStreamClient(api_url=stream_url(config))
        self._market_name = market_name
        self._task: Optional[asyncio.Task] = None
        self._bid_prices: SortedDict[Decimal, OrderBookEntry] = SortedDict()
        self._ask_prices: SortedDict[Decimal, OrderBookEntry] = SortedDict()
        self.best_ask_change_callback = best_ask_change_callback
        self.best_bid_change_callback = best_bid_change_callback
        self.orderbook_update_callback = orderbook_update_callback
        self.sequence_gap_callback = sequence_gap_callback
        self.snapshot_callback = snapshot_callback
        self.depth = depth
        self._last_seq: Optional[int] = None

    async def _notify_orderbook_update(self) -> None:
        if self.orderbook_update_callback is None:
            return
        try:
            await self.orderbook_update_callback()
        except Exception as exc:
            logger.error("Error in orderbook update callback: %s", exc, exc_info=True)

    async def _notify_bid_if_changed(
        self, before: Optional[OrderBookEntry]
    ) -> None:
        current = self.best_bid()
        if before != current and self.best_bid_change_callback is not None:
            await self.best_bid_change_callback(current)

    async def _notify_ask_if_changed(
        self, before: Optional[OrderBookEntry]
    ) -> None:
        current = self.best_ask()
        if before != current and self.best_ask_change_callback is not None:
            await self.best_ask_change_callback(current)

    async def update_orderbook(self, data) -> None:
        best_bid_before = self.best_bid()
        for bid in data.bid:
            if bid.price in self._bid_prices:
                entry = self._bid_prices[bid.price]
                entry.amount = entry.amount + bid.qty
                if entry.amount == 0:
                    del self._bid_prices[bid.price]
            else:
                self._bid_prices[bid.price] = OrderBookEntry(
                    price=bid.price,
                    amount=bid.qty,
                )
        await self._notify_bid_if_changed(best_bid_before)

        best_ask_before = self.best_ask()
        for ask in data.ask:
            if ask.price in self._ask_prices:
                entry = self._ask_prices[ask.price]
                entry.amount = entry.amount + ask.qty
                if entry.amount == 0:
                    del self._ask_prices[ask.price]
            else:
                self._ask_prices[ask.price] = OrderBookEntry(
                    price=ask.price,
                    amount=ask.qty,
                )
        await self._notify_ask_if_changed(best_ask_before)
        await self._notify_orderbook_update()

    async def init_orderbook(self, data) -> None:
        self._bid_prices.clear()
        self._ask_prices.clear()

        best_bid_before = self.best_bid()
        for bid in data.bid:
            self._bid_prices[bid.price] = OrderBookEntry(
                price=bid.price,
                amount=bid.qty,
            )
        await self._notify_bid_if_changed(best_bid_before)

        best_ask_before = self.best_ask()
        for ask in data.ask:
            self._ask_prices[ask.price] = OrderBookEntry(
                price=ask.price,
                amount=ask.qty,
            )
        await self._notify_ask_if_changed(best_ask_before)
        await self._notify_orderbook_update()

    async def start_orderbook(self) -> asyncio.Task:
        async def _run() -> None:
            while True:
                async with self._stream_client.subscribe_to_orderbooks(
                    self._market_name,
                    depth=self.depth,
                ) as stream:
                    self._last_seq = None
                    async for event in stream:
                        current_seq = int(getattr(event, "seq", 0) or 0)
                        if (
                            self._last_seq is not None
                            and current_seq != (self._last_seq + 1)
                        ):
                            prev = self._last_seq
                            logger.critical(
                                "Orderbook sequence gap for %s: prev=%s current=%s",
                                self._market_name,
                                prev,
                                current_seq,
                            )
                            if self.sequence_gap_callback is not None:
                                await self.sequence_gap_callback(prev, current_seq)
                            break
                        self._last_seq = current_seq

                        if event.type == StreamDataType.SNAPSHOT:
                            if not event.data:
                                continue
                            if self.snapshot_callback is not None:
                                await self.snapshot_callback(current_seq)
                            await self.init_orderbook(event.data)
                        elif event.type == StreamDataType.DELTA:
                            if not event.data:
                                continue
                            await self.update_orderbook(event.data)
                await asyncio.sleep(1)

        self._task = asyncio.create_task(_run(), name=f"mm-sdk-ob-{self._market_name}")
        return self._task

    def stop_orderbook(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    def best_bid(self) -> Optional[OrderBookEntry]:
        try:
            return self._bid_prices.peekitem(-1)[1]
        except IndexError:
            return None

    def best_ask(self) -> Optional[OrderBookEntry]:
        try:
            return self._ask_prices.peekitem(0)[1]
        except IndexError:
            return None

    async def close(self) -> None:
        self.stop_orderbook()
