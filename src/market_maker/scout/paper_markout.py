"""Paper maker-markout estimation for market scout candidate filtering.

This module intentionally does not place orders. It infers hypothetical maker
fills from public trades + top-of-book updates and computes short-horizon
markouts/toxicity metrics.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote, urlparse

from market_maker.public_markets import resolve_default_api_base

DEFAULT_HORIZONS_MS: tuple[int, ...] = (250, 1000, 5000, 30000, 120000)
DEFAULT_TRADE_TYPES: tuple[str, ...] = ("TRADE",)
_MIN_PRICE_TOLERANCE = 1e-12


@dataclass(slots=True)
class SequenceTracker:
    """Track monotonic sequence values for one (market, stream) slot."""

    last_seq: Optional[int] = None
    reconnect_needed: bool = False
    require_increment: bool = True

    def observe(self, seq: int) -> tuple[bool, Optional[int]]:
        prev = self.last_seq
        if prev is not None:
            if seq <= prev:
                self.reconnect_needed = True
                self.last_seq = seq
                return False, prev
            if self.require_increment and seq != (prev + 1):
                self.reconnect_needed = True
                self.last_seq = seq
                return False, prev
        self.last_seq = seq
        return True, prev

    def reset(self) -> None:
        self.last_seq = None
        self.reconnect_needed = False


@dataclass(slots=True)
class InferredPaperFill:
    maker_side: str  # BUY or SELL
    fill_px: float
    mid_at_fill: float
    ts_ms: int


@dataclass(slots=True)
class PendingPaperFill:
    ts_ms: int
    maker_side: str
    fill_px: float
    mid_at_fill: float
    pending_horizons_ms: set[int]


@dataclass(slots=True)
class MarketRealtimeState:
    bid_px: Optional[float] = None
    ask_px: Optional[float] = None
    mid_px: Optional[float] = None
    last_bbo_ts_ms: Optional[int] = None
    observed_tick: Optional[float] = None
    mid_series: Deque[tuple[int, float]] = field(default_factory=deque)
    pending_fills: Deque[PendingPaperFill] = field(default_factory=deque)
    orderbook_seq: SequenceTracker = field(default_factory=SequenceTracker)
    trades_seq: SequenceTracker = field(default_factory=SequenceTracker)


@dataclass(slots=True)
class MarketAggregate:
    horizons_ms: Sequence[int]
    fill_count: int = 0
    side_fill_counts: Dict[str, int] = field(default_factory=lambda: {"BUY": 0, "SELL": 0})
    markouts: Dict[int, list[float]] = field(default_factory=dict)
    side_markouts: Dict[int, Dict[str, list[float]]] = field(default_factory=dict)
    toxic_counts: Dict[int, int] = field(default_factory=dict)
    data_quality_warnings: list[str] = field(default_factory=list)
    seq_reset_count: int = 0
    trade_rows_seen: int = 0
    trade_rows_used: int = 0
    bbo_updates_seen: int = 0
    bbo_ready: bool = False
    trade_ts_fallback_count: int = 0

    def __post_init__(self) -> None:
        for horizon_ms in self.horizons_ms:
            self.markouts.setdefault(int(horizon_ms), [])
            self.side_markouts.setdefault(int(horizon_ms), {"BUY": [], "SELL": []})
            self.toxic_counts.setdefault(int(horizon_ms), 0)

    def add_warning(self, warning: str) -> None:
        if warning not in self.data_quality_warnings:
            self.data_quality_warnings.append(warning)

    def add_markout(self, horizon_ms: int, maker_side: str, markout_bps: float) -> None:
        self.markouts[horizon_ms].append(markout_bps)
        self.side_markouts[horizon_ms][maker_side].append(markout_bps)
        if markout_bps < 0:
            self.toxic_counts[horizon_ms] += 1


@dataclass(slots=True)
class PaperMarkoutEngine:
    """Stateful estimator fed by orderbook/trade stream messages."""

    markets: Sequence[str]
    horizons_ms: Sequence[int]
    queue_capture: float
    bbo_match_mode: str
    include_trade_types: Sequence[str]
    use_trade_price_on_match: bool = False
    max_trade_lag_ms: int = 5_000
    mid_series_margin_ms: int = 60_000
    _max_horizon_ms: int = field(init=False, default=0)
    _mid_keep_ms: int = field(init=False, default=0)
    _state: Dict[str, MarketRealtimeState] = field(init=False, default_factory=dict)
    _agg: Dict[str, MarketAggregate] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        norm_markets = []
        seen = set()
        for market in self.markets:
            key = str(market).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            norm_markets.append(key)
        self.markets = norm_markets

        self.horizons_ms = _normalize_horizons(self.horizons_ms)
        self._max_horizon_ms = max(self.horizons_ms) if self.horizons_ms else 0
        self._mid_keep_ms = self._max_horizon_ms + max(1, int(self.mid_series_margin_ms))
        self.queue_capture = max(0.0, float(self.queue_capture))

        mode = str(self.bbo_match_mode or "strict").strip().lower()
        self.bbo_match_mode = mode if mode in {"strict", "loose"} else "strict"

        include = {str(v).strip().upper() for v in self.include_trade_types if str(v).strip()}
        self.include_trade_types = sorted(include or {"TRADE"})
        self.max_trade_lag_ms = max(0, int(self.max_trade_lag_ms))

        self._state: Dict[str, MarketRealtimeState] = {
            market: MarketRealtimeState() for market in self.markets
        }
        self._agg: Dict[str, MarketAggregate] = {
            market: MarketAggregate(horizons_ms=self.horizons_ms) for market in self.markets
        }

    def add_warning_to_all(self, warning: str) -> None:
        for market in self.markets:
            self._agg[market].add_warning(warning)

    def add_warning(self, market: str, warning: str) -> None:
        agg = self._agg.get(market)
        if agg is not None:
            agg.add_warning(warning)

    def observe_orderbook_message(
        self,
        payload: Mapping[str, Any],
        *,
        market_hint: Optional[str] = None,
    ) -> bool:
        """Consume one orderbook WS message.

        Returns True when a sequence fault was detected and reconnect is advised.
        """
        rows = _extract_rows(payload)
        if not rows:
            return False

        reconnect_needed = False
        checked_seq: set[tuple[str, int]] = set()
        seq_faulted_markets: set[str] = set()
        top_ts = _safe_int(payload.get("ts"))

        for row in rows:
            market = str(_get_first(row, "m", "market") or market_hint or "").strip()
            if market not in self._state:
                continue

            seq = _safe_int(_get_first(row, "seq", "s"))
            if seq is None:
                continue

            key = (market, seq)
            if key in checked_seq:
                continue
            checked_seq.add(key)

            ok = self._observe_seq(market=market, stream_name="orderbook", seq=seq)
            if not ok:
                reconnect_needed = True
                seq_faulted_markets.add(market)

        for row in rows:
            market = str(_get_first(row, "m", "market") or market_hint or "").strip()
            if market not in self._state or market in seq_faulted_markets:
                continue

            bid = _extract_book_px(_get_first(row, "b", "bids"))
            ask = _extract_book_px(_get_first(row, "a", "asks"))
            if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
                self.add_warning(market, "invalid_bbo_payload")
                continue

            ts_ms = _safe_int(_get_first(row, "ts", "T", "timestamp"))
            if ts_ms is None:
                ts_ms = top_ts
            if ts_ms is None:
                ts_ms = int(time.time() * 1000)

            self._on_bbo_update(market=market, bid_px=bid, ask_px=ask, ts_ms=ts_ms)

        return reconnect_needed

    def observe_trades_message(
        self,
        payload: Mapping[str, Any],
        *,
        market_hint: Optional[str] = None,
    ) -> bool:
        """Consume one public-trades WS message.

        Returns True when a sequence fault was detected and reconnect is advised.
        """
        rows = _extract_rows(payload)
        if not rows:
            return False

        reconnect_needed = False
        checked_markets: set[str] = set()
        seq_faulted_markets: set[str] = set()
        top_ts = _safe_int(payload.get("ts"))

        for row in rows:
            market = str(_get_first(row, "m", "market") or market_hint or "").strip()
            if market not in self._state or market in checked_markets:
                continue
            checked_markets.add(market)

            seq = _safe_int(_get_first(row, "seq", "s"))
            if seq is None:
                continue

            ok = self._observe_seq(market=market, stream_name="trades", seq=seq)
            if not ok:
                reconnect_needed = True
                seq_faulted_markets.add(market)

        include_types = {t.upper() for t in self.include_trade_types}

        for row in rows:
            market = str(_get_first(row, "m", "market") or market_hint or "").strip()
            if market not in self._state or market in seq_faulted_markets:
                continue

            agg = self._agg[market]
            agg.trade_rows_seen += 1

            trade_type = str(_get_first(row, "tT", "tradeType") or "").strip().upper()
            if include_types and trade_type and trade_type not in include_types:
                continue
            if include_types and not trade_type and "" not in include_types:
                continue

            taker_side = str(_get_first(row, "S", "side") or "").strip().upper()
            trade_px = _safe_float(_get_first(row, "p", "price"))
            if trade_px is None or trade_px <= 0:
                continue

            trade_ts_ms = _safe_int(_get_first(row, "T", "tradeTimestamp", "ts", "timestamp"))
            ts_ms = self._resolve_trade_timestamp(
                market=market,
                trade_ts_ms=trade_ts_ms,
                msg_ts_ms=top_ts,
            )

            state = self._state[market]
            if state.bid_px is None or state.ask_px is None or state.mid_px is None:
                continue

            agg.trade_rows_used += 1
            tol = max((state.observed_tick or 0.0) / 2.0, _MIN_PRICE_TOLERANCE)
            inferred = infer_paper_fill(
                trade_side=taker_side,
                trade_price=trade_px,
                bid_px=state.bid_px,
                ask_px=state.ask_px,
                mid_px=state.mid_px,
                ts_ms=ts_ms,
                bbo_match_mode=self.bbo_match_mode,
                tolerance=tol,
                use_trade_price_on_match=self.use_trade_price_on_match,
            )
            if inferred is None:
                continue

            agg.fill_count += 1
            agg.side_fill_counts[inferred.maker_side] += 1
            state.pending_fills.append(
                PendingPaperFill(
                    ts_ms=inferred.ts_ms,
                    maker_side=inferred.maker_side,
                    fill_px=inferred.fill_px,
                    mid_at_fill=inferred.mid_at_fill,
                    pending_horizons_ms=set(self.horizons_ms),
                )
            )

        return reconnect_needed

    def build_stats(self, duration_s: float) -> Dict[str, Dict[str, Any]]:
        minutes = float(duration_s) / 60.0 if duration_s and duration_s > 0 else 0.0
        out: Dict[str, Dict[str, Any]] = {}

        for market in self.markets:
            agg = self._agg[market]
            fill_rate = (agg.fill_count / minutes) if minutes > 0 else 0.0
            fill_rate_adj = fill_rate * self.queue_capture

            row: Dict[str, Any] = {
                "paper_fills": int(agg.fill_count),
                "paper_fill_rate_per_min": _round_or_none(fill_rate),
                "paper_fill_rate_per_min_adjusted": _round_or_none(fill_rate_adj),
                "paper_queue_capture": _round_or_none(self.queue_capture),
                "data_quality_warnings": list(agg.data_quality_warnings),
                "seq_reset_count": int(agg.seq_reset_count),
                "paper_trade_rows_seen": int(agg.trade_rows_seen),
                "paper_trade_rows_used": int(agg.trade_rows_used),
                "paper_bbo_updates_seen": int(agg.bbo_updates_seen),
                "paper_bbo_ready": bool(agg.bbo_ready),
                "paper_trade_ts_fallback_count": int(agg.trade_ts_fallback_count),
            }

            markout_summary: Dict[str, Dict[str, Any]] = {}
            toxicity_summary: Dict[str, Optional[float]] = {}
            side_summary: Dict[str, Dict[str, Optional[float]]] = {}

            for horizon_ms in self.horizons_ms:
                label = horizon_label(horizon_ms)
                values = agg.markouts[horizon_ms]
                count = len(values)
                mean_val = (sum(values) / count) if count else None
                med_val = median(values) if values else None
                tox_val = (agg.toxic_counts[horizon_ms] / count) if count else None

                buy_values = agg.side_markouts[horizon_ms]["BUY"]
                sell_values = agg.side_markouts[horizon_ms]["SELL"]
                buy_mean = (sum(buy_values) / len(buy_values)) if buy_values else None
                sell_mean = (sum(sell_values) / len(sell_values)) if sell_values else None

                row[f"paper_markout_bps_{label}_mean"] = _round_or_none(mean_val)
                row[f"paper_markout_bps_{label}_median"] = _round_or_none(med_val)
                row[f"paper_markout_bps_{label}_count"] = count
                row[f"paper_toxicity_share_{label}"] = _round_or_none(tox_val)
                row[f"paper_markout_buy_bps_{label}_mean"] = _round_or_none(buy_mean)
                row[f"paper_markout_sell_bps_{label}_mean"] = _round_or_none(sell_mean)

                markout_summary[label] = {
                    "mean": _round_or_none(mean_val),
                    "median": _round_or_none(med_val),
                    "count": count,
                }
                toxicity_summary[label] = _round_or_none(tox_val)
                side_summary[label] = {
                    "buy_mean": _round_or_none(buy_mean),
                    "sell_mean": _round_or_none(sell_mean),
                }

            row["paper_markout"] = markout_summary
            row["paper_toxicity"] = toxicity_summary
            row["paper_markout_side_split"] = side_summary
            out[market] = row

        return out

    def _observe_seq(self, *, market: str, stream_name: str, seq: int) -> bool:
        state = self._state[market]
        tracker = state.orderbook_seq if stream_name == "orderbook" else state.trades_seq
        ok, prev = tracker.observe(seq)
        if ok:
            return True

        agg = self._agg[market]
        agg.seq_reset_count += 1
        agg.add_warning(f"{stream_name}_seq_gap prev={prev} current={seq}")
        self._reset_market_runtime_state(market)
        return False

    def _reset_market_runtime_state(self, market: str) -> None:
        state = self._state[market]
        state.bid_px = None
        state.ask_px = None
        state.mid_px = None
        state.last_bbo_ts_ms = None
        state.observed_tick = None
        state.mid_series.clear()
        state.pending_fills.clear()
        state.orderbook_seq.reset()
        state.trades_seq.reset()

    def _on_bbo_update(self, *, market: str, bid_px: float, ask_px: float, ts_ms: int) -> None:
        agg = self._agg[market]
        agg.bbo_updates_seen += 1
        state = self._state[market]

        if state.bid_px is not None:
            d_bid = abs(bid_px - state.bid_px)
            if d_bid > 0:
                state.observed_tick = d_bid if state.observed_tick is None else min(state.observed_tick, d_bid)
        if state.ask_px is not None:
            d_ask = abs(ask_px - state.ask_px)
            if d_ask > 0:
                state.observed_tick = d_ask if state.observed_tick is None else min(state.observed_tick, d_ask)

        mid_px = (bid_px + ask_px) / 2.0
        if mid_px <= 0:
            self.add_warning(market, "non_positive_mid")
            return

        if state.last_bbo_ts_ms is not None and ts_ms < state.last_bbo_ts_ms:
            self.add_warning(market, "out_of_order_bbo_timestamp")
            return

        state.bid_px = bid_px
        state.ask_px = ask_px
        state.mid_px = mid_px
        state.last_bbo_ts_ms = ts_ms
        agg.bbo_ready = True

        if state.mid_series and state.mid_series[-1][0] == ts_ms:
            state.mid_series[-1] = (ts_ms, mid_px)
        else:
            state.mid_series.append((ts_ms, mid_px))

        cutoff = ts_ms - self._mid_keep_ms
        while state.mid_series and state.mid_series[0][0] < cutoff:
            state.mid_series.popleft()

        self._resolve_pending_fills(market=market, up_to_ts_ms=ts_ms)

    def _resolve_pending_fills(self, *, market: str, up_to_ts_ms: int) -> None:
        state = self._state[market]
        agg = self._agg[market]
        if not state.pending_fills:
            return

        next_pending: Deque[PendingPaperFill] = deque()
        for pending in state.pending_fills:
            ready_horizons = [
                h for h in pending.pending_horizons_ms
                if pending.ts_ms + h <= up_to_ts_ms
            ]

            for horizon_ms in sorted(ready_horizons):
                target_ts = pending.ts_ms + horizon_ms
                pending.pending_horizons_ms.discard(horizon_ms)
                mid_future = _mid_at_or_before(state.mid_series, target_ts)
                if mid_future is None:
                    continue
                markout_bps = compute_markout_bps(
                    maker_side=pending.maker_side,
                    fill_px=pending.fill_px,
                    mid_at_fill=pending.mid_at_fill,
                    mid_future=mid_future,
                )
                agg.add_markout(horizon_ms=horizon_ms, maker_side=pending.maker_side, markout_bps=markout_bps)

            if pending.pending_horizons_ms:
                next_pending.append(pending)

        state.pending_fills = next_pending

    def _resolve_trade_timestamp(
        self,
        *,
        market: str,
        trade_ts_ms: Optional[int],
        msg_ts_ms: Optional[int],
    ) -> int:
        if trade_ts_ms is not None and msg_ts_ms is not None:
            if abs(msg_ts_ms - trade_ts_ms) <= self.max_trade_lag_ms:
                return trade_ts_ms
            agg = self._agg[market]
            agg.trade_ts_fallback_count += 1
            self.add_warning(
                market,
                f"trade_ts_fallback_to_msg_ts(max_lag_ms={self.max_trade_lag_ms})",
            )
            return msg_ts_ms
        if trade_ts_ms is not None:
            return trade_ts_ms
        if msg_ts_ms is not None:
            return msg_ts_ms
        return int(time.time() * 1000)

    def missing_bbo_markets(self) -> list[str]:
        return [
            market
            for market in self.markets
            if self._state.get(market) is None or self._state[market].mid_px is None
        ]


class _ReconnectRequested(RuntimeError):
    pass


def infer_paper_fill(
    *,
    trade_side: str,
    trade_price: float,
    bid_px: float,
    ask_px: float,
    mid_px: float,
    ts_ms: int,
    bbo_match_mode: str,
    tolerance: float = _MIN_PRICE_TOLERANCE,
    use_trade_price_on_match: bool = False,
) -> Optional[InferredPaperFill]:
    """Infer a hypothetical maker fill from a public trade."""
    side = str(trade_side or "").strip().upper()
    mode = str(bbo_match_mode or "strict").strip().lower()
    tol = max(float(tolerance), _MIN_PRICE_TOLERANCE)

    if side == "BUY":
        maker_side = "SELL"
        bbo_px = ask_px
        strict_match = abs(trade_price - ask_px) <= tol
        loose_match = trade_price >= (ask_px - tol)
    elif side == "SELL":
        maker_side = "BUY"
        bbo_px = bid_px
        strict_match = abs(trade_price - bid_px) <= tol
        loose_match = trade_price <= (bid_px + tol)
    else:
        return None

    matched = strict_match if mode == "strict" else loose_match
    if not matched:
        return None

    fill_px = float(trade_price) if use_trade_price_on_match else float(bbo_px)
    return InferredPaperFill(
        maker_side=maker_side,
        fill_px=fill_px,
        mid_at_fill=float(mid_px),
        ts_ms=int(ts_ms),
    )


def compute_markout_bps(
    *,
    maker_side: str,
    fill_px: float,
    mid_at_fill: float,
    mid_future: float,
) -> float:
    denom = float(mid_at_fill)
    if denom <= 0:
        return 0.0

    if str(maker_side).upper() == "BUY":
        pnl_px = float(mid_future) - float(fill_px)
    else:
        pnl_px = float(fill_px) - float(mid_future)
    return (pnl_px / denom) * 10_000.0


def horizon_label(horizon_ms: int) -> str:
    ms = int(horizon_ms)
    if ms < 1000:
        return f"{ms}ms"
    if ms % 60_000 == 0:
        return f"{ms // 60_000}m"
    if ms % 1000 == 0:
        return f"{ms // 1000}s"
    return f"{ms}ms"


def run_paper_markout(
    markets: list[str],
    duration_s: float,
    horizons_ms: list[int],
    queue_capture: float,
    bbo_match_mode: str,
    include_trade_types: list[str],
    *,
    api_base: Optional[str] = None,
    prefer_all_markets: bool = True,
    use_trade_price_on_match: bool = False,
    warmup_s: float = 8.0,
    max_trade_lag_ms: int = 5_000,
) -> Dict[str, Dict[str, Any]]:
    """Run paper markout sampling over public WS streams.

    Returns a market -> stats mapping. This function never submits orders.
    """
    market_list = [str(m).strip() for m in markets if str(m).strip()]
    if not market_list:
        return {}

    engine = PaperMarkoutEngine(
        markets=market_list,
        horizons_ms=horizons_ms,
        queue_capture=queue_capture,
        bbo_match_mode=bbo_match_mode,
        include_trade_types=include_trade_types,
        use_trade_price_on_match=use_trade_price_on_match,
        max_trade_lag_ms=max_trade_lag_ms,
    )

    duration_s = max(0.0, float(duration_s))
    if duration_s <= 0:
        return engine.build_stats(duration_s=duration_s)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop in this thread: safe to call asyncio.run below.
        pass
    else:
        engine.add_warning_to_all(
            "paper_markout_runtime_error:run_paper_markout cannot run inside an active event loop"
        )
        return engine.build_stats(duration_s=duration_s)

    stream_base = _resolve_stream_base(api_base)

    async def _runner() -> None:
        try:
            await _collect_streams(
                engine=engine,
                stream_base=stream_base,
                duration_s=duration_s,
                prefer_all_markets=prefer_all_markets,
                warmup_s=warmup_s,
            )
        except Exception as collect_exc:  # pragma: no cover - network/runtime safety path
            engine.add_warning_to_all(
                f"paper_markout_collect_error:{type(collect_exc).__name__}:{collect_exc}"
            )

    asyncio.run(_runner())
    return engine.build_stats(duration_s=duration_s)


async def _collect_streams(
    *,
    engine: PaperMarkoutEngine,
    stream_base: str,
    duration_s: float,
    prefer_all_markets: bool,
    warmup_s: float,
) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_s))

    try:
        import websockets  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        engine.add_warning_to_all(f"websockets_import_failed:{type(exc).__name__}:{exc}")
        return

    connect = websockets.connect

    use_all_markets = False
    if prefer_all_markets and len(engine.markets) > 1:
        use_all_markets = await _probe_all_market_support(
            stream_base=stream_base,
            connect=connect,
            deadline=deadline,
            engine=engine,
        )

    orderbook_tasks: list[asyncio.Task[None]] = []
    trade_tasks: list[asyncio.Task[None]] = []
    if use_all_markets:
        orderbook_tasks.append(
            asyncio.create_task(
                _stream_loop(
                    engine=engine,
                    url=f"{stream_base}/orderbooks?depth=1",
                    stream_name="orderbook",
                    deadline=deadline,
                    connect=connect,
                )
            )
        )
        trade_tasks.append(
            asyncio.create_task(
                _stream_loop(
                    engine=engine,
                    url=f"{stream_base}/publicTrades",
                    stream_name="trades",
                    deadline=deadline,
                    connect=connect,
                )
            )
        )
    else:
        for market in engine.markets:
            encoded = quote(market, safe="-_.")
            orderbook_tasks.append(
                asyncio.create_task(
                    _stream_loop(
                        engine=engine,
                        url=f"{stream_base}/orderbooks/{encoded}?depth=1",
                        stream_name="orderbook",
                        deadline=deadline,
                        connect=connect,
                        market_hint=market,
                    )
                )
            )
            trade_tasks.append(
                asyncio.create_task(
                    _stream_loop(
                        engine=engine,
                        url=f"{stream_base}/publicTrades/{encoded}",
                        stream_name="trades",
                        deadline=deadline,
                        connect=connect,
                        market_hint=market,
                    )
                )
            )

    if orderbook_tasks:
        await _wait_for_bbo_warmup(
            engine=engine,
            deadline=deadline,
            warmup_s=warmup_s,
        )
    tasks = [*orderbook_tasks, *trade_tasks]
    if tasks:
        await asyncio.gather(*tasks)


async def _wait_for_bbo_warmup(
    *,
    engine: PaperMarkoutEngine,
    deadline: float,
    warmup_s: float,
) -> None:
    warmup_s = max(0.0, float(warmup_s))
    if warmup_s <= 0:
        return
    warmup_deadline = min(deadline, time.monotonic() + warmup_s)
    while time.monotonic() < warmup_deadline:
        if not engine.missing_bbo_markets():
            return
        await asyncio.sleep(0.1)
    missing = engine.missing_bbo_markets()
    if missing:
        engine.add_warning_to_all(
            "paper_markout_bbo_warmup_timeout_missing=" + ",".join(missing)
        )


async def _probe_all_market_support(
    *,
    stream_base: str,
    connect: Any,
    deadline: float,
    engine: PaperMarkoutEngine,
) -> bool:
    urls = [
        f"{stream_base}/orderbooks?depth=1",
        f"{stream_base}/publicTrades",
    ]

    for url in urls:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        try:
            async with connect(url, open_timeout=min(5.0, max(1.0, remaining))):
                pass
        except Exception:
            engine.add_warning_to_all("all_market_stream_probe_failed_falling_back_to_per_market")
            return False
    return True


async def _stream_loop(
    *,
    engine: PaperMarkoutEngine,
    url: str,
    stream_name: str,
    deadline: float,
    connect: Any,
    market_hint: Optional[str] = None,
) -> None:
    backoff_s = 0.5

    while time.monotonic() < deadline:
        try:
            async with connect(
                url,
                open_timeout=10,
                close_timeout=5,
                ping_interval=20,
                ping_timeout=20,
                max_size=2**22,
            ) as ws:
                backoff_s = 0.5
                stream_seq = SequenceTracker(require_increment=True)
                while time.monotonic() < deadline:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break

                    payload = _decode_json_payload(raw)
                    if payload is None:
                        continue

                    seq = _safe_int(payload.get("seq"))
                    if seq is not None:
                        ok, prev = stream_seq.observe(seq)
                        if not ok:
                            warning = f"{stream_name}_stream_seq_gap prev={prev} current={seq}"
                            if market_hint:
                                engine.add_warning(market_hint, warning)
                            else:
                                engine.add_warning_to_all(warning)
                            raise _ReconnectRequested(warning)

                    reconnect = (
                        engine.observe_orderbook_message(payload, market_hint=market_hint)
                        if stream_name == "orderbook"
                        else engine.observe_trades_message(payload, market_hint=market_hint)
                    )
                    if reconnect:
                        raise _ReconnectRequested(f"{stream_name}_market_seq_gap")

        except _ReconnectRequested:
            wait_s = min(backoff_s, max(0.0, deadline - time.monotonic()))
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            backoff_s = min(backoff_s * 2.0, 10.0)
            continue
        except Exception as exc:
            warning = f"{stream_name}_stream_error:{type(exc).__name__}:{exc}"
            if market_hint:
                engine.add_warning(market_hint, warning)
            else:
                engine.add_warning_to_all(warning)
            wait_s = min(backoff_s, max(0.0, deadline - time.monotonic()))
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            backoff_s = min(backoff_s * 2.0, 10.0)


def _resolve_stream_base(api_base: Optional[str]) -> str:
    explicit = str(os.getenv("EXTENDED_STREAM_BASE", "")).strip()
    if explicit:
        return explicit.rstrip("/")

    base = str(api_base or resolve_default_api_base()).strip()
    parsed = urlparse(base)
    if not parsed.scheme or not parsed.netloc:
        host = base.replace("https://", "").replace("http://", "").strip("/")
        return f"wss://{host}/stream.extended.exchange/v1"

    scheme = "wss" if parsed.scheme in {"https", "wss"} else "ws"
    return f"{scheme}://{parsed.netloc}/stream.extended.exchange/v1"


def _extract_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = payload.get("data")
    rows: list[Mapping[str, Any]] = []

    if isinstance(data, dict):
        rows.append(data)
    elif isinstance(data, list):
        rows.extend(item for item in data if isinstance(item, dict))

    if rows:
        return rows

    if isinstance(payload, dict):
        market = payload.get("m")
        bids = payload.get("b")
        asks = payload.get("a")
        side = payload.get("S")
        if market is not None and (bids is not None or asks is not None or side is not None):
            return [payload]

    return []


def _extract_book_px(levels: Any) -> Optional[float]:
    if not isinstance(levels, list) or not levels:
        return None
    top = levels[0]
    if isinstance(top, dict):
        return _safe_float(_get_first(top, "p", "price"))
    if isinstance(top, (list, tuple)) and top:
        return _safe_float(top[0])
    return None


def _mid_at_or_before(series: Deque[tuple[int, float]], target_ts: int) -> Optional[float]:
    for ts_ms, mid_px in reversed(series):
        if ts_ms <= target_ts:
            return mid_px
    return None


def _decode_json_payload(raw: Any) -> Optional[Mapping[str, Any]]:
    text: str
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="ignore")
    elif isinstance(raw, str):
        text = raw
    elif isinstance(raw, dict):
        return raw
    else:
        return None

    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_horizons(raw: Iterable[int] | Sequence[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in raw:
        ms = _safe_int(value)
        if ms is None or ms <= 0 or ms in seen:
            continue
        seen.add(ms)
        out.append(ms)
    return out or list(DEFAULT_HORIZONS_MS)


def _safe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not (out == out):
        return None
    return out


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_first(obj: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def _round_or_none(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)
