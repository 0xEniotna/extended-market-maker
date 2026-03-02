from __future__ import annotations

import asyncio
import json
import time

import pytest

from market_maker.scout.paper_markout import (
    PaperMarkoutEngine,
    SequenceTracker,
    _collect_streams,
    _stream_loop,
    compute_markout_bps,
    infer_paper_fill,
)


def _book(seq: int, ts_ms: int, market: str, bid: float, ask: float) -> dict:
    return {
        "seq": seq,
        "ts": ts_ms,
        "data": {
            "m": market,
            "b": [{"p": str(bid), "q": "1"}],
            "a": [{"p": str(ask), "q": "1"}],
        },
    }


def _trade(seq: int, market: str, taker_side: str, ts_ms: int, px: float, t_type: str = "TRADE") -> dict:
    return {
        "seq": seq,
        "data": [
            {
                "m": market,
                "S": taker_side,
                "tT": t_type,
                "T": ts_ms,
                "p": str(px),
                "q": "1",
            }
        ],
    }


def test_paper_fill_inference_side_and_fill_price_from_bbo():
    market = "TEST-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )

    assert engine.observe_orderbook_message(_book(1, 1000, market, 99, 101)) is False
    assert engine.observe_trades_message(_trade(1, market, "BUY", 1000, 101)) is False
    assert engine.observe_trades_message(_trade(2, market, "SELL", 1000, 99)) is False
    assert engine.observe_orderbook_message(_book(2, 1250, market, 100, 100)) is False

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["paper_fills"] == 2
    assert stats["paper_markout_sell_bps_250ms_mean"] == 100.0
    assert stats["paper_markout_buy_bps_250ms_mean"] == 100.0


def test_strict_vs_loose_bbo_match_behavior():
    strict_fill = infer_paper_fill(
        trade_side="BUY",
        trade_price=101.1,
        bid_px=99.0,
        ask_px=101.0,
        mid_px=100.0,
        ts_ms=1234,
        bbo_match_mode="strict",
    )
    loose_fill = infer_paper_fill(
        trade_side="BUY",
        trade_price=101.1,
        bid_px=99.0,
        ask_px=101.0,
        mid_px=100.0,
        ts_ms=1234,
        bbo_match_mode="loose",
    )

    assert strict_fill is None
    assert loose_fill is not None
    assert loose_fill.maker_side == "SELL"
    # Default fill price is current BBO side price, not trade price.
    assert loose_fill.fill_px == 101.0


def test_markout_bps_formula_for_buy_and_sell():
    buy_markout = compute_markout_bps(
        maker_side="BUY",
        fill_px=99.0,
        mid_at_fill=100.0,
        mid_future=101.0,
    )
    sell_markout = compute_markout_bps(
        maker_side="SELL",
        fill_px=101.0,
        mid_at_fill=100.0,
        mid_future=99.0,
    )

    assert buy_markout == 200.0
    assert sell_markout == 200.0


def test_missing_future_mid_excludes_horizon_markout():
    market = "MISS-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[1000],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )

    engine.observe_orderbook_message(_book(1, 1000, market, 99, 101))
    engine.observe_trades_message(_trade(1, market, "SELL", 1000, 99))
    engine.observe_orderbook_message(_book(2, 1500, market, 99, 101))

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["paper_fills"] == 1
    assert stats["paper_markout_bps_1s_count"] == 0
    assert stats["paper_markout_bps_1s_mean"] is None


def test_toxicity_share_aggregation():
    market = "TOX-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )

    engine.observe_orderbook_message(_book(1, 1000, market, 99, 101))
    engine.observe_trades_message(_trade(1, market, "SELL", 1000, 99))

    engine.observe_orderbook_message(_book(2, 1250, market, 100, 100))
    engine.observe_orderbook_message(_book(3, 1300, market, 99, 101))
    engine.observe_trades_message(_trade(2, market, "SELL", 1300, 99))
    engine.observe_orderbook_message(_book(4, 1550, market, 97, 99))

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["paper_markout_bps_250ms_count"] == 2
    assert stats["paper_toxicity_share_250ms"] == 0.5


def test_seq_gap_marks_reconnect_needed():
    tracker = SequenceTracker()
    assert tracker.observe(10)[0] is True
    ok, prev = tracker.observe(12)
    assert ok is False
    assert prev == 10
    assert tracker.reconnect_needed is True

    market = "SEQ-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )
    row_seq_1 = _book(1, 1000, market, 99, 101)
    row_seq_1["data"]["seq"] = 1
    row_seq_3 = _book(3, 1010, market, 99, 101)
    row_seq_3["data"]["seq"] = 3
    assert engine.observe_orderbook_message(row_seq_1) is False
    # Seq gap should request reconnect and increment reset count.
    assert engine.observe_orderbook_message(row_seq_3) is True

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["seq_reset_count"] == 1


def test_top_level_seq_is_not_enforced_per_market_without_row_seq():
    engine = PaperMarkoutEngine(
        markets=["AAA-USD", "BBB-USD"],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )
    msg_a = {
        "seq": 100,
        "ts": 1000,
        "data": {"m": "AAA-USD", "b": [{"p": "99", "q": "1"}], "a": [{"p": "101", "q": "1"}]},
    }
    msg_b = {
        "seq": 250,
        "ts": 1001,
        "data": {"m": "BBB-USD", "b": [{"p": "49", "q": "1"}], "a": [{"p": "51", "q": "1"}]},
    }
    assert engine.observe_orderbook_message(msg_a) is False
    assert engine.observe_orderbook_message(msg_b) is False

    a_stats = engine.build_stats(duration_s=30.0)["AAA-USD"]
    b_stats = engine.build_stats(duration_s=30.0)["BBB-USD"]
    assert a_stats["seq_reset_count"] == 0
    assert b_stats["seq_reset_count"] == 0
    assert a_stats["paper_bbo_ready"] is True
    assert b_stats["paper_bbo_ready"] is True


def test_trade_timestamp_falls_back_to_message_ts_when_trade_t_is_stale():
    market = "LAG-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
        max_trade_lag_ms=100,
    )
    engine.observe_orderbook_message(_book(1, 1000, market, 99, 101))
    stale_trade = {
        "seq": 1,
        "ts": 10_000,
        "data": [{"m": market, "S": "SELL", "tT": "TRADE", "T": 1000, "p": "99", "q": "1"}],
    }
    engine.observe_trades_message(stale_trade)
    engine.observe_orderbook_message(_book(2, 1250, market, 100, 100))

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["paper_fills"] == 1
    assert stats["paper_markout_bps_250ms_count"] == 0
    assert stats["paper_trade_ts_fallback_count"] == 1
    assert "trade_ts_fallback_to_msg_ts(max_lag_ms=100)" in stats["data_quality_warnings"]


def test_trade_timestamp_uses_trade_t_when_lag_is_within_threshold():
    market = "FRESH-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
        max_trade_lag_ms=50,
    )
    engine.observe_orderbook_message(_book(1, 1000, market, 99, 101))
    fresh_trade = {
        "seq": 1,
        "ts": 1040,
        "data": [{"m": market, "S": "SELL", "tT": "TRADE", "T": 1000, "p": "99", "q": "1"}],
    }
    engine.observe_trades_message(fresh_trade)
    engine.observe_orderbook_message(_book(2, 1270, market, 100, 100))

    stats = engine.build_stats(duration_s=60.0)[market]
    assert stats["paper_fills"] == 1
    assert stats["paper_markout_bps_250ms_count"] == 1
    assert stats["paper_trade_ts_fallback_count"] == 0


def test_stream_level_seq_gap_adds_warning_and_requests_reconnect():
    market = "SEQ2-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )

    class _FakeWS:
        def __init__(self, messages):
            self._messages = list(messages)

        async def recv(self):
            if self._messages:
                return self._messages.pop(0)
            await asyncio.sleep(1.0)
            return ""

    class _FakeConn:
        def __init__(self, messages):
            self._ws = _FakeWS(messages)

        async def __aenter__(self):
            return self._ws

        async def __aexit__(self, *_):
            return False

    calls = {"count": 0}

    def _connect(*_, **__):
        calls["count"] += 1
        if calls["count"] == 1:
            payloads = [
                json.dumps(_book(1, 1000, market, 99, 101)),
                json.dumps(_book(3, 1010, market, 99, 101)),
            ]
        else:
            payloads = []
        return _FakeConn(payloads)

    asyncio.run(
        _stream_loop(
            engine=engine,
            url="wss://example.invalid/orderbooks/SEQ2-USD?depth=1",
            stream_name="orderbook",
            deadline=time.monotonic() + 0.25,
            connect=_connect,
            market_hint=market,
        )
    )
    stats = engine.build_stats(duration_s=60.0)[market]
    assert any(w.startswith("orderbook_stream_seq_gap") for w in stats["data_quality_warnings"])


def test_collect_streams_warms_orderbook_before_starting_trades(monkeypatch):
    events: list[str] = []
    market = "WARM-USD"
    engine = PaperMarkoutEngine(
        markets=[market],
        horizons_ms=[250],
        queue_capture=0.2,
        bbo_match_mode="strict",
        include_trade_types=["TRADE"],
    )

    async def _fake_probe(**_kwargs):
        return False

    async def _fake_stream_loop(*, stream_name, market_hint=None, **_kwargs):
        events.append(stream_name)
        if stream_name == "orderbook" and market_hint:
            engine._on_bbo_update(market=market_hint, bid_px=99.0, ask_px=101.0, ts_ms=1000)
        await asyncio.sleep(0)

    monkeypatch.setattr("market_maker.scout.paper_markout._probe_all_market_support", _fake_probe)
    monkeypatch.setattr("market_maker.scout.paper_markout._stream_loop", _fake_stream_loop)

    asyncio.run(
        _collect_streams(
            engine=engine,
            stream_base="wss://example.invalid/stream.extended.exchange/v1",
            duration_s=0.2,
            prefer_all_markets=False,
            warmup_s=0.1,
        )
    )
    assert events
    assert events[0] == "orderbook"
    assert "trades" in events
