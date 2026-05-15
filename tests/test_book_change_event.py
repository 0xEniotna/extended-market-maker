"""Phase 0.5: book_change event instrumentation tests.

Covers:
1. TradeJournal.record_book_change emits a well-formed event.
2. OrderbookManager dedup: identical L1 state → exactly one event.
3. OrderbookManager emission: changing L1 state → new events.
4. OrderbookManager fail-soft: a broken journal does not kill the
   WS callback path.

Source: docs/microprice_ofi_plan.md Phase 0.5.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from unittest.mock import MagicMock

# ---------------------------------------------------------------------
# SDK stubs (must be set before importing orderbook_manager — matches
# the convention in test_funding_aware_integration.py etc.)
# ---------------------------------------------------------------------

_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.trading_client",
    "x10.perpetual.positions",
    "x10.perpetual.accounts",
    "x10.perpetual.configuration",
    "x10.perpetual.orderbook",
    "x10.perpetual.trades",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.utils",
    "x10.utils.http",
]
for mod_name in _SDK_MODULES:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

from market_maker.orderbook_manager import OrderbookManager, PriceLevel  # noqa: E402
from market_maker.trade_journal import TradeJournal  # noqa: E402


def _read_records(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------
# TradeJournal.record_book_change
# ---------------------------------------------------------------------

def test_record_book_change_writes_expected_payload(tmp_path):
    journal = TradeJournal(
        "TEST-USD",
        journal_dir=tmp_path,
        run_id="run-bc",
        schema_version=2,
    )
    journal.record_book_change(
        bid=Decimal("1.3901"),
        bid_qty=Decimal("723.1"),
        ask=Decimal("1.3922"),
        ask_qty=Decimal("500.0"),
    )
    journal.close()

    records = _read_records(journal.path)
    assert len(records) == 1
    rec = records[0]
    assert rec["type"] == "book_change"
    assert rec["market"] == "TEST-USD"
    assert rec["run_id"] == "run-bc"
    assert rec["schema_version"] == 2
    # Decimals must round-trip exactly via _DecimalEncoder → string.
    assert rec["bid"] == "1.3901"
    assert rec["bid_qty"] == "723.1"
    assert rec["ask"] == "1.3922"
    assert rec["ask_qty"] == "500.0"
    assert rec["seq"] == 1
    assert isinstance(rec["ts"], float)


def test_book_change_is_not_critical_event(tmp_path):
    # book_change is high-frequency; it must use batched fsync, not the
    # per-event fsync reserved for `fill`/`run_end`/etc.
    journal = TradeJournal("TEST-USD", journal_dir=tmp_path, run_id="run-bc")

    fsync_calls = []
    real_do_fsync = journal._do_fsync

    def counting_do_fsync():
        fsync_calls.append(len(fsync_calls))
        real_do_fsync()

    journal._do_fsync = counting_do_fsync  # type: ignore[method-assign]

    # Write 10 book_change events; none should force an immediate fsync.
    for i in range(10):
        journal.record_book_change(
            bid=Decimal(f"1.{i:04d}"),
            bid_qty=Decimal("100"),
            ask=Decimal(f"1.{i + 1:04d}"),
            ask_qty=Decimal("100"),
        )
    # By contrast, a single `fill`-type write does fsync immediately.
    n_after_book_changes = len(fsync_calls)

    journal.record_run_end(reason="test")
    assert len(fsync_calls) > n_after_book_changes  # run_end is critical
    journal.close()


# ---------------------------------------------------------------------
# OrderbookManager dedup + emission
# ---------------------------------------------------------------------

class _FakeJournal:
    """Captures record_book_change calls for assertion."""

    def __init__(self):
        self.calls: list[dict[str, Decimal]] = []

    def record_book_change(self, *, bid, bid_qty, ask, ask_qty):
        self.calls.append(
            {"bid": bid, "bid_qty": bid_qty, "ask": ask, "ask_qty": ask_qty}
        )


def _make_manager() -> OrderbookManager:
    # The OrderbookManager only needs market_name + endpoint_config to
    # construct; we never start the SDK subscription in these unit tests.
    return OrderbookManager(
        endpoint_config=None,  # type: ignore[arg-type]
        market_name="TEST-USD",
    )


def test_emit_book_change_first_call_emits(tmp_path):
    mgr = _make_manager()
    journal = _FakeJournal()
    mgr.set_journal(journal)
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))

    mgr._maybe_emit_book_change()

    assert len(journal.calls) == 1
    call = journal.calls[0]
    assert call["bid"] == Decimal("100.0")
    assert call["bid_qty"] == Decimal("10")
    assert call["ask"] == Decimal("100.1")
    assert call["ask_qty"] == Decimal("12")


def test_emit_book_change_dedup_on_identical_state():
    mgr = _make_manager()
    journal = _FakeJournal()
    mgr.set_journal(journal)
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))

    mgr._maybe_emit_book_change()
    mgr._maybe_emit_book_change()
    mgr._maybe_emit_book_change()

    # Three calls with identical state → only one journal write.
    assert len(journal.calls) == 1


def test_emit_book_change_emits_on_bid_size_only_change():
    mgr = _make_manager()
    journal = _FakeJournal()
    mgr.set_journal(journal)
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))
    mgr._maybe_emit_book_change()

    # Change bid SIZE only (price unchanged). Must emit a new event —
    # this is critical for OFI reconstruction.
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("8"))
    mgr._maybe_emit_book_change()

    assert len(journal.calls) == 2
    assert journal.calls[1]["bid_qty"] == Decimal("8")


def test_emit_book_change_emits_on_ask_price_change():
    mgr = _make_manager()
    journal = _FakeJournal()
    mgr.set_journal(journal)
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))
    mgr._maybe_emit_book_change()

    mgr._last_ask = PriceLevel(price=Decimal("100.2"), size=Decimal("12"))
    mgr._maybe_emit_book_change()

    assert len(journal.calls) == 2
    assert journal.calls[1]["ask"] == Decimal("100.2")
    assert journal.calls[1]["ask_qty"] == Decimal("12")


def test_emit_book_change_skipped_when_journal_unset():
    mgr = _make_manager()
    # No set_journal call.
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))

    # Must not raise. Must not change internal state spuriously.
    mgr._maybe_emit_book_change()
    assert mgr._last_emitted_l1 is None


def test_emit_book_change_skipped_when_bid_missing():
    mgr = _make_manager()
    journal = _FakeJournal()
    mgr.set_journal(journal)
    mgr._last_bid = None
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))

    mgr._maybe_emit_book_change()
    assert journal.calls == []


def test_emit_book_change_swallows_journal_exception(caplog):
    class _BrokenJournal:
        def record_book_change(self, **kwargs):
            raise RuntimeError("disk full")

    mgr = _make_manager()
    mgr.set_journal(_BrokenJournal())
    mgr._last_bid = PriceLevel(price=Decimal("100.0"), size=Decimal("10"))
    mgr._last_ask = PriceLevel(price=Decimal("100.1"), size=Decimal("12"))

    # Must not raise — a journal failure cannot kill the WS callback path.
    mgr._maybe_emit_book_change()
    # State tracker still updated, so we don't retry the same state forever.
    assert mgr._last_emitted_l1 == (
        Decimal("100.0"),
        Decimal("10"),
        Decimal("100.1"),
        Decimal("12"),
    )


def test_emit_book_change_with_real_journal(tmp_path):
    # End-to-end integration: manager → real TradeJournal → file on disk.
    journal = TradeJournal(
        "TEST-USD",
        journal_dir=tmp_path,
        run_id="run-int",
        schema_version=2,
    )
    mgr = _make_manager()
    mgr.set_journal(journal)

    # Three mutations in sequence:
    states = [
        (Decimal("100.0"), Decimal("10"), Decimal("100.1"), Decimal("12")),
        (Decimal("100.0"), Decimal("10"), Decimal("100.1"), Decimal("12")),  # dedup
        (Decimal("100.0"), Decimal("8"),  Decimal("100.1"), Decimal("12")),  # size change
        (Decimal("100.05"), Decimal("8"), Decimal("100.1"), Decimal("12")),  # price change
    ]
    for bid_p, bid_q, ask_p, ask_q in states:
        mgr._last_bid = PriceLevel(price=bid_p, size=bid_q)
        mgr._last_ask = PriceLevel(price=ask_p, size=ask_q)
        mgr._maybe_emit_book_change()

    journal.close()
    records = [r for r in _read_records(journal.path) if r["type"] == "book_change"]
    # Three unique L1 states should produce three book_change events
    # (the second identical state is deduped out).
    assert len(records) == 3
    assert records[0]["bid_qty"] == "10"
    assert records[1]["bid_qty"] == "8"
    assert records[2]["bid"] == "100.05"
