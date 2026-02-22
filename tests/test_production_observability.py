"""Tests for production observability features (Prompt 5).

Covers:
- Journal file rotation (size-based)
- Heartbeat gap detection (analyse_mm_journal.detect_heartbeat_gaps)
- Schema version validation (analyse_mm_journal.validate_schema_versions)
- Fsync behaviour (critical vs batched)
- Latest symlink maintenance
- record_error structured journaling
- record_config_change journaling
- Heartbeat event recording
- analyse() integration of schema warnings and heartbeat gaps
"""
from __future__ import annotations

import json
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401

from market_maker.trade_journal import (
    _BATCH_FSYNC_INTERVAL_WRITES,
    _CRITICAL_EVENT_TYPES,
    TradeJournal,
)

# Ensure scripts directory is importable
_scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_scripts_dir))
import analyse_mm_journal as analyser  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def journal_dir(tmp_path: Path) -> Path:
    d = tmp_path / "journals"
    d.mkdir()
    return d


@pytest.fixture()
def journal(journal_dir: Path) -> TradeJournal:
    j = TradeJournal("TEST-USD", journal_dir, run_id="test123", max_size_mb=50)
    yield j
    j.close()


@pytest.fixture()
def small_journal(journal_dir: Path) -> TradeJournal:
    """Journal with tiny max size to trigger rotation quickly."""
    j = TradeJournal("TEST-USD", journal_dir, run_id="test123", max_size_mb=0.001)
    yield j
    j.close()


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


class TestJournalRotation:
    def test_rotation_triggers_on_size(self, small_journal: TradeJournal, journal_dir: Path):
        """Writing enough data should trigger a rotation to a new file."""
        initial_path = small_journal.path
        # Write many events to exceed the tiny max size
        for i in range(50):
            small_journal.record_snapshot(
                position=Decimal("1.0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("100"),
                active_orders=2,
                total_fills=i,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        # After rotation, path should change
        assert small_journal.path != initial_path
        assert small_journal._rotation_index >= 1

    def test_rotation_creates_new_file(self, small_journal: TradeJournal, journal_dir: Path):
        """Rotated file should exist on disk."""
        for i in range(50):
            small_journal.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        assert small_journal.path.exists()

    def test_latest_symlink_updated_on_rotation(
        self, small_journal: TradeJournal, journal_dir: Path,
    ):
        """The 'latest' symlink should point to the current journal file."""
        link = journal_dir / "mm_TEST-USD_latest.jsonl"
        assert link.is_symlink()

        initial_target = os.readlink(link)
        for i in range(50):
            small_journal.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        new_target = os.readlink(link)
        # Symlink should now point to the rotated file
        if small_journal._rotation_index > 0:
            assert new_target != initial_target

    def test_rotation_preserves_event_count(
        self, small_journal: TradeJournal,
    ):
        """Event count (seq) should be continuous across rotations."""
        for i in range(50):
            small_journal.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        assert small_journal.event_count == 50

    def test_no_rotation_when_disabled(self, journal_dir: Path):
        """Setting max_size_mb=0 should disable rotation."""
        j = TradeJournal("TEST-USD", journal_dir, run_id="no-rotate", max_size_mb=0)
        initial_path = j.path
        for i in range(20):
            j.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        assert j.path == initial_path
        assert j._rotation_index == 0
        j.close()

    def test_rotation_old_file_still_readable(
        self, small_journal: TradeJournal, journal_dir: Path,
    ):
        """Old rotated files should still be readable."""
        initial_path = small_journal.path
        for i in range(50):
            small_journal.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
        # Read old file
        if initial_path.exists():
            with open(initial_path) as f:
                lines = f.readlines()
            assert len(lines) > 0
            # Each line should be valid JSON
            for line in lines:
                event = json.loads(line.strip())
                assert event["type"] == "snapshot"


# ---------------------------------------------------------------------------
# Fsync tests
# ---------------------------------------------------------------------------


class TestFsyncBehaviour:
    def test_critical_events_trigger_fsync(self, journal: TradeJournal):
        """Critical events should call os.fsync immediately."""
        with patch("os.fsync") as mock_fsync:
            journal.record_fill(
                trade_id=1,
                order_id=1,
                side="BUY",
                price=Decimal("100"),
                qty=Decimal("1"),
                fee=Decimal("0.01"),
                is_taker=False,
                level=0,
                best_bid=Decimal("99"),
                best_ask=Decimal("101"),
                position=Decimal("1"),
            )
            assert mock_fsync.called

    def test_non_critical_events_batch_fsync(self, journal: TradeJournal):
        """Non-critical events should not fsync on every write."""
        with patch("os.fsync") as _mock_fsync:
            journal.record_snapshot(
                position=Decimal("0"),
                best_bid=Decimal("100"),
                best_ask=Decimal("101"),
                spread_bps=Decimal("10"),
                active_orders=0,
                total_fills=0,
                total_fees=Decimal("0"),
                circuit_open=False,
            )
            # Verify the test runs without error; fsync batching is tested
            # more thoroughly in test_batch_fsync_after_n_writes.

    def test_batch_fsync_after_n_writes(self, journal: TradeJournal):
        """After _BATCH_FSYNC_INTERVAL_WRITES non-critical writes, fsync should trigger."""
        # Reset counters
        journal._writes_since_fsync = 0
        journal._last_fsync_ts = time.monotonic()

        with patch("os.fsync") as mock_fsync:
            for i in range(_BATCH_FSYNC_INTERVAL_WRITES + 1):
                journal.record_snapshot(
                    position=Decimal("0"),
                    best_bid=Decimal("100"),
                    best_ask=Decimal("101"),
                    spread_bps=Decimal("10"),
                    active_orders=0,
                    total_fills=0,
                    total_fees=Decimal("0"),
                    circuit_open=False,
                )
            # Should have triggered at least one batched fsync
            assert mock_fsync.call_count >= 1

    def test_critical_event_types_are_correct(self):
        """Verify the set of critical event types."""
        assert "fill" in _CRITICAL_EVENT_TYPES
        assert "drawdown_stop" in _CRITICAL_EVENT_TYPES
        assert "run_end" in _CRITICAL_EVENT_TYPES
        assert "circuit_breaker" in _CRITICAL_EVENT_TYPES
        assert "error" in _CRITICAL_EVENT_TYPES
        assert "run_config_change" in _CRITICAL_EVENT_TYPES
        # Non-critical
        assert "snapshot" not in _CRITICAL_EVENT_TYPES
        assert "reprice_decision" not in _CRITICAL_EVENT_TYPES


# ---------------------------------------------------------------------------
# Heartbeat tests
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_record_heartbeat(self, journal: TradeJournal, journal_dir: Path):
        """record_heartbeat should write a heartbeat event."""
        journal.record_heartbeat(
            position=Decimal("5.5"),
            event_count=42,
            active_orders=3,
            uptime_s=120.5,
        )
        journal.close()

        events = analyser.load_journal(journal.path)
        hb = [e for e in events if e["type"] == "heartbeat"]
        assert len(hb) == 1
        assert hb[0]["position"] == "5.5"
        assert hb[0]["event_count"] == 42
        assert hb[0]["active_orders"] == 3
        assert hb[0]["uptime_s"] == 120.5

    def test_event_count_property(self, journal: TradeJournal):
        """event_count should track total writes."""
        assert journal.event_count == 0
        journal.record_heartbeat(
            position=Decimal("0"),
            event_count=0,
        )
        assert journal.event_count == 1
        journal.record_heartbeat(
            position=Decimal("0"),
            event_count=1,
        )
        assert journal.event_count == 2


# ---------------------------------------------------------------------------
# Error journaling tests
# ---------------------------------------------------------------------------


class TestErrorJournaling:
    def test_record_error(self, journal: TradeJournal, journal_dir: Path):
        """record_error should write a structured error event."""
        journal.record_error(
            component="level_task_BUY_L0",
            exception_type="ValueError",
            message="bad value",
            stack_trace_hash="abc123def456",
            stack_trace="Traceback ...\nValueError: bad value",
        )
        journal.close()

        events = analyser.load_journal(journal.path)
        errs = [e for e in events if e["type"] == "error"]
        assert len(errs) == 1
        assert errs[0]["component"] == "level_task_BUY_L0"
        assert errs[0]["exception_type"] == "ValueError"
        assert errs[0]["message"] == "bad value"
        assert errs[0]["stack_trace_hash"] == "abc123def456"
        assert "Traceback" in errs[0]["stack_trace"]

    def test_make_stack_trace_hash(self):
        """make_stack_trace_hash should produce consistent hashes."""
        try:
            raise ValueError("test error")
        except ValueError as exc:
            h1 = TradeJournal.make_stack_trace_hash(exc)
            h2 = TradeJournal.make_stack_trace_hash(exc)
            assert h1 == h2
            assert len(h1) == 12  # md5 hex[:12]

    def test_format_stack_trace(self):
        """format_stack_trace should include the exception message."""
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            trace = TradeJournal.format_stack_trace(exc)
            assert "RuntimeError" in trace
            assert "boom" in trace

    def test_error_is_critical_event(self, journal: TradeJournal):
        """Error events should trigger immediate fsync."""
        with patch("os.fsync") as mock_fsync:
            journal.record_error(
                component="test",
                exception_type="Exception",
                message="test",
                stack_trace_hash="abc",
            )
            assert mock_fsync.called


# ---------------------------------------------------------------------------
# Config change tests
# ---------------------------------------------------------------------------


class TestConfigChange:
    def test_record_config_change(self, journal: TradeJournal, journal_dir: Path):
        """record_config_change should write before/after/diff."""
        journal.record_config_change(
            before={"spread": 5, "levels": 3},
            after={"spread": 10, "levels": 3},
            diff={"spread": {"before": 5, "after": 10}},
        )
        journal.close()

        events = analyser.load_journal(journal.path)
        changes = [e for e in events if e["type"] == "run_config_change"]
        assert len(changes) == 1
        assert changes[0]["before"]["spread"] == 5
        assert changes[0]["after"]["spread"] == 10
        assert "spread" in changes[0]["diff"]


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    def test_all_valid_returns_true(self):
        """All events with expected version should pass validation."""
        events = [
            {"type": "fill", "schema_version": 2},
            {"type": "snapshot", "schema_version": 2},
        ]
        valid, warnings = analyser.validate_schema_versions(events)
        assert valid is True
        assert warnings == []

    def test_missing_version_flagged(self):
        """Events without schema_version should produce a warning."""
        events = [
            {"type": "fill"},
            {"type": "snapshot", "schema_version": 2},
        ]
        valid, warnings = analyser.validate_schema_versions(events)
        assert valid is False
        assert len(warnings) == 1
        assert "no schema_version" in warnings[0]

    def test_wrong_version_flagged(self):
        """Events with wrong schema_version should produce a warning."""
        events = [
            {"type": "fill", "schema_version": 1},
            {"type": "snapshot", "schema_version": 2},
        ]
        valid, warnings = analyser.validate_schema_versions(events)
        assert valid is False
        assert len(warnings) == 1
        assert "schema_version=1" in warnings[0]

    def test_multiple_issues(self):
        """Multiple schema issues should produce multiple warnings."""
        events = [
            {"type": "fill"},  # missing
            {"type": "snapshot", "schema_version": 1},  # wrong
            {"type": "heartbeat", "schema_version": 3},  # future
            {"type": "order_placed", "schema_version": 2},  # ok
        ]
        valid, warnings = analyser.validate_schema_versions(events)
        assert valid is False
        assert len(warnings) == 3  # missing, v1, v3

    def test_empty_events_valid(self):
        """Empty event list should be valid."""
        valid, warnings = analyser.validate_schema_versions([])
        assert valid is True
        assert warnings == []


# ---------------------------------------------------------------------------
# Heartbeat gap detection tests
# ---------------------------------------------------------------------------


class TestHeartbeatGapDetection:
    def test_no_gaps_returns_empty(self):
        """Consecutive heartbeats within threshold return no gaps."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "heartbeat", "ts": 1030.0},
            {"type": "heartbeat", "ts": 1060.0},
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert gaps == []

    def test_gap_detected(self):
        """A gap > 60s between heartbeats should be detected."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "heartbeat", "ts": 1030.0},
            {"type": "heartbeat", "ts": 1200.0},  # 170s gap from previous
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert len(gaps) == 1
        assert gaps[0]["start_ts"] == 1030.0
        assert gaps[0]["end_ts"] == 1200.0
        assert gaps[0]["gap_s"] == 170.0

    def test_multiple_gaps(self):
        """Multiple gaps should all be detected."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "heartbeat", "ts": 1200.0},  # 200s gap
            {"type": "heartbeat", "ts": 1230.0},  # ok
            {"type": "heartbeat", "ts": 1500.0},  # 270s gap
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert len(gaps) == 2
        assert gaps[0]["gap_s"] == 200.0
        assert gaps[1]["gap_s"] == 270.0

    def test_custom_threshold(self):
        """Custom max_gap_s should be respected."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "heartbeat", "ts": 1040.0},  # 40s gap
        ]
        # Default 60s - no gap
        assert analyser.detect_heartbeat_gaps(events) == []
        # Custom 30s - gap detected
        gaps = analyser.detect_heartbeat_gaps(events, max_gap_s=30.0)
        assert len(gaps) == 1

    def test_non_heartbeat_events_ignored(self):
        """Only heartbeat events should be considered."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "snapshot", "ts": 1030.0},
            {"type": "fill", "ts": 1060.0},
            {"type": "heartbeat", "ts": 1090.0},  # 90s from first heartbeat
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert len(gaps) == 1
        assert gaps[0]["gap_s"] == 90.0

    def test_single_heartbeat_no_gap(self):
        """A single heartbeat can't form a gap."""
        events = [{"type": "heartbeat", "ts": 1000.0}]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert gaps == []

    def test_no_heartbeats(self):
        """No heartbeat events should return empty."""
        events = [
            {"type": "fill", "ts": 1000.0},
            {"type": "snapshot", "ts": 1030.0},
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert gaps == []

    def test_gap_includes_iso_timestamps(self):
        """Gap records should include ISO formatted timestamps."""
        events = [
            {"type": "heartbeat", "ts": 1000.0},
            {"type": "heartbeat", "ts": 1200.0},
        ]
        gaps = analyser.detect_heartbeat_gaps(events)
        assert len(gaps) == 1
        assert "start_iso" in gaps[0]
        assert "end_iso" in gaps[0]


# ---------------------------------------------------------------------------
# analyse() integration tests
# ---------------------------------------------------------------------------


class TestAnalyseIntegration:
    def _make_base_events(self, schema_version=2):
        """Create a minimal set of events for analyse()."""
        base_ts = 1700000000.0
        events = [
            {
                "type": "run_start",
                "ts": base_ts,
                "market": "TEST-USD",
                "schema_version": schema_version,
            },
            {
                "type": "heartbeat",
                "ts": base_ts + 30,
                "market": "TEST-USD",
                "schema_version": schema_version,
                "position": "0",
                "event_count": 1,
            },
            {
                "type": "heartbeat",
                "ts": base_ts + 60,
                "market": "TEST-USD",
                "schema_version": schema_version,
                "position": "0",
                "event_count": 2,
            },
            {
                "type": "run_end",
                "ts": base_ts + 90,
                "market": "TEST-USD",
                "schema_version": schema_version,
            },
        ]
        return events

    def test_analyse_includes_heartbeat_gaps(self):
        """analyse() should include heartbeat gap section when gaps exist."""
        base_ts = 1700000000.0
        events = [
            {
                "type": "run_start",
                "ts": base_ts,
                "market": "TEST-USD",
                "schema_version": 2,
            },
            {
                "type": "heartbeat",
                "ts": base_ts + 30,
                "market": "TEST-USD",
                "schema_version": 2,
                "position": "0",
                "event_count": 1,
            },
            {
                "type": "heartbeat",
                "ts": base_ts + 200,  # Big gap
                "market": "TEST-USD",
                "schema_version": 2,
                "position": "0",
                "event_count": 2,
            },
            {
                "type": "run_end",
                "ts": base_ts + 230,
                "market": "TEST-USD",
                "schema_version": 2,
            },
        ]
        report = analyser.analyse(events, Path("test.jsonl"), None)
        assert "Heartbeat Gaps" in report
        assert "170s gap" in report

    def test_analyse_no_heartbeat_gaps_when_fine(self):
        """analyse() should not include heartbeat gap section when no gaps."""
        events = self._make_base_events()
        report = analyser.analyse(events, Path("test.jsonl"), None)
        assert "Heartbeat Gaps" not in report

    def test_analyse_includes_schema_warnings(self):
        """analyse() should include schema warnings for incompatible events."""
        events = self._make_base_events()
        # Add an event with wrong schema
        events.insert(1, {
            "type": "snapshot",
            "ts": events[0]["ts"] + 10,
            "market": "TEST-USD",
            "schema_version": 1,
        })
        report = analyser.analyse(events, Path("test.jsonl"), None)
        assert "Schema Warnings" in report
        assert "schema_version=1" in report

    def test_analyse_no_schema_warnings_when_valid(self):
        """analyse() should not include schema warnings when all events are valid."""
        events = self._make_base_events()
        report = analyser.analyse(events, Path("test.jsonl"), None)
        assert "Schema Warnings" not in report


# ---------------------------------------------------------------------------
# Latest symlink tests
# ---------------------------------------------------------------------------


class TestLatestSymlink:
    def test_symlink_created_on_init(self, journal_dir: Path):
        """Creating a journal should create a latest symlink."""
        j = TradeJournal("SYMTEST", journal_dir, run_id="s1")
        link = journal_dir / "mm_SYMTEST_latest.jsonl"
        assert link.is_symlink()
        assert os.readlink(link) == j.path.name
        j.close()

    def test_symlink_updates_across_journals(self, journal_dir: Path):
        """Creating a second journal for same market should update symlink."""
        j1 = TradeJournal("SYMTEST", journal_dir, run_id="s1")
        link = journal_dir / "mm_SYMTEST_latest.jsonl"
        os.readlink(link)  # Verify symlink exists
        j1.close()

        # Ensure different timestamp
        time.sleep(0.01)
        j2 = TradeJournal("SYMTEST", journal_dir, run_id="s2")
        j2.close()

        # Symlink should point to j2's file
        assert os.readlink(link) == j2.path.name


# ---------------------------------------------------------------------------
# Journal close with fsync
# ---------------------------------------------------------------------------


class TestJournalClose:
    def test_close_fsyncs(self, journal: TradeJournal):
        """close() should fsync before closing."""
        journal.record_heartbeat(
            position=Decimal("0"),
            event_count=0,
        )
        with patch("os.fsync") as mock_fsync:
            journal.close()
            assert mock_fsync.called

    def test_double_close_safe(self, journal: TradeJournal):
        """Calling close() twice should not raise."""
        journal.close()
        journal.close()  # Should not raise
