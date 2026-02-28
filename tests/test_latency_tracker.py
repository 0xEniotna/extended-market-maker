from __future__ import annotations

from market_maker.order_tracking import LatencyTracker
import market_maker.order_tracking as order_tracking


def test_latency_tracker_drains_only_new_samples(monkeypatch) -> None:
    now = 1000.0

    def _mono() -> float:
        return now

    monkeypatch.setattr(order_tracking.time, "monotonic", _mono)

    tracker = LatencyTracker(maxlen=10)
    tracker.record_send("a")
    now = 1000.050
    lat_a = tracker.record_ack("a")
    assert lat_a is not None

    first = tracker.drain_samples()
    assert len(first) == 1
    assert first[0] > 0
    assert tracker.drain_samples() == []

    tracker.record_send("b")
    now = 1000.125
    lat_b = tracker.record_ack("b")
    assert lat_b is not None

    second = tracker.drain_samples()
    assert len(second) == 1
    assert second[0] > 0
    assert second[0] != first[0]
