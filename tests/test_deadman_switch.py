"""Dead-man switch heartbeat behavior.

Extended/x10 exposes no server-side dead-man switch. The heartbeat task must
degrade gracefully (one warning, status off, no error loop) when the SDK
account lacks ``set_deadman_switch``, and still arm it when present.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

# Stub the x10 SDK submodules so importing the full strategy_runner graph
# (which pulls in the WS stream client) works without the real SDK installed —
# matching the convention used by the other runtime-level tests in this suite,
# extended to cover the stream_client path that strategy_runner reaches.
_SDK_MODULES = [
    "x10",
    "x10.perpetual",
    "x10.perpetual.orders",
    "x10.perpetual.positions",
    "x10.perpetual.trading_client",
    "x10.perpetual.stream_client",
    "x10.perpetual.stream_client.stream_client",
    "x10.utils",
    "x10.utils.http",
]
for _mod_name in _SDK_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

from market_maker import strategy_runner  # noqa: E402


class _Metrics:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def set_deadman_status(self, *, armed, countdown_s, last_ok_ts) -> None:
        self.calls.append((armed, countdown_s, last_ok_ts))


class _Journal:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def record_exchange_event(self, **kwargs) -> None:
        self.events.append(kwargs)


def _make_ctx(account) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        trading_client=types.SimpleNamespace(account=account),
        settings=types.SimpleNamespace(
            deadman_countdown_s=60,
            deadman_heartbeat_s=20,
            market_name="BTC-USD",
        ),
        metrics=_Metrics(),
        journal=_Journal(),
    )


async def test_deadman_unsupported_disables_without_error_loop():
    """No set_deadman_switch -> returns promptly, reports off, never raises."""
    ctx = _make_ctx(types.SimpleNamespace())  # account without the method
    # Must complete quickly (not spin in the retry loop) and not raise.
    await asyncio.wait_for(strategy_runner._deadman_heartbeat_task(ctx), timeout=2.0)
    assert ctx.metrics.calls, "expected deadman status to be reported"
    armed, _countdown, last_ok = ctx.metrics.calls[-1]
    assert armed is False
    assert last_ok is None


async def test_deadman_supported_arms(monkeypatch):
    """When set_deadman_switch exists, the task arms it and reports armed."""
    armed_with: list[int] = []

    class _Acct:
        async def set_deadman_switch(self, countdown_s) -> None:
            armed_with.append(countdown_s)

    ctx = _make_ctx(_Acct())

    async def _stop_after_arm(*_a, **_k):
        raise asyncio.CancelledError  # break the heartbeat loop after first arm

    monkeypatch.setattr(strategy_runner.asyncio, "sleep", _stop_after_arm)
    await strategy_runner._deadman_heartbeat_task(ctx)

    assert armed_with == [60]
    armed, countdown, _last_ok = ctx.metrics.calls[-1]
    assert armed is True
    assert countdown == 60
