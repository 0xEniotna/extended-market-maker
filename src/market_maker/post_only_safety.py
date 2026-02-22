from __future__ import annotations

import math
import time
from collections import deque
from decimal import Decimal
from typing import Callable, Optional, cast

from .types import PostOnlySettingsLike


class PostOnlySafety:
    """Stateful post-only clamp and adaptive POF handling."""

    def __init__(
        self,
        settings: object,
        tick_size: Decimal,
        round_to_tick: Callable[[Decimal, object], Decimal],
    ) -> None:
        self._settings = cast(PostOnlySettingsLike, settings)
        self._tick_size = tick_size
        self._round_to_tick = round_to_tick

        self._level_pof_until: dict[tuple[str, int], float] = {}
        self._level_pof_streak: dict[tuple[str, int], int] = {}
        self._level_pof_last_ts: dict[tuple[str, int], float] = {}
        self._level_dynamic_safety_ticks: dict[tuple[str, int], int] = {}
        self._level_consecutive_successes: dict[tuple[str, int], int] = {}

        # POF-spread correlation: auto-widen min_offset when POF rate is high.
        self._pof_timestamps: deque[float] = deque()
        self._pof_offset_boost_bps: Decimal = Decimal("0")

    @property
    def pof_until(self) -> dict[tuple[str, int], float]:
        return self._level_pof_until

    @property
    def pof_streak(self) -> dict[tuple[str, int], int]:
        return self._level_pof_streak

    @property
    def pof_last_ts(self) -> dict[tuple[str, int], float]:
        return self._level_pof_last_ts

    @property
    def dynamic_safety_ticks(self) -> dict[tuple[str, int], int]:
        return self._level_dynamic_safety_ticks

    @property
    def pof_offset_boost_bps(self) -> Decimal:
        """Current dynamic offset boost (bps) driven by POF-spread correlation."""
        return self._pof_offset_boost_bps

    def effective_ticks(
        self,
        key: tuple[str, int],
        *,
        avg_latency_ms: float = 0.0,
        tick_time_ms: float = 0.0,
    ) -> int:
        """Return effective safety ticks for a level.

        Incorporates base safety, adaptive POF escalation, and latency buffer.
        """
        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        if not self._settings.adaptive_pof_enabled:
            latency_ticks = self._latency_ticks(avg_latency_ms, tick_time_ms)
            return base_ticks + latency_ticks

        dynamic_ticks = self._level_dynamic_safety_ticks.get(key, base_ticks)
        result = max(base_ticks, int(dynamic_ticks))

        latency_ticks = self._latency_ticks(avg_latency_ms, tick_time_ms)
        return result + latency_ticks

    @staticmethod
    def _latency_ticks(avg_latency_ms: float, tick_time_ms: float) -> int:
        """Compute extra safety ticks from average placement latency."""
        if avg_latency_ms <= 0 or tick_time_ms <= 0:
            return 0
        return math.ceil(avg_latency_ms / tick_time_ms)

    def on_rejection(self, key: tuple[str, int]) -> None:
        now = time.monotonic()
        self._level_consecutive_successes[key] = 0

        # Record POF timestamp for rate tracking.
        self._pof_timestamps.append(now)

        if not self._settings.adaptive_pof_enabled:
            self._level_pof_until[key] = now + float(self._settings.pof_cooldown_s)
            return

        last_ts = self._level_pof_last_ts.get(key)
        reset_window_s = float(self._settings.pof_streak_reset_s)
        if (
            last_ts is None
            or (reset_window_s > 0 and (now - last_ts) > reset_window_s)
        ):
            streak = 0
        else:
            streak = self._level_pof_streak.get(key, 0)
        streak += 1
        self._level_pof_streak[key] = streak
        self._level_pof_last_ts[key] = now

        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        max_ticks = max(base_ticks, int(self._settings.pof_max_safety_ticks))
        dynamic_ticks = min(max_ticks, base_ticks + streak)
        self._level_dynamic_safety_ticks[key] = dynamic_ticks

        multiplier = max(Decimal("1"), self._settings.pof_backoff_multiplier)
        cooldown_s = float(
            Decimal(str(self._settings.pof_cooldown_s))
            * (multiplier ** (streak - 1))
        )
        self._level_pof_until[key] = now + min(cooldown_s, 120.0)

    def on_success(self, key: tuple[str, int]) -> None:
        """Handle successful order placement.

        Uses faster decay: streak *= 0.5 (rounded down), and after 3
        consecutive successes fully reset the streak.
        """
        if not self._settings.adaptive_pof_enabled:
            return
        streak = self._level_pof_streak.get(key, 0)
        if streak <= 0:
            return

        # Track consecutive successes for full reset
        consec = self._level_consecutive_successes.get(key, 0) + 1
        self._level_consecutive_successes[key] = consec

        if consec >= 3:
            # Full reset after 3 consecutive successes
            self._level_pof_streak[key] = 0
            base_ticks = max(1, int(self._settings.post_only_safety_ticks))
            self._level_dynamic_safety_ticks[key] = base_ticks
            self._level_pof_until[key] = 0.0
            return

        # Decay: streak *= 0.5 (rounded down), minimum reduction of 2
        new_streak = max(0, min(streak // 2, streak - 2))
        self._level_pof_streak[key] = new_streak
        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        self._level_dynamic_safety_ticks[key] = min(
            max(base_ticks, int(self._settings.pof_max_safety_ticks)),
            base_ticks + new_streak,
        )
        if new_streak == 0:
            self._level_pof_until[key] = 0.0

    def reset(self, key: tuple[str, int]) -> None:
        self._level_pof_streak[key] = 0
        self._level_pof_until[key] = 0.0
        self._level_pof_last_ts[key] = time.monotonic()
        self._level_dynamic_safety_ticks[key] = max(
            1, self._settings.post_only_safety_ticks
        )
        self._level_consecutive_successes[key] = 0

    # ------------------------------------------------------------------
    # POF-spread correlation
    # ------------------------------------------------------------------

    def pof_rate_60s(self) -> float:
        """Return POF rejection count in the last 60 seconds."""
        now = time.monotonic()
        cutoff = now - 60.0
        while self._pof_timestamps and self._pof_timestamps[0] < cutoff:
            self._pof_timestamps.popleft()
        return float(len(self._pof_timestamps))

    def update_pof_offset_boost(self, total_placements_60s: int) -> Decimal:
        """Recalculate dynamic offset boost based on POF rate.

        If POF rate in last 60s exceeds 10% of placements, increase
        boost by 1 bps (capped at +3 bps). If POF rate drops below 5%,
        decrease boost by 1 bps (floored at 0).

        Returns the current boost.
        """
        if total_placements_60s <= 0:
            return self._pof_offset_boost_bps

        pof_count = self.pof_rate_60s()
        pof_rate = pof_count / total_placements_60s

        if pof_rate > 0.10:
            self._pof_offset_boost_bps = min(
                Decimal("3"), self._pof_offset_boost_bps + Decimal("1")
            )
        elif pof_rate < 0.05:
            self._pof_offset_boost_bps = max(
                Decimal("0"), self._pof_offset_boost_bps - Decimal("1")
            )

        return self._pof_offset_boost_bps

    def clamp_price(
        self,
        *,
        side,
        target_price: Decimal,
        bid_price: Decimal,
        ask_price: Decimal,
        safety_ticks: Optional[int] = None,
    ) -> Optional[Decimal]:
        if bid_price <= 0 or ask_price <= 0:
            return None
        if self._tick_size <= 0:
            return target_price

        safety_ticks = max(
            1,
            safety_ticks
            if safety_ticks is not None
            else self._settings.post_only_safety_ticks,
        )
        safety_buffer = self._tick_size * Decimal(safety_ticks)
        side_name = str(side).upper()
        is_buy = side_name == "BUY" or side_name.endswith("BUY")

        if is_buy:
            max_buy = ask_price - safety_buffer
            safe = min(target_price, max_buy)
            safe = self._round_to_tick(safe, side)
            if safe >= ask_price:
                safe = ask_price - self._tick_size
        else:
            min_sell = bid_price + safety_buffer
            safe = max(target_price, min_sell)
            safe = self._round_to_tick(safe, side)
            if safe <= bid_price:
                safe = bid_price + self._tick_size

        if safe <= 0:
            return None
        return safe
