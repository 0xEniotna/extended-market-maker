from __future__ import annotations

import time
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

    def effective_ticks(self, key: tuple[str, int]) -> int:
        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        if not self._settings.adaptive_pof_enabled:
            return base_ticks
        dynamic_ticks = self._level_dynamic_safety_ticks.get(key, base_ticks)
        return max(base_ticks, int(dynamic_ticks))

    def on_rejection(self, key: tuple[str, int]) -> None:
        now = time.monotonic()
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
        if not self._settings.adaptive_pof_enabled:
            return
        streak = self._level_pof_streak.get(key, 0)
        if streak <= 0:
            return
        streak -= 1
        self._level_pof_streak[key] = streak
        base_ticks = max(1, int(self._settings.post_only_safety_ticks))
        self._level_dynamic_safety_ticks[key] = min(
            max(base_ticks, int(self._settings.pof_max_safety_ticks)),
            base_ticks + streak,
        )
        if streak == 0:
            self._level_pof_until[key] = 0.0

    def reset(self, key: tuple[str, int]) -> None:
        self._level_pof_streak[key] = 0
        self._level_pof_until[key] = 0.0
        self._level_pof_last_ts[key] = time.monotonic()
        self._level_dynamic_safety_ticks[key] = max(
            1, self._settings.post_only_safety_ticks
        )

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
