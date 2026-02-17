from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class DrawdownStopState:
    enabled: bool
    current_pnl: Decimal
    peak_pnl: Decimal
    drawdown: Decimal
    threshold_usd: Decimal
    triggered: bool


class DrawdownStop:
    """Track per-market PnL drawdown and emit a one-shot trigger on breach."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_position_notional_usd: Decimal,
        drawdown_pct_of_max_notional: Decimal,
        use_high_watermark: bool,
    ) -> None:
        self._enabled = bool(enabled)
        self._use_high_watermark = bool(use_high_watermark)
        safe_notional = max(Decimal("0"), Decimal(str(max_position_notional_usd)))
        safe_pct = max(Decimal("0"), Decimal(str(drawdown_pct_of_max_notional)))
        self._threshold_usd = safe_notional * safe_pct / Decimal("100")
        self._initialised = False
        self._baseline_pnl = Decimal("0")
        self._peak_pnl = Decimal("0")
        self._tripped = False

    @property
    def threshold_usd(self) -> Decimal:
        return self._threshold_usd

    @property
    def tripped(self) -> bool:
        return self._tripped

    def evaluate(self, current_pnl: Decimal) -> DrawdownStopState:
        pnl = Decimal(str(current_pnl))

        if not self._initialised:
            self._baseline_pnl = pnl
            self._peak_pnl = pnl
            self._initialised = True
        else:
            self._peak_pnl = max(self._peak_pnl, pnl)

        reference_pnl = self._peak_pnl if self._use_high_watermark else self._baseline_pnl
        drawdown = max(Decimal("0"), reference_pnl - pnl)
        should_trip = (
            self._enabled
            and self._threshold_usd > 0
            and drawdown >= self._threshold_usd
        )
        triggered = should_trip and not self._tripped
        if should_trip:
            self._tripped = True

        return DrawdownStopState(
            enabled=self._enabled,
            current_pnl=pnl,
            peak_pnl=self._peak_pnl,
            drawdown=drawdown,
            threshold_usd=self._threshold_usd,
            triggered=triggered,
        )
