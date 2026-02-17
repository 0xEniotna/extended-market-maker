from __future__ import annotations

from decimal import Decimal

from market_maker.drawdown_stop import DrawdownStop


def test_high_water_drawdown_triggers_at_threshold_boundary():
    stop = DrawdownStop(
        enabled=True,
        max_position_notional_usd=Decimal("1000"),
        drawdown_pct_of_max_notional=Decimal("1.5"),
        use_high_watermark=True,
    )
    assert stop.threshold_usd == Decimal("15")

    first = stop.evaluate(Decimal("50"))
    second = stop.evaluate(Decimal("35"))

    assert first.triggered is False
    assert second.drawdown == Decimal("15")
    assert second.triggered is True


def test_high_water_drawdown_does_not_trigger_below_threshold():
    stop = DrawdownStop(
        enabled=True,
        max_position_notional_usd=Decimal("2000"),
        drawdown_pct_of_max_notional=Decimal("1.5"),
        use_high_watermark=True,
    )
    stop.evaluate(Decimal("120"))
    state = stop.evaluate(Decimal("95"))  # drawdown=25, threshold=30

    assert state.threshold_usd == Decimal("30")
    assert state.drawdown == Decimal("25")
    assert state.triggered is False


def test_disabled_drawdown_stop_never_triggers():
    stop = DrawdownStop(
        enabled=False,
        max_position_notional_usd=Decimal("1000"),
        drawdown_pct_of_max_notional=Decimal("1.5"),
        use_high_watermark=True,
    )
    stop.evaluate(Decimal("100"))
    state = stop.evaluate(Decimal("-1000"))

    assert state.triggered is False


def test_run_start_mode_uses_first_pnl_as_reference():
    stop = DrawdownStop(
        enabled=True,
        max_position_notional_usd=Decimal("1000"),
        drawdown_pct_of_max_notional=Decimal("1.5"),
        use_high_watermark=False,
    )
    stop.evaluate(Decimal("40"))
    state = stop.evaluate(Decimal("24"))

    assert state.drawdown == Decimal("16")
    assert state.triggered is True
