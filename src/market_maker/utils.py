"""Shared utility helpers for the market_maker package."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


def safe_decimal(value, default: str = "0") -> Decimal:
    """Convert *value* to Decimal, returning *default* on failure."""
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError, ArithmeticError):
        return Decimal(default)


def safe_float(value, default: float = 0.0) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
