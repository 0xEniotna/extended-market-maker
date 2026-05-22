"""
Microprice — size-weighted fair-value estimator (Stoikov 2018).

The raw mid ``(bid + ask) / 2`` ignores depth asymmetry. When the book is
skewed (e.g. far more size resting on the bid), the short-horizon-fair price
sits closer to the ask, because that imbalance tends to precede an up-move.
The microprice corrects for this:

    microprice = (bid · ask_qty + ask · bid_qty) / (bid_qty + ask_qty)

Note the cross-weighting: a *large bid_qty* puts more weight on ``ask`` (the
higher price), pulling the estimate up — consistent with "bid-heavy book →
mid about to rise". This direction was validated empirically on our crypto
markets in Stage 3 (``docs/stage3_microprice_diagnostic.md``): on ETH the
Pearson corr of ``(microprice − mid)`` with the +5s forward mid return was
**+0.41** (Spearman +0.60), i.e. microprice *leads* mid. The same diagnostic
showed the **wrong sign on TradFi 24/5 markets** (MU/SPX), so the live
quoting use is gated to crypto only (see ``PricingEngine.compute_target_price``).

This module is a single pure function — the single source of truth shared by
the hot path, the unit tests, and any offline replay. Edge cases (empty or
one-sided book) fall back to the plain mid so the caller never sees a
degenerate value.
"""
from __future__ import annotations

from decimal import Decimal


def microprice(
    bid: Decimal,
    ask: Decimal,
    bid_qty: Decimal,
    ask_qty: Decimal,
) -> Decimal:
    """Return the size-weighted microprice, or the plain mid on a bad book.

    Falls back to ``(bid + ask) / 2`` when either side has non-positive
    quantity (empty / one-sided / stale book) — in that regime the
    cross-weighting is meaningless and the safe reference is the mid.

    Pure and side-effect free. Decimal in, Decimal out: the caller converts
    to float once for the hot path (mirrors the ``funding_aware`` pattern),
    while tests exercise exact Decimal precision.
    """
    mid = (bid + ask) / 2
    # Degenerate book: don't trust the cross-weighting, return the mid.
    if bid_qty <= 0 or ask_qty <= 0:
        return mid
    total = bid_qty + ask_qty
    if total <= 0:  # defensive; unreachable given the guard above
        return mid
    return (bid * ask_qty + ask * bid_qty) / total
