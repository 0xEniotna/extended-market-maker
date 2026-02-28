"""
Quote Halt Manager

Centralises all quote halt/resume logic that was previously scattered
across MarketMakerStrategy.  The strategy now delegates to this class
for halt state management, stream health checks, and margin-breach
tracking.

Halt reasons accumulate as a set of strings.  Quoting is paused when
the set is non-empty and resumes when all reasons are cleared.
"""
from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


class QuoteHaltManager:
    """Manages quote halt state with reason tracking and journaling."""

    def __init__(
        self,
        market_name: str,
        journal: object,
        metrics: object,
    ) -> None:
        self._market_name = market_name
        self._journal = journal
        self._metrics = metrics
        self._reasons: Set[str] = set()
        self._margin_breach_since: Optional[float] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def reasons(self) -> Set[str]:
        """Return the current set of active halt reasons."""
        return self._reasons

    @property
    def is_halted(self) -> bool:
        return bool(self._reasons)

    @property
    def margin_breach_since(self) -> Optional[float]:
        return self._margin_breach_since

    @margin_breach_since.setter
    def margin_breach_since(self, value: Optional[float]) -> None:
        self._margin_breach_since = value

    def set_halt(self, reason: str) -> None:
        """Add a halt reason.  No-op if the reason is already active."""
        if reason in self._reasons:
            return
        self._reasons.add(reason)
        logger.warning(
            "Quote halt engaged for %s: reasons=%s",
            self._market_name,
            sorted(self._reasons),
        )
        self._journal.record_exchange_event(
            event_type="quote_halt",
            details={"reason": reason, "reasons": sorted(self._reasons)},
        )
        self._metrics.set_quote_halt_state(self._reasons)

    def clear_halt(self, reason: str) -> None:
        """Remove a halt reason.  No-op if the reason is not active."""
        if reason not in self._reasons:
            return
        self._reasons.discard(reason)
        logger.info(
            "Quote halt reason cleared for %s: %s (remaining=%s)",
            self._market_name,
            reason,
            sorted(self._reasons),
        )
        self._journal.record_exchange_event(
            event_type="quote_halt_cleared",
            details={"reason": reason, "remaining": sorted(self._reasons)},
        )
        self._metrics.set_quote_halt_state(self._reasons)

    def sync_state(
        self,
        *,
        rate_limit_halt: bool,
        streams_healthy: bool,
    ) -> None:
        """Synchronise external halt conditions into the reason set."""
        if rate_limit_halt:
            self.set_halt("rate_limit_halt")
        else:
            self.clear_halt("rate_limit_halt")

        if streams_healthy:
            self.clear_halt("stream_desync")
        else:
            self.set_halt("stream_desync")

        self._metrics.set_quote_halt_state(self._reasons)
        self._metrics.set_margin_guard_breached(
            self._margin_breach_since is not None,
        )

    @staticmethod
    def check_streams_healthy(
        account_stream: object,
        orderbook_mgr: object,
    ) -> bool:
        """Return True if both account and orderbook streams are healthy."""
        account_ok = True
        if hasattr(account_stream, "is_sequence_healthy"):
            account_state = account_stream.is_sequence_healthy()
            account_ok = account_state if isinstance(account_state, bool) else True
        book_ok = True
        if hasattr(orderbook_mgr, "is_sequence_healthy"):
            book_state = orderbook_mgr.is_sequence_healthy()
            book_ok = book_state if isinstance(book_state, bool) else True
        has_data_fn = getattr(orderbook_mgr, "has_data", None)
        has_data = has_data_fn() if callable(has_data_fn) else True
        return account_ok and book_ok and has_data
