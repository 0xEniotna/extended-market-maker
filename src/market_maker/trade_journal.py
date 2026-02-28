"""
Trade Journal

Records every market-maker event (fills, order placements, cancellations,
reprices, rejections, periodic snapshots) to a JSONL file.

Each line is a self-contained JSON object with a ``type`` field and all
the context needed for post-hoc analysis.  The file can be shared as-is
or fed to ``scripts/analyse_mm_journal.py`` for a compact summary.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL_DIR = Path("data/mm_journal")

# Event types that require immediate os.fsync for durability.
_CRITICAL_EVENT_TYPES = frozenset({
    "fill", "drawdown_stop", "run_end", "circuit_breaker",
    "exchange_maintenance", "run_config_change", "error",
})

# For non-critical events, fsync every N writes or every N seconds.
_BATCH_FSYNC_INTERVAL_WRITES = 100
_BATCH_FSYNC_INTERVAL_S = 10.0


class _DecimalEncoder(json.JSONEncoder):
    """Encode Decimal as string to preserve precision in JSON."""

    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


class TradeJournal:
    """Append-only JSONL writer for market-maker events.

    Features:
    - Durable fsync for critical events, batched fsync for high-frequency events
    - Automatic rotation when file exceeds ``max_size_mb``
    - "latest" symlink always points to the current journal file
    """

    def __init__(
        self,
        market_name: str,
        journal_dir: Optional[Path] = None,
        *,
        run_id: Optional[str] = None,
        schema_version: int = 2,
        max_size_mb: float = 50.0,
    ) -> None:
        self._market = market_name
        self._dir = journal_dir or _DEFAULT_JOURNAL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or uuid.uuid4().hex
        self._schema_version = schema_version
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._seq = 0
        self._rotation_index = 0

        # Batched fsync tracking.
        self._writes_since_fsync = 0
        self._last_fsync_ts = time.monotonic()
        # Single-thread executor for non-blocking fsync on SD cards.
        self._fsync_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="journal_fsync")

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._base_stem = f"mm_{market_name}_{ts}"
        self._path = self._dir / f"{self._base_stem}.jsonl"
        self._fh = open(self._path, "a")  # noqa: SIM115
        self._update_latest_symlink()
        logger.info(
            "Trade journal: %s (run_id=%s schema=v%s max_size=%.0fMB)",
            self._path,
            self._run_id,
            self._schema_version,
            max_size_mb,
        )

    # ------------------------------------------------------------------
    # Core writer
    # ------------------------------------------------------------------

    def _write(self, event_type: str, data: Dict[str, Any]) -> None:
        self._seq += 1
        record = {
            "ts": time.time(),
            "seq": self._seq,
            "run_id": self._run_id,
            "schema_version": self._schema_version,
            "type": event_type,
            "market": self._market,
            **data,
        }
        self._fh.write(json.dumps(record, cls=_DecimalEncoder) + "\n")
        self._fh.flush()

        # --- Durable fsync logic ---
        if event_type in _CRITICAL_EVENT_TYPES:
            self._do_fsync()
        else:
            self._writes_since_fsync += 1
            now = time.monotonic()
            if (
                self._writes_since_fsync >= _BATCH_FSYNC_INTERVAL_WRITES
                or (now - self._last_fsync_ts) >= _BATCH_FSYNC_INTERVAL_S
            ):
                self._do_fsync()

        # --- Rotation check ---
        self._maybe_rotate()

    def _do_fsync(self) -> None:
        """Submit fsync to a background thread so the event loop is not blocked.

        On SD-card media (Raspberry Pi), ``os.fsync()`` can stall for
        10-500 ms during wear-leveling / garbage-collection.  Running it
        in a single-thread executor keeps the asyncio loop responsive.
        """
        self._writes_since_fsync = 0
        self._last_fsync_ts = time.monotonic()
        try:
            fh = self._fh
            if fh and not fh.closed:
                fh.flush()
                self._fsync_executor.submit(self._fsync_fd, fh.fileno())
        except (OSError, ValueError):
            pass

    @staticmethod
    def _fsync_fd(fd: int) -> None:
        """Perform the actual os.fsync — runs in the thread-pool."""
        try:
            os.fsync(fd)
        except (OSError, ValueError):
            pass

    def _do_fsync_sync(self) -> None:
        """Synchronous flush+fsync for shutdown / rotation (must complete)."""
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except (OSError, ValueError):
            pass
        self._writes_since_fsync = 0
        self._last_fsync_ts = time.monotonic()

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------

    def _maybe_rotate(self) -> None:
        """Rotate the journal file if it exceeds max_size_bytes."""
        if self._max_size_bytes <= 0:
            return
        try:
            pos = self._fh.tell()
        except (OSError, ValueError):
            return
        if pos < self._max_size_bytes:
            return
        self._rotate()

    def _rotate(self) -> None:
        """Close the current file and open a new one with incremented suffix."""
        self._do_fsync_sync()
        try:
            self._fh.close()
        except Exception:
            pass
        self._rotation_index += 1
        self._path = self._dir / f"{self._base_stem}.{self._rotation_index}.jsonl"
        self._fh = open(self._path, "a")  # noqa: SIM115
        self._update_latest_symlink()
        logger.info("Journal rotated to: %s", self._path)

    def _update_latest_symlink(self) -> None:
        """Maintain a 'latest' symlink pointing to the current journal file."""
        link_path = self._dir / f"mm_{self._market}_latest.jsonl"
        try:
            if link_path.is_symlink() or link_path.exists():
                link_path.unlink()
            link_path.symlink_to(self._path.name)
        except OSError as exc:
            logger.debug("Failed to update latest symlink: %s", exc)

    # ------------------------------------------------------------------
    # Event methods — called by strategy components
    # ------------------------------------------------------------------

    def record_run_start(
        self,
        *,
        environment: str,
        config: Dict[str, Any],
        market_static: Dict[str, Any],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write("run_start", {
            "environment": environment,
            "config": config,
            "market_static": market_static,
            "provenance": provenance or {},
        })

    def record_run_end(
        self,
        *,
        reason: str = "shutdown",
        stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._write("run_end", {
            "reason": reason,
            "stats": stats or {},
        })

    def record_fill(
        self,
        *,
        trade_id: int,
        order_id: int,
        side: str,
        price: Decimal,
        qty: Decimal,
        fee: Decimal,
        is_taker: bool,
        level: Optional[int],
        best_bid: Optional[Decimal],
        best_ask: Optional[Decimal],
        position: Decimal,
        quote_lifetime_ms: Optional[Decimal] = None,
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> None:
        mid = None
        spread_bps = None
        edge_bps = None
        if best_bid and best_ask and best_bid > 0:
            mid = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid * Decimal("10000")
            # Edge: how far our fill was from mid (positive = favorable)
            if side == "BUY":
                edge_bps = (mid - price) / mid * Decimal("10000")
            else:
                edge_bps = (price - mid) / mid * Decimal("10000")

        self._write("fill", {
            "trade_id": trade_id,
            "order_id": order_id,
            "side": side,
            "price": price,
            "qty": qty,
            "fee": fee,
            "is_taker": is_taker,
            "level": level,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "edge_bps": edge_bps,
            "position": position,
            "quote_lifetime_ms": quote_lifetime_ms,
            "market_snapshot": market_snapshot,
        })

    def record_order_placed(
        self,
        *,
        external_id: str,
        exchange_id: Optional[str],
        side: str,
        price: Decimal,
        size: Decimal,
        level: int,
        best_bid: Optional[Decimal],
        best_ask: Optional[Decimal],
        position: Decimal,
    ) -> None:
        spread_bps = None
        if best_bid and best_ask and best_bid > 0:
            mid = (best_bid + best_ask) / 2
            spread_bps = (best_ask - best_bid) / mid * Decimal("10000")

        self._write("order_placed", {
            "external_id": external_id,
            "exchange_id": exchange_id,
            "side": side,
            "price": price,
            "size": size,
            "level": level,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": spread_bps,
            "position": position,
        })

    def record_order_cancelled(
        self,
        *,
        external_id: str,
        exchange_id: Optional[str] = None,
        side: str,
        level: int,
        reason: str = "reprice",
    ) -> None:
        self._write("order_cancelled", {
            "external_id": external_id,
            "exchange_id": exchange_id,
            "side": side,
            "level": level,
            "reason": reason,
        })

    def record_rejection(
        self,
        *,
        external_id: str,
        exchange_id: Optional[str] = None,
        side: str,
        price: Decimal,
        reason: str,
    ) -> None:
        self._write("rejection", {
            "external_id": external_id,
            "exchange_id": exchange_id,
            "side": side,
            "price": price,
            "reason": reason,
        })

    def record_snapshot(
        self,
        *,
        position: Decimal,
        best_bid: Optional[Decimal],
        best_ask: Optional[Decimal],
        spread_bps: Optional[Decimal],
        active_orders: int,
        total_fills: int,
        total_fees: Decimal,
        circuit_open: bool,
    ) -> None:
        self._write("snapshot", {
            "position": position,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": spread_bps,
            "active_orders": active_orders,
            "total_fills": total_fills,
            "total_fees": total_fees,
            "circuit_open": circuit_open,
        })

    def record_reprice_decision(
        self,
        *,
        side: str,
        level: int,
        reason: str,
        current_best: Optional[Decimal] = None,
        prev_price: Optional[Decimal] = None,
        target_price: Optional[Decimal] = None,
        spread_bps: Optional[Decimal] = None,
        extra_offset_bps: Optional[Decimal] = None,
        regime: Optional[str] = None,
        trend_direction: Optional[str] = None,
        trend_strength: Optional[Decimal] = None,
        inventory_band: Optional[str] = None,
        funding_bias_bps: Optional[Decimal] = None,
    ) -> None:
        self._write("reprice_decision", {
            "side": side,
            "level": level,
            "reason": reason,
            "current_best": current_best,
            "prev_price": prev_price,
            "target_price": target_price,
            "spread_bps": spread_bps,
            "extra_offset_bps": extra_offset_bps,
            "regime": regime,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "inventory_band": inventory_band,
            "funding_bias_bps": funding_bias_bps,
        })

    def record_drawdown_stop(
        self,
        *,
        current_pnl: Decimal,
        peak_pnl: Decimal,
        drawdown: Decimal,
        threshold_usd: Decimal,
        action: str,
    ) -> None:
        self._write("drawdown_stop", {
            "current_pnl": current_pnl,
            "peak_pnl": peak_pnl,
            "drawdown": drawdown,
            "threshold_usd": threshold_usd,
            "action": action,
        })

    def record_exchange_event(
        self,
        *,
        event_type: str,
        details: dict,
    ) -> None:
        self._write(event_type, details)

    def record_heartbeat(
        self,
        *,
        position: Decimal,
        event_count: int,
        active_orders: int = 0,
        uptime_s: float = 0.0,
    ) -> None:
        self._write("heartbeat", {
            "position": position,
            "event_count": event_count,
            "active_orders": active_orders,
            "uptime_s": uptime_s,
        })

    def record_error(
        self,
        *,
        component: str,
        exception_type: str,
        message: str,
        stack_trace_hash: str,
        stack_trace: Optional[str] = None,
    ) -> None:
        self._write("error", {
            "component": component,
            "exception_type": exception_type,
            "message": message,
            "stack_trace_hash": stack_trace_hash,
            "stack_trace": stack_trace,
        })

    def record_config_change(
        self,
        *,
        before: Dict[str, Any],
        after: Dict[str, Any],
        diff: Dict[str, Any],
    ) -> None:
        self._write("run_config_change", {
            "before": before,
            "after": after,
            "diff": diff,
        })

    # ------------------------------------------------------------------
    # Helpers for structured error recording
    # ------------------------------------------------------------------

    @staticmethod
    def make_stack_trace_hash(exc: BaseException) -> str:
        """Create a short hash of the stack trace for deduplication."""
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        return hashlib.md5(tb_str.encode()).hexdigest()[:12]

    @staticmethod
    def format_stack_trace(exc: BaseException) -> str:
        """Format exception traceback as a string."""
        return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._do_fsync_sync()
            self._fh.close()
        self._fsync_executor.shutdown(wait=True)
        logger.info("Trade journal closed: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def event_count(self) -> int:
        return self._seq
