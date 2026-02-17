"""
Trade Journal

Records every market-maker event (fills, order placements, cancellations,
reprices, rejections, periodic snapshots) to a JSONL file.

Each line is a self-contained JSON object with a ``type`` field and all
the context needed for post-hoc analysis.  The file can be shared as-is
or fed to ``scripts/analyse_mm_journal.py`` for a compact summary.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL_DIR = Path("data/mm_journal")


class _DecimalEncoder(json.JSONEncoder):
    """Encode Decimal as string to preserve precision in JSON."""

    def default(self, o):
        if isinstance(o, Decimal):
            return str(o)
        return super().default(o)


class TradeJournal:
    """Append-only JSONL writer for market-maker events."""

    def __init__(
        self,
        market_name: str,
        journal_dir: Optional[Path] = None,
        *,
        run_id: Optional[str] = None,
        schema_version: int = 2,
    ) -> None:
        self._market = market_name
        self._dir = journal_dir or _DEFAULT_JOURNAL_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or uuid.uuid4().hex
        self._schema_version = schema_version
        self._seq = 0

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = self._dir / f"mm_{market_name}_{ts}.jsonl"
        self._fh = open(self._path, "a")  # noqa: SIM115
        logger.info(
            "Trade journal: %s (run_id=%s schema=v%s)",
            self._path,
            self._run_id,
            self._schema_version,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._fh and not self._fh.closed:
            self._fh.close()
            logger.info("Trade journal closed: %s", self._path)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def run_id(self) -> str:
        return self._run_id
