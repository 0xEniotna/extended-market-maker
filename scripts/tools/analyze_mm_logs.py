#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

LOG_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(?P<ms>\d{3}) "
    r"\[(?P<level>\w+)\] (?P<logger>[^:]+): (?P<msg>.*)$"
)

REPRICE_RE = re.compile(
    r"^Reprice (?P<side>BUY|SELL) L(?P<level>\d+): best=(?P<best>\S+) target=(?P<target>\S+)"
)

ORDER_PLACED_RE = re.compile(
    r"^Order placed: side=(?P<side>BUY|SELL) price=(?P<price>\S+) size=(?P<size>\S+) "
    r"level=(?P<level>\d+) ext_id=(?P<ext_id>\S+) exch_id=(?P<exch_id>\S+)"
)

ORDER_CANCEL_REQ_RE = re.compile(
    r"^Order cancel requested: ext_id=(?P<ext_id>\S+)"
)

ORDER_TERMINAL_RE = re.compile(
    r"^Order (?P<status>CANCELLED|FILLED|REJECTED) \(ext_id=(?P<ext_id>\S+), level=(?P<level>\d+)\)"
)

POST_ONLY_RE = re.compile(
    r"^POST_ONLY_FAILED: order ext_id=(?P<ext_id>\S+) side=(?P<side>BUY|SELL) price=(?P<price>\S+)"
)

FILL_RE = re.compile(
    r"^Fill: side=(?P<side>BUY|SELL) price=(?P<price>\S+) qty=(?P<qty>\S+) fee=(?P<fee>\S+) taker=(?P<taker>\S+)"
)

STATUS_RE = re.compile(
    r"^STATUS \| pos=(?P<pos>\S+) \| bid=(?P<bid>\S+) ask=(?P<ask>\S+) spread=(?P<spread>\S+)bps \| "
    r"orders=(?P<orders>\d+) \| fills=(?P<fills>\d+) cancel=(?P<cancels>\d+) reject=(?P<rejects>\d+) pof=(?P<pof>\d+) \| "
    r"fees=(?P<fees>\S+) \| fails=(?P<fails>\d+) cb=(?P<cb>\S+) \| uptime=(?P<uptime>\d+)s"
)


def _to_decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except Exception:
        return Decimal("0")


def _parse_ts(ts: str, ms: str) -> dt.datetime:
    base = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return base.replace(microsecond=int(ms) * 1000)


@dataclass
class RepriceInfo:
    ts: dt.datetime
    side: str
    level: int
    best: Decimal
    target: Decimal
    offset_bps: Decimal


@dataclass
class OrderInfo:
    ts: dt.datetime
    side: str
    level: int
    price: Decimal
    size: Decimal
    ext_id: str
    exch_id: str
    offset_bps: Optional[Decimal] = None
    spread_bps: Optional[Decimal] = None
    status: Optional[str] = None
    terminal_ts: Optional[dt.datetime] = None


@dataclass
class StatusInfo:
    ts: dt.datetime
    bid: Decimal
    ask: Decimal
    spread_bps: Decimal


def _iter_log_lines(path: str) -> Iterable[Tuple[dt.datetime, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = LOG_RE.match(line)
            if not match:
                continue
            ts = _parse_ts(match.group("ts"), match.group("ms"))
            yield ts, match.group("msg")


def _nearest_status(statuses: List[StatusInfo], ts: dt.datetime) -> Optional[StatusInfo]:
    if not statuses:
        return None
    # statuses are appended in time order
    for status in reversed(statuses):
        if status.ts <= ts:
            return status
    return None


def _percentile(values: List[Decimal], pct: float) -> Optional[Decimal]:
    if not values:
        return None
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * Decimal(str(c - k))
    d1 = values_sorted[c] * Decimal(str(k - f))
    return d0 + d1


def analyze_log(path: str, max_reprice_gap_s: float, max_status_age_s: float) -> Tuple[List[OrderInfo], Dict[str, int], Dict[str, List[float]], List[RepriceInfo]]:
    reprices: Dict[Tuple[str, int], RepriceInfo] = {}
    orders: Dict[str, OrderInfo] = {}
    statuses: List[StatusInfo] = []
    counters = defaultdict(int)
    latencies: Dict[str, List[float]] = {"filled": [], "cancelled": [], "rejected": []}
    reprice_events: List[RepriceInfo] = []

    for ts, msg in _iter_log_lines(path):
        match = STATUS_RE.match(msg)
        if match:
            statuses.append(
                StatusInfo(
                    ts=ts,
                    bid=_to_decimal(match.group("bid")),
                    ask=_to_decimal(match.group("ask")),
                    spread_bps=_to_decimal(match.group("spread")),
                )
            )
            continue

        match = REPRICE_RE.match(msg)
        if match:
            side = match.group("side")
            level = int(match.group("level"))
            best = _to_decimal(match.group("best"))
            target = _to_decimal(match.group("target"))
            offset_bps = (abs(target - best) / best) * Decimal("10000") if best != 0 else Decimal("0")
            info = RepriceInfo(ts=ts, side=side, level=level, best=best, target=target, offset_bps=offset_bps)
            reprices[(side, level)] = info
            reprice_events.append(info)
            counters["reprices"] += 1
            continue

        match = ORDER_PLACED_RE.match(msg)
        if match:
            ext_id = match.group("ext_id")
            side = match.group("side")
            level = int(match.group("level"))
            price = _to_decimal(match.group("price"))
            size = _to_decimal(match.group("size"))
            exch_id = match.group("exch_id")

            info = OrderInfo(ts=ts, side=side, level=level, price=price, size=size, ext_id=ext_id, exch_id=exch_id)
            last_reprice = reprices.get((side, level))
            if last_reprice:
                gap = (ts - last_reprice.ts).total_seconds()
                if gap <= max_reprice_gap_s:
                    info.offset_bps = last_reprice.offset_bps
            last_status = _nearest_status(statuses, ts)
            if last_status and (ts - last_status.ts).total_seconds() <= max_status_age_s:
                info.spread_bps = last_status.spread_bps

            orders[ext_id] = info
            counters["placed"] += 1
            continue

        match = ORDER_TERMINAL_RE.match(msg)
        if match:
            ext_id = match.group("ext_id")
            status = match.group("status").lower()
            counters[status] += 1
            order = orders.get(ext_id)
            if order:
                order.status = status
                order.terminal_ts = ts
                latency = (ts - order.ts).total_seconds()
                latencies[status].append(latency)
            continue

        match = ORDER_CANCEL_REQ_RE.match(msg)
        if match:
            counters["cancel_requests"] += 1
            continue

        match = POST_ONLY_RE.match(msg)
        if match:
            counters["post_only_failed"] += 1
            continue

        match = FILL_RE.match(msg)
        if match:
            counters["fills"] += 1
            continue

    return list(orders.values()), counters, latencies, reprice_events


def _fmt_seconds(values: List[float]) -> str:
    if not values:
        return "-"
    return f"avg={mean(values):.2f}s med={median(values):.2f}s n={len(values)}"


def _fmt_decimal(value: Optional[Decimal]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _safe_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "inf"
    return f"{(numerator / denominator):.2f}"


def render_report(orders: List[OrderInfo], counters: Dict[str, int], latencies: Dict[str, List[float]], reprice_events: List[RepriceInfo]) -> str:
    offsets = [o.offset_bps for o in orders if o.offset_bps is not None]
    spreads = [o.spread_bps for o in orders if o.spread_bps is not None]
    ratio_values = [
        o.offset_bps / o.spread_bps
        for o in orders
        if o.offset_bps is not None and o.spread_bps is not None and o.spread_bps > 0
    ]

    placed_by_side_level: Dict[Tuple[str, int], int] = defaultdict(int)
    fills_by_side_level: Dict[Tuple[str, int], int] = defaultdict(int)
    cancels_by_level: Dict[int, int] = defaultdict(int)
    fills_by_level: Dict[int, int] = defaultdict(int)

    for order in orders:
        key = (order.side, order.level)
        placed_by_side_level[key] += 1
        if order.status == "filled":
            fills_by_side_level[key] += 1
            fills_by_level[order.level] += 1
        if order.status == "cancelled":
            cancels_by_level[order.level] += 1

    offset_p90 = _percentile(offsets, 90) if offsets else None
    spread_p50 = _percentile(spreads, 50) if spreads else None
    ratio_p90 = _percentile(ratio_values, 90) if ratio_values else None

    lines = []
    lines.append("MM log analysis")
    lines.append("")
    lines.append(f"Placed orders: {counters.get('placed', 0)}")
    lines.append(f"Fills: {counters.get('filled', 0)} | Cancels: {counters.get('cancelled', 0)} | Rejects: {counters.get('rejected', 0)}")
    lines.append(f"Post-only failed: {counters.get('post_only_failed', 0)}")
    lines.append(f"Reprices: {counters.get('reprices', 0)}")
    lines.append("")
    lines.append("Order lifetime:")
    lines.append(f"  filled:   {_fmt_seconds(latencies.get('filled', []))}")
    lines.append(f"  cancelled:{_fmt_seconds(latencies.get('cancelled', []))}")
    lines.append(f"  rejected: {_fmt_seconds(latencies.get('rejected', []))}")
    lines.append("")
    lines.append("Offsets (bps) at placement:")
    if offsets:
        lines.append(f"  avg={mean(offsets):.2f} med={median(offsets):.2f} p90={_fmt_decimal(offset_p90)}")
    else:
        lines.append("  -")
    lines.append("Spread bps at placement (nearest STATUS):")
    if spreads:
        lines.append(f"  avg={mean(spreads):.2f} med={median(spreads):.2f} p50={_fmt_decimal(spread_p50)}")
    else:
        lines.append("  -")
    lines.append("Offset/spread ratio at placement:")
    if ratio_values:
        lines.append(
            f"  avg={mean(ratio_values):.2f} med={median(ratio_values):.2f} p90={_fmt_decimal(ratio_p90)}"
        )
    else:
        lines.append("  -")
    lines.append("")
    lines.append("Fill rate by side/level:")
    if placed_by_side_level:
        for side in ("BUY", "SELL"):
            side_keys = sorted([key for key in placed_by_side_level if key[0] == side], key=lambda x: x[1])
            for key in side_keys:
                placed = placed_by_side_level[key]
                filled = fills_by_side_level.get(key, 0)
                fill_rate = (filled / placed * 100.0) if placed > 0 else 0.0
                lines.append(
                    f"  {side} L{key[1]}: placed={placed} filled={filled} fill_rate={fill_rate:.2f}%"
                )
    else:
        lines.append("  -")
    lines.append("Cancel per fill by level:")
    levels = sorted(set(cancels_by_level) | set(fills_by_level))
    if levels:
        for level in levels:
            cancels = cancels_by_level.get(level, 0)
            fills = fills_by_level.get(level, 0)
            lines.append(
                f"  L{level}: cancels={cancels} fills={fills} cancel_per_fill={_safe_ratio(cancels, fills)}"
            )
    else:
        lines.append("  -")

    return "\n".join(lines)


def write_csv(path: str, orders: List[OrderInfo]) -> None:
    fields = [
        "ts",
        "ext_id",
        "side",
        "level",
        "price",
        "size",
        "offset_bps",
        "spread_bps",
        "offset_to_spread_ratio",
        "status",
        "terminal_ts",
        "lifetime_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for order in orders:
            lifetime = None
            if order.terminal_ts is not None:
                lifetime = (order.terminal_ts - order.ts).total_seconds()
            writer.writerow(
                {
                    "ts": order.ts.isoformat(),
                    "ext_id": order.ext_id,
                    "side": order.side,
                    "level": order.level,
                    "price": str(order.price),
                    "size": str(order.size),
                    "offset_bps": str(order.offset_bps) if order.offset_bps is not None else "",
                    "spread_bps": str(order.spread_bps) if order.spread_bps is not None else "",
                    "offset_to_spread_ratio": (
                        str(order.offset_bps / order.spread_bps)
                        if order.offset_bps is not None
                        and order.spread_bps is not None
                        and order.spread_bps > 0
                        else ""
                    ),
                    "status": order.status or "",
                    "terminal_ts": order.terminal_ts.isoformat() if order.terminal_ts else "",
                    "lifetime_s": f"{lifetime:.6f}" if lifetime is not None else "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze market-maker logs for order placement vs spread/offsets."
    )
    parser.add_argument("logfile", help="Path to market_maker log file")
    parser.add_argument("--csv", help="Write per-order CSV to this path")
    parser.add_argument(
        "--max-reprice-gap",
        type=float,
        default=5.0,
        help="Max seconds between reprice and order placement to link offsets",
    )
    parser.add_argument(
        "--max-status-age",
        type=float,
        default=120.0,
        help="Max seconds between STATUS line and order placement to link spreads",
    )
    args = parser.parse_args()

    orders, counters, latencies, reprices = analyze_log(
        args.logfile, args.max_reprice_gap, args.max_status_age
    )

    print(render_report(orders, counters, latencies, reprices))

    if args.csv:
        write_csv(args.csv, orders)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
