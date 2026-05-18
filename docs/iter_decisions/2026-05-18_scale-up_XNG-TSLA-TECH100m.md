# Scale-up — XNG / TSLA_24_5 / TECH100m

**Date**: 2026-05-18 ~15:15 UTC
**Type**: Sizing change on 3 markets that demonstrated positive PnL over
the May 15-18 window.
**Iter files to create**:
- `/root/MM/.env.xng.iter001` (XNG was on baseline `.env.xng`)
- `/root/MM/.env.tsla_24_5.iter002` (was iter001)
- `/root/MM/.env.tech100m.iter002` (was iter001)

---

## 1. Why now

72h PnL since 2026-05-15 15:00 UTC (per `pnl_attribution` event sums):

| Market | TOTAL | Spread cap | Realized | Fills | Verdict |
|---|---|---|---|---|---|
| TECH100m | **+$84.41** | +$40.42 | +$38.94 | 31 (16/15) | clear winner, +$28/day |
| XNG | **+$28.06** | +$9.68 | +$15.48 | 16 (8/8) | profitable, +$9.3/day |
| TSLA_24_5 | **+$14.38** | +$1.25 | +$10.94 | 17 (7/10) | profitable, +$4.8/day |
| (NEAR) | +$0.44 | — | — | 15 | break-even, don't scale yet |
| (GOOG_24_5) | −$9.61 | — | — | 20 | losing modeste, don't scale |
| (DOT) | −$78.88 | — | — | 20 | inventory cleanup, don't scale |

Doubling these 3 → expected ~+$84/day if linear (vs current $42/day on
these 3). Caveat: not exactly linear — fills/h doesn't change, only
$/fill does.

## 2. What we change

Same pattern as the DOT doubling on 2026-05-13 (real fix was
`MM_MAX_ORDER_NOTIONAL_USD`, not the position-cap knobs).

| Knob | XNG (baseline) | XNG iter001 | TSLA (iter001) | TSLA iter002 | TECH100m (iter001) | TECH100m iter002 |
|---|---|---|---|---|---|---|
| `MM_MAX_ORDER_NOTIONAL_USD` | 400 | **800** | 300 | **600** | 300 | **600** |
| `MM_MAX_POSITION_NOTIONAL_USD` | 2000 | **4000** | 2000 | **4000** | 2000 | **4000** |
| `MM_MAX_POSITION_SIZE` | 800 | **1600** | 10 | 20 (headroom) | 0.5 | 0.5 (unchanged — plenty of room) |
| Everything else | — | unchanged | — | unchanged | — | unchanged |

Drawdown ABSOLUTE threshold scales with position notional:
- TSLA/TECH100m: 7.5% × $4000 = **$300 abs** (was $150)
- XNG: 5.0% × $4000 = **$200 abs** (was $100)

This is OK because we expect bigger absolute swings on bigger positions.
The drawdown pct does NOT need adjustment.

## 3. Success / rollback

**Success criteria** (24-48h window after scale-up):
- PnL/day scales **at least 1.5×** (so +$13 → +$20 minimum). Pure linear
  would be 2×; accept 1.5× because not all fills come in equal size.
- Fill rate stable or higher
- No incidents (drawdown_stop, etc.)
- Inventory drift not significantly worse

**Rollback** if any:
- `drawdown_stop` fires on any of the 3
- PnL/day drops below current level (i.e., doubling *destroys* value)
- Position pinned at hard limit for >2h

Rollback: stop iter, start prior baseline (XNG: `.env.xng`; TSLA/TECH:
their respective `iter001`). All snapshots preserved.

## 4. What we don't change

- **NEAR** is break-even at current size; need more data
- **DOT** is losing — bigger size = bigger loss. Wait for inventory unwind
- **GOOG_24_5** is losing; needs diagnosis (likely a TradFi-day-bias issue)
- **HOOD/ORCL** were prepared but not launched; separate decision

## 5. Procedure

For each market:
1. Snapshot the current iter file (preserve audit trail)
2. Create the new iter file with bumped values
3. Stop the current iter
4. Start the new iter
5. Verify settings live + first orders quoting both sides
