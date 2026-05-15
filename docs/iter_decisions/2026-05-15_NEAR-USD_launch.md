# NEAR-USD launch — iter001

**Date drafted**: 2026-05-15 13:35 UTC
**Type**: New market launch (no prior baseline)
**Iter file**: `/root/MM/.env.near.iter001`
**Snapshot of prior config**: `/root/MM/.env.near.pre-iter001.20260515`
**Run ID**: TBD (at launch)

This is a lightweight decision doc for adding a new market to the
running fleet. Full iter-doc structure isn't appropriate because there's
no prior baseline to compare against — we're establishing one.

---

## 1. Why NEAR

From fresh `mmctl markets screen` (2026-05-15 13:34 UTC):

- Score 15.9 (rank 9 in screen)
- Spread median 17.1 bps, p90 19.3 bps — **in our profitable bucket** (≥15)
- 24h volume $750K — higher than MU baseline ($475K)
- 100% spread-coverage at ≥3 bps
- Tick size 0.65 bps — reasonable granularity
- Min order $15 — easy
- **24/7 perp** — will trade this weekend, unlike 24/5 markets

Our pattern from killed-vs-running markets:
- Profitable: DOT (12bp), MU (14bp), XNG (15bp) — all ≥12 bps
- Killed for AS: ETH (1bp), SPX (5bp), 1000PEPE (7bp), ZRO (6bp)

NEAR at 17 bps fits comfortably in the profitable zone.

---

## 2. Config diff vs `.env.near` (which was last used long ago)

| Knob | Old `.env.near` | iter001 |
|---|---|---|
| `MM_DEADMAN_ENABLED` | true | **false** (SDK has no `set_deadman_switch`; XNG showed this errors every 5s) |
| `MM_MIN_OFFSET_BPS` | 4.5 | **6.0** (skip the toxic <5bps bucket per Stage 2 diagnostic) |

Everything else preserved from the old `.env.near`:
- Sizing: max_position 2000 contracts × NEAR price ~$1.52 ≈ $3,040 notional, capped by max_position_notional_usd=$5000
- Per-order cap: $500 (matches DOT's pre-doubled state)
- Drawdown: 5% × $5000 = **$250 abs threshold**
- Funding/trend/funding-aware/markout-feedback: all OFF (clean baseline)

---

## 3. Success criteria

Phase A — first 24h (mostly weekend, low activity expected):
- **No incidents**: no drawdown_stop, no excessive errors (<10/h excl. routine clips)
- **Stream health**: desync rate similar to peer markets (~5-20/h)
- **Quoting active**: orders placed both sides, balanced ratio (40-60% per side)
- **Position drift OK**: not pinned at one inventory boundary for >2h
- **At least 1 fill** would be a positive signal but not required (weekend)

Phase B — Monday morning (after futures markets reopen, fuller liquidity):
- **First markout diagnostic** when ≥20 resting fills accumulated:
  - Mean +5s markout ≥ −2.0 bps (vs killed markets: ETH was -2.4, SPX similar)
  - Per-side: no extreme asymmetry suggesting toxic flow on one side

If both phases clean → continue. If Phase B shows AS biting → re-evaluate
(possibly bump min_offset further, possibly kill).

---

## 4. Rollback triggers (immediate stop)

- `drawdown_stop` fires (auto: at $250 realized drawdown)
- Realized PnL < **−$30 in any 6h window**
- **>20 ERROR/h** sustained (excl. routine `risk_sizing` clips)
- Position pinned at hard limit for >1h with no rebalancing
- Quote latency p95 > **2× DOT/XNG baseline** sustained 10 min

Rollback procedure:
```bash
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl stop near.iter001'
```

The bot will auto-flatten any position on shutdown. No relaunch on
baseline `.env.near` (the old config is stale anyway — we've snapshotted
it pre-edit; baseline is now this iter001 by default).

---

## 5. Post-launch checks (immediate)

After launch, within 10 minutes:
1. `mmctl status` shows NEAR-USD RUNNING from `.env.near.iter001`
2. Log free of `ERROR`/`CRITICAL` (excl. routine warmup QTR)
3. **Deadman error gone** (the whole reason for changing the env)
4. Orders being placed both sides at reasonable offsets (min_offset≥6
   means orders at ≥6 bps from BBO)
5. `journal_config_history.py --market NEAR-USD` shows the iter001
   run_start with the new config

---

## 6. Post-mortem (filled when bot is stopped or first markout
   diagnostic is run)

- Window: TBD
- Resting fills: TBD
- Markout +5s: TBD
- W/L: TBD
- Realized PnL: TBD
- Decision: **[KEEP / KILL / EXTEND / INCONCLUSIVE]**
