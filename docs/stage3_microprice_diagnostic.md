# Stage 3 — Microprice Diagnostic (Phase 1 / A1.1)

**Status**: NOT STARTED
**Pre-registered**: 2026-05-15
**Author**: Antoine
**Parent plan**: `docs/microprice_ofi_plan.md` Phase 1

---

## Goal

Confirm that microprice differs from mid in a way that **predicts where mid is heading on a short horizon**. If null on all markets, A1.2 (the live quoting change) is not worth shipping.

---

## Pre-registered decision criterion

**Proceed to A1.2** if and only if:

> On at least one market with n ≥ 100 fills, the Pearson correlation
> `corr(microprice − mid, +5s mid markout)` has magnitude ≥ 0.05 at p < 0.05,
> with sign consistent across markets ("microprice leads mid" — i.e. positive
> (microprice − mid) precedes positive markout on bid fills, negative on ask fills).

**Stop A1** if:
- All markets are null (|corr| < 0.05 or p ≥ 0.05)
- Or signs are inconsistent across markets (suggests the relationship is noise)

---

## Phase 0 finding (2026-05-15)

Spot-check on `mm_DOT-USD_20260510_140533.jsonl` confirmed:
- `fill` events include `market_snapshot.bids_top` and `asks_top` — top-5 levels with **price + size**. Microprice reconstructible directly from `bids_top[0]` and `asks_top[0]`.
- DOT had only 61 fills across 7 journals (~10 days) — too few for a power claim on DOT alone.
- ETH historical (per Stage 2: `stage2_markout_ETH_pooled.md`) has ≥1,538 resting fills available — sufficient power for the diagnostic. Will need to verify ETH journal structure is the same as DOT (likely yes; both schema_version 2).
- MU_24_5-USD has ~100 fills (per Stage 2) — useful as a 24/5 cross-check.
- `+5s mid markout`: reconstructible from the dense `order_placed` event stream (`best_bid`, `best_ask` every quote update) joined to fill timestamps. Same approach as `diagnose_markout.py`.

Conclusion: diagnostic is feasible offline. Will be ETH-pooled-dominated for power.

---

## Methodology

`scripts/diagnose_microprice.py` (to be written):

1. Accept `--journal <path>` (repeatable, pooled — same convention as `diagnose_markout.py`).
2. For each `fill` event:
   - Extract `bid`, `ask`, `bid_qty`, `ask_qty` from `market_snapshot`.
   - Compute `microprice = (bid·ask_qty + ask·bid_qty) / (bid_qty + ask_qty)`.
   - Compute `mid = (bid + ask) / 2`.
   - Compute `mp_minus_mid = microprice − mid` in absolute price units AND in bps of mid.
3. Join each fill to the +5s post-fill book snapshot (reuse `diagnose_markout.py` joining logic).
4. Compute `+5s markout` in bps, signed per side (positive = adverse for resting fills).
5. Output:
   - Pooled scatter / correlation: `corr(mp_minus_mid_bps, +5s_markout_bps)`, per side.
   - Stratified by inventory bucket (same buckets as `diagnose_markout.py`).
   - Distribution of `mp_minus_mid_bps` (mean, std, percentiles 1/5/25/50/75/95/99).
   - Sanity: how often does microprice differ from mid by ≥ 1 bp? ≥ 5 bps?

---

## Markets to test

| Market | Source | Fill count expected | Why |
|---|---|---|---|
| DOT-USD | `/root/MM/data/mm_journal/mm_DOT-USD_*.jsonl` | 50-80 in 1 week, 200+ in 1 month | Phase 3 live target |
| NEAR-USD | `/root/MM/data/mm_journal/mm_NEAR-USD_*.jsonl` | <50 (launched 2026-05-15) | Phase 5 target; may be too thin |
| XNG-USD | `/root/MM/data/mm_journal/mm_XNG-USD_*.jsonl` | TBD | Phase 5 target |
| ETH-USD (historical) | `/root/MM/data/mm_journal/mm_ETH-USD_*.jsonl` | ≥1,538 (from Stage 2 pool) | Power. Killed market — won't go live, but the relationship should hold there too if it's real |
| MU_24_5-USD (historical) | `/root/MM/data/mm_journal/mm_MU_24_5-USD_*.jsonl` | ~100 | Stage 2 markout baseline known |

---

## Expected runtime

- Script write: ~3-4 hours
- Run + analysis: ~1-2 hours
- Writeup: ~1 hour
- **Total**: ½ day

---

## Results section (to be filled)

### Per-market correlations

| Market | n_fills | corr(mp−mid, +5s markout) | p-value | Sign consistent? |
|---|---|---|---|---|
| DOT-USD | TBD | TBD | TBD | TBD |
| NEAR-USD | TBD | TBD | TBD | TBD |
| XNG-USD | TBD | TBD | TBD | TBD |
| ETH-USD | TBD | TBD | TBD | TBD |
| MU_24_5-USD | TBD | TBD | TBD | TBD |

### Distribution of (microprice − mid)

| Market | mean (bps) | std (bps) | p50 | p95 | p99 |
|---|---|---|---|---|---|
| DOT-USD | TBD | TBD | TBD | TBD | TBD |
| ... | | | | | |

### How often microprice ≠ mid materially

| Market | % events with \|mp−mid\| ≥ 1 bp | ≥ 3 bps | ≥ 5 bps |
|---|---|---|---|
| ... | | | |

---

## Verdict

**[PASS / FAIL / INCONCLUSIVE]** — TBD

Justification: TBD

Decision: proceed to Phase 3 / wait for more data / stop A1.
