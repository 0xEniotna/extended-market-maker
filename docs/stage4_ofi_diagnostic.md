# Stage 4 — OFI Mean-Reversion Diagnostic (Phase 2 / A2.1)

**Status**: NOT STARTED
**Pre-registered**: 2026-05-15
**Author**: Antoine
**Parent plan**: `docs/microprice_ofi_plan.md` Phase 2

---

## Goal

Validate Brief 18's claim ([2505.17388](https://arxiv.org/abs/2505.17388)) that
**Order Flow Imbalance shocks mean-revert on short horizons**. Brief 18 showed
this empirically on CSI 300 Index Futures (Chinese equity index). The
brief author flagged a crypto-specific caveat:

> "OFI shocks in crypto can be liquidations that don't mean-revert (they trend)."

We need to verify which regime our specific markets (DOT, NEAR, XNG, ETH, MU)
are in **before wiring OFI as a skew term in quoting**.

---

## Pre-registered decision criterion

**Proceed to A2.2** if and only if:

> On at least one market with n ≥ 100 fills, OFI is **monotonically related**
> to subsequent +5s markout with the **mean-reverting sign**:
> - High positive OFI (buying pressure) quintile → negative mean markout for **bid** fills (i.e., price reverts down after the shock).
> - Symmetric for ask fills.
> - Spearman rank correlation across OFI quintiles, p < 0.05.

**Skip A2 entirely** if:
- High-positive OFI predicts **positive** markout (trending regime — Brief 18's crypto caveat confirmed).
- Or no monotone relationship in either direction.

If only some markets pass, A2.2 only ships to the passing markets.

---

## Methodology

`scripts/diagnose_ofi.py` (to be written):

1. Accept `--journal <path>` (repeatable, pooled).
2. For each tick where we have book state, compute the **rolling OFI**:
   - `Δbid_qty = bid_qty_t − bid_qty_{t-1}` (at the best bid level)
   - `Δask_qty = ask_qty_t − ask_qty_{t-1}` (at the best ask level)
   - Sum these signed quantity changes over a rolling window of `imbalance_window_s` seconds (default 2.0s).
   - `OFI_t = (Σ Δbid_qty − Σ Δask_qty) / (|Σ Δbid_qty| + |Σ Δask_qty|)` ∈ [−1, +1]
3. For each `fill` event:
   - Compute the prevailing `OFI` at fill time.
   - Compute `+5s mid markout` (same as Stage 3).
4. Bucket fills by OFI quintile (5 buckets per side).
5. Report:
   - Mean +5s markout per OFI quintile, per side.
   - Spearman ρ between OFI quintile rank and mean markout.
   - Sanity: distribution of OFI (mean, std, percentiles).
   - Stratified by inventory bucket.

---

## Reconstruction caveat (Phase 0 finding — 2026-05-15)

**Spot-check on `mm_DOT-USD_20260510_140533.jsonl` revealed:**

| Event type | Has bid/ask **sizes**? | Has top-of-book **prices**? | Has pre-computed `imbalance`? |
|---|---|---|---|
| `fill` (n=61 across 7 journals) | ✅ in `market_snapshot.bids_top/asks_top` (top 5 levels) | ✅ | ✅ in `market_snapshot.imbalance` |
| `order_placed` (n=341,750) | ❌ only `best_bid`, `best_ask`, `spread_bps` | ✅ | ❌ |
| `snapshot` (n=14,570, ~every 60s) | ❌ | ✅ | ❌ |
| `book_change` | doesn't exist | — | — |

**Implication**: true Brief 18-style OFI (signed flow deltas over a rolling
window between consecutive book updates) is **NOT reconstructible offline**
from existing journals. The book-update stream is not journaled at sub-event
resolution; only fill-time and quote-update-time snapshots exist, and
only fill events include sizes.

### Pragmatic alternative (adopted)

Use the **pre-computed `market_snapshot.imbalance`** at fill events as the
signal to validate. This is the L1 depth-ratio
`(bid_size - ask_size) / (bid_size + ask_size)` at top of book, **already
computed by the bot's `orderbook_manager.orderbook_imbalance(window_s)`** —
which is what a production OFI-skew policy would consume at runtime.

Diagnostic shifts from "validate Brief 18's flow-OFI mean-reversion" to
"validate that the L1-window-imbalance signal the bot already produces
predicts adverse markout in a tradeable mean-reverting way."

Distinction from Brief 18 must be documented in the verdict:
- Brief 18 measures signed flow (Δqty deltas). Our signal measures L1 depth ratio.
- The two correlate but are not identical. We are testing the signal we *can* use, not the signal Brief 18 used.

### Purist alternative (deferred)

If the pragmatic diagnostic is inconclusive or yields counterintuitive
results, the purist path is:

1. Add a `book_change` event type to journaling: every time the L1 book mutates,
   emit `{ts, type:"book_change", best_bid, best_ask, bid_qty, ask_qty}`.
2. Wait ~2 weeks of journal accrual on DOT/NEAR.
3. Re-run Stage 4 with true flow-OFI.

This is a separate phase (~1 day instrumentation + 2 weeks wait), deferred
unless Stage 4 with the pragmatic signal is inconclusive.

---

## Markets to test

Same set as Stage 3.

---

## Expected runtime

- Script write: ~6-8 hours (more complex than diagnose_microprice; OFI rolling
  window over journal stream needs care)
- Run + analysis: ~1-2 hours
- Writeup: ~1 hour
- **Total**: 1 day (or 1.5 if reconstruction requires instrumentation first)

---

## Results section (to be filled)

### OFI reconstruction feasibility

- DOT journal book-update resolution: TBD (Phase 0 spot-check)
- Decision: offline reconstruction OK / need instrumentation

### Per-market OFI distribution

| Market | n_observations | OFI mean | OFI std | p5 | p50 | p95 |
|---|---|---|---|---|---|---|
| DOT-USD | TBD | TBD | TBD | TBD | TBD | TBD |
| ... | | | | | | |

### Per-quintile +5s markout

#### DOT-USD — bid fills

| OFI quintile | n | mean +5s markout (bps) | std |
|---|---|---|---|
| Q1 (most negative) | TBD | TBD | TBD |
| Q2 | TBD | TBD | TBD |
| Q3 | TBD | TBD | TBD |
| Q4 | TBD | TBD | TBD |
| Q5 (most positive) | TBD | TBD | TBD |

Spearman ρ: TBD (target: negative ρ — high OFI → negative markout — for bid fills under mean-reversion)
p-value: TBD

#### DOT-USD — ask fills

(symmetric table)

#### Other markets

(same per-market subsections)

### Calibration outputs (if proceed)

If verdict is PASS, derive initial `k_ofi` per market from the slope of mean
markout per OFI unit. Save to `data/ofi_calibration/<MARKET>.json` for
Phase 4 use.

---

## Verdict

**[PASS / FAIL / INCONCLUSIVE]** — TBD

Per-market verdicts (one of these patterns expected):
- All markets PASS → proceed Phase 4 on all
- Only crypto markets PASS, 24/5 markets FAIL → proceed Phase 4 on DOT/NEAR/XNG only
- All FAIL → skip A2 entirely, document Brief 18's caveat as confirmed for our universe

Justification: TBD
