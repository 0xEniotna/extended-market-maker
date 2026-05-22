# Stage 4 — OFI Mean-Reversion Diagnostic (Phase 2 / A2.1)

**Status**: NOT STARTED (blocked on Phase 0.5 book_change accrual)
**Pre-registered**: 2026-05-15
**Updated**: 2026-05-15 (purist signal switch — see below)
**Author**: Antoine
**Parent plan**: `docs/microprice_ofi_plan.md` Phase 2

---

## Goal

Validate Brief 18's claim ([2505.17388](https://arxiv.org/abs/2505.17388))
that **signed Order Flow Imbalance shocks mean-revert on short horizons**,
and concurrently compare it against the bot's existing L1 depth-imbalance
signal. Pick whichever is the stronger predictor of adverse markout for our
markets (if any).

Brief 18's CSI 300 result was on Chinese equity index futures. The brief
author flagged a crypto caveat:

> "OFI shocks in crypto can be liquidations that don't mean-revert (they trend)."

We need to verify the regime empirically on DOT, NEAR, XNG, plus ETH/MU
historical for power.

---

## Two signals tested

### Signal A — Flow-OFI (Brief 18 definition)

```
ΔV_b(t) =  bid_qty_t                if bid_price_t > bid_price_{t-1}
        =  bid_qty_t − bid_qty_{t-1} if bid_price_t = bid_price_{t-1}
        = −bid_qty_{t-1}             if bid_price_t < bid_price_{t-1}
(symmetric for ask)

OFI_window(t) = Σ_{s ∈ [t−w, t]}  ΔV_b(s) − ΔV_a(s)
```

Computed from the `book_change` event stream (Phase 0.5 output). Aggregated
over rolling window `w = imbalance_window_s` (default 2.0s).

### Signal B — L1 depth-imbalance (existing bot signal)

```
imb(t) = (bid_qty(t) − ask_qty(t)) / (bid_qty(t) + ask_qty(t))   ∈ [−1, +1]
imb_window(t) = EWMA of imb over imbalance_window_s
```

This is exactly what `orderbook_manager.orderbook_imbalance(window_s)` returns
today. Reconstructible from either the L1 state in `book_change` events, or
the pre-computed `fill.market_snapshot.imbalance` field.

---

## Pre-registered decision criterion

Run both signals against the same `+5s mid markout` outcome. Decide per
the following matrix:

| Signal A (flow-OFI) | Signal B (depth-imb) | Action |
|---|---|---|
| PASS | PASS | Use whichever has stronger Spearman ρ (typically A) in Phase 4 |
| PASS | FAIL | Use A. Brief 18 confirmed; depth-imbalance was a weaker proxy. |
| FAIL | PASS | Use B. Brief 18 doesn't transfer; the depth-ratio still works. |
| FAIL | FAIL | Skip A2. Crypto trends after order-flow shocks. Document. |

**PASS** = monotone mean-reverting relationship with subsequent markout,
Spearman p < 0.05, on at least one market with n ≥ 100 fills, and sign
consistent across that market's bid + ask sides.

**FAIL** = null, or trending sign (high positive signal → positive markout
for bid fills).

If only some markets PASS for a given signal, A2.2 only ships to the passing
markets.

---

## Phase 0.5 dependency

Signal A (flow-OFI) requires the `book_change` event stream from Phase 0.5.
Without it, only Signal B can be computed (from fill-time snapshots).

If, for whatever reason, Phase 0.5 is delayed: fall back to Signal B-only
diagnostic. Document this fallback in the verdict.

---

## Methodology

`scripts/diagnose_ofi.py` (to be written):

1. Accept `--journal <path>` (repeatable, pooled). Loads both `fill` and
   `book_change` events.
2. **For Signal A (flow-OFI)** — requires `book_change` events:
   - Walk the `book_change` stream chronologically.
   - At each event, compute `ΔV_b` and `ΔV_a` per the formula above.
   - Maintain a deque of `(ts, ΔV_b, ΔV_a)` over the last `imbalance_window_s` seconds.
   - On each `fill` event, sum the deque and compute the normalized OFI.
3. **For Signal B (depth-imbalance)** — works with either source:
   - For fills where `fill.market_snapshot.imbalance` is present, use it directly.
   - For fills predating Phase 0.5, recompute from `bids_top[0]/asks_top[0]` sizes.
4. Compute `+5s mid markout` for each fill (same as Stage 3).
5. Bucket fills by signal quintile (5 buckets per side, per signal).
6. Report:
   - Mean +5s markout per quintile, per side, per signal.
   - Spearman ρ between quintile rank and mean markout, per signal.
   - Side-by-side: Signal A ρ vs Signal B ρ.
   - Distribution of each signal (mean, std, percentiles).
   - Inventory-bucket stratification.

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


---

## VERDICT (ran 2026-05-22)

Status: COMPLETE.

Script: scripts/diagnose_ofi.py. Correlates trailing flow-OFI (Cont-
Kukanov-Stoikov, from book_change events) and the depth-imbalance ratio
against forward mid return, sampled spaced >= horizon.

| Market | n | Flow-OFI Pearson/Spearman | depth-imb | Verdict |
|---|---|---|---|---|
| DOT-USD | 11-24k | -0.20 / -0.23 | -0.03 | PASS (strong, mean-reverting) |
| TECH100m-USD | 9-22k | -0.01 to -0.11 | +0.05 to +0.08 | INCONCLUSIVE (weak, mixed) |

Findings:
1. Flow-OFI (CKS) is ~6x stronger than the depth-imbalance ratio the bot
   currently computes. Use FLOW-OFI from book_change deltas.
2. Sign is NEGATIVE (mean-reverting) on DOT: net buy pressure precedes a
   downward mid revert over 5-30s. Holds on crypto at this horizon.
3. Strong on crypto (DOT), weak/mixed on index (TECH100m) -- same pattern
   as the microprice diagnostic.
4. Horizon caveat: short-horizon (5-30s) transient-impact reversion,
   distinct from the multi-day macro trend bleed.

Decision: build OFI skew for DOT first (flow-OFI, mean-reversion sign),
after microprice ships. Skip TECH100m. Per-market reports:
docs/stage4_ofi_DOT.md, docs/stage4_ofi_TECH100m.md.
