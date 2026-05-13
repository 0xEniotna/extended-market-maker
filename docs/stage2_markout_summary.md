# Stage-2 Diagnostic — Per-Fill Markout Across 4 Markets (pooled)

**Date**: 2026-05-13
**Question**: is adverse selection biting hard enough on our MM to justify
Stage-2 (adverse-selection-aware quoting)?

**Method**: `scripts/diagnose_markout.py` reads each market's journals,
builds a mid-price timeline (per-journal, to avoid cross-boundary
interpolation), then for each resting `fill` computes the MM-perspective
markout at +1s, +5s, +30s, +5min. Fills pooled across all journals for
the market. Positive markout = good for MM, negative = adverse selection
biting.

**Inputs** — pooled across ALL substantial journals on mm-bot VPS for
each market (May 2026 run; March run on different code excluded):

| Market | N journals | Total raw fills | Taker excl. | Resting analyzed |
|---|---|---|---|---|
| ETH-USD | 4 | 1,620 | 82 | **1,538** |
| DOT-USD | 4 | 52 | 2 | **50** |
| SPX500m-USD | 3 | 65 | 28 | **37** |
| MU_24_5-USD | 4 | 104 | 3 | **101** |

---

## TL;DR — verdict per market

| Market | n | mean +5s | tight-edge mean | wide-edge mean | verdict |
|---|---|---|---|---|---|
| **ETH** | 1,538 | **−2.43 bps** | −2.37 (n=1,489) | (no wide fills) | **Symmetric AS, not edge-dependent** |
| **MU_24_5** | 101 | +20.9 bps overall | **−5.55** (n=24) | +47.7 (n=44) | **Edge-dependent toxicity** |
| DOT | 50 | +21.8 bps | +0.6 (n=3) | +26.0 (n=38) | No AS problem |
| SPX500m | 37 | +12.2 bps | – (n=0) | +20.2 (n=17) | No AS problem |

### Headline conclusion

**Adverse selection is real on ETH and on the tight bucket of MU.
It's NOT a fleet-wide problem.**

- **ETH**: −2.43 bps mean markout at +5s on n=1,538 fills pooled across
  4 journals. Standard error ~0.07 bps → ~35σ from zero. Symmetric on
  both sides (BUY −2.5, SELL −2.3) and not edge-dependent — even the
  small mid-edge bucket is worse (−4.0 bps). Widening alone won't fix
  this; needs a markout-feedback policy.

- **MU**: overall markout is +20.9 bps, but this hides a clear pattern.
  The 24 fills at tight edge (0-5 bps) lose −5.5 bps each; the 77 fills
  at med+wide edge make +4.6 to +47.7. **The tight bucket is toxic and
  the rest is clean** — exactly Paper A's `β > κ` (informed-trader-
  exploits-tight-quote) regime. Raising `MM_MIN_OFFSET_BPS` would
  selectively cut the toxic bucket.

- **DOT** and **SPX**: both clean. n is moderate (50 / 37) but with
  >+10 bps mean markout and 0-10% negative, AS is not a problem.

---

## ETH detail

### Overall (n=1,538 pooled across 4 May journals)

| horizon | mean | median |
|---|---|---|
| +1s | −1.92 bps | −1.50 |
| +5s | −2.43 bps | −2.18 |
| +30s | −2.89 bps | −2.70 |
| +300s | −2.44 bps | −1.91 |

Monotonic deepening to 30s, then plateau. Classic AS signature where
post-fill mid drifts against us in the first half-minute, then
neutralizes (consistent with informed-flow horizon ~30s).

### By side — symmetric AS

| side | n | h1s | h5s | h30s | h300s |
|---|---|---|---|---|---|
| BUY | 757 | −2.02 | −2.54 | −3.09 | −0.38 |
| SELL | 781 | −1.82 | −2.32 | −2.70 | −4.43 |

Both sides bleed roughly equally in the short run. **Symmetric
adverse selection** — informed flow hits both bids and asks. This is
expected on a competitive CLOB where directional information flows
randomly.

### By edge bucket at fill — wider quotes don't help

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| neg_edge (<0) | 1 | −6.14 | 100.0% |
| tight (0–5 bps) | 1,489 | −2.37 | 78.4% |
| med (5–15 bps) | 48 | **−4.04** | 70.8% |
| wide (≥15 bps) | 0 | – | – |

96.8% of ETH fills happen at tight (0-5 bps) edge — that's where AS
manifests. The mid-edge bucket (48 fills) is **worse**, not better.
Implication: the toxic flow on ETH isn't a simple "informed trader
picks off our tight quote" — it's general post-trade drift. Static
widening probably won't help; a markout-feedback policy that responds
to recent fill outcomes can.

### Economic impact

- 1,538 resting fills over ~10 days of pooled journal coverage.
- ETH `MM_MAX_POSITION_SIZE=5`, mid ~$2,300 → typical fill notional
  small (a fraction of max, depending on level), probably $500–$1,500.
- At $1,000 notional × 2.4 bps × 1,538 fills = **~$370 of AS bleed**
  over the window, or ~$37/day on ETH alone.
- Reference: fleet PnL ~$50/day across 3 markets. ETH AS bleed is
  on the same order. **Recovering it is meaningful.**

---

## MU detail (n=101 pooled across 4 journals)

### Edge-dependent toxicity — confirmed at n=101

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| tight (0–5 bps) | 24 | **−5.55** | **70.8%** |
| med (5–15 bps) | 33 | +4.57 | 24.2% |
| wide (≥15 bps) | 44 | +47.68 | 9.1% |

The earlier n=42 finding replicates. On MU:
- Tight quotes **lose** −5.5 bps on average (24 fills, 71% negative).
- Mid-edge quotes **win** +4.6 bps (33 fills).
- Wide quotes **win big** +47.7 bps (44 fills, 91% positive).

Interpretation: MU's order flow has an "informed-trader-exploits-tight-
quote" component. This is Paper A's `β > κ` regime — informed
counterparties trade aggressively into thin top-of-book. Raising
`MM_MIN_OFFSET_BPS` (currently 4 on MU) selectively cuts the toxic
bucket while preserving the profitable mid+wide buckets.

The aggregate markout is **+20.9 bps** because the wide bucket
dominates; this masks the tight-bucket toxicity that's worth fixing.

### By side — strongly asymmetric on MU

| side | n | h1s | h5s | h30s | h300s |
|---|---|---|---|---|---|
| BUY | 55 | +7.96 | +7.31 | +7.13 | +6.49 |
| SELL | 46 | **+39.03** | **+37.24** | +37.81 | +40.22 |

SELL fills make 4-5× more than BUY fills. The pattern persists across
all horizons including +5min, so this isn't a short-term post-trade
drift artifact — it's a structural asymmetry in MU's price action
over the journal window (likely downtrend favoring shorts). Not a
classical AS signature; more a market-regime artifact. Less actionable
than the edge-bucket finding.

---

## DOT & SPX detail — confirmed no AS problem

### DOT (n=50)

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| tight (0–5 bps) | 3 | +0.61 | 33% |
| med (5–15 bps) | 9 | +10.88 | 0% |
| wide (≥15 bps) | 38 | +26.02 | 0% |

By side: BUY +27.7, SELL +15.3 — both positive. 76% of fills happen
at wide edge; the bot is essentially printing money at +26 bps/fill
on the wide bucket. No intervention needed.

### SPX500m (n=37)

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| med (5–15 bps) | 20 | +5.42 | 10% |
| wide (≥15 bps) | 17 | +20.19 | 0% |

By side: BUY +10.9, SELL +14.1 — symmetric and clean. Zero tight fills
(bot doesn't quote tight on SPX). No intervention needed.

---

## What this changes for Stage 2

The pooled diagnostic told us:

1. **AS is real on ETH** (n=1,538, +5s markout −2.43 bps, ~35σ from
   zero). Worth ~$37/day in avoidable bleed. Symmetric, not edge-
   dependent. Needs a **markout-feedback policy**.

2. **AS is real on MU's tight bucket** (n=24, −5.55 bps, 71% negative).
   But MU's overall markout is +20.9 bps because the wider buckets are
   very profitable. Selectively cutting the tight bucket via
   `MM_MIN_OFFSET_BPS` would isolate the toxic flow without losing
   the good flow. **Config-only experiment.**

3. **DOT and SPX have no AS problem.** With n=50/37 and overall markouts
   +21.8/+12.2 bps, those bots work. Nothing to do.

4. **Paper A and Paper B are wrong tools.** Both are dealer/broker
   models for client-identified flow. Our ETH problem is a simple
   per-side feedback signal; a 50-LOC overlay captures it.

### Proposed Stage 2 plan (revised, scoped down)

**A. ETH markout-feedback overlay** — new feature, behind
`MM_MARKOUT_FEEDBACK_ENABLED=false` flag:

```python
# Maintain per-side EWMA of recent post-fill markout (+5s horizon).
# Update on every fill event.
ewma_markout_bid: Decimal  # if negative → BUY fills are toxic
ewma_markout_ask: Decimal  # if negative → SELL fills are toxic

# In compute_target_price, add a per-side widening:
if ewma_markout_bid < -threshold_bps:
    bid_offset_extra = clamp(α × |ewma_markout_bid|, 0, cap_bps)
    bid raw price ← bid raw price − bid_offset_extra
# Symmetric for ask.
```

Knobs: EWMA half-life, threshold, gain α, cap. Calibrate on journal
replay against measured ETH markout series; should reduce mean
markout magnitude while accepting some fill-rate loss.

Same rollout discipline as Stage 1: replay → paper-trade → live A/B,
per-flag.

**B. MU config-only experiment** — bump `MM_MIN_OFFSET_BPS` from 4 to
6 on a `.env.mu_24_5.iter002` and run 48h. No code change. Compare to
baseline.

**C. DOT/SPX**: re-run diagnostic in 2-4 weeks for more fills, but
prior is "leave them alone — they work".

---

## Files

- `scripts/diagnose_markout.py` — the tool (supports multi-journal pooling)
- `docs/stage2_markout_ETH_pooled.md` — ETH report (n=1,538, 4 journals)
- `docs/stage2_markout_MU_pooled.md` — MU report (n=101, 4 journals)
- `docs/stage2_markout_DOT_pooled.md` — DOT report (n=50, 4 journals)
- `docs/stage2_markout_SPX_pooled.md` — SPX report (n=37, 3 journals)
- Earlier per-journal reports retained for reference (single-journal series)
