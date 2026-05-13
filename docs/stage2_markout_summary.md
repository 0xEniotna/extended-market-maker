# Stage-2 Diagnostic — Per-Fill Markout Across 4 Markets

**Date**: 2026-05-13
**Question**: is adverse selection biting hard enough on our MM to justify
Stage-2 (adverse-selection-aware quoting)?

**Method**: `scripts/diagnose_markout.py` reads each market's journal,
builds a mid-price timeline from every event carrying BBO, then for
each resting `fill` computes the MM-perspective markout at +1s, +5s,
+30s, +5min. Positive markout = good for MM, negative = adverse
selection biting.

**Inputs** (live journals on mm-bot VPS):

| Market | Journal | Resting fills | Taker excluded |
|---|---|---|---|
| ETH-USD (latest) | `mm_ETH-USD_20260506_102154.1.jsonl` | 682 | 0 |
| ETH-USD (prior) | `mm_ETH-USD_20260506_102154.jsonl` | 677 | 68 |
| DOT-USD | `mm_DOT-USD_20260510_140533.jsonl` | 19 | 0 |
| SPX500m-USD | `mm_SPX500m-USD_20260510_140536.jsonl` | 12 | 0 |
| MU_24_5-USD | `mm_MU_24_5-USD_20260510_140543.jsonl` | 42 | 0 |

---

## TL;DR — verdict per market

| Market | n | mean markout +5s | mean markout +30s | %neg @ 5s | verdict |
|---|---|---|---|---|---|
| **ETH (latest)** | 682 | **−2.46 bps** | −3.06 bps | 80.8% | **AS biting, statistically rock-solid** |
| **ETH (prior)** | 677 | **−2.37 bps** | −2.61 bps | – | replicates latest, n=1,359 combined |
| DOT | 19 | +22.6 bps | +23.0 bps | – | positive but sample too small |
| SPX500m | 12 | +18.4 bps | +17.4 bps | – | positive but sample too small |
| MU_24_5 | 42 | +13.3 bps | +12.5 bps | 40.5% | mixed; edge-dependent (see below) |

### Headline conclusion

**Adverse selection is real and measurable on ETH.** Two independent
journals, 1,359 combined resting fills, mean markout −2.4 bps at +5s
with 80%+ of fills negative. Standard error of the mean is roughly
0.07 bps — the −2.4 bps signal is ~35σ from zero. This is not noise.

**On the other 3 markets, we don't have enough fills to call it.** DOT
and SPX show positive markouts but with n=12–19 the confidence interval
is wider than the effect. MU shows an interesting edge-bucket pattern
worth following up.

---

## ETH detail

### Overall (n=682, latest journal)

| horizon | mean | median | %neg |
|---|---|---|---|
| +1s | −1.88 bps | −1.51 | **86.4%** |
| +5s | −2.46 bps | −2.31 | **80.8%** |
| +30s | −3.06 bps | −2.79 | 65.8% |
| +300s | −2.90 bps | −2.36 | 54.3% |

Monotonic deepening to 30s, then plateau. Classic AS signature where
post-fill mid drifts against us in the first half-minute, then
neutralizes (consistent with informed-flow horizon ~30s).

### By side — symmetric AS

| side | n | h1s | h5s | h30s | h300s |
|---|---|---|---|---|---|
| BUY | 338 | −2.00 | −2.53 | −3.89 | +0.53 |
| SELL | 344 | −1.76 | −2.39 | −2.24 | −6.26 |

Both sides bleed roughly equally in the short run. **Symmetric
adverse selection** — informed flow hits both bids and asks. This is
expected on a competitive CLOB where directional information flows
randomly.

### By edge bucket at fill — wider quotes don't help (this is the key finding)

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| tight (0–5 bps) | 671 | −2.43 | 81.1% |
| med (5–15 bps) | 11 | **−4.04** | 63.6% |
| wide (≥15 bps) | 0 | – | – |

Naively you'd expect wider quotes to be safer (informed flow only
crosses tight quotes). On ETH that's NOT what we see — the small
sample of mid-edge fills is **worse**, not better. Implication: the
toxic flow on ETH isn't a simple "informed trader picks off our
tight quote" — it's more general post-trade drift. May respond to a
markout-feedback policy but probably not to a static-widen.

### Economic impact

- 1,359 resting fills over ~10 days of journal coverage.
- ETH `MM_MAX_POSITION_SIZE=5`, mid ~$2,300 → typical fill notional
  small (level-spread fractions of max), probably $500–$1,500 per fill.
- At $1,000 notional × 2.4 bps × 1,359 fills = **~$326 of AS bleed
  over the window**, or ~$32/day on ETH alone.
- Reference: fleet PnL is on the order of $50/day across 3 markets.
  ETH AS is roughly the same order of magnitude. **Recovering this
  is meaningful.**

---

## MU detail

### Edge-dependent toxicity

| edge bucket | n | mean markout +5s | %neg |
|---|---|---|---|
| tight (0–5 bps) | 20 | **−6.4** | **70.0%** |
| med (5–15 bps) | 11 | +7.3 | 9.1% |
| wide (≥15 bps) | 11 | +55.3 | 18.2% |

This is the most interesting per-market finding. On MU:
- Tight quotes (level-0) **lose** −6.4 bps on average.
- Wider quotes (level-1 + wide-regime) **win** +7 to +55 bps.

Interpretation: MU's order flow has a "informed-trader-exploits-tight-
quote" component. This is Paper A's `β > κ` regime — informed
counterparties trade aggressively into thin top-of-book. Widening
`MM_MIN_OFFSET_BPS` (currently 4 on MU) by a few bps would likely
move PnL.

Small caveat: n=20 in the tight bucket. Replicate on more journals
before betting on this.

### By side — strongly asymmetric on MU

| side | n | h1s | h5s | h30s | h300s |
|---|---|---|---|---|---|
| BUY | 22 | +2.1 | +5.4 | +11.0 | −2.8 |
| SELL | 20 | **+23.2** | **+22.1** | +14.2 | −11.6 |

Our SELL fills make money short-term, our BUY fills break even. This
likely means MU has had a downtrend over the journal window (selling
into a falling market is profitable). Not a static AS signature; a
market-regime signature. Less actionable.

---

## What this changes for Stage 2

**The diagnostic was the right first step.** It told us:

1. **Stage 2 is worth doing — but only as a markout-feedback policy on
   ETH, not as a full Paper-A/Paper-B port.** ETH has the signal, the
   sample size, and the symmetric AS pattern that responds to a simple
   per-side feedback loop.

2. **Paper A and Paper B's full machinery is overkill.** Neither was
   designed for our anonymous CLOB. A 50-LOC markout-feedback overlay
   captures the actionable insight.

3. **MU's pattern points to a config-level fix first.** Before any
   stage-2 code, try `MM_MIN_OFFSET_BPS` 4 → 6 or 4 → 8 on MU iter002.
   That's a one-line env change with no new code.

4. **DOT and SPX need more data before any decision.** Not enough
   fills to distinguish signal from noise.

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

**C. Re-run diagnostic in 2 weeks.** With another week of data, DOT
and SPX should have enough fills to call.

---

## Files

- `scripts/diagnose_markout.py` — the tool
- `docs/stage2_markout_ETH-USD.md` — ETH full report (n=682)
- `docs/stage2_markout_ETH_prior.md` — ETH cross-check (n=677)
- `docs/stage2_markout_DOT-USD.md` — DOT (n=19, inconclusive)
- `docs/stage2_markout_SPX500m-USD.md` — SPX (n=12, inconclusive)
- `docs/stage2_markout_MU_24_5-USD.md` — MU full report (n=42)
