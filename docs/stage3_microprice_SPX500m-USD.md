# Stage 3 — Microprice Diagnostic — SPX500m-USD

## Funnel

- Journals scanned: 5
- Total `fill` events: 78
- Skipped (taker): 30
- Skipped (no book snapshot): 0
- Skipped (no +5s mid lookup): 3
- **Used in analysis: 45** (BUY 29 / SELL 16)

## (microprice − mid) distribution (bps of mid)

- mean: -1.676 bps
- stdev: 5.136 bps
- p05 / p50 / p95: -3.979 / 0.000 / 1.109 bps
- fraction \|mp−mid\| ≥ 1 bps: 53.3%
- fraction \|mp−mid\| ≥ 3 bps: 20.0%
- fraction \|mp−mid\| ≥ 5 bps: 4.4%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): -0.4901 (p=0.0002271) ***
- Spearman ρ (pooled): -0.2026 (p=0.1749)
- Pearson r (BUY fills): -0.4325 (p=0.01268) *
- Pearson r (SELL fills): -0.6552 (p=0.001174) **

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: -6.120 bps
- stdev: 6.224 bps
- p05 / p50 / p95: -17.075 / -4.423 / 1.093 bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: INCONCLUSIVE — only 45 fills (need ≥100). Pool with other markets.
