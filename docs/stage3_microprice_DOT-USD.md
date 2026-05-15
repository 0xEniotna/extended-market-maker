# Stage 3 — Microprice Diagnostic — DOT-USD

## Funnel

- Journals scanned: 7
- Total `fill` events: 61
- Skipped (taker): 3
- Skipped (no book snapshot): 0
- Skipped (no +5s mid lookup): 5
- **Used in analysis: 53** (BUY 29 / SELL 24)

## (microprice − mid) distribution (bps of mid)

- mean: 0.304 bps
- stdev: 2.431 bps
- p05 / p50 / p95: -4.347 / 0.000 / 4.161 bps
- fraction \|mp−mid\| ≥ 1 bps: 60.4%
- fraction \|mp−mid\| ≥ 3 bps: 20.8%
- fraction \|mp−mid\| ≥ 5 bps: 5.7%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): +0.0218 (p=0.8763)
- Spearman ρ (pooled): +0.0463 (p=0.7408)
- Pearson r (BUY fills): -0.1064 (p=0.5781)
- Pearson r (SELL fills): +0.1100 (p=0.6037)

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: -2.160 bps
- stdev: 7.324 bps
- p05 / p50 / p95: -13.529 / -2.254 / 8.034 bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: INCONCLUSIVE — only 53 fills (need ≥100). Pool with other markets.
