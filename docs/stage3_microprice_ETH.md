# Stage 3 — Microprice Diagnostic — ETH-USD

## Funnel

- Journals scanned: 24
- Total `fill` events: 2331
- Skipped (taker): 159
- Skipped (no book snapshot): 17
- Skipped (no +5s mid lookup): 10
- **Used in analysis: 2145** (BUY 1055 / SELL 1090)

## (microprice − mid) distribution (bps of mid)

- mean: 0.002 bps
- stdev: 0.479 bps
- p05 / p50 / p95: -0.525 / 0.036 / 0.245 bps
- fraction \|mp−mid\| ≥ 1 bps: 3.9%
- fraction \|mp−mid\| ≥ 3 bps: 0.3%
- fraction \|mp−mid\| ≥ 5 bps: 0.1%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): +0.4085 (p=2.492e-95) ***
- Spearman ρ (pooled): +0.5955 (p=5.069e-258) ***
- Pearson r (BUY fills): +0.1048 (p=0.000626) ***
- Pearson r (SELL fills): +0.1727 (p=7.257e-09) ***

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: -3.431 bps
- stdev: 5.324 bps
- p05 / p50 / p95: -12.704 / -2.626 / 3.113 bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: PASS (r=+0.4085, p=2.492e-95).
