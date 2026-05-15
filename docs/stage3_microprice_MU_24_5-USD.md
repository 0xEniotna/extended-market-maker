# Stage 3 — Microprice Diagnostic — MU_24_5-USD

## Funnel

- Journals scanned: 5
- Total `fill` events: 162
- Skipped (taker): 5
- Skipped (no book snapshot): 0
- Skipped (no +5s mid lookup): 0
- **Used in analysis: 157** (BUY 85 / SELL 72)

## (microprice − mid) distribution (bps of mid)

- mean: -0.688 bps
- stdev: 9.644 bps
- p05 / p50 / p95: -10.901 / -3.015 / 14.395 bps
- fraction \|mp−mid\| ≥ 1 bps: 82.8%
- fraction \|mp−mid\| ≥ 3 bps: 70.7%
- fraction \|mp−mid\| ≥ 5 bps: 52.9%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): -0.2456 (p=0.001606) **
- Spearman ρ (pooled): +0.0696 (p=0.3854)
- Pearson r (BUY fills): +0.2451 (p=0.02126) *
- Pearson r (SELL fills): -0.6047 (p=2.125e-10) ***

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: -3.475 bps
- stdev: 18.630 bps
- p05 / p50 / p95: -29.034 / -3.640 / 22.255 bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: WRONG SIGN (r=-0.2456, p=0.001606) — microprice anti-predicts mid. Investigate before shipping.
