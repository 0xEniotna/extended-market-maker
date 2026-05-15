# Stage 3 — Microprice Diagnostic — XNG-USD

## Funnel

- Journals scanned: 15
- Total `fill` events: 112
- Skipped (taker): 5
- Skipped (no book snapshot): 0
- Skipped (no +5s mid lookup): 3
- **Used in analysis: 104** (BUY 52 / SELL 52)

## (microprice − mid) distribution (bps of mid)

- mean: -0.139 bps
- stdev: 5.082 bps
- p05 / p50 / p95: -10.581 / 0.098 / 6.624 bps
- fraction \|mp−mid\| ≥ 1 bps: 64.4%
- fraction \|mp−mid\| ≥ 3 bps: 33.7%
- fraction \|mp−mid\| ≥ 5 bps: 21.2%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): -0.0567 (p=0.5664)
- Spearman ρ (pooled): +0.0805 (p=0.4146)
- Pearson r (BUY fills): -0.1654 (p=0.2356)
- Pearson r (SELL fills): +0.0835 (p=0.5534)

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: -7.287 bps
- stdev: 16.466 bps
- p05 / p50 / p95: -33.721 / -4.349 / 4.273 bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: NULL (r=-0.0567, p=0.5664).
