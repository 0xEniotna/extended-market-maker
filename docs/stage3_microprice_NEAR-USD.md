# Stage 3 — Microprice Diagnostic — NEAR-USD

## Funnel

- Journals scanned: 2
- Total `fill` events: 0
- Skipped (taker): 0
- Skipped (no book snapshot): 0
- Skipped (no +5s mid lookup): 0
- **Used in analysis: 0** (BUY 0 / SELL 0)

## (microprice − mid) distribution (bps of mid)

- mean: n/a bps
- stdev: n/a bps
- p05 / p50 / p95: n/a / n/a / n/a bps
- fraction \|mp−mid\| ≥ 1 bps: 0.0%
- fraction \|mp−mid\| ≥ 3 bps: 0.0%
- fraction \|mp−mid\| ≥ 5 bps: 0.0%

## +5s raw-markout vs (microprice − mid)

Raw markout = (mid_after − fill_mid) / fill_mid × 1e4. Sign is directional (positive = mid went up), not side-flipped. Test: does microprice predict the direction of subsequent mid moves?

- Pearson r (pooled): n/a
- Spearman ρ (pooled): n/a
- Pearson r (BUY fills): n/a
- Pearson r (SELL fills): n/a

## MM-perspective markout (sign-flipped; positive = good for MM)

- mean: n/a bps
- stdev: n/a bps
- p05 / p50 / p95: n/a / n/a / n/a bps

## Pre-registered decision criterion

PROCEED to A1.2 if **|Pearson r| ≥ 0.05 with p < 0.05** on at least
one market with n ≥ 100 fills, sign positive (microprice leads mid).

**Verdict for this market**: INCONCLUSIVE — only 0 fills (need ≥100). Pool with other markets.
