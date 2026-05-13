# Per-Fill Markout Diagnostic — DOT-USD

- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260510_140533.jsonl`
- Total fills: 19
- Taker fills (excluded — shutdown flatten / hedge): 0
- Resting-order fills analyzed: 19
- Side distribution: {'BUY': 11, 'SELL': 8}
- Mid timeline size: 72,140 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 19 | +20.902 | +16.538 | +10.565 | +25.239 | +41.617 | 5.3% | 94.7% | 15.360 |
| +5s | 19 | +22.627 | +17.617 | +10.565 | +26.560 | +41.617 | 0.0% | 100.0% | 15.320 |
| +30s | 19 | +22.976 | +19.183 | +9.572 | +25.615 | +48.740 | 5.3% | 94.7% | 17.059 |
| +300s | 19 | +19.536 | +20.031 | -0.366 | +36.683 | +55.489 | 26.3% | 73.7% | 27.385 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 11 | +24.406 | +26.716 | +26.059 | +27.292 |
| SELL | 8 | +16.085 | +17.005 | +18.737 | +8.873 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 2 | +8.744 | +10.565 | 0.0% |
| normal(5-20bps) | 17 | +24.260 | +19.302 | 0.0% |
| wide(>=20bps) | 0 | – | – | – |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 1 | +17.617 | +17.617 | 0.0% |
| med(5-15) | 5 | +10.265 | +10.752 | 0.0% |
| wide(>=15) | 13 | +27.767 | +25.239 | 0.0% |
| unknown | 0 | – | – | – |