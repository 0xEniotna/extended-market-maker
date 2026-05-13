# Per-Fill Markout Diagnostic — ETH-USD

- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.1.jsonl`
- Total fills: 682
- Taker fills (excluded — shutdown flatten / hedge): 0
- Resting-order fills analyzed: 682
- Side distribution: {'SELL': 344, 'BUY': 338}
- Mid timeline size: 66,716 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 682 | -1.881 | -1.511 | -3.171 | -0.641 | +1.106 | 86.4% | 13.5% | 2.221 |
| +5s | 682 | -2.460 | -2.312 | -4.165 | -0.218 | +2.862 | 80.8% | 19.1% | 3.943 |
| +30s | 682 | -3.060 | -2.789 | -7.514 | +2.339 | +10.305 | 65.8% | 34.0% | 9.455 |
| +300s | 682 | -2.898 | -2.358 | -14.328 | +10.861 | +25.404 | 54.3% | 45.6% | 21.166 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 338 | -2.002 | -2.530 | -3.890 | +0.526 |
| SELL | 344 | -1.761 | -2.391 | -2.243 | -6.262 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 679 | -2.477 | -2.312 | 81.0% |
| normal(5-20bps) | 3 | +1.309 | +2.850 | 33.3% |
| wide(>=20bps) | 0 | – | – | – |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 671 | -2.434 | -2.312 | 81.1% |
| med(5-15) | 11 | -4.039 | -2.366 | 63.6% |
| wide(>=15) | 0 | – | – | – |
| unknown | 0 | – | – | – |