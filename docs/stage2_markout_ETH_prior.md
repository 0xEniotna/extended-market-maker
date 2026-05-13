# Per-Fill Markout Diagnostic — ETH-USD-prior

- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.jsonl`
- Total fills: 745
- Taker fills (excluded — shutdown flatten / hedge): 68
- Resting-order fills analyzed: 677
- Side distribution: {'SELL': 377, 'BUY': 368}
- Mid timeline size: 67,184 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 677 | -1.792 | -1.484 | -2.747 | -0.219 | +1.986 | 79.9% | 20.1% | 2.883 |
| +5s | 677 | -2.374 | -2.302 | -4.150 | -0.211 | +3.715 | 76.1% | 23.8% | 4.119 |
| +30s | 677 | -2.612 | -2.282 | -6.763 | +1.976 | +11.235 | 65.1% | 34.7% | 9.490 |
| +300s | 677 | -1.942 | -1.099 | -13.971 | +10.677 | +26.932 | 52.0% | 48.0% | 18.221 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 337 | -1.943 | -2.566 | -2.423 | -1.773 |
| SELL | 340 | -1.641 | -2.184 | -2.800 | -2.109 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 676 | -2.377 | -2.302 | 76.0% |
| normal(5-20bps) | 1 | -0.652 | -0.652 | 100.0% |
| wide(>=20bps) | 0 | – | – | – |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 1 | -6.142 | -6.142 | 100.0% |
| tight(0-5) | 654 | -2.298 | -2.283 | 76.1% |
| med(5-15) | 22 | -4.459 | -4.014 | 72.7% |
| wide(>=15) | 0 | – | – | – |
| unknown | 0 | – | – | – |