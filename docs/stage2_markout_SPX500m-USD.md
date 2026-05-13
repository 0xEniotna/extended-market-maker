# Per-Fill Markout Diagnostic — SPX500m-USD

- Journal: `/root/MM/data/mm_journal/mm_SPX500m-USD_20260510_140536.jsonl`
- Total fills: 12
- Taker fills (excluded — shutdown flatten / hedge): 0
- Resting-order fills analyzed: 12
- Side distribution: {'SELL': 6, 'BUY': 6}
- Mid timeline size: 65,971 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 12 | +16.915 | +11.001 | +5.480 | +11.543 | +26.456 | 0.0% | 100.0% | 20.996 |
| +5s | 12 | +18.372 | +11.260 | +5.671 | +15.123 | +30.534 | 0.0% | 100.0% | 20.572 |
| +30s | 12 | +17.409 | +11.260 | +5.323 | +15.256 | +30.534 | 0.0% | 100.0% | 21.212 |
| +300s | 12 | +5.679 | +9.035 | -3.383 | +15.715 | +23.532 | 25.0% | 75.0% | 19.836 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 6 | +6.617 | +7.560 | +6.568 | +5.280 |
| SELL | 6 | +27.214 | +29.185 | +28.250 | +6.078 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 0 | – | – | – |
| normal(5-20bps) | 11 | +12.930 | +11.050 | 0.0% |
| wide(>=20bps) | 1 | +78.234 | +78.234 | 0.0% |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 0 | – | – | – |
| med(5-15) | 7 | +8.290 | +7.144 | 0.0% |
| wide(>=15) | 5 | +32.487 | +27.545 | 0.0% |
| unknown | 0 | – | – | – |