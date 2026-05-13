# Per-Fill Markout Diagnostic — MU_24_5-USD

- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_20260510_140543.jsonl`
- Total fills: 42
- Taker fills (excluded — shutdown flatten / hedge): 0
- Resting-order fills analyzed: 42
- Side distribution: {'BUY': 22, 'SELL': 20}
- Mid timeline size: 72,860 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 42 | +12.145 | +2.166 | -5.332 | +13.228 | +35.730 | 42.9% | 57.1% | 51.903 |
| +5s | 42 | +13.340 | +7.026 | -10.159 | +20.263 | +48.081 | 40.5% | 59.5% | 55.908 |
| +30s | 42 | +12.517 | -2.017 | -23.978 | +34.358 | +75.057 | 52.4% | 47.6% | 64.784 |
| +300s | 42 | -6.964 | -13.663 | -62.028 | +69.149 | +180.273 | 54.8% | 45.2% | 145.801 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 22 | +2.062 | +5.401 | +10.997 | -2.776 |
| SELL | 20 | +23.237 | +22.072 | +14.191 | -11.571 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 9 | +4.231 | -12.165 | 66.7% |
| normal(5-20bps) | 19 | -2.906 | +3.732 | 47.4% |
| wide(>=20bps) | 14 | +41.243 | +16.274 | 14.3% |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 20 | -6.396 | -7.948 | 70.0% |
| med(5-15) | 11 | +7.298 | +9.532 | 9.1% |
| wide(>=15) | 11 | +55.265 | +39.310 | 18.2% |
| unknown | 0 | – | – | – |