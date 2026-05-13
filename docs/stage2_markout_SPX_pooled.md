# Per-Fill Markout Diagnostic — SPX500m-USD

- N journals pooled: 3
- Journal: `/root/MM/data/mm_journal/mm_SPX500m-USD_20260506_095117.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_SPX500m-USD_20260506_095117.1.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_SPX500m-USD_20260510_140536.jsonl`
- Total fills: 65
- Taker fills (excluded — shutdown flatten / hedge): 28
- Resting-order fills analyzed: 37
- Side distribution: {'BUY': 30, 'SELL': 35}
- Mid timeline size (pooled): 147,011 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 37 | +11.170 | +7.479 | +4.583 | +10.385 | +38.895 | 8.1% | 91.9% | 15.264 |
| +5s | 37 | +12.204 | +7.144 | +4.878 | +13.962 | +38.895 | 5.4% | 94.6% | 15.114 |
| +30s | 37 | +9.895 | +5.323 | +0.946 | +15.123 | +38.895 | 18.9% | 81.1% | 17.048 |
| +300s | 37 | +10.870 | +9.035 | +0.404 | +21.821 | +37.909 | 24.3% | 75.7% | 16.975 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 22 | +9.288 | +10.917 | +10.096 | +16.859 |
| SELL | 15 | +13.929 | +14.090 | +9.599 | +2.086 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 0 | – | – | – |
| normal(5-20bps) | 33 | +8.203 | +5.671 | 6.1% |
| wide(>=20bps) | 4 | +45.206 | +41.533 | 0.0% |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 0 | – | – | – |
| med(5-15) | 20 | +5.417 | +5.142 | 10.0% |
| wide(>=15) | 17 | +20.188 | +14.490 | 0.0% |
| unknown | 0 | – | – | – |