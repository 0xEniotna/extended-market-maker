# Per-Fill Markout Diagnostic — MU_24_5-USD

- N journals pooled: 4
- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_20260507_140129.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_20260507_140129.1.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_20260510_140543.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_20260510_140543.1.jsonl`
- Total fills: 104
- Taker fills (excluded — shutdown flatten / hedge): 3
- Resting-order fills analyzed: 101
- Side distribution: {'BUY': 57, 'SELL': 47}
- Mid timeline size (pooled): 203,490 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 101 | +22.113 | +11.095 | +0.768 | +28.426 | +95.312 | 23.8% | 76.2% | 43.753 |
| +5s | 101 | +20.943 | +9.928 | -1.132 | +34.488 | +94.266 | 28.7% | 71.3% | 44.398 |
| +30s | 101 | +21.099 | +8.824 | -10.636 | +37.539 | +99.081 | 33.7% | 66.3% | 51.357 |
| +300s | 101 | +21.854 | +31.365 | -30.471 | +73.852 | +184.567 | 40.6% | 59.4% | 110.412 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 55 | +7.963 | +7.314 | +7.125 | +6.491 |
| SELL | 46 | +39.030 | +37.240 | +37.807 | +40.223 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 10 | +3.691 | -11.402 | 70.0% |
| normal(5-20bps) | 47 | +5.657 | +8.146 | 36.2% |
| wide(>=20bps) | 44 | +41.193 | +28.426 | 11.4% |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 24 | -5.553 | -5.684 | 70.8% |
| med(5-15) | 33 | +4.567 | +8.146 | 24.2% |
| wide(>=15) | 44 | +47.678 | +41.961 | 9.1% |
| unknown | 0 | – | – | – |