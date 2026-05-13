# Per-Fill Markout Diagnostic — ETH-USD

- N journals pooled: 4
- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260505_171314.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.1.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_20260506_102154.2.jsonl`
- Total fills: 1620
- Taker fills (excluded — shutdown flatten / hedge): 82
- Resting-order fills analyzed: 1538
- Side distribution: {'BUY': 795, 'SELL': 825}
- Mid timeline size (pooled): 149,812 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 1538 | -1.919 | -1.502 | -2.856 | -0.624 | +1.900 | 83.0% | 16.9% | 2.717 |
| +5s | 1538 | -2.428 | -2.182 | -4.165 | -0.214 | +3.709 | 78.2% | 21.7% | 4.216 |
| +30s | 1538 | -2.889 | -2.697 | -7.324 | +1.975 | +11.519 | 65.8% | 34.1% | 9.585 |
| +300s | 1538 | -2.439 | -1.914 | -14.253 | +10.301 | +27.007 | 53.2% | 46.7% | 19.761 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 757 | -2.021 | -2.538 | -3.087 | -0.381 |
| SELL | 781 | -1.820 | -2.322 | -2.696 | -4.433 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 1534 | -2.437 | -2.185 | 78.2% |
| normal(5-20bps) | 4 | +0.818 | +2.850 | 50.0% |
| wide(>=20bps) | 0 | – | – | – |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 1 | -6.142 | -6.142 | 100.0% |
| tight(0-5) | 1489 | -2.374 | -1.986 | 78.4% |
| med(5-15) | 48 | -4.035 | -3.609 | 70.8% |
| wide(>=15) | 0 | – | – | – |
| unknown | 0 | – | – | – |