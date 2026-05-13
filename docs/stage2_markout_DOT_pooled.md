# Per-Fill Markout Diagnostic — DOT-USD

- N journals pooled: 4
- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260506_095117.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260506_095117.1.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260510_140533.jsonl`
- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260510_140533.1.jsonl`
- Total fills: 52
- Taker fills (excluded — shutdown flatten / hedge): 2
- Resting-order fills analyzed: 50
- Side distribution: {'BUY': 26, 'SELL': 26}
- Mid timeline size (pooled): 243,748 observations

Convention: **markout in bps, signed from MM perspective**.
Positive = good for MM (we were on the right side of post-fill mid drift).
Negative = adverse selection biting (we got picked off).

## Overall markout distribution (all resting fills)

| horizon | count | mean | median | p25 | p75 | p95 | %neg | %pos | stdev |
|---|---|---|---|---|---|---|---|---|---|
| +1s | 50 | +21.031 | +20.206 | +12.505 | +27.351 | +41.126 | 4.0% | 96.0% | 13.021 |
| +5s | 50 | +21.770 | +20.573 | +12.137 | +31.021 | +41.131 | 2.0% | 98.0% | 14.286 |
| +30s | 50 | +21.479 | +22.068 | +10.720 | +29.014 | +43.365 | 4.0% | 94.0% | 15.698 |
| +300s | 50 | +18.077 | +23.172 | +2.501 | +34.074 | +44.107 | 22.0% | 78.0% | 23.816 |

## By side (median across all horizons; look for asymmetry)

| side | count | h1s mean | h5s mean | h30s mean | h300s mean |
|---|---|---|---|---|---|
| BUY | 26 | +25.340 | +27.721 | +27.374 | +24.236 |
| SELL | 24 | +16.363 | +15.323 | +15.093 | +11.405 |

## By regime at fill (5s markout)

| regime | count | mean | median | %neg |
|---|---|---|---|---|
| calm(<5bps) | 3 | -3.008 | +6.924 | 33.3% |
| normal(5-20bps) | 47 | +23.352 | +21.169 | 0.0% |
| wide(>=20bps) | 0 | – | – | – |

## By edge bucket at fill (5s markout)

edge_bps = post-only edge from BBO at fill time. Tighter quote =
more aggressive = should be more toxic if AS is biting.

| edge bucket | count | mean | median | %neg |
|---|---|---|---|---|
| neg_edge(<0) | 0 | – | – | – |
| tight(0-5) | 3 | +0.608 | +10.720 | 33.3% |
| med(5-15) | 9 | +10.883 | +10.752 | 0.0% |
| wide(>=15) | 38 | +26.019 | +24.142 | 0.0% |
| unknown | 0 | – | – | – |