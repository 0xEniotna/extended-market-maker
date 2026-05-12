# Funding-Aware Counterfactual Backtest — SPX500m-USD

- Journal: `/root/MM/data/mm_journal/mm_SPX500m-USD_latest.jsonl`
- Funding history: `data/funding_history/SPX500m-USD.json`
- Policy: coupling_bps_max=8, hold_horizon_periods=4, dollar_cap_pct_of_notional=0.001
- Events replayed: 49282
- Events skipped (bad data): 0
- Sanity-cap violations (|perturb| > coupling_bps_max): **0**
- Pearson(|F|, |perturb|): **+0.9981**  (positive ⇒ monotonic, expected)

## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)

| metric | value |
|---|---|
| count | 49282 |
| mean | +1.6111 |
| median | +1.1209 |
| min | -3.0841 |
| max | +6.0094 |
| p95 | +5.8840 |
| p99 | +6.0045 |
| abs max | 6.0094 |

## By side

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| BUY | 16361 | +1.2902 | +0.7591 | 5.9989 |
| SELL | 32921 | +1.7705 | +1.1613 | 6.0094 |

## By regime

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| calm | 9 | +2.5403 | +5.8789 | 5.9989 |
| normal | 35943 | +1.5579 | +1.1205 | 6.0060 |
| wide | 13330 | +1.7539 | +1.2016 | 6.0094 |

## By funding sign

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| + | 5604 | -1.4929 | -0.6406 | 3.0841 |
| 0 | 5430 | +0.0000 | +0.0000 | 0.0000 |
| - | 38248 | +2.2946 | +1.2791 | 6.0094 |
