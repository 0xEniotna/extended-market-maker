# Funding-Aware Counterfactual Backtest — DOT-USD

- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_20260506_095117.1.jsonl`
- Funding history: `data/funding_history/DOT-USD.json`
- Policy: coupling_bps_max=8, hold_horizon_periods=4, dollar_cap_pct_of_notional=0.001
- Events replayed: 63592
- Events skipped (bad data): 0
- Sanity-cap violations (|perturb| > coupling_bps_max): **0**
- Pearson(|F|, |perturb|): **+1.0000**  (positive ⇒ monotonic, expected)

## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)

| metric | value |
|---|---|
| count | 63592 |
| mean | +0.3482 |
| median | -0.2399 |
| min | -0.5209 |
| max | +4.0039 |
| p95 | +2.0817 |
| p99 | +3.9985 |
| abs max | 4.0039 |

## By side

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| BUY | 31703 | +0.3486 | -0.2398 | 3.9988 |
| SELL | 31889 | +0.3477 | -0.2402 | 4.0039 |

## By regime

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| calm | 99 | +0.6687 | +1.6397 | 1.8404 |
| normal | 62870 | +0.3490 | -0.2399 | 4.0039 |
| wide | 623 | +0.2109 | -0.2398 | 3.2035 |

## By funding sign

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| + | 37074 | -0.4560 | -0.5197 | 0.5209 |
| 0 | 0 | – | – | – |
| - | 26518 | +1.4725 | +1.4391 | 4.0039 |
