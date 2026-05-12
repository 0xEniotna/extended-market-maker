# Funding-Aware Counterfactual Backtest — MU_24_5-USD

- Journal: `/root/MM/data/mm_journal/mm_MU_24_5-USD_latest.jsonl`
- Funding history: `data/funding_history/MU_24_5-USD.json`
- Policy: coupling_bps_max=8, hold_horizon_periods=4, dollar_cap_pct_of_notional=0.001
- Events replayed: 29930
- Events skipped (bad data): 0
- Sanity-cap violations (|perturb| > coupling_bps_max): **0**
- Pearson(|F|, |perturb|): **+1.0000**  (positive ⇒ monotonic, expected)

## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)

| metric | value |
|---|---|
| count | 29930 |
| mean | +0.6008 |
| median | +0.3596 |
| min | -0.5268 |
| max | +3.2461 |
| p95 | +3.2368 |
| p99 | +3.2442 |
| abs max | 3.2461 |

## By side

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| BUY | 15039 | +0.5730 | +0.3595 | 3.2390 |
| SELL | 14891 | +0.6290 | +0.3604 | 3.2461 |

## By regime

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| calm | 2 | +1.7202 | +1.7202 | 1.7202 |
| normal | 8083 | +1.0618 | +0.8406 | 3.2432 |
| wide | 21845 | +0.4302 | -0.0400 | 3.2461 |

## By funding sign

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| + | 14287 | -0.4346 | -0.5193 | 0.5268 |
| 0 | 0 | – | – | – |
| - | 15643 | +1.5465 | +1.0410 | 3.2461 |
