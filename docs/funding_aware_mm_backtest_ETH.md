# Funding-Aware Counterfactual Backtest — ETH-USD

- Journal: `/root/MM/data/mm_journal/mm_ETH-USD_latest.jsonl`
- Funding history: `data/funding_history/ETH-USD.json`
- Policy: coupling_bps_max=8, hold_horizon_periods=4, dollar_cap_pct_of_notional=0.001
- Events replayed: 49011
- Events skipped (bad data): 0
- Sanity-cap violations (|perturb| > coupling_bps_max): **0**
- Pearson(|F|, |perturb|): **+0.9923**  (positive ⇒ monotonic, expected)

## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)

| metric | value |
|---|---|
| count | 49011 |
| mean | -0.4239 |
| median | -0.5200 |
| min | -0.5203 |
| max | +0.8801 |
| p95 | +0.0800 |
| p99 | +0.8000 |
| abs max | 0.8801 |

## By side

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| BUY | 24527 | -0.4239 | -0.5200 | 0.8800 |
| SELL | 24484 | -0.4239 | -0.5200 | 0.8801 |

## By regime

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| calm | 49006 | -0.4239 | -0.5200 | 0.8801 |
| normal | 5 | -0.4481 | -0.5202 | 0.5203 |
| wide | 0 | – | – | – |

## By funding sign

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| + | 46207 | -0.4809 | -0.5200 | 0.5203 |
| 0 | 0 | – | – | – |
| - | 2804 | +0.5156 | +0.6000 | 0.8801 |
