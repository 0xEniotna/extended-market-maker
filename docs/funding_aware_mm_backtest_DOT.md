# Funding-Aware Counterfactual Backtest — DOT-USD

- Journal: `/root/MM/data/mm_journal/mm_DOT-USD_latest.jsonl`
- Funding history: `data/funding_history/DOT-USD.json`
- Policy: coupling_bps_max=8, hold_horizon_periods=4, dollar_cap_pct_of_notional=0.001
- Events replayed: 4216
- Events skipped (bad data): 0
- Sanity-cap violations (|perturb| > coupling_bps_max): **0**
- Pearson(|F|, |perturb|): **+0.0000**  (positive ⇒ monotonic, expected)

## Overall perturbation distribution (quote_overlay − quote_base, bps of mid)

| metric | value |
|---|---|
| count | 4216 |
| mean | -0.5200 |
| median | -0.5202 |
| min | -0.5214 |
| max | -0.5186 |
| p95 | -0.5196 |
| p99 | -0.5194 |
| abs max | 0.5214 |

## By side

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| BUY | 2102 | -0.5196 | -0.5196 | 0.5199 |
| SELL | 2114 | -0.5204 | -0.5204 | 0.5214 |

## By regime

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| calm | 0 | – | – | – |
| normal | 4074 | -0.5200 | -0.5202 | 0.5206 |
| wide | 142 | -0.5200 | -0.5205 | 0.5214 |

## By funding sign

| bucket | count | mean | median | abs_max |
|---|---|---|---|---|
| + | 4216 | -0.5200 | -0.5202 | 0.5214 |
| 0 | 0 | – | – | – |
| - | 0 | – | – | – |
