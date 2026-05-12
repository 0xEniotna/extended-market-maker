# Phase 5 — MU_24_5-USD Paper-Trade Baseline

## Cutover

- **Cutover UTC**: 2026-05-12 15:48:28 (Wed)
- **Old PID**: 326445 (stopped via mmctl stop mu_24_5)
- **New PID**: 336299 (mmctl start mu_24_5.iter001 from ~/MM-funding-aware)
- **Old position at cutover**: -1.046 contracts (short), notional $766.53
  - Position is preserved on the exchange — bot picks it up at startup.
- **New process working dir**: /root/MM-funding-aware
- **New process PYTHONPATH**: /root/MM-funding-aware/src
- **New process env file**: /root/MM-funding-aware/.env.mu_24_5.iter001
  → symlinks to /root/MM/.env.mu_24_5.iter001
- **New process log**: /root/MM-funding-aware/data/mm_journal/mm_env_mu_24_5_iter001.log
- **New process journal**: /root/MM-funding-aware/data/mm_journal/mm_MU_24_5-USD_20260512_154828.jsonl

## Baseline window (overlay OFF) — for comparison

- Journal: /root/MM/data/mm_journal/mm_MU_24_5-USD_20260510_140543.1.jsonl
- Window: 2026-05-10 14:05:43 → 2026-05-12 15:48:28 UTC (≈49h35m)
- Closed positions: 8 (5W / 3L)
- Realized PnL: +91.737530 USD
  - trade_pnl: +91.051700 USD
  - funding_fees: +0.912702 USD (net positive — MU paid us)
  - close_fees: -0.226872 USD
- Open position: -1.046 contracts (short) @ notional 766.53 USD
  - realized_component: +35.796433 USD
  - unrealized_component: +30.872268 USD
- **Total incl open: +158.41 USD over 49h35m → ~+3.19 USD/h**

## Overrides in .env.mu_24_5.iter001

| Var | Baseline | iter001 |
|---|---|---|
| MM_FUNDING_AWARE_ENABLED | false (default) | **true** |
| MM_FUNDING_AWARE_COUPLING_BPS_MAX | n/a | 8 |
| MM_FUNDING_AWARE_HOLD_HORIZON_PERIODS | n/a | 4 |
| MM_FUNDING_AWARE_DOLLAR_CAP_PCT_OF_NOTIONAL | n/a | 0.001 |
| MM_MAX_POSITION_NOTIONAL_USD | 3000 | **500** |
| MM_FUNDING_BIAS_ENABLED | false | false |
| (everything else) | unchanged | unchanged |

Notional reduced 6x as a Phase-5 safety. Counterfactual gate (Gate 3 in
docs/funding_aware_mm_backtest.md) suggested ~3.2 bps overlay perturbations
on MU; with $500 max notional this caps the worst-case directional impact
of the new code at ~$0.16 per contract — within rollback envelope.

## 48h test window

- Start: 2026-05-12 15:48:28 UTC
- Target end: 2026-05-14 15:48:28 UTC
- Rollback path: cd /root/MM-funding-aware && mmctl stop mu_24_5.iter001
  && cd /root/MM && mmctl start mu_24_5  (re-uses original .env.mu_24_5)

## Caveats for comparison

1. **Notional dropped 6x.** Absolute PnL will be smaller per hour — must
   compare on a per-contract or per-notional basis, not absolute USD.
2. **Different code path** (worktree vs main). The 30-test integration
   suite asserts byte-identical quotes with overlay off, so any deviation
   from baseline IS attributable to the overlay (or to funding-rate /
   market drift over the comparison window).
3. **Market regime drift.** Compare 49h pre vs 48h post; check if
   funding rate, spread distribution, and trade volume look similar
   over both windows before claiming a delta is real.
4. **MU is 24/5.** Both windows include a Saturday (May 9 weekend
   in the baseline, May 16-17 weekend in the test). Funding accrues
   on weekends but no trading — exclude weekend hours from PnL/hour
   normalization.
