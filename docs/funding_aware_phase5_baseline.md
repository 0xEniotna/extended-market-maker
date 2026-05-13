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

---

## Restart after drawdown_stop incident

### What happened (first run)
- 2026-05-12 15:48:28 UTC → started (pid 336299).
- 2026-05-12 18:02:24 UTC → `drawdown_stop` triggered (2h13m uptime).
  - peak_pnl=+11.36, current_pnl=-14.26, drawdown=$25.62 vs threshold $25.
  - Flatten BUY 0.701 @ 748.95, slippage 20 bps.
  - 3 fills, 5,773 cancellations, total fees $0.131.
- Position cleanly flattened to 0 on shutdown.

### Root cause
`MM_DRAWDOWN_STOP_PCT_OF_MAX_NOTIONAL=5.0` scales with `MM_MAX_POSITION_NOTIONAL_USD`:

- Production (baseline): 5% × $3,000 = **$150 absolute drawdown threshold**.
- iter001 (first run): 5% × $500 = **$25 absolute drawdown threshold**.

Cutting notional 6× also cut the drawdown threshold 6×. With MU's
~2% daily range and an inherited -1.046 short position, $25 fires fast.

**The funding-aware overlay did not cause this.** Max overlay shift on MU
≈ 3.25 bps ≈ $0.16/contract — three orders of magnitude smaller than
the $25 drawdown trigger. The PnL move was driven by mid going
733.59 → ~747.5 over 2h while we were short.

### Fix (option B)
Bumped `MM_DRAWDOWN_STOP_PCT_OF_MAX_NOTIONAL` from 5.0 → 30.0 in iter001
so the absolute threshold matches production ($150). The 6× notional cut
still preserves Phase-5 conservatism on position size.

### Second run
- 2026-05-12 20:06:19 UTC → started (pid 337803).
- Position at restart: 0 (clean slate).
- Settings verified live: drawdown_pct=30.0 → $150 threshold,
  funding_aware=True, max_notional=$500.
- New journal: `mm_MU_24_5-USD_20260512_200619.jsonl`.
- 48h target: 2026-05-14 20:06 UTC.

### Updated comparison plan
The first 2h13m of run 1 produced a clean drawdown_stop event. That
window is **excluded** from the Phase-5 analysis because the drawdown
fired on a *config* problem, not on the overlay. The 48h test starts
fresh from 20:06 UTC.

---

## iter001 → iter002 cutover (Stage-2 widening experiment)

### Why
Stage-2 markout diagnostic (`docs/stage2_markout_summary.md`) showed
MU has edge-dependent toxicity: the tight (0-5 bps edge) fill bucket
loses −5.55 bps mean markout on n=24, while med+wide buckets are
strongly profitable. Hypothesis: raising `MM_MIN_OFFSET_BPS` from 4
to 6 cuts the toxic bucket without losing the profitable ones.

### iter001 final state (stopped at 2026-05-13 ~11:11 UTC)
- Window: 2026-05-12 20:06:19 → 2026-05-13 11:11 UTC (≈15h)
- Realized PnL contribution: +$72.52 (cumulative $164.26 vs $91.74 at
  cutover yesterday)
- 2 closed positions during the window (10 → 12)
- 6 fills earlier check, more after; not enough to verdict Stage-1
- Position at stop: −0.173 short (preserved on exchange)

### iter002 launch
- 2026-05-13 11:12 UTC → started (pid 342581)
- Inherits −0.173 short from iter001 (exchange-side position).
- Single diff vs iter001: `MM_MIN_OFFSET_BPS=4 → 6`. Everything else
  identical (funding-aware ON, max notional $500, drawdown pct 30%).
- New journal: `mm_MU_24_5-USD_<ts>.jsonl` under
  `/root/MM-funding-aware/data/mm_journal/`.
- 48h target: 2026-05-15 ~11:12 UTC.

### Two intertwined experiments now
The MU iter001 window was the funding-aware Stage-1 test. iter002
keeps funding-aware ON and changes only `MM_MIN_OFFSET_BPS`. So:
- The funding-aware overlay continues to be tested (versus
  pre-Stage-1 production baseline at `.env.mu_24_5`).
- The widening is tested against iter001 — same funding-aware code,
  different min-offset.

Cleanest A/B is iter002 vs iter001 on the min_offset effect. If we
later want to isolate funding-aware on its own, that would require a
separate iter003 with overlay OFF.

### Rollback
- Stop iter002: `cd /root/MM-funding-aware && PYTHONPATH=... PATH=... mmctl stop mu_24_5.iter002`
- Resume iter001: `mmctl start mu_24_5.iter001`
- Or revert to production: `cd /root/MM && mmctl start mu_24_5`
