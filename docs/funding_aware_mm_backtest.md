# Funding-Aware Counterfactual Backtest — Phase 4 report

**Date**: 2026-05-12
**Branch**: `funding-aware-mm`
**Commit**: `181d8a0`
**Harness**: `scripts/backtest_funding_aware_counterfactual.py`
**Inputs**:
- Per-market live journals from `/root/MM/data/mm_journal/` (mm-bot VPS).
- 7-day funding-rate history pulled from Extended via
  `scripts/download_funding_history.py` (168 hourly entries per market).
- Policy: `coupling_bps_max=8`, `hold_horizon_periods=4`,
  `dollar_cap_pct_of_notional=0.001`.
- Tick size: `1e-8` (forced tight so quantization noise doesn't mask the
  intended signal — the real production tick is asymmetric and identical
  on both engines, so it cancels in the perturbation difference).

This report covers Gate 3 of the rollout plan: a **counterfactual** quote
comparison of the funding-aware overlay vs the current baseline
(`funding_aware=None`). No orders are placed, no fills are simulated,
no live state is touched.

---

## TL;DR — Phase 4 gate verdict: **PASS**

| Market         | Events    | abs_max bps | p99 bps  | Pearson(\|F\|, \|perturb\|) | Cap violations |
|----------------|-----------|-------------|----------|-----------------------------|----------------|
| ETH-USD        | 49,011    | 0.880       | +0.800   | **+0.9923**                 | **0**          |
| DOT-USD (prior)| 63,592    | 4.004       | +3.999   | **+1.0000**                 | **0**          |
| DOT-USD (latest, 2.15h)| 4,216 | 0.521  | -0.519   | +0.0000 †                   | **0**          |
| SPX500m-USD    | 49,282    | 6.009       | +6.005   | **+0.9981**                 | **0**          |
| MU_24_5-USD    | 29,930    | 3.246       | +3.244   | **+1.0000**                 | **0**          |

† Pearson denominator collapses because the DOT-latest journal only
covers 2.15h and the funding rate was a single value (`0.000013`) for
that entire window. With the previous DOT journal (full week), Pearson
recovers to +1.0000 as expected. Not a defect.

All three Gate-3 conditions hold for every market with measurable \|F\|
variance:

1. **No exceptions / no NaN / no Inf** — 0 events skipped on bad data
   across 196,031 replayed quotes.
2. **Monotone in \|F\|** — Pearson(\|F\|, \|perturb\|) ≥ +0.9923 on
   ETH/DOT-prior/SPX500m/MU. Inconclusive for DOT-latest by construction.
3. **\|perturbation\| ≤ coupling_bps_max=8 bps** — 0 sanity violations
   across all 196,031 events.

No claim about fill quality, PnL, or quote latency is made here. Those
belong to Phase 5 (paper-trade) and Phase 6 (live).

---

## Per-market reports

Per-market detail tables (overall distribution, by side, by regime, by
funding sign) live in:

- [`funding_aware_mm_backtest_ETH.md`](funding_aware_mm_backtest_ETH.md)
- [`funding_aware_mm_backtest_DOT_prior.md`](funding_aware_mm_backtest_DOT_prior.md)
- [`funding_aware_mm_backtest_DOT.md`](funding_aware_mm_backtest_DOT.md) (latest, 2.15h slice)
- [`funding_aware_mm_backtest_SPX500m.md`](funding_aware_mm_backtest_SPX500m.md)
- [`funding_aware_mm_backtest_MU_24_5.md`](funding_aware_mm_backtest_MU_24_5.md)

---

## Cap-binding analysis (calibration sanity)

The overlay has two saturation knobs:

1. `coupling_bps_max` — hard cap on signed bps (8 bps default).
2. `dollar_cap_pct_of_notional` — clamp on funding_dollar before bps
   conversion: implied bps cap = `pct × max_position_size × 1e4` (note
   it does *not* depend on mid).

Implied dollar-cap (in bps) per market:

| Market | `MM_MAX_POSITION_SIZE` | dollar_cap implied bps | bps cap | Binding |
|---|---|---|---|---|
| ETH-USD     |    5    |   50 | 8 | bps cap (loose; observed signal ≪ 8) |
| DOT-USD     | 4500    | 45000 | 8 | bps cap (loose; observed signal ≪ 8) |
| SPX500m-USD |   0.6   |    6 | 8 | **dollar cap** (observed max=6.009) |
| MU_24_5-USD |   6.0   |   60 | 8 | bps cap (loose; observed signal ≪ 8) |

For SPX500m the dollar cap is the active constraint (6 bps), and the
observed abs_max=6.009 matches that exactly. Whether that's tight enough
for SPX is a Phase-5 calibration question. The current default keeps the
overlay strictly inside the operator-facing 8-bps envelope on every
market, which is what we want for the first live experiment.

---

## Quote-shift sign convention (sanity)

The overlay subtracts `signal_offset_f` (proportional to `F`) from both
`raw_f` paths (BUY and SELL) in `PricingEngine.compute_target_price`. So
when `F > 0`:

- BUY raw price → **down** (less aggressive long; we don't want to grow
  long while paying funding).
- SELL raw price → **down** (more aggressive sell; we want to shed
  long).

Both quotes shift in the same direction by the same magnitude → the
spread is unchanged, the reservation price moves. The asymmetric
inventory term is already produced by the existing `_skew_component_f`;
adding `sign(q)` here would double-count (Codex P1-2 fix).

The per-market `By funding sign` tables confirm this: rows with `F>0`
show negative mean perturb_bps, rows with `F<0` show positive mean
perturb_bps. The magnitudes are tiny compared to the existing
`min_offset_bps` (typically 3-4 bps) and `max_offset_bps` (90 bps).

---

## What this report does NOT establish

- It does **not** prove the overlay improves PnL. The harness only
  compares quote prices — it does not re-simulate fills, queue
  position, or adverse selection. The paper's empirical uplift on
  ETH/BTC (1-2%) requires a fill simulator we deliberately chose not
  to build (Codex P1-6 fix).
- It does **not** prove the overlay is safe at the boundary of the
  inventory critical/hard pcts. Those tests live in
  `tests/test_funding_aware_integration.py` (Gate 2).
- It does **not** establish whether `dollar_cap_pct_of_notional=0.001`
  is the right number for SPX500m. The current default merely keeps
  the overlay inside the 8-bps hard cap on every market.

---

## Next step — Phase 5 paper-trade

With Gate 3 passing, the recommended Phase 5 plan is:

1. Pick **one** market with the highest funding-to-spread ratio in the
   per-market reports. Candidate from these results: **MU_24_5-USD**
   (max funding 0.0998% per period, ~3.2 bps overlay perturbation, well
   inside cap, Pearson +1.0).
2. Restart that market on the VPS with `MM_FUNDING_AWARE_ENABLED=true`
   in an iter-suffixed env copy (e.g. `.env.mu_24_5.iter001`). Do NOT
   modify `.env.mu_24_5` directly.
3. Use a small `MM_MAX_POSITION_NOTIONAL_USD` cap (≤ $500) for the
   first 48h.
4. Compare realized PnL, post-only failure rate, and quote latency
   against the immediately-preceding 48h baseline on the same env file.
5. Decision: green-light Phase 6 (live cutover with operator approval)
   or rollback (flag off + restart, Gate-2-verified path).

**Phase 5 requires explicit operator green light to start.** No
`.env.*` files have been edited; no live process has been touched.
