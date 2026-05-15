# Iter Decision Doc — DOT-USD iter002 (microprice as fair value)

**Date launched**: TBD (after Phase 3 code + replay complete; target ~2026-05-20)
**Author**: Antoine
**Iter file**: `/root/MM/.env.DOT-USD.iter002`
**Baseline**: `/root/MM/.env.DOT-USD` (iter001 production, +$91/wk over 1 week)
**Run ID** (filled at launch): TBD
**Parent plan**: `docs/microprice_ofi_plan.md` Phase 3 / A1.2

---

## 1. Hypothesis (one sentence, falsifiable)

Replacing the mid-price reference in `compute_target_price` with the depth-weighted microprice will reduce mean +5s post-fill adverse markout by ≥ 0.5 bps on DOT-USD resting fills (vs iter001 baseline), without cutting fill rate per hour by more than 20%.

---

## 2. Config diff vs baseline

| Knob | Baseline (iter001) | This iter (iter002) |
|---|---|---|
| `MM_USE_MICROPRICE` | `false` (default) | `true` |

All other knobs **identical** to baseline. Will verify via `journal_config_history.py` after launch (post-mortem).

Code change: `src/market_maker/microprice.py` (new) + `src/market_maker/pricing_engine.py` (~10 lines). All on branch `microprice-ofi`. Worktree path: `/root/MM-funding-aware/ (after `git checkout microprice-ofi` in that worktree)` (to be created on VPS).

---

## 3. Pre-registered success criteria

The iter is a **success** if **all** of the following hold over the test window:

1. **Primary metric — markout improvement**:
   - Mean +5s post-fill markout improves by ≥ 0.5 bps vs baseline (e.g., baseline −0.8 bps → iter ≥ −0.3 bps). Measured per-side and pooled.
   - Computed via `scripts/diagnose_markout.py --journal <iter002_journal>` and compared to a baseline pull over the matched preceding window.

2. **Secondary metric — fill rate preserved**:
   - Fills per hour ≥ 0.8 × baseline fills per hour (DOT baseline ~5-10 fills/day → accept ≥ 4-8/day).

3. **Safety metric**:
   - No `drawdown_stop` triggered.
   - No `circuit_breaker` open.
   - Quote latency p95 ≤ 1.5× baseline.

---

## 4. Sample size required

- Expected fills per hour under this config: ~0.25-0.5 (DOT averages 5-10 resting fills/day).
- Sample size needed for **primary metric** at α=0.05, β=0.20 with effect size 0.5 bps and per-side σ_markout ≈ 2 bps (from `stage2_markout_DOT-USD.md`):
  - N ≈ 60 per side → ≥ 120 total fills.
  - At ~0.3 fills/h → ≥ 400 hours ≈ **17 days**.
  - At ~0.5 fills/h → ~10 days.
- **Minimum wall-clock duration to commit**: **10 days** (with mid-window interim check at day 6).

If at day 6 the directional sign of the effect is unambiguous AND |effect| > 1.0 bps, may bake in early. If null at day 10, extend to day 14.

---

## 5. Rollback trigger (immediate stop)

The iter is **aborted and rolled back** within the test window if **any**:

- `drawdown_stop` fires
- Cumulative realized PnL < −2% of `MM_MAX_POSITION_NOTIONAL_USD` (= −$200 on $10k notional)
- Quote latency p95 > 2× baseline
- More than 10 errors of a single type in the log
- One-sided fill bias (BUY/SELL ratio < 0.3 or > 3.0) sustained for ≥ 4 hours

Specifically here:
- At day 3 with ≥ 30 fills accumulated: if mean +5s markout is materially **worse** than baseline (Δ < −0.5 bps at p < 0.10) → rollback.

Rollback procedure:
```bash
ssh mm-bot 'cd /root/MM-funding-aware && PATH=/root/MM/.venv/bin:$PATH \
  PYTHONPATH=/root/MM-funding-aware/ (after `git checkout microprice-ofi` in that worktree)src mmctl stop DOT-USD.iter002'
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl start DOT-USD'
```

---

## 6. Comparison baseline

- **Baseline window**: the 7-day journal slice immediately preceding iter002 launch on DOT-USD production. Pinned via `journal_config_history.py` (must show iter001 config stable across the slice).
- **Iter window**: starts at iter002 launch UTC, ends at ≥ 10 days OR rollback, whichever first.

Required cross-checks before drawing any conclusion:
- `journal_config_history.py` confirms config was stable in both windows.
- Direction drift: note DOT close-to-close move in both windows. If |Δbaseline − Δiter| > 5% on a directional metric, the comparison is weak → extend or accept inconclusive.
- Spread band: confirm DOT natural spread stayed ≥ 12 bps in both windows (per `project_learnings.md` profitability rule).

---

## 7. Post-mortem (filled after test ends)

### Actual window
- Started: TBD
- Stopped: TBD
- Duration: TBD
- Rollback triggered: TBD

### Actual fills
- Total resting fills: TBD
- Fills per hour: TBD
- Side balance: BUY TBD / SELL TBD

### Primary metric measured
- Mean +5s markout: TBD bps (target: ≥ baseline +0.5 bps)
- ✅ MET / ❌ NOT MET / ⚠️ INCONCLUSIVE

### Secondary metric measured
- Fills per hour: TBD (target: ≥ 0.8× baseline)
- ✅ / ❌ / ⚠️

### Safety
- Incidents: TBD

### Market context
- DOT direction drift this iter: TBD
- DOT direction drift baseline: TBD

### Decision

**[KEEP / ROLLBACK / EXTEND / INCONCLUSIVE]**

Justification: TBD.

If KEEP: promote iter002 config to `.env.DOT-USD` baseline (with snapshot of pre-microprice baseline at `.env.DOT-USD.pre_microprice.YYYYMMDD`). Document promotion in `docs/fleet_status_log.md`.

If INCONCLUSIVE: next-step plan (extend to 14d / abandon).

---

## 8. Links

- Parent plan: `docs/microprice_ofi_plan.md`
- Phase 1 diagnostic: `docs/stage3_microprice_diagnostic.md`
- Decision gate: `docs/stage3_4_gate_decision.md`
- Replay verification: `docs/stage3_replay_DOT_verdict.md` (after Phase 3 replay)
- Config history: `docs/config_history_DOT-USD.md`
- Markout baseline: `docs/stage2_markout_DOT-USD.md`
- Markout for this iter: TBD after test
- Journals analyzed:
  - Baseline: TBD
  - Iter: TBD
