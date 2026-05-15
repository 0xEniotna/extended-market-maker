# Iter Decision Doc — DOT-USD iter003 (OFI skew in quoting)

**Date launched**: TBD (after iter002 baked in; target ~2026-06-01)
**Author**: Antoine
**Iter file**: `/root/MM/.env.DOT-USD.iter003`
**Baseline**: `/root/MM/.env.DOT-USD` (post-iter002 baseline with microprice ON)
**Run ID** (filled at launch): TBD
**Parent plan**: `docs/microprice_ofi_plan.md` Phase 4 / A2.2

---

## 1. Hypothesis (one sentence, falsifiable)

Adding a calibrated OFI-driven skew term to `signal_offset_f` will reduce mean +5s post-fill adverse markout by an additional ≥ 0.3 bps on DOT-USD resting fills (vs post-microprice baseline), conditional on the OFI mean-reversion premise (Phase 2 diagnostic) being confirmed on DOT.

---

## 2. Config diff vs baseline

| Knob | Baseline (post-iter002) | This iter (iter003) |
|---|---|---|
| `MM_USE_MICROPRICE` | `true` (inherited) | `true` |
| `MM_OFI_SKEW_ENABLED` | `false` (default) | `true` |
| `MM_OFI_SKEW_K` | (n/a) | loaded from `data/ofi_calibration/DOT-USD.json` |
| `MM_OFI_CAP_BPS` | `3` (default) | `3` (initial; revisit after calibration) |

All other knobs **identical** to post-iter002 baseline. Will verify via `journal_config_history.py` after launch.

Code change: `src/market_maker/ofi_signal.py` (new) + `src/market_maker/pricing_engine.py` (~5 lines additive). All on branch `microprice-ofi`. Worktree path: `/root/MM-microprice-ofi/`.

---

## 3. Pre-registered success criteria

The iter is a **success** if **all** of the following hold over the test window:

1. **Primary metric — incremental markout improvement**:
   - Mean +5s post-fill markout improves by ≥ 0.3 bps vs post-iter002 baseline. Measured per-side and pooled.
   - This is **incremental** on top of microprice. Total cumulative improvement vs iter001 should be ≥ 0.8 bps (microprice +0.5 + OFI +0.3).

2. **Secondary metric — fill rate preserved**:
   - Fills per hour ≥ 0.8 × post-iter002 baseline fills per hour.
   - OFI skew widens one side; a small reduction in fill rate is acceptable if markout improves more than proportionally.

3. **Active fraction (sanity)**:
   - `ofi_skew_bps` is non-zero on ≥ 20% of `order_placed` events (it shouldn't be perpetually flat). Computed from `order_placed.ofi_skew_bps`.
   - If it's flat the whole time, calibration is mis-set.

4. **Safety metric**:
   - No `drawdown_stop`.
   - No `circuit_breaker`.
   - Quote latency p95 ≤ 1.5× baseline.

---

## 4. Sample size required

- Expected fills per hour under this config: similar to iter002 (~0.25-0.5).
- Sample size needed for **primary metric** at α=0.05, β=0.20 with effect size 0.3 bps and σ_markout ≈ 2 bps:
  - N ≈ 175 per side → ≥ 350 total fills.
  - At ~0.3 fills/h → ~1200 hours ≈ **50 days**. Way too long.
  - **Pragmatic compromise**: 14-day window. Effect must be ≥ 0.5 bps (relaxed from 0.3) to be detected; else inconclusive.
- **Minimum wall-clock duration to commit**: **14 days**, with day-7 interim check.

If at day 7 the directional sign of the effect is unambiguous AND |effect| > 1.0 bps, may bake in early.

Note: OFI's marginal improvement on top of microprice is harder to detect than microprice's marginal improvement on top of mid. May need to accept inconclusive and re-test on a higher-fill-rate market (HOOD 24/5 once it has journal).

---

## 5. Rollback trigger (immediate stop)

The iter is **aborted and rolled back** within the test window if **any**:

- `drawdown_stop` fires
- Cumulative realized PnL < −2% of `MM_MAX_POSITION_NOTIONAL_USD`
- Quote latency p95 > 2× baseline
- `ofi_skew_bps` saturates at cap on > 50% of events (indicates calibration off, or extreme regime)
- One-sided fill bias sustained for ≥ 4 hours

Specifically here:
- At day 4 with ≥ 30 fills: if mean +5s markout materially worse than baseline (Δ < −0.5 bps at p < 0.10) → rollback.

Rollback procedure:
```bash
ssh mm-bot 'cd /root/MM-microprice-ofi && PATH=/root/MM/.venv/bin:$PATH \
  PYTHONPATH=/root/MM-microprice-ofi/src mmctl stop DOT-USD.iter003'
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl start DOT-USD'
```

(Rolls back to post-iter002 baseline — microprice stays ON, OFI turns OFF.)

---

## 6. Comparison baseline

- **Baseline window**: the post-iter002 journal slice (microprice baked in) ≥ 7 days preceding iter003 launch.
- **Iter window**: starts at iter003 launch UTC, ends at ≥ 14 days OR rollback.

Required cross-checks:
- `journal_config_history.py` confirms config stable in both windows (microprice ON throughout baseline; microprice ON + OFI ON throughout iter).
- DOT direction drift comparison.
- Spread band ≥ 12 bps maintained.
- OFI calibration `k_ofi` matches the value emitted in `run_start` (i.e., the bot loaded the right calibration file).

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
- OFI-skew-active fraction: TBD %

### Primary metric measured
- Mean +5s markout: TBD bps (target: ≥ baseline +0.3 bps; relaxed to +0.5 with 14d window)
- Cumulative vs iter001: TBD bps (target: +0.8 bps)
- ✅ / ❌ / ⚠️

### Secondary metric measured
- Fills per hour: TBD (target: ≥ 0.8× baseline)
- ✅ / ❌ / ⚠️

### Active fraction
- `ofi_skew_bps` non-zero on TBD % of events (target: ≥ 20%)
- ✅ / ❌

### Safety
- Incidents: TBD

### Market context
- DOT direction drift this iter: TBD
- DOT direction drift baseline: TBD
- DOT spread band: TBD bps mean

### Decision

**[KEEP / ROLLBACK / EXTEND / INCONCLUSIVE]**

Justification: TBD.

If KEEP: promote iter003 to `.env.DOT-USD` baseline (snapshot `.env.DOT-USD.pre_ofi.YYYYMMDD`). Document in `docs/fleet_status_log.md`. Then proceed to Phase 5 (roll to NEAR/XNG).

If INCONCLUSIVE: extend to 21d OR accept and re-test on higher-fill-rate market (HOOD 24/5).

---

## 8. Links

- Parent plan: `docs/microprice_ofi_plan.md`
- Phase 2 diagnostic: `docs/stage4_ofi_diagnostic.md`
- Decision gate: `docs/stage3_4_gate_decision.md`
- Replay verification: `docs/stage4_replay_DOT_verdict.md`
- Calibration: `data/ofi_calibration/DOT-USD.json`
- Config history: `docs/config_history_DOT-USD.md`
- Previous iter (microprice bake-in): `docs/iter_decisions/YYYY-MM-DD_DOT-USD_microprice_iter002.md`
- Journals analyzed:
  - Baseline: TBD
  - Iter: TBD
