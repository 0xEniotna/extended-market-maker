# Iter Decision Doc — ETH-USD iter001 (markout-feedback overlay)

**Date drafted**: 2026-05-15 (pre-build; the code does not yet exist)
**Date launched**: TBD (after Phase 2 build + Phase 3 replay gates)
**Author**: AM + Claude
**Iter file**: `/root/MM/.env.eth.iter001` (to be created from `.env.eth` baseline)
**Baseline**: production `.env.eth` (killed 2026-05-13 21:42 UTC)
**Run ID**: TBD (filled at launch from journal `run_start`)

---

## 1. Hypothesis (one sentence, falsifiable)

> Enabling a **per-side EWMA markout-feedback overlay** on ETH will
> reduce the mean +5s post-fill markout from the baseline −2.43 bps
> toward −1.0 bps or better, by widening the side that is actively
> bleeding while keeping the other side at baseline offset.

**Why this hypothesis is plausible**: the Phase 1 calibration (Wednesday,
n=1,751 fills across 4 ETH journals) showed:
- Per-side lag-1 autocorrelation of markout: BUY +0.466, SELL +0.430
- Pearson(prior-5 mean, next markout): BUY +0.18, SELL +0.27

These say the bleed is **temporally clustered**, not independent draws.
A reactive feedback policy can target the clusters. The 108-combo
parameter sweep showed all combinations produce `markout_diff(active −
inactive) < 0`, meaning the policy fires during measurably worse
windows.

**Why this hypothesis might fail**: the policy reacts at the granularity
of fills. The first bad fill in a streak is never prevented (we only
have data after it). On a thin tape, an isolated bad fill won't trigger
the widening. And the widening protects placement, not orders already
resting (the "BBO drift" issue we hit on MU iter002).

---

## 2. Config diff vs baseline

The baseline (`.env.eth`) has the standard MM knobs. This iter adds the
4 new envs **plus** uses a smaller notional cap for safety during the
test.

| Knob | Baseline `.env.eth` | iter001 |
|---|---|---|
| `MM_MARKOUT_FEEDBACK_ENABLED` | false (default) | **true** |
| `MM_MARKOUT_FEEDBACK_HALF_LIFE_S` | — | **30** |
| `MM_MARKOUT_FEEDBACK_THRESHOLD_BPS` | — | **2.0** |
| `MM_MARKOUT_FEEDBACK_GAIN` | — | **0.5** |
| `MM_MARKOUT_FEEDBACK_CAP_BPS` | — | **5.0** |
| `MM_MARKOUT_FEEDBACK_HORIZON_S` | — | **5** |
| `MM_MAX_POSITION_NOTIONAL_USD` | $\<production value\> | **$1,000** (cut for safety) |
| `MM_DRAWDOWN_STOP_PCT_OF_MAX_NOTIONAL` | $\<production value\> | **30** (preserves $300 abs threshold at the cut notional) |

**All other knobs identical to baseline** — to be verified post-launch
via `journal_config_history.py` (cross-check is a Rule 4 requirement).

Param choice rationale (from Phase 1 calibration sweep):

```
| half_life | threshold | gain | cap | %active | diff (act−inact) |
|---|---|---|---|---|---|
| 30s        | 2.0       | 0.5  | 5   | 40.7%   | -1.71 bps  ← chosen
| 30s        | 1.0       | 0.5  | 5   | 45.6%   | -1.48
| 60s        | 0.5       | 0.5  | 5   | 53.9%   | -1.43
```

The chosen combo targets the active window most precisely (most-negative
diff) with the smallest %active footprint (40.7%, lower than the
alternatives that take 45-50% of fill opportunities). Higher gains don't
improve targeting (the active/inactive partition is binary), so 0.5
suffices.

---

## 3. Pre-registered success criteria

The iter is a **success** if **all** of the following hold over the test
window (excluding the first 30 minutes of warmup):

### Primary metric — markout improvement
- **Mean +5s post-fill markout on resting fills ≥ −1.0 bps**
- Baseline reference: −2.43 bps on n=1,538 pooled fills
- Target = 60% reduction in bleed
- Measured via `scripts/diagnose_markout.py` on the iter journal

### Secondary metric — fill rate preserved
- **Fill rate per hour ≥ 0.5 × baseline**
- Baseline fill rate: ~6 fills/hour during ETH's high-volume periods
  (220 fills/day observed on 2026-05-13)
- Target threshold = 3 fills/hour during similar conditions
- Accepts up to 50% volume drop (the policy is expected to skip toxic
  windows; some volume loss is the *point*)

### Safety metric — no incidents
- No `drawdown_stop` fires
- No `circuit_breaker` open during iter
- Quote latency p95 ≤ 1.5× baseline (baseline ~500ms → max ~750ms)
- Position never exceeds 90% of `MM_MAX_POSITION_NOTIONAL_USD`
- No more than 5 `ERROR`/`CRITICAL` log lines per hour (excluding
  routine `risk_sizing` clips)

---

## 4. Sample size required

ETH baseline activity (Stage 2 diagnostic): ~150 fills/day during high
volume.

**Sample for primary metric** (mean +5s markout, two-sample t-test):
- Effect size: |Δμ| = |−2.43 − (−1.0)| = 1.43 bps
- Std dev (baseline): σ ≈ 4 bps
- Significance: α = 0.05, power: β = 0.20
- n per group ≈ (σ / Δμ)² × (z_α/2 + z_β)² × 2 ≈ (4/1.43)² × (1.96+0.84)² × 2 ≈ **60 fills**

**Minimum test duration**: 60 fills ÷ (150 × 0.5 / 24) fills/hour ≈
**20 wall-clock hours minimum** (accepting that 50% notional cut roughly
halves fill rate vs original ETH).

**Committed minimum**: **48h test window** (gives n ~ 150 fills,
2.5× the minimum, robust to slow trading periods).

---

## 5. Rollback trigger (immediate stop)

The iter is **aborted and rolled back** within the test window if **any**:

- `drawdown_stop` fires (the bot will do this automatically)
- Cumulative realized PnL on this iter < **−$50 (= −5% of $1,000 notional)** in any 6h sliding window
- Quote latency p95 > **2× baseline** sustained for 10 minutes
- Mean +5s markout on **first 50 fills** is **WORSE** than baseline
  (< −3.0 bps) — i.e., the overlay actively hurts → kill before more
  damage
- Position size hits 95% of `MM_MAX_POSITION_NOTIONAL_USD` and stays
  there for >30 minutes (one-sided fills, no offset)
- More than 30 `ERROR` log lines in any 1-hour window (excl. routine clips)

**Rollback procedure**:
```bash
# Stop the iter (from worktree where overlay code lives)
ssh mm-bot 'cd /root/MM-funding-aware && PATH=/root/MM/.venv/bin:$PATH \
  PYTHONPATH=/root/MM-funding-aware/src mmctl stop eth.iter001'

# ETH baseline is currently killed; we do NOT auto-relaunch.
# Decision on relaunching baseline is a separate discussion.
```

The shutdown flatten will close any open position. Expected slippage on
ETH: ~20 bps (consistent with prior shutdowns observed in this session).

---

## 6. Comparison baseline

**Baseline window** (for markout comparison):
- Journals: pooled 4 ETH May journals used in Stage 2 diagnostic
  - `mm_ETH-USD_20260505_171314.jsonl`
  - `mm_ETH-USD_20260506_102154.jsonl`
  - `mm_ETH-USD_20260506_102154.1.jsonl`
  - `mm_ETH-USD_20260506_102154.2.jsonl`
- Stats: n=1,538 resting fills, mean +5s markout −2.43 bps
- Cross-checked via `journal_config_history.py` to confirm config was
  stable (no funding-aware overlay, default min_offset, default notional)

**Iter window**:
- Starts at iter launch UTC
- Ends after 48h or rollback trigger, whichever first
- Excludes first 30 min of warmup (stream-desyncs, position-build phase)

**Cross-checks required before drawing conclusions**:
1. Run `python scripts/journal_config_history.py --market ETH-USD` to
   confirm iter config was stable for the full window (no SIGHUP drift).
2. Note market direction during iter window. Baseline coverage included
   a ~−1% drift (ETH range ~$2250-$2340). If iter window has materially
   different drift (>3% in any direction), note it in the post-mortem
   and treat the conclusion as weak.
3. Verify funding rate magnitude during iter window is comparable to
   baseline (ETH funding ~ ±3 bps/period). If unusual funding regime,
   adjust expectations.

---

## 7. Pre-launch gates (must pass before iter goes live)

This decision doc is filed BEFORE the code exists. The build sequence:

### Phase 2 — implementation (estimated 8-12h)
- `src/market_maker/markout_feedback.py` (~120 LOC)
  - Config dataclass + policy class
  - Per-side EWMA with horizon-delayed fill ingestion
  - Plug point in `pricing_engine.compute_target_price`
- Settings in `config.py` + `config_metadata.py` (new "markout_feedback"
  group)
- `tests/test_markout_feedback.py` (~250 LOC):
  - Unit: disabled-returns-zero, EWMA decay math, threshold gating
  - Property: widening non-negative, magnitude monotone in |EWMA|
  - Integration: byte-identical quotes when flag off (rollback safety)

### Phase 3 — counterfactual replay (estimated 2-3h)
- Re-run the calibration script with the **actual implementation** (not
  the calibration sim) on the same 4 ETH journals.
- Verify the policy fires on the same fill clusters the calibration
  identified.
- Verify perturbation is bounded (≤ `cap_bps`).

### Phase 4 — Gate 1 + Gate 2 (existing structure)
- Ruff clean on all touched files
- All existing tests still pass
- Rollback test (Gate 2): flag-off byte-identical to baseline engine

### Phase 5 — paper trade (this iter)
- Pre-condition: all gates above passed; the decision doc is referenced
  by the launch.
- This decision doc fixes the criteria.

**Without all gates passing, this iter does NOT launch.**

---

## 8. Post-mortem (filled after test ends)

### Actual window
- Started: \<UTC timestamp\>
- Stopped: \<UTC timestamp\>
- Duration: \<hours\>
- Rollback triggered: \<yes/no, reason\>

### Actual fills
- Total resting fills: \<n\>
- Fills per hour: \<n/h\>
- Side balance: BUY \<n\> / SELL \<n\>

### Primary metric measured
- Mean +5s markout (resting fills): \<value\> bps
- Target: ≥ −1.0 bps
- Result: ✅ MET / ❌ NOT MET / ⚠️ INCONCLUSIVE

### Secondary metric measured
- Fill rate per hour: \<value\>
- Target: ≥ 3.0/h
- Result: ✅ / ❌ / ⚠️

### Safety
- Any incidents? \<yes/no, details\>
- Max position observed: \<value\>
- Realized PnL during iter: \<value\>

### Market context
- ETH price drift during iter window: \<%\>
- Compared to baseline window drift: \<comparison\>
- Funding rate during iter: \<range\>
- Compared to baseline funding: \<comparison\>

### Policy activity
- % fills with `widen_bps > 1`: \<value\> (target: ~40% from calibration)
- Mean widening when active: \<value\> (target: ~3.5 bps)
- Markout during active periods: \<value\>
- Markout during inactive periods: \<value\>
- Diff (active − inactive): \<value\> (calibration predicted −1.71)

### Decision

**[KEEP / ROLLBACK / EXTEND / INCONCLUSIVE]**

Justification (one paragraph): \<why\>

**If KEEP**: 
- Promote `MM_MARKOUT_FEEDBACK_ENABLED=true` and the 4 params to the
  `.env.eth` baseline.
- Snapshot the prior `.env.eth` as `.env.eth.pre-markout-feedback.YYYYMMDD`.
- Update `docs/fleet_status_log.md`.
- Consider extending the overlay to other markets where markout was
  measured-negative (none currently; re-measure DOT/XNG/MU after more
  fills accumulate).

**If ROLLBACK**:
- Keep the overlay code (it's well-tested and behind a flag).
- Note the failure mode in `docs/stage2_markout_summary.md`.
- The overlay design might be sound but ETH-specific conditions don't
  match. Reconsider for other markets after more diagnostics.

**If INCONCLUSIVE**:
- Likely cause: not enough fills (test window too short or trading
  was unusually slow).
- Action: extend the window by 48h, document the extension.
- If still inconclusive after 96h total: abandon and revisit when ETH
  is showing baseline-like fill rate.

---

## 9. Links

- Phase 1 calibration: `docs/stage2_calibration_ETH.md`
- Stage 2 diagnostic baseline: `docs/stage2_markout_ETH_pooled.md`
- Stage 2 summary: `docs/stage2_markout_summary.md`
- ETH config history (post-launch): `docs/config_history_ETH-USD.md`
- Iter journals (post-launch): `\<paths\>`
