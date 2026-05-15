# Iter Decision Doc — \<MARKET\> iter\<NNN\>

**Date launched**: YYYY-MM-DD HH:MM UTC
**Author**: \<who\>
**Iter file**: `/root/MM/.env.\<market\>.iter\<NNN\>`
**Baseline**: `\<market.iter(NNN-1) or "production .env.\<m\>"\>`
**Run ID** (filled at launch): \<from journal `run_start`, 8 char short hash\>

---

## 1. Hypothesis (one sentence, falsifiable)

> Example: "Bumping `MM_MIN_OFFSET_BPS` from 4 to 6 on MU will reduce
> the proportion of fills in the toxic <5bps edge bucket from ~23% to
> <10% without cutting fill rate by more than 50%."

---

## 2. Config diff vs baseline

| Knob | Baseline | This iter |
|---|---|---|
| `MM_FOO_KNOB` | `old_value` | `new_value` |
| ... | ... | ... |

All other knobs **identical** to baseline (verified via
`journal_config_history.py` after launch — see post-mortem).

---

## 3. Pre-registered success criteria

The iter is a **success** if **all** of the following hold over the
test window:

1. **Primary metric** (specific number, not vague):
   - Example: "Mean +5s post-fill markout ≥ −1.0 bps on resting fills
     (vs baseline −2.4 bps)."
   - Example: "% fills in tight edge bucket (<5 bps) ≤ 10% (vs 23%
     baseline)."

2. **Secondary metric** (defended against the obvious bad-side-effect):
   - Example: "Fill rate per hour ≥ 0.9 (vs baseline 1.8 — accepts
     50% drop but not more)."

3. **Safety metric** (no incidents):
   - No `drawdown_stop` triggered.
   - No `circuit_breaker` open.
   - Quote latency p95 ≤ 1.5× baseline.

---

## 4. Sample size required

- Expected fills per hour under this config: \<estimate\>
- Sample size needed for **primary metric** at α=0.05, β=0.20:
  \<computed; if unsure, default to N=30 per side\>
- → Minimum test duration: \<sample / fill-rate\> hours
- **Minimum wall-clock duration to commit**: \<at least the duration
  above; can extend if conditions are unusual\>

---

## 5. Rollback trigger (immediate stop)

The iter is **aborted and rolled back** within the test window if **any**:

- `drawdown_stop` fires
- Cumulative realized PnL < −X% of `MM_MAX_POSITION_NOTIONAL_USD`
- Quote latency p95 > 2× baseline
- More than \<threshold\> errors of a single type in the log
- Position grows unbounded (e.g., one-sided fills with no offset)

Specifically here:
- \<exact rollback trigger for this iter\>

Rollback procedure:
```bash
ssh mm-bot 'cd /root/MM-funding-aware && PATH=/root/MM/.venv/bin:$PATH \
  PYTHONPATH=/root/MM-funding-aware/src mmctl stop <market>.iter<NNN>'
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl start <market>'
```

---

## 6. Comparison baseline

How will the iter be compared to the baseline? Be specific to avoid
post-hoc cherry-picking of windows.

- Baseline window: \<exact journal slice, e.g.,
  `mm_<m>_20260510_140543.jsonl` events from t1 to t2\>
- Iter window: starts at iter launch UTC, ends after \<duration above\>
  or rollback, whichever first.

Required cross-check before drawing any conclusion:
- Run `journal_config_history.py` to confirm config was stable during
  both windows.
- Note market direction drift in both windows (e.g., "baseline: −5%,
  iter: +3%"). If direction differs significantly, the conclusion is
  weak.

---

## 7. Post-mortem (filled after test ends)

### Actual window
- Started: \<UTC\>
- Stopped: \<UTC\>
- Duration: \<h\>
- Rollback triggered: \<yes/no, reason\>

### Actual fills
- Total resting fills: \<n\>
- Fills per hour: \<n/h\>
- Side balance: BUY \<n\> / SELL \<n\>

### Primary metric measured
- \<metric\>: \<value\> (target was \<target\>)
- ✅ MET / ❌ NOT MET / ⚠️ INCONCLUSIVE (sample too small)

### Secondary metric measured
- \<metric\>: \<value\> (target was \<target\>)
- ✅ / ❌ / ⚠️

### Safety
- Any incidents? \<yes/no, details\>

### Market context
- Direction drift during iter window: \<%, e.g., MU went from $800 to $750 = −6%\>
- Compared to baseline window direction: \<...\>

### Decision

**[KEEP / ROLLBACK / EXTEND / INCONCLUSIVE]**

Justification (one paragraph): \<why\>

If KEEP: promote iter config to `.env.<market>` baseline? \<yes/no\> If
yes, document the promotion in `docs/fleet_status_log.md` and snapshot
the previous baseline.

If INCONCLUSIVE: next-step plan (e.g., extend window, try larger size,
abandon).

---

## 8. Links

- Config history: `docs/config_history_<MARKET>.md`
- Markout diagnostic baseline: `docs/stage2_markout_<MARKET>_pooled.md`
- Markout diagnostic for this iter: \<path after test\>
- Journals analyzed:
  - Baseline: \<paths\>
  - Iter: \<paths\>
