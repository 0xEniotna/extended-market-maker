# Phase 3 Verdict — Markout-Feedback Overlay on ETH

**Date**: 2026-05-15
**Status**: ⚠️ **DO NOT LAUNCH** — replay revealed a model-vs-implementation gap.

This is **exactly the kind of finding Phase 3 is meant to catch**: the
calibration simulator made a simplification that turned out to materially
overstate the policy's effectiveness. We caught it before paper-trading
or live deployment.

---

## What the replay measured

Ran the **actual** `MarkoutFeedbackPolicy` (from `src/market_maker/`)
against 4 ETH journals (1,751 resting fills total) at three parameter
combos:

| Combo | Params | %active | mean widen | markout(act) | markout(inact) | **diff** |
|---|---|---|---|---|---|---|
| recommended | hl=30s, th=2, g=0.5, cap=5 | 26.0% | 3.45 bps | -2.557 | -2.278 | **-0.28** |
| aggressive | hl=30s, th=1, g=1, cap=10 | 34.6% | 6.12 bps | -2.441 | -2.303 | **-0.14** |
| conservative | hl=60s, th=2, g=0.5, cap=5 | 32.6% | 3.81 bps | -2.494 | -2.281 | **-0.21** |

**Compare to Phase 1 calibration** (same recommended combo): diff = **-1.71 bps**.

The real implementation produces a diff **6× smaller** than the
calibration sim predicted.

---

## Why the discrepancy

The Phase 1 calibration script
(`scripts/calibrate_markout_feedback.py`) walks the fill stream and
updates the EWMA **immediately** when each fill is encountered. There's
a comment in the code acknowledging this simplification:

```python
# Update EWMA with this fill's markout, accounting for the
# horizon_s delay (we'd only know markout at ts + horizon_s).
# For calibration purposes treat update as immediate; the delay
# would slightly slow the policy but not change the
# qualitative picture.
```

This turns out to be wrong. The horizon delay does NOT just "slightly
slow the policy" — it materially reduces effectiveness.

In the **real implementation** (`src/market_maker/markout_feedback.py`):
1. `on_fill(ts, side, price)` enqueues the fill at `ts`.
2. `tick(now_ts)` only processes pending fills with `deadline ≤ now_ts`,
   where `deadline = ts + horizon_s`.
3. So fill N's markout is incorporated into the EWMA only **after** at
   least `horizon_s` seconds have elapsed.
4. If fill N+1 happens within `horizon_s` of fill N, the EWMA at the
   time of fill N+1's widening lookup does NOT include fill N's
   contribution.

For ETH with 1,751 fills over ~10 days (~7.3 fills/hour, mean inter-fill
~8 minutes), most fill pairs ARE >5s apart. But the bleed clusters
themselves are tighter — and that's exactly when fills happen close
together. The policy can't react in time within a cluster.

The calibration sim was implicitly assuming "perfect information"; the
real policy operates on stale information during the most important
moments (bleed clusters).

---

## Implication for the iter

The pre-registered success criteria from
`docs/iter_decisions/2026-05-15_ETH-USD_iter001-markout-feedback.md`:

> Primary metric: Mean +5s post-fill markout on resting fills ≥ −1.0 bps
> (vs baseline −2.43 bps). Target = 60% reduction in bleed.

Implied PnL impact:
- The overall replay markout (= weighted average of active −2.56 and
  inactive −2.28) ≈ **−2.35 bps**.
- Versus baseline of −2.43 bps.
- Improvement: **0.08 bps** — essentially noise.

**The primary metric cannot be met by this implementation as-is.**

Estimated PnL impact:
- Baseline: ~$37/day AS bleed on ETH (per Stage 2 diagnostic)
- After overlay: ~$36/day bleed
- **Improvement: ~$1/day** — negligible

---

## Decision

**INCONCLUSIVE → DO NOT LAUNCH.**

Reasons:
1. The implementation does what it's designed to do (the math is right,
   tests pass, no bugs in the policy code itself).
2. But the implementation reflects reality, and reality includes the
   horizon delay that the calibration ignored.
3. The expected PnL improvement is too small to justify even
   paper-trade overhead.
4. ETH is currently killed anyway (per docs/fleet_status_log) — no live
   target.

Reasons NOT to abandon the overlay entirely:
1. The code is well-tested (49 new tests pass), behind a default-off
   flag, byte-identical to baseline when off — zero risk to current
   production.
2. The *qualitative* finding (autocorrelated markouts → reactive
   policy targets bad windows) is real, just smaller than predicted.
3. A future variant with **shorter horizon** (e.g., 1s instead of 5s)
   might recover more of the predicted gain. The shorter horizon
   reduces the lag at the cost of using a noisier markout signal — a
   tradeoff worth measuring.
4. The infrastructure (config, wiring, tests, replay script) is ready
   for any future reactive policy.

---

## What this validates about the process

The new decision-doc workflow is doing its job:

1. **Pre-registered criteria caught this.** Without the criteria, I
   would have probably noticed the smaller diff but rationalized it
   ("still negative, still works"). With the criteria, the result is
   unambiguously "doesn't meet target".

2. **Phase 3 caught it before live deployment.** If we'd launched the
   iter and only checked PnL after 48h, we would have observed "no
   meaningful improvement" but couldn't isolate cause from market
   noise. Phase 3 isolates the implementation vs the model.

3. **The replay script will be reusable** for future overlays — point
   it at the actual policy code, run on journals, compare to the
   prediction.

---

## Next steps (if pursuing this further later)

1. **Re-write the calibration script** to include the horizon delay.
   The current sim is misleading — fix it before any future
   calibration sweep.

2. **Sweep horizons** (1s, 2s, 5s, 10s) to find the optimal point
   between reactivity and signal quality.

3. **Consider alternative formulations**:
   - Use the trade tape (public market trades) rather than just our
     own fills — much higher signal frequency, possibly tighter
     timing.
   - Compute markout against the **fill price at placement time** rather
     than the mid at fill time — different framing of toxicity.

4. **Re-test on a market where we have a live position**. ETH is
   killed; DOT, MU, XNG are the candidates. DOT and XNG show clean
   markouts in the Stage 2 diagnostic — the overlay has nothing to
   correct there. MU's edge-dependent toxicity might be a better
   target.

---

## Artifacts

- Implementation: `src/market_maker/markout_feedback.py` (~290 LOC,
  default-off, fully tested)
- Tests: `tests/test_markout_feedback.py` + `_integration.py` (49 tests,
  all pass; full suite 584 pass + 1 skip)
- Replay script: `scripts/replay_markout_feedback.py`
- Replay report: `docs/stage2_replay_ETH_markout_feedback.md`
- Decision doc (now superseded by this verdict):
  `docs/iter_decisions/2026-05-15_ETH-USD_iter001-markout-feedback.md`
- Phase 1 calibration (note flagged simplification):
  `docs/stage2_calibration_ETH.md`
