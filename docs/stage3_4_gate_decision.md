# Decision Gate 1 — Stage 3 + Stage 4 outcomes

**Status**: NOT STARTED
**Pre-registered**: 2026-05-15
**Parent plan**: `docs/microprice_ofi_plan.md`

This doc consolidates the outcomes of Stage 3 (microprice diagnostic) and
Stage 4 (OFI diagnostic) and records the decision on whether to proceed to
Phase 3 / Phase 4.

---

## Outcome matrix (pre-registered)

| Stage 3 (microprice) | Stage 4 (OFI) | Action |
|---|---|---|
| PASS | PASS | Proceed Phase 3 (microprice) → Phase 4 (OFI), sequenced. |
| PASS | FAIL (trending) | Proceed Phase 3 only. Document A2 as not viable on our universe. |
| PASS | INCONCLUSIVE | Proceed Phase 3. Re-run Stage 4 after Phase 3 bake-in with more journal data. |
| FAIL | PASS | Proceed Phase 4 only. Be cautious about quoting around mid vs microprice — A2 will compose with the existing mid-anchored quoting. |
| FAIL | FAIL | Stop the plan. Investigate. Don't ship either. |
| FAIL | INCONCLUSIVE | Stop the plan. Investigate. |
| INCONCLUSIVE | any | Extend Stage 3 with more data first. |

---

## Stage 3 result

(To be filled after Stage 3 runs.)

- Verdict: TBD
- Pooled n_fills: TBD
- Best market correlation: TBD
- Worst market correlation: TBD
- Sign consistent across markets? TBD

Link: `docs/stage3_microprice_diagnostic.md`

---

## Stage 4 result

(To be filled after Stage 4 runs.)

- Verdict: TBD
- Per-market verdicts:
  - DOT-USD: TBD
  - NEAR-USD: TBD
  - XNG-USD: TBD
  - ETH-USD (historical): TBD
  - MU_24_5-USD (historical): TBD
- OFI reconstruction was feasible offline: TBD (yes / required instrumentation)

Link: `docs/stage4_ofi_diagnostic.md`

---

## Decision

**[PROCEED BOTH / PROCEED MICROPRICE ONLY / PROCEED OFI ONLY / STOP]** — TBD

Justification: TBD

Resulting work plan:
- Phase 3 launch target: TBD (or N/A)
- Phase 4 launch target: TBD (or N/A)
- Phase 5 (NEAR/XNG roll) launch target: TBD

Recorded by: TBD
Date: TBD
