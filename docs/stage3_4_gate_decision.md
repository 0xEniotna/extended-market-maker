# Decision Gate 1 — Stage 3 + Stage 4 outcomes

**Status**: DECIDED 2026-05-22 → **PROCEED BOTH** (microprice first, then OFI)
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

- **Verdict: PASS (crypto-only)**
- Pooled n_fills: ETH historical 1,538 (power); DOT 53 / XNG 104 underpowered
- Best market correlation: ETH Pearson **+0.41**, Spearman **+0.60** (microprice
  leads mid)
- Worst / wrong-sign: MU_24_5 r = −0.25 (p=0.002), SPX500m r = −0.49 (p=0.0002)
  — TradFi 24/5 stale-book regime ⇒ **crypto-only rollout**
- Sign consistent across crypto markets? Yes (and the TradFi wrong-sign is a
  known dust-quote failure mode, gated out in code)

Link: `docs/stage3_microprice_diagnostic.md`

---

## Stage 4 result

- **Verdict: PASS on DOT** (flow-OFI mean-reverting)
- Per-market verdicts:
  - DOT-USD: **PASS** — flow-OFI Pearson −0.20 / Spearman −0.23 at 5–30s
    (mean-reverting); ~6× stronger than depth-imbalance
  - TECH100m: INCONCLUSIVE (−0.01 to −0.11)
  - NEAR-USD / XNG-USD: killed before book_change accrual; N/A
- OFI reconstruction was feasible offline: required the Phase 0.5
  `book_change` instrumentation (now live); reconstructed via Cont-Kukanov-
  Stoikov in `scripts/diagnose_ofi.py`

Link: `docs/stage4_ofi_diagnostic.md`

---

## Decision

**PROCEED BOTH** — microprice (Phase 3) first, then OFI skew (Phase 4),
sequenced on DOT-USD per anchoring decision #3 (one change per iter).

Justification: both diagnostics passed on crypto. Microprice has the strongest
evidence (ETH r=+0.41) and the simplest implementation, so it ships first and
bakes in before OFI starts live testing — avoiding the correlated-signal
confound that hid Stage 2's calibration/impl gap.

Resulting work plan:
- **Phase 3 (microprice)**: CODE + replay verification **COMPLETE 2026-05-22**
  (`dc30f94`, replay PASS, cap added). Live A/B = DOT iter002, pending user GO
  (`docs/iter_decisions/2026-05-22_DOT-USD_microprice_iter002.md`).
- **Phase 4 (OFI skew)**: queued — starts after DOT microprice bakes in.
- **Phase 5 (roll)**: NEAR/XNG are killed; re-target to current crypto fleet
  (DOT + any future crypto adds) when the time comes.

Recorded by: Claude (microprice-ofi session)
Date: 2026-05-22
