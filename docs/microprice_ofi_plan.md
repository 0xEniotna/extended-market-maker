# Plan: A1 (Microprice as Fair Value) + A2 (OFI in Quoting)

**Status**: Draft — pending kick-off
**Created**: 2026-05-15
**Backbone**: `project_learnings.md` ranked top-5, items 1 and 2
**Parent branch**: `funding-aware-mm`
**Working branch**: `microprice-ofi` (to be created)

---

## Context

The fleet's structural pattern (`project_learnings.md`): spread band determines viability — ≤8 bps toxic, ≥12 bps profitable. Current bots (DOT, NEAR, XNG) all sit in the ≥12 bps band. The next-mile improvement isn't more markets — it's better quoting on the markets we keep.

Microprice and OFI-in-quoting are the top two ranked improvements. Together they address two known blind spots in the current quoting engine:

- **Microprice** corrects the fair-value reference. Today `compute_target_price` centers quotes on `mid_dec`, ignoring depth asymmetry. When the book is skewed (e.g., 10× more size on the bid), the true short-horizon-fair price is closer to the ask. Quoting symmetrically around mid is mispricing.
- **OFI** adds a signed, mean-reverting skew term. The bot already computes `orderbook_imbalance(window_s)` but only consumes it as a halt gate. OFI in *quoting* lets the bot lean against transient flow imbalance.

These are individually small (~80 LOC + ~300 LOC) but compose into a meaningful $/h improvement on borderline markets per the project-learnings projection.

---

## Anchoring decisions

1. **Branch off `funding-aware-mm`**, not main. The experiment-tracking infrastructure (`OPERATING_PROTOCOL.md`, `iter_decisions/`, `journal_config_history.py`, Stage 1/Stage 2 modules) is only on that branch.
2. **First live market: DOT-USD.** Highest baseline (+$91/wk), oldest journal, lowest blast radius.
3. **One change per iter.** Microprice ships and bakes in before OFI starts live testing. Avoids the correlated-signal confound that hid Stage 2's calibration/implementation gap.
4. **Two diagnostic gates before any live change.** They might tell us not to ship.
5. **Stage 1/Stage 2 replay pattern is mandatory.** Phase 3 (real-impl replay) caught the markout-feedback overstate (calibration −1.71 bps vs real −0.28 bps). Both new overlays must clear the same gate.

---

## Code touchpoints (verified)

| File | Today | Change for A1 / A2 |
|---|---|---|
| `src/market_maker/pricing_engine.py:177` (`compute_target_price`) | Uses `mid_dec` as fair value | A1: replace with `fair_value_dec = microprice(...) if MM_USE_MICROPRICE else mid_dec` |
| `src/market_maker/pricing_engine.py:222` (`signal_offset_f`) | Adds funding-aware overlay term | A2: also add `ofi_skew_bps` term, capped |
| `src/market_maker/orderbook_manager.py:257` (`orderbook_imbalance`) | Returns Decimal mean over `imbalance_window_s` window, used only by `guard_policy` halt | A2: also consumed by new `ofi_signal.py` |
| `src/market_maker/config_base.py` | `imbalance_window_s` field exists | Add `use_microprice`, `ofi_skew_enabled`, `ofi_skew_k`, `ofi_cap_bps` |
| `src/market_maker/types.py` | Same | Mirror config fields |

New modules:
- `src/market_maker/microprice.py` (~50 LOC) — pure function + edge cases
- `src/market_maker/ofi_signal.py` (~150 LOC) — OFI → skew_bps with calibration

---

## Order of operations

```
Phase 0  ─ Setup branch + decision-doc skeletons
Phase 1  ─ A1.1 microprice diagnostic (analytics, ~½ day)
Phase 2  ─ A2.1 OFI mean-reversion diagnostic (analytics, ~1 day)
Gate 1   ─ Do diagnostics support shipping?
Phase 3  ─ A1.2 microprice in quoting (code + replay, ~1.5 days)
         ─ DOT iter002 live A/B (≥6 days)
Gate 2   ─ Bake in microprice?
Phase 4  ─ A2.2 OFI skew in quoting (code + calibration + replay, ~2-3 days)
         ─ DOT iter003 live A/B (≥6 days)
Gate 3   ─ Bake in OFI skew?
Phase 5  ─ Roll to NEAR + XNG (~2 days)
```

---

## Phase 0 — Setup (½ day)

| Deliverable | Path |
|---|---|
| New branch | `microprice-ofi` off `funding-aware-mm` |
| Decision-doc skeleton (microprice) | `docs/iter_decisions/2026-05-XX_DOT-USD_microprice_iter002.md` |
| Decision-doc skeleton (OFI) | `docs/iter_decisions/2026-05-XX_DOT-USD_ofi_iter003.md` |
| Stage doc placeholders | `docs/stage3_microprice_*.md`, `docs/stage4_ofi_*.md` |

---

## Phase 1 — A1.1: Microprice diagnostic (analytics only)

**Goal**: confirm microprice differs from mid in a way that correlates with adverse markout. If not, A1.2 isn't worth shipping.

**Deliverables**:
- `scripts/diagnose_microprice.py`
  - Walks any journal; for each `fill` and `order_placed` event with a `market_snapshot`, computes:
    - `microprice = (bid · ask_size + ask · bid_size) / (bid_size + ask_size)`
    - `(microprice − mid)` at event time
    - `+5s mid markout` after fills (use existing `diagnose_markout.py` joining logic)
  - Outputs per-side, per-market table:
    - `corr((microprice − mid), +5s_markout)` — does microprice predict where mid is heading?
    - Decomposition by inventory bucket (matches `diagnose_markout.py` conventions)
  - Pooled across journals via `--journal` repeat flag (Stage 2 pattern)
- Run on DOT-USD, NEAR-USD, XNG-USD, plus historical ETH (1,538 fills there — gold for power)

**Decision criteria** (pre-registered in `docs/stage3_microprice_diagnostic.md`):
- **Proceed to A1.2** if: `|corr(microprice − mid, +5s markout)| ≥ 0.05` at `p < 0.05` on at least one market with n≥100 fills, sign consistent with "microprice leads mid."
- **Stop A1** if: correlation null on all markets, or signs inconsistent across markets.

**Effort**: ½ day script + ½ day analysis & writeup.

---

## Phase 2 — A2.1: OFI mean-reversion diagnostic (analytics only)

**Goal**: validate Brief 18's claim that OFI shocks mean-revert. The brief flagged that crypto liquidations may trend instead — we must know which regime our markets are in.

**Deliverables**:
- `scripts/diagnose_ofi.py`
  - OFI per event: `(Δbid_qty − Δask_qty) / (|Δbid_qty| + |Δask_qty|)` rolling over `imbalance_window_s` (default 2.0s — keep config-aligned)
  - Bucket each fill by OFI sign + magnitude (quintiles)
  - Compute mean `+5s markout` per bucket, per side
  - Output: monotonicity test (Spearman ρ between OFI quintile rank and signed markout)
- Run on DOT/NEAR/XNG/ETH historical

**Decision criteria** (pre-registered in `docs/stage4_ofi_diagnostic.md`):
- **Proceed to A2.2** if: high-positive OFI quintile → negative mean markout for bid fills (and symmetric for ask). Monotone across quintiles. `p < 0.05`.
- **Skip A2** if: high-positive OFI predicts positive markout (trending — Brief 18's crypto caveat confirmed for our markets). Document, archive, move on.

**Effort**: 1 day script + ½ day analysis.

---

## ✋ Decision Gate 1

Outcome matrix (record in `docs/stage3_4_gate_decision.md`):

| Phase 1 result | Phase 2 result | Action |
|---|---|---|
| Positive | Positive | Run both A1.2 and A2.2 (sequenced) |
| Positive | Null/trending | A1.2 only |
| Null | Positive | A2.2 only — but be cautious without microprice |
| Null | Null/trending | Stop the plan. Investigate. Don't ship either. |

---

## Phase 3 — A1.2: Microprice in quoting (live)

### Code (~1.5 days)

- `src/market_maker/microprice.py` (~50 LOC):
  ```python
  def microprice(bid, ask, bid_qty, ask_qty) -> Decimal:
      total = bid_qty + ask_qty
      if total <= 0:
          return (bid + ask) / 2  # fallback to mid
      return (bid * ask_qty + ask * bid_qty) / total
  ```
  Edge cases: zero/one-sided book (fall back to mid). Decimal arithmetic on the slow path; the hot path uses float (matches existing pattern).
- `src/market_maker/pricing_engine.py:177`:
  ```python
  if self._settings.use_microprice:
      fair_value_dec = microprice(best_bid, best_ask, bid_qty, ask_qty)
  else:
      fair_value_dec = mid_dec
  ```
- Config:
  - `MM_USE_MICROPRICE` (default `false`) in `config_base.py` and `types.py`
- Journaling: emit `microprice` value alongside `mid` in `order_placed` and `fill` events for downstream A/B analysis.
- Tests: `tests/test_microprice.py` — formula correctness, edge cases, Decimal precision.

### Verification (Stage 2 replay pattern)

- `scripts/replay_microprice.py` (mirrors `replay_markout_feedback.py`):
  - Loads journal, walks `order_placed` events, recomputes `compute_target_price` with `use_microprice=True` vs `False` using the **real** `PricingEngine` (not a simplified sim).
  - Reports per-fill quote-price perturbation distribution, theoretical edge impact, max/min divergence.
  - This is the gate against the Stage 2 trap.
- All existing tests + new microprice tests pass. `ruff` clean.

### Live A/B: DOT iter002

Decision doc: `docs/iter_decisions/2026-05-XX_DOT-USD_microprice_iter002.md`.

- **Control**: `.env.DOT-USD` (iter001 baseline, currently live, +$91/wk)
- **Treatment**: `.env.DOT-USD.iter002` — identical to baseline + `MM_USE_MICROPRICE=true`
- **Primary metric**: $/h, matched window (use `journal_config_history.py` to pin)
- **Secondary**: fill rate, mean edge_bps, mean +5s markout
- **Safety**: drawdown_stop unchanged, deadman OFF, MAX_ORDER_NOTIONAL preserved
- **Sample size**: ≥30 fills per condition (`project_learnings.md` statistical-trap rule). DOT averages ~5-10 fills/day → plan for ≥6 days.
- **Pre-registered rollback triggers**:
  - At 50% sample size (≥15 fills): if treatment $/h < baseline $/h − $0.50 with `p < 0.10` → rollback.
  - Any drawdown_stop fire → rollback automatically.
- **Operating discipline**: snapshot `.env.DOT-USD` to `.env.DOT-USD.pre_microprice.20260515` before any change (per `OPERATING_PROTOCOL.md` §2).

### Decision Gate 2

After ≥6 days, ≥30 fills:
- **Bake in** if: treatment $/h ≥ baseline $/h, no safety regression. Rename iter002 → baseline. Snapshot old baseline.
- **Extend** if: neutral. Run to 10 days.
- **Rollback** if: worse. Post-mortem in decision doc.

---

## Phase 4 — A2.2: OFI skew in quoting (live)

### Code (~2-3 days)

- `src/market_maker/ofi_signal.py` (~150 LOC):
  - Consumes `orderbook_manager.orderbook_imbalance(window_s)` (existing primitive)
  - `ofi_skew_bps = clip(k_ofi · OFI, -cap, +cap)` where `k_ofi` and `cap` are calibrated per market from A2.1
  - Sign: positive OFI (buying pressure) → widen bid offset (less eager to buy at the now-likely-elevated price), tighten ask offset (more eager to sell into reversion)
  - Convention: matches the existing `signal_offset_f` sign in `pricing_engine.py:228`
- `src/market_maker/pricing_engine.py:222`:
  ```python
  signal_bps = funding_aware_bps + (ofi_skew_bps if self._settings.ofi_skew_enabled else 0)
  ```
- Config:
  - `MM_OFI_SKEW_ENABLED` (default `false`)
  - `MM_OFI_SKEW_K` (per-market, loaded from calibration file)
  - `MM_OFI_CAP_BPS` (default 3)
- Journaling: emit `OFI` and `ofi_skew_bps` in `order_placed`.
- Tests: `tests/test_ofi_signal.py` — sign convention, cap, composition with funding-aware overlay.

### Calibration

- `scripts/calibrate_ofi_signal.py`:
  - Re-runs Phase 2 diagnostic, fits `k_ofi` = slope of mean markout per OFI unit (signed appropriately)
  - Fits `cap_bps` = 95th percentile of |OFI · k_ofi|
  - Saves to `data/ofi_calibration/<MARKET>.json` (follow `data/funding_history/` pattern)
- Auto-load on bot start; emit calibration metadata in `run_start` event.

### Verification (Stage 2 replay pattern)

- `scripts/replay_ofi_signal.py`:
  - Walks journal with **real** `OfiSignalPolicy` from `src/`
  - Reports active%, mean skew_bps, would-have-fill distribution under skew vs no-skew
  - **The Stage 2 trap reapplies**: ensure the replay uses the real `imbalance_window_s` rolling window (not an idealized instantaneous compute). Pull state directly from `orderbook_manager` if needed.

### Live A/B: DOT iter003 (after microprice is baked in)

Decision doc: `docs/iter_decisions/2026-05-XX_DOT-USD_ofi_iter003.md`.

- **Control**: post-Phase-3 baseline (microprice ON)
- **Treatment**: iter003 — control + `MM_OFI_SKEW_ENABLED=true`
- Same sample-size + rollback logic as Phase 3.
- Both microprice and OFI on simultaneously. To attribute, the control already includes microprice → the delta is OFI's contribution.

### Decision Gate 3

After ≥6 days, ≥30 fills: bake in / extend / rollback per Phase 3 pattern.

---

## Phase 5 — Roll to NEAR + XNG (~2 days)

After DOT iter003 is baked in:

- **NEAR-USD iter002**: bump from current iter001 baseline. Both flags ON. `k_ofi` calibrated from NEAR journal (will need ~7-10 days of journal first — NEAR was just launched).
- **XNG-USD iter002**: same.
- **HOOD/ORCL Sunday launch**:
  - If Phase 5 complete: launch with both flags ON from iter001 (use ORCL_24_5 template).
  - If not complete: launch with flags OFF, treat as A/B markets later.

---

## Files added / modified (summary)

```
NEW:
  src/market_maker/microprice.py
  src/market_maker/ofi_signal.py
  tests/test_microprice.py
  tests/test_ofi_signal.py
  scripts/diagnose_microprice.py
  scripts/diagnose_ofi.py
  scripts/calibrate_ofi_signal.py
  scripts/replay_microprice.py
  scripts/replay_ofi_signal.py
  data/ofi_calibration/<MARKET>.json (one per market)
  docs/stage3_microprice_diagnostic.md
  docs/stage4_ofi_diagnostic.md
  docs/stage3_4_gate_decision.md
  docs/iter_decisions/2026-05-XX_DOT-USD_microprice_iter002.md
  docs/iter_decisions/2026-05-XX_DOT-USD_ofi_iter003.md

MOD (small):
  src/market_maker/pricing_engine.py            (~10 LOC)
  src/market_maker/config_base.py               (4 fields)
  src/market_maker/types.py                     (4 fields)
```

Approximate LOC: A1 ~150 (incl tests + scripts), A2 ~400 (incl tests + scripts + calibration).

---

## Timeline

| Day | Work |
|---|---|
| 1 (½) | Phase 0 setup + Phase 1 diagnostic script |
| 1 (½) – 2 | Run Phase 1, write `stage3_microprice_diagnostic.md` |
| 2 – 3 | Phase 2 diagnostic + write `stage4_ofi_diagnostic.md` |
| 3 | Decision Gate 1 (`stage3_4_gate_decision.md`) |
| 4 – 5 | Phase 3 microprice code, tests, replay |
| 5 – 11 | DOT iter002 live A/B (≥6 days) |
| 11 | Decision Gate 2 + bake-in or rollback |
| 12 – 14 | Phase 4 OFI code, calibration, replay |
| 14 – 20 | DOT iter003 live A/B (≥6 days) |
| 20 | Decision Gate 3 + bake-in or rollback |
| 21 – 22 | Phase 5 roll to NEAR + XNG |

**Total elapsed**: ~3 weeks. Heads-down dev: ~6-7 days. Rest is waiting on live A/B sample accrual.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phase 1/2 diagnostic shows null → wasted setup | Acceptable cost — diagnostic is ~1.5 days total. Better to know than to ship blind. |
| Microprice and OFI are correlated signals → double-counting | Verified in replay: emit both metrics in `order_placed`, inspect Pearson correlation, document. If `|r| > 0.7` consider scaling `k_ofi` down. |
| Replay shows real-impl differs from calibration (Stage 2 redux) | The whole point of Phase 3 replay is to catch this. If detected, **do not launch live** — patch the gap in the real impl first. |
| DOT regime change during A/B invalidates comparison | Use `journal_config_history.py` to pin windows. If DOT spread band shifts (≥12 bps → <12 bps), abort iter and re-baseline. |
| Sample-size insufficient at 6 days | Extend to 10 days. If still inconclusive at 10 days, treat as "no measurable improvement" and keep flag OFF. |
| Both overlays interact badly with existing funding-aware overlay if it gets re-enabled later | `signal_offset_f` composition is linear — should be additive — but verify in tests that all three terms (`funding_aware` + `ofi_skew` + microprice-shifted base) compose without sign errors. |

---

## Open questions to resolve before kickoff

1. The `OFI` we'd compute from journals is reconstructed from `book_change` deltas — does the journal log enough resolution? (Verify: spot-check on DOT journal: is each book update logged, or only on fill?)
2. Sample-size sufficiency on NEAR (just launched, <2 days of journal at plan start) — likely OK by the time Phase 5 starts.
3. HOOD/ORCL — calibrate `k_ofi` from a related-asset proxy, or wait for 7 days of own journal?

These questions belong in Phase 0 / Phase 1; they don't block plan approval.
