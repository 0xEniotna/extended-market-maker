# Plan: A1 (Microprice as Fair Value) + A2 (OFI in Quoting)

**Status**: Phase 0 complete; Phase 0.5 in progress
**Created**: 2026-05-15
**Last updated**: 2026-05-15 (added Phase 0.5 — book_change instrumentation)
**Backbone**: `project_learnings.md` ranked top-5, items 1 and 2
**Parent branch**: `funding-aware-mm`
**Working branch**: `microprice-ofi`

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
Phase 0    ─ Setup branch + decision-doc skeletons (DONE)
Phase 0.5  ─ Instrument book_change journal event (code + 24-48h accrual)
Phase 1    ─ A1.1 microprice diagnostic (analytics, ~½ day) — runs in PARALLEL with 0.5
Phase 2    ─ A2.1 true flow-OFI diagnostic (analytics, ~1 day) — gated on 0.5
Gate 1     ─ Do diagnostics support shipping?
Phase 3    ─ A1.2 microprice in quoting (code + replay, ~1.5 days)
           ─ DOT iter002 live A/B (≥6 days)
Gate 2     ─ Bake in microprice?
Phase 4    ─ A2.2 OFI skew in quoting (code + calibration + replay, ~2-3 days)
           ─ DOT iter003 live A/B (≥6 days)
Gate 3     ─ Bake in OFI skew?
Phase 5    ─ Roll to NEAR + XNG (~2 days)
```

**Why Phase 0.5 was added.** Phase 0's journal spot-check revealed that
existing journals carry bid/ask **sizes** only in `fill.market_snapshot` events.
Brief 18's mean-reversion result is specifically about signed flow
(`ΔV_b − ΔV_a` over time), which is **not reconstructible** from existing
journals because L1 book mutations between fills aren't journaled at
sub-event resolution. Three options considered:

1. **Pragmatic**: validate the existing L1 depth-ratio signal
   (`market_snapshot.imbalance`). Risk: false positive/negative because this
   is not Brief 18's signal.
2. **Purist** (selected): instrument a `book_change` journal event capturing
   every L1 mutation with `(bid, bid_qty, ask, ask_qty)`. Adds ~2-3 days.
   Once instrumented, the same diagnostic validates **both** flow-OFI and
   depth-imbalance against subsequent markout and picks the winner.
3. **Defer**: ship microprice first, skip A2 entirely.

The purist option was chosen because (a) the marginal cost is small,
(b) it produces a clean comparison that resolves Brief 18's crypto caveat
empirically, and (c) the same instrumentation is reusable for future
microstructure research (e.g., the eventual fill-model simulator in the
user's top-5 item 4).

---

## Phase 0 — Setup (DONE 2026-05-15)

| Deliverable | Path | Status |
|---|---|---|
| New branch | `microprice-ofi` off `funding-aware-mm` (at `a37b904`) | ✅ |
| Decision-doc skeletons | `docs/iter_decisions/TBD_DOT-USD_*` | ✅ |
| Stage doc placeholders | `docs/stage3_*.md`, `docs/stage4_*.md` | ✅ |
| Journal resolution spot-check | (DOT, ETH, etc.) | ✅ surfaced the book_change need |

Phase 0 was committed in `4e69bd1`.

---

## Phase 0.5 — Instrument `book_change` event (~1-2 days dev + 24-48h accrual)

**Goal**: capture every L1 book mutation in the journal so Brief 18-style
flow-OFI can be reconstructed offline.

### Design

- **New event type**: `book_change`, emitted from `orderbook_manager.py`.
- **Payload (minimal)**: `{ts, seq, run_id, schema_version, type:"book_change", market, bid, bid_qty, ask, ask_qty}` — 4 numeric fields beyond the standard envelope.
- **Emission gate** (in `OrderbookManager`): dedup by comparing
  `(bid_price, bid_qty, ask_price, ask_qty)` to last-emitted tuple. Emit
  only when at least one of the four changed. This avoids redundant events
  when deeper book levels mutate but L1 is unchanged.
- **Trigger point**: end of `_on_orderbook_update`, after the existing
  `_record_mid`/`_record_imbalance` calls. Single emission per book event.
- **Schema version**: stays at v2. Adding a new event type doesn't break
  consumers — unknown types are ignored.
- **fsync class**: non-critical (batched). High frequency expected; durability
  per-event would dominate latency. The existing batch-fsync covers it.
- **Failure mode**: emission wrapped in try/except so a journal write error
  cannot kill the WS callback path.

### Files

| File | Change |
|---|---|
| `src/market_maker/trade_journal.py` | Add `record_book_change(bid, bid_qty, ask, ask_qty)` method |
| `src/market_maker/orderbook_manager.py` | Add `set_journal()`, dedup tracker, `_maybe_emit_book_change()` call from `_on_orderbook_update` |
| `src/market_maker/strategy_runner.py` | Wire `orderbook_manager.set_journal(journal)` at startup (matches the existing `account_stream.set_journal` / `order_mgr.set_journal` pattern) |
| `tests/test_book_change_event.py` | Unit tests: dedup correctness, payload structure, journal write |

### Volume / disk budget

Rough estimate at current DOT volatility:
- ~10-50 L1 mutations/sec during active hours = 36k-180k events/hour
- Each event ~150 bytes JSON = 5-25 MB/hour per market
- 50-100 MB/market/day, well within the existing 50 MB rotation limit
- 3 active markets = 150-300 MB/day total → ~5-10 GB/month. Manageable.

If volume becomes a problem later:
- Add a `min_emit_interval_ms` throttle (e.g., 50ms) — at the cost of resolution
- Or filter by `|Δqty| > threshold`
- Both are easy follow-ups; ship full-resolution first.

### Rollout

Reuses the existing `/root/MM-funding-aware/` worktree on the VPS (currently
idle — no bots running from it per `project_fleet.md`). Just switch its
branch to `microprice-ofi`:

```bash
ssh mm-bot 'cd /root/MM-funding-aware && git fetch && git checkout microprice-ofi'
# Migrate DOT-USD off /root/MM/ (main) to /root/MM-funding-aware/ (microprice-ofi)
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl stop DOT-USD'
ssh mm-bot 'ln -sf /root/MM/.env.DOT-USD /root/MM-funding-aware/.env.DOT-USD'
ssh mm-bot 'cd /root/MM-funding-aware && PYTHONPATH=/root/MM-funding-aware/src \
  PATH=/root/MM/.venv/bin:$PATH mmctl start DOT-USD'
```

1. Code + tests on `microprice-ofi` branch, push to origin.
2. Checkout `microprice-ofi` in `/root/MM-funding-aware/`.
3. Stop DOT-USD on `/root/MM/` (main), restart on `/root/MM-funding-aware/`.
   This is just an env change — DOT runs the same `.env.DOT-USD` baseline,
   only the code path differs (with book_change instrumentation enabled).
4. Monitor 24h: check journal size growth, no perf regression, no error spike.
5. If clean, do the same migration for NEAR-USD and XNG-USD.
6. After ≥24-48h of `book_change` accrual on DOT, Phase 2 can run.

To roll back: `mmctl stop DOT-USD` on the worktree, `mmctl start DOT-USD` on
`/root/MM/`. Branch separation is preserved — `funding-aware-mm` is unchanged
in origin; the worktree just temporarily tracks `microprice-ofi`.

### Acceptance criteria

Pre-registered:
- `book_change` events present in fresh DOT journal at expected volume (≥1k events/h).
- No new error log entries above baseline.
- Bot quote latency p95 unchanged.
- Events parse cleanly with `analyse_mm_journal.py` (or are ignored without traceback).

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

**Prerequisite**: Phase 0.5 must have accrued ≥24-48h of `book_change` events on at least one running market (DOT).

**Goal**: validate Brief 18's claim that signed flow-OFI shocks mean-revert. Compare against the existing depth-imbalance signal to determine which is the better predictor of adverse markout for our markets.

**Deliverables**:
- `scripts/diagnose_ofi.py`
  - **Two signals computed in parallel** from the `book_change` event stream:
    - **Flow-OFI (Brief 18)**: standard signed-flow formula
      ```
      ΔV_b(t) = bid_qty_t               if bid_price_t > bid_price_{t-1}
              = bid_qty_t - bid_qty_{t-1}  if bid_price_t == bid_price_{t-1}
              = -bid_qty_{t-1}            if bid_price_t < bid_price_{t-1}
      (symmetric for ask)
      OFI(t) = ΔV_b(t) - ΔV_a(t)
      ```
      Aggregated over rolling window `imbalance_window_s` (default 2.0s).
    - **Depth-imbalance** (existing bot signal): `(bid_qty − ask_qty) / (bid_qty + ask_qty)` at L1, EWMA-smoothed.
  - Bucket each fill by each signal's quintile.
  - Compute mean `+5s markout` per bucket, per side, per signal.
  - Output two monotonicity tests (Spearman ρ for each signal).
- Run on DOT first; expand to NEAR/XNG once they have `book_change` data too.

**Decision criteria** (pre-registered in `docs/stage4_ofi_diagnostic.md`):
- **Proceed to A2.2 with flow-OFI** if: flow-OFI shows monotone mean-reverting relationship (high positive flow-OFI quintile → negative markout for bid fills), p < 0.05.
- **Proceed to A2.2 with depth-imbalance** if: only depth-imbalance is monotone mean-reverting (and flow-OFI is null/trending). Use the cheaper, already-computed signal.
- **Proceed with whichever signal is stronger** if both pass.
- **Skip A2** if: both signals fail (trending or null). Document Brief 18's caveat as confirmed for our markets.

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
| 0 | Phase 0 setup (DONE — branch + decision-doc skeletons + journal spot-check) |
| 1 | Phase 0.5 code (book_change instrumentation) + tests |
| 1 | Phase 1 diagnostic script (in parallel — uses fill snapshots, doesn't need 0.5) |
| 2 | Phase 0.5 rollout to DOT-USD; monitor 24h |
| 2 – 3 | Run Phase 1, write `stage3_microprice_diagnostic.md` |
| 3 – 4 | book_change journal accumulation; Phase 0.5 expand to NEAR/XNG |
| 4 – 5 | Phase 2 diagnostic (flow-OFI + depth-imbalance compared); write `stage4_*` |
| 5 | Decision Gate 1 (`stage3_4_gate_decision.md`) |
| 6 – 7 | Phase 3 microprice code, tests, replay |
| 7 – 13 | DOT iter002 live A/B (≥6 days) |
| 13 | Decision Gate 2 + bake-in or rollback |
| 14 – 16 | Phase 4 OFI code, calibration, replay |
| 16 – 22 | DOT iter003 live A/B (≥6 days) |
| 22 | Decision Gate 3 + bake-in or rollback |
| 23 – 24 | Phase 5 roll to NEAR + XNG |

**Total elapsed**: ~3.5 weeks (was 3 — Phase 0.5 adds ~2-3 days). Heads-down dev: ~8-9 days. Rest is waiting on journal/A-B sample accrual.

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
