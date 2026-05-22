# DOT-USD microprice A/B — iter002

**Status**: PENDING USER GO. Phase 3 code + replay verification are COMPLETE
(see below); this is a real-money change and is **not launched**. Awaiting
explicit "go live".
**Date drafted**: 2026-05-22
**Plan**: `docs/microprice_ofi_plan.md` Phase 3
**Feature**: `MM_USE_MICROPRICE` (size-weighted fair-value recenter, Stoikov 2018)

---

## 1. What ships

| Knob | `.env.dot` (control) | `.env.dot.iter002` (treatment) |
|---|---|---|
| `MM_USE_MICROPRICE` | (absent ⇒ false) | **true** |
| `MM_MICROPRICE_CAP_BPS` | (absent ⇒ 10) | **10** (explicit) |
| Everything else | — | **byte-identical to `.env.dot`** |

No sizing / credential / market changes. Treatment is the control plus one
flag (and an explicit cap value). Crypto profile ⇒ the gate is satisfied.

**Deploy prerequisite**: `/root/MM` is live at `6c230de`; the microprice code
is on `origin/microprice-ofi` at `dc30f94` but **not deployed**. Launch
requires `git -C /root/MM pull` to `dc30f94` first. This is safe — with the
flag off the quote path is byte-identical (proven by
`tests/test_microprice_integration.py` + 630 passing tests), and no other
`.env.*` sets the flag, so the other 5 live bots are unaffected by the pull
(they only pick up new code on their next restart, and even then default-off).

## 2. Why now — gates already cleared

- **Stage 3 diagnostic (PASS, crypto-only)**: microprice leads mid on crypto
  (ETH corr +0.41 / ρ +0.60); WRONG sign on TradFi → crypto-gated in code.
  `docs/stage3_microprice_diagnostic.md`.
- **Phase 3 replay (PASS)**: real `compute_target_price` on 17,155 DOT
  `book_change` snapshots — correct sign (100% on non-clamped), symmetric
  bid/ask shift, 0 exceptions. Surfaced a 75 bps dislocation tail → added
  `MM_MICROPRICE_CAP_BPS=10` → max |shift| 75.35 → 9.84 bps, calm/normal
  distribution untouched. `docs/stage3_replay_DOT_microprice.md`.

## 3. Pre-registered success / rollback (pin BEFORE launch)

DOT is a single market ⇒ this is a **before/after** comparison, confounded by
regime. Mitigations: (a) pin the matched baseline window with
`scripts/journal_config_history.py`; (b) abort if DOT's spread band leaves the
≥12 bps profitable zone during the test (regime change invalidates it).

**Sample**: ≥30 resting fills under treatment (DOT ~5–10 fills/day ⇒ plan ≥6
days). Primary metric **$/h** over the matched window; secondary: fill rate,
mean edge_bps, mean +5s markout (use `scripts/diagnose_markout.py`).

**Bake-in** (after ≥6 days, ≥30 fills): treatment $/h ≥ baseline $/h with no
safety regression → rename iter002 → baseline, snapshot old.

**Rollback triggers** (any ⇒ flag off + restart on `.env.dot`):
- At 50% sample (≥15 fills): treatment $/h < baseline − $0.50 at p < 0.10.
- Any `drawdown_stop` fire.
- Mean +5s markout degrades vs baseline by > 1 bps on ≥30 fills.

## 4. Launch procedure (on GO)

1. `git -C /root/MM pull` → `dc30f94`. Confirm `git -C /root/MM log -1`.
2. Snapshot baseline: `cp /root/MM/.env.dot /root/MM/.env.dot.pre_microprice.20260522`.
3. Create `/root/MM/.env.dot.iter002` = copy of `.env.dot` + the two lines:
   `MM_USE_MICROPRICE=true` / `MM_MICROPRICE_CAP_BPS=10`. Confirm
   `MM_DEADMAN_ENABLED=false` is present (SDK lacks the switch).
4. `mmctl stop DOT-USD` (note: default flattens — DOT is near-flat, cost
   negligible; verify position first).
5. `mmctl start DOT-USD` from `.env.dot.iter002`.
6. Verify live: `use_microprice=true` in the fresh `run_start`; bot quoting
   both sides; first `order_placed` events present; no error spike.
7. Pin the baseline window for comparison; set a 50%-sample check reminder.

**Rollback**: `mmctl stop DOT-USD` → `mmctl start DOT-USD` from `.env.dot`.
Snapshots preserved; origin branches untouched.
