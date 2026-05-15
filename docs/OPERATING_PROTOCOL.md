# MM Bot Operating Protocol

Hard rules for running the market-maker fleet. Read this before touching
any live state.

The bot was built for trading, not for A/B testing. The infra to track
which config produced which PnL exists (every `run_start` event carries
the full config) but it's easy to defeat by editing `.env` files in
place. These rules close that gap.

---

## Rule 0 — Production sanity rules (never break)

These are absolute. Never violate them, even under pressure.

| Rule | Why |
|---|---|
| **Never edit credentials** (`MM_VAULT_ID`, `MM_STARK_*`, `MM_API_KEY`) | Credential rotation has its own procedure. Bot uptime depends on these being stable. |
| **Never edit `MM_ENVIRONMENT` or `MM_MARKET_NAME` in a running env file** | The bot identity is tied to these. Changing them mid-run is undefined behavior. |
| **Never skip hooks or signing** in git operations on this repo | The MM repo has tests gated on hooks for a reason. |
| **Never force-push to `main`** | Main is what `~/MM` on the VPS tracks. A bad push touches production. |

---

## Rule 1 — Never edit `.env.<market>` in place

The file `/root/MM/.env.<market>` is the "production baseline" for that
market. It must remain stable so its content always reflects the config
that produced the all-time PnL for that market.

If you want to change a knob for a test:

```bash
# 1. Pick an iter suffix (incrementing).
ITER=iter003

# 2. Copy the production baseline.
ssh mm-bot 'cp /root/MM/.env.<market> /root/MM/.env.<market>.$ITER && chmod 600 /root/MM/.env.<market>.$ITER'

# 3. Edit the COPY only (never the source).
ssh mm-bot 'sed -i "s/^MM_FOO=.*/MM_FOO=newvalue/" /root/MM/.env.<market>.$ITER'

# 4. Document the diff in docs/iter_decisions/<date>_<market>_$ITER.md
#    (see decision template below).

# 5. Stop the running prod instance and start the iter.
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl stop <market> && mmctl start <market>.$ITER'
```

The `.env.<market>` original is now untouched. The iter file lives next to
it. To roll back, `mmctl stop <market>.$ITER && mmctl start <market>`.

**Exception**: in-place edits of `.env.<market>` are allowed ONLY when:
1. You're committing a permanent change after the iter has been validated
2. You snapshot the previous version first (`cp .env.<m> .env.<m>.preX.YYYYMMDD`)
3. You document the change in `docs/iter_decisions/` with the journal
   timeline showing the change was deliberate.

---

## Rule 2 — Worktree-based dev follows the same pattern

Code changes that aren't ready for `main` live in a git worktree at
`/root/MM-funding-aware/` (or similar). The bot started from a worktree
uses the worktree's `src/market_maker/` via PYTHONPATH override.

The env file used by a worktree bot is symlinked from `/root/MM/`:

```bash
ssh mm-bot 'ln -sf /root/MM/.env.<market>.$ITER /root/MM-funding-aware/.env.<market>.$ITER'
ssh mm-bot 'cd /root/MM-funding-aware && PYTHONPATH=/root/MM-funding-aware/src PATH=/root/MM/.venv/bin:$PATH mmctl start <market>.$ITER'
```

The credentials live in `/root/MM/` only. The symlink keeps the worktree
filesystem free of duplicated secrets while still giving mmctl an env file
to resolve.

Worktree `mmctl status` lives at `/root/MM-funding-aware/data/pids/`,
**separate from production** at `/root/MM/data/pids/`. To see the full
fleet, you must check **both**:

```bash
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl status'
ssh mm-bot 'cd /root/MM-funding-aware && PATH=/root/MM/.venv/bin:$PATH PYTHONPATH=/root/MM-funding-aware/src mmctl status'
```

---

## Rule 3 — Every iter needs a pre-registered decision doc

Before launching `<market>.iterNNN`, write a decision doc:

```
docs/iter_decisions/YYYY-MM-DD_<market>_iterNNN.md
```

Use the template at `docs/iter_decisions/_TEMPLATE.md`. It captures:
- Hypothesis (what you expect to change)
- Config diff from baseline
- Pre-registered success criteria (specific numbers)
- Sample-size required for statistical significance
- Rollback trigger
- Outcome (filled in post-mortem)

**Without this doc, the iter result is unfalsifiable.** You'll find a
narrative to fit whatever happened.

---

## Rule 4 — Always check `journal_config_history.py` before drawing
   conclusions from PnL data

```bash
python scripts/journal_config_history.py --market <MARKET> \
    --journals-dir /root/MM/data/mm_journal \
    --journals-dir /root/MM-funding-aware/data/mm_journal \
    --out docs/config_history_<MARKET>.md
```

The output shows every `run_start` event in chronological order with
diffs. Use it to:

1. Verify the iter you expected to be running was actually running.
2. Spot unintended config changes (hot-reloads, manual edits).
3. Match PnL deltas to config-change boundaries.

`mmctl pnl` returns all-time cumulative. You must combine it with the
config history to know "PnL under config X from t1 to t2".

---

## Rule 5 — Kill criteria for an underperforming market

Before killing a market with `mmctl stop`, verify:

1. **The PnL signal is robust** — at least 3 weeks of data, or ≥100
   closed trades. (Counter-example: BABA killed after 17h with 0 closes
   was an exception — there was no signal at all because no fills
   happened. The decision was "no data is its own data".)
2. **The bleed isn't transient** — funding regime change, single bad
   trading day, etc. don't justify a kill.
3. **No structural fix is cheap** — if widening `MM_MIN_OFFSET_BPS` or
   reducing `MM_MAX_POSITION_NOTIONAL_USD` would plausibly fix the
   bleed, try that first via iter.
4. **The kill is documented** in `docs/iter_decisions/` with the
   cumulative PnL and the analysis that justified it.

---

## Rule 6 — Fleet status hygiene

After every fleet change, log a snapshot in `docs/fleet_status_log.md`:

```
## 2026-MM-DD HH:MM UTC

| Market | PID | Code path | Iter / config | Status |
|---|---|---|---|---|
| ... | ... | main / worktree | iterNNN | RUNNING |

### Changes since last snapshot
- ...
```

This becomes the audit trail. If something goes wrong later, this is
where you find "what was running when".

---

## Rule 7 — No SIGHUP-style hot reloads without a `run_start` re-emit

The bot supports SIGHUP reload via `strategy_components.rebuild_components`.
But the journal does NOT emit a new `run_start` when this happens, so the
post-hoc analysis can't see the change.

Until we add `config_change` events on SIGHUP, **always restart the bot
(stop + start) to apply config changes**. The new journal will have an
accurate `run_start`.

---

## Rule 8 — Trust `run_start` over `.env.<m>` files

The `.env.<m>` files are mutable; the `run_start` event in each journal
is immutable. When you're trying to understand what produced an old PnL
number, the `run_start` is the source of truth, not the current contents
of the env file.

---

## Quick reference: common operations

### Launch a new iter

```bash
ITER=iter003
MARKET=mu_24_5
ssh mm-bot 'cp /root/MM/.env.'$MARKET' /root/MM/.env.'$MARKET'.'$ITER' && chmod 600 /root/MM/.env.'$MARKET'.'$ITER
# Edit the iter file
ssh mm-bot 'sed -i "s/^MM_MIN_OFFSET_BPS=.*/MM_MIN_OFFSET_BPS=8/" /root/MM/.env.'$MARKET'.'$ITER
# Symlink into worktree if running from there
ssh mm-bot 'ln -sf /root/MM/.env.'$MARKET'.'$ITER' /root/MM-funding-aware/.env.'$MARKET'.'$ITER
# Launch
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl start '$MARKET'.'$ITER
# OR from worktree:
ssh mm-bot 'cd /root/MM-funding-aware && PYTHONPATH=/root/MM-funding-aware/src PATH=/root/MM/.venv/bin:$PATH mmctl start '$MARKET'.'$ITER
```

### Rollback an iter

```bash
# Stop the iter, restart the baseline
ssh mm-bot 'cd /root/MM-funding-aware && mmctl stop '$MARKET'.'$ITER
ssh mm-bot 'cd /root/MM && PATH=/root/MM/.venv/bin:$PATH mmctl start '$MARKET
```

### Compare two iters

```bash
# 1. Pull the config history
python scripts/journal_config_history.py --market <MARKET-UPPER> \
    --journals-dir /root/MM/data/mm_journal \
    --journals-dir /root/MM-funding-aware/data/mm_journal

# 2. Pull the markout diagnostic for each iter's journal slice
python scripts/diagnose_markout.py --market <MARKET-UPPER> \
    --journal /path/to/iter1_journal.jsonl --out reports/iter1.md
python scripts/diagnose_markout.py --market <MARKET-UPPER> \
    --journal /path/to/iter2_journal.jsonl --out reports/iter2.md

# 3. Cross-check by hand or build a comparison script (TODO).
```

---

## Things that are broken / known infra gaps

| Gap | Workaround | Long-term fix |
|---|---|---|
| `mmctl pnl` has no `--since/--until` | Subtract two cumulative snapshots, use journal timestamps to anchor | Add date filtering to mmctl |
| No `config_change` event on SIGHUP | Restart bot (stop+start) for any config change | Emit event in `rebuild_components` |
| No automatic `.env.<m>` snapshot on edit | Manual `cp .env.<m> .env.<m>.snap.YYYYMMDD` before edit | Wrap edit in a script that snapshots |
| No statistical-power check before launching iters | Estimate fills/h × hours-of-test; require ≥30 for any directional claim | Add a `tools/power_check.py` |
| No automated A/B comparison report | Hand-walk the journals and diagnostics | `mmctl journal compare <iter1> <iter2>` |
