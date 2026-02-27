# Config Proposal Pipeline

## Overview

The config proposal pipeline is deterministic and fail-closed:

1. Agents write proposal JSON files to `mm_config/proposals/`.
2. `mmctl apply-proposal` validates the proposal against policy (`whitelist` + `bounds`).
3. `mmctl` snapshots the target `.env.<market>` file.
4. `mmctl` applies changes atomically with a file lock.
5. `mmctl` runs an optional reload hook (default NOOP).
6. `mmctl` archives the applied proposal and metadata in `mm_config/applied/`.

Direct edits by agents to `.env.*` are not allowed.

## Policy Files

- `mm_config/policy/whitelist.json`
  Defines allowed env keys.
- `mm_config/policy/bounds.json`
  Defines key type and bounds.
- `mm_config/policy/protected_keys.json` (optional)
  Defines keys that are never mutable via proposals (e.g. API/private keys).

Supported value types:

- `int` with `min`/`max`
- `float` with `min`/`max`
- `bool` (`true/false/1/0`, normalized to `true`/`false`)
- `string` (optional `allowed` list)

## Proposal Format

```json
{
  "proposal_id": "2026-02-26_1430Z_MON_001",
  "market": "MON",
  "env_file": "mm-env/.env.mon",
  "changes": [
    {"key": "QUOTE_OFFSET_BPS", "op": "set", "value": "9.0"},
    {"key": "REPRICE_MIN_INTERVAL_MS", "op": "set", "value": "450"},
    {"key": "TOXICITY_GUARD_ENABLED", "op": "set", "value": "true"}
  ],
  "reason": {"hypothesis": "spread too tight"},
  "canary": {"duration_minutes": 30, "success_metrics": {"pnl": ">=0"}},
  "created_at": "2026-02-26T14:30:00Z"
}
```

Validation rules:

- `market` must match `^[A-Z0-9]+$`.
- Only op `set` is accepted.
- All changed keys must be whitelisted and bounded.
- Protected keys are rejected even if they appear in whitelist.
- Secret-like key names (`*PRIVATE_KEY*`, `*_SECRET*`, `*PASSPHRASE*`, `*MNEMONIC*`, `*_API_KEY*`) are always rejected.
- `env_file` (if provided) must match `.env.<lowercase market>`.
- Unknown fields are rejected in strict mode.

## Env File Location

Env directory is controlled by `MM_ENV_DIR`.

- Default: `./mm-env/`
- Market file for `MON`: `${MM_ENV_DIR}/.env.mon`

## Commands

### Preview Diff

```bash
mmctl diff-proposal <proposal_id_or_path>
```

No writes are performed.

### List Pending Advisor Proposals

```bash
python scripts/mm_advisor_apply.py --repo /home/flexouille/Code/MM --list-pending --json
```

Use this before applying by id to confirm the proposal exists and is still pending.

### Submit Manual/Analyst Proposals Safely

Do not append raw JSON lines directly to `proposals.jsonl`.
Use the deterministic submitter so rows are normalized and validated:

```bash
python scripts/mm_advisor_submit.py \
  --repo /home/flexouille/Code/MM \
  --market ETH-USD \
  --param MM_MIN_OFFSET_BPS \
  --proposed 3 \
  --reason "tighten for unwind" \
  --json
```

Bulk import (JSONL intents from analyst):

```bash
python scripts/mm_advisor_submit.py \
  --repo /home/flexouille/Code/MM \
  --input-jsonl /path/to/analyst_intents.jsonl \
  --json
```

Notes:
- Submission uses live `.env` values for `old` to avoid stale-ID mismatch.
- Protected/sensitive keys are rejected.
- Import is all-or-nothing: if any JSONL row is malformed/invalid, nothing is written.
- Output IDs are immediately applyable by `mm_advisor_apply.py`.

### Apply Proposal

```bash
mmctl apply-proposal <proposal_id_or_path> --json
```

Behavior:

- Validates proposal.
- Locks target env file (`.env.<market>.lock`).
- Creates before/after snapshots in `mm_config/snapshots/`.
- Applies changes atomically.
- Archives proposal and metadata in `mm_config/applied/`.
- Returns deterministic JSON output.

### Rollback

```bash
mmctl rollback MON --to <snapshot_path_or_snapshot_id> --json
```

Behavior:

- Resolves snapshot by path or id.
- Locks target env file.
- Creates pre-rollback snapshot.
- Restores snapshot atomically.
- Runs reload hook.

## Reload Hook

Default reload behavior is NOOP:

- message: `Reload is NOOP; process must be restarted to pick changes.`

Optional command hook via env var:

```bash
export MM_RELOAD_CMD_TEMPLATE='systemctl --user restart mm-executor@{market}'
```

`{market}` is substituted with lowercase market, `{MARKET}` with uppercase.

## Safety Rules

- No automatic tuning by runtime bots.
- No delete/unset ops.
- Validation failure means no writes.
- Atomic writes with lock.
- Snapshots always created before apply/rollback.
