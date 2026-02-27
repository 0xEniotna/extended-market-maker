# extended-market-maker

Standalone market-making engine for Extended Exchange, extracted from a larger research monorepo and hardened for open-source usage.

## What This Repo Contains

- `src/market_maker/`: async MM engine and core components.
- `scripts/`: runbook, journal analysis, market screening, and tuning utilities.
- `tests/`: MM-focused unit tests.
- `config/examples/`: sanitized example MM environment presets.
- `MM_CONFIG_GUIDE.md`: detailed parameter reference.

## Architecture

Core runtime flow:

1. `OrderbookManager` streams best bid/ask and microstructure signals.
2. `MarketMakerStrategy` computes target quotes per `(side, level)`.
3. `RiskManager` clips order size to position and notional limits.
4. `OrderManager` places/cancels post-only orders and tracks lifecycle.
5. `AccountStreamManager` consumes fills/order updates/positions.
6. `TradeJournal` records structured JSONL events.
7. `analyse_mm_journal.py` turns raw journal data into tuning diagnostics.

## Quick Start

### 1) Clone + submodule

```bash
git clone git@github.com:0xEniotna/extended-market-maker.git
cd extended-market-maker
git submodule update --init --recursive
```

### 2) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e python_sdk
pip install -e ".[dev]"
```

### 3) Configure

```bash
cp .env.example .env
# Fill MM credentials and market parameters in .env
```

### 4) Run

```bash
PYTHONPATH=src python -m market_maker
```

Alternative entrypoint:

```bash
PYTHONPATH=src python scripts/run_market_maker.py
```

## Main Scripts

- `scripts/analyse_mm_journal.py`: summarize one run or the latest journal in a directory.
- `mmctl apply-proposal|rollback|diff-proposal`: deterministic config proposal control plane (see `docs/config_proposals.md`).
- `scripts/mm_advisor_loop.py`: advisor-only config proposal loop (never mutates env, never restarts bots).
- `scripts/mm_advisor_apply.py`: approval-gated proposal apply tool (writes `.env`, apply receipts, changelog apply rows).
- `scripts/mm_openclaw_controller.sh`: single-instance long-running advisor controller (per env).
- `scripts/mm_openclaw_fleet.sh`: run multiple advisor controllers (one per `.env.*` instance).
- `scripts/screen_mm_markets.py`: spread/tick/volume suitability screening.
- `scripts/tools/find_mm_markets.py`: rolling market filter for MM candidates.
- `scripts/tools/fetch_market_info.py`: inspect trading config and stats for one market.
- `scripts/tools/fetch_pnl.py`: account-level market PnL summary.
- `scripts/tools/fetch_total_pnl.py`: total account PnL across all markets since a timestamp, with APR/APY.
- `scripts/tools/close_mm_position.py`: flatten one market position (reduce-only `MARKET+IOC`) for ops/agent workflows.
- `scripts/tools/analyze_mm_logs.py`: parse text logs for lifecycle and latency diagnostics.
- `scripts/tools/market_scout_pipeline.py`: deterministic market scouting and action-pack generation.
- `scripts/tools/auditor_apply_scout.py`: auditor decisioning (`APPROVE`/`HOLD`/`REJECT`) over scout actions.
- `scripts/tools/auditor_followup.py`: 30-minute pending-action follow-up and escalation checks.
- `config/market_scout_policy.yaml`: policy thresholds/guardrails for scout + auditor workflow.

## Multi-Instance Supervision

Run one controller per strategy instance:

```bash
cd /path/to/repo
scripts/mm_openclaw_fleet.sh start .env.asset
scripts/mm_openclaw_fleet.sh status .env.asset
scripts/mm_openclaw_fleet.sh logs .env.asset
```

Each controller monitors only its own market journal and emits advisory config proposals.  
Controllers do not self-edit `.env` files and do not restart strategy instances.

Apply authority:
- Dead-man proposals (`deadman=true` + `guardrail_status=passed`) are escalated to Warren auto-apply.
- All other proposals are routed for human review.

Apply examples:

```bash
# Human apply for one approved proposal
.venv/bin/python scripts/mm_advisor_apply.py \
  --proposal-id <proposal_id> \
  --approve \
  --json

# Warren auto-apply pass (deadman-only proposals)
.venv/bin/python scripts/mm_advisor_apply.py \
  --mode warren-auto \
  --approve \
  --json
```

## Scout + Auditor Workflow

Deterministic scout + auditor scripts (Discord-first, recommend-only):

```bash
cd /path/to/repo
.venv/bin/python scripts/tools/market_scout_pipeline.py
.venv/bin/python scripts/tools/auditor_apply_scout.py --print-target auditor
.venv/bin/python scripts/tools/auditor_apply_scout.py --print-target mm
.venv/bin/python scripts/tools/auditor_followup.py --print-target auditor
.venv/bin/python scripts/tools/auditor_followup.py --print-target mm
```

Note on analyst file access:
- If your analyst agent is `messaging`-only (no filesystem tools), it should read config context from scout artifacts, not from direct `.env.*` reads.
- `market_scout_report.json` now includes `active_markets[].config_snapshot` with allowlisted MM tuning keys and `MM_TOXICITY_*` keys.

Primary artifacts:
- `data/mm_audit/scout/market_scout_report.json`
- `data/mm_audit/scout/action_pack.json`
- `data/mm_audit/scout/market_scout_report.md`
- `data/mm_audit/scout/market_scout_actions.sh`
- `data/mm_audit/advisor/proposals.jsonl`
- `data/mm_audit/advisor/apply_receipts.jsonl`
- `data/mm_audit/autotune_baselines/<MARKET>.json`
- `data/mm_audit/auditor/auditor_decisions.jsonl`
- `data/mm_audit/auditor/pending_actions.json`
- `data/mm_audit/auditor/auditor_followup_log.jsonl`

## Safety Notes

- Never commit real credentials.
- Start on testnet and small size before mainnet.
- Keep `MM_ENABLED` as a kill switch.
- Keep risk limits (`MM_MAX_*`) conservative during bring-up.

## Testing

```bash
PYTHONPATH=src pytest tests/ -q
```

## License

MIT. See `LICENSE`.
