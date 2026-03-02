# extended-market-maker

Standalone market-making engine for Extended Exchange.

## Architecture

Core runtime flow:

1. `OrderbookManager` streams best bid/ask and microstructure signals.
2. `MarketMakerStrategy` computes target quotes per `(side, level)`.
3. `RiskManager` clips order size to position and notional limits.
4. `OrderManager` places/cancels post-only orders and tracks lifecycle.
5. `AccountStreamManager` consumes fills/order updates/positions.
6. `TradeJournal` records structured JSONL events.

## Setup

```bash
git clone git@github.com:0xEniotna/extended-market-maker.git
cd extended-market-maker
./setup.sh
source .venv/bin/activate
cp .env.example .env
# Fill MM credentials and market parameters
```

Manual setup (if you prefer):

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
pip install -e python_sdk       # x10 SDK (git submodule)
pip install -e ".[dev]"         # MM engine + dev tools
```

## CLI (`mmctl`)

All operations go through `mmctl`. Every command supports `--json` for agent/bot consumption.

### Instance management

```bash
mmctl start eth                    # Start MM instance
mmctl stop eth                     # Stop (SIGINT -> SIGTERM)
mmctl restart eth                  # Stop + start
mmctl status --json                # All running instances
mmctl status eth --json            # One instance
```

### PnL

```bash
mmctl pnl ETH-USD --days 7 --json           # Per-market PnL
mmctl pnl --all --since 2026-01-01 --json   # Account-wide PnL + APR/APY
mmctl pnl --scorecard --json                # Daily fleet scorecard
```

### Positions & risk

```bash
mmctl positions --json                      # Position risk across all markets
mmctl close ETH-USD --dry-run --json        # Preview flatten
mmctl close ETH-USD --json                  # Execute flatten
```

### Markets

```bash
mmctl markets info ETH-USD --json           # Trading config and stats
mmctl markets find --json                   # Find MM candidates (rolling spread)
mmctl markets screen --json                 # Screen markets for suitability
```

### Journal analysis

```bash
mmctl journal analyze                       # Analyze latest journal
mmctl journal export --market ETH-USD       # Inventory CSV export
mmctl journal reprice-quality               # Reprice quality audit
```

### Config proposals

```bash
mmctl config apply <proposal-id> --json     # Apply a config proposal
mmctl config rollback MON --to <snapshot>   # Rollback env file
mmctl config diff <proposal-id>             # Show diff without applying
```

## Advisor & Scout Workflow

Advisory scripts (recommend-only, never mutate env or restart bots):

```bash
# Advisor loop (generates config proposals)
.venv/bin/python scripts/mm_advisor_loop.py

# Approval-gated proposal apply
.venv/bin/python scripts/mm_advisor_apply.py --proposal-id <id> --approve --json

# Market scout pipeline
.venv/bin/python scripts/tools/market_scout_pipeline.py

# Auditor decisioning
.venv/bin/python scripts/tools/auditor_apply_scout.py --print-target auditor
.venv/bin/python scripts/tools/auditor_followup.py --print-target auditor
```

## Testing

```bash
pytest tests/ -q
```

## Safety Notes

- Never commit real credentials.
- Start on testnet and small size before mainnet.
- Keep `MM_ENABLED` as a kill switch.
- Keep risk limits (`MM_MAX_*`) conservative during bring-up.

## License

MIT. See `LICENSE`.
