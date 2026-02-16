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
- `scripts/mm_autotune_loop.py`: run-analyze-adjust loop using bounded MM\_\* key updates.
- `scripts/mm_openclaw_controller.sh`: single-instance long-running autotune controller.
- `scripts/mm_openclaw_fleet.sh`: run multiple autotune controllers (one per `.env.*` instance).
- `scripts/screen_mm_markets.py`: spread/tick/volume suitability screening.
- `scripts/tools/find_mm_markets.py`: rolling market filter for MM candidates.
- `scripts/tools/fetch_market_info.py`: inspect trading config and stats for one market.
- `scripts/tools/fetch_pnl.py`: account-level market PnL summary.
- `scripts/tools/analyze_mm_logs.py`: parse text logs for lifecycle and latency diagnostics.

## Multi-Instance Supervision

Run one controller per strategy instance:

```bash
cd /path/to/repo
scripts/mm_openclaw_fleet.sh start .env.asset
scripts/mm_openclaw_fleet.sh status .env.asset
scripts/mm_openclaw_fleet.sh logs .env.asset
```

Each controller monitors only its own market journal and will stop/tune/restart only that instance when degradation is detected.  
On shutdown/restart, the runtime now attempts to flatten that market's open position (reduce-only `MARKET+IOC`) after cancelling resting orders.

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
