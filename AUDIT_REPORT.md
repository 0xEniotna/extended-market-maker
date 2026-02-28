# Market Maker Comprehensive Audit Report

**Date**: 2026-02-27
**Codebase**: extended-market-maker (Python async market maker for Extended Exchange / Starknet)
**Target**: Raspberry Pi deployment, solo operator, 3 monitoring agents

---

## Executive Summary

The codebase is significantly more mature than a typical solo-operator project. The market
making math (Avellaneda-Stoikov skew, markout calculation, funding bias) is **correct**.
However, I found **1 critical bug** that will gradually brick quoting slots during normal
operation, several high-severity operational gaps, and meaningful performance concerns for
the Raspberry Pi target.

**Critical findings**:
- `sweep_pending_cancels()` does not fire `_level_freed_callbacks`, permanently blocking
  level slots when WS terminal events are missed (CRITICAL)
- `os.fsync()` in journal writes is synchronous and blocks the entire event loop — up to
  200ms on an SD card (HIGH)
- SIGHUP config reload does not guard against changes to `market_name`, `environment`, or
  risk limits, leaving the bot in a silently broken state (HIGH)
- Drawdown stop resets on position close, failing to catch cumulative losses across
  position lifecycles (HIGH)

---

## 1. MARKET MAKING CORRECTNESS

### 1.1 Inventory Skew (Avellaneda-Stoikov Style)

**Severity**: OK — No issues found
**Location**: `pricing_engine.py:95-128` (`_skew_component`) and `pricing_engine.py:130-163`
(`compute_target_price`)

The sign logic is correct. When long (inventory_norm > 0):
- `skew_bps > 0` → `skew_offset > 0`
- BUY: `raw = best_price - offset - skew_offset` → lower bid (less aggressive buying) ✓
- SELL: `raw = best_price + offset - skew_offset` → lower ask (more aggressive selling) ✓

This matches the A-S reservation price: `r = s - q·γ·σ²`. When long, both quotes shift
downward to shed inventory. The deadband, tanh shaping, and inventory band amplification
(1.25x at WARN+) are all sound.

**One subtlety**: The `skew_max_bps * inventory_skew_factor` multiplication at line 118-120
means both parameters act as multipliers. If an advisor agent sets both independently, the
effective skew can be multiplicatively larger than intended. Consider documenting that the
effective max skew in bps is `skew_max_bps × inventory_skew_factor`.

### 1.2 Dynamic Mode Spread Multiplier

**Severity**: OK — Geometry is correct
**Location**: `pricing_engine.py:62-72`

```python
per_level_bps = spread_bps * spread_multiplier * Decimal(level + 1)
```

With `spread_multiplier=1.0`, the offset equals one full spread width from the same-side
BBO. Since the BBO IS the edge of the spread, this places quotes one spread width behind
it — not "at the edge." A multiplier of `0` would quote at the BBO.

**Issue**: The config documentation may mislead operators. `spread_multiplier=0.5` places
you half a spread behind the BBO, not "at half the spread." The geometry is correct for
market making — any positive multiplier produces quotes wider than the natural spread.

The floor/ceiling clamping (`min_offset_bps` / `max_offset_bps`) per-level is correctly
applied, preventing quotes from collapsing to zero spread during tight markets.

### 1.3 Markout Calculation

**Severity**: OK — Sign-correct for both sides
**Location**: `scripts/analyse_mm_journal.py:223-257` and `src/market_maker/fill_quality.py:129-153`

Both implementations use the same convention:
- BUY markout = `(future_mid - fill_price) / fill_price × 10000`
- SELL markout = `(fill_price - future_mid) / fill_price × 10000`

Positive markout = favorable (price moved in our favor after fill). Negative = adverse
selection. This is consistent between the offline analysis script and the live
`FillQualityTracker`. ✓

### 1.4 Post-Only Failed (POF) Adaptive Logic

**Severity**: OK — Sound with one minor observation
**Location**: `post_only_safety.py`

The adaptive logic: on rejection → increment streak, exponential cooldown (capped at 120s),
dynamic safety ticks increase. On success → halve streak, full reset after 3 consecutive
successes. The cooldown formula `base × backoff^streak` with cap is standard.

**Minor**: The success decay uses `min(streak // 2, streak - 2)` which takes the FASTER
decay path. For streak=10: `min(5, 8) = 5`. This is fine — aggressive reset after
conditions normalize.

### 1.5 Trend Signal EMA

**Severity**: LOW — Irregular sampling bias
**Location**: `trend_signal.py`

The EMA uses standard `alpha = 2/(N+1)` and computes over `mids[-slow_n:]`. The strength
normalization by micro-vol (`abs_bps / ref_vol`) is appropriate — it makes the
`trend_strong_threshold` comparable across volatility regimes.

**Issue**: The EMA treats each mid-price sample as equally spaced in time. If the orderbook
updates at irregular intervals (common in low-liquidity perpetuals on Starknet), the EMA
will be biased toward periods with more frequent updates. This introduces a subtle timing
bias — fast markets are overweighted relative to quiet periods.

**Fix**: For a Pi deployment this is acceptable. To fix properly, use a time-weighted EMA
where `alpha_effective = 1 - exp(-dt / halflife)` for each observation.

### 1.6 Funding Bias Direction

**Severity**: OK — Direction is correct
**Location**: `strategy.py` (`_funding_bias_bps`) and `pricing_engine.py:147-153`

When `funding_rate > 0` (longs pay shorts):
- `funding_bias_bps > 0` → `funding_offset > 0`
- BUY: `raw = best - offset - funding_offset` → lower bid (discourages going long) ✓
- SELL: `raw = best + offset - funding_offset` → lower ask (encourages selling/going short) ✓

This correctly incentivizes the bot to hold the side that receives funding payments. ✓

### 1.7 Inventory Band Effects

**Severity**: OK — Correct progressive protection
**Location**: `reprice_pipeline.py:580-603` and `pricing_engine.py:126-127`

- NORMAL: no modification
- WARN: +25% max skew amplification
- CRITICAL/HARD: cancel inventory-increasing orders, +25% skew on remaining
- Inventory-reducing orders override imbalance pauses and trend blocks

All correctly implemented.

### 1.8 Reprice Tolerance Logic

**Severity**: OK — Correctly scales with offset
**Location**: `reprice_pipeline.py:43-99`

`max_deviation = target_offset × tolerance_percent`. The tolerance is a fraction of the
offset itself, so it scales naturally — wide offsets get wider tolerance bands, preventing
unnecessary reprice churn. The additional tick gate (`min_reprice_move_ticks`) and edge
delta gate prevent micro-oscillation. ✓

---

## 2. ARCHITECTURE & COMPLEXITY

### 2.1 God Object: strategy.py (1169 lines)

**Severity**: MEDIUM — Technical debt, not a correctness issue
**Location**: `src/market_maker/strategy.py`

`MarketMakerStrategy` holds too many responsibilities:
- Level task lifecycle management
- Fill/order callbacks (partially moved to `strategy_callbacks.py`)
- Circuit breaker monitoring
- Drawdown watchdog
- Margin guard
- KPI watchdog
- Position/balance refresh scheduling
- Funding rate refresh
- Quote halt management
- Config reload handling
- Signal handling

**Fix**: Extract these into focused classes:
1. `QuoteHaltManager` — halt/resume logic, sync state
2. `FundingManager` — funding rate refresh + bias calculation
3. Move circuit breaker, drawdown watchdog, margin guard into `RiskManager` or a
   `RiskWatchdog` class
4. Keep `MarketMakerStrategy` as the thin orchestrator that wires components together

### 2.2 RepricePipeline Mixed Concerns

**Severity**: MEDIUM
**Location**: `src/market_maker/reprice_pipeline.py`

`RepricePipeline.evaluate()` takes `strategy` as a parameter and directly accesses:
- `strategy._level_ext_ids`
- `strategy._level_cancel_pending_ext_id`
- `strategy._orders`
- `strategy._ob`
- `strategy._pricing`
- `strategy._risk`
- `strategy._guards`
- `strategy._journal`

This is a "friend class" pattern that creates tight coupling. The pipeline is essentially
a method of the strategy with its own file.

**Fix**: Define a `StrategyContext` protocol/interface that exposes only the needed
accessors. The pipeline should depend on this protocol, not the concrete strategy.

### 2.3 Double-Tracking: `_level_ext_ids` + `OrderManager._active_orders`

**Severity**: LOW — Necessary two-tier lookup, not a bug
**Location**: `strategy.py:106` and `order_manager.py:116`

- `_level_ext_ids[(side, level)] → ext_id`: Logical slot → current order
- `_active_orders[ext_id] → OrderInfo`: Order ID → details

These serve different purposes. The risk of desync is managed: when `_level_ext_ids`
points to an ext_id not in `_active_orders`, `get_active_order()` returns None, triggering
a fresh quote. The `_clear_level_slot()` method atomically resets all tracking for a slot.

### 2.4 Cancel-Barrier Race Condition — CRITICAL BUG

**Severity**: CRITICAL — Permanently bricks quoting slots
**Location**: `order_manager.py:732-759` (`sweep_pending_cancels`) and
`strategy.py:292-294`

**The bug**: `sweep_pending_cancels()` force-removes orders from `_active_orders` and
`_pending_cancel` after a 10-second timeout, but it does NOT fire
`_level_freed_callbacks`. This means `on_level_freed()` is never called, so
`_clear_level_slot()` is never called, so `_level_cancel_pending_ext_id[key]` stays set.

The level task at `strategy.py:292` then spins forever:
```python
if self._level_cancel_pending_ext_id.get(key) is not None:
    await asyncio.sleep(0.1)
    continue
```

**Impact**: On a Raspberry Pi with occasional WebSocket disconnections, each missed
terminal event permanently disables one (side, level) slot. Over hours/days of operation,
all slots can be bricked, leaving the bot with zero quoting capacity while appearing
healthy in logs.

**Fix**: `sweep_pending_cancels()` must fire `_level_freed_callbacks` for each
force-removed order, exactly like `handle_order_update()` does for terminal orders. See
the attached code fix below.

### 2.5 Asyncio Anti-patterns

**Severity**: LOW
**Location**: `strategy.py` init via `_run_provenance()`

`subprocess.run(["git", "rev-parse", "HEAD"], ...)` is a blocking call. It runs only at
startup (not in the hot path), but it blocks the event loop for ~50-200ms on a Pi. Use
`asyncio.create_subprocess_exec()` or move to a pre-init step.

---

## 3. PERFORMANCE (Raspberry Pi Constraints)

### 3.1 Decimal Arithmetic in Hot Path

**Severity**: MEDIUM — ~10-50x slower than float on every reprice evaluation
**Location**: `pricing_engine.py`, `reprice_pipeline.py`

Every book update triggers `compute_target_price()` and `needs_reprice()` for each
(side, level) slot. With 2 levels per side, that is 4 evaluations per book tick. Each
evaluation performs ~20+ Decimal operations:
- `compute_offset`: 5 Decimal ops
- `_skew_component`: 12+ Decimal ops (includes `math.tanh` via float conversion)
- `compute_target_price`: 6+ Decimal ops
- `needs_reprice`: 8+ Decimal ops

On a Raspberry Pi (ARM Cortex-A72), Decimal arithmetic is ~10-50x slower than float.
With 10 book updates/second and 4 levels, that is 40 Decimal-heavy evaluations/second.

**Fix**: Convert the pricing hot path to float arithmetic. The precision requirements for
quote pricing (~4-6 significant digits) are well within float's 15-digit mantissa.
Reserve Decimal for order submission (final tick rounding) and risk calculations.

### 3.2 Mid-History Deque Pruning

**Severity**: LOW — Acceptable for Pi
**Location**: `orderbook_manager.py`

The `mid_history` deque prunes on every bid/ask change. With 120s of data at typical
update frequencies (10 updates/s), this is ~1200 entries × ~108 bytes = ~130KB. Acceptable.

During volatile periods with rapid updates, this could grow to ~12,000 entries (~1.3MB).
Still fine. The `popleft()` pruning on every change is O(1) for a deque.

### 3.3 Analysis Script Memory

**Severity**: LOW — Offline only
**Location**: `scripts/analyse_mm_journal.py`

`load_journal()` reads entire JSONL into memory. This is an offline analysis script, not
in the hot path. For a 50MB journal (~500k events), this uses ~200MB RAM. Fine for
offline use. If the script ever runs as a monitoring sidecar on the Pi, it would need
streaming/incremental processing.

### 3.4 Markout Bisect Lookups

**Severity**: LOW — Acceptable for analysis
**Location**: `scripts/analyse_mm_journal.py:192-209`

`_build_mid_timeline()` does O(log n) bisect lookups per fill per horizon. For 1000 fills
× 5 horizons × O(log 10000) = ~65,000 operations. Runs in <1 second even on a Pi.

---

## 4. RISK MANAGEMENT GAPS

### 4.1 `allowed_order_size()` Clipping Order

**Severity**: OK — Order does not affect correctness
**Location**: `risk_manager.py:344-483`

The 6 clipping layers are all `min()` operations applied sequentially:
1. Per-side position size headroom
2. Per-order notional cap
3. Total position notional cap
4. Gross exposure limit
5. Reducing/opening split
6. Balance-aware sizing (opening qty only)

Since each clip is a `min()`, the order of operations does not matter for final size.
The only ordering dependency is the reducing/opening split (step 5) which correctly
happens after the gross exposure clip.

### 4.2 Stale Balance Handling

**Severity**: MEDIUM
**Location**: `risk_manager.py:436-438`

When balance is stale, balance-aware sizing is **skipped entirely** — the order sizes
based on position/notional limits alone. This is a fail-open design.

**Risk**: If the account stream stops delivering balance updates (WS issue), the bot
continues sizing orders without balance awareness. On a small Pi account, this could
lead to oversized orders relative to available collateral.

**Fix**: Add a `balance_stale_action` config option: `skip` (current behavior),
`reduce` (halve order size), or `halt` (zero order size). Default to `reduce` for
production.

### 4.3 Drawdown Stop Resets on Position Close

**Severity**: HIGH — Fails to catch cumulative losses
**Location**: `risk_manager.py:107-109` and `drawdown_stop.py`

When a position is closed (`PositionStatus.CLOSED`), `_reset_position_pnl()` zeroes all
PnL caches. `DrawdownStop.evaluate()` then sees PnL=0 → no drawdown → no trigger.

**Impact**: If the bot opens a position, loses $50, the position gets liquidated or
flattened (PnL resets to 0), then opens a new position and loses another $50 — the
drawdown stop never fires. Each individual position lifecycle looks fine, but the
cumulative session loss is $100.

**Fix**: Track cumulative session P&L separately from position P&L. Add a
`_session_realized_pnl` accumulator that sums realized P&L across position lifecycles.
The drawdown stop should evaluate `session_realized_pnl + current_unrealized_pnl`.

### 4.4 Margin Guard Defaults

**Severity**: LOW
**Location**: `config.py`

`min_available_balance_for_trading = 100` (USD) is reasonable for mainnet but will
constantly trip on testnet with small balances (typical testnet balance: $100-$1000). The
margin guard will halt quoting immediately.

**Fix**: Consider a relative threshold (`min_available_balance_ratio`) as the primary
guard, with the absolute floor as a secondary safety. This scales across account sizes.
The `min_available_balance_ratio` config already exists but defaults to 0 (disabled).

### 4.5 `flatten_position()` One-Sided Book

**Severity**: OK — Fallback logic is correct and safe
**Location**: `order_manager.py:765-972`

The three-tier fallback (natural BBO → opposite BBO → last known mid with 50bps slippage)
is well-designed. The progressive slippage on retries (`base + step × (attempt-1)`,
capped at `max_bps`) is correct. The `reduce_only=True` flag prevents the flatten from
accidentally opening a position on the opposite side. ✓

### 4.6 Position Notional Cap Direction Ambiguity

**Severity**: LOW
**Location**: `risk_manager.py:400-415`

`current_notional = abs(current) * reference_price` uses the absolute position for the
notional cap. This means if we are short -100 and trying to sell more, the notional
check counts our existing |100| notional against the cap. This is correct — selling more
increases our position magnitude, which should count against the position notional cap.

However, if we are short -100 and trying to **buy** (reduce), the per-side headroom
check at step 1 correctly handles this as a reducing order. The notional cap may still
over-restrict buys that would reduce our position. The `_split_reducing_and_opening_qty`
at step 5 partially mitigates this by exempting reducing quantity from balance-aware
sizing, but not from the notional cap at step 3.

**Fix**: Skip position notional cap check for reducing orders. This requires reordering
the clips to put the reducing/opening split earlier.

---

## 5. OPERATIONAL RELIABILITY

### 5.1 Account Stream Fail-Safe / Cancel Storm

**Severity**: MEDIUM
**Location**: `account_stream.py`

On sequence gap: `cancel_all_orders()` → raise RuntimeError → reconnect with backoff.
The cancel is a single mass_cancel API call, not per-order. On reconnect, seq resets.

**Risk**: If the stream oscillates rapidly (connect → gap → disconnect → connect → gap),
each cycle triggers a mass cancel. At 10 reconnects/minute, that is 10 mass-cancel API
calls — unlikely to be rate-limited but will cause extended periods with no quotes.

**Mitigation**: The existing exponential backoff on reconnect bounds this. With 1s initial
backoff and 2x multiplier, reconnect attempts are: 1s, 2s, 4s, 8s... At 6+ reconnects,
the backoff exceeds 30s, naturally throttling the cancel storm.

### 5.2 Dead-Man Switch Timing

**Severity**: OK — Intentional and safe
**Location**: `strategy_runner.py:434-469` and `strategy_runner.py:774-790`

- Heartbeat interval: 20s (configurable)
- DMS countdown: 60s (configurable)
- Worst-case exposure after crash: 20s (last heartbeat) + 60s (countdown) = 80s

`set_deadman_switch(0)` on clean shutdown disarms it. If the process crashes, the DMS
fires after countdown, cancelling all orders on the exchange. This is the correct
design for an exchange-side safety mechanism.

**Recommendation**: For a Pi (power loss risk), consider reducing countdown to 30s and
heartbeat to 10s. This reduces worst-case exposure to 40s at the cost of more API calls.

### 5.3 `_level_task` Catch-All Exception Handling

**Severity**: MEDIUM
**Location**: `strategy.py:305-316`

```python
except Exception as exc:
    logger.error(...)
    await asyncio.sleep(1.0)
```

All exceptions cause a 1-second sleep and retry. This includes irrecoverable errors
(credential errors, API key revoked, market delisted) that will never succeed.

**Fix**: Classify exceptions:
- `AuthenticationError`, `PermissionError` → trigger shutdown
- `RateLimitException` → already handled upstream
- Network errors → retry (current behavior)
- Everything else → increment failure counter, let circuit breaker decide

### 5.4 Journal `os.fsync()` Blocks Event Loop

**Severity**: HIGH — Stalls all async operations during write
**Location**: `trade_journal.py:125-133`

`_do_fsync()` calls `os.fsync(fd)` synchronously. On an SD card:
- Typical latency: 10-50ms
- Worst case (wear leveling, garbage collection): 100-500ms

During this time, the entire event loop is blocked. No book updates, no order management,
no WS processing. For critical events (fills, drawdown stops), this happens immediately.
The batched fsync (every 100 writes or 10s) mitigates frequency but not per-call latency.

**Fix**: Run fsync in a thread executor:
```python
async def _do_fsync_async(self) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, self._do_fsync_sync)
```
Or better: accept the risk of losing the last few non-critical events on power loss and
only fsync on shutdown. Fills are reconstructable from exchange history.

### 5.5 SIGHUP Reload Does Not Guard Critical Parameters

**Severity**: HIGH — Silently broken state
**Location**: `strategy.py:147-169` (`_handle_reload`) and `strategy.py:710-762`
(`_rebuild_components`)

`_rebuild_components()` recreates: PricingEngine, PostOnlySafety, VolatilityRegime,
TrendSignal, GuardPolicy, RepricePipeline, DrawdownStop, FillQualityTracker.

It does NOT touch: OrderbookManager (still subscribed to original market),
AccountStreamManager (still tracking original market), OrderManager (still using original
trading client), RiskManager (still using original position/notional limits).

**Impact**: If `MM_MARKET_NAME` changes via SIGHUP, the bot quotes the OLD market with
parameters tuned for the NEW market. If `MM_MAX_POSITION_SIZE` changes, the RiskManager
still enforces the original limit.

**Fix**: Add a guard in `_handle_reload`:
```python
IMMUTABLE_KEYS = {"market_name", "environment", "vault_id", "api_key", "stark_private_key"}
for key in IMMUTABLE_KEYS:
    if getattr(new_settings, key) != getattr(self._settings, key):
        logger.error("Cannot change %s via SIGHUP — restart required", key)
        return
```
Also propagate changed risk limits to `RiskManager` or document that risk parameter
changes require a restart.

---

## 6. CODE QUALITY & SIMPLIFICATION

### 6.1 `strategy_runner.py` (915 lines)

**Severity**: MEDIUM
**Location**: `src/market_maker/strategy_runner.py`

The `RuntimeContext` dataclass and all `_build_*` functions form an ad-hoc factory.
Extract into `StrategyFactory` with clear phases: validate → build components → wire
callbacks → start services.

### 6.2 `analyse_mm_journal.py` Duplicate Markout Logic

**Severity**: LOW
**Location**: `scripts/analyse_mm_journal.py`

`build_summary()` (line 355) and `analyse()` (line 444) both independently compute
markouts over the same fills. They use slightly different code paths (`_compute_markouts`
vs inline loop). Consolidate into a shared `MarkoutResult` computed once and consumed by
both functions.

### 6.3 `_to_decimal()` / `_d()` Duplication

**Severity**: LOW
**Location**: `pricing_engine.py:29-34`, `analyse_mm_journal.py`, and others

The pattern `Decimal(str(value))` with fallback is repeated in multiple files. Extract to
`src/market_maker/utils.py`:
```python
def to_decimal(value, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)
```

### 6.4 `config.py` (961 lines)

**Severity**: LOW
**Location**: `src/market_maker/config.py`

200+ Field() definitions in a single class. Split into sub-configs:
- `RiskConfig` — position limits, margin guard, drawdown
- `PricingConfig` — offset mode, spread multiplier, skew
- `TrendConfig` — EMA parameters, thresholds
- `OperationalConfig` — deadman, circuit breaker, journal

Use pydantic model composition. This also makes it easier for advisor agents to validate
parameter groups.

### 6.5 `bounds.json` Provides Zero Validation

**Severity**: MEDIUM — False sense of safety
**Location**: `mm_config/policy/bounds.json`

Every float parameter has `"min": -1000000000.0, "max": 1000000000.0`. This means:
- `MM_SPREAD_MULTIPLIER = -999999999` passes validation
- `MM_DRAWDOWN_STOP_PCT_OF_MAX_NOTIONAL = -999999999` passes validation
- `MM_CIRCUIT_BREAKER_COOLDOWN_S = -999999999` passes validation

An advisor agent could set catastrophically wrong values and pass bounds checking.

**Fix**: Set meaningful bounds. Examples:
```json
"MM_SPREAD_MULTIPLIER": {"type": "float", "min": 0.01, "max": 100.0},
"MM_DRAWDOWN_STOP_PCT_OF_MAX_NOTIONAL": {"type": "float", "min": 0.001, "max": 1.0},
"MM_CIRCUIT_BREAKER_COOLDOWN_S": {"type": "float", "min": 1.0, "max": 3600.0},
"MM_MIN_OFFSET_BPS": {"type": "float", "min": 0.0, "max": 500.0},
"MM_MAX_OFFSET_BPS": {"type": "float", "min": 0.0, "max": 5000.0},
"MM_INVENTORY_SKEW_FACTOR": {"type": "float", "min": 0.0, "max": 10.0}
```

### 6.6 Policy Files Drift from config.py

**Severity**: MEDIUM
**Location**: `mm_config/policy/` vs `src/market_maker/config.py`

`bounds.json`, `whitelist.json`, and `protected_keys.json` are maintained separately from
the pydantic model in `config.py`. When a new config field is added to `config.py`, the
policy files must be manually updated or the advisor agents will silently ignore the new
parameter.

**Fix**: Generate `bounds.json` and `whitelist.json` from the pydantic model at build time:
```python
# scripts/generate_policy.py
from market_maker.config import MarketMakerSettings
for field_name, field_info in MarketMakerSettings.model_fields.items():
    # Extract type, min/max from field_info.metadata
```

---

## 7. MISSING INSTITUTIONAL FEATURES (Top 5)

### 7.1 Per-Session P&L Attribution

**Priority**: HIGH
**Current state**: The journal records fills and the analysis script computes gross P&L,
but there is no real-time decomposition into:
- **Spread capture** (half-spread earned on each maker fill pair)
- **Inventory P&L** (mark-to-market on held inventory)
- **Fee P&L** (maker rebates minus taker fees)
- **Funding P&L** (funding payments received/paid)

**Why it matters**: Without attribution, you cannot determine whether losses come from
adverse selection (widen spreads), inventory risk (reduce position limits), or fees
(change fee tier). You are flying blind on the levers that matter.

### 7.2 Quote-to-Trade Ratio Monitoring

**Priority**: HIGH
**Current state**: No tracking. The `fill_rate_pct` in the analysis script is
fills/placements, not the exchange-mandated quote-to-trade ratio.

**Why it matters**: Many exchanges (including those on Starknet L2) impose or will impose
quote-to-trade ratio limits for market makers. Exceeding the limit can result in API
throttling, fee penalties, or account suspension. The current repricing frequency (every
book tick) likely generates a very high quote-to-trade ratio.

### 7.3 Kill Switch with Exchange-Side Hard Position Limit

**Priority**: HIGH
**Current state**: The dead-man switch cancels orders on process death, but there is no
exchange-side enforcement of position limits. All limits are local. If the local process
has a bug that ignores risk limits, positions can grow unbounded.

**Why it matters**: A software bug, memory corruption, or cosmic ray bit-flip on a Pi
could bypass local risk checks. Exchange-side position limits are the last line of defense.
Check if Extended Exchange supports configurable position limits per API key.

### 7.4 Venue Connectivity Monitoring with Latency SLAs

**Priority**: MEDIUM
**Current state**: `avg_placement_latency_ms()` tracks rolling order latency, but there
are no SLA thresholds, alerting, or adaptive behavior based on latency.

**Why it matters**: On a Pi with WiFi, network latency can spike from 50ms to 2000ms
during interference. During high-latency periods, the bot's quotes are stale by the time
they reach the exchange, increasing adverse selection. The bot should widen spreads or
halt quoting when latency exceeds a threshold.

### 7.5 Audit Trail for Config Changes with Rollback

**Priority**: MEDIUM
**Current state**: Config changes are journaled, and the advisor pipeline generates
proposals with receipts. But there is no automated rollback mechanism that can revert to
a known-good config if performance degrades after a change.

**Why it matters**: With 3 advisor agents modifying config, a bad parameter change could
go unnoticed for hours. An automated "watchdog rollback" that reverts to the last
known-good config when key metrics (markout, fill rate, P&L) degrade would prevent
prolonged losses from misconfiguration.

---

## CRITICAL BUG FIX

### `sweep_pending_cancels` must fire level-freed callbacks

The fix adds callback firing to `sweep_pending_cancels()` in `order_manager.py`, matching
the behavior of `handle_order_update()` for terminal orders:

**File**: `src/market_maker/order_manager.py`, method `sweep_pending_cancels`

After force-removing an order, fire `_level_freed_callbacks` so that the strategy's
`_level_cancel_pending_ext_id` gets cleared via `_clear_level_slot()`.

---

## PRIORITIZED REFACTOR ROADMAP

### Phase 1: Immediate (stops losing money / prevents bricking)
1. **Fix `sweep_pending_cancels` callback bug** — CRITICAL, ~20 lines changed
2. **Add SIGHUP immutable-key guard** — HIGH, ~15 lines
3. **Add cumulative session P&L tracking for drawdown stop** — HIGH, ~50 lines
4. **Set meaningful bounds.json values** — MEDIUM, config-only change

### Phase 2: Short-term (operational reliability on Pi)
5. **Make journal fsync async** (thread executor) — HIGH, ~30 lines
6. **Classify exceptions in `_level_task`** — MEDIUM, ~20 lines
7. **Add stale-balance order reduction** — MEDIUM, ~15 lines
8. **Reduce DMS countdown to 30s, heartbeat to 10s** — LOW, config change

### Phase 3: Medium-term (code quality)
9. **Extract `QuoteHaltManager` from strategy.py** — MEDIUM
10. **Define `StrategyContext` protocol for RepricePipeline** — MEDIUM
11. **Split config.py into sub-configs** — LOW
12. **Generate policy files from pydantic model** — LOW
13. **Convert pricing hot path to float** — MEDIUM (perf gain on Pi)

### Phase 4: Institutional features
14. **Per-session P&L attribution** — HIGH value
15. **Quote-to-trade ratio monitoring** — HIGH value
16. **Venue latency SLA monitoring** — MEDIUM value
17. **Automated config rollback watchdog** — MEDIUM value

---

## FILE DISPOSITION

| Action | File | Reason |
|--------|------|--------|
| **Split** | `strategy.py` (1169 lines) | Extract QuoteHaltManager, FundingManager |
| **Split** | `strategy_runner.py` (915 lines) | Extract StrategyFactory class |
| **Split** | `config.py` (961 lines) | Sub-configs: Risk, Pricing, Trend, Ops |
| **Merge** | `drawdown_stop.py` (74 lines) | Into risk_manager.py (same domain) |
| **Create** | `src/market_maker/utils.py` | Shared `to_decimal()`, common helpers |
| **Refactor** | `reprice_pipeline.py` | Depend on StrategyContext protocol |
| **Auto-generate** | `mm_config/policy/bounds.json` | From pydantic model metadata |
| **Auto-generate** | `mm_config/policy/whitelist.json` | From pydantic model metadata |

---

## FUNDAMENTAL MARKET MAKING LOGIC ERRORS

**None found.** The core market making math is correct:
- Avellaneda-Stoikov inventory skew: correct sign, correct direction
- Markout calculation: sign-correct for both sides
- Funding bias: correct direction (shorts incentivized when longs pay)
- Dynamic offset geometry: correctly places quotes outside the natural spread
- Risk clipping: correct min() chain, order-independent

The codebase has sound quantitative foundations. The issues are operational (the
sweep_pending_cancels bug, fsync blocking, config reload safety) rather than mathematical.
