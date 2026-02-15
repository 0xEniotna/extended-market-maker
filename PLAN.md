# Extended Market Maker — Crypto Strategy Analysis & Refactoring Plan

## Part 1: Why It Doesn't Work for Crypto Pairs

### 1.1 Volatility Regime Mismatch (Critical)

The toxicity guard defaults are calibrated for low-volatility markets (stablecoins, tight perps):

- `micro_vol_max_bps = 8` — crypto pairs routinely move 20-100+ bps in 5 seconds
- `micro_drift_max_bps = 6` — any normal crypto price movement exceeds this
- Hard pause triggers at 1.25x (10 bps vol, 7.5 bps drift) — in crypto this fires *constantly*

**Effect**: The bot spends most of its time in toxicity-pause, cancelling all resting orders. It quotes for brief windows between normal crypto price movements, making almost no fills.

**Where**: `strategy.py:1095-1126` (`_toxicity_adjustment`), `config.py:336-365`

### 1.2 Inventory Skew Too Weak for Crypto Dynamics (Critical)

Maximum effective skew = `skew_max_bps × inventory_skew_factor` = 20 × 0.5 = **10 bps**.

In a market that can move 50-200 bps in seconds:
- The skew can only adjust quotes by 10 bps at the absolute maximum (when at position limit)
- This is negligible compared to the price moves that caused the inventory buildup
- Inventory accumulates on the wrong side during trends far faster than skew can unwind it

**Where**: `strategy.py:1189-1216`, `config.py:293-324`

### 1.3 No Volatility Regime Adaptation (Critical)

All thresholds are **static**. Crypto alternates between:
- **Low-vol regimes** (consolidation): 2-5 bps/s range — current defaults work here
- **High-vol regimes** (breakouts, news): 50-500 bps/s range — current defaults shut the bot down

There is no mechanism to:
- Scale offsets proportionally to realized volatility
- Widen thresholds in high-vol periods (where the biggest MM profits are)
- Tighten in low-vol (where competition for fills is highest)

The `volatility_offset_multiplier` (0.35) adds a fraction of excess vol as extra offset, but the hard-pause at 1.25x the limit means it never gets used in real crypto conditions.

**Where**: `strategy.py:1095-1126`, `orderbook_manager.py:205-227`

### 1.4 No Trend Detection or Directional Bias (Major)

The Avellaneda-Stoikov framework assumes **symmetric order arrival** — buy and sell market orders arrive with equal probability. This is fundamentally wrong for crypto:

- Crypto trends strongly for extended periods (hours/days)
- During a trend, the MM accumulates inventory against the trend
- The gentle tanh skew curve takes too long to respond
- By the time skew reaches maximum, the position is already at risk limits

The `micro_drift_max_bps` guard detects *rapid* drift but not sustained directional movement. A steady 2 bps/s trend (12 bps in 6 seconds) is below the guard but will steadily fill one side.

**Where**: `strategy.py:1176-1238` (no trend signal in target price), `orderbook_manager.py:218-227`

### 1.5 No Funding Rate Awareness (Major)

Perpetual futures have funding rates that materially affect inventory carrying cost:

- **Positive funding** (longs pay shorts): Being short earns income — can carry short inventory longer
- **Negative funding** (shorts pay longs): Being long earns income — can carry long inventory longer
- Current strategy treats all inventory as equally undesirable regardless of funding

For crypto perps, funding rates can be 0.01-0.1% per 8 hours (3.65-36.5% annualized). This is significant.

**Where**: Not implemented anywhere in the codebase.

### 1.6 Spread/Offset Assumptions Don't Scale (Moderate)

- **Fixed mode** (0.3%/level = 30 bps): Way too wide for liquid BTC-USD (typical spread 1-3 bps), reasonable for mid-cap alts
- **Dynamic mode**: Clamped between 3-100 bps. The 100 bps ceiling is too low for volatile alt-perps during events
- The EMA spread smoothing (alpha=0.15, ~7-sample half-life) lags during rapid spread regime changes

**Where**: `strategy.py:1147-1174`, `orderbook_manager.py:402-418`

### 1.7 Order Age and Reprice Churn (Moderate)

- `max_order_age_s = 15` forces constant repricing in naturally volatile markets
- `min_reprice_interval_s = 0.5` means up to 2 reprices/second per level
- With 2 levels × 2 sides = 4 tasks, this is up to 8 cancel+place per second
- On volatile crypto pairs this creates heavy API load and exchange rate-limit risk

**Where**: `strategy.py:765-796`, `config.py:259-266`

### 1.8 Missing Cross-Market Hedging (Moderate)

Single-market design means no awareness that:
- Most altcoins are 70-90% correlated with BTC
- Inventory in SOL-USD could be partially hedged with BTC-USD
- Running multiple independent MMs on correlated pairs compounds directional risk

**Where**: Architectural — the strategy only sees one market.

---

## Part 2: Strategy Alternatives for Crypto

### 2.1 Volatility-Adaptive Market Making (Recommended — Incremental)

**Concept**: Replace static thresholds with volatility-regime-scaled parameters.

- Compute realized volatility over multiple horizons (1m, 5m, 1h)
- Classify regime: LOW / MEDIUM / HIGH / EXTREME
- Scale offsets, skew intensity, position limits, and reprice intervals by regime
- In HIGH vol: wider spreads but still quoting (this is where MM profits are highest)
- In EXTREME vol: pause (genuine circuit breaker, not normal operation)

**Pros**: Minimal architectural change, directly addresses the #1 problem.
**Cons**: Requires volatility estimator calibration per-market.

### 2.2 Trend-Following Market Maker (Recommended — Moderate Change)

**Concept**: Add a trend signal that biases quoting direction.

- Compute short-term trend (EMA crossover, linear regression slope on 1-5m window)
- In uptrend: reduce bid sizes / increase ask sizes (don't fight the trend)
- In strong uptrend: pull bids entirely, only quote asks
- Trend signal feeds into both skew intensity and size asymmetry

**Pros**: Directly addresses inventory buildup during trends.
**Cons**: Risk of whipsaws in choppy markets; needs careful signal design.

### 2.3 Grid Trading with Dynamic Bounds (Alternative)

**Concept**: Replace continuous repricing with a grid of resting orders at fixed intervals.

- Place N buy orders below mid at intervals, N sell orders above
- When a grid level fills, place the opposite order at the next grid level
- Grid spacing adapts to realized volatility
- No constant repricing — orders rest until filled

**Pros**: Much simpler, less API churn, natural mean-reversion capture.
**Cons**: Poor in trending markets, needs wide grids for volatile pairs.

### 2.4 Funding-Rate-Aware Inventory Management (Complementary)

**Concept**: Adjust inventory tolerance based on funding rate.

- Fetch funding rate periodically (every 1-8 hours)
- If funding favors current position direction: relax skew, carry longer
- If funding punishes current position: increase skew aggressively
- Adjust max_position_size dynamically based on funding regime

**Pros**: Exploits a crypto-specific edge that most MMs ignore.
**Cons**: Funding data requires additional API integration.

### 2.5 Order-Flow-Informed MM (Advanced)

**Concept**: Use trade flow (not just top-of-book) to predict short-term direction.

- Track recent trade tape: volume-weighted buy/sell ratio
- Compute order flow toxicity (VPIN or similar metric)
- High-toxicity periods: widen or pull quotes
- Directional flow: skew quotes away from predicted direction

**Pros**: Most sophisticated adverse-selection protection.
**Cons**: Requires trade feed integration (not just orderbook), more complex.

---

## Part 3: Strategy File Refactoring Plan

The current `strategy.py` is 1,405 lines handling 8+ distinct responsibilities. Proposed decomposition:

### 3.1 New Module Structure

```
src/market_maker/
├── strategy.py              (~250 lines) Main orchestrator — just task management & lifecycle
├── pricing_engine.py        (~200 lines) Target price computation, offset calculation, inventory skew
├── reprice_pipeline.py       (~300 lines) Reprice gates, decision logic, tolerance checks
├── post_only_safety.py      (~120 lines) POST_ONLY safety clamping, adaptive POF streak tracking
├── toxicity_guard.py        (~100 lines) Volatility, drift, imbalance checks
├── volatility_regime.py     (~150 lines) NEW: Multi-horizon vol estimation, regime classification
├── trend_signal.py          (~100 lines) NEW: Trend detection for directional bias
├── order_manager.py         (existing, unchanged)
├── orderbook_manager.py     (existing, unchanged)
├── risk_manager.py          (existing, minor changes)
├── account_stream.py        (existing, unchanged)
├── config.py                (existing, extended with new params)
├── trade_journal.py         (existing, unchanged)
├── metrics.py               (existing, unchanged)
└── public_markets.py        (existing, unchanged)
```

### 3.2 Refactoring Steps (in order)

#### Step 1: Extract `PricingEngine` from strategy.py
Move these methods out of `MarketMakerStrategy`:
- `_compute_offset()` → `PricingEngine.compute_offset()`
- `_compute_target_price()` → `PricingEngine.compute_target_price()`
- `_round_to_tick()` → `PricingEngine.round_to_tick()`
- `_theoretical_edge_bps()` → `PricingEngine.theoretical_edge_bps()`
- `_level_size()` → `PricingEngine.level_size()`
- Inventory skew logic (lines 1189-1216) → `PricingEngine.inventory_skew()`

**Dependencies**: Needs `settings`, `orderbook_manager`, `risk_manager`, `tick_size`.

#### Step 2: Extract `PostOnlySafety` from strategy.py
Move these methods:
- `_apply_post_only_safety()` → `PostOnlySafety.clamp_price()`
- `_effective_safety_ticks()` → `PostOnlySafety.effective_ticks()`
- `_apply_adaptive_pof_reject()` → `PostOnlySafety.on_rejection()`
- `_on_successful_quote()` → `PostOnlySafety.on_success()`
- `_reset_pof_state()` → `PostOnlySafety.reset()`
- All `_level_pof_*` and `_level_dynamic_safety_ticks` state

**Dependencies**: Needs `settings`, `tick_size`.

#### Step 3: Extract `ToxicityGuard` from strategy.py
Move:
- `_toxicity_adjustment()` → `ToxicityGuard.check()`
- Imbalance pause logic (lines 709-737) → `ToxicityGuard.imbalance_check()`
- Min spread check (lines 681-707) → `ToxicityGuard.spread_check()`

**Dependencies**: Needs `settings`, `orderbook_manager`.

#### Step 4: Extract `RepricePipeline` from strategy.py
Move:
- `_maybe_reprice()` → `RepricePipeline.evaluate()`
- `_needs_reprice()` → `RepricePipeline.needs_reprice()`
- `_order_age_exceeded()` → `RepricePipeline.age_exceeded()`
- All reprice gate logic (stale check, min spread, imbalance, toxicity, age)
- `_cancel_level_order()` → stays in strategy (needs order_manager)

**Dependencies**: Needs `settings`, `pricing_engine`, `toxicity_guard`, `post_only_safety`, `order_manager`.

#### Step 5: Add `VolatilityRegime` (new module)
- Multi-horizon realized vol estimation (1m, 5m, 15m)
- Regime classification: CALM / NORMAL / ELEVATED / EXTREME
- Dynamic parameter scaling: `scale_offsets(regime)`, `scale_thresholds(regime)`
- Replace static `micro_vol_max_bps` / `micro_drift_max_bps` with regime-relative checks

#### Step 6: Add `TrendSignal` (new module)
- Short-term trend detection (EMA crossover on 1m/5m, or linear regression slope)
- Output: direction (BULLISH / NEUTRAL / BEARISH) + strength (0-1)
- Feeds into `PricingEngine` to adjust skew and size asymmetry
- Feeds into `ToxicityGuard` to be more tolerant of drift in the trend direction

#### Step 7: Slim down `strategy.py` to orchestrator
After extractions, strategy.py should only contain:
- `__init__()` — wire components together
- `run()` — classmethod entry point (bootstrap + lifecycle)
- `_level_task()` — the async per-level loop (delegates to RepricePipeline)
- `_on_fill()` — fill callback (journal recording)
- `_on_level_freed()` — level-freed callback (delegates to PostOnlySafety)
- Signal handlers, position refresh task, circuit breaker task

Target: ~250-300 lines, down from 1,405.

### 3.3 New Config Parameters (added to `config.py`)

```python
# --- Volatility Regime ---
vol_regime_enabled: bool = True
vol_short_window_s: float = 60.0        # 1-minute vol
vol_medium_window_s: float = 300.0      # 5-minute vol
vol_long_window_s: float = 900.0        # 15-minute vol
vol_regime_calm_bps: Decimal = 5        # Below this = CALM
vol_regime_elevated_bps: Decimal = 25   # Above this = ELEVATED
vol_regime_extreme_bps: Decimal = 60    # Above this = EXTREME (pause)
vol_offset_scale_elevated: Decimal = 2.0  # Multiply offsets by this in ELEVATED
vol_offset_scale_calm: Decimal = 0.7    # Multiply offsets by this in CALM

# --- Trend Signal ---
trend_enabled: bool = True
trend_fast_ema_s: float = 60.0          # 1-minute EMA
trend_slow_ema_s: float = 300.0         # 5-minute EMA
trend_strong_threshold: Decimal = 0.5   # Strength above this = strong trend
trend_skew_multiplier: Decimal = 2.0    # Multiply skew by this in strong trend
trend_size_reduction: Decimal = 0.5     # Reduce counter-trend size by this factor
```

### 3.4 Test Plan

Each extracted module gets its own test file:
- `test_pricing_engine.py` — offset computation, skew curves, tick rounding
- `test_post_only_safety.py` — price clamping, adaptive streak logic
- `test_toxicity_guard.py` — vol/drift/imbalance thresholds
- `test_reprice_pipeline.py` — gate evaluation, decision reasons
- `test_volatility_regime.py` — regime classification, parameter scaling
- `test_trend_signal.py` — trend detection, direction/strength output

Existing `test_mm_strategy.py` tests should still pass (import paths updated).

---

## Part 4: Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| **P0** | Extract `PricingEngine` | Enables independent pricing improvements | Low |
| **P0** | Extract `ToxicityGuard` | Enables crypto-specific threshold tuning | Low |
| **P0** | Add `VolatilityRegime` | Fixes the #1 reason crypto pairs don't work | Medium |
| **P1** | Extract `PostOnlySafety` | Clean separation of concerns | Low |
| **P1** | Extract `RepricePipeline` | Reduces strategy.py to orchestrator | Medium |
| **P1** | Add `TrendSignal` | Fixes inventory buildup during trends | Medium |
| **P2** | Funding rate integration | Crypto-specific edge improvement | Medium |
| **P2** | Cross-market correlation | Multi-market hedging | High |
