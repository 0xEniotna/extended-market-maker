# Market Maker Config Guide

## 0. Profiles (Legacy vs Crypto)

Use `MM_MARKET_PROFILE` to switch behavior:
- `legacy` (default): keeps prior static guard/skew behavior for non-crypto parity.
- `crypto`: enables volatility regimes, trend bias, inventory bands, and funding bias.

Recommended crypto baseline:
- `MM_MARKET_PROFILE=crypto`
- `MM_VOL_REGIME_ENABLED=true`
- `MM_VOL_REGIME_SHORT_WINDOW_S=15`
- `MM_VOL_REGIME_MEDIUM_WINDOW_S=60`
- `MM_VOL_REGIME_LONG_WINDOW_S=120`
- `MM_VOL_REGIME_CALM_BPS=8`
- `MM_VOL_REGIME_ELEVATED_BPS=20`
- `MM_VOL_REGIME_EXTREME_BPS=45`
- `MM_VOL_OFFSET_SCALE_CALM=0.8`
- `MM_VOL_OFFSET_SCALE_ELEVATED=1.5`
- `MM_VOL_OFFSET_SCALE_EXTREME=2.2`
- `MM_TREND_ENABLED=true`
- `MM_TREND_FAST_EMA_S=15`
- `MM_TREND_SLOW_EMA_S=60`
- `MM_TREND_STRONG_THRESHOLD=0.7`
- `MM_TREND_COUNTER_SIDE_SIZE_CUT=0.6`
- `MM_TREND_SKEW_BOOST=1.5`
- `MM_INVENTORY_WARN_PCT=0.5`
- `MM_INVENTORY_CRITICAL_PCT=0.8`
- `MM_INVENTORY_HARD_PCT=0.95`
- `MM_FUNDING_BIAS_ENABLED=true`
- `MM_FUNDING_INVENTORY_WEIGHT=1.0`
- `MM_FUNDING_BIAS_CAP_BPS=5`

## 1. Change Log (Grouped by Theme)

### Reprice Hysteresis (NEW)
Prevents excessive cancel/replace churn on noisy books.
- `MM_MIN_REPRICE_MOVE_TICKS` — minimum price movement in ticks to trigger reprice
- `MM_MIN_REPRICE_EDGE_DELTA_BPS` — minimum change in theoretical edge (bps) before repricing
- `MM_MIN_REPRICE_INTERVAL_S` — rate-limit reprices per slot (seconds)

### Adaptive POST_ONLY Handling (NEW)
Dynamically widens safety buffer after POST_ONLY_FAILED rejections.
- `MM_ADAPTIVE_POF_ENABLED` — enable/disable the adaptive system
- `MM_POF_MAX_SAFETY_TICKS` — ceiling on adaptive tick buffer
- `MM_POF_BACKOFF_MULTIPLIER` — exponential backoff on cooldown per consecutive POF
- `MM_POF_STREAK_RESET_S` — reset streak if no POF within this window
- `MM_POST_ONLY_SAFETY_TICKS` — base tick buffer from opposite BBO
- `MM_POF_COOLDOWN_S` — base cooldown after a POF rejection

### Exposure / Notional Limits (NEW)
USD-denominated caps on order and position size.
- `MM_MAX_ORDER_NOTIONAL_USD` — per-order USD cap
- `MM_MAX_POSITION_NOTIONAL_USD` — total position USD cap

### Inventory Skew Shaping (NEW)
Nonlinear skew curve replacing the old linear model.
- `MM_INVENTORY_DEADBAND_PCT` — fraction of max_position with zero skew
- `MM_SKEW_SHAPE_K` — tanh curvature coefficient
- `MM_SKEW_MAX_BPS` — maximum skew contribution in bps (before skew_factor scaling)

### Imbalance & Toxicity Filter (NEW)
Microstructure-aware guards that pause or widen quotes.
- `MM_IMBALANCE_WINDOW_S` — smoothing window for book imbalance
- `MM_IMBALANCE_PAUSE_THRESHOLD` — signed imbalance level that pauses one side
- `MM_MICRO_VOL_WINDOW_S` / `MM_MICRO_VOL_MAX_BPS` — micro-volatility guard
- `MM_MICRO_DRIFT_WINDOW_S` / `MM_MICRO_DRIFT_MAX_BPS` — micro-drift guard
- `MM_VOLATILITY_OFFSET_MULTIPLIER` — extra offset widening in moderate stress

### Volatility Regime (NEW)
Profile-aware volatility classifier used by `MM_MARKET_PROFILE=crypto`.
- `MM_VOL_REGIME_ENABLED`
- `MM_VOL_REGIME_SHORT_WINDOW_S`, `MM_VOL_REGIME_MEDIUM_WINDOW_S`, `MM_VOL_REGIME_LONG_WINDOW_S`
- `MM_VOL_REGIME_CALM_BPS`, `MM_VOL_REGIME_ELEVATED_BPS`, `MM_VOL_REGIME_EXTREME_BPS`
- `MM_VOL_OFFSET_SCALE_CALM`, `MM_VOL_OFFSET_SCALE_ELEVATED`, `MM_VOL_OFFSET_SCALE_EXTREME`

### Trend and Funding Bias (NEW)
- `MM_TREND_ENABLED`
- `MM_TREND_FAST_EMA_S`, `MM_TREND_SLOW_EMA_S`
- `MM_TREND_STRONG_THRESHOLD`, `MM_TREND_COUNTER_SIDE_SIZE_CUT`, `MM_TREND_SKEW_BOOST`
- `MM_FUNDING_BIAS_ENABLED`, `MM_FUNDING_INVENTORY_WEIGHT`, `MM_FUNDING_BIAS_CAP_BPS`
- `MM_INVENTORY_WARN_PCT`, `MM_INVENTORY_CRITICAL_PCT`, `MM_INVENTORY_HARD_PCT`

### Failure-Rate Circuit Breaker (NEW)
Rolling-window failure rate check (complements consecutive-failure breaker).
- `MM_FAILURE_WINDOW_S` — rolling window for failure rate
- `MM_FAILURE_RATE_TRIP` — trip threshold (failures/attempts)
- `MM_MIN_ATTEMPTS_FOR_BREAKER` — minimum attempts before rate breaker can trip

### Stale Book Safety (NEW)
- `MM_CANCEL_ON_STALE_BOOK` — cancel resting orders on stale data
- `MM_STALE_CANCEL_GRACE_S` — grace period before cancelling

### Telemetry (NEW)
- `MM_JOURNAL_REPRICE_DECISIONS` — log every reprice skip/trigger to journal

### Existing Core Knobs (unchanged behavior)
- `MM_SPREAD_MULTIPLIER`, `MM_MIN_OFFSET_BPS`, `MM_MAX_OFFSET_BPS`, `MM_MIN_SPREAD_BPS`
- `MM_REPRICE_TOLERANCE_PERCENT`, `MM_MAX_ORDER_AGE_S`
- `MM_ORDER_SIZE_MULTIPLIER`, `MM_MAX_POSITION_SIZE`, `MM_SIZE_SCALE_PER_LEVEL`
- `MM_INVENTORY_SKEW_FACTOR`, `MM_NUM_PRICE_LEVELS`

---

## 2. Variable Reference

### Reprice Hysteresis

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_MIN_REPRICE_MOVE_TICKS` | 2 | Min price move (ticks) to cancel+replace | Higher → less churn, but staler quotes | 1–5 | Increase if cancel rate >50%. Decrease if quotes lag price. |
| `MM_MIN_REPRICE_EDGE_DELTA_BPS` | 0.5 | Min edge change (bps) to reprice | Higher → fewer reprices, risk stale edge | 0.3–2.0 | Increase if too many cancels. Decrease if fill rate drops. |
| `MM_MIN_REPRICE_INTERVAL_S` | 0.5 | Rate-limit per slot | Higher → less API usage, staler quotes | 0.2–2.0 | Increase on rate-limit errors. Decrease for fast markets. |

### Adaptive POST_ONLY Handling

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_ADAPTIVE_POF_ENABLED` | true | Enable adaptive safety tick widening | On = fewer POF rejections; Off = tighter quotes | true/false | Disable only if POF rate is already <1%. |
| `MM_POST_ONLY_SAFETY_TICKS` | 2 | Base tick buffer from opposite BBO | Higher → fewer POFs, wider quotes | 1–4 | Increase if POF rate >5%. Decrease if quotes too wide. |
| `MM_POF_MAX_SAFETY_TICKS` | 8 | Ceiling on adaptive buffer | Higher → more room to adapt, wider quotes in worst case | 4–12 | Increase on high-POF markets. Decrease on wide-tick markets. |
| `MM_POF_BACKOFF_MULTIPLIER` | 1.7 | Exponential backoff on cooldown per streak | Higher → longer pauses after repeated POFs | 1.3–2.5 | Increase if POF storms cause API rate limits. |
| `MM_POF_STREAK_RESET_S` | 45 | Reset streak after quiet period | Lower → faster recovery, risk re-triggering | 15–90 | Decrease for liquid markets. Increase for thin books. |
| `MM_POF_COOLDOWN_S` | 2.0 | Base cooldown after POF (seconds) | Higher → less churn, miss fills | 0.5–5.0 | Increase if POFs cause retry storms. |

### Exposure / Notional Limits

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_MAX_ORDER_NOTIONAL_USD` | 250 | Max USD per single order | Lower → limits tail risk, may cap fill size | 50–1000 | Increase if order sizes are being clipped. Decrease for tighter risk. |
| `MM_MAX_POSITION_NOTIONAL_USD` | 2500 | Max USD total position | Lower → less inventory risk, fewer fills at limits | 500–10000 | Scale with account equity. Decrease if drawdowns too large. |

### Inventory Skew

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_INVENTORY_SKEW_FACTOR` | 0.5 | Overall skew intensity multiplier | Higher → faster inventory reduction, wider quotes on loaded side | 0.05–1.0 | Increase if inventory drifts >50% of max. Decrease if fill rate too asymmetric. |
| `MM_INVENTORY_DEADBAND_PCT` | 0.10 | Fraction of max_position with zero skew | Higher → more symmetric near flat, slower response to small positions | 0.05–0.25 | Increase for mean-reverting markets. Decrease for trending/toxic flow. |
| `MM_SKEW_SHAPE_K` | 2.0 | Tanh curvature: 0=linear, higher=more aggressive at extremes | Higher → gentle near flat, steep at limits; Lower → linear response | 1.0–4.0 | Increase if you want to tolerate small positions but aggressively shed large ones. |
| `MM_SKEW_MAX_BPS` | 20 | Max skew offset contribution (bps, before skew_factor) | Higher → more aggressive inventory reduction at limits | 10–50 | Increase if inventory regularly hits max_position. |

### Toxicity / Microstructure

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_IMBALANCE_WINDOW_S` | 2.0 | EMA window for top-of-book imbalance | Shorter → faster reaction, noisier | 1.0–5.0 | Decrease for thin books. Increase for noisy thick books. |
| `MM_IMBALANCE_PAUSE_THRESHOLD` | 0.70 | Pause one side when imbalance exceeds this | Lower → more conservative (pauses more), fewer adverse fills | 0.5–0.85 | Decrease if negative markouts on one side. Increase if too many skipped cycles. |
| `MM_MICRO_VOL_WINDOW_S` | 5.0 | Window for mid-price volatility (seconds) | Shorter → faster detection, noisier | 2.0–10.0 | Decrease for fast markets. |
| `MM_MICRO_VOL_MAX_BPS` | 8 | Soft vol threshold: above → widen; 1.25x → pause | Lower → more conservative | 4–15 | Decrease if adverse fills cluster around vol spikes. |
| `MM_MICRO_DRIFT_WINDOW_S` | 3.0 | Window for directional drift detection | Shorter → faster, more false positives | 1.0–5.0 | Decrease for momentum-driven markets. |
| `MM_MICRO_DRIFT_MAX_BPS` | 6 | Soft drift threshold: above → widen; 1.25x → pause | Lower → more conservative, fewer fills | 3–10 | Decrease if markouts negative during trends. |
| `MM_VOLATILITY_OFFSET_MULTIPLIER` | 0.35 | Extra offset in moderate stress (fraction of excess vol/drift) | Higher → wider quotes in stress, fewer adverse fills | 0.1–0.7 | Increase if stress fills are unprofitable. |

### Circuit Breaker

| Variable | Default | What it controls | Tradeoff | Safe range | Tune |
|---|---|---|---|---|---|
| `MM_CIRCUIT_BREAKER_MAX_FAILURES` | 5 | Consecutive failures to trip | Lower → trips faster, more downtime | 3–10 | Decrease if failures cascade. |
| `MM_CIRCUIT_BREAKER_COOLDOWN_S` | 30 | Pause duration after trip | Longer → more protection, more missed fills | 15–120 | Increase if issues persist after reset. |
| `MM_FAILURE_WINDOW_S` | 60 | Rolling window for rate-based breaker | Shorter → more responsive | 30–120 | Match to your typical failure burst duration. |
| `MM_FAILURE_RATE_TRIP` | 0.35 | Trip when failure/attempts > this | Lower → more sensitive | 0.2–0.5 | Decrease if you see sustained partial failures. |
| `MM_MIN_ATTEMPTS_FOR_BREAKER` | 10 | Min attempts before rate check applies | Higher → more forgiving during startup | 5–20 | Increase if cold-start causes false trips. |

### Stale Book & Telemetry

| Variable | Default | What it controls | Safe range |
|---|---|---|---|
| `MM_CANCEL_ON_STALE_BOOK` | true | Cancel orders when OB data is stale | true (always) |
| `MM_STALE_CANCEL_GRACE_S` | 3.0 | Grace before cancelling on stale data | 1.0–5.0 |
| `MM_JOURNAL_REPRICE_DECISIONS` | true | Log skip/trigger reasons to journal | true for debugging, false to reduce journal size |

---

## 3. Interaction Groups

### Spread Controls
`SPREAD_MULTIPLIER × spread_ema` → clamped by [`MIN_OFFSET_BPS`, `MAX_OFFSET_BPS`] → this is your **placement distance**.
`MIN_SPREAD_BPS` is a **separate gate**: if the book spread < this value, all quoting pauses and resting orders cancel.
- MIN_SPREAD_BPS protects against being the only liquidity in a tight book
- MIN_OFFSET_BPS is your actual minimum distance from best — **this is your real floor**

### Reprice Controls
Three gates must ALL pass for a reprice to fire:
1. `REPRICE_TOLERANCE_PERCENT` — price deviation exceeds this fraction of target offset
2. `MIN_REPRICE_MOVE_TICKS` — price move exceeds this many ticks
3. `MIN_REPRICE_EDGE_DELTA_BPS` — edge improvement exceeds this many bps

Plus rate limiting: `MIN_REPRICE_INTERVAL_S` between reprices per slot.
And forced refresh: `MAX_ORDER_AGE_S` forces reprice regardless of above.

### Size / Risk Controls
`ORDER_SIZE_MULTIPLIER × min_order_size` → capped by `MAX_ORDER_NOTIONAL_USD / price`.
Position capped by BOTH `MAX_POSITION_SIZE` (contracts) AND `MAX_POSITION_NOTIONAL_USD` (USD).
The tighter constraint wins.

### Skew Controls
`INVENTORY_SKEW_FACTOR` scales the total skew output.
Inside, `SKEW_SHAPE_K` controls the tanh curve shape, `INVENTORY_DEADBAND_PCT` creates a zero-skew zone near flat, and `SKEW_MAX_BPS` caps the raw skew contribution.
Effective max skew = `SKEW_MAX_BPS × INVENTORY_SKEW_FACTOR` bps at position limits.

For XCU: effective max = 20 × 0.15 = **3 bps** (very gentle)
For ZRO: effective max = 20 × 0.6 = **12 bps** (aggressive)

---
