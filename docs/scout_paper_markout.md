# Scout Stage B: Paper Maker Markout

This document describes the optional Stage B pre-launch filter used by
`scripts/tools/market_scout_pipeline.py --paper-markout`.

## Overview

- Stage A (`find_mm_markets.py` + `screen_mm_markets.py`) scores candidates from spread, tick, volume, OI, and coverage.
- Stage B (paper markout) estimates adverse selection using public streams only.
- Stage B is **paper-only**: no live or simulated order placement on exchange APIs.

When Stage B is disabled, scout behavior remains Stage A only.

## Data Sources

Stage B consumes public WS streams:

- Orderbook top-of-book (`depth=1`) for bid/ask and mid updates.
- Public trades for taker side, trade type, trade timestamp, and price.

Stream-level sequence numbers are checked per websocket connection. On seq gap or
out-of-order, Stage B reconnects that stream. Per-market runtime state resets are
applied only when explicit row-level per-market sequence fields are out-of-order.

Before trade processing starts, Stage B performs a short orderbook warmup
(`--paper-warmup-s`, default `8`) so BBO is initialized first. If warmup times
out, sampling continues with warnings listing markets still missing BBO.

## Paper Fill Model

Given current BBO and one trade event:

- Taker `BUY` means maker `SELL` at ask.
- Taker `SELL` means maker `BUY` at bid.

Default filters:

- Include only trade type `TRADE`.
- `bbo_match_mode=strict`: trade must match BBO side price within tolerance.
- Optional `bbo_match_mode=loose`: allow `trade_price >= ask` for taker BUY and `trade_price <= bid` for taker SELL.

Default fill price is BBO side price at inference time (not trade price).

## Markout Definitions

For fill at time `t` with price `fill_px` and mid `mid(t)`:

- Maker BUY:
  - `markout_h_bps = (mid(t+h) - fill_px) / mid(t) * 10_000`
- Maker SELL:
  - `markout_h_bps = (fill_px - mid(t+h)) / mid(t) * 10_000`

Default horizons (ms): `250,1000,5000,30000,120000`.
If `mid(t+h)` is unavailable, that fill is excluded for that horizon.

Trade timestamp policy is hybrid (`--paper-max-trade-lag-ms`, default `5000`):

- Use trade row `T` when `abs(message.ts - T) <= max_trade_lag_ms`.
- Otherwise, fall back to message `ts` and record a data-quality warning.

## Queue Capture

Queue position is unknown in public data. Stage B applies
`queue_capture` to fill-rate expectation only:

- `paper_fill_rate_per_min_adjusted = paper_fill_rate_per_min * queue_capture`

Markout values are not scaled by queue capture.

## Gates and Ranking

When enabled, Stage B adds gates:

- `paper_fills >= paper_min_fills`
- `paper_toxicity_share_250ms <= paper_max_toxicity_250ms`
- `paper_markout_250ms >= paper_min_markout_250ms`
- `paper_markout_1s >= paper_min_markout_1s`

Pass logic:

- `stageA_pass`: existing Stage A gates
- `stageB_pass`: all Stage B conditions true
- `passes_all`: `stageA_pass && stageB_pass`

When Stage B samples only top-K markets, non-sampled markets are marked
`paper_sampled=false`, `stageB_pass=null`, and `passes_all=false` (Top-K only).

Score adjustment:

- `base_score`: existing Stage A score
- `markout_score = clamp((paper_markout_1s_bps + shift) / scale, 0, max)`
- `toxicity_penalty = clamp((paper_toxicity_share_250ms - center) * scale, 0, max)`
- `score2 = base_score + markout_score - toxicity_penalty`

Ranking (paper mode): `passes_all desc`, then `score2 desc`.

## Degraded Stage B Fallback

Stage B health counters are recorded per sampled market:

- `paper_trade_rows_seen`
- `paper_trade_rows_used`
- `paper_bbo_updates_seen`
- `paper_bbo_ready`

Run-level degraded criteria:

- sampled markets exist, and
- total inferred paper fills equals `0`, and
- either trades or orderbook updates were observed.

`--paper-fallback-mode` controls behavior:

- `stageA` (default): apply Stage A gating/ranking for this run, keep Stage B diagnostics in report.
- `strict`: keep Stage B gating/ranking even when degraded.

## Output Fields

Stage B adds market fields (examples):

- `paper_fills`
- `paper_fill_rate_per_min`
- `paper_fill_rate_per_min_adjusted`
- `paper_markout_bps_250ms_mean`, `paper_markout_bps_1s_mean`
- `paper_toxicity_share_250ms`, `paper_toxicity_share_1s`
- side splits such as `paper_markout_buy_bps_1s_mean` and `paper_markout_sell_bps_1s_mean`
- health counters: `paper_trade_rows_seen`, `paper_trade_rows_used`,
  `paper_bbo_updates_seen`, `paper_bbo_ready`

The scout markdown report adds a `Paper Markout (Stage B)` section and extends
Top Candidates with Stage B columns when paper scoring is active. The Stage B
table includes sampled markets only.

## Limitations

- Queue priority is unknown, so fill probabilities are approximate.
- Public data quality can vary; seq gaps trigger resets/reconnects.
- Markout depends on observed top-of-book mid path and sampling window length.
