# TECH100m iter003 — double again (×2 from iter002)

**Date**: 2026-05-22 ~12:30 UTC
**Iter**: iter002 → iter003 (MAX_ORDER $600→$1200, MAX_POSITION_NOTIONAL $4000→$8000)
**Rollback**: stop iter003, start iter002.

## Why TECH100m and NOT DOT

The decision rule crystallized this session: **scale only markets with
positive REALIZED PnL** (genuine directional/mean-reversion edge), not
markets that are net-positive only via spread capture (fragile).

| | TECH100m | DOT |
|---|---|---|
| Realized since scale-up (4d) | +$29 | −$12.80 |
| Total | +$75 | +$28 |
| Type | NASDAQ-100 index (diversified) | crypto single-asset |
| Profit source | spread + direction | spread only |
| Verdict | **DOUBLE** | **hold, don't scale** |

TECH100m's positive realized = the index mean-reverts and our fills are
not toxic. DOT's negative realized = we're only earning the spread; the
+$28 total came from a lucky bounce after a −$24 directional drawdown.
Doubling DOT would repeat the XNG scale-up mistake (lever a market right
before it drifts).

## Single-stock vs index (the TSLA lesson)

TSLA (−$17 since scale-up, realized −$58) demonstrates why single-stock
perps are worse MM targets than indices: idiosyncratic event risk
produces sharp one-sided directional moves ("0% SELL fills" flag). The
index (TECH100m) averages 100 names, so idiosyncratic moves wash out and
the flow mean-reverts. Prefer indices for passive MM.

## Verified at launch
- Orders placed at ~$1200 (was $596) — clean 2× scale
- multiplier=50 already over-requests, so MAX_ORDER cap binds; no
  multiplier change needed
- Drawdown abs threshold now 7.5% × $8000 = $600
