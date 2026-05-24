# Live experiments + speed test — 2026-05-24

**Status**: ACTIVE. Consolidated tracker for everything launched 2026-05-24.
Reconvene checkpoint: **+24–48 h** (agenda at bottom).
**Branch**: `microprice-ofi`. **Live dir**: `/root/MM` @ `14b8c77` (deployed).

---

## Code shipped today

- **`14b8c77` — microprice decoupled from the crypto-profile gate.** Microprice
  now gates on `MM_USE_MICROPRICE` alone, validated **per-market by diagnostic**
  (not by asset class). Rationale: all Extended perps trade 24/7; microprice
  leads mid where the perp book is **liquid** (ETH +0.41, DOT, GOOG +0.13,
  TECH100m +0.07) and nulls/inverts where **thin** (MU). The old crypto-vs-
  TradFi gate was the wrong axis. `ruff` clean, 632 tests pass. Magnitude cap
  (`MM_MICROPRICE_CAP_BPS=10`) still applies.

## Fleet (7 running, ~20:40 UTC)

| Market | Instance | microprice | Notes |
|---|---|---|---|
| DOT-USD | env_dot_iter002 | ✅ | A/B since 05-22 17:04 |
| GOOG_24_5 | env_goog_24_5_iter002 | ✅ (new) | legacy profile (decoupled) |
| TECH100m | env_tech100m_iter004 | ✅ (new) | top earner — tight rollback |
| EDGE-USD | EDGE-USD (env_edge) | ✅ (new) | NEW crypto, probe $500 |
| SPCX (XYZSPCX_ORCLPX) | env_xyzspcx_orclpx_iter003 | ⛔ | scaled + reaction-speed test |
| MU_24_5 | env_mu_24_5_iter003 | ⛔ | diagnostic null — correctly off |
| EWY_24_5 | env_ewy_24_5_iter001 | ⛔ | untested |

---

## Experiments + pre-registered criteria

### 1. DOT microprice (env_dot_iter002, since 05-22 17:04)
- A/B vs `.env.dot`. Capped 10 bps. ~4–8 fills/day → ≥15-fill half-sample due.
- **Rollback**: flag off → `mmctl start .env.dot`. Any drawdown_stop → auto.

### 2. GOOG microprice (env_goog_24_5_iter002, 05-24)
- Diagnostic +0.13/ρ+0.19. Replay PASS. legacy profile (works post-decoupling).
- **Watch**: A/B vs iter001 **+ active-hours check** (confirm signal holds while
  the NASDAQ underlying is open, not just aggregate).
- **Rollback** → `.env.goog_24_5.iter001`.

### 3. TECH100m microprice (env_tech100m_iter004, 05-24) — TIGHT GATE
- Diagnostic +0.07 (weakest positive). Replay PASS. **Top earner → highest
  blast radius.**
- **Rollback (fast)**: check at **≥10 fills (~1 day)**; if treatment +5s markout
  < `iter003` baseline or $/day drops → roll back to `iter003` immediately.
  Daily watch.

### 4. EDGE-USD new crypto (env_edge, 05-24)
- New market. crypto profile, probe **$500 pos / $150 order**, drawdown $150,
  microprice **on ahead of its own diagnostic** (liquid-crypto generalization:
  $665k/day, 12.7 bps, fine tick). Order notional binds at $150 (sizing OK).
- **Pending gate**: run EDGE's own microprice diagnostic at **24–48 h**; if sign
  not positive → `MM_USE_MICROPRICE=false`. Standard probe rollback otherwise.

### 5. SPCX scale-up + reaction-speed (env_xyzspcx_orclpx_iter003, 05-24)
- iter002 scaled order $150→$250, pos $500→$900 (binds, +1.7× throughput).
- iter003 adds **faster reaction** (this is **Step 1 of the speed thesis**):
  `MIN_REPRICE_INTERVAL_S 2.0→0.5`, `MIN_REPRICE_MOVE_TICKS 10→4` (~2 bps vs
  ~5 bps trigger). Size held constant; microprice off (clean isolation).
- **First result (in)**: rate-safe — `rl_hits=0`, cb closed, 0 rejects; churn
  0.58→0.97 actions/s. **We are NOT rate-constrained → faster is feasible.**
- **Signal (pending, ~2–3 days, SPCX fills slowly)**: does markout improve vs
  iter002 baseline (+5s mean −0.50, SELL −3.06)?
- **Rollback** → `.env.xyzspcx_orclpx.iter002`.

---

## Speed / Tokyo decision (the bigger thread)

- **VPS**: Hostinger KVM, **Kuala Lumpur**, 2 vCPU EPYC, 8 GB. Compute NOT the
  bottleneck (~54% load).
- **Network**: Extended gateways are **AWS Tokyo** (ap-northeast-1). Measured
  **~76 ms RTT** KL→Tokyo. Bot self-reports `latency≈200–300 ms`; reaction
  (book→quote) p50 ~187 ms.
- **Bottleneck**: geography (76 ms) + the **1 s reprice floor** (config, not
  hardware — the EPYC can do sub-second easily).
- **Why it matters**: this staleness window IS the spread-band rule — ≤8 bps
  toxic (picked off inside our window), ≥12 bps profitable.
- **Decision logic**: Step 1 (SPCX reaction test) → if markout improves +
  rate-safe ⇒ a **Tokyo VPS move is justified** (would cut 76 ms→~2–10 ms,
  amplifying the gain + opening tighter markets). If markout unchanged ⇒ skip.
- **Tokyo options if we proceed**: a plain **Vultr/Linode Tokyo VPS** (~$20–48/mo,
  Hostinger-like simplicity, ~2–10 ms to Extended) captures ~90% — **don't need
  AWS** unless going fully latency-competitive. Cost is noise vs ~$36/day PnL.

---

## Operating reminders (hard-won today)
- **`mmctl stop` by the ENV LABEL you started with** (`.env.X.iterNNN`), NOT the
  market name — and **confirm stopped via `mmctl status` BEFORE starting the
  replacement** (2026-05-24 SPCX double-run incident).
- All env edits via `.env.<m>.iterNNN` copies; never edit live in place.
- `MM_DEADMAN_ENABLED=false` on every env (SDK lacks the switch).

## +24–48 h checkpoint agenda
1. **EDGE diagnostic** → keep microprice on, or flip off.
2. **Microprice markout reads** — DOT / GOOG (active-hours split) / TECH100m
   (tight gate, may already have ≥10 fills).
3. **SPCX reaction-speed markout** — the Step-1 signal for the Tokyo decision.
4. **Decide Tokyo** (and whether to lower the reprice floor fleet-wide).
