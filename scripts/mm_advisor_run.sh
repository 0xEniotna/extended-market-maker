#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${MM_REPO_ROOT:-/home/flexouille/Code/MM}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
OPENCLAW_BIN="${OPENCLAW_BIN:-/home/flexouille/.nvm/versions/node/v22.22.0/bin/openclaw}"
STATE_DIR="${STATE_DIR:-/home/flexouille/.openclaw/mm-advisor}"
STATE_FILE="${STATE_DIR}/state.json"
MEDIA_DIR="${MEDIA_DIR:-/home/flexouille/.openclaw/media}"
DISCORD_ACCOUNT_PUBLISHER="${DISCORD_ACCOUNT_PUBLISHER:-publisher}"
DISCORD_CHANNEL_ID_AUDITOR="${DISCORD_CHANNEL_ID_AUDITOR:-1474058159332004034}"
DISCORD_CHANNEL_ID_MAIN="${DISCORD_CHANNEL_ID_MAIN:-1476353346502922270}"

mkdir -p "${STATE_DIR}" "${MEDIA_DIR}"
LOCK_FILE="${STATE_DIR}/advisor.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

send_discord() {
  local account="$1"
  local channel_id="$2"
  local message="$3"
  local media_file="$4"
  local out_json="$5"
  local out_err="$6"

  if [[ -n "${media_file}" && -f "${media_file}" ]]; then
    "${OPENCLAW_BIN}" message send \
      --channel discord \
      --account "${account}" \
      --target "channel:${channel_id}" \
      --message "${message}" \
      --media "${media_file}" \
      --json >"${out_json}" 2>"${out_err}" || return 1
  else
    "${OPENCLAW_BIN}" message send \
      --channel discord \
      --account "${account}" \
      --target "channel:${channel_id}" \
      --message "${message}" \
      --json >"${out_json}" 2>"${out_err}" || return 1
  fi
  return 0
}

TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
STDOUT_LOG="${STATE_DIR}/advisor_${STAMP}.stdout.log"
STDERR_LOG="${STATE_DIR}/advisor_${STAMP}.stderr.log"

PROPOSALS_PATH="${REPO_ROOT}/data/mm_audit/advisor/proposals.jsonl"
SUMMARY_PATH="${REPO_ROOT}/data/mm_audit/advisor/latest_summary.md"
ANALYST_PACKET_PATH="${REPO_ROOT}/data/mm_audit/advisor/analyst_packet.md"

BEFORE_LINES=0
if [[ -f "${PROPOSALS_PATH}" ]]; then
  BEFORE_LINES=$(wc -l < "${PROPOSALS_PATH}")
fi

LAST_SIGNATURE=""
if [[ -f "${STATE_FILE}" ]]; then
  LAST_SIGNATURE=$(jq -r '.last_signature // ""' "${STATE_FILE}" 2>/dev/null || echo "")
fi

cd "${REPO_ROOT}"
if ! "${PYTHON_BIN}" scripts/mm_advisor_loop.py \
  --repo "${REPO_ROOT}" \
  --iterations 1 \
  --sleep-s 0 \
  --journal-dir data/mm_journal \
  --advisor-dir data/mm_audit/advisor \
  --baselines-dir data/mm_audit/autotune_baselines \
  --config-changelog data/mm_audit/config_changelog.jsonl \
  --apply-receipts data/mm_audit/advisor/apply_receipts.jsonl \
  >"${STDOUT_LOG}" 2>"${STDERR_LOG}"; then
  ERR_SUMMARY="$(tail -n 20 "${STDERR_LOG}" 2>/dev/null | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g' | cut -c1-500)"
  MSG="Advisor FAILED at ${TS_UTC}.\nError: ${ERR_SUMMARY}"
  send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_AUDITOR}" "${MSG}" "" "${STATE_DIR}/send_fail_auditor_${STAMP}.json" "${STATE_DIR}/send_fail_auditor_${STAMP}.err" || true
  send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_MAIN}" "${MSG}" "" "${STATE_DIR}/send_fail_main_${STAMP}.json" "${STATE_DIR}/send_fail_main_${STAMP}.err" || true
  printf "%s status=error stderr=%s\n" "${TS_UTC}" "${STDERR_LOG}" >> "${STATE_DIR}/runs.log"
  exit 1
fi

AFTER_LINES=0
if [[ -f "${PROPOSALS_PATH}" ]]; then
  AFTER_LINES=$(wc -l < "${PROPOSALS_PATH}")
fi
DELTA=$((AFTER_LINES - BEFORE_LINES))
if [[ "${DELTA}" -lt 0 ]]; then
  DELTA=0
fi

SIGNATURE="${LAST_SIGNATURE}"
if [[ "${DELTA}" -gt 0 && -f "${SUMMARY_PATH}" ]]; then
  METRICS_JSON=$(tail -n "${DELTA}" "${PROPOSALS_PATH}" | python3 -c 'import hashlib,json,sys
rows=[]
for raw in sys.stdin:
 raw=raw.strip()
 if not raw: continue
 try: row=json.loads(raw)
 except Exception: continue
 if isinstance(row,dict): rows.append(row)
passed=sum(1 for r in rows if r.get("guardrail_status")=="passed")
rejected=sum(1 for r in rows if r.get("guardrail_status")=="rejected")
warren=sum(1 for r in rows if r.get("deadman") is True and r.get("guardrail_status")=="passed" and r.get("confidence")=="high" and r.get("escalation_target")=="warren")
markets=sorted({str(r.get("market","")) for r in rows if r.get("market")})
fingerprint=[]
for r in rows:
 fingerprint.append([
  str(r.get("market","")),
  str(r.get("param","")),
  str(r.get("proposed","")),
  str(r.get("guardrail_status","")),
  str(r.get("escalation_target","")),
  bool(r.get("deadman") is True),
  str(r.get("guardrail_reason","")),
 ])
fingerprint.sort()
sig=hashlib.sha256(json.dumps(fingerprint,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
print(json.dumps({"total":len(rows),"passed":passed,"rejected":rejected,"warren_auto_candidates":warren,"markets":markets,"signature":sig}))')

  SIGNATURE=$(echo "${METRICS_JSON}" | jq -r '.signature // ""')
  if [[ -n "${SIGNATURE}" && "${SIGNATURE}" != "${LAST_SIGNATURE}" ]]; then
    TOTAL=$(echo "${METRICS_JSON}" | jq -r '.total // 0')
    PASSED=$(echo "${METRICS_JSON}" | jq -r '.passed // 0')
    REJECTED=$(echo "${METRICS_JSON}" | jq -r '.rejected // 0')
    WARREN=$(echo "${METRICS_JSON}" | jq -r '.warren_auto_candidates // 0')
    MARKETS=$(echo "${METRICS_JSON}" | jq -r '.markets | if length==0 then "none" else join(", ") end')

    SUMMARY_MSG="Advisor OK ${TS_UTC} | proposals=${TOTAL} passed=${PASSED} rejected=${REJECTED} warren_auto=${WARREN}\nmarkets: ${MARKETS}\nartifacts:\n- ${SUMMARY_PATH}\n- ${PROPOSALS_PATH}"
    SUMMARY_MEDIA="${MEDIA_DIR}/mm_advisor_summary_${STAMP}.md"
    cp "${SUMMARY_PATH}" "${SUMMARY_MEDIA}"
    send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_AUDITOR}" "${SUMMARY_MSG}" "${SUMMARY_MEDIA}" "${STATE_DIR}/send_summary_auditor_${STAMP}.json" "${STATE_DIR}/send_summary_auditor_${STAMP}.err" || true
  fi
fi

"${PYTHON_BIN}" scripts/tools/audit_position_risk.py \
  --journal-dir data/mm_journal \
  --output-dir data/mm_audit \
  --lookback-hours 2 \
  >"${STATE_DIR}/position_risk_${STAMP}.md" 2>"${STATE_DIR}/position_risk_${STAMP}.err" || true

PNL_TXT="/tmp/mm_advisor_pnl_${STAMP}.txt"
"${PYTHON_BIN}" scripts/tools/fetch_total_pnl.py \
  --env .env.eth \
  --since "$(date -u +%Y-%m-%dT00:00:00Z)" \
  >"${PNL_TXT}" 2>"${STATE_DIR}/fetch_total_pnl_${STAMP}.err" || true

REPO_ROOT="${REPO_ROOT}" PNL_TXT="${PNL_TXT}" ANALYST_PACKET_PATH="${ANALYST_PACKET_PATH}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(os.environ['REPO_ROOT'])
PNL_TXT = Path(os.environ['PNL_TXT'])
OUT = Path(os.environ['ANALYST_PACKET_PATH'])

sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'tools'))
try:
    from analyse_mm_journal import build_summary
    from audit_report_common import discover_market_journals, load_recent_entries
except Exception:
    build_summary = None
    discover_market_journals = None
    load_recent_entries = None

def _fmt_num(value, digits=2, suffix=''):
    if value is None:
        return 'n/a'
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return f"{value}{suffix}"

SENSITIVE_KEYS = {
    'MM_API_KEY',
    'MM_STARK_PRIVATE_KEY',
    'MM_STARK_PUBLIC_KEY',
    'MM_BUILDER_ID',
}

def _mask_value(key, value):
    if value is None:
        return ''
    text = str(value)
    if key in SENSITIVE_KEYS:
        if len(text) <= 8:
            return '***'
        return text[:4] + '...' + text[-4:]
    return text

def _fmt_pct_ratio(value, digits=1):
    if value is None:
        return 'n/a'
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return 'n/a'

def _event_mid(event):
    if not isinstance(event, dict):
        return None
    for key in ('mid', 'mark_price'):
        raw = event.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except Exception:
            continue
        if value > 0:
            return value
    bid = event.get('best_bid')
    ask = event.get('best_ask')
    try:
        bid_v = float(bid) if bid is not None else None
        ask_v = float(ask) if ask is not None else None
    except Exception:
        return None
    if bid_v is None or ask_v is None:
        return None
    if bid_v <= 0 or ask_v <= 0:
        return None
    return (bid_v + ask_v) / 2.0

def _latest_mid_from_journal(path):
    try:
        tail = deque(maxlen=1200)
        with path.open() as fh:
            for raw in fh:
                raw = raw.strip()
                if raw:
                    tail.append(raw)
        for raw in reversed(tail):
            try:
                event = json.loads(raw)
            except Exception:
                continue
            mid = _event_mid(event)
            if mid is not None:
                return mid
    except Exception:
        return None
    return None

fallback_keys = [
    'MM_SPREAD_MULTIPLIER',
    'MM_MIN_OFFSET_BPS',
    'MM_REPRICE_TOLERANCE_PERCENT',
    'MM_INVENTORY_SKEW_FACTOR',
    'MM_IMBALANCE_PAUSE_THRESHOLD',
    'MM_TOXICITY_BLOCK_SECONDS',
    'MM_TOXICITY_EXTREME_THRESHOLD',
    'MM_TOXICITY_HIGH_THRESHOLD',
    'MM_TOXICITY_MEDIUM_THRESHOLD',
    'MM_TOXICITY_RISK_OFF_THRESHOLD',
    'MM_ORDER_SIZE_MULTIPLIER',
    'MM_MAX_NOTIONAL',
]

advisor_dir = ROOT / 'data/mm_audit/advisor'
journal_dir = ROOT / 'data/mm_journal'

mid_prices = {}
if discover_market_journals and journal_dir.exists():
    try:
        for market_name, journal_path in sorted(discover_market_journals(journal_dir).items()):
            mid = _latest_mid_from_journal(journal_path)
            if mid is not None:
                mid_prices[market_name] = mid
    except Exception:
        mid_prices = {}

lines = [f"# Analyst Context Packet ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})"]

lines.append("\n## Current mid prices")
if mid_prices:
    for market_name in sorted(mid_prices):
        lines.append(f"- {market_name}: mid={_fmt_num(mid_prices[market_name], 8)}")
else:
    lines.append('- unavailable (no mid price found)')

risk_path = ROOT / 'data/mm_audit/position_risk.json'
lines.append("\n## Current positions & exposure")
if risk_path.exists():
    risk = json.loads(risk_path.read_text())
    markets = risk.get('markets', {})
    for m in sorted(markets):
        row = markets[m]
        mid_text = _fmt_num(mid_prices.get(m), 8)
        lines.append(
            f"- {m}: mid={mid_text} position={row.get('current_position')} util={row.get('inventory_utilization_pct')}% "
            f"exposure_usd={row.get('directional_exposure_usd')} flags={','.join(row.get('flags', [])) or '-'}"
        )
else:
    lines.append('- missing position_risk.json')

lines.append("\n## PnL since UTC midnight")
pnl_txt = PNL_TXT.read_text() if PNL_TXT.exists() else ''
m = re.search(r"total_pnl=([-0-9.]+)\s+USD", pnl_txt)
lines.append(f"- total_pnl_usd={m.group(1) if m else 'unavailable'}")

lines.append("\n## Fill-rate and markout (last 2h)")
if build_summary and discover_market_journals and load_recent_entries:
    min_ts = time.time() - (2 * 3600)
    journals = discover_market_journals(journal_dir) if journal_dir.exists() else {}
    emitted = 0
    for market_name, journal_path in sorted(journals.items()):
        events = load_recent_entries(journal_path, min_ts)
        if not events:
            continue
        try:
            summary = build_summary(events, journal_path, None)
            metrics = summary.get('metrics', {})
            counts = summary.get('counts', {})
            lines.append(
                f"- {market_name}: fills={counts.get('fills', 0)} orders={counts.get('orders', 0)} "
                f"fill_rate={_fmt_num(metrics.get('fill_rate_pct'), 2, '%')} "
                f"markout_250ms={_fmt_num(metrics.get('markout_250ms_bps'), 2, 'bps')} "
                f"markout_1s={_fmt_num(metrics.get('markout_1s_bps'), 2, 'bps')} "
                f"markout_5s={_fmt_num(metrics.get('markout_5s_bps'), 2, 'bps')} "
                f"adverse={_fmt_pct_ratio(metrics.get('adverse_fill_ratio'), 1)} "
                f"lifetime={_fmt_num(metrics.get('quote_lifetime_ms_avg'), 1, 'ms')}"
            )
            emitted += 1
        except Exception as exc:
            lines.append(f"- {market_name}: unavailable ({exc})")
    if emitted == 0:
        lines.append('- unavailable (no recent events in last 2h)')
else:
    lines.append('- unavailable (markout analyzers not importable)')

lines.append("\n## Passed advisor proposals (latest 10)")
props = advisor_dir / 'proposals.jsonl'
rows = []
if props.exists():
    for ln in props.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get('guardrail_status') == 'passed':
            rows.append(r)
for r in rows[-10:]:
    lines.append(
        f"- {r.get('ts')} | {r.get('proposal_id')} | {r.get('market')} | "
        f"{r.get('param')}: {r.get('old')} -> {r.get('proposed')} | deadman={r.get('deadman')} | escalation={r.get('escalation_target')}"
    )
if not rows:
    lines.append('- none')

lines.append("\n## Current config values (policy/fallback keys)")
keys = []
policy_file = ROOT / 'mm_config/policy/whitelist.json'
if policy_file.exists():
    try:
        obj = json.loads(policy_file.read_text())
        keys = obj.get('allowed_keys', []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
    except Exception:
        keys = []
if not keys:
    keys = fallback_keys

for env in sorted(ROOT.glob('.env.*')):
    if not env.is_file():
        continue
    lines.append(f"- {env.name}")
    kv = {}
    for raw in env.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith('#'):
            continue
        if s.startswith('export '):
            s = s[len('export '):]
        if '=' not in s:
            continue
        k, v = s.split('=', 1)
        kv[k.strip()] = v.strip()
    matched = 0
    for key in keys:
        if key in kv:
            lines.append(f"  {key}={_mask_value(key, kv[key])}")
            matched += 1
    if matched == 0:
        lines.append('  (no policy keys found in this env file)')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
PY

PACKET_MEDIA="${MEDIA_DIR}/analyst_packet_${STAMP}.md"
if [[ -f "${ANALYST_PACKET_PATH}" ]]; then
  cp "${ANALYST_PACKET_PATH}" "${PACKET_MEDIA}"
  PACKET_MSG="Analyst packet ${TS_UTC}\nartifact:\n- ${ANALYST_PACKET_PATH}"
  send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_AUDITOR}" "${PACKET_MSG}" "${PACKET_MEDIA}" "${STATE_DIR}/send_packet_auditor_${STAMP}.json" "${STATE_DIR}/send_packet_auditor_${STAMP}.err" || true
  send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_MAIN}" "${PACKET_MSG}" "${PACKET_MEDIA}" "${STATE_DIR}/send_packet_main_${STAMP}.json" "${STATE_DIR}/send_packet_main_${STAMP}.err" || true
fi

jq -nc --arg sig "${SIGNATURE}" --arg ts "${TS_UTC}" '{last_signature:$sig,last_run_at:$ts}' > "${STATE_FILE}"
printf "%s status=ok proposals_delta=%s signature=%s packet=%s\n" "${TS_UTC}" "${DELTA}" "${SIGNATURE}" "${ANALYST_PACKET_PATH}" >> "${STATE_DIR}/runs.log"
