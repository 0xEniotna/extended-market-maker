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
ADVISOR_PROPOSALS_ENABLED="${ADVISOR_PROPOSALS_ENABLED:-0}"

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
DELTA=0
SIGNATURE="${LAST_SIGNATURE}"
ADVISOR_PROPOSALS_ENABLED_LC="$(printf '%s' "${ADVISOR_PROPOSALS_ENABLED}" | tr '[:upper:]' '[:lower:]')"
if [[ "${ADVISOR_PROPOSALS_ENABLED_LC}" == "1" || "${ADVISOR_PROPOSALS_ENABLED_LC}" == "true" || "${ADVISOR_PROPOSALS_ENABLED_LC}" == "yes" || "${ADVISOR_PROPOSALS_ENABLED_LC}" == "on" ]]; then
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
else
  printf "%s status=ok advisor_loop=disabled packet_only=true\n" "${TS_UTC}" >> "${STATE_DIR}/runs.log"
fi

"${PYTHON_BIN}" scripts/tools/audit_position_risk.py \
  --journal-dir data/mm_journal \
  --output-dir data/mm_audit \
  --lookback-hours 2 \
  >"${STATE_DIR}/position_risk_${STAMP}.md" 2>"${STATE_DIR}/position_risk_${STAMP}.err" || true

PNL_TXT="/tmp/mm_advisor_pnl_${STAMP}.txt"
"${PYTHON_BIN}" -m market_maker.cli pnl --all \
  --env .env.eth \
  --since "$(date -u +%Y-%m-%dT00:00:00Z)" \
  >"${PNL_TXT}" 2>"${STATE_DIR}/fetch_total_pnl_${STAMP}.err" || true

REPO_ROOT="${REPO_ROOT}" PNL_TXT="${PNL_TXT}" ANALYST_PACKET_PATH="${ANALYST_PACKET_PATH}" ADVISOR_PROPOSALS_ENABLED="${ADVISOR_PROPOSALS_ENABLED_LC}" "${PYTHON_BIN}" - <<'PY'
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

ROOT = Path(os.environ['REPO_ROOT'])
PNL_TXT = Path(os.environ['PNL_TXT'])
OUT = Path(os.environ['ANALYST_PACKET_PATH'])
PROPOSALS_ENABLED = str(os.environ.get('ADVISOR_PROPOSALS_ENABLED', '0')).lower() in {'1', 'true', 'yes', 'on'}

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

def _parse_env_file(path):
    kv = {}
    try:
        for raw in path.read_text().splitlines():
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            if s.startswith('export '):
                s = s[len('export '):]
            if '=' not in s:
                continue
            k, v = s.split('=', 1)
            kv[k.strip()] = v.strip()
    except Exception:
        return {}
    return kv

def _norm_market_name(value):
    return str(value or '').strip().upper()

def _to_decimal(value):
    try:
        return Decimal(str(value))
    except Exception:
        return None

def _build_env_index(root):
    snapshots = []
    max_position_by_market = {}
    market_by_env_path = {}
    for env in sorted(root.glob('.env.*')):
        if not env.is_file():
            continue
        kv = _parse_env_file(env)
        snapshots.append((env, kv))
        market_name = _norm_market_name(kv.get('MM_MARKET_NAME'))
        max_position = _to_decimal(kv.get('MM_MAX_POSITION_SIZE'))
        if market_name and max_position is not None:
            max_position_by_market[market_name] = max_position
        if market_name:
            market_by_env_path[str(env)] = market_name
            try:
                market_by_env_path[str(env.resolve())] = market_name
            except Exception:
                pass
    return snapshots, max_position_by_market, market_by_env_path

def _resolve_market_from_env_path(path_str, market_by_env_path):
    if not path_str:
        return ''
    path = Path(str(path_str))
    keys = [str(path)]
    if path.exists():
        try:
            keys.append(str(path.resolve()))
        except Exception:
            pass
    for key in keys:
        if key in market_by_env_path:
            return market_by_env_path[key]
    if path.exists():
        return _norm_market_name(_parse_env_file(path).get('MM_MARKET_NAME'))
    return ''

def _running_markets_from_mmctl(root, market_by_env_path):
    mmctl_path = Path('/home/flexouille/bin/mmctl')
    if not mmctl_path.exists():
        return set(), 'mmctl_not_found'
    try:
        proc = subprocess.run(
            [str(mmctl_path), 'status', '--json'],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return set(), f"mmctl_exec_error: {exc}"
    if proc.returncode != 0:
        return set(), f"mmctl_status_failed: rc={proc.returncode}"
    try:
        payload = json.loads(proc.stdout or '{}')
    except Exception as exc:
        return set(), f"mmctl_json_error: {exc}"
    markets = set()
    for row in payload.get('markets', []):
        if not isinstance(row, dict):
            continue
        if not bool(row.get('running')):
            continue
        market_name = _norm_market_name(
            row.get('market_name')
            or row.get('marketName')
            or row.get('mm_market_name')
            or row.get('mmMarketName')
        )
        if not market_name:
            market_name = _resolve_market_from_env_path(row.get('envPath'), market_by_env_path)
        if market_name:
            markets.add(market_name)
    return markets, None

def _running_markets_from_pid_files(root, market_by_env_path):
    markets = set()
    journal_dir_local = root / 'data' / 'mm_journal'
    if not journal_dir_local.exists():
        return markets
    for pid_file in sorted(journal_dir_local.glob('mm_openclaw_controller_*.pid')):
        try:
            pid = int(pid_file.read_text().strip())
        except Exception:
            continue
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except Exception:
            continue
        stem = pid_file.stem
        marker = 'env_'
        idx = stem.rfind(marker)
        slug = stem[idx + len(marker):] if idx >= 0 else ''
        env_path = root / f'.env.{slug}' if slug else None
        market_name = ''
        if env_path is not None:
            market_name = _resolve_market_from_env_path(str(env_path), market_by_env_path)
        if market_name:
            markets.add(market_name)
    return markets

async def _fetch_live_positions(root):
    try:
        sys.path.insert(0, str(root / 'src'))
        from dotenv import load_dotenv
        from market_maker.config import MarketMakerSettings
        from x10.perpetual.accounts import StarkPerpetualAccount
        from x10.perpetual.positions import PositionSide
        from x10.perpetual.trading_client import PerpetualTradingClient
        from x10.utils.http import ResponseStatus
    except Exception as exc:
        return {}, f"imports_unavailable: {exc}"

    env_file = root / '.env.eth'
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    else:
        load_dotenv()

    try:
        settings = MarketMakerSettings()
    except Exception as exc:
        return {}, f"settings_error: {exc}"
    if not settings.is_configured:
        return {}, "missing_credentials"

    account = StarkPerpetualAccount(
        vault=int(settings.vault_id),
        private_key=settings.stark_private_key,
        public_key=settings.stark_public_key,
        api_key=settings.api_key,
    )
    client = PerpetualTradingClient(settings.endpoint_config, account)
    try:
        resp = await client.account.get_positions(market_names=None)
        if resp.status != ResponseStatus.OK:
            return {}, f"get_positions_failed: {resp.status}"
        rows = resp.data or []
        merged = {}
        for pos in rows:
            market = _norm_market_name(getattr(pos, 'market', ''))
            if not market:
                continue
            side = getattr(pos, 'side', None)
            sign = Decimal('1') if side == PositionSide.LONG else Decimal('-1')
            size = _to_decimal(getattr(pos, 'size', 0)) or Decimal('0')
            value = _to_decimal(getattr(pos, 'value', 0)) or Decimal('0')
            unrealized = _to_decimal(getattr(pos, 'unrealised_pnl', 0)) or Decimal('0')
            entry = merged.setdefault(
                market,
                {
                    'position': Decimal('0'),
                    'exposure_usd': Decimal('0'),
                    'unrealized_pnl_usd': Decimal('0'),
                },
            )
            entry['position'] += sign * size
            entry['exposure_usd'] += sign * value
            entry['unrealized_pnl_usd'] += unrealized
        return merged, None
    except Exception as exc:
        return {}, f"live_position_error: {exc}"
    finally:
        try:
            await client.close()
        except Exception:
            pass

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
env_snapshots, max_position_by_market, market_by_env_path = _build_env_index(ROOT)
running_markets, running_markets_error = _running_markets_from_mmctl(ROOT, market_by_env_path)
if not running_markets:
    fallback_running_markets = _running_markets_from_pid_files(ROOT, market_by_env_path)
    if fallback_running_markets:
        running_markets = fallback_running_markets
        if running_markets_error:
            running_markets_error = f"{running_markets_error}; pid_file_fallback_used"

mid_prices = {}
if discover_market_journals and journal_dir.exists():
    try:
        for market_name, journal_path in sorted(discover_market_journals(journal_dir).items()):
            if running_markets and market_name not in running_markets:
                continue
            mid = _latest_mid_from_journal(journal_path)
            if mid is not None:
                mid_prices[market_name] = mid
    except Exception:
        mid_prices = {}

lines = [f"# Analyst Context Packet ({datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})"]

lines.append("\n## Running instances")
if running_markets:
    for market_name in sorted(running_markets):
        lines.append(f"- {market_name}")
else:
    lines.append(f"- unavailable ({running_markets_error or 'none running'})")

lines.append("\n## Current mid prices")
if running_markets:
    for market_name in sorted(running_markets):
        lines.append(f"- {market_name}: mid={_fmt_num(mid_prices.get(market_name), 8)}")
elif mid_prices:
    for market_name in sorted(mid_prices):
        lines.append(f"- {market_name}: mid={_fmt_num(mid_prices.get(market_name), 8)}")
else:
    lines.append('- unavailable (no mid price found)')

live_positions, live_positions_error = asyncio.run(_fetch_live_positions(ROOT))
lines.append("\n## Current positions on exchange (live)")
if live_positions_error:
    lines.append(f"- unavailable ({live_positions_error})")
else:
    open_markets = []
    for market_name, row in live_positions.items():
        if running_markets and market_name not in running_markets:
            continue
        if abs(row.get('position', Decimal('0'))) > Decimal('0.0000001'):
            open_markets.append((market_name, row))
    if not open_markets:
        lines.append('- no open positions on exchange')
    else:
        for market_name, row in sorted(open_markets):
            position = row.get('position', Decimal('0'))
            exposure = row.get('exposure_usd', Decimal('0'))
            unrealized = row.get('unrealized_pnl_usd', Decimal('0'))
            max_position = max_position_by_market.get(market_name)
            util_pct = None
            if max_position is not None and max_position != 0:
                try:
                    util_pct = (abs(position) / abs(max_position)) * Decimal('100')
                except Exception:
                    util_pct = None
            lines.append(
                f"- {market_name}: mid={_fmt_num(mid_prices.get(market_name), 8)} "
                f"position={_fmt_num(position, 6)} util={_fmt_num(util_pct, 2, '%')} "
                f"exposure_usd={_fmt_num(exposure, 2)} unrealized_pnl_usd={_fmt_num(unrealized, 2)}"
            )

risk_path = ROOT / 'data/mm_audit/position_risk.json'
lines.append("\n## Journal inventory risk snapshot (last 2h)")
if risk_path.exists():
    risk = json.loads(risk_path.read_text())
    markets = risk.get('markets', {})
    emitted_risk = 0
    for m in sorted(markets):
        if running_markets and _norm_market_name(m) not in running_markets:
            continue
        row = markets[m]
        lines.append(
            f"- {m}: journal_position={row.get('current_position')} util={row.get('inventory_utilization_pct')}% "
            f"exposure_usd={row.get('directional_exposure_usd')} flags={','.join(row.get('flags', [])) or '-'}"
        )
        emitted_risk += 1
    if emitted_risk == 0:
        lines.append('- unavailable (no running markets in risk snapshot)')
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
        if running_markets and market_name not in running_markets:
            continue
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
if not PROPOSALS_ENABLED:
    lines.append('- advisor proposal loop is disabled on this host')
else:
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
                market_name = _norm_market_name(r.get('market'))
                if running_markets and market_name and market_name not in running_markets:
                    continue
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

emitted_env = 0
for env, kv in env_snapshots:
    market_name = _norm_market_name(kv.get('MM_MARKET_NAME'))
    if running_markets and market_name and market_name not in running_markets:
        continue
    lines.append(f"- {env.name}")
    emitted_env += 1
    matched = 0
    for key in keys:
        if key in kv:
            lines.append(f"  {key}={_mask_value(key, kv[key])}")
            matched += 1
    if matched == 0:
        lines.append('  (no policy keys found in this env file)')
if emitted_env == 0:
    lines.append('- no running market env snapshots')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
PY

PACKET_MEDIA="${MEDIA_DIR}/analyst_packet_${STAMP}.md"
if [[ -f "${ANALYST_PACKET_PATH}" ]]; then
  cp "${ANALYST_PACKET_PATH}" "${PACKET_MEDIA}"
  PACKET_MSG="Analyst packet ${TS_UTC}\nartifact:\n- ${ANALYST_PACKET_PATH}"
  send_discord "${DISCORD_ACCOUNT_PUBLISHER}" "${DISCORD_CHANNEL_ID_AUDITOR}" "${PACKET_MSG}" "${PACKET_MEDIA}" "${STATE_DIR}/send_packet_auditor_${STAMP}.json" "${STATE_DIR}/send_packet_auditor_${STAMP}.err" || true
fi

jq -nc --arg sig "${SIGNATURE}" --arg ts "${TS_UTC}" '{last_signature:$sig,last_run_at:$ts}' > "${STATE_FILE}"
printf "%s status=ok proposals_delta=%s signature=%s packet=%s\n" "${TS_UTC}" "${DELTA}" "${SIGNATURE}" "${ANALYST_PACKET_PATH}" >> "${STATE_DIR}/runs.log"
