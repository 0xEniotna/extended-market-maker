#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JOURNAL_DIR="${REPO_ROOT}/data/mm_journal"

BASE_ENV="${BASE_ENV:-.env.cop}"
CONTROLLER_ID="${CONTROLLER_ID:-}"
ITERATIONS="${ITERATIONS:-999999}"
ADVISOR_INTERVAL="${ADVISOR_INTERVAL:-1800}"
JOBS_JSON="${JOBS_JSON:-/home/flexouille/.openclaw/cron/jobs.json}"
DEADMAN_WINDOW_S="${DEADMAN_WINDOW_S:-3600}"
DEADMAN_COOLDOWN_S="${DEADMAN_COOLDOWN_S:-1800}"
INVENTORY_WINDOW_S="${INVENTORY_WINDOW_S:-7200}"
MAX_CHANGES_PER_HOUR="${MAX_CHANGES_PER_HOUR:-10}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -n "${CONTROLLER_ID}" ]]; then
  PID_FILE="${JOURNAL_DIR}/mm_openclaw_controller_${CONTROLLER_ID}.pid"
  LOG_FILE="${JOURNAL_DIR}/mm_openclaw_controller_${CONTROLLER_ID}.log"
else
  PID_FILE="${JOURNAL_DIR}/mm_openclaw_controller.pid"
  LOG_FILE="${JOURNAL_DIR}/mm_openclaw_controller.log"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/mm_openclaw_controller.sh start
  scripts/mm_openclaw_controller.sh stop
  scripts/mm_openclaw_controller.sh restart
  scripts/mm_openclaw_controller.sh status
  scripts/mm_openclaw_controller.sh logs

Config via env vars:
  BASE_ENV=.env.cop
  CONTROLLER_ID=amzn
  ITERATIONS=999999
  ADVISOR_INTERVAL=1800
  JOBS_JSON=/home/flexouille/.openclaw/cron/jobs.json
  DEADMAN_WINDOW_S=3600
  DEADMAN_COOLDOWN_S=1800
  INVENTORY_WINDOW_S=7200
  MAX_CHANGES_PER_HOUR=10
  PYTHON_BIN=python3
  # Defaults to repo .venv python when present.
EOF
}

is_running() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "${PID_FILE}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi
  kill -0 "${pid}" 2>/dev/null
}

start_controller() {
  mkdir -p "${JOURNAL_DIR}"

  if is_running; then
    echo "Controller already running (pid=$(cat "${PID_FILE}"))."
    exit 0
  fi

  cd "${REPO_ROOT}"
  local cmd=(
    "${PYTHON_BIN}" scripts/mm_advisor_loop.py
    --repo "${REPO_ROOT}"
    --env-path "${BASE_ENV}"
    --iterations "${ITERATIONS}"
    --sleep-s "${ADVISOR_INTERVAL}"
    --jobs-json "${JOBS_JSON}"
    --deadman-window-s "${DEADMAN_WINDOW_S}"
    --deadman-cooldown-s "${DEADMAN_COOLDOWN_S}"
    --inventory-window-s "${INVENTORY_WINDOW_S}"
    --max-changes-per-hour "${MAX_CHANGES_PER_HOUR}"
  )

  nohup "${cmd[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "$!" > "${PID_FILE}"
  echo "Controller started (pid=$!, log=${LOG_FILE})."
}

stop_controller() {
  if ! is_running; then
    echo "Controller is not running."
    rm -f "${PID_FILE}"
    exit 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  kill -INT "${pid}" 2>/dev/null || true

  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 1
    waited=$((waited + 1))
    if [[ "${waited}" -ge 30 ]]; then
      kill -TERM "${pid}" 2>/dev/null || true
      break
    fi
  done

  rm -f "${PID_FILE}"
  echo "Controller stopped."
}

status_controller() {
  if is_running; then
    echo "RUNNING pid=$(cat "${PID_FILE}")"
    echo "log=${LOG_FILE}"
  else
    echo "STOPPED"
    echo "log=${LOG_FILE}"
  fi
}

logs_controller() {
  mkdir -p "${JOURNAL_DIR}"
  touch "${LOG_FILE}"
  tail -n 120 -f "${LOG_FILE}"
}

action="${1:-}"
case "${action}" in
  start)
    start_controller
    ;;
  stop)
    stop_controller
    ;;
  restart)
    stop_controller
    start_controller
    ;;
  status)
    status_controller
    ;;
  logs)
    logs_controller
    ;;
  *)
    usage
    exit 1
    ;;
esac
