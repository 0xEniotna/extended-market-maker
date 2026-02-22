#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOURNAL_DIR="${REPO_ROOT}/data/mm_journal"

CONTROLLER_PREFIX="${CONTROLLER_PREFIX:-fleet}"
ENV_LIST="${ENV_LIST:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/mm_openclaw_fleet.sh start <env1> [env2 ...]
  scripts/mm_openclaw_fleet.sh stop <env1> [env2 ...]
  scripts/mm_openclaw_fleet.sh restart <env1> [env2 ...]
  scripts/mm_openclaw_fleet.sh status <env1> [env2 ...]
  scripts/mm_openclaw_fleet.sh logs <env1> [env2 ...]

Inputs:
  - Pass env files as positional args (e.g. .env.amzn .env.pump)
  - Shorthand is accepted (e.g. amzn, AMZN-USD -> .env.amzn)
  - Or set ENV_LIST as comma-separated env files
  - If neither is provided, defaults to .env.cop

Config passthrough:
  CONTROLLER_PREFIX=fleet
  ITERATIONS=999999
  ANALYSIS_INTERVAL=60
  MIN_FILLS=10
  ASSUMED_FEE_BPS=0
  MAX_RUN_SECONDS=0
  ALLOW_MAINNET=1
  PYTHON_BIN=.venv/bin/python
EOF
}

trim() {
  local s="$1"
  # shellcheck disable=SC2001
  s="$(echo "${s}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
  echo "${s}"
}

controller_id_for_env() {
  local env_label="$1"
  local base
  base="$(basename "${env_label}")"
  # shellcheck disable=SC2001
  base="$(echo "${base}" | sed 's/[^[:alnum:]]/_/g;s/_\+/_/g;s/^_//;s/_$//')"
  if [[ -z "${base}" ]]; then
    base="default"
  fi
  echo "${CONTROLLER_PREFIX}_${base}"
}

resolve_env_path() {
  local env_label="$1"
  if [[ "${env_label}" = /* ]]; then
    echo "${env_label}"
  else
    echo "${REPO_ROOT}/${env_label}"
  fi
}

normalize_env_label() {
  local raw
  raw="$(trim "$1")"
  if [[ -z "${raw}" ]]; then
    echo ""
    return
  fi

  # Keep explicit paths / filenames unchanged.
  if [[ "${raw}" = /* || "${raw}" == .env* || "${raw}" == */* ]]; then
    echo "${raw}"
    return
  fi

  # Accept market shorthand like "AZTEC-USD" and plain aliases like "aztec".
  local lowered
  lowered="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${lowered}" == *-usd ]]; then
    lowered="${lowered%-usd}"
  fi
  echo ".env.${lowered}"
}

action="${1:-}"
case "${action}" in
  start|stop|restart|status|logs)
    shift || true
    ;;
  *)
    usage
    exit 1
    ;;
esac

declare -a envs=()
if [[ "$#" -gt 0 ]]; then
  envs=("$@")
elif [[ -n "${ENV_LIST}" ]]; then
  IFS=',' read -r -a raw <<< "${ENV_LIST}"
  for item in "${raw[@]}"; do
    item="$(trim "${item}")"
    if [[ -n "${item}" ]]; then
      envs+=("${item}")
    fi
  done
else
  envs=(".env.cop")
fi

if [[ "${#envs[@]}" -eq 0 ]]; then
  echo "No env files provided."
  exit 1
fi

declare -a normalized_envs=()
for env_label in "${envs[@]}"; do
  normalized="$(normalize_env_label "${env_label}")"
  if [[ -n "${normalized}" ]]; then
    normalized_envs+=("${normalized}")
  fi
done
envs=("${normalized_envs[@]}")

if [[ "${action}" == "logs" ]]; then
  mkdir -p "${JOURNAL_DIR}"
  declare -a logs=()
  for env_label in "${envs[@]}"; do
    controller_id="$(controller_id_for_env "${env_label}")"
    log_file="${JOURNAL_DIR}/mm_openclaw_controller_${controller_id}.log"
    touch "${log_file}"
    logs+=("${log_file}")
    echo "[${env_label}] log=${log_file}"
  done
  tail -n 120 -F "${logs[@]}"
  exit 0
fi

for env_label in "${envs[@]}"; do
  controller_id="$(controller_id_for_env "${env_label}")"
  env_path="$(resolve_env_path "${env_label}")"

  if [[ "${action}" == "start" || "${action}" == "restart" ]]; then
    if [[ ! -f "${env_path}" ]]; then
      echo "[${env_label}] missing env file: ${env_path}" >&2
      exit 1
    fi
  fi

  echo "[${env_label}] action=${action} controller_id=${controller_id}"
  (
    cd "${REPO_ROOT}"
    BASE_ENV="${env_path}" \
    CONTROLLER_ID="${controller_id}" \
    ITERATIONS="${ITERATIONS:-999999}" \
    ANALYSIS_INTERVAL="${ANALYSIS_INTERVAL:-60}" \
    MIN_FILLS="${MIN_FILLS:-10}" \
    ASSUMED_FEE_BPS="${ASSUMED_FEE_BPS:-0}" \
    MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-0}" \
    ALLOW_MAINNET="${ALLOW_MAINNET:-1}" \
    PYTHON_BIN="${PYTHON_BIN:-}" \
      scripts/mm_openclaw_controller.sh "${action}"
  )
done
