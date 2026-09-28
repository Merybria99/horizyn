#!/usr/bin/env bash
# Run read-only integrity checks when reconstruction stages finish.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/data/reconstructed/horizyn1_2023_05}"
PIPELINE_SESSION="${PIPELINE_SESSION:-horizyn1_dataset}"
PYTHON="${PYTHON:-python3}"
AUDITOR="${PROJECT_ROOT}/scripts/audit_horizyn1_reconstruction.py"
LOG_DIR="${RUN_ROOT}/logs"

log() {
  printf '[%s] %s\n' "$(date -Iseconds)" "$*"
}

wait_for_manifest() {
  local manifest="$1"
  log "Waiting for ${manifest}"
  while [[ ! -s "${manifest}" ]]; do
    if ! tmux has-session -t "${PIPELINE_SESSION}" 2>/dev/null; then
      # The manifest may have appeared between the two checks.
      if [[ -s "${manifest}" ]]; then
        break
      fi
      log "Pipeline session ended before the required manifest appeared; audit not started."
      return 2
    fi
    sleep 30
  done
}

run_audit() {
  local stage="$1"
  local manifest="$2"
  local report="${LOG_DIR}/${stage}_integrity_audit.json"
  local audit_status
  wait_for_manifest "${manifest}"
  log "Starting ${stage} integrity audit; report: ${report}"
  if nice -n 10 "${PYTHON}" "${AUDITOR}" \
    --run-root "${RUN_ROOT}" --stage "${stage}" --output "${report}"; then
    log "${stage} integrity audit passed."
  else
    audit_status=$?
    log "${stage} integrity audit did not pass (exit ${audit_status}); inspect the report."
    return "${audit_status}"
  fi
}

if [[ ! -d "${RUN_ROOT}" || ! -f "${AUDITOR}" ]]; then
  log "Missing reconstruction directory or audit script."
  exit 2
fi
mkdir -p "${LOG_DIR}"
run_audit raw "${RUN_ROOT}/raw/raw_manifest.json"
run_audit clustered "${RUN_ROOT}/clustered/clustered_manifest.json"
log "Both reconstruction integrity audits passed."
