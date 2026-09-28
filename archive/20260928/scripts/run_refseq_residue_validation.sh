#!/usr/bin/env bash
set -eEuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
H5_PATH="${H5_PATH:-$PROJECT_ROOT/wet_lab/databases/refseq/prokaryotes/five_shards/proteins_prott5_residue.h5}"
VALIDATION_WORKERS="${VALIDATION_WORKERS:-3}"
ROWS_PER_CHUNK="${ROWS_PER_CHUNK:-32768}"
RUN_NAME="${RUN_NAME:-refseq_residue_validation_20260902}"
RUNTIME_ROOT="$PROJECT_ROOT/wet_lab/runtime/$RUN_NAME"
LOG_PATH="$PROJECT_ROOT/wet_lab/logs/$RUN_NAME.log"
PID_FILE="$PROJECT_ROOT/run_pids/$RUN_NAME.pid"

for value_name in VALIDATION_WORKERS ROWS_PER_CHUNK; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer, got %s\n' "$value_name" "$value" >&2
    exit 1
  fi
done

mkdir -p "$RUNTIME_ROOT/home" "$RUNTIME_ROOT/tmp" "$(dirname "$LOG_PATH")" \
  "$(dirname "$PID_FILE")"
if [[ -s "$PID_FILE" ]]; then
  existing_pid="$(cat "$PID_FILE")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    printf 'Residue validation is already running as PID %s\n' "$existing_pid" >&2
    exit 1
  fi
fi

exec >> "$LOG_PATH" 2>&1
VALIDATION_SHELL_PID="$BASHPID"
printf '%s\n' "$VALIDATION_SHELL_PID" > "$PID_FILE"
cleanup() {
  local status=$?
  if [[ "$BASHPID" != "$VALIDATION_SHELL_PID" ]]; then
    return
  fi
  if [[ -f "$PID_FILE" ]] && [[ "$(cat "$PID_FILE")" == "$VALIDATION_SHELL_PID" ]]; then
    rm -f "$PID_FILE"
  fi
  printf 'VALIDATION_EXIT=%s AT=%s\n' "$status" "$(date -Is)"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

export HOME="$RUNTIME_ROOT/home"
export TMPDIR="$RUNTIME_ROOT/tmp"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
printf 'VALIDATION_START=%s H5=%s WORKERS=%s ROWS_PER_CHUNK=%s\n' \
  "$(date -Is)" "$H5_PATH" "$VALIDATION_WORKERS" "$ROWS_PER_CHUNK"
"$PYTHON_BIN" -m scripts.validate_residue_hdf5 "$H5_PATH" \
  --rows-per-chunk "$ROWS_PER_CHUNK" \
  --progress-every-chunks 16 \
  --workers "$VALIDATION_WORKERS"
