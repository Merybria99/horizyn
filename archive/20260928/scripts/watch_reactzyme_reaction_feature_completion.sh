#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT CONTROLLER_PID" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$(realpath "$1")"
CONTROLLER_PID="$2"
RUNTIME_ENV="$RUN_ROOT/runtime.env"

if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing runtime env: $RUNTIME_ENV" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$RUNTIME_ENV"
cd "$ROOT"

echo "Watching controller PID $CONTROLLER_PID for $RUN_ROOT"
while kill -0 "$CONTROLLER_PID" 2>/dev/null; do
  command_line="$(
    tr '\0' ' ' < "/proc/$CONTROLLER_PID/cmdline" 2>/dev/null || true
  )"
  if [[ "$command_line" != *"$RUN_ROOT"* ]]; then
    echo "PID $CONTROLLER_PID no longer belongs to this run"
    break
  fi
  sleep 60
done

echo "Controller exited; validating and generating the TIGER comparison"
"$PYTHON_BIN" scripts/report_reactzyme_reaction_feature_ablation.py \
  --run-root "$RUN_ROOT"
echo "Completion report: $RUN_ROOT/eval/tiger_comparison.md"
