#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
session=cyp_external_v1
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "Session already exists: $session; no duplicate launched."; exit 1
fi
mkdir -p runs/cyp_external_v1
printf -v command '%q ' "$PWD/.deps/cyp-external-env/bin/python" -u "$PWD/scripts/run_cyp_external.py" run "$@"
tmux new-session -d -s "$session" -c "$PWD" "exec $command >> '$PWD/runs/cyp_external_v1/pipeline.log' 2>&1"
echo "Detached session: $session"
echo "Log: $PWD/runs/cyp_external_v1/pipeline.log"
