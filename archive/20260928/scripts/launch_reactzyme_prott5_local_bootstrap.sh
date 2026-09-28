#!/usr/bin/env bash
# When NFS script reads hang, paste this block into node014's terminal instead
# of asking that node to read this file. No shared-storage access precedes tmux.
(
set -euo pipefail
cd /
[[ "$(hostname -s)" == "slurm-node-014" ]] || {
  echo "Run this on slurm-node-014. Nothing was launched."
  exit 1
}
command -v tmux >/dev/null
circe_session=reactzyme_prott5_reextract
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Session already exists: $circe_session. Not launching another copy."
  exit 1
fi
circe_bootdir=$(mktemp -d /tmp/reactzyme_bootstrap.XXXXXX)
circe_bootlog="$circe_bootdir/bootstrap.log"
echo "Local bootstrap log: $circe_bootlog"
tmux new-session -d -s "$circe_session" -c / \
  /usr/bin/env -u BASH_ENV /bin/bash --noprofile --norc -c '
    exec > "$1" 2>&1
    set -euo pipefail
    echo "Local bootstrap started; no NFS launcher or login profile was read."
    circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
    circe_controller="$circe_project/scripts/rebuild_reactzyme_prott5_cache.py"
    echo "Checking extraction controller access (15-second limit)..."
    if timeout --kill-after=2s 15s head -c 65536 "$circe_controller" >/dev/null; then
      echo "Controller is readable. Starting its storage/runtime/pilot checks..."
    else
      circe_code=$?
      echo "STOPPED: controller read failed (exit $circe_code). No extraction started."
      echo "A local launcher cannot fix access to the remaining NFS dependencies."
      exit "$circe_code"
    fi
    exec /usr/bin/python3 -u "$circe_controller" run \
      --run-root "$circe_project/runs/reactzyme_prott5_reextract_20260910" \
      --gpus 0,1,2,3 --batch-size 256 --max-tokens 65536
  ' circe-bootstrap "$circe_bootlog"
echo "Detached session: $circe_session"
echo "Watch with: tail -f $circe_bootlog"
)
