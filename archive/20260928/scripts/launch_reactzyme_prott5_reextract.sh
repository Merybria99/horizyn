#!/usr/bin/env bash
set -euo pipefail

[[ "$(hostname -s)" == "slurm-node-014" ]] || {
  echo "Run this on slurm-node-014. Nothing was launched."
  exit 1
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_root="$circe_project/runs/reactzyme_prott5_reextract_20260910"
circe_session=reactzyme_prott5_reextract
cd "$circe_project"
command -v tmux >/dev/null
if tmux has-session -t "$circe_session" 2>/dev/null; then
  echo "Session already exists: $circe_session. Nothing was launched."
  exit 1
fi
if [[ -d "$circe_root/.controller.lock" ]]; then
  echo "Controller lock exists. Verify its recorded process has stopped before retrying."
  exit 1
fi
if [[ -s "$circe_root/READY.json" ]]; then
  echo "A completed cache already exists: $circe_root/READY.json"
  exit 0
fi
printf -v circe_command '%q ' /usr/bin/python3 -u \
  scripts/rebuild_reactzyme_prott5_cache.py run \
  --run-root "$circe_root" --gpus 0,1,2,3 \
  --batch-size 256 --max-tokens 65536
printf -v circe_log '%q' "$circe_root.log"
tmux new-session -d -s "$circe_session" -c "$circe_project" \
  "exec $circe_command >> $circe_log 2>&1"
echo "Detached cache rebuild: $circe_session"
echo "Pipeline log: $circe_root.log"
echo "This extracts and verifies features only. It neither starts training nor deletes the old cache."
