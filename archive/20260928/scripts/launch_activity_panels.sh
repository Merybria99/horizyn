#!/usr/bin/env bash
set -euo pipefail
panel_gpu=${1:?Usage: bash scripts/launch_activity_panels.sh GPU_ID}
[[ "$panel_gpu" =~ ^[0-9]+$ ]] || { echo 'Specify one physical GPU index'; exit 1; }
panel_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$panel_root"
panel_session=activity_panels_nitrilase_v2
if tmux has-session -t "=$panel_session" 2>/dev/null; then
  echo 'Benchmark session already exists; no duplicate launched.'; exit 1
fi
panel_apps=$(nvidia-smi -i "$panel_gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "$panel_apps" ]] || { echo "GPU $panel_gpu is occupied; nothing launched."; exit 1; }
mkdir -p runs/activity_panels_nitrilase_v2
printf -v panel_command '%q ' env -u BASH_ENV -u ENV CUDA_VISIBLE_DEVICES="$panel_gpu" \
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 WANDB_MODE=disabled \
  "$panel_root/../.capability-run-py/bin/python" "$panel_root/scripts/run_activity_panels.py" all --device cuda
printf -v panel_log '%q' "$panel_root/runs/activity_panels_nitrilase_v2/pipeline.log"
printf -v panel_readout '%q ' env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$panel_root/../.capability-run-py/bin/python" "$panel_root/scripts/summarize_activity_panel_run.py"
tmux new-session -d -s "$panel_session" -c "$panel_root" -e BASH_ENV=/dev/null -e ENV=/dev/null \
  "$panel_command >> $panel_log 2>&1 && exec $panel_readout >> $panel_log 2>&1"
echo "Started $panel_session on idle GPU $panel_gpu"
echo "Report: $panel_root/runs/activity_panels_nitrilase_v2/report/readout.md"
