#!/usr/bin/env bash
# One explicitly selected, idle GPU. Never signals or replaces another job.
set -euo pipefail
circe_gpu=${1:?Usage: bash scripts/launch_cyp_specificity.sh GPU_ID [--models f3 residual circe_v2]}
shift
[[ "$circe_gpu" =~ ^[0-9]+$ ]] || { echo 'GPU_ID must be one physical GPU index'; exit 1; }
for circe_option in "$@"; do
  case "$circe_option" in
    --run-root|--run-root=*|--device|--device=*)
      echo 'Use the Python runner directly to override the fixed launcher run root/device.'; exit 1 ;;
  esac
done
circe_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$circe_project"
circe_python="$circe_project/../.capability-run-py/bin/python"
circe_run="$circe_project/runs/cyp_specificity_v1"
circe_session=cyp_specificity_v1
command -v tmux >/dev/null
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Session already exists: $circe_session. Nothing was stopped."
  exit 1
fi
circe_uuid=$(timeout --kill-after=2s 10s nvidia-smi -i "$circe_gpu" --query-gpu=uuid --format=csv,noheader,nounits)
circe_apps=$(timeout --kill-after=2s 10s nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
circe_busy=$(awk -F, -v gpu="$circe_uuid" '$1 == gpu {print $2}' <<< "$circe_apps")
[[ -z "$circe_busy" ]] || { echo "GPU $circe_gpu is occupied; nothing launched."; exit 1; }
mkdir -p "$circe_run"
printf -v circe_cmd '%q ' env -u BASH_ENV -u ENV \
  CUDA_VISIBLE_DEVICES="$circe_gpu" PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$circe_python" scripts/run_cyp_specificity.py all --device cuda --run-root "$circe_run" "$@"
printf -v circe_log '%q' "$circe_run/pipeline.log"
tmux new-session -d -s "$circe_session" -c "$circe_project" \
  -e BASH_ENV=/dev/null -e ENV=/dev/null \
  "exec $circe_cmd >> $circe_log 2>&1"
echo "Detached session: $circe_session"
echo "Log: $circe_run/pipeline.log"
