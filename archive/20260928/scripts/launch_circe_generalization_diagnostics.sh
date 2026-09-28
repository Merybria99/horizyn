#!/usr/bin/env bash
# Detached, validation-only diagnostics. Never stop or share another GPU job.
set -euo pipefail
circe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
circe_output="$circe_root/runs/circe_generalization_diagnostics_v1"
circe_args=("$@")
circe_has_gpus=false
while (($#)); do
  case "$1" in
    --output) circe_output=${2:?Missing output directory}; shift 2 ;;
    --output=*) circe_output=${1#*=}; shift ;;
    --gpus) : "${2:?Missing GPU IDs}"; circe_has_gpus=true; shift 2 ;;
    --gpus=*) circe_has_gpus=true; shift ;;
    *) shift ;;
  esac
done
if ! "$circe_has_gpus"; then
  echo "Usage: bash $0 --gpus FREE_GPU_IDS [--threads 8] [--target-batch-size 128]"
  echo "Example: --gpus 2 (only if GPU 2 is free). No GPUs are stopped or shared."
  exit 2
fi
command -v tmux >/dev/null
cd -- "$circe_root"
mkdir -p -- "$circe_output"
circe_output=$(cd -- "$circe_output" && pwd)
# A path-derived session name avoids collisions and tmux's special '.' syntax.
circe_digest=$(printf '%s' "$circe_output" | sha256sum)
circe_session="circe_diagnostics_${circe_digest:0:12}"
if tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Already launched: $circe_session"
  echo "Watch: tail -f $circe_output/pipeline.log"
  exit 1
fi
printf -v circe_command '%q ' /usr/bin/env -u BASH_ENV -u ENV \
  PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$circe_root/../env/bin/python" "$circe_root/scripts/run_circe_generalization_diagnostics.py" \
  all "${circe_args[@]}"
printf -v circe_log '%q' "$circe_output/pipeline.log"
tmux new-session -d -s "$circe_session" -c "$circe_root" \
  /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc -c \
  "exec $circe_command >> $circe_log 2>&1"
echo "Detached launch requested: $circe_session"
echo "Preflight will refuse occupied GPUs or changed experiment inputs."
echo "Watch: tail -f $circe_output/pipeline.log"
