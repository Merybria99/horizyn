#!/usr/bin/env bash
# One free GPU; existing residue features, checkpoint-specific resumable cache.
set -euo pipefail

project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project"
if [[ "${1:-}" == --parallel || "${1:-}" == --restart-parallel ]]; then
  [[ "$#" == 1 ]] || { echo "Usage: bash $0 --parallel or --restart-parallel"; exit 1; }
  circe_mode=--launch
  [[ "$1" != --restart-parallel ]] || circe_mode=--restart
  exec "$project/../.capability-run-py/bin/python" \
    "$project/scripts/run_case1_refseq_best16k_parallel.py" "$circe_mode"
fi
gpu="${1:?Usage: bash scripts/launch_case1_refseq_best16k.sh GPU_INDEX or --parallel}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "GPU_INDEX must be a physical GPU index."; exit 1; }
session=case1_refseq_best16k
run_root="$project/wet_lab/runs/refseq/prokaryotes/horizyn1_best16k"
python="$project/../.capability-run-py/bin/python"

check_gpu() {
  local uuid busy free_memory
  uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
  [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || return 1
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits |
    awk -F, -v uuid="$uuid" '{gsub(/[[:space:]]/, "", $1); if ($1 == uuid) print $2}')
  if [[ -n "$busy" ]]; then
    echo "GPU $gpu is occupied by PID(s): $busy. Nothing was stopped or launched."
    return 1
  fi
  free_memory=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ "$free_memory" =~ ^[0-9]+$ ]] && (( free_memory >= 40000 )) || {
    echo "This configuration requires a free GPU with at least 40,000 MiB available."
    return 1
  }
}

[[ -x "$python" ]] || { echo "Python missing: $python"; exit 1; }
check_gpu
if [[ "${2:-}" == --worker ]]; then
  # Hold one shared-storage lock across hosts. Never signal other jobs.
  exec 9>"$run_root/launch.lock"
  flock -n 9 || { echo "A RefSeq best16k run already holds the lock."; exit 1; }
  printf 'Starting Case 1 RefSeq on %s, physical GPU %s, at %s\n' "$(hostname -s)" "$gpu" "$(date -Is)"
  exec env -u WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE -u WET_LAB_RESIDUE_EMBEDDINGS_OVERRIDE \
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 HORIZYN_CPU_THREADS=4 \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    "$python" -m wet_lab.query --config wet_lab/configs/refseq/horizyn1_best16k.yaml
fi
[[ "$#" == 1 ]] || { echo "Usage: bash $0 GPU_INDEX"; exit 1; }
command -v tmux >/dev/null
command -v flock >/dev/null
if tmux has-session -t "=$session" 2>/dev/null; then
  echo "Session $session already exists; not starting another copy."
  exit 1
fi
mkdir -p "$run_root"
flock -n "$run_root/launch.lock" true || { echo "A RefSeq run already holds the lock."; exit 1; }
printf -v command '%q ' /usr/bin/env -u BASH_ENV -u ENV /bin/bash --noprofile --norc \
  "$project/scripts/launch_case1_refseq_best16k.sh" "$gpu" --worker
printf -v log_path '%q' "$run_root/pipeline.log"
tmux new-session -d -s "$session" -c "$project" "$command >> $log_path 2>&1"
echo "Detached launch requested: $session"
echo "Watch: tail -f $run_root/pipeline.log"
