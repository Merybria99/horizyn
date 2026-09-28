#!/usr/bin/env bash
# Dedicated detached reaction-smi run. No extraction or old-job cancellation.
set -euo pipefail

[[ "$(/bin/hostname -s)" == "glutamine" ]] || {
  echo "Run this on glutamine. Nothing was launched." >&2
  exit 1
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_launcher="$circe_project/scripts/launch_reactzyme_glutamine_cached.sh"
circe_python=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python
circe_session=reactzyme_circe_v2_glutamine

if [[ "${1:-launch}" == launch ]]; then
  cd /
  if /usr/bin/tmux has-session -t "=$circe_session" 2>/dev/null; then
    echo "Session already exists: $circe_session. No duplicate launched." >&2
    exit 1
  fi
  circe_bootdir=$(/usr/bin/mktemp -d /tmp/reactzyme_train.XXXXXX)
  printf 'Detached launch requested on glutamine; GPUs 1 and 2 only.\n' > "$circe_bootdir/pipeline.log"
  /usr/bin/tmux new-session -d -s "$circe_session" -c / \
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /bin/bash --noprofile --norc "$circe_launcher" run "$circe_bootdir"
  echo "Detached session: $circe_session"
  echo "Watch: tail -f $circe_bootdir/pipeline.log"
  exit 0
fi

[[ "${1:-}" == run && "${2:-}" == /tmp/reactzyme_train.* && -d "$2" ]] || {
  echo "Invalid internal launcher arguments" >&2; exit 1;
}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2
exec >> "$2/pipeline.log" 2>&1
echo "Launcher running; checking selected GPUs before shared-storage setup."

gpu_check() {
  local circe_busy
  circe_busy=$(/usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
    --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$circe_busy" ]] || { echo "GPUs 1 or 2 are occupied. No processes were stopped."; return 1; }
  /usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
    --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
    | /usr/bin/awk -F, '
      {if (($1+0 != 1 && $1+0 != 2) || $2 !~ /A100/ || $3+0 < 60000) bad=1; print; n++}
      END {exit (bad || n != 2)}'
}

gpu_check
cd "$circe_project"
circe_run="$circe_project/runs/reactzyme_reaction_smi_tyrosine_recipe_glutamine_$(/bin/date -u +%Y%m%d_%H%M%S)"
/bin/mkdir "$circe_run"
echo "Run directory: $circe_run"
# Keep the local bootstrap log even if shared logging later stalls.
exec > >(/usr/bin/tee -a "$circe_run/pipeline.log") 2>&1
echo "Run directory: $circe_run"
echo "Stage 1/4: CPU cache-checker and preparation regression tests"
CUDA_VISIBLE_DEVICES= /usr/bin/timeout --kill-after=5s 180s \
  "$circe_python" -I -B -u tests/unit/test_check_reactzyme_prott5_cache.py
CUDA_VISIBLE_DEVICES= /usr/bin/timeout --kill-after=5s 180s \
  "$circe_python" -I -B -u tests/unit/test_reactzyme_tyrosine_recipe.py
echo "Stage 2/4: existing protein-cache screening (read only, no extraction)"
CUDA_VISIBLE_DEVICES= /usr/bin/timeout --kill-after=5s 300s \
  "$circe_python" -I -B -u scripts/check_reactzyme_prott5_cache.py --samples 32
echo "Stage 3/4: CPU train-only annotations/index and configuration"
CUDA_VISIBLE_DEVICES= /usr/bin/timeout --kill-after=5s 1800s \
  "$circe_python" -I -B -u scripts/prepare_reactzyme_tyrosine_recipe.py --run-root "$circe_run"
gpu_check
echo "Stage 4/4: reaction-smi training; GPUs 1,2; batch 2x200; no extraction"
exec "$circe_python" -B -u scripts/train_protein_pooling_fast_io.py \
  --config "$circe_run/configs/train.yaml" --io-mode fast --io-prefetch 1 \
  --io-output-dir "$circe_run/training" --wandb-mode disabled
