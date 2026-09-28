#!/usr/bin/env bash
# Reload the validation fix on Glutamine; reuse the already prepared experiment.
set -euo pipefail
cd /
[[ "$(/bin/hostname -s)" == glutamine ]] || {
  echo "Run this on glutamine. No processes were stopped or launched." >&2
  exit 1
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_old="$circe_project/runs/reactzyme_reaction_smi_tyrosine_recipe_glutamine_20260911_031941"
circe_new="$circe_old/training_validation_fixed"
circe_old_session=reactzyme_circe_v2_glutamine
circe_session=reactzyme_circe_v2_glutamine_fixed
circe_python=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python

[[ -s "$circe_old/configs/train.yaml" && -x "$circe_python" ]] || {
  echo "Prepared configuration or Python environment is unavailable; stopping."; exit 1;
}
[[ ! -e "$circe_new" && ! -e "$circe_new.log" ]] || {
  echo "Replacement output already exists; no duplicate launched."; exit 1;
}
if /usr/bin/tmux has-session -t "=$circe_session" 2>/dev/null; then
  echo "Replacement session already exists; stopping."; exit 1
fi

if /usr/bin/tmux has-session -t "=$circe_old_session" 2>/dev/null; then
  circe_pid=$(/usr/bin/timeout --kill-after=2s 5s /usr/bin/tmux list-panes \
    -s -t "=$circe_old_session" -F '#{pane_pid}')
  [[ "$circe_pid" =~ ^[0-9]+$ ]] || {
    echo "Expected exactly one old training pane; no signals sent."; exit 1;
  }
  echo "Verifying and stopping the previous ReactZyme training only..."
  /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C /usr/bin/python3 -I -B \
    "$circe_project/scripts/stop_horizyn1_circe_v2.py" \
    --pid "$circe_pid" --run-root "$circe_old/training" \
    --timeout 30 --force-after-timeout
fi

# The process tree can exit before the driver finishes removing its CUDA
# contexts. Poll without signalling any additional process or resetting GPUs.
circe_release_deadline=$((SECONDS + 30))
while :; do
  circe_busy=$(/usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
    --query-compute-apps=pid --format=csv,noheader)
  [[ -n "$circe_busy" ]] || break
  (( SECONDS < circe_release_deadline )) || break
  echo "Waiting for GPUs 1 and 2 to be released; driver still reports PIDs: $circe_busy"
  /usr/bin/sleep 5
done
[[ -z "$circe_busy" ]] || {
  echo "GPUs 1 or 2 are still occupied; replacement NOT launched. PIDs: $circe_busy"
  exit 1
}
/usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
  --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F, '
    {print; seen[$1+0]++; if (($1+0 != 1 && $1+0 != 2) || $2 !~ /A100/ || $3+0 < 60000) bad=1}
    END {exit (bad || seen[1] != 1 || seen[2] != 1)}'

circe_resume=()
for circe_checkpoint in "$circe_old/training/checkpoints/recovery/last.ckpt" \
                        "$circe_old/training/checkpoints/last.ckpt"; do
  if [[ -s "$circe_checkpoint" ]]; then
    circe_resume=(--resume "$circe_checkpoint")
    echo "Resuming: $circe_checkpoint"
    break
  fi
done
if [[ ${#circe_resume[@]} == 0 ]]; then
  echo "No saved checkpoint: restarting from initialization; existing features/index reused."
fi
circe_args=(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C
  CUDA_VISIBLE_DEVICES=1,2 CUDA_DEVICE_ORDER=PCI_BUS_ID
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  "$circe_python" -B -u "$circe_project/scripts/train_protein_pooling_fast_io.py"
  --config "$circe_old/configs/train.yaml" --io-mode fast --io-prefetch 1
  --io-recovery-every-n-train-steps 100 --io-output-dir "$circe_new"
  --wandb-mode disabled "${circe_resume[@]}")
printf -v circe_command '%q ' "${circe_args[@]}"
/usr/bin/tmux new-session -d -s "$circe_session" -c "$circe_project" \
  "exec $circe_command >> $circe_new.log 2>&1"
echo "Detached session: $circe_session"
echo "Watch: tail -f $circe_new.log"
