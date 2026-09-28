#!/usr/bin/env bash
# Resume the five-epoch run with corrected validation/early-stopping timing.
# Never stop processes or overwrite an earlier experiment.
set -euo pipefail
cd /
[[ "$(/bin/hostname -s)" == glutamine ]] || {
  echo "Run this on glutamine. Nothing launched." >&2; exit 1;
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_base="$circe_project/runs/reactzyme_reaction_smi_tyrosine_recipe_glutamine_20260911_031941"
circe_new="$circe_base/training_30epochs_resume_fixed"
circe_session=reactzyme_30epochs_glutamine_fixed
circe_config="$circe_base/configs/train_30epochs.yaml"
circe_checkpoint="$circe_base/training_validation_fixed/checkpoints/last.ckpt"
circe_python=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python

[[ -s "$circe_checkpoint" && -s "$circe_config" && -x "$circe_python" ]] || {
  echo "Checkpoint, configuration, or Python environment missing; stopping." >&2; exit 1;
}
[[ ! -e "$circe_new" && ! -e "$circe_new.log" ]] || {
  echo "Replacement output already exists; nothing overwritten." >&2; exit 1;
}
for circe_existing in reactzyme_30epochs_glutamine "$circe_session"; do
  if /usr/bin/tmux has-session -t "=$circe_existing" 2>/dev/null; then
    echo "Session $circe_existing already exists; check it before relaunching." >&2
    exit 1
  fi
done

# Verify the actual Glutamine Python/Lightning environment before committing
# another training run. The fixtures use CPU and tiny local temporary files.
circe_check_dir=$(/usr/bin/mktemp -d /tmp/reactzyme_resume_check.XXXXXX)
echo "Checking epoch-boundary resume on CPU before launching training..."
echo "Temporary regression artifacts: $circe_check_dir"
# Stage just this standalone regression file so pytest does not traverse the
# large shared tests tree (including unrelated slow NFS paths).
/usr/bin/cp -- "$circe_project/tests/unit/test_training_epoch_resume.py" \
  "$circe_check_dir/test_training_epoch_resume.py"
cd "$circe_project"
/usr/bin/timeout --kill-after=5s 240s /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
  CUDA_VISIBLE_DEVICES= PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$circe_python" -B -u -m pytest "$circe_check_dir/test_training_epoch_resume.py" \
  -q -c /dev/null --rootdir="$circe_check_dir" --confcutdir="$circe_check_dir" \
  -p no:cacheprovider --basetemp="$circe_check_dir/pytest"
echo "Resume regression checks passed; checking GPUs 1 and 2."

circe_busy=$(/usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
  --query-compute-apps=pid --format=csv,noheader)
[[ -z "$circe_busy" ]] || {
  echo "GPUs 1 or 2 are occupied. No processes were stopped. PIDs: $circe_busy" >&2; exit 1;
}
/usr/bin/timeout --kill-after=2s 10s /usr/bin/nvidia-smi -i 1,2 \
  --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
  | /usr/bin/awk -F, '
    {print; seen[$1+0]++; if (($1+0 != 1 && $1+0 != 2) || $2 !~ /A100/ || $3+0 < 60000) bad=1}
    END {exit (bad || seen[1] != 1 || seen[2] != 1)}'

circe_args=(/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C
  CUDA_VISIBLE_DEVICES=1,2 CUDA_DEVICE_ORDER=PCI_BUS_ID
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  "$circe_python" -B -u "$circe_project/scripts/train_protein_pooling_fast_io.py"
  --config "$circe_config" --resume "$circe_checkpoint"
  --io-mode fast --io-prefetch 1 --io-recovery-every-n-train-steps 100
  --io-output-dir "$circe_new" --wandb-mode disabled)
printf -v circe_command '%q ' "${circe_args[@]}"
/usr/bin/tmux new-session -d -s "$circe_session" -c "$circe_project" \
  "exec $circe_command > $circe_new.log 2>&1"
echo "Detached resume launched: $circe_session"
echo "Resuming the five completed epochs; 30 total maximum, early stopping enabled."
echo "Original checkpoints, test results and failed-run logs were preserved."
echo "Watch: tail -f $circe_new.log"
