#!/usr/bin/env bash
# Stop the resumed experiment and launch a fresh seed-42 run at lambda=0.05.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

circe_old="$PWD/runs/circe_v3_reaction_smi_cls005_resume.I4pJh7"
circe_labels="$PWD/runs/circe_v3_reaction_smi_cls002.DfNq1C"
circe_base=runs/circe_v3_reactzyme_3gpu/reactzyme/reaction_smi/seed42
circe_session=circe_v3_reaction_smi_cls005_resume_I4pJh7
circe_mode=${1:-run}
[[ "$circe_mode" == run || "$circe_mode" == --check ]] || {
  echo "Usage: bash $0 [--check]"; exit 1;
}
[[ "$(hostname -s)" == slurm-node-014 ]] || {
  echo "Run this on slurm-node-014. No processes stopped."; exit 1;
}
command -v tmux >/dev/null
command -v nvidia-smi >/dev/null
for circe_file in "$circe_labels/biofp_targets.npz" "$circe_labels/biofp_vocab.json" \
  "$circe_base/configs/train.yaml"; do
  [[ -s "$circe_file" ]] || { echo "Missing: $circe_file"; exit 1; }
done

if tmux has-session -t "=$circe_session" 2>/dev/null; then
  # Use a pane ID, avoiding punctuation-dependent tmux target parsing.
  circe_panes=$(tmux list-panes -s -t "=$circe_session" -F '#{pane_id} #{pane_pid}')
  [[ "$circe_panes" != *$'\n'* ]] || {
    echo "Multiple panes found; refusing an ambiguous shutdown."; exit 1;
  }
  read -r circe_pane circe_pid <<< "$circe_panes"
  circe_argv=$(tr '\0' ' ' < "/proc/$circe_pid/cmdline")
  [[ "$circe_argv" == *scripts/train_protein_pooling.py* &&
     "$circe_argv" == *"--logging.checkpoint_dir=$circe_old/checkpoints"* ]] || {
    echo "Pane does not match the expected training run. Nothing stopped."; exit 1;
  }
  echo "Verified original training: pane $circe_pane, PID $circe_pid."
  [[ "$circe_mode" != --check ]] || { echo "Check passed; no changes made."; exit 0; }
  echo "Requesting graceful stop..."
  tmux send-keys -t "$circe_pane" C-c
elif [[ "$circe_mode" == --check ]]; then
  echo "Original tmux session absent. No changes made."; exit 1
fi

for circe_attempt in {1..15}; do
  if ! pgrep -u "$(id -u)" -f "[t]rain_protein_pooling[.]py.*$circe_old/" >/dev/null; then
    break
  fi
  sleep 2
done
if pgrep -u "$(id -u)" -f "[t]rain_protein_pooling[.]py.*$circe_old/"; then
  echo "Old processes remain. Replacement NOT launched."; exit 1
fi
circe_busy=$(nvidia-smi -i 0,1,3 --query-compute-apps=pid --format=csv,noheader)
[[ -z "$circe_busy" ]] || {
  echo "GPUs 0,1,3 still occupied. Replacement NOT launched. PIDs: $circe_busy"; exit 1;
}

circe_new=$(mktemp -d "$PWD/runs/circe_v3_reaction_smi_cls005_fresh.XXXXXX")
circe_name=$(basename "$circe_new")
circe_name=${circe_name//./_}
printf -v circe_cmd '%q ' /usr/bin/env -u BASH_ENV -u ENV \
  CUDA_VISIBLE_DEVICES=0,1,3 PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PWD/../env/bin/python" scripts/train_protein_pooling.py \
  --config "$circe_base/configs/train.yaml" --seed 42 \
  --wandb-mode disabled \
  --data.protein_biofp_targets_path="$circe_labels/biofp_targets.npz" \
  --data.protein_biofp_vocab_path="$circe_labels/biofp_vocab.json" \
  --training.loss.biofp_aux_weight=0.05 \
  --training.loss.biofp_aux_warmup_epochs=3 \
  --training.loss.biofp_family_weights.mechanism=1.0 \
  --training.loss.biofp_family_weights.cofactor=1.0 \
  --training.devices=3 \
  --logging.log_dir="$circe_new/logs" \
  --logging.checkpoint_dir="$circe_new/checkpoints" \
  --ablation.variant=CIRCE-v3-classification-005-fresh
printf -v circe_log '%q' "$circe_new/pipeline.log"
tmux new-session -d -s "$circe_name" -c "$PWD" \
  "exec $circe_cmd > $circe_log 2>&1"
echo "Fresh seed-42 training; lambda=0.05 with 3-epoch warm-up. No checkpoint resumed."
echo "Detached session: $circe_name"
echo "Watch: tail -f $circe_new/pipeline.log"
