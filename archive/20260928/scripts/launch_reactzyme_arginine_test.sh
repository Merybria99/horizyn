#!/usr/bin/env bash
# Evaluate the validation-selected classification/50-50 checkpoint, not F3 weights.
set -euo pipefail
cd /
[[ "$(hostname -s)" == slurm-node-014 ]] || {
  echo "Run this on arginine (slurm-node-014). Nothing launched." >&2; exit 1;
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_run="$circe_project/runs/reactzyme_reaction_smi_arginine_cls02_5050_20260911_233523"
circe_checkpoint="$circe_run/training/checkpoints/protein-pooling-epoch=27.ckpt"
# This existing YAML supplies the established test features/candidate protocol.
# The evaluator reconstructs the model and all its weights from the checkpoint.
circe_config="$circe_project/runs/reactzyme_reaction_features_v1/configs/F3_set_chemistry/reaction_smi/test.yaml"
circe_session=reactzyme_arginine_cls02_test
circe_log="$circe_run/test_reaction_smi.log"
circe_result="$circe_run/test_reaction_smi.json"

if [[ "${1:-launch}" == launch ]]; then
  [[ ! -e "$circe_log" && ! -e "$circe_result" ]] || {
    echo "Test log/results already exist; refusing to overwrite them."; exit 1;
  }
  if tmux has-session -t "=$circe_session" 2>/dev/null; then
    echo "Test session already exists; no duplicate launched."; exit 1
  fi
  tmux new-session -d -s "$circe_session" -c / \
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /bin/bash --noprofile --norc "$circe_project/scripts/launch_reactzyme_arginine_test.sh" run
  echo "Detached test session: $circe_session"
  echo "Watch: tail -f $circe_log"
  echo "Results: $circe_result"
  exit 0
fi
[[ "${1:-}" == run ]] || exit 1
set -o noclobber
exec > "$circe_log" 2>&1
trap 'echo "Test stopped (exit $?) at line $LINENO; no other jobs were stopped."' ERR
echo "Checking GPU 0; testing both retrieval directions with the best checkpoint."
circe_uuid=$(timeout --kill-after=2s 10s nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
circe_busy=$(timeout --kill-after=2s 10s nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
  | awk -F, -v uuid="$circe_uuid" '$1 == uuid {print $2}')
[[ -z "$circe_busy" ]] || {
  echo "GPU 0 is occupied by PID(s): $circe_busy. Test NOT started."; exit 1;
}
[[ -s "$circe_checkpoint" && -s "$circe_config" && ! -e "$circe_result" ]] || {
  echo "Missing checkpoint/configuration, or results already exist. Stopping."; exit 1;
}
cd "$circe_project"
echo "Checkpoint: $circe_checkpoint"
echo "Protocol: paper_test_candidates; ordinary cosine; no training or extraction."
exec "$circe_project/../env/bin/python" -B -u scripts/evaluate_protein_pooling.py \
  --checkpoint "$circe_checkpoint" --config "$circe_config" \
  --device cuda:0 --direction both --evaluation-protocol paper_test_candidates \
  --batch-size 128 --target-batch-size 512 --cosine-feature-power 1.0 \
  --output "$circe_result"
