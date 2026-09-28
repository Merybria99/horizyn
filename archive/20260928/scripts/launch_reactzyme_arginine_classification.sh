#!/usr/bin/env bash
# Four H200s, cached backbones, fresh towers, classification=0.2, pairs=50/50.
set -euo pipefail
cd /
[[ "$(hostname -s)" == slurm-node-014 ]] || {
  echo "Run this on arginine (slurm-node-014). Nothing launched." >&2; exit 1;
}
circe_project=/datastor2/deep-proteins/EnzymeDiscovery/horizyn
circe_launcher="$circe_project/scripts/launch_reactzyme_arginine_classification.sh"
circe_python=/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python
circe_session=reactzyme_arginine_cls02_5050

if [[ "${1:-launch}" == launch ]]; then
  if tmux has-session -t "=$circe_session" 2>/dev/null; then
    echo "Session already exists: $circe_session. No duplicate launched."; exit 1
  fi
  circe_bootdir=$(mktemp -d /tmp/reactzyme_arginine.XXXXXX)
  # The log exists before tmux or any shared-storage access inside the pane.
  printf 'Arginine launch requested: GPUs 0,1,2,3; no existing jobs stopped.\n' > "$circe_bootdir/pipeline.log"
  tmux new-session -d -s "$circe_session" -c / \
    /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /bin/bash --noprofile --norc "$circe_launcher" run "$circe_bootdir"
  echo "Detached session: $circe_session"
  echo "Watch: tail -f $circe_bootdir/pipeline.log"
  exit 0
fi
[[ "${1:-}" == run && "${2:-}" == /tmp/reactzyme_arginine.* && -d "$2" ]] || exit 1
exec >> "$2/pipeline.log" 2>&1
trap 'echo "Launcher stopped (exit $?) at line $LINENO; inspect the log. No other jobs were stopped."' ERR

gpu_check() {
  local circe_busy
  circe_busy=$(timeout --kill-after=2s 10s nvidia-smi -i 0,1,2,3 \
    --query-compute-apps=pid --format=csv,noheader)
  [[ -z "$circe_busy" ]] || { echo "GPU processes present: $circe_busy. Stopping without signalling them."; return 1; }
  timeout --kill-after=2s 10s nvidia-smi -i 0,1,2,3 \
    --query-gpu=index,name,memory.free --format=csv,noheader,nounits \
    | awk -F, '{print; seen[$1+0]++; if ($2 !~ /H200/ || $3+0 < 130000) bad=1}
               END {exit (bad || seen[0]!=1 || seen[1]!=1 || seen[2]!=1 || seen[3]!=1)}'
}
wait_for_gpus() {
  local circe_deadline=$((SECONDS + 30)) circe_busy
  while :; do
    circe_busy=$(timeout --kill-after=2s 10s nvidia-smi -i 0,1,2,3 \
      --query-compute-apps=pid --format=csv,noheader)
    [[ -n "$circe_busy" ]] || break
    (( SECONDS < circe_deadline )) || break
    echo "Waiting for CUDA contexts to release; no signals sent."
    sleep 2
  done
  gpu_check
}
gpu_check
echo "Checking shared storage; timeouts mean NFS still needs repair, not re-extraction."
timeout --kill-after=2s 15s head -c 65536 "$circe_project/scripts/prepare_reactzyme_arginine_classification.py" >/dev/null
cd "$circe_project"
CUDA_VISIBLE_DEVICES= timeout --kill-after=5s 300s "$circe_python" -B -u \
  scripts/check_reactzyme_prott5_cache.py --samples 32

circe_run="$circe_project/runs/reactzyme_reaction_smi_arginine_cls02_5050_$(date -u +%Y%m%d_%H%M%S)"
echo "Preparing: $circe_run"
CUDA_VISIBLE_DEVICES= timeout --kill-after=5s 600s "$circe_python" -B -u \
  scripts/prepare_reactzyme_arginine_classification.py prepare --run-root "$circe_run"
exec > >(tee -a "$circe_run/pipeline.log") 2>&1
echo "Run: $circe_run"
echo "Calibrating 400, 800, 1600, 3200 pair rows/GPU; 15 steps each, no validation/checkpoints."
for circe_batch in 400 800 1600 3200; do
  wait_for_gpus
  if "$circe_python" -B -u scripts/train_protein_pooling_fast_io.py \
    --config "$circe_run/batch_$circe_batch.yaml" --io-mode fast --io-prefetch 1 \
    --io-benchmark-steps 15 --io-output-dir "$circe_run/pilot_$circe_batch" \
    --wandb-mode disabled 2>&1 | tee "$circe_run/pilot_$circe_batch.log"; then
    continue
  fi
  # Only a verified OOM permits falling back to an earlier successful pilot.
  # Other errors stop immediately; never signal unverified GPU processes.
  if ! /usr/bin/grep -Eq 'CUDA out of memory|torch.OutOfMemoryError' "$circe_run/pilot_$circe_batch.log"; then
    echo "Pilot failed for a non-memory reason. Full training NOT launched."; exit 1
  fi
  echo "Batch $circe_batch exhausted VRAM; choosing among smaller successful pilots."
  wait_for_gpus
  break
done
CUDA_VISIBLE_DEVICES= "$circe_python" -B -u scripts/prepare_reactzyme_arginine_classification.py \
  select --run-root "$circe_run"
wait_for_gpus
echo "Starting 30 full epochs from fresh tower initialization; see batch_selection.json."
exec "$circe_python" -B -u scripts/train_protein_pooling_fast_io.py \
  --config "$circe_run/train.yaml" --io-mode fast --io-prefetch 1 \
  --io-recovery-every-n-train-steps 50 --io-output-dir "$circe_run/training" \
  --wandb-mode disabled
