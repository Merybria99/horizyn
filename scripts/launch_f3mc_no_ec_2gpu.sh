#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-2,3}"
MASTER_PORT="${2:-31504}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python"
CONFIG="$ROOT/runs/reactzyme_f3_biological_no_ec_v1/configs/F3MC_no_ec/train_2gpu_retry.yaml"
LOG_DIR="$ROOT/runs/reactzyme_f3_biological_no_ec_v1/logs/F3MC_no_ec_2gpu_retry"
LOG="$LOG_DIR/train.stdout.log"

mkdir -p "$LOG_DIR"

count="$(CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$count" != "2" ]]; then
  echo "Expected two visible GPUs from '$GPU_LIST', found $count" >&2
  exit 1
fi

cd "$ROOT"
setsid nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$ROOT" \
  "$PYTHON" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    scripts/train_protein_pooling.py \
    --config "$CONFIG" \
    --wandb \
    --wandb-project horizyn-reactzyme-f3-biological-no-ec \
    --wandb-entity omnai \
    --wandb-run-name F3MC_no_ec-2gpu-retry-reaction-smi-seed42 \
    --wandb-mode online \
    > "$LOG" 2>&1 < /dev/null &

echo "$!"
