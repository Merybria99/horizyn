#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-0,1,2,3}"
MASTER_PORT="${2:-31575}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python"
RUN_ROOT="$ROOT/runs/source_collapse_f3mc_ec_lambda3_95_5_v1"
CONFIG="$RUN_ROOT/configs/train.yaml"
LOG_DIR="$RUN_ROOT/logs/train"
LOG="$LOG_DIR/train.stdout.log"
PID_FILE="$LOG_DIR/launcher.pid"

mkdir -p "$LOG_DIR" "$RUN_ROOT/checkpoints/train" \
  "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/matplotlib" \
  "$RUN_ROOT/cache/wandb" "$RUN_ROOT/wandb"

if [[ ! -s "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  echo "Prepare it first with scripts/create_source_collapse_f3mc_ec_95_5_run.py" >&2
  exit 1
fi

count="$(CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$count" != "4" ]]; then
  echo "Expected four visible GPUs from '$GPU_LIST', found $count" >&2
  exit 1
fi

cd "$ROOT"
setsid nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_LIST" \
  NCCL_P2P_DISABLE=1 \
  NCCL_IB_DISABLE=1 \
  TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  HF_HOME="$RUN_ROOT/cache/huggingface" \
  TORCH_HOME="$RUN_ROOT/cache/torch" \
  XDG_CACHE_HOME="$RUN_ROOT/cache/xdg" \
  MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib" \
  WANDB_CACHE_DIR="$RUN_ROOT/cache/wandb" \
  WANDB_DIR="$RUN_ROOT/wandb" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH="$ROOT" \
  "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr=127.0.0.1 \
    --master_port="$MASTER_PORT" \
    scripts/train_protein_pooling.py \
    --config "$CONFIG" \
    --wandb \
    --wandb-project horizyn-source-collapse-f3mc-ec-lambda3-95-5-v1 \
    --wandb-entity omnai \
    --wandb-run-name F3MC-EC-lambda3-source-collapse-rxn-disjoint-95-5-seed42 \
    --wandb-mode online \
    > "$LOG" 2>&1 < /dev/null &

pid="$!"
printf '%s\n' "$pid" > "$PID_FILE"
echo "$pid"
