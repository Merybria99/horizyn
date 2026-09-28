#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
CONFIG="${CONFIG:-configs/retrieval_hybrid_multi_alignment_4gpu.yaml}"
RUN_KEY="${1:-anchor}"
GPUS="${GPUS:-0,1,2,3}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-retrieval-source-collapse}"
WANDB_ENTITY="${WANDB_ENTITY:-omnai}"

case "$RUN_KEY" in
  anchor)
    RUN_NAME="hybrid_multi_alignment_allknown_anchor_4gpu"
    SESSION="retrieval_hybrid_multi_align_allknown_anchor_4gpu"
    MASTER_PORT="${MASTER_PORT:-29681}"
    LAMBDA_RR="0.0"
    LAMBDA_EE="0.0"
    LAMBDA_GW="0.0"
    EC_MIN_SHARED_DEPTH="2"
    ;;
  ee_depth3)
    RUN_NAME="hybrid_multi_alignment_anchor_ee_depth3_4gpu"
    SESSION="retrieval_hybrid_multi_align_anchor_ee_depth3_4gpu"
    MASTER_PORT="${MASTER_PORT:-29685}"
    LAMBDA_RR="0.0"
    LAMBDA_EE="0.02"
    LAMBDA_GW="0.0"
    EC_MIN_SHARED_DEPTH="3"
    ;;
  rr)
    RUN_NAME="hybrid_multi_alignment_anchor_rr_4gpu"
    SESSION="retrieval_hybrid_multi_align_anchor_rr_4gpu"
    MASTER_PORT="${MASTER_PORT:-29682}"
    LAMBDA_RR="0.02"
    LAMBDA_EE="0.0"
    LAMBDA_GW="0.0"
    EC_MIN_SHARED_DEPTH="2"
    ;;
  rr_ee)
    RUN_NAME="hybrid_multi_alignment_anchor_rr_ee_4gpu"
    SESSION="retrieval_hybrid_multi_align_anchor_rr_ee_4gpu"
    MASTER_PORT="${MASTER_PORT:-29683}"
    LAMBDA_RR="0.02"
    LAMBDA_EE="0.02"
    LAMBDA_GW="0.0"
    EC_MIN_SHARED_DEPTH="2"
    ;;
  rr_ee_gw)
    RUN_NAME="hybrid_multi_alignment_anchor_rr_ee_gw_4gpu"
    SESSION="retrieval_hybrid_multi_align_anchor_rr_ee_gw_4gpu"
    MASTER_PORT="${MASTER_PORT:-29684}"
    LAMBDA_RR="0.02"
    LAMBDA_EE="0.02"
    LAMBDA_GW="0.005"
    EC_MIN_SHARED_DEPTH="2"
    ;;
  *)
    echo "Usage: $0 {anchor|ee_depth3|rr|rr_ee|rr_ee_gw}" >&2
    exit 2
    ;;
esac

LOG_DIR="logs/retrieval_${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/retrieval_${RUN_NAME}"
LOG_FILE="${LOG_DIR}/tmux.log"

cd "$ROOT_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p \
  "$LOG_DIR/wandb_runtime/cache" \
  "$LOG_DIR/wandb_runtime/config" \
  "$CHECKPOINT_DIR"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

train_cmd=(
  env
  MASTER_PORT="$MASTER_PORT"
  WANDB_DIR="$LOG_DIR/wandb_runtime"
  WANDB_CACHE_DIR="$LOG_DIR/wandb_runtime/cache"
  WANDB_CONFIG_DIR="$LOG_DIR/wandb_runtime/config"
  "$PYTHON_BIN"
  scripts/train_protein_pooling.py
  --config "$CONFIG"
  --wandb
  --wandb-entity "$WANDB_ENTITY"
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$RUN_NAME"
  --logging.log_dir "$LOG_DIR"
  --logging.checkpoint_dir "$CHECKPOINT_DIR"
  --logging.wandb.run_name "$RUN_NAME"
  --training.loss.name MultiAlignmentRetrievalLoss
  --training.loss.positive_pair_source all_known_in_batch
  --training.loss.lambda_rr "$LAMBDA_RR"
  --training.loss.lambda_ee "$LAMBDA_EE"
  --training.loss.lambda_gw "$LAMBDA_GW"
  --training.loss.ec_min_shared_depth "$EC_MIN_SHARED_DEPTH"
)

printf -v quoted_cmd "%q " "${train_cmd[@]}"
printf -v quoted_log "%q" "$LOG_FILE"

tmux new -d -s "$SESSION" -c "$ROOT_DIR" "${quoted_cmd} >> ${quoted_log} 2>&1"

echo "Launched $RUN_KEY as tmux session: $SESSION"
echo "Log: $LOG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT: $MASTER_PORT"
