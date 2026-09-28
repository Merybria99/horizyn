#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"

RUN_NAME="${RUN_NAME:-hyperbolic-enzyme-esmc-lorentz-c0p25-b512-4gpu-$(date +%Y%m%d_%H%M%S)}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

DEFAULT_READABLE_RESIDUE_PATH="data/sota/prots_esmc_6B_residue.h5"
if [[ -r "$DEFAULT_READABLE_RESIDUE_PATH" ]]; then
  ESMC_RESIDUE_PATH="${ESMC_RESIDUE_PATH:-$DEFAULT_READABLE_RESIDUE_PATH}"
else
  ESMC_RESIDUE_PATH="${ESMC_RESIDUE_PATH:-data/sota/prots_esmc_6B_residue_mbriglia.h5}"
fi
ESMC_TMP_DIR="${ESMC_TMP_DIR:-data/sota/prots_esmc_6B_residue_mbriglia_shards}"
SLEEC_CHECKPOINT_PATH="${SLEEC_CHECKPOINT_PATH:-checkpoints/SLEEC/sleec_stage1_esmc_6b_50k_threshold_sweep_4gpu_retry3_20260601_154048/best.ckpt}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/hyperbolic_enzyme/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-logs/hyperbolic_enzyme}"
LOG_PATH="${LOG_PATH:-$LOG_DIR/$RUN_NAME.log}"
PID_DIR="${PID_DIR:-run_pids/hyperbolic_enzyme}"

cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR" "$PID_DIR"
printf '%s\n' "$$" > "$PID_DIR/$RUN_NAME.pid"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

{
  echo "[$(date --iso-8601=seconds)] Starting ESMC SLEEC-guided Lorentz pretraining launcher"
  echo "Run name: $RUN_NAME"
  echo "Project root: $PROJECT_ROOT"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "ESMC residue path: $ESMC_RESIDUE_PATH"
  echo "ESMC temp shards: $ESMC_TMP_DIR"
  echo "SLEEC checkpoint: $SLEEC_CHECKPOINT_PATH"
  echo "Checkpoint dir: $CHECKPOINT_DIR"

  if [[ ! -r "$SLEEC_CHECKPOINT_PATH" ]]; then
    echo "Missing readable ESMC SLEEC checkpoint: $SLEEC_CHECKPOINT_PATH" >&2
    exit 1
  fi

  if [[ ! -r "$ESMC_RESIDUE_PATH" ]]; then
    echo "[$(date --iso-8601=seconds)] ESMC residue HDF5 is not readable; extracting it first"
    scripts/run_extract_esmc_residue_embeddings_4gpu.sh \
      --run-name "${RUN_NAME}_extract" \
      --output "$ESMC_RESIDUE_PATH" \
      --tmp-dir "$ESMC_TMP_DIR" \
      --batch-size 1 \
      --max-tokens-per-batch 2048 \
      --max-sequence-length 1022 \
      --dtype float16 \
      --compression none
  else
    echo "[$(date --iso-8601=seconds)] Using existing readable ESMC residue HDF5"
  fi

  echo "[$(date --iso-8601=seconds)] Launching hyperbolic enzyme pretraining"
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=4 \
    scripts/pretrain_hyperbolic_enzyme.py \
    --config configs/hyperbolic_enzyme_pretrain.yaml \
    --residue-embeddings-path "$ESMC_RESIDUE_PATH" \
    --input-dim 2560 \
    --batch-size 512 \
    --sleec-checkpoint-path "$SLEEC_CHECKPOINT_PATH" \
    --output-checkpoint "$CHECKPOINT_DIR/last.ckpt" \
    --wandb \
    --wandb-project horizyn-hyperbolic-enzyme \
    --wandb-run-name "$RUN_NAME" \
    --wandb-tags hyperbolic-enzyme lorentz esmc sleec-prior c0p25 b512 \
    --log-every 5

  echo "[$(date --iso-8601=seconds)] Finished ESMC SLEEC-guided Lorentz pretraining"
} 2>&1 | tee "$LOG_PATH"
