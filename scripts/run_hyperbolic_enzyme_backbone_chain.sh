#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {esm2|prott5|esmc}" >&2
  exit 2
fi

BACKBONE="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"
FASTA_PATH="${FASTA_PATH:-data/sota/prots.fasta}"
EC_LABELS_PATH="${EC_LABELS_PATH:-data/sota/uniprot_all_ec_labels.csv}"
RUN_NAME="${RUN_NAME:-hyperbolic-enzyme-${BACKBONE}-lorentz-c0p25-ecentail-b512-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/hyperbolic_enzyme/$RUN_NAME}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CURVATURE="${CURVATURE:-0.25}"
P0_SLEEC_THRESHOLD="${P0_SLEEC_THRESHOLD:-0.34}"
USE_EC_ENTAILMENT="${USE_EC_ENTAILMENT:-true}"
ALPHA_EC_ENTAILMENT="${ALPHA_EC_ENTAILMENT:-0.05}"
EC_ENTAILMENT_WARMUP_EPOCHS="${EC_ENTAILMENT_WARMUP_EPOCHS:-2}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-hyperbolic-enzyme}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-omnai}"
LOG_EVERY="${LOG_EVERY:-5}"

cd "$PROJECT_ROOT"
mkdir -p "$CHECKPOINT_DIR"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache}"

case "$BACKBONE" in
  esm2)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    INPUT_DIM="${INPUT_DIM:-1280}"
    RESIDUE_OUTPUT="${RESIDUE_OUTPUT:-data/sota/prots_esm2_650m_residue.h5}"
    TMP_DIR="${TMP_DIR:-data/sota/prots_esm2_650m_residue_shards}"
    SLEEC_CHECKPOINT_PATH="${SLEEC_CHECKPOINT_PATH:-checkpoints/SLEEC/sleec_stage1_esm2_650m_uniref90_msa_50k_threshold_sweep_4gpu_20260602_103942/best.ckpt}"
    EXTRACT_CMD=(
      bash scripts/run_extract_esm2_residue_embeddings_4gpu.sh
      --run-name "${RUN_NAME}_extract"
      --fasta "$FASTA_PATH"
      --output "$RESIDUE_OUTPUT"
      --tmp-dir "$TMP_DIR"
      --batch-size "${EXTRACT_BATCH_SIZE:-8}"
      --max-tokens-per-batch "${EXTRACT_MAX_TOKENS_PER_BATCH:-4096}"
      --max-sequence-length "${MAX_SEQUENCE_LENGTH:-1022}"
      --dtype "${EXTRACT_DTYPE:-float16}"
      --compression "${EXTRACT_COMPRESSION:-none}"
    )
    ;;
  prott5)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
    INPUT_DIM="${INPUT_DIM:-1024}"
    RESIDUE_OUTPUT="${RESIDUE_OUTPUT:-data/sota/prots_t5_residue.h5}"
    TMP_DIR="${TMP_DIR:-data/sota/prots_t5_residue_shards}"
    SLEEC_CHECKPOINT_PATH="${SLEEC_CHECKPOINT_PATH:-checkpoints/SLEEC/sleec_stage1_prott5_uniref90_msa_4gpu_20260601_235157/best.ckpt}"
    EXTRACT_CMD=(
      bash scripts/run_extract_prott5_residue_embeddings_4gpu.sh
      --run-name "${RUN_NAME}_extract"
      --fasta "$FASTA_PATH"
      --output "$RESIDUE_OUTPUT"
      --tmp-dir "$TMP_DIR"
      --batch-size "${EXTRACT_BATCH_SIZE:-8}"
      --max-tokens-per-batch "${EXTRACT_MAX_TOKENS_PER_BATCH:-4096}"
      --max-sequence-length "${MAX_SEQUENCE_LENGTH:-1024}"
      --dtype "${EXTRACT_DTYPE:-float16}"
      --compression "${EXTRACT_COMPRESSION:-none}"
    )
    ;;
  esmc)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
    INPUT_DIM="${INPUT_DIM:-2560}"
    RESIDUE_OUTPUT="${RESIDUE_OUTPUT:-data/sota/prots_esmc_6B_residue_mbriglia.h5}"
    TMP_DIR="${TMP_DIR:-data/sota/prots_esmc_6B_residue_mbriglia_shards}"
    SLEEC_CHECKPOINT_PATH="${SLEEC_CHECKPOINT_PATH:-checkpoints/SLEEC/sleec_stage1_esmc_6b_uniref90_msa_4gpu_20260601_235157/best.ckpt}"
    EXTRACT_CMD=(
      bash scripts/run_extract_esmc_residue_embeddings_4gpu.sh
      --run-name "${RUN_NAME}_extract"
      --fasta "$FASTA_PATH"
      --output "$RESIDUE_OUTPUT"
      --tmp-dir "$TMP_DIR"
      --batch-size "${EXTRACT_BATCH_SIZE:-1}"
      --max-tokens-per-batch "${EXTRACT_MAX_TOKENS_PER_BATCH:-2048}"
      --max-sequence-length "${MAX_SEQUENCE_LENGTH:-1022}"
      --dtype "${EXTRACT_DTYPE:-float16}"
      --compression "${EXTRACT_COMPRESSION:-none}"
    )
    ;;
  *)
    echo "Unknown backbone: $BACKBONE" >&2
    exit 2
    ;;
esac

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
WORLD_SIZE="${#GPUS[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] Hyperbolic enzyme backbone chain"
echo "Backbone: $BACKBONE"
echo "Run name: $RUN_NAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "World size: $WORLD_SIZE"
echo "Residue output: $RESIDUE_OUTPUT"
echo "SLEEC checkpoint: $SLEEC_CHECKPOINT_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Per-process batch size: $BATCH_SIZE"
echo "Curvature: $CURVATURE"

if [[ ! -r "$SLEEC_CHECKPOINT_PATH" ]]; then
  echo "Missing readable SLEEC checkpoint: $SLEEC_CHECKPOINT_PATH" >&2
  exit 1
fi

if [[ -r "$RESIDUE_OUTPUT" ]]; then
  echo "[$(date --iso-8601=seconds)] Using existing residue HDF5"
else
  echo "[$(date --iso-8601=seconds)] Extracting residue HDF5"
  "${EXTRACT_CMD[@]}"
fi

TRAIN_ARGS=(
  scripts/pretrain_hyperbolic_enzyme.py
  --config configs/hyperbolic_enzyme_pretrain.yaml
  --residue-embeddings-path "$RESIDUE_OUTPUT"
  --data-path "$EC_LABELS_PATH"
  --input-dim "$INPUT_DIM"
  --curvature "$CURVATURE"
  --p0-sleec-threshold "$P0_SLEEC_THRESHOLD"
  --batch-size "$BATCH_SIZE"
  --sleec-checkpoint-path "$SLEEC_CHECKPOINT_PATH"
  --output-checkpoint "$CHECKPOINT_DIR/last.ckpt"
  --log-every "$LOG_EVERY"
)

if [[ "$USE_EC_ENTAILMENT" == "true" ]]; then
  TRAIN_ARGS+=(
    --use-ec-entailment
    --alpha-ec-entailment "$ALPHA_EC_ENTAILMENT"
    --ec-entailment-warmup-epochs "$EC_ENTAILMENT_WARMUP_EPOCHS"
  )
fi

if [[ "$WANDB_ENABLED" == "true" ]]; then
  TRAIN_ARGS+=(
    --wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-entity "$WANDB_ENTITY"
    --wandb-mode "$WANDB_MODE"
    --wandb-run-name "$RUN_NAME"
    --wandb-tags hyperbolic-enzyme lorentz "$BACKBONE" sleec-prior c0p25 ecentail b512 uniref90
  )
fi

echo "[$(date --iso-8601=seconds)] Starting hyperbolic enzyme pretraining"
if [[ "$WORLD_SIZE" -gt 1 ]]; then
  "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="$WORLD_SIZE" \
    "${TRAIN_ARGS[@]}"
else
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi
echo "[$(date --iso-8601=seconds)] Chain finished"
