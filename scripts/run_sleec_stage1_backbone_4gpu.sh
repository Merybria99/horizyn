#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
ENV_PATH="${ENV_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_PATH/bin/python}"

BACKBONE="esm2"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME=""
CONFIG_PATH=""
OUTPUT_DIR=""
FASTA_PATH="data/sleec_stage1/fasta/mcsa_reference.fasta"
RESIDUE_H5=""
MCSA_LABELS_PATH="data/sleec_stage1/mcsa_residue_labels.csv"
PSEUDO_LABELS_PATH="data/sleec_stage1/msa_pseudo_residue_labels.balanced.csv"
MODEL_NAME=""
ESMC_BACKEND="biohub"
BIOHUB_ESM_PATH="${BIOHUB_ESM_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/sources/Biohub_esm}"
ESMC_OVERLAY_PATH="${ESMC_OVERLAY_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn/.deps/esmc_transformers_457_min}"
EXTRACT_BATCH_SIZE=""
MAX_TOKENS_PER_BATCH=""
MAX_SEQUENCE_LENGTH=""
HIDDEN_LAYER="-1"
EXPECTED_DIM=""
DTYPE="float16"
COMPRESSION="none"
SKIP_EXTRACT="false"
FORCE_EXTRACT="false"
CLEANUP_SHARDS="false"
EXTRACT_ONLY="false"
TRAIN_ONLY="false"
WANDB="${WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-sleec-stage1}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRUST_REMOTE_CODE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone)
      BACKBONE="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --fasta)
      FASTA_PATH="$2"
      shift 2
      ;;
    --residue-h5|--embeddings)
      RESIDUE_H5="$2"
      shift 2
      ;;
    --mcsa-labels)
      MCSA_LABELS_PATH="$2"
      shift 2
      ;;
    --pseudo-labels)
      PSEUDO_LABELS_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --esmc-backend|--backend)
      ESMC_BACKEND="$2"
      shift 2
      ;;
    --extract-batch-size)
      EXTRACT_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-tokens-per-batch)
      MAX_TOKENS_PER_BATCH="$2"
      shift 2
      ;;
    --max-sequence-length)
      MAX_SEQUENCE_LENGTH="$2"
      shift 2
      ;;
    --hidden-layer)
      HIDDEN_LAYER="$2"
      shift 2
      ;;
    --expected-dim)
      EXPECTED_DIM="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --compression)
      COMPRESSION="$2"
      shift 2
      ;;
    --skip-extract)
      SKIP_EXTRACT="true"
      shift
      ;;
    --force-extract)
      FORCE_EXTRACT="true"
      shift
      ;;
    --cleanup-shards)
      CLEANUP_SHARDS="true"
      shift
      ;;
    --extract-only)
      EXTRACT_ONLY="true"
      shift
      ;;
    --train-only)
      TRAIN_ONLY="true"
      shift
      ;;
    --wandb)
      WANDB="true"
      shift
      ;;
    --no-wandb)
      WANDB="false"
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$BACKBONE" in
  esm2|esm2_650m|esm2-650m)
    BACKBONE="esm2"
    DEFAULT_CONFIG="configs/sleec_stage1_esm2_650m.yaml"
    DEFAULT_RESIDUE_H5="data/sleec_stage1/mcsa_cath_esm2_650m_residue.h5"
    DEFAULT_MODEL_NAME="facebook/esm2_t33_650M_UR50D"
    EXTRACT_RUNNER="scripts/run_extract_esm2_residue_embeddings_4gpu.sh"
    DEFAULT_EXTRACT_BATCH_SIZE="8"
    DEFAULT_MAX_TOKENS_PER_BATCH="4096"
    DEFAULT_MAX_SEQUENCE_LENGTH="1022"
    DEFAULT_EXPECTED_DIM="1280"
    ;;
  prott5|prot_t5|prot-t5|t5)
    BACKBONE="prott5"
    DEFAULT_CONFIG="configs/sleec_stage1_prott5_unpooled.yaml"
    DEFAULT_RESIDUE_H5="data/sleec_stage1/mcsa_cath_prott5_residue.h5"
    DEFAULT_MODEL_NAME="Rostlab/prot_t5_xl_half_uniref50-enc"
    EXTRACT_RUNNER="scripts/run_extract_prott5_residue_embeddings_4gpu.sh"
    DEFAULT_EXTRACT_BATCH_SIZE="8"
    DEFAULT_MAX_TOKENS_PER_BATCH="4096"
    DEFAULT_MAX_SEQUENCE_LENGTH="1024"
    DEFAULT_EXPECTED_DIM="1024"
    ;;
  esmc|esmc_6b|esmc-6b)
    BACKBONE="esmc"
    DEFAULT_CONFIG="configs/sleec_stage1_esmc_6b.yaml"
    DEFAULT_RESIDUE_H5="data/sleec_stage1/mcsa_cath_esmc_6b_residue.h5"
    DEFAULT_MODEL_NAME="Biohub/ESMC-6B"
    EXTRACT_RUNNER="scripts/run_extract_esmc_residue_embeddings_4gpu.sh"
    DEFAULT_EXTRACT_BATCH_SIZE="1"
    DEFAULT_MAX_TOKENS_PER_BATCH="2048"
    DEFAULT_MAX_SEQUENCE_LENGTH="1022"
    DEFAULT_EXPECTED_DIM="2560"
    ;;
  *)
    echo "Unsupported backbone '$BACKBONE'. Use one of: esm2, prott5, esmc." >&2
    exit 2
    ;;
esac

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="sleec_stage1_${BACKBONE}_${RUN_STAMP}"
fi
if [[ -z "$CONFIG_PATH" ]]; then
  CONFIG_PATH="$DEFAULT_CONFIG"
fi
if [[ -z "$RESIDUE_H5" ]]; then
  RESIDUE_H5="$DEFAULT_RESIDUE_H5"
fi
if [[ -z "$MODEL_NAME" ]]; then
  MODEL_NAME="$DEFAULT_MODEL_NAME"
fi
if [[ -z "$EXTRACT_BATCH_SIZE" ]]; then
  EXTRACT_BATCH_SIZE="$DEFAULT_EXTRACT_BATCH_SIZE"
fi
if [[ -z "$MAX_TOKENS_PER_BATCH" ]]; then
  MAX_TOKENS_PER_BATCH="$DEFAULT_MAX_TOKENS_PER_BATCH"
fi
if [[ -z "$MAX_SEQUENCE_LENGTH" ]]; then
  MAX_SEQUENCE_LENGTH="$DEFAULT_MAX_SEQUENCE_LENGTH"
fi
if [[ -z "$EXPECTED_DIM" ]]; then
  EXPECTED_DIM="$DEFAULT_EXPECTED_DIM"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="checkpoints/SLEEC/${RUN_NAME}"
fi

cd "$PROJECT_ROOT"
source "$ENV_PATH/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache}"
if [[ "$BACKBONE" == "esmc" ]]; then
  export PYTHONPATH="$BIOHUB_ESM_PATH:$ESMC_OVERLAY_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi

IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
WORLD_SIZE="${#GPUS[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain at least one GPU" >&2
  exit 2
fi
LOCAL_DEVICE_IDS="$(seq -s, 0 $((WORLD_SIZE - 1)))"

LOG_DIR="logs/SLEEC/${RUN_NAME}"
RUN_LOG="${LOG_DIR}/run.log"
PID_DIR="run_pids/SLEEC"
mkdir -p "$LOG_DIR" "$PID_DIR" "$OUTPUT_DIR" "$(dirname "$RESIDUE_H5")" "$HF_HOME"
printf '%s\n' "$$" > "${PID_DIR}/${RUN_NAME}.pid"

{
  echo "[$(date --iso-8601=seconds)] Starting SLEEC Stage 1 backbone run"
  echo "Backbone: $BACKBONE"
  echo "Run name: $RUN_NAME"
  echo "Project: $PROJECT_ROOT"
  echo "Environment: $ENV_PATH"
  echo "Python: $PYTHON_BIN"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Local device ids for DataParallel: $LOCAL_DEVICE_IDS"
  echo "Config: $CONFIG_PATH"
  echo "FASTA: $FASTA_PATH"
  echo "Residue HDF5: $RESIDUE_H5"
  echo "Model name: $MODEL_NAME"
  if [[ "$BACKBONE" == "esmc" ]]; then
    echo "ESMC backend: $ESMC_BACKEND"
    echo "Biohub ESM source: $BIOHUB_ESM_PATH"
    echo "ESMC dependency overlay: $ESMC_OVERLAY_PATH"
  fi
  echo "Output dir: $OUTPUT_DIR"

  if [[ "$TRAIN_ONLY" != "true" ]]; then
    if [[ -e "$RESIDUE_H5" && "$FORCE_EXTRACT" != "true" ]]; then
      echo "[$(date --iso-8601=seconds)] Residue HDF5 exists; skipping extraction"
    elif [[ "$SKIP_EXTRACT" == "true" ]]; then
      echo "Residue HDF5 is missing and --skip-extract was set: $RESIDUE_H5" >&2
      exit 1
    else
      EXTRACT_ARGS=(
        --run-name "SLEEC/${RUN_NAME}_extract"
        --fasta "$FASTA_PATH"
        --output "$RESIDUE_H5"
        --model-name "$MODEL_NAME"
        --batch-size "$EXTRACT_BATCH_SIZE"
        --max-tokens-per-batch "$MAX_TOKENS_PER_BATCH"
        --max-sequence-length "$MAX_SEQUENCE_LENGTH"
        --dtype "$DTYPE"
        --compression "$COMPRESSION"
      )
      if [[ "$FORCE_EXTRACT" == "true" ]]; then
        EXTRACT_ARGS+=(--force)
      fi
      if [[ "$CLEANUP_SHARDS" == "true" ]]; then
        EXTRACT_ARGS+=(--cleanup-shards)
      fi
      if [[ "$BACKBONE" == "esmc" ]]; then
        EXTRACT_ARGS+=(--backend "$ESMC_BACKEND")
        EXTRACT_ARGS+=(--hidden-layer "$HIDDEN_LAYER")
        if [[ "$TRUST_REMOTE_CODE" == "true" ]]; then
          EXTRACT_ARGS+=(--trust-remote-code)
        fi
      fi
      echo "[$(date --iso-8601=seconds)] Extracting residue embeddings with $EXTRACT_RUNNER"
      "$EXTRACT_RUNNER" "${EXTRACT_ARGS[@]}"
    fi
  fi

  echo "[$(date --iso-8601=seconds)] Validating SLEEC Stage 1 data"
  "$PYTHON_BIN" scripts/validate_sleec_stage1_data.py \
    --labels "$MCSA_LABELS_PATH" \
    --embeddings "$RESIDUE_H5" \
    --fasta "$FASTA_PATH" \
    --expected-dim "$EXPECTED_DIM" \
    > "${LOG_DIR}/validation.json"
  cat "${LOG_DIR}/validation.json"

  if [[ "$EXTRACT_ONLY" == "true" ]]; then
    echo "[$(date --iso-8601=seconds)] --extract-only set; stopping before training"
    exit 0
  fi

  TRAIN_ARGS=(
    scripts/train_sleec_stage1.py
    --config "$CONFIG_PATH"
    --output-dir "$OUTPUT_DIR"
    --data.residue_embeds_path "$RESIDUE_H5"
    --data.mcsa_labels_path "$MCSA_LABELS_PATH"
    --data.pseudo_labels_path "$PSEUDO_LABELS_PATH"
    --model.input_dim auto
    --training.data_parallel true
    --training.device_ids "$LOCAL_DEVICE_IDS"
    --logging.output_dir "$OUTPUT_DIR"
    --logging.wandb.run_name "$RUN_NAME"
  )
  if [[ "$WANDB" == "true" ]]; then
    TRAIN_ARGS+=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-run-name "$RUN_NAME"
      --wandb-tags sleec stage1 "$BACKBONE" unpooled threshold-sweep 4gpu
    )
    if [[ -n "$WANDB_ENTITY" ]]; then
      TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
  else
    TRAIN_ARGS+=(--wandb-mode disabled --logging.wandb.enabled false)
  fi

  echo "[$(date --iso-8601=seconds)] Training SLEEC Stage 1 classifier"
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  echo "[$(date --iso-8601=seconds)] SLEEC Stage 1 run complete"
} 2>&1 | tee "$RUN_LOG"
