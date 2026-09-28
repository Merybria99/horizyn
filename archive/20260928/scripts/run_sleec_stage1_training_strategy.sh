#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PAPER_CONFIG="${PAPER_CONFIG:-configs/sleec_stage1_esm2_650m.yaml}"
REGULARIZED_CONFIG="${REGULARIZED_CONFIG:-configs/sleec_stage1_esm2_650m_regularized.yaml}"
MAX_STEPS="${MAX_STEPS:-10000}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
RUN_FULL="${RUN_FULL:-1}"
LOG_DIR="${LOG_DIR:-logs/SLEEC}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/datastor2/deep-proteins/EnzymeDiscovery/.uv-cache}"
export UV_CACHE_DIR

PAPER_OUT="${PAPER_OUT:-checkpoints/SLEEC/sleec_stage1_esm2_650m_paper}"
REGULARIZED_OUT="${REGULARIZED_OUT:-checkpoints/SLEEC/sleec_stage1_esm2_650m_regularized}"
PAPER_SMOKE_OUT="${PAPER_SMOKE_OUT:-checkpoints/SLEEC/sleec_stage1_esm2_650m_paper_smoke}"
REGULARIZED_SMOKE_OUT="${REGULARIZED_SMOKE_OUT:-checkpoints/SLEEC/sleec_stage1_esm2_650m_regularized_smoke}"
SUMMARY_JSON="${SUMMARY_JSON:-checkpoints/SLEEC/sleec_stage1_esm2_650m_comparison.json}"
SUMMARY_CSV="${SUMMARY_CSV:-checkpoints/SLEEC/sleec_stage1_esm2_650m_comparison.csv}"
WANDB_ENABLED="${WANDB_ENABLED:-${WANDB:-0}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-training}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"
GPUS="${GPUS:-0,1,2,3}"
DATA_PARALLEL="${DATA_PARALLEL:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPUS}"

mkdir -p "$LOG_DIR"

run_train() {
  local label="$1"
  local config="$2"
  local out_dir="$3"
  local steps="$4"
  local eval_every="$5"
  local save_every="$6"
  local log_file="$LOG_DIR/sleec_stage1_${label}_$(date +%Y%m%d_%H%M%S).log"
  local train_args=(
    --config "$config"
    --output-dir "$out_dir"
    --training.max_steps="$steps"
    --training.eval_every_steps="$eval_every"
    --training.save_every_steps="$save_every"
    --data.num_workers=0
    --training.data_parallel="$DATA_PARALLEL"
    --training.device_ids="$GPUS"
  )
  case "${WANDB_ENABLED,,}" in
    1|true|yes)
      train_args+=(
        --wandb
        --wandb-mode "$WANDB_MODE"
        --wandb-project "$WANDB_PROJECT"
        --wandb-run-name "$label"
        --wandb-tags sleec stage1 "$label" esm2-650m uniref50
      )
      if [[ -n "$WANDB_ENTITY" ]]; then
        train_args+=(--wandb-entity "$WANDB_ENTITY")
      fi
      case "${WANDB_LOG_MODEL,,}" in
        1|true|yes) train_args+=(--wandb-log-model) ;;
      esac
      ;;
  esac

  echo "[$(date --iso-8601=seconds)] Starting ${label}: config=${config}, steps=${steps}, out=${out_dir}"
  uv run --python 3.12 python scripts/train_sleec_stage1.py \
    "${train_args[@]}" \
    2>&1 | tee "$log_file"
}

echo "[$(date --iso-8601=seconds)] Validating SLEEC stage-1 data"
uv run --python 3.12 python scripts/validate_sleec_stage1_data.py \
  --labels data/sleec_stage1/mcsa_residue_labels.csv \
  --embeddings data/sleec_stage1/mcsa_cath_esm2_650m_residue.h5 \
  --fasta data/sleec_stage1/fasta/mcsa_reference.fasta \
  --msa-dir data/sleec_stage1/msas/mcsa_uniref50 \
  --msa-glob "**/*_rm_ins.a3m"

run_train "paper_smoke" "$PAPER_CONFIG" "$PAPER_SMOKE_OUT" "$SMOKE_STEPS" 1 1
run_train "regularized_smoke" "$REGULARIZED_CONFIG" "$REGULARIZED_SMOKE_OUT" "$SMOKE_STEPS" 1 1

if [[ "$RUN_FULL" == "1" || "$RUN_FULL" == "true" ]]; then
  run_train "paper" "$PAPER_CONFIG" "$PAPER_OUT" "$MAX_STEPS" 500 1000
  run_train "regularized" "$REGULARIZED_CONFIG" "$REGULARIZED_OUT" "$MAX_STEPS" 500 1000

  uv run --python 3.12 python scripts/summarize_sleec_stage1_runs.py \
    --run "paper=$PAPER_OUT" \
    --run "regularized=$REGULARIZED_OUT" \
    --output-json "$SUMMARY_JSON" \
    --output-csv "$SUMMARY_CSV"
else
  echo "[$(date --iso-8601=seconds)] RUN_FULL=${RUN_FULL}; skipping full training runs"
fi
