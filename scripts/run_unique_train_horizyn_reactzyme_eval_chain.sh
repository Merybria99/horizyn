#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
cd "$ROOT_DIR"

POLICY="${POLICY:-nr90}"
case "$POLICY" in
  exact|nr90|nr50) ;;
  *)
    echo "POLICY must be one of: exact, nr90, nr50 (got '$POLICY')" >&2
    exit 2
    ;;
esac

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/horizyn_unique_nr90_esm2_sleec_hyperbolic_c0p25.yaml}"
BENCHMARK_SUITE="${BENCHMARK_SUITE:-configs/benchmarks/horizyn_reactzyme_eval.yaml}"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
EXTRACT_EMBEDDINGS="${EXTRACT_EMBEDDINGS:-true}"
FORCE_EXTRACT="${FORCE_EXTRACT:-false}"
CLEANUP_SHARDS="${CLEANUP_SHARDS:-true}"
USE_VALID_REACTIONS="${USE_VALID_REACTIONS:-false}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-unique-training}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"

BENCHMARK_WANDB_PROJECT="${BENCHMARK_WANDB_PROJECT:-horizyn-unique-benchmarks}"
BENCHMARK_WANDB_MODE="${BENCHMARK_WANDB_MODE:-$WANDB_MODE}"
BENCHMARK_WANDB_ENTITY="${BENCHMARK_WANDB_ENTITY:-$WANDB_ENTITY}"
LOG_BENCHMARK_WANDB="${LOG_BENCHMARK_WANDB:-true}"

ESM_BATCH_SIZE="${ESM_BATCH_SIZE:-8}"
ESM_MAX_TOKENS_PER_BATCH="${ESM_MAX_TOKENS_PER_BATCH:-4096}"
ESM_DTYPE="${ESM_DTYPE:-float16}"
ESM_COMPRESSION="${ESM_COMPRESSION:-none}"
ESM_MAX_SEQUENCE_LENGTH="${ESM_MAX_SEQUENCE_LENGTH:-1022}"

EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-128}"
EVAL_TARGET_BATCH_SIZE="${EVAL_TARGET_BATCH_SIZE:-256}"
STORE_TARGETS_ON_CPU="${STORE_TARGETS_ON_CPU:-true}"

UNIQUE_ROOT="data/standardized/global_unique_retrieval/${POLICY}"
EVAL_ROOT="data/standardized/horizyn_reactzyme_eval"
UNIQUE_FASTA="${UNIQUE_ROOT}/proteins.fasta"
UNIQUE_H5="${UNIQUE_ROOT}/proteins_esm2_650m_residue.h5"
EVAL_FASTA="${EVAL_ROOT}/proteins.fasta"
EVAL_H5="${EVAL_ROOT}/proteins_esm2_650m_residue.h5"
if [[ "${USE_VALID_REACTIONS,,}" =~ ^(1|true|yes|y)$ ]]; then
  TRAIN_PAIRS="${UNIQUE_ROOT}/train_pairs_valid_rxn.csv"
  TEST_PAIRS="${UNIQUE_ROOT}/test_pairs_valid_rxn.csv"
  TRAIN_RXNS="${UNIQUE_ROOT}/train_rxns_valid_rxn.csv"
  TEST_RXNS="${UNIQUE_ROOT}/test_rxns_valid_rxn.csv"
else
  TRAIN_PAIRS="${UNIQUE_ROOT}/train_pairs.csv"
  TEST_PAIRS="${UNIQUE_ROOT}/test_pairs.csv"
  TRAIN_RXNS="${UNIQUE_ROOT}/train_rxns.csv"
  TEST_RXNS="${UNIQUE_ROOT}/test_rxns.csv"
fi

RUN_NAME="${RUN_NAME:-horizyn-unique-${POLICY}-esm2-sleec-hyperbolic-c0p25-published-eval-4gpu-${RUN_STAMP}}"
LOG_DIR="logs/SLEEC/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/SLEEC/${RUN_NAME}"
RESULTS_DIR="results/benchmarks/${RUN_NAME}/horizyn_reactzyme_eval"
CHAIN_LOG="${LOG_DIR}/chain.log"
TRAIN_LOG="${LOG_DIR}/training.log"
BENCHMARK_LOG="${LOG_DIR}/benchmark.log"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR" "$RESULTS_DIR" run_pids/SLEEC
printf '%s\n' "$$" > "run_pids/SLEEC/${RUN_NAME}.pid"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y) return 0 ;;
    *) return 1 ;;
  esac
}

extract_if_missing() {
  local fasta="$1"
  local output="$2"
  local name="$3"

  require_path "$fasta"
  if [[ -s "$output" ]] && ! bool_true "$FORCE_EXTRACT"; then
    echo "[$(date --iso-8601=seconds)] Reusing existing ESM2 residue HDF5: $output"
    return 0
  fi

  if ! bool_true "$EXTRACT_EMBEDDINGS"; then
    echo "Missing ESM2 residue HDF5 and EXTRACT_EMBEDDINGS=false: $output" >&2
    exit 2
  fi

  local args=(
    --run-name "$name"
    --fasta "$fasta"
    --output "$output"
    --batch-size "$ESM_BATCH_SIZE"
    --max-tokens-per-batch "$ESM_MAX_TOKENS_PER_BATCH"
    --max-sequence-length "$ESM_MAX_SEQUENCE_LENGTH"
    --dtype "$ESM_DTYPE"
    --compression "$ESM_COMPRESSION"
  )
  if bool_true "$FORCE_EXTRACT"; then
    args+=(--force)
  fi
  if bool_true "$CLEANUP_SHARDS"; then
    args+=(--cleanup-shards)
  fi

  echo "[$(date --iso-8601=seconds)] Extracting ESM2 residue embeddings: $output"
  scripts/run_extract_esm2_residue_embeddings_4gpu.sh "${args[@]}"
}

resolve_checkpoint_for_eval() {
  local best=""
  if [[ -f "$TRAIN_LOG" ]]; then
    best="$(grep -E '^Best checkpoint:' "$TRAIN_LOG" | tail -n 1 | sed 's/^Best checkpoint:[[:space:]]*//')"
  fi
  if [[ -n "$best" && -f "$best" ]]; then
    printf '%s\n' "$best"
    return 0
  fi
  if [[ -f "${CHECKPOINT_DIR}/last.ckpt" ]]; then
    printf '%s\n' "${CHECKPOINT_DIR}/last.ckpt"
    return 0
  fi
  local newest
  newest="$(find "$CHECKPOINT_DIR" -maxdepth 1 -name '*.ckpt' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  if [[ -n "$newest" && -f "$newest" ]]; then
    printf '%s\n' "$newest"
    return 0
  fi
  echo "Could not resolve a checkpoint for evaluation in $CHECKPOINT_DIR" >&2
  return 1
}

run_training() {
  local train_args=(
    --config "$CONFIG"
    --wandb
    --wandb-mode "$WANDB_MODE"
    --wandb-project "$WANDB_PROJECT"
    --wandb-run-name "$RUN_NAME"
    --wandb-tags horizyn unique-training "unique-${POLICY}" esm2-650m sleec-guided-attention hyperbolic-c0p25 frozen-stage1 mlnce 4gpu
    --data.train_pairs_path "$TRAIN_PAIRS"
    --data.test_pairs_path "$TEST_PAIRS"
    --data.train_reactions_path "$TRAIN_RXNS"
    --data.test_reactions_path "$TEST_RXNS"
    --data.protein_residue_embeds_path "$UNIQUE_H5"
    --data.train_batch_size "$TRAIN_BATCH_SIZE"
    --training.max_epochs "$EPOCHS"
    --logging.log_dir "$LOG_DIR"
    --logging.checkpoint_dir "$CHECKPOINT_DIR"
  )
  if [[ -n "$WANDB_ENTITY" ]]; then
    train_args+=(--wandb-entity "$WANDB_ENTITY")
  fi
  case "${WANDB_LOG_MODEL,,}" in
    1|true|yes) train_args+=(--wandb-log-model) ;;
  esac

  require_path "$TRAIN_PAIRS"
  require_path "$TEST_PAIRS"
  require_path "$TRAIN_RXNS"
  require_path "$TEST_RXNS"
  require_path "$UNIQUE_H5"

  echo "[$(date --iso-8601=seconds)] Starting unique-dataset Horizyn training"
  "$PYTHON_BIN" scripts/train_protein_pooling.py "${train_args[@]}" 2>&1 | tee "$TRAIN_LOG"
}

run_benchmark() {
  local checkpoint="$1"
  local benchmark_args=(
    --checkpoint "$checkpoint"
    --config "$CONFIG"
    --suite "$BENCHMARK_SUITE"
    --tasks all
    --protein-embedding esm2
    --score-protein-embedding esm2
    --output-dir "$RESULTS_DIR"
    --device "$EVAL_DEVICE"
    --query-batch-size "$EVAL_QUERY_BATCH_SIZE"
    --target-batch-size "$EVAL_TARGET_BATCH_SIZE"
  )
  if bool_true "$STORE_TARGETS_ON_CPU"; then
    benchmark_args+=(--store-targets-on-cpu)
  fi

  require_path "$checkpoint"
  require_path "$EVAL_H5"

  echo "[$(date --iso-8601=seconds)] Running published Horizyn/ReactZyme benchmark"
  "$PYTHON_BIN" scripts/run_unified_retrieval_benchmark.py "${benchmark_args[@]}" 2>&1 | tee "$BENCHMARK_LOG"

  if bool_true "$LOG_BENCHMARK_WANDB"; then
    local wandb_args=(
      "$RESULTS_DIR"
      --project "$BENCHMARK_WANDB_PROJECT"
      --mode "$BENCHMARK_WANDB_MODE"
      --run-name "${RUN_NAME}-benchmark"
      --expected-tasks horizyn_sota reactzyme_time reactzyme_enzyme_smi reactzyme_reaction_smi
    )
    if [[ -n "$BENCHMARK_WANDB_ENTITY" ]]; then
      wandb_args+=(--entity "$BENCHMARK_WANDB_ENTITY")
    fi
    "$PYTHON_BIN" scripts/log_unified_retrieval_benchmark_to_wandb.py "${wandb_args[@]}"
  fi
}

{
  echo "[$(date --iso-8601=seconds)] Starting unique-train -> Horizyn/ReactZyme-eval chain"
  echo "Root: $ROOT_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Policy: $POLICY"
  echo "Config: $CONFIG"
  echo "Benchmark suite: $BENCHMARK_SUITE"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Run name: $RUN_NAME"
  echo "Unique FASTA: $UNIQUE_FASTA"
  echo "Unique ESM2 HDF5: $UNIQUE_H5"
  echo "Use valid reactions: $USE_VALID_REACTIONS"
  echo "Train pairs: $TRAIN_PAIRS"
  echo "Train reactions: $TRAIN_RXNS"
  echo "Validation pairs: $TEST_PAIRS"
  echo "Validation reactions: $TEST_RXNS"
  echo "Eval FASTA: $EVAL_FASTA"
  echo "Eval ESM2 HDF5: $EVAL_H5"
  echo "Checkpoint dir: $CHECKPOINT_DIR"
  echo "Results dir: $RESULTS_DIR"
  echo "W&B training project: $WANDB_PROJECT"
  echo "W&B benchmark project: $BENCHMARK_WANDB_PROJECT"
  nvidia-smi || true

  if bool_true "$RUN_TRAIN"; then
    extract_if_missing "$UNIQUE_FASTA" "$UNIQUE_H5" "extract-unique-${POLICY}-esm2-${RUN_STAMP}"
    run_training
  else
    echo "[$(date --iso-8601=seconds)] RUN_TRAIN=false; skipping training"
  fi

  if bool_true "$RUN_EVAL"; then
    extract_if_missing "$EVAL_FASTA" "$EVAL_H5" "extract-horizyn-reactzyme-eval-esm2-${RUN_STAMP}"
    EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-$(resolve_checkpoint_for_eval)}"
    run_benchmark "$EVAL_CHECKPOINT"
  else
    echo "[$(date --iso-8601=seconds)] RUN_EVAL=false; skipping benchmark"
  fi

  echo "[$(date --iso-8601=seconds)] Chain complete"
} 2>&1 | tee "$CHAIN_LOG"
