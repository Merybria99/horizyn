#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/env}"
PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/setup_wandb_env.sh"

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
CONFIG="${CONFIG:-configs/retrieval_source_collapse_nr90_esmc_sleec_hyperbolic_c0p25.yaml}"
BENCHMARK_SUITE="${BENCHMARK_SUITE:-configs/benchmarks/retrieval_source_collapse_tests.yaml}"

RUN_PREPARE="${RUN_PREPARE:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
EXTRACT_EMBEDDINGS="${EXTRACT_EMBEDDINGS:-true}"
FORCE_EXTRACT="${FORCE_EXTRACT:-false}"
CLEANUP_SHARDS="${CLEANUP_SHARDS:-true}"
USE_VALID_REACTIONS="${USE_VALID_REACTIONS:-true}"
FILTER_VALID_REACTIONS="${FILTER_VALID_REACTIONS:-true}"
VALID_RXN_SUFFIX="${VALID_RXN_SUFFIX:-_valid_rxn}"
FILTER_NONEMPTY_RESIDUE_PROTEINS="${FILTER_NONEMPTY_RESIDUE_PROTEINS:-true}"
NONEMPTY_RESIDUE_SUFFIX="${NONEMPTY_RESIDUE_SUFFIX:-_nonempty_esmc}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-100}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-retrieval-source-collapse}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_LOG_MODEL="${WANDB_LOG_MODEL:-0}"

BENCHMARK_WANDB_PROJECT="${BENCHMARK_WANDB_PROJECT:-horizyn-retrieval-source-collapse-benchmarks}"
BENCHMARK_WANDB_MODE="${BENCHMARK_WANDB_MODE:-$WANDB_MODE}"
BENCHMARK_WANDB_ENTITY="${BENCHMARK_WANDB_ENTITY:-$WANDB_ENTITY}"
LOG_BENCHMARK_WANDB="${LOG_BENCHMARK_WANDB:-true}"

ESMC_BATCH_SIZE="${ESMC_BATCH_SIZE:-1}"
ESMC_MAX_TOKENS_PER_BATCH="${ESMC_MAX_TOKENS_PER_BATCH:-2048}"
ESMC_MAX_SEQUENCE_LENGTH="${ESMC_MAX_SEQUENCE_LENGTH:-1022}"
ESMC_HIDDEN_LAYER="${ESMC_HIDDEN_LAYER:--1}"
ESMC_DTYPE="${ESMC_DTYPE:-float16}"
ESMC_COMPRESSION="${ESMC_COMPRESSION:-none}"
ESMC_BACKEND="${ESMC_BACKEND:-biohub}"
ESMC_MODEL_NAME="${ESMC_MODEL_NAME:-Biohub/ESMC-6B}"

EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
EVAL_QUERY_BATCH_SIZE="${EVAL_QUERY_BATCH_SIZE:-128}"
EVAL_TARGET_BATCH_SIZE="${EVAL_TARGET_BATCH_SIZE:-128}"
STORE_TARGETS_ON_CPU="${STORE_TARGETS_ON_CPU:-true}"

DATA_ROOT="data/standardized/retrieval_training_source_collapse"
TRAIN_ROOT="${DATA_ROOT}/train_${POLICY}"
VAL_ROOT="${DATA_ROOT}/validation/clipzyme_eval"
TEST_SHARED_ROOT="${DATA_ROOT}/test/horizyn_reactzyme_shared_candidates"

TRAIN_PAIRS_RAW="${TRAIN_ROOT}/train_pairs.csv"
TRAIN_RXNS_RAW="${TRAIN_ROOT}/train_rxns.csv"
TRAIN_PAIRS_VALID="${TRAIN_ROOT}/train_pairs${VALID_RXN_SUFFIX}.csv"
TRAIN_RXNS_VALID="${TRAIN_ROOT}/train_rxns${VALID_RXN_SUFFIX}.csv"
TRAIN_PAIRS="$TRAIN_PAIRS_RAW"
TRAIN_RXNS="$TRAIN_RXNS_RAW"
VAL_PAIRS_RAW="${VAL_ROOT}/pairs_horizyn.csv"
VAL_PAIRS="$VAL_PAIRS_RAW"
VAL_RXNS="${VAL_ROOT}/reactions.csv"
FIT_FASTA="${TRAIN_ROOT}/fit_proteins_with_clipzyme_eval.fasta"
FIT_H5="${TRAIN_ROOT}/fit_proteins_with_clipzyme_eval_esmc_6b_residue.h5"
FIT_TMP_DIR="${TRAIN_ROOT}/fit_proteins_with_clipzyme_eval_esmc_6b_residue_shards"
TEST_FASTA="${TEST_SHARED_ROOT}/proteins.fasta"
TEST_H5="${TEST_SHARED_ROOT}/proteins_esmc_6b_residue.h5"
TEST_TMP_DIR="${TEST_SHARED_ROOT}/proteins_esmc_6b_residue_shards"

RUN_NAME="${RUN_NAME:-horizyn-source-collapse-${POLICY}-esmc-sleec-lorentz-c0p25-4gpu-${RUN_STAMP}}"
LOG_DIR="logs/SLEEC/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/SLEEC/${RUN_NAME}"
RESULTS_DIR="results/benchmarks/${RUN_NAME}/retrieval_source_collapse_tests"
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

select_train_reaction_files() {
  if ! bool_true "$USE_VALID_REACTIONS"; then
    TRAIN_PAIRS="$TRAIN_PAIRS_RAW"
    TRAIN_RXNS="$TRAIN_RXNS_RAW"
    echo "[$(date --iso-8601=seconds)] USE_VALID_REACTIONS=false; using raw train reactions"
    return 0
  fi

  if [[ ! -s "$TRAIN_PAIRS_VALID" || ! -s "$TRAIN_RXNS_VALID" ]]; then
    if ! bool_true "$FILTER_VALID_REACTIONS"; then
      echo "Missing valid-reaction files and FILTER_VALID_REACTIONS=false:" >&2
      echo "  $TRAIN_PAIRS_VALID" >&2
      echo "  $TRAIN_RXNS_VALID" >&2
      exit 2
    fi
    echo "[$(date --iso-8601=seconds)] Creating valid-reaction filtered source-collapse train files"
    "$PYTHON_BIN" scripts/filter_standardized_retrieval_reaction_smiles.py \
      --root "$DATA_ROOT" \
      --policies "$POLICY" \
      --policy-dir-template 'train_{policy}' \
      --splits train \
      --suffix "$VALID_RXN_SUFFIX"
  fi

  TRAIN_PAIRS="$TRAIN_PAIRS_VALID"
  TRAIN_RXNS="$TRAIN_RXNS_VALID"
  require_path "$TRAIN_PAIRS"
  require_path "$TRAIN_RXNS"
  echo "[$(date --iso-8601=seconds)] Using valid-reaction train pairs: $TRAIN_PAIRS"
  echo "[$(date --iso-8601=seconds)] Using valid-reaction train reactions: $TRAIN_RXNS"
}

filter_pairs_nonempty() {
  local input_pairs="$1"
  local output_pairs="$2"
  local label="$3"

  if [[ ! -s "$output_pairs" || "$input_pairs" -nt "$output_pairs" || "$FIT_H5" -nt "$output_pairs" ]]; then
    echo "[$(date --iso-8601=seconds)] Filtering $label pairs to non-empty ESMC residue embeddings"
    "$PYTHON_BIN" scripts/filter_pairs_by_residue_h5.py \
      --pairs "$input_pairs" \
      --output "$output_pairs" \
      --h5 "$FIT_H5"
  else
    echo "[$(date --iso-8601=seconds)] Reusing non-empty $label pairs: $output_pairs"
  fi
}

select_nonempty_residue_pair_files() {
  if ! bool_true "$FILTER_NONEMPTY_RESIDUE_PROTEINS"; then
    echo "[$(date --iso-8601=seconds)] FILTER_NONEMPTY_RESIDUE_PROTEINS=false; using unfiltered protein pairs"
    return 0
  fi

  require_path "$FIT_H5"
  local train_pairs_nonempty="${TRAIN_PAIRS%.csv}${NONEMPTY_RESIDUE_SUFFIX}.csv"
  local val_pairs_nonempty="${VAL_PAIRS%.csv}${NONEMPTY_RESIDUE_SUFFIX}.csv"

  filter_pairs_nonempty "$TRAIN_PAIRS" "$train_pairs_nonempty" "train"
  filter_pairs_nonempty "$VAL_PAIRS" "$val_pairs_nonempty" "validation"

  TRAIN_PAIRS="$train_pairs_nonempty"
  VAL_PAIRS="$val_pairs_nonempty"
  require_path "$TRAIN_PAIRS"
  require_path "$VAL_PAIRS"
  echo "[$(date --iso-8601=seconds)] Using non-empty train pairs: $TRAIN_PAIRS"
  echo "[$(date --iso-8601=seconds)] Using non-empty validation pairs: $VAL_PAIRS"
}

extract_esmc_if_missing() {
  local fasta="$1"
  local output="$2"
  local tmp_dir="$3"
  local name="$4"

  require_path "$fasta"
  if [[ -s "$output" ]] && ! bool_true "$FORCE_EXTRACT"; then
    echo "[$(date --iso-8601=seconds)] Reusing existing ESMC residue HDF5: $output"
    return 0
  fi

  if ! bool_true "$EXTRACT_EMBEDDINGS"; then
    echo "Missing ESMC residue HDF5 and EXTRACT_EMBEDDINGS=false: $output" >&2
    exit 2
  fi

  local args=(
    --run-name "$name"
    --fasta "$fasta"
    --output "$output"
    --tmp-dir "$tmp_dir"
    --model-name "$ESMC_MODEL_NAME"
    --backend "$ESMC_BACKEND"
    --batch-size "$ESMC_BATCH_SIZE"
    --max-tokens-per-batch "$ESMC_MAX_TOKENS_PER_BATCH"
    --max-sequence-length "$ESMC_MAX_SEQUENCE_LENGTH"
    --hidden-layer "$ESMC_HIDDEN_LAYER"
    --dtype "$ESMC_DTYPE"
    --compression "$ESMC_COMPRESSION"
  )
  if bool_true "$FORCE_EXTRACT"; then
    args+=(--force)
  fi
  if bool_true "$CLEANUP_SHARDS"; then
    args+=(--cleanup-shards)
  fi

  echo "[$(date --iso-8601=seconds)] Extracting ESMC residue embeddings: $output"
  scripts/run_extract_esmc_residue_embeddings_4gpu.sh "${args[@]}"
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
    --wandb-tags horizyn source-collapse "source-collapse-${POLICY}" esmc-6b sleec-guided-attention frozen-lorentz-hyperbolic-c0p25 mlnce 4gpu
    --data.train_pairs_path "$TRAIN_PAIRS"
    --data.train_reactions_path "$TRAIN_RXNS"
    --data.test_pairs_path "$VAL_PAIRS"
    --data.test_reactions_path "$VAL_RXNS"
    --data.protein_residue_embeds_path "$FIT_H5"
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
  require_path "$TRAIN_RXNS"
  require_path "$VAL_PAIRS"
  require_path "$VAL_RXNS"
  require_path "$FIT_H5"

  echo "[$(date --iso-8601=seconds)] Starting collapsed-source ESMC Horizyn retrieval training"
  "$PYTHON_BIN" scripts/train_protein_pooling.py "${train_args[@]}" 2>&1 | tee "$TRAIN_LOG"
}

run_benchmark() {
  local checkpoint="$1"
  local benchmark_args=(
    --checkpoint "$checkpoint"
    --config "$CONFIG"
    --suite "$BENCHMARK_SUITE"
    --tasks all
    --protein-embedding esmc
    --score-protein-embedding esmc
    --output-dir "$RESULTS_DIR"
    --device "$EVAL_DEVICE"
    --query-batch-size "$EVAL_QUERY_BATCH_SIZE"
    --target-batch-size "$EVAL_TARGET_BATCH_SIZE"
  )
  if bool_true "$STORE_TARGETS_ON_CPU"; then
    benchmark_args+=(--store-targets-on-cpu)
  fi

  require_path "$checkpoint"
  require_path "$TEST_H5"

  echo "[$(date --iso-8601=seconds)] Running Horizyn + ReactZyme ESMC held-out tests"
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
  echo "[$(date --iso-8601=seconds)] Starting collapsed-source ESMC retrieval train/eval chain"
  echo "Root: $ROOT_DIR"
  echo "Python: $PYTHON_BIN"
  echo "Policy: $POLICY"
  echo "Config: $CONFIG"
  echo "Benchmark suite: $BENCHMARK_SUITE"
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Run name: $RUN_NAME"
  echo "Raw train pairs: $TRAIN_PAIRS_RAW"
  echo "Raw train reactions: $TRAIN_RXNS_RAW"
  echo "Use valid reactions: $USE_VALID_REACTIONS"
  echo "Raw validation pairs: $VAL_PAIRS_RAW"
  echo "Validation reactions: $VAL_RXNS"
  echo "Filter non-empty residue proteins: $FILTER_NONEMPTY_RESIDUE_PROTEINS"
  echo "Fit FASTA: $FIT_FASTA"
  echo "Fit ESMC HDF5: $FIT_H5"
  echo "Test FASTA: $TEST_FASTA"
  echo "Test ESMC HDF5: $TEST_H5"
  echo "Checkpoint dir: $CHECKPOINT_DIR"
  echo "Results dir: $RESULTS_DIR"
  echo "W&B training project: $WANDB_PROJECT"
  echo "W&B benchmark project: $BENCHMARK_WANDB_PROJECT"
  nvidia-smi || true

  if bool_true "$RUN_PREPARE"; then
    "$PYTHON_BIN" scripts/prepare_retrieval_source_collapse_pipeline_inputs.py \
      --root "$DATA_ROOT" \
      --variants exact nr90 nr50
  fi

  select_train_reaction_files

  if bool_true "$RUN_TRAIN"; then
    extract_esmc_if_missing "$FIT_FASTA" "$FIT_H5" "$FIT_TMP_DIR" "extract-source-collapse-${POLICY}-fit-esmc-${RUN_STAMP}"
    select_nonempty_residue_pair_files
    run_training
  else
    echo "[$(date --iso-8601=seconds)] RUN_TRAIN=false; skipping training"
  fi

  if bool_true "$RUN_EVAL"; then
    extract_esmc_if_missing "$TEST_FASTA" "$TEST_H5" "$TEST_TMP_DIR" "extract-source-collapse-test-esmc-${RUN_STAMP}"
    EVAL_CHECKPOINT="${EVAL_CHECKPOINT:-$(resolve_checkpoint_for_eval)}"
    run_benchmark "$EVAL_CHECKPOINT"
  else
    echo "[$(date --iso-8601=seconds)] RUN_EVAL=false; skipping benchmark"
  fi

  echo "[$(date --iso-8601=seconds)] Chain complete"
} 2>&1 | tee "$CHAIN_LOG"
