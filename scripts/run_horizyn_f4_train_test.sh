#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/horizyn_f4_in_domain_v1}"
case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT" ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
REACTION_T5_MODEL_PATH="${REACTION_T5_MODEL_PATH:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache/hub/models--sagawa--ReactionT5v2-forward/snapshots/933114058cb2604dc1bf536dbebdfcefbe83d4fc}"
COFACTOR_DICTIONARY="${COFACTOR_DICTIONARY:-$ROOT/data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv}"
PROTEIN_H5="$ROOT/data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/proteins_prott5_residue.h5"
TRAIN_CONFIG="$ROOT/configs/horizyn_f4_train.yaml"
TEST_CONFIG="$ROOT/configs/horizyn_f4_test.yaml"
SUITE_CONFIG="$ROOT/configs/benchmarks/horizyn_f4.yaml"
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-27840}"
WANDB_PROJECT="${WANDB_PROJECT:-horizyn-f4-in-domain-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-omnai}"
WANDB_MODE="${WANDB_MODE:-online}"
SHORT_TMPDIR="${HORIZYN_TMPDIR:-/tmp/hz_f4_${UID}}"

mkdir -p \
  "$RUN_ROOT"/{checkpoints,features/train,features/test,features/reaction_set,logs,protocol,results,target_cache,nohome,tmp} \
  "$RUN_ROOT"/cache/{huggingface,torch} \
  "$RUN_ROOT"/{xdg_cache,xdg_config,xdg_state,wandb,matplotlib} \
  "$SHORT_TMPDIR"

export HOME="$RUN_ROOT/nohome"
export TMPDIR="$SHORT_TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UNIMOL_WEIGHT_DIR="$ROOT/../unimol_weights"
export PYTHONPATH="$ROOT"

if [[ -z "${WANDB_API_KEY:-}" && -s "$ROOT/runs/reactzyme_reaction_features_v1/wandb/api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "$ROOT/runs/reactzyme_reaction_features_v1/wandb/api_key")"
fi

status() {
  printf '{"time":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "${3:-}" >> "$RUN_ROOT/logs/status.jsonl"
}

run_logged() {
  local stage="$1"
  local log="$2"
  shift 2
  status "$stage" start
  set +e
  "$@" > "$log" 2>&1
  local rc=$?
  set -e
  status "$stage" end "$rc"
  return "$rc"
}

cd "$ROOT"

if [[ ! -s "$RUN_ROOT/protocol/manifest.json" ]]; then
  run_logged protocol "$RUN_ROOT/logs/protocol.log" \
    "$PYTHON_BIN" scripts/build_horizyn_f4_protocol.py \
      --source-dir data/sota \
      --out-dir "$RUN_ROOT/protocol" \
      --protein-h5 "$PROTEIN_H5" \
      --validation-fraction 0.1 \
      --seed 42
fi

if [[ ! -s "$RUN_ROOT/features/reaction_set/test_reaction_set_features.npz" ]]; then
  run_logged reaction_set "$RUN_ROOT/logs/reaction_set.log" \
    "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
      --train-reactions "$RUN_ROOT/protocol/train_rxns.csv" \
      --validation-reactions "$RUN_ROOT/protocol/validation_rxns.csv" \
      --test-reactions data/sota/test_rxns.csv \
      --cofactor-dictionary "$COFACTOR_DICTIONARY" \
      --out-dir "$RUN_ROOT/features/reaction_set"
fi

feature_pids=()
feature_names=()
if [[ ! -s "$RUN_ROOT/features/train/reactiont5v2.h5" ]]; then
  run_logged reactiont5_train "$RUN_ROOT/logs/reactiont5_train.log" \
    env CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions data/sota/train_rxns.csv \
      --output "$RUN_ROOT/features/train/reactiont5v2.h5" \
      --model-name "$REACTION_T5_MODEL_PATH" \
      --batch-size 64 --max-length 512 --pooling mean \
      --device cuda --dtype float16 --bidirectional --no-allow-pseudo-reactions --force &
  feature_pids+=("$!")
  feature_names+=(reactiont5_train)
fi
if [[ ! -s "$RUN_ROOT/features/train/unimol2.h5" ]]; then
  run_logged unimol2_train "$RUN_ROOT/logs/unimol2_train.log" \
    env PYTHONPATH="$ROOT/.deps/unimol_tools:$ROOT/../env/unimol2_site:$ROOT" \
      CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions data/sota/train_rxns.csv \
      --output "$RUN_ROOT/features/train/unimol2.h5" \
      --batch-size 64 --dtype float16 --compression none --bidirectional \
      --no-allow-pseudo-reactions --skip-invalid-molecules --skip-invalid-reactions --force &
  feature_pids+=("$!")
  feature_names+=(unimol2_train)
fi
if [[ ! -s "$RUN_ROOT/features/train/chiro.h5" ]]; then
  run_logged chiro_train "$RUN_ROOT/logs/chiro_train.log" \
    env PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
      CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions data/sota/train_rxns.csv \
      --output "$RUN_ROOT/features/train/chiro.h5" \
      --device cuda --num-workers 4 --batch-size 64 --bidirectional \
      --no-allow-pseudo-reactions --force &
  feature_pids+=("$!")
  feature_names+=(chiro_train)
fi

for index in "${!feature_pids[@]}"; do
  if ! wait "${feature_pids[$index]}"; then
    echo "Feature extraction failed: ${feature_names[$index]}" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY'
from horizyn.config import load_config
load_config("configs/horizyn_f4_train.yaml")
print("Horizyn F4 training config is valid")
PY

if [[ ! -s "$RUN_ROOT/checkpoints/selected_checkpoint.txt" ]]; then
  train_args=(
    "$PYTHON_BIN" scripts/train_protein_pooling.py
    --config "$TRAIN_CONFIG"
  )
  if [[ -s "$RUN_ROOT/checkpoints/last.ckpt" ]]; then
    train_args+=(--resume "$RUN_ROOT/checkpoints/last.ckpt")
  fi
  if [[ "$WANDB_MODE" != disabled ]]; then
    train_args+=(
      --wandb
      --wandb-project "$WANDB_PROJECT"
      --wandb-entity "$WANDB_ENTITY"
      --wandb-run-name horizyn-F4-factorized-reaction-seed42
      --wandb-mode "$WANDB_MODE"
    )
  fi
  run_logged train "$RUN_ROOT/logs/train.log" \
    env CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$MASTER_PORT" "${train_args[@]}"
  checkpoint="$(
    awk -F'Best checkpoint: ' '/Best checkpoint:/ {value=$2} END {print value}' \
      "$RUN_ROOT/logs/train.log" | tr -d '\r'
  )"
  if [[ -z "$checkpoint" || ! -s "$checkpoint" ]]; then
    echo "Training completed without a readable best checkpoint" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint" > "$RUN_ROOT/checkpoints/selected_checkpoint.txt"
fi

test_pids=()
test_names=()
if [[ ! -s "$RUN_ROOT/features/test/reactiont5v2.h5" ]]; then
  run_logged reactiont5_test "$RUN_ROOT/logs/reactiont5_test.log" \
    env CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test/reactiont5v2.h5" \
      --model-name "$REACTION_T5_MODEL_PATH" \
      --batch-size 64 --max-length 512 --pooling mean \
      --device cuda --dtype float16 --no-bidirectional --no-allow-pseudo-reactions --force &
  test_pids+=("$!")
  test_names+=(reactiont5_test)
fi
if [[ ! -s "$RUN_ROOT/features/test/unimol2.h5" ]]; then
  run_logged unimol2_test "$RUN_ROOT/logs/unimol2_test.log" \
    env PYTHONPATH="$ROOT/.deps/unimol_tools:$ROOT/../env/unimol2_site:$ROOT" \
      CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test/unimol2.h5" \
      --batch-size 64 --dtype float16 --compression none --no-bidirectional \
      --no-allow-pseudo-reactions --skip-invalid-molecules --skip-invalid-reactions --force &
  test_pids+=("$!")
  test_names+=(unimol2_test)
fi
if [[ ! -s "$RUN_ROOT/features/test/chiro.h5" ]]; then
  run_logged chiro_test "$RUN_ROOT/logs/chiro_test.log" \
    env PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
      CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions data/sota/test_rxns.csv \
      --output "$RUN_ROOT/features/test/chiro.h5" \
      --device cuda --num-workers 4 --batch-size 64 --no-bidirectional \
      --no-allow-pseudo-reactions --force &
  test_pids+=("$!")
  test_names+=(chiro_test)
fi

for index in "${!test_pids[@]}"; do
  if ! wait "${test_pids[$index]}"; then
    echo "Test feature extraction failed: ${test_names[$index]}" >&2
    exit 1
  fi
done

checkpoint="$(tr -d '\r\n' < "$RUN_ROOT/checkpoints/selected_checkpoint.txt")"
if [[ ! -s "$RUN_ROOT/results/horizyn_f4_test.json" ]]; then
  run_logged test "$RUN_ROOT/logs/test.log" \
    env CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" scripts/run_unified_retrieval_benchmark.py \
      --checkpoint "$checkpoint" \
      --config "$TEST_CONFIG" \
      --suite "$SUITE_CONFIG" \
      --tasks horizyn_f4_test \
      --protein-embedding prott5 \
      --score-protein-embedding prott5 \
      --output-dir "$RUN_ROOT/results" \
      --target-cache-dir "$RUN_ROOT/target_cache" \
      --device cuda \
      --query-batch-size 256 \
      --target-batch-size 2048
fi

echo "Horizyn F4 train/test chain complete"
echo "Checkpoint: $checkpoint"
echo "Results: $RUN_ROOT/results/horizyn_f4_test.json"
