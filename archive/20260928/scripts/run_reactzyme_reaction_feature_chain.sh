#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/runs/reactzyme_reaction_features_v1"

RUN_ROOT_ARG="$DEFAULT_RUN_ROOT"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_ROOT_ARG="$1"
  shift
fi
case "$RUN_ROOT_ARG" in
  /*) RUN_ROOT="$RUN_ROOT_ARG" ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT_ARG" ;;
esac

DETACH=0
SKIP_SETUP=0
SKIP_TRAIN=0
FORCE_SETUP=0
WAIT_FOR_JOBS=1
ONLY_VARIANT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --force-setup) FORCE_SETUP=1; shift ;;
    --no-wait) WAIT_FOR_JOBS=0; shift ;;
    --variant) ONLY_VARIANT="${2:-}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  pid_file="$RUN_ROOT/logs/controller/pid"
  rm -f "$pid_file"
  detach_args=("$RUN_ROOT" --foreground)
  [[ "$SKIP_SETUP" -eq 1 ]] && detach_args+=(--skip-setup)
  [[ "$SKIP_TRAIN" -eq 1 ]] && detach_args+=(--skip-train)
  [[ "$FORCE_SETUP" -eq 1 ]] && detach_args+=(--force-setup)
  [[ "$WAIT_FOR_JOBS" -eq 0 ]] && detach_args+=(--no-wait)
  [[ -n "$ONLY_VARIANT" ]] && detach_args+=(--variant "$ONLY_VARIANT")
  setsid -f bash -c '
    pid_file="$1"
    shift
    printf "%s\n" "$$" > "$pid_file"
    exec bash "$@"
  ' bash "$pid_file" "$0" "${detach_args[@]}" \
    > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  for _ in {1..50}; do
    [[ -s "$pid_file" ]] && break
    sleep 0.1
  done
  echo "Launched ReactZyme reaction-feature controller"
  echo "PID: $(cat "$pid_file" 2>/dev/null || printf unknown)"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  echo "Status: $RUN_ROOT/logs/status.jsonl"
  exit 0
fi

RUNTIME_ENV="$RUN_ROOT/runtime.env"
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Missing runtime env: $RUNTIME_ENV" >&2
  echo "Run scripts/create_reactzyme_reaction_feature_matrix.py first." >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$RUNTIME_ENV"
cd "$ROOT"

SHORT_TMPDIR="${HORIZYN_TMPDIR:-/tmp/hz_reaction_features_v1}"
mkdir -p \
  "$RUN_ROOT/logs/controller" "$RUN_ROOT/logs/setup" "$RUN_ROOT/nohome" \
  "$RUN_ROOT/tmp" "$RUN_ROOT/cache/huggingface" "$RUN_ROOT/cache/torch" \
  "$RUN_ROOT/cache/pip" "$RUN_ROOT/xdg_cache" "$RUN_ROOT/xdg_config" \
  "$RUN_ROOT/xdg_state" "$RUN_ROOT/wandb" "$RUN_ROOT/matplotlib" \
  "$RUN_ROOT/deps/rxnmapper" "$SHORT_TMPDIR"

export HOME="$RUN_ROOT/nohome"
export TMPDIR="$SHORT_TMPDIR"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export WANDB_ROOT="$RUN_ROOT/wandb"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export UNIMOL_WEIGHT_DIR="$ROOT/../unimol_weights"
export PYTHONPATH="$ROOT"
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY
  WANDB_API_KEY="$(tr -d '\r\n' < "$RUN_ROOT/wandb/api_key")"
fi

STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
status() {
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "$3" "${4:-}" >> "$STATUS_LOG"
}

wait_for_unrelated_training() {
  [[ "$WAIT_FOR_JOBS" -eq 1 ]] || return 0
  while true; do
    local active
    active="$(pgrep -af 'scripts/train_protein_pooling.py' | grep -Fv "$RUN_ROOT" || true)"
    [[ -z "$active" ]] && return 0
    echo "[$(date -Iseconds)] Waiting for unrelated protein-pooling jobs to finish"
    status controller wait_for_gpus poll ""
    sleep 60
  done
}

ensure_rxnmapper() {
  local site="$RUN_ROOT/deps/rxnmapper"
  local pip_bin="${PIP_BIN:-/usr/bin/pip3}"
  if PYTHONPATH="$site:$ROOT" "$SETUP_PYTHON_BIN" -c \
      'from horizyn.capability.reaction_features import _build_rxnmapper; _build_rxnmapper(True)' \
      >/dev/null 2>&1; then
    return 0
  fi
  echo "[$(date -Iseconds)] Installing rxnmapper into $site"
  "$pip_bin" install --no-deps --target "$site" \
    'rxnmapper==0.4.2' 'rxn-chem-utils>=1.0.3' 'rxn-utils>=1.1.9' \
    'transformers==4.46.3' 'tokenizers==0.20.3' 'huggingface-hub==0.26.5' \
    > "$RUN_ROOT/logs/setup/rxnmapper_install.log" 2>&1
  PYTHONPATH="$site:$ROOT" "$SETUP_PYTHON_BIN" -c \
    'from horizyn.capability.reaction_features import _build_rxnmapper; _build_rxnmapper(True)'
}

setup_fingerprint() {
  local split="$1"
  sha256sum \
    configs/benchmarks/reactzyme_paper/reaction_features.yaml \
    scripts/run_reactzyme_reaction_feature_chain.sh \
    scripts/build_reaction_set_features.py \
    scripts/build_reaction_only_rhea_map.py \
    scripts/build_reaction_features.py \
    scripts/build_reaction_directional_vectors.py \
    scripts/extract_reaction_t5v2_embeddings.py \
    scripts/extract_unimol2_reaction_embeddings.py \
    scripts/extract_chiro_reaction_embeddings.py \
    horizyn/capability/reaction_features.py \
    horizyn/capability/reaction_set_features.py \
    horizyn/capability/reaction_directional_features.py \
    "$PROTOCOL_ROOT/$split/train_rxns.csv" \
    "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
    "$PROTOCOL_ROOT/$split/test_rxns.csv" \
    "$RHEA_MOLECULES" "$COFACTOR_DICTIONARY" | sha256sum | awk '{print $1}'
}

setup_split() {
  local split="$1"
  local gpu="$2"
  local split_root="$RUN_ROOT/data/$split"
  local set_dir="$split_root/reaction_set"
  local directional_dir="$split_root/reaction_directional"
  local source_dir="$directional_dir/source"
  local marker="$split_root/reaction_feature_setup.sha256"
  local expected_hash
  expected_hash="$(setup_fingerprint "$split")"
  local required=(
    "$set_dir/train_reaction_set_features.npz"
    "$set_dir/validation_reaction_set_features.npz"
    "$set_dir/test_reaction_set_features.npz"
    "$directional_dir/train_reaction_directional_f5.npz"
    "$directional_dir/validation_reaction_directional_f5.npz"
    "$directional_dir/test_reaction_directional_f5.npz"
    "$directional_dir/train_reaction_directional_f6.npz"
    "$directional_dir/validation_reaction_directional_f6.npz"
    "$directional_dir/test_reaction_directional_f6.npz"
    "$directional_dir/schema.json"
  )
  local complete=1
  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || complete=0
  done
  if [[ "$FORCE_SETUP" -eq 0 && "$complete" -eq 1 && -s "$marker" ]] \
      && [[ "$(tr -d '\r\n' < "$marker")" == "$expected_hash" ]]; then
    echo "[$split] setup artifacts match source fingerprint"
    status "$split" reaction_feature_setup skip 0
    return 0
  fi

  status "$split" reaction_feature_setup start ""
  mkdir -p "$set_dir" "$source_dir"
  "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
    --train-reactions "$PROTOCOL_ROOT/$split/train_rxns.csv" \
    --validation-reactions "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
    --test-reactions "$PROTOCOL_ROOT/$split/test_rxns.csv" \
    --cofactor-dictionary "$COFACTOR_DICTIONARY" \
    --out-dir "$set_dir" \
    > "$RUN_ROOT/logs/setup/${split}_reaction_set.log" 2>&1

  local subset subset_dir reactions mapping directional report center_dir
  for subset in train validation test; do
    subset_dir="$source_dir/$subset"
    reactions="$PROTOCOL_ROOT/$split/${subset}_rxns.csv"
    mapping="$subset_dir/rhea_mapping.csv"
    directional="$subset_dir/rhea_directional_reactions.csv"
    report="$subset_dir/rhea_mapping_report.json"
    center_dir="$subset_dir/reaction_centers"
    mkdir -p "$subset_dir" "$center_dir"

    "$SETUP_PYTHON_BIN" scripts/build_reaction_only_rhea_map.py \
      --reactions "$reactions" --rhea-molecules "$RHEA_MOLECULES" \
      --out-mapping "$mapping" --out-directional-reactions "$directional" \
      --out-report "$report" \
      > "$RUN_ROOT/logs/setup/${split}_${subset}_rhea_map.log" 2>&1

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions "$directional" --output "$subset_dir/reactiont5v2.h5" \
      --model-name "$REACTION_T5_MODEL_PATH" --batch-size 64 --max-length 512 \
      --pooling mean --device cuda --dtype float16 --bidirectional \
      --no-allow-pseudo-reactions --force \
      > "$RUN_ROOT/logs/setup/${split}_${subset}_reactiont5.log" 2>&1

    PYTHONPATH="$ROOT/.deps/unimol_tools:$ROOT/../env/unimol2_site:$ROOT" \
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions "$directional" --output "$subset_dir/unimol2.h5" \
      --batch-size 64 --dtype float16 --compression none --bidirectional \
      --no-allow-pseudo-reactions --skip-invalid-molecules --skip-invalid-reactions --force \
      > "$RUN_ROOT/logs/setup/${split}_${subset}_unimol2.log" 2>&1

    PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions "$directional" --output "$subset_dir/chiro.h5" \
      --device cuda --num-workers 4 --batch-size 64 --bidirectional \
      --no-allow-pseudo-reactions --force \
      > "$RUN_ROOT/logs/setup/${split}_${subset}_chiro.log" 2>&1

    PYTHONPATH="$ROOT:$RUN_ROOT/deps/rxnmapper" \
      CUDA_VISIBLE_DEVICES="$gpu" "$SETUP_PYTHON_BIN" scripts/build_reaction_features.py \
      --reaction-smiles "$directional" --cofactor-dictionary "$COFACTOR_DICTIONARY" \
      --out-dir "$center_dir" --use-rxnmapper true --rxnmapper-batch-size 64 \
      > "$RUN_ROOT/logs/setup/${split}_${subset}_rxnmapper.log" 2>&1
  done

  "$SETUP_PYTHON_BIN" scripts/build_reaction_directional_vectors.py \
    --train-reactions "$PROTOCOL_ROOT/$split/train_rxns.csv" \
    --train-mapping "$source_dir/train/rhea_mapping.csv" \
    --train-reaction-t5 "$source_dir/train/reactiont5v2.h5" \
    --train-unimol2 "$source_dir/train/unimol2.h5" \
    --train-chiro "$source_dir/train/chiro.h5" \
    --train-center-features "$source_dir/train/reaction_centers/reaction_features.parquet" \
    --validation-reactions "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
    --validation-mapping "$source_dir/validation/rhea_mapping.csv" \
    --validation-reaction-t5 "$source_dir/validation/reactiont5v2.h5" \
    --validation-unimol2 "$source_dir/validation/unimol2.h5" \
    --validation-chiro "$source_dir/validation/chiro.h5" \
    --validation-center-features "$source_dir/validation/reaction_centers/reaction_features.parquet" \
    --test-reactions "$PROTOCOL_ROOT/$split/test_rxns.csv" \
    --test-mapping "$source_dir/test/rhea_mapping.csv" \
    --test-reaction-t5 "$source_dir/test/reactiont5v2.h5" \
    --test-unimol2 "$source_dir/test/unimol2.h5" \
    --test-chiro "$source_dir/test/chiro.h5" \
    --test-center-features "$source_dir/test/reaction_centers/reaction_features.parquet" \
    --out-dir "$directional_dir" \
    > "$RUN_ROOT/logs/setup/${split}_directional_vectors.log" 2>&1
  printf '%s\n' "$expected_hash" > "$marker"
  status "$split" reaction_feature_setup end 0
}

run_setup() {
  [[ "$SKIP_SETUP" -eq 0 ]] || return 0
  wait_for_unrelated_training
  ensure_rxnmapper
  local pids=()
  setup_split time 0 & pids+=("$!")
  setup_split enzyme_smi 1 & pids+=("$!")
  setup_split reaction_smi 2 & pids+=("$!")
  local failed=0 pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" -ne 0 ]]; then
    status controller reaction_feature_setup end 1
    echo "Reaction-feature setup failed; inspect $RUN_ROOT/logs/setup" >&2
    exit 1
  fi
}

echo "Run root: $RUN_ROOT"
echo "W&B: ${WANDB_ENTITY:-}/$WANDB_PROJECT mode=$WANDB_MODE"
echo "Schedule: seven waves; time/enzyme_smi/reaction_smi concurrent; four GPUs per run"
status controller reaction_feature_controller start ""
run_setup

generic_args=("$RUN_ROOT" --foreground --skip-setup)
[[ "$SKIP_TRAIN" -eq 1 ]] && generic_args+=(--skip-train)
[[ -n "$ONLY_VARIANT" ]] && generic_args+=(--variant "$ONLY_VARIANT")
exec "$ROOT/scripts/run_reactzyme_paper_ablation_matrix.sh" "${generic_args[@]}"
