#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_RUN_ROOT="$ROOT/runs/reactzyme_f3_loss_ablation_v1"

RUN_ROOT_ARG="$DEFAULT_RUN_ROOT"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  RUN_ROOT_ARG="$1"
  shift
fi
case "$RUN_ROOT_ARG" in
  /*) RUN_ROOT="$RUN_ROOT_ARG" ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT_ARG" ;;
esac

PHASE="all"
ONLY_VARIANT=""
ONLY_SPLIT=""
ONLY_SEED=""
DETACH=0
FORCE_SETUP=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --variant) ONLY_VARIANT="${2:?missing variant}"; shift 2 ;;
    --split) ONLY_SPLIT="${2:?missing split}"; shift 2 ;;
    --seed) ONLY_SEED="${2:?missing seed}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --force-setup) FORCE_SETUP=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in setup|screen|replicate|freeze|test|all) ;; *) echo "Invalid phase: $PHASE" >&2; exit 2 ;; esac
[[ -z "$ONLY_VARIANT" || "$ONLY_VARIANT" =~ ^L[0-7]$ ]] || { echo "Invalid variant: $ONLY_VARIANT" >&2; exit 2; }
[[ -z "$ONLY_SPLIT" || "$ONLY_SPLIT" =~ ^(time|enzyme_smi|reaction_smi)$ ]] || { echo "Invalid split: $ONLY_SPLIT" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs/controller"
if [[ "$DETACH" -eq 1 ]]; then
  args=("$RUN_ROOT" --phase "$PHASE" --foreground)
  [[ -n "$ONLY_VARIANT" ]] && args+=(--variant "$ONLY_VARIANT")
  [[ -n "$ONLY_SPLIT" ]] && args+=(--split "$ONLY_SPLIT")
  [[ -n "$ONLY_SEED" ]] && args+=(--seed "$ONLY_SEED")
  [[ "$FORCE_SETUP" -eq 1 ]] && args+=(--force-setup)
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  pid_file="$RUN_ROOT/logs/controller/pid"
  : > "$pid_file"
  setsid -f bash -c 'printf "%s\n" "$$" > "$1"; shift; exec "$@"' bash \
    "$pid_file" "$0" "${args[@]}" \
    > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  for _ in {1..50}; do [[ -s "$pid_file" ]] && break; sleep 0.1; done
  echo "Launched F3 loss campaign controller"
  echo "PID: $(cat "$pid_file" 2>/dev/null || printf unknown)"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  echo "Status: $RUN_ROOT/logs/status.jsonl"
  exit 0
fi

BOOTSTRAP_PYTHON="${SETUP_PYTHON_BIN:-$ROOT/../.capability-run-py/bin/python}"
if [[ ! -s "$RUN_ROOT/campaign.tsv" ]]; then
  "$BOOTSTRAP_PYTHON" "$ROOT/scripts/create_reactzyme_f3_loss_campaign.py" --run-root "$RUN_ROOT"
fi
# shellcheck source=/dev/null
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

HOST="$(hostname -s)"
SHORT_TMPDIR="${F3_TMPDIR:-/tmp/horizyn-f3-${UID}-${HOST}}"
mkdir -p \
  "$RUN_ROOT/nohome/$HOST" "$RUN_ROOT/cache/$HOST/huggingface" \
  "$RUN_ROOT/cache/$HOST/torch" "$RUN_ROOT/cache/$HOST/pip" \
  "$RUN_ROOT/cache/$HOST/xdg" "$RUN_ROOT/cache/$HOST/matplotlib" \
  "$RUN_ROOT/wandb/$HOST" "$RUN_ROOT/logs/setup" "$RUN_ROOT/locks" \
  "$RUN_ROOT/tmp/$HOST" "$RUN_ROOT/results" "$RUN_ROOT/frozen" "$SHORT_TMPDIR"
chmod 700 "$SHORT_TMPDIR"
export HOME="$RUN_ROOT/nohome/$HOST"
export TMPDIR="$SHORT_TMPDIR"
export HF_HOME="$RUN_ROOT/cache/$HOST/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/$HOST/torch"
export PIP_CACHE_DIR="$RUN_ROOT/cache/$HOST/pip"
export XDG_CACHE_HOME="$RUN_ROOT/cache/$HOST/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/$HOST/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb/$HOST"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export UNIMOL_WEIGHT_DIR="$ROOT/../unimol_weights"
MIN_FEATURE_COVERAGE="${MIN_FEATURE_COVERAGE:-0.95}"
CHIRO_NUM_WORKERS="${CHIRO_NUM_WORKERS:-8}"
CHIRO_MAX_PENDING_TASKS="${CHIRO_MAX_PENDING_TASKS:-$((CHIRO_NUM_WORKERS * 2))}"
CHIRO_MOLECULE_CACHE="$RUN_ROOT/data/features/chiro_molecules.sqlite3"
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$RUN_ROOT/wandb/api_key")"
fi

STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
status() {
  printf '{"time":"%s","host":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$HOST" "$1" "$2" "$3" "${4:-}" >> "$STATUS_LOG"
}

acquire_four_gpu_lock() {
  exec 9>"$RUN_ROOT/locks/$HOST.four_gpu.lock"
  if ! flock -n 9; then
    echo "Another F3 campaign process owns the four-GPU lock on $HOST" >&2
    exit 1
  fi
}

normalize_split() {
  local split="$1" subset source output log
  for subset in train validation test; do
    source="$ROOT/data/revised_protocols/reactzyme_paper/$split/${subset}_rxns.csv"
    output="$RUN_ROOT/data/isomeric/$split/${subset}_rxns.csv"
    log="$RUN_ROOT/logs/setup/${split}_${subset}_isomeric.log"
    mkdir -p "$(dirname "$output")"
    "$SETUP_PYTHON_BIN" scripts/materialize_isomeric_reactions.py \
      --input "$source" --output "$output" > "$log" 2>&1
  done
}

feature_is_valid() {
  local artifact="$1" source="$2"
  [[ "$FORCE_SETUP" -eq 0 && -s "$artifact" ]] || return 1
  "$SETUP_PYTHON_BIN" scripts/stamp_reaction_feature_provenance.py \
    --artifact "$artifact" --source-csv "$source" --validate \
    --minimum-coverage "$MIN_FEATURE_COVERAGE" \
    >/dev/null 2>&1
}

stamp_feature() {
  local artifact="$1" source="$2" extractor="$3" version="$4"
  "$SETUP_PYTHON_BIN" scripts/stamp_reaction_feature_provenance.py \
    --artifact "$artifact" --source-csv "$source" \
    --extractor-name "$extractor" --extractor-version "$version" >/dev/null
  "$SETUP_PYTHON_BIN" scripts/stamp_reaction_feature_provenance.py \
    --artifact "$artifact" --source-csv "$source" --validate \
    --minimum-coverage "$MIN_FEATURE_COVERAGE" >/dev/null
}

extract_non_chiro_subset() {
  local split="$1" subset="$2" gpu="$3"
  local source="$RUN_ROOT/data/isomeric/$split/${subset}_rxns.csv"
  local out_dir="$RUN_ROOT/data/features/$split/$subset"
  local artifact log
  mkdir -p "$out_dir"

  artifact="$out_dir/reactiont5v2.h5"
  log="$RUN_ROOT/logs/setup/${split}_${subset}_reactiont5v2.log"
  if ! feature_is_valid "$artifact" "$source"; then
    rm -f "$artifact"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_reaction_t5v2_embeddings.py \
      --reactions "$source" --output "$artifact" --model-name "$REACTION_T5_MODEL_PATH" \
      --batch-size 64 --max-length 512 --pooling mean --device cuda --dtype float16 \
      --bidirectional --allow-pseudo-reactions --no-standardize --force > "$log" 2>&1
    stamp_feature "$artifact" "$source" reactiont5v2 "sagawa-forward-9331140"
  fi

  artifact="$out_dir/unimol2.h5"
  log="$RUN_ROOT/logs/setup/${split}_${subset}_unimol2.log"
  if ! feature_is_valid "$artifact" "$source"; then
    rm -f "$artifact"
    PYTHONPATH="$ROOT/.deps/unimol_tools:$ROOT/../env/unimol2_site:$ROOT" \
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
      --reactions "$source" --output "$artifact" --batch-size 64 --dtype float16 \
      --compression none --bidirectional --allow-pseudo-reactions --no-standardize \
      --skip-invalid-molecules --skip-invalid-reactions --force > "$log" 2>&1
    stamp_feature "$artifact" "$source" unimol2 "84m"
  fi

}

extract_chiro_subset() {
  local split="$1" subset="$2"
  local source="$RUN_ROOT/data/isomeric/$split/${subset}_rxns.csv"
  local out_dir="$RUN_ROOT/data/features/$split/$subset"
  local artifact="$out_dir/chiro.h5"
  local log="$RUN_ROOT/logs/setup/${split}_${subset}_chiro.log"
  mkdir -p "$out_dir"
  log="$RUN_ROOT/logs/setup/${split}_${subset}_chiro.log"
  if ! feature_is_valid "$artifact" "$source"; then
    rm -f "$artifact"
    PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
      CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions "$source" --output "$artifact" --device cpu --num-workers 0 \
      --molecule-cache "$CHIRO_MOLECULE_CACHE" --cache-read-only \
      --batch-size 64 --bidirectional --allow-pseudo-reactions --no-standardize \
      --skip-invalid-molecules --skip-invalid-reactions --force > "$log" 2>&1
    stamp_feature "$artifact" "$source" chiro "keiradams-chiro-cached-v1"
  fi
}

extract_non_chiro_split() {
  local split="$1" gpu="$2" subset
  for subset in train validation test; do
    extract_non_chiro_subset "$split" "$subset" "$gpu"
  done
}

precompute_chiro_cache() {
  local splits=("$@") split subset
  local sources=()
  for split in "${splits[@]}"; do
    for subset in train validation test; do
      sources+=("$RUN_ROOT/data/isomeric/$split/${subset}_rxns.csv")
    done
  done
  if [[ "$FORCE_SETUP" -eq 1 ]]; then
    rm -f "$CHIRO_MOLECULE_CACHE" "${CHIRO_MOLECULE_CACHE}-journal" \
      "${CHIRO_MOLECULE_CACHE}-shm" "${CHIRO_MOLECULE_CACHE}-wal"
  fi
  status chiro_molecule_cache setup start ""
  if OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONPATH="$ROOT/.deps/python:$ROOT/.deps/ChIRo:$ROOT" \
    CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
      --reactions "${sources[@]}" --molecule-cache "$CHIRO_MOLECULE_CACHE" --cache-only \
      --device cuda --num-workers "$CHIRO_NUM_WORKERS" \
      --max-pending-tasks "$CHIRO_MAX_PENDING_TASKS" --batch-size 64 \
      --no-bidirectional --allow-pseudo-reactions --no-standardize \
      --skip-invalid-molecules --skip-invalid-reactions \
      > "$RUN_ROOT/logs/setup/chiro_molecule_cache.log" 2>&1
  then
    status chiro_molecule_cache setup end 0
  else
    local rc=$?
    status chiro_molecule_cache setup end "$rc"
    return "$rc"
  fi
}

build_reaction_set_for_split() {
  local split="$1"
  local set_dir="$RUN_ROOT/data/features/$split/reaction_set"
  local schema="$set_dir/schema.json"
  local set_valid=0
  if [[ "$FORCE_SETUP" -eq 0 && -s "$schema" ]]; then
    if "$SETUP_PYTHON_BIN" - "$schema" \
      "$RUN_ROOT/data/isomeric/$split/train_rxns.csv" \
      "$RUN_ROOT/data/isomeric/$split/validation_rxns.csv" \
      "$RUN_ROOT/data/isomeric/$split/test_rxns.csv" <<'PY' >/dev/null 2>&1
import hashlib, json, sys
p = json.load(open(sys.argv[1]))
assert p.get("smiles_mode") == "canonical_isomeric"
assert p.get("normalizer_version") == "horizyn_canonical_isomeric_v1"
expected = {
    name: hashlib.sha256(open(path, "rb").read()).hexdigest()
    for name, path in zip(("train", "validation", "test"), sys.argv[2:])
}
assert p.get("source_csv_sha256") == expected
PY
    then set_valid=1; fi
  fi
  if [[ "$set_valid" -eq 0 ]]; then
    rm -rf "$set_dir"
    mkdir -p "$set_dir"
    "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
      --train-reactions "$RUN_ROOT/data/isomeric/$split/train_rxns.csv" \
      --validation-reactions "$RUN_ROOT/data/isomeric/$split/validation_rxns.csv" \
      --test-reactions "$RUN_ROOT/data/isomeric/$split/test_rxns.csv" \
      --cofactor-dictionary "$ROOT/data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv" \
      --out-dir "$set_dir" > "$RUN_ROOT/logs/setup/${split}_reaction_set.log" 2>&1
  fi
}

finalize_split() {
  local split="$1" subset
  for subset in train validation test; do
    extract_chiro_subset "$split" "$subset"
  done
  build_reaction_set_for_split "$split"
  status "$split" setup end 0
}

gpu_for_split() {
  case "$1" in
    time) printf '0\n' ;;
    enzyme_smi) printf '1\n' ;;
    reaction_smi) printf '2\n' ;;
  esac
}

run_setup() {
  acquire_four_gpu_lock
  local splits=(time enzyme_smi reaction_smi)
  local pids=() failed=0 pid split
  if [[ -n "$ONLY_SPLIT" ]]; then splits=("$ONLY_SPLIT"); fi

  for split in "${splits[@]}"; do
    status "$split" setup start ""
    normalize_split "$split" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  [[ "$failed" -eq 0 ]] || { echo "Isomeric normalization failed; inspect $RUN_ROOT/logs/setup" >&2; exit 1; }

  pids=(); failed=0
  precompute_chiro_cache "${splits[@]}" & pids+=("$!")
  for split in "${splits[@]}"; do
    extract_non_chiro_split "$split" "$(gpu_for_split "$split")" & pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  [[ "$failed" -eq 0 ]] || { echo "Reaction feature extraction failed; inspect $RUN_ROOT/logs/setup" >&2; exit 1; }

  pids=(); failed=0
  for split in "${splits[@]}"; do finalize_split "$split" & pids+=("$!"); done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
  [[ "$failed" -eq 0 ]] || { echo "Feature materialization failed; inspect $RUN_ROOT/logs/setup" >&2; exit 1; }
}

select_checkpoint() {
  local checkpoint_dir="$1" log="$2" best=""
  [[ -s "$log" ]] && best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {v=$2} END {print v}' "$log" | tr -d '\r')"
  if [[ -n "$best" && -s "$best" ]]; then printf '%s\n' "$best"
  elif [[ -s "$checkpoint_dir/last.ckpt" ]]; then printf '%s\n' "$checkpoint_dir/last.ckpt"
  else find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,"");print}'; fi
}

run_train() {
  local variant="$1" split="$2" seed="$3"
  local run_id="${variant}_${split}_seed${seed}"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/train.yaml"
  local checkpoint_dir="$RUN_ROOT/checkpoints/$run_id"
  local log="$RUN_ROOT/logs/$run_id/train.log"
  local marker="$RUN_ROOT/logs/$run_id/complete.json"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$marker" ]] && [[ -n "$(select_checkpoint "$checkpoint_dir" "$log")" ]]; then
    status "$run_id" train skip 0
    return
  fi
  local command=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
  [[ -s "$checkpoint_dir/last.ckpt" ]] && command+=(--resume "$checkpoint_dir/last.ckpt")
  if [[ "$WANDB_MODE" != disabled ]]; then
    command+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "reactzyme-f3-loss-$run_id" --wandb-mode "$WANDB_MODE")
    [[ -n "${WANDB_ENTITY:-}" ]] && command+=(--wandb-entity "$WANDB_ENTITY")
  fi
  status "$run_id" train start ""
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$((BASE_MASTER_PORT + seed % 100))" \
    "${command[@]}" > "$log" 2>&1
  local rc=$?
  set -e
  status "$run_id" train end "$rc"
  [[ "$rc" -eq 0 ]] || return "$rc"
  local checkpoint
  checkpoint="$(select_checkpoint "$checkpoint_dir" "$log")"
  [[ -s "$checkpoint" ]] || { echo "No checkpoint produced for $run_id" >&2; return 1; }
  printf '{"checkpoint":"%s","sha256":"%s"}\n' "$checkpoint" "$(sha256sum "$checkpoint" | awk '{print $1}')" > "$marker"
}

run_training_phase() {
  local phase="$1" variants seeds variant split seed
  acquire_four_gpu_lock
  if [[ "$phase" == screen ]]; then
    variants="L0 L1 L3 L4 L2 L5 L6 L7"
    seeds="42"
  else
    if [[ -n "$ONLY_VARIANT" ]]; then variants="$ONLY_VARIANT"
    else
      [[ -s "$RUN_ROOT/reports/selected_variants.txt" ]] || "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_loss_campaign select --run-root "$RUN_ROOT"
      variants="$(tr '\n' ' ' < "$RUN_ROOT/reports/selected_variants.txt")"
    fi
    seeds="17 73"
  fi
  [[ -n "$ONLY_VARIANT" ]] && variants="$ONLY_VARIANT"
  [[ -n "$ONLY_SEED" ]] && seeds="$ONLY_SEED"
  for variant in $variants; do
    for seed in $seeds; do
      for split in time enzyme_smi reaction_smi; do
        [[ -z "$ONLY_SPLIT" || "$split" == "$ONLY_SPLIT" ]] || continue
        run_train "$variant" "$split" "$seed"
      done
    done
  done
  if [[ "$phase" == screen && -z "$ONLY_VARIANT" && -z "$ONLY_SPLIT" ]]; then
    "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_loss_campaign select --run-root "$RUN_ROOT"
  fi
}

selected_variants() {
  if [[ -n "$ONLY_VARIANT" ]]; then printf '%s\n' "$ONLY_VARIANT"
  else
    [[ -s "$RUN_ROOT/reports/selected_variants.txt" ]] || "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_loss_campaign select --run-root "$RUN_ROOT" >/dev/null
    cat "$RUN_ROOT/reports/selected_variants.txt"
  fi
}

run_freeze() {
  local variant seed split run_id source destination manifest
  while read -r variant; do
    for seed in 42 17 73; do
      [[ -z "$ONLY_SEED" || "$seed" == "$ONLY_SEED" ]] || continue
      for split in time enzyme_smi reaction_smi; do
        [[ -z "$ONLY_SPLIT" || "$split" == "$ONLY_SPLIT" ]] || continue
        run_id="${variant}_${split}_seed${seed}"
        source="$(select_checkpoint "$RUN_ROOT/checkpoints/$run_id" "$RUN_ROOT/logs/$run_id/train.log")"
        [[ -s "$source" ]] || { echo "Cannot freeze missing checkpoint for $run_id" >&2; exit 1; }
        destination="$RUN_ROOT/frozen/$variant/seed$seed/$split/best.ckpt"
        manifest="$destination.json"
        mkdir -p "$(dirname "$destination")"
        if [[ ! -s "$destination" ]]; then cp --reflink=auto "$source" "$destination"; fi
        printf '{"source":"%s","frozen":"%s","sha256":"%s"}\n' \
          "$source" "$destination" "$(sha256sum "$destination" | awk '{print $1}')" > "$manifest"
      done
    done
  done < <(selected_variants)
}

run_eval_one() {
  local variant="$1" seed="$2" split="$3" gpu="$4"
  local checkpoint="$RUN_ROOT/frozen/$variant/seed$seed/$split/best.ckpt"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/test.yaml"
  local output="$RUN_ROOT/results/$variant/seed$seed/$split.json"
  local log="$RUN_ROOT/logs/${variant}_${split}_seed${seed}/test.log"
  [[ -s "$output" ]] && { status "${variant}_${split}_seed${seed}" test skip 0; return; }
  mkdir -p "$(dirname "$output")" "$(dirname "$log")"
  status "${variant}_${split}_seed${seed}" test start ""
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
    --checkpoint "$checkpoint" --config "$config" --device cuda \
    --direction both --evaluation-protocol paper_test_candidates \
    --batch-size 128 --target-batch-size 512 --output "$output" > "$log" 2>&1
  status "${variant}_${split}_seed${seed}" test end 0
}

run_test() {
  acquire_four_gpu_lock
  local variant seed pids failed pid
  while read -r variant; do
    for seed in 42 17 73; do
      [[ -z "$ONLY_SEED" || "$seed" == "$ONLY_SEED" ]] || continue
      pids=(); failed=0
      if [[ -z "$ONLY_SPLIT" || "$ONLY_SPLIT" == time ]]; then run_eval_one "$variant" "$seed" time 0 & pids+=("$!"); fi
      if [[ -z "$ONLY_SPLIT" || "$ONLY_SPLIT" == enzyme_smi ]]; then run_eval_one "$variant" "$seed" enzyme_smi 1 & pids+=("$!"); fi
      if [[ -z "$ONLY_SPLIT" || "$ONLY_SPLIT" == reaction_smi ]]; then run_eval_one "$variant" "$seed" reaction_smi 2 & pids+=("$!"); fi
      for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
      [[ "$failed" -eq 0 ]] || { echo "Test evaluation failed for $variant seed$seed" >&2; exit 1; }
    done
  done < <(selected_variants)
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_loss_campaign report --run-root "$RUN_ROOT"
}

echo "Run root: $RUN_ROOT"
echo "Phase: $PHASE; W&B: ${WANDB_ENTITY:-}/$WANDB_PROJECT ($WANDB_MODE)"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only; no setup, training, freezing, or evaluation will execute."
  if [[ "$PHASE" == screen || "$PHASE" == all ]]; then
    for variant in L0 L1 L3 L4 L2 L5 L6 L7; do
      [[ -z "$ONLY_VARIANT" || "$variant" == "$ONLY_VARIANT" ]] || continue
      for split in time enzyme_smi reaction_smi; do
        [[ -z "$ONLY_SPLIT" || "$split" == "$ONLY_SPLIT" ]] || continue
        echo "screen $variant $split seed${ONLY_SEED:-42} GPUs=$GPUS"
      done
    done
  fi
  if [[ "$PHASE" == replicate || "$PHASE" == all ]]; then
    echo "replicate: top two validation variants; seeds ${ONLY_SEED:-17,73}; splits ${ONLY_SPLIT:-time,enzyme_smi,reaction_smi}"
  fi
  if [[ "$PHASE" == freeze || "$PHASE" == test || "$PHASE" == all ]]; then
    echo "$PHASE: top two validation variants; seeds ${ONLY_SEED:-42,17,73}; splits ${ONLY_SPLIT:-time,enzyme_smi,reaction_smi}"
  fi
  exit 0
fi
status controller "$PHASE" start ""
case "$PHASE" in
  setup) run_setup ;;
  screen) run_training_phase screen ;;
  replicate) run_training_phase replicate ;;
  freeze) run_freeze ;;
  test) run_test ;;
  all)
    run_setup
    run_training_phase screen
    run_training_phase replicate
    run_freeze
    run_test
    ;;
esac
status controller "$PHASE" end 0
