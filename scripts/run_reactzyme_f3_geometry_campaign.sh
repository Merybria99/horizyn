#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT_ARG="${1:-$ROOT/runs/reactzyme_f3_geometry_v1}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then shift; fi
case "$RUN_ROOT_ARG" in
  /*) RUN_ROOT="$RUN_ROOT_ARG" ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT_ARG" ;;
esac

PHASE=screen
DETACH=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in
  setup|screen|replicate|freeze|test|all) ;;
  *) echo "Invalid phase: $PHASE" >&2; exit 2 ;;
esac

mkdir -p "$RUN_ROOT/logs/controller"
if [[ "$DETACH" -eq 1 ]]; then
  nohup setsid "$0" "$RUN_ROOT" --phase "$PHASE" --foreground \
    > "$RUN_ROOT/logs/controller/nohup.log" 2>&1 < /dev/null &
  controller_pid=$!
  printf '%s\n' "$controller_pid" > "$RUN_ROOT/logs/controller/pid"
  echo "Launched F3 geometry campaign controller PID $controller_pid"
  echo "Log: $RUN_ROOT/logs/controller/nohup.log"
  exit 0
fi

PROTOCOL_ROOT="${PROTOCOL_ROOT:-$ROOT/data/revised_protocols/reactzyme_unseen_reaction_v1}"
BOOTSTRAP_PYTHON="${SETUP_PYTHON_BIN:-$ROOT/../.capability-run-py/bin/python}"
if [[ ! -s "$PROTOCOL_ROOT/manifest.json" ]]; then
  "$BOOTSTRAP_PYTHON" "$ROOT/scripts/build_reaction_disjoint_validation.py" \
    --out-root "$PROTOCOL_ROOT"
fi
if [[ ! -s "$RUN_ROOT/campaign.tsv" ]]; then
  "$BOOTSTRAP_PYTHON" "$ROOT/scripts/create_reactzyme_f3_geometry_campaign.py" generate \
    --run-root "$RUN_ROOT" --protocol-root "$PROTOCOL_ROOT"
fi
# shellcheck source=/dev/null
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

HOST="$(hostname -s)"
SHORT_TMPDIR="${F3_GEOMETRY_TMPDIR:-/tmp/hfg-${UID}-${HOST}}"
mkdir -p \
  "$RUN_ROOT/nohome/$HOST" "$SHORT_TMPDIR" \
  "$RUN_ROOT/cache/$HOST/huggingface" "$RUN_ROOT/cache/$HOST/torch" \
  "$RUN_ROOT/cache/$HOST/xdg" "$RUN_ROOT/cache/$HOST/matplotlib" \
  "$RUN_ROOT/wandb/$HOST" "$RUN_ROOT/logs/setup" \
  "$RUN_ROOT/logs/controller" "$RUN_ROOT/results" "$RUN_ROOT/reports" \
  "$RUN_ROOT/locks"
chmod 700 "$SHORT_TMPDIR"
export HOME="$RUN_ROOT/nohome/$HOST"
export TMPDIR="$SHORT_TMPDIR"
export HF_HOME="$RUN_ROOT/cache/$HOST/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/$HOST/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/$HOST/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/$HOST/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb/$HOST"
export WANDB_CACHE_DIR="$WANDB_DIR/cache"
export WANDB_CONFIG_DIR="$WANDB_DIR/config"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' < "$RUN_ROOT/wandb/api_key")"
fi

STATUS_LOG="$RUN_ROOT/logs/status.jsonl"
status() {
  printf '{"time":"%s","host":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s","gpu":"%s"}\n' \
    "$(date -Iseconds)" "$HOST" "$1" "$2" "$3" "${4:-}" "${5:-}" >> "$STATUS_LOG"
}

acquire_lock() {
  exec 9>"$RUN_ROOT/locks/$HOST.controller.lock"
  flock -n 9 || { echo "Another F3 geometry controller owns the host lock" >&2; exit 1; }
}

setup_chemistry() {
  local split="$1" out="$RUN_ROOT/data/$1/reaction_set"
  local schema="$out/schema.json" valid=0
  if [[ -s "$out/train_reaction_set_features.npz" \
      && -s "$out/validation_reaction_set_features.npz" \
      && -s "$out/test_reaction_set_features.npz" \
      && -s "$schema" ]]; then
    if "$SETUP_PYTHON_BIN" - "$schema" \
      "$PROTOCOL_ROOT/$split/train_rxns.csv" \
      "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
      "$PROTOCOL_ROOT/$split/test_rxns.csv" <<'PY' >/dev/null 2>&1
import hashlib
import json
import sys

schema = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    name: hashlib.sha256(open(path, "rb").read()).hexdigest()
    for name, path in zip(("train", "validation", "test"), sys.argv[2:])
}
assert schema.get("fit_split") == "train"
assert schema.get("source_csv_sha256") == expected
PY
    then valid=1; fi
  fi
  if [[ "$valid" -eq 1 ]]; then
    status "$split" setup skip 0 ""
    return
  fi
  rm -rf "$out"
  mkdir -p "$out"
  status "$split" setup start "" ""
  "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
    --train-reactions "$PROTOCOL_ROOT/$split/train_rxns.csv" \
    --validation-reactions "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
    --test-reactions "$PROTOCOL_ROOT/$split/test_rxns.csv" \
    --cofactor-dictionary "$ROOT/data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv" \
    --out-dir "$out" > "$RUN_ROOT/logs/setup/${split}_reaction_set.log" 2>&1
  status "$split" setup end 0 ""
}

run_setup() {
  local split
  for split in time enzyme_smi reaction_smi; do setup_chemistry "$split"; done
}

run_train() {
  local variant="$1" split="$2" seed="$3" gpu="$4"
  local run_id="${variant}_${split}_seed${seed}"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/train.yaml"
  local checkpoint_dir="$RUN_ROOT/checkpoints/$run_id"
  local log="$RUN_ROOT/logs/$run_id/train.log"
  local marker="$RUN_ROOT/logs/$run_id/train.complete"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$marker" ]]; then status "$run_id" train skip 0 "$gpu"; return; fi
  local command=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
  if [[ -s "$checkpoint_dir/last.ckpt" ]]; then command+=(--resume "$checkpoint_dir/last.ckpt"); fi
  if [[ "$WANDB_MODE" != disabled ]]; then
    command+=(--wandb --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "reactzyme-f3-geometry-$run_id" --wandb-mode "$WANDB_MODE")
    [[ -n "${WANDB_ENTITY:-}" ]] && command+=(--wandb-entity "$WANDB_ENTITY")
  fi
  status "$run_id" train start "" "$gpu"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "${command[@]}" > "$log" 2>&1
  local rc=$?
  set -e
  status "$run_id" train end "$rc" "$gpu"
  [[ "$rc" -eq 0 ]] || return "$rc"
  find "$checkpoint_dir" -maxdepth 1 -name 'protein-pooling-*.ckpt' -type f | grep -q . \
    || { echo "No monitored checkpoint produced for $run_id" >&2; return 1; }
  date -Iseconds > "$marker"
}

run_validation() {
  local variant="$1" split="$2" seed="$3" gpu="$4"
  local run_id="${variant}_${split}_seed${seed}"
  local checkpoint_dir="$RUN_ROOT/checkpoints/$run_id"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/validation.yaml"
  local output_dir="$RUN_ROOT/results/validation/$variant/$split/seed$seed"
  local checkpoint stem output log rc
  mkdir -p "$output_dir" "$RUN_ROOT/logs/$run_id/validation"
  while IFS= read -r checkpoint; do
    stem="$(basename "${checkpoint%.ckpt}")"
    output="$output_dir/$stem.json"
    log="$RUN_ROOT/logs/$run_id/validation/$stem.log"
    [[ -s "$output" ]] && continue
    status "$run_id" validation start "" "$gpu"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device cuda --direction both \
      --evaluation-protocol configured_forward_candidates --batch-size 128 \
      --target-batch-size 512 --output "$output" > "$log" 2>&1
    rc=$?
    set -e
    status "$run_id" validation end "$rc" "$gpu"
    [[ "$rc" -eq 0 ]] || return "$rc"
  done < <(find "$checkpoint_dir" -maxdepth 1 -name 'protein-pooling-*.ckpt' -type f | sort)
}

run_variant_seed() {
  local variant="$1" seed="$2" gpu="$3" split
  for split in time enzyme_smi reaction_smi; do
    run_train "$variant" "$split" "$seed" "$gpu"
    run_validation "$variant" "$split" "$seed" "$gpu"
  done
}

wait_for_workers() {
  local rc=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then rc=1; fi
  done
  return "$rc"
}

run_screen() {
  local variants=(M0 M1 M2 M3) gpus pids=() index
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  [[ "${#gpus[@]}" -ge 4 ]] || { echo "Screen requires four GPU IDs" >&2; return 1; }
  for index in 0 1 2 3; do
    run_variant_seed "${variants[$index]}" 42 "${gpus[$index]}" &
    pids+=("$!")
  done
  wait_for_workers "${pids[@]}"
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_geometry_campaign \
    select-screen --run-root "$RUN_ROOT" > "$RUN_ROOT/reports/screen_selection.stdout.json"
}

run_replicate() {
  [[ -s "$RUN_ROOT/reports/selected_variants.txt" ]] || \
    "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_geometry_campaign \
      select-screen --run-root "$RUN_ROOT" >/dev/null
  mapfile -t variants < "$RUN_ROOT/reports/selected_variants.txt"
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  local pids=() index=0 variant seed
  for variant in "${variants[@]}"; do
    for seed in 17 73; do
      run_variant_seed "$variant" "$seed" "${gpus[$index]}" &
      pids+=("$!")
      index=$((index + 1))
    done
  done
  wait_for_workers "${pids[@]}"
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_geometry_campaign \
    select-final --run-root "$RUN_ROOT" > "$RUN_ROOT/reports/final_selection.stdout.json"
}

run_freeze() {
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_geometry_campaign \
    freeze --run-root "$RUN_ROOT" > "$RUN_ROOT/reports/freeze.stdout.json"
}

run_test_seed() {
  local winner="$1" seed="$2" gpu="$3" split checkpoint config output log rc
  for split in time enzyme_smi reaction_smi; do
    checkpoint="$RUN_ROOT/frozen/$winner/seed$seed/$split/best.ckpt"
    config="$RUN_ROOT/configs/$winner/$split/seed$seed/test.yaml"
    output="$RUN_ROOT/results/test/$winner/seed$seed/$split.json"
    log="$RUN_ROOT/logs/${winner}_${split}_seed${seed}/test.log"
    [[ -s "$output" ]] && continue
    mkdir -p "$(dirname "$output")" "$(dirname "$log")"
    status "${winner}_${split}_seed${seed}" test start "" "$gpu"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device cuda --direction both \
      --evaluation-protocol paper_test_candidates --batch-size 128 \
      --target-batch-size 512 --output "$output" > "$log" 2>&1
    rc=$?
    set -e
    status "${winner}_${split}_seed${seed}" test end "$rc" "$gpu"
    [[ "$rc" -eq 0 ]] || return "$rc"
  done
}

run_test() {
  local manifest="$RUN_ROOT/frozen/manifest.json"
  [[ -s "$manifest" ]] || { echo "Refusing test access before winner freeze" >&2; return 1; }
  local winner gpus pids=()
  winner="$($SETUP_PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$manifest")"
  IFS=',' read -r -a gpus <<< "$GPU_LIST"
  run_test_seed "$winner" 42 "${gpus[0]}" & pids+=("$!")
  run_test_seed "$winner" 17 "${gpus[1]}" & pids+=("$!")
  run_test_seed "$winner" 73 "${gpus[2]}" & pids+=("$!")
  wait_for_workers "${pids[@]}"
}

echo "Run root: $RUN_ROOT"
echo "Phase: $PHASE"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: M0-M3 seed42 on four GPUs; top two replicate at seeds 17/73; frozen winner only on test."
  exit 0
fi
acquire_lock
status controller "$PHASE" start "" ""
case "$PHASE" in
  setup) run_setup ;;
  screen) run_setup; run_screen ;;
  replicate) run_replicate ;;
  freeze) run_freeze ;;
  test) run_test ;;
  all) run_setup; run_screen; run_replicate; run_freeze; run_test ;;
esac
status controller "$PHASE" end 0 ""
