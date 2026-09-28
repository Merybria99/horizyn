#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT_ARG="${1:-$ROOT/runs/reactzyme_f3_shortcut_ablation_v1}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then shift; fi
case "$RUN_ROOT_ARG" in
  /*) RUN_ROOT="$RUN_ROOT_ARG" ;;
  *) RUN_ROOT="$ROOT/$RUN_ROOT_ARG" ;;
esac

PHASE=all
ONLY_VARIANT=""
ONLY_SPLIT=""
ONLY_SEED=""
DETACH=0
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --variant) ONLY_VARIANT="${2:?missing variant}"; shift 2 ;;
    --split) ONLY_SPLIT="${2:?missing split}"; shift 2 ;;
    --seed) ONLY_SEED="${2:?missing seed}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in setup|screen|replicate|freeze|test|all) ;; *) echo "Invalid phase: $PHASE" >&2; exit 2 ;; esac
[[ -z "$ONLY_VARIANT" || "$ONLY_VARIANT" =~ ^S[0-5]$ ]] || { echo "Invalid variant" >&2; exit 2; }
[[ -z "$ONLY_SPLIT" || "$ONLY_SPLIT" =~ ^(time|enzyme_smi|reaction_smi)$ ]] || { echo "Invalid split" >&2; exit 2; }

mkdir -p "$RUN_ROOT/logs/controller"
if [[ "$DETACH" -eq 1 ]]; then
  args=("$RUN_ROOT" --phase "$PHASE" --foreground)
  [[ -n "$ONLY_VARIANT" ]] && args+=(--variant "$ONLY_VARIANT")
  [[ -n "$ONLY_SPLIT" ]] && args+=(--split "$ONLY_SPLIT")
  [[ -n "$ONLY_SEED" ]] && args+=(--seed "$ONLY_SEED")
  [[ "$DRY_RUN" -eq 1 ]] && args+=(--dry-run)
  setsid -f "$0" "${args[@]}" > "$RUN_ROOT/logs/controller/nohup.log" 2>&1
  echo "Launched F3 shortcut campaign controller"
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
  "$BOOTSTRAP_PYTHON" "$ROOT/scripts/create_reactzyme_f3_shortcut_campaign.py" generate \
    --run-root "$RUN_ROOT" --protocol-root "$PROTOCOL_ROOT"
fi
# shellcheck source=/dev/null
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

HOST="$(hostname -s)"
SHORT_TMPDIR="${F3_SHORTCUT_TMPDIR:-/tmp/horizyn-f3-shortcut-${UID}-${HOST}}"
mkdir -p \
  "$RUN_ROOT/nohome/$HOST" "$RUN_ROOT/cache/$HOST/huggingface" \
  "$RUN_ROOT/cache/$HOST/torch" "$RUN_ROOT/cache/$HOST/xdg" \
  "$RUN_ROOT/cache/$HOST/matplotlib" "$RUN_ROOT/wandb/$HOST" \
  "$RUN_ROOT/logs/setup" "$RUN_ROOT/logs/controller" "$RUN_ROOT/results" \
  "$RUN_ROOT/reports" "$RUN_ROOT/locks" "$SHORT_TMPDIR"
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
  printf '{"time":"%s","host":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$HOST" "$1" "$2" "$3" "${4:-}" >> "$STATUS_LOG"
}

acquire_lock() {
  exec 9>"$RUN_ROOT/locks/$HOST.controller.lock"
  flock -n 9 || { echo "Another shortcut campaign owns the host lock" >&2; exit 1; }
}

setup_chemistry() {
  local split="$1"
  local out="$RUN_ROOT/data/$split/reaction_set"
  local schema="$out/schema.json"
  local valid=0
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
    status "$split" setup skip 0
    return
  fi
  rm -rf "$out"
  mkdir -p "$out"
  status "$split" setup start ""
  "$SETUP_PYTHON_BIN" scripts/build_reaction_set_features.py \
    --train-reactions "$PROTOCOL_ROOT/$split/train_rxns.csv" \
    --validation-reactions "$PROTOCOL_ROOT/$split/validation_rxns.csv" \
    --test-reactions "$PROTOCOL_ROOT/$split/test_rxns.csv" \
    --cofactor-dictionary "$ROOT/data/processed/capability_features/train_exact_rhea_reconstructed/cofactor_dictionary.csv" \
    --out-dir "$out" > "$RUN_ROOT/logs/setup/${split}_reaction_set.log" 2>&1
  status "$split" setup end 0
}

run_setup() {
  local split
  for split in time enzyme_smi reaction_smi; do
    [[ -z "$ONLY_SPLIT" || "$split" == "$ONLY_SPLIT" ]] || continue
    setup_chemistry "$split"
  done
}

run_train() {
  local variant="$1" split="$2" seed="$3"
  local run_id="${variant}_${split}_seed${seed}"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/train.yaml"
  local checkpoint_dir="$RUN_ROOT/checkpoints/$run_id"
  local log="$RUN_ROOT/logs/$run_id/train.log"
  local marker="$RUN_ROOT/logs/$run_id/train.complete"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$marker" ]]; then status "$run_id" train skip 0; return; fi
  local command=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
  [[ -s "$checkpoint_dir/last.ckpt" ]] && command+=(--resume "$checkpoint_dir/last.ckpt")
  if [[ "$WANDB_MODE" != disabled ]]; then
    command+=(--wandb --wandb-project "$WANDB_PROJECT" \
      --wandb-run-name "reactzyme-f3-shortcut-$run_id" --wandb-mode "$WANDB_MODE")
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
  find "$checkpoint_dir" -maxdepth 1 -name 'protein-pooling-*.ckpt' -type f | grep -q . \
    || { echo "No top-three checkpoint produced for $run_id" >&2; return 1; }
  date -Iseconds > "$marker"
}

run_validation() {
  local variant="$1" split="$2" seed="$3"
  local run_id="${variant}_${split}_seed${seed}"
  local checkpoint_dir="$RUN_ROOT/checkpoints/$run_id"
  local config="$RUN_ROOT/configs/$variant/$split/seed$seed/validation.yaml"
  local output_dir="$RUN_ROOT/results/validation/$variant/$split/seed$seed"
  local eval_gpu="${GPUS%%,*}" checkpoint stem output log
  mkdir -p "$output_dir" "$RUN_ROOT/logs/$run_id/validation"
  while IFS= read -r checkpoint; do
    stem="$(basename "${checkpoint%.ckpt}")"
    output="$output_dir/$stem.json"
    log="$RUN_ROOT/logs/$run_id/validation/$stem.log"
    [[ -s "$output" ]] && continue
    status "$run_id" validation start ""
    CUDA_VISIBLE_DEVICES="$eval_gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device cuda --direction both \
      --evaluation-protocol configured_forward_candidates --batch-size 128 \
      --target-batch-size 512 --output "$output" > "$log" 2>&1
    status "$run_id" validation end 0
  done < <(find "$checkpoint_dir" -maxdepth 1 -name 'protein-pooling-*.ckpt' -type f | sort)
}

run_variant_seed() {
  local variant="$1" seed="$2" split
  for split in time enzyme_smi reaction_smi; do
    [[ -z "$ONLY_SPLIT" || "$split" == "$ONLY_SPLIT" ]] || continue
    run_train "$variant" "$split" "$seed"
    run_validation "$variant" "$split" "$seed"
  done
}

run_screen() {
  local variants="S0 S1 S2 S3 S4 S5" variant seed="${ONLY_SEED:-42}"
  [[ -n "$ONLY_VARIANT" ]] && variants="$ONLY_VARIANT"
  for variant in $variants; do run_variant_seed "$variant" "$seed"; done
  if [[ -z "$ONLY_VARIANT" && -z "$ONLY_SPLIT" && -z "$ONLY_SEED" ]]; then
    "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_shortcut_campaign \
      select-screen --run-root "$RUN_ROOT"
  fi
}

run_replicate() {
  [[ -s "$RUN_ROOT/reports/selected_variants.txt" ]] || \
    "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_shortcut_campaign \
      select-screen --run-root "$RUN_ROOT"
  local variants seed variant
  variants="$(tr '\n' ' ' < "$RUN_ROOT/reports/selected_variants.txt")"
  [[ -n "$ONLY_VARIANT" ]] && variants="$ONLY_VARIANT"
  for variant in $variants; do
    for seed in ${ONLY_SEED:-17 73}; do run_variant_seed "$variant" "$seed"; done
  done
  if [[ -z "$ONLY_VARIANT" && -z "$ONLY_SPLIT" && -z "$ONLY_SEED" ]]; then
    "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_shortcut_campaign \
      select-final --run-root "$RUN_ROOT"
  fi
}

run_freeze() {
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_shortcut_campaign \
    freeze --run-root "$RUN_ROOT"
}

run_test() {
  local manifest="$RUN_ROOT/frozen/manifest.json"
  [[ -s "$manifest" ]] || { echo "Refusing test access before winner freeze" >&2; exit 1; }
  local winner
  winner="$($SETUP_PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"])' "$manifest")"
  local seed split checkpoint config output log eval_gpu="${GPUS%%,*}"
  for seed in 42 17 73; do
    for split in time enzyme_smi reaction_smi; do
      checkpoint="$RUN_ROOT/frozen/$winner/seed$seed/$split/best.ckpt"
      config="$RUN_ROOT/configs/$winner/$split/seed$seed/test.yaml"
      output="$RUN_ROOT/results/test/$winner/seed$seed/$split.json"
      log="$RUN_ROOT/logs/${winner}_${split}_seed${seed}/test.log"
      [[ -s "$output" ]] && continue
      mkdir -p "$(dirname "$output")" "$(dirname "$log")"
      status "${winner}_${split}_seed${seed}" test start ""
      CUDA_VISIBLE_DEVICES="$eval_gpu" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
        --checkpoint "$checkpoint" --config "$config" --device cuda --direction both \
        --evaluation-protocol paper_test_candidates --batch-size 128 \
        --target-batch-size 512 --output "$output" > "$log" 2>&1
      status "${winner}_${split}_seed${seed}" test end 0
    done
  done
  "$SETUP_PYTHON_BIN" -m horizyn.benchmarks.reactzyme_f3_shortcut_campaign \
    report --run-root "$RUN_ROOT"
}

echo "Run root: $RUN_ROOT"
echo "Phase: $PHASE"
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: S0-S5 screen at seed42; top two repeat at seeds 17 and 73; frozen winner only on test."
  exit 0
fi
acquire_lock
status controller "$PHASE" start ""
case "$PHASE" in
  setup) run_setup ;;
  screen) run_screen ;;
  replicate) run_replicate ;;
  freeze) run_freeze ;;
  test) run_test ;;
  all)
    run_setup
    run_screen
    run_replicate
    run_freeze
    run_test
    ;;
esac
status controller "$PHASE" end 0
