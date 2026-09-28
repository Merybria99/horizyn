#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_b0_e2r_geometry_v1}"
[[ "$RUN_ROOT" = /* ]] || RUN_ROOT="$ROOT/$RUN_ROOT"
[[ $# -gt 0 ]] && shift

PHASE="full"
DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
case "$PHASE" in
  full|prepare|recipe|reaction|enzyme|hard-negative|freeze|test|status) ;;
  *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$RUN_ROOT/logs/controller"
  setsid -f "$0" "$RUN_ROOT" --phase "$PHASE" --foreground \
    >"$RUN_ROOT/logs/controller/nohup.log" 2>&1
  echo "Launched controller: $RUN_ROOT/logs/controller/nohup.log"
  exit 0
fi

[[ -s "$RUN_ROOT/runtime.env" ]] || {
  echo "Run root is not initialized: $RUN_ROOT" >&2
  echo "Run scripts/create_reactzyme_geometry_ablation_matrix.py init first." >&2
  exit 2
}
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

mkdir -p "$RUN_ROOT"/{logs/controller,nohome,tmp,cache/huggingface,cache/torch,xdg_cache,xdg_config,xdg_state,wandb/data,wandb/cache,wandb/config,matplotlib,selection,promotions,reports}
export HOME="$RUN_ROOT/nohome"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export WANDB_ROOT="$RUN_ROOT/wandb"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_DATA_DIR="$RUN_ROOT/wandb/data"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export WANDB_MODE TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT"
export MASTER_ADDR="${HORIZYN_MASTER_ADDR:-127.0.0.1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
SHORT_TMPDIR="${HORIZYN_SHORT_TMPDIR:-/tmp/hz-${UID}-${BASE_MASTER_PORT}}"
mkdir -p "$SHORT_TMPDIR"
[[ "$(stat -c %u "$SHORT_TMPDIR")" == "$UID" ]] || {
  echo "Short TMPDIR is owned by another user: $SHORT_TMPDIR" >&2
  exit 1
}
chmod 700 "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR" TEMP="$SHORT_TMPDIR" TMP="$SHORT_TMPDIR"
if [[ -z "${WANDB_API_KEY:-}" && -s "$RUN_ROOT/wandb/api_key" ]]; then
  export WANDB_API_KEY="$(tr -d '\r\n' <"$RUN_ROOT/wandb/api_key")"
fi

GEOMETRY="$ROOT/scripts/create_reactzyme_geometry_ablation_matrix.py"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"

status() {
  printf '{"time":"%s","run_id":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "$3" "${4:-}" >>"$STATUS_LOG"
}

config_value() {
  "$PYTHON_BIN" -c \
    'import sys,yaml; from functools import reduce; data=yaml.safe_load(open(sys.argv[1],encoding="utf-8")); print(reduce(lambda value,key:value[key],sys.argv[2].split("."),data))' \
    "$1" "$2"
}

select_checkpoint() {
  local checkpoint_dir="$1" log="$2" best=""
  [[ -f "$log" ]] && best="$(awk -F'Best checkpoint: ' '/Best checkpoint:/ {x=$2} END {print x}' "$log" | tr -d '\r')"
  if [[ -n "$best" && -f "$best" ]]; then
    printf '%s\n' "$best"
  elif [[ -f "$checkpoint_dir/last.ckpt" ]]; then
    printf '%s\n' "$checkpoint_dir/last.ckpt"
  else
    find "$checkpoint_dir" -maxdepth 1 -name '*.ckpt' -type f -printf '%T@ %p\n' \
      | sort -nr | awk 'NR==1{sub(/^[^ ]+ /,"");print}'
  fi
}

prepare_data() {
  "$ROOT/scripts/run_reactzyme_biological_ablation_matrix.sh" \
    "$RUN_ROOT" --phase prepare --foreground
}

run_pretrain_row() {
  local run_id="$1" port="$2" biological_config="$3" ec_config="$4"
  if [[ "$ec_config" != "-" ]]; then
    local ec_checkpoint ec_log ec_rc
    ec_checkpoint="$(config_value "$ec_config" output_checkpoint)"
    ec_log="$(dirname "$ec_config")/../../../ec_pretrain.stdout.log"
    if [[ ! -s "$ec_checkpoint" ]]; then
      mkdir -p "$(dirname "$ec_checkpoint")" "$(dirname "$ec_log")"
      status "$run_id" ec_pretrain start
      set +e
      CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc-per-node=4 --master-port="$((port + 1000))" \
        scripts/pretrain_hyperbolic_enzyme.py --config "$ec_config" >"$ec_log" 2>&1
      ec_rc=$?
      set -e
      status "$run_id" ec_pretrain end "$ec_rc"
      [[ "$ec_rc" -eq 0 ]] || return "$ec_rc"
    fi
  fi
  if [[ "$biological_config" != "-" ]]; then
    local biological_dir biological_checkpoint biological_log biological_rc
    biological_dir="$(config_value "$biological_config" logging.checkpoint_dir)"
    biological_checkpoint="$biological_dir/best.ckpt"
    biological_log="$biological_dir/stdout.log"
    if [[ ! -s "$biological_checkpoint" ]]; then
      mkdir -p "$biological_dir"
      status "$run_id" biological_pretrain start
      set +e
      CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$((port + 1000))" \
        "$PYTHON_BIN" scripts/pretrain_enzyme_biofp_split.py \
        --config "$biological_config" >"$biological_log" 2>&1
      biological_rc=$?
      set -e
      status "$run_id" biological_pretrain end "$biological_rc"
      [[ "$biological_rc" -eq 0 ]] || return "$biological_rc"
    fi
  fi
}

mine_hard_negatives_row() {
  local run_id="$1" parent_checkpoint="$2" mining_config="$3" output="$4" device="$5"
  [[ "$output" != "-" ]] || return 0
  [[ -s "$output" ]] && return 0
  mkdir -p "$(dirname "$output")"
  status "$run_id" hard_negative_mining start
  local rc
  set +e
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$parent_checkpoint" --config "$mining_config" --device "$device" \
      --direction reaction_to_enzyme \
      --evaluation-protocol configured_forward_candidates \
      --batch-size 128 --target-batch-size 2048 \
      --hard-negative-output "$output" --hard-negative-top-k 64 \
      --output "${output%.json}.metrics.json" \
      >"${output%.json}.stdout.log" 2>&1
  rc=$?
  set -e
  status "$run_id" hard_negative_mining end "$rc"
  return "$rc"
}

run_train_row() {
  local run_id="$1" config="$2" checkpoint_dir="$3" sidecar="$4" wandb_name="$5" port="$6" parent_checkpoint="$7"
  local existing checkpoint log rc resume=()
  existing="$($PYTHON_BIN -c 'from pathlib import Path; from horizyn.benchmarks.reactzyme_runtime import read_checkpoint_sidecar; import sys; print(read_checkpoint_sidecar(Path(sys.argv[1])) or "")' "$sidecar")"
  [[ -n "$existing" ]] && return 0
  log="$RUN_ROOT/logs/train/${run_id}.log"
  mkdir -p "$checkpoint_dir" "$(dirname "$log")"
  if [[ -s "$checkpoint_dir/last.ckpt" ]]; then
    resume=(--resume "$checkpoint_dir/last.ckpt")
  elif [[ "$parent_checkpoint" != "-" ]]; then
    resume=(--resume "$parent_checkpoint")
  fi
  status "$run_id" retrieval start
  local cmd=("$PYTHON_BIN" scripts/train_protein_pooling.py --config "$config")
  [[ "${#resume[@]}" -gt 0 ]] && cmd+=("${resume[@]}")
  if [[ "$WANDB_MODE" != disabled ]]; then
    cmd+=(--wandb --wandb-project "$WANDB_PROJECT" --wandb-run-name "$wandb_name" --wandb-mode "$WANDB_MODE")
    [[ -n "${WANDB_ENTITY:-}" ]] && cmd+=(--wandb-entity "$WANDB_ENTITY")
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$GPUS" MASTER_PORT="$port" "${cmd[@]}" >"$log" 2>&1
  rc=$?
  set -e
  status "$run_id" retrieval end "$rc"
  [[ "$rc" -eq 0 ]] || return "$rc"
  checkpoint="$(select_checkpoint "$checkpoint_dir" "$log")"
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  "$PYTHON_BIN" "$GEOMETRY" write-checkpoint "$sidecar" "$run_id" "$checkpoint"
}

run_validation_row() {
  local run_id="$1" sidecar="$2" config="$3" output="$4" device="$5"
  [[ -s "$output" ]] && return 0
  local checkpoint rc
  checkpoint="$($PYTHON_BIN -c 'from pathlib import Path; from horizyn.benchmarks.reactzyme_runtime import read_checkpoint_sidecar; import sys; print(read_checkpoint_sidecar(Path(sys.argv[1])) or "")' "$sidecar")"
  [[ -n "$checkpoint" ]] || return 1
  mkdir -p "$(dirname "$output")"
  status "$run_id" validation start
  set +e
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device "$device" \
      --direction both --evaluation-protocol configured_forward_candidates \
      --batch-size 512 --target-batch-size 2048 --output "$output" \
      >"${output%.json}.stdout.log" 2>&1
  rc=$?
  set -e
  status "$run_id" validation end "$rc"
  return "$rc"
}

run_wave() {
  local plan="$1" variant="$2" seed="$3"
  local wave_file="$RUN_ROOT/tmp/wave_${variant}_seed${seed}.tsv"
  awk -F'\t' -v v="$variant" -v s="$seed" 'NR==1 || ($2==v && $4==s)' "$plan" >"$wave_file"
  local pids=() failed=0 slot=0
  local stage row_variant label row_seed split run_id config validation_config mining_config test_config checkpoint_dir checkpoint_sidecar validation_eval test_eval wandb_name port parent biological ec hardneg description

  {
    IFS=$'\t' read -r _
    while IFS=$'\t' read -r stage row_variant label row_seed split run_id config validation_config mining_config test_config checkpoint_dir checkpoint_sidecar validation_eval test_eval wandb_name port parent biological ec hardneg description; do
      (run_pretrain_row "$run_id" "$port" "$biological" "$ec") & pids+=("$!")
    done
  } <"$wave_file"
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1

  pids=(); failed=0; slot=0
  {
    IFS=$'\t' read -r _
    while IFS=$'\t' read -r stage row_variant label row_seed split run_id config validation_config mining_config test_config checkpoint_dir checkpoint_sidecar validation_eval test_eval wandb_name port parent biological ec hardneg description; do
      if [[ "$hardneg" != "-" ]]; then
        (mine_hard_negatives_row "$run_id" "$parent" "$mining_config" "$hardneg" "cuda:$slot") & pids+=("$!")
        slot=$((slot + 1))
      fi
    done
  } <"$wave_file"
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1

  pids=(); failed=0
  {
    IFS=$'\t' read -r _
    while IFS=$'\t' read -r stage row_variant label row_seed split run_id config validation_config mining_config test_config checkpoint_dir checkpoint_sidecar validation_eval test_eval wandb_name port parent biological ec hardneg description; do
      (run_train_row "$run_id" "$config" "$checkpoint_dir" "$checkpoint_sidecar" "$wandb_name" "$port" "$parent") & pids+=("$!")
    done
  } <"$wave_file"
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1

  pids=(); failed=0; slot=0
  {
    IFS=$'\t' read -r _
    while IFS=$'\t' read -r stage row_variant label row_seed split run_id config validation_config mining_config test_config checkpoint_dir checkpoint_sidecar validation_eval test_eval wandb_name port parent biological ec hardneg description; do
      (run_validation_row "$run_id" "$checkpoint_sidecar" "$validation_config" "$validation_eval" "cuda:$slot") & pids+=("$!")
      slot=$((slot + 1))
    done
  } <"$wave_file"
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
}

run_plan() {
  local plan="$1" variant seed
  while IFS=$'\t' read -r variant seed; do
    run_wave "$plan" "$variant" "$seed"
  done < <(awk -F'\t' 'NR>1 {key=$2 FS $4; if (!seen[key]++) print $2 FS $4}' "$plan")
}

generate_stage() {
  local stage="$1" parent="${2:-}"; shift 2 || true
  local args=(
    "$PYTHON_BIN" "$GEOMETRY" generate-stage
    --run-root "$RUN_ROOT" --protocol-root "$PROTOCOL_ROOT" --feature-root "$FEATURE_ROOT"
    --stage "$stage" --python-bin "$PYTHON_BIN" --setup-python-bin "$SETUP_PYTHON_BIN"
    --wandb-project "$WANDB_PROJECT" --wandb-entity "$WANDB_ENTITY" --wandb-mode "$WANDB_MODE"
  )
  [[ -n "$parent" ]] && args+=(--parent-promotion "$parent")
  args+=("$@")
  "${args[@]}"
}

run_recipe_stage() {
  local screen="$RUN_ROOT/promotions/recipe_screen.json"
  local final="$RUN_ROOT/promotions/recipe.json"
  generate_stage recipe "" --seed 42 --base-master-port 24800
  run_plan "$RUN_ROOT/plans/recipe.tsv"
  "$PYTHON_BIN" "$GEOMETRY" promote --plan "$RUN_ROOT/plans/recipe.tsv" --output "$screen" --top-k 2
  local selected_args=()
  while IFS= read -r variant; do selected_args+=(--variant "$variant"); done < <(
    "$PYTHON_BIN" -c 'import json,sys; p=json.load(open(sys.argv[1])); values=list(dict.fromkeys([*p["selected_variants"],"R0"])); print("\n".join(values))' "$screen"
  )
  generate_stage recipe_confirm "$screen" --seed 7 --seed 42 --seed 137 \
    --base-master-port 25800 "${selected_args[@]}"
  run_plan "$RUN_ROOT/plans/recipe_confirm.tsv"
  "$PYTHON_BIN" "$GEOMETRY" promote --plan "$RUN_ROOT/plans/recipe_confirm.tsv" --output "$final" --top-k 1
}

run_reaction_stage() {
  generate_stage reaction "$RUN_ROOT/promotions/recipe.json" --seed 42 --base-master-port 26800
  run_plan "$RUN_ROOT/plans/reaction.tsv"
  "$PYTHON_BIN" "$GEOMETRY" promote --plan "$RUN_ROOT/plans/reaction.tsv" \
    --output "$RUN_ROOT/promotions/reaction.json" --top-k 1
}

run_enzyme_stage() {
  generate_stage enzyme "$RUN_ROOT/promotions/reaction.json" --seed 42 --base-master-port 27800
  run_plan "$RUN_ROOT/plans/enzyme.tsv"
  "$PYTHON_BIN" "$GEOMETRY" promote --plan "$RUN_ROOT/plans/enzyme.tsv" \
    --output "$RUN_ROOT/promotions/enzyme.json" --top-k 1
}

run_hard_negative_stage() {
  generate_stage hard_negative "$RUN_ROOT/promotions/enzyme.json" --seed 42 --base-master-port 28800
  run_plan "$RUN_ROOT/plans/hard_negative.tsv"
  "$PYTHON_BIN" "$GEOMETRY" promote --plan "$RUN_ROOT/plans/hard_negative.tsv" \
    --output "$RUN_ROOT/promotions/hard_negative.json" --top-k 1
}

freeze_all() {
  "$PYTHON_BIN" "$GEOMETRY" freeze-all --run-root "$RUN_ROOT" \
    --output "$RUN_ROOT/selection/frozen_all.json"
}

run_test_row() {
  local run_id="$1" checkpoint="$2" config="$3" output="$4" device="$5"
  [[ -s "$output" ]] && return 0
  mkdir -p "$(dirname "$output")"
  status "$run_id" official_test start
  local rc
  set +e
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    CUDA_VISIBLE_DEVICES="$GPUS" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
      --checkpoint "$checkpoint" --config "$config" --device "$device" \
      --direction both --evaluation-protocol paper_test_candidates \
      --batch-size 512 --target-batch-size 2048 --output "$output" \
      >"${output%.json}.stdout.log" 2>&1
  rc=$?
  set -e
  status "$run_id" official_test end "$rc"
  return "$rc"
}

test_all() {
  local selection="$RUN_ROOT/selection/frozen_all.json"
  [[ -s "$selection" ]] || freeze_all
  local rows="$RUN_ROOT/selection/verified_rows.tsv"
  "$PYTHON_BIN" "$GEOMETRY" selection-rows "$selection" >"$rows"
  local gpu_ids slot pids=() failed=0
  IFS=',' read -r -a gpu_ids <<<"$GPUS"
  local workers="${#gpu_ids[@]}"
  for ((slot=0; slot<workers; slot++)); do
    (
      local index=0 stage variant seed split run_id checkpoint sha test_config test_eval
      {
        IFS=$'\t' read -r _
        while IFS=$'\t' read -r stage variant seed split run_id checkpoint sha test_config test_eval; do
          if (( index % workers == slot )); then
            run_test_row "$run_id" "$checkpoint" "$test_config" "$test_eval" "cuda:$slot"
          fi
          index=$((index + 1))
        done
      } <"$rows"
    ) & pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]] || return 1
  "$PYTHON_BIN" "$GEOMETRY" write-report --run-root "$RUN_ROOT" \
    --output-csv "$RUN_ROOT/reports/official_test.tsv" \
    --output-markdown "$RUN_ROOT/reports/official_test.md"
}

show_status() {
  echo "Run root: $RUN_ROOT"
  [[ -s "$STATUS_LOG" ]] && tail -n 30 "$STATUS_LOG" || echo "No status events yet."
  pgrep -af 'train_protein_pooling.py|pretrain_enzyme_biofp_split.py|pretrain_hyperbolic_enzyme.py|evaluate_protein_pooling.py' || true
}

echo "Run root: $RUN_ROOT"
echo "Phase: $PHASE"
case "$PHASE" in
  prepare) prepare_data ;;
  recipe) run_recipe_stage ;;
  reaction) run_reaction_stage ;;
  enzyme) run_enzyme_stage ;;
  hard-negative) run_hard_negative_stage ;;
  freeze) freeze_all ;;
  test) test_all ;;
  status) show_status ;;
  full)
    prepare_data
    run_recipe_stage
    run_reaction_stage
    run_enzyme_stage
    run_hard_negative_stage
    freeze_all
    test_all
    ;;
esac
