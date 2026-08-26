#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_b0_e2r_geometry_v1}"
[[ "$RUN_ROOT" = /* ]] || RUN_ROOT="$ROOT/$RUN_ROOT"
[[ $# -gt 0 ]] && shift

DETACH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

OUT_ROOT="$RUN_ROOT/test/recipe_all_seed42"
if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$OUT_ROOT"
  setsid -f "$0" "$RUN_ROOT" --foreground \
    >"$OUT_ROOT/controller.log" 2>&1
  echo "Launched recipe-screen official tests: $OUT_ROOT/controller.log"
  exit 0
fi

[[ -s "$RUN_ROOT/runtime.env" ]] || {
  echo "Run root is not initialized: $RUN_ROOT" >&2
  exit 2
}
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

mkdir -p "$OUT_ROOT" "$RUN_ROOT"/{nohome,cache/huggingface,cache/torch,xdg_cache,xdg_config,xdg_state,matplotlib}
export HOME="$RUN_ROOT/nohome"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export XDG_CONFIG_HOME="$RUN_ROOT/xdg_config"
export XDG_STATE_HOME="$RUN_ROOT/xdg_state"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT"
export TOKENIZERS_PARALLELISM=false
SHORT_TMPDIR="/tmp/hz-${UID}-recipe-official-test"
mkdir -p "$SHORT_TMPDIR"
chmod 700 "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR" TEMP="$SHORT_TMPDIR" TMP="$SHORT_TMPDIR"

mapfile -t JOBS < <(
  for variant in R0 R1 R2 R3 R4 R5 R6; do
    for split in time enzyme_smi reaction_smi; do
      validation_json="$RUN_ROOT/validation/recipe/recipe_${variant}_seed42/${split}.json"
      checkpoint="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$validation_json")"
      test_config="$RUN_ROOT/configs/recipe/recipe_${variant}_seed42/${split}/test.yaml"
      output="$OUT_ROOT/${variant}/${split}.json"
      printf '%s\t%s\t%s\t%s\n' "$variant" "$split" "$checkpoint" "$test_config|$output"
    done
  done
)

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
WORKERS="${#GPU_IDS[@]}"
PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do
  (
    failed=0
    for index in "${!JOBS[@]}"; do
      (( index % WORKERS == slot )) || continue
      IFS=$'\t' read -r variant split checkpoint config_and_output <<<"${JOBS[$index]}"
      test_config="${config_and_output%%|*}"
      output="${config_and_output#*|}"
      if [[ -s "$output" ]] && "$PYTHON_BIN" -c '
import json, pathlib, sys
d = json.load(open(sys.argv[1]))
required = ("enzyme_to_reaction/reactzyme_mrr", "reaction_to_enzyme/reactzyme_mrr")
ok = (
    all(key in d for key in required)
    and d.get("evaluation_protocol") == "paper_test_candidates"
    and pathlib.Path(d.get("checkpoint", "")).resolve() == pathlib.Path(sys.argv[2]).resolve()
    and pathlib.Path(d.get("config", "")).resolve() == pathlib.Path(sys.argv[3]).resolve()
)
raise SystemExit(0 if ok else 1)
' "$output" "$checkpoint" "$test_config"; then
        continue
      fi
      rm -f "$output"
      mkdir -p "$(dirname "$output")"
      printf '{"time":"%s","variant":"%s","split":"%s","event":"start","gpu":%s}\n' \
        "$(date -Iseconds)" "$variant" "$split" "${GPU_IDS[$slot]}" >>"$OUT_ROOT/status.jsonl"
      set +e
      OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        CUDA_VISIBLE_DEVICES="${GPU_IDS[$slot]}" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
          --checkpoint "$checkpoint" --config "$test_config" --device cuda:0 \
          --direction both --evaluation-protocol paper_test_candidates \
          --batch-size 512 --target-batch-size 2048 --output "$output" \
          >"${output%.json}.stdout.log" 2>&1
      rc=$?
      set -e
      printf '{"time":"%s","variant":"%s","split":"%s","event":"end","gpu":%s,"returncode":%s}\n' \
        "$(date -Iseconds)" "$variant" "$split" "${GPU_IDS[$slot]}" "$rc" >>"$OUT_ROOT/status.jsonl"
      [[ "$rc" -eq 0 ]] || failed=1
    done
    exit "$failed"
  ) & PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done

{
  printf 'variant\tsplit\te2r_hit1\te2r_reactzyme_mrr\te2r_first_positive_mrr\tr2e_hit1\tr2e_reactzyme_mrr\tr2e_first_positive_mrr\n'
  for variant in R0 R1 R2 R3 R4 R5 R6; do
    for split in time enzyme_smi reaction_smi; do
      output="$OUT_ROOT/${variant}/${split}.json"
      [[ -s "$output" ]] || continue
      "$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[3])); print("\t".join([sys.argv[1],sys.argv[2],str(d["enzyme_to_reaction/top_1"]),str(d["enzyme_to_reaction/reactzyme_mrr"]),str(d["enzyme_to_reaction/first_positive_mrr"]),str(d["reaction_to_enzyme/top_1"]),str(d["reaction_to_enzyme/reactzyme_mrr"]),str(d["reaction_to_enzyme/first_positive_mrr"])]))' \
        "$variant" "$split" "$output"
    done
  done
} >"$OUT_ROOT/summary.tsv"

exit "$failed"
