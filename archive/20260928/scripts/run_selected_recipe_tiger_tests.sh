#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/reactzyme_b0_e2r_geometry_v1}"
[[ "$RUN_ROOT" = /* ]] || RUN_ROOT="$ROOT/$RUN_ROOT"
[[ $# -gt 0 ]] && shift

DETACH=0
POLL_SECONDS=60
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detach) DETACH=1; shift ;;
    --foreground) DETACH=0; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

OUT_ROOT="$RUN_ROOT/test/tiger_comparison"
REPORT_ROOT="$RUN_ROOT/reports/tiger_comparison"
CONTROLLER_LOG="$OUT_ROOT/controller.log"
if [[ "$DETACH" -eq 1 ]]; then
  mkdir -p "$OUT_ROOT"
  setsid -f "$0" "$RUN_ROOT" --foreground --poll-seconds "$POLL_SECONDS" \
    >"$CONTROLLER_LOG" 2>&1
  echo "Launched validation-selected TIGER comparison: $CONTROLLER_LOG"
  exit 0
fi

[[ -s "$RUN_ROOT/runtime.env" ]] || {
  echo "Run root is not initialized: $RUN_ROOT" >&2
  exit 2
}
source "$RUN_ROOT/runtime.env"
cd "$ROOT"

mkdir -p "$OUT_ROOT" "$REPORT_ROOT" "$RUN_ROOT"/{nohome,cache/huggingface,cache/torch,xdg_cache,xdg_config,xdg_state,matplotlib}
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
SHORT_TMPDIR="/tmp/hz-${UID}-tiger-comparison"
mkdir -p "$SHORT_TMPDIR"
chmod 700 "$SHORT_TMPDIR"
export TMPDIR="$SHORT_TMPDIR" TEMP="$SHORT_TMPDIR" TMP="$SHORT_TMPDIR"

PROMOTION="$RUN_ROOT/promotions/recipe.json"
PLAN="$RUN_ROOT/plans/recipe_confirm.tsv"
STATUS="$OUT_ROOT/status.jsonl"
JOBS="$OUT_ROOT/jobs.tsv"

while [[ ! -s "$PROMOTION" ]]; do
  printf '{"time":"%s","event":"waiting_for_validation_promotion"}\n' "$(date -Iseconds)" >>"$STATUS"
  sleep "$POLL_SECONDS"
done

"$PYTHON_BIN" scripts/reactzyme_tiger_comparison.py write-jobs \
  --promotion "$PROMOTION" --plan "$PLAN" --output-root "$OUT_ROOT/results" --output "$JOBS"

IFS=',' read -r -a GPU_IDS <<<"$GPUS"
WORKERS="${#GPU_IDS[@]}"
PIDS=()
for ((slot=0; slot<WORKERS; slot++)); do
  (
    failed=0
    index=0
    {
      IFS=$'\t' read -r _
      while IFS=$'\t' read -r run_id variant seed split checkpoint checkpoint_sha config output; do
        if (( index % WORKERS == slot )); then
          valid=0
          if [[ -s "$output" ]]; then
            "$PYTHON_BIN" scripts/reactzyme_tiger_comparison.py validate-output \
              --output "$output" --checkpoint "$checkpoint" --config "$config" \
              >/dev/null 2>&1 && valid=1
          fi
          if [[ "$valid" -eq 0 ]]; then
            mkdir -p "$(dirname "$output")"
            printf '{"time":"%s","event":"start","run_id":"%s","variant":"%s","seed":%s,"split":"%s","gpu":%s}\n' \
              "$(date -Iseconds)" "$run_id" "$variant" "$seed" "$split" "${GPU_IDS[$slot]}" >>"$STATUS"
            success=0
            for attempt in 1 2; do
              set +e
              OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
                CUDA_VISIBLE_DEVICES="${GPU_IDS[$slot]}" "$PYTHON_BIN" scripts/evaluate_protein_pooling.py \
                  --checkpoint "$checkpoint" --config "$config" --device cuda:0 \
                  --direction both --evaluation-protocol paper_test_candidates \
                  --batch-size 512 --target-batch-size 2048 --output "$output" \
                  >"${output%.json}.stdout.log" 2>&1
              rc=$?
              set -e
              if [[ "$rc" -eq 0 ]] && "$PYTHON_BIN" scripts/reactzyme_tiger_comparison.py validate-output \
                  --output "$output" --checkpoint "$checkpoint" --config "$config" >/dev/null 2>&1; then
                success=1
                break
              fi
              rm -f "$output"
              sleep 10
            done
            printf '{"time":"%s","event":"end","run_id":"%s","seed":%s,"split":"%s","gpu":%s,"success":%s}\n' \
              "$(date -Iseconds)" "$run_id" "$seed" "$split" "${GPU_IDS[$slot]}" "$success" >>"$STATUS"
            [[ "$success" -eq 1 ]] || failed=1
          fi
        fi
        index=$((index + 1))
      done
    } <"$JOBS"
    exit "$failed"
  ) & PIDS+=("$!")
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
[[ "$failed" -eq 0 ]] || {
  echo "At least one paper-test evaluation failed; see $STATUS" >&2
  exit 1
}

"$PYTHON_BIN" scripts/reactzyme_tiger_comparison.py aggregate \
  --promotion "$PROMOTION" --plan "$PLAN" --output-root "$OUT_ROOT/results" \
  --output-json "$REPORT_ROOT/comparison.json" \
  --output-tsv "$REPORT_ROOT/ours_long.tsv" \
  --output-markdown "$REPORT_ROOT/comparison.md"
printf '{"time":"%s","event":"complete","report":"%s"}\n' \
  "$(date -Iseconds)" "$REPORT_ROOT/comparison.md" >>"$STATUS"
