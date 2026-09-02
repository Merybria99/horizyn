#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="$ROOT/runs/reactzyme_f3_biological_no_ec_v1"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/.capability-run-py/bin/python}"
LOG="$RUN_ROOT/finalizer.log"
TAGS=(lambda_0p03 lambda_0p10 lambda_0p30 lambda_1p00)

mkdir -p "$RUN_ROOT/locks" "$RUN_ROOT/local_controllers"
exec 9>"$RUN_ROOT/locks/finalizer.lock"
flock -n 9 || { echo "Finalizer already active" >&2; exit 1; }
cd "$ROOT"

calibration_complete() {
  local tag="$1" status="$RUN_ROOT/calibration/$tag/status.jsonl"
  [[ -s "$status" ]] || return 1
  tail -n 1 "$status" | grep -q '"event":"end","returncode":0'
}

for tag in "${TAGS[@]}"; do
  while ! calibration_complete "$tag"; do
    echo "[$(date -Iseconds)] waiting for $tag" >> "$LOG"
    sleep 30
  done
done

selected="$($PYTHON_BIN scripts/select_f3_no_ec_aux_weight.py)"
echo "[$(date -Iseconds)] selected auxiliary weight $selected" >> "$LOG"
"$PYTHON_BIN" scripts/create_f3_biological_no_ec_campaign.py \
  --aux-weight "$selected" --max-epochs 30 >> "$LOG" 2>&1

for variant in F3_no_ec F3M_no_ec F3C_no_ec F3MC_no_ec; do
  "$PYTHON_BIN" scripts/train_protein_pooling.py \
    --config "$RUN_ROOT/configs/$variant/train.yaml" --model-preflight-only \
    >> "$LOG" 2>&1
done

printf 'ready\n' > "$RUN_ROOT/READY"
echo "[$(date -Iseconds)] final configs ready; launching local variants" >> "$LOG"

setsid nohup bash scripts/run_f3_no_ec_variant.sh F3_no_ec 0,1 30501 \
  > "$RUN_ROOT/local_controllers/F3_no_ec.controller.log" 2>&1 &
echo "$!" > "$RUN_ROOT/local_controllers/F3_no_ec.pid"

setsid nohup bash scripts/run_f3_no_ec_variant.sh F3M_no_ec 2,3 30502 \
  > "$RUN_ROOT/local_controllers/F3M_no_ec.controller.log" 2>&1 &
echo "$!" > "$RUN_ROOT/local_controllers/F3M_no_ec.pid"

echo "[$(date -Iseconds)] local launch complete" >> "$LOG"
