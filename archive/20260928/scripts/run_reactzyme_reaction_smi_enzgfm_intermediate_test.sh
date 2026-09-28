#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/reactzyme_reaction_smi_enzgfm650m_hybrid/intermediate_tests/epoch10}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$ROOT/../env/bin/python}"
ENZGFM_PYTHON="${ENZGFM_PYTHON:-$ROOT/.deps/enzgfm-env-uv/bin/python}"
ENZGFM_REPO="${ENZGFM_REPO:-$ROOT/.deps/EnzGFM}"
ENZGFM_MODEL="${ENZGFM_MODEL:-$ROOT/.deps/EnzGFM_650M}"
EXTRACT_GPUS="${EXTRACT_GPUS:-1,3}"
EVAL_GPU="${EVAL_GPU:-1}"

FASTA="$RUN_ROOT/inputs/test_proteins.fasta"
RESIDUE_H5="$RUN_ROOT/inputs/test_proteins_enzgfm_650m_residue.h5"
CHECKPOINT="$RUN_ROOT/checkpoint.ckpt"
CONFIG="$RUN_ROOT/test.yaml"
RESULT="$RUN_ROOT/test_both.json"
TARGET_CACHE="$RUN_ROOT/cache/target_embeddings.pt"
STATUS_LOG="$RUN_ROOT/logs/status.jsonl"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/cache" "$RUN_ROOT/locks"
exec 9>"$RUN_ROOT/locks/controller.lock"
flock -n 9 || { echo "Another epoch-10 intermediate-test controller is active" >&2; exit 1; }

cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"

status() {
  printf '{"time":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "${3:-}" >> "$STATUS_LOG"
}

for required in "$TRAIN_PYTHON" "$ENZGFM_PYTHON" "$FASTA" "$CHECKPOINT" "$CONFIG"; do
  if [[ ! -s "$required" ]]; then
    echo "Required input is missing or empty: $required" >&2
    exit 2
  fi
done

if [[ ! -s "$RESIDUE_H5" ]]; then
  status extract start ""
  set +e
  CUDA_VISIBLE_DEVICES="$EXTRACT_GPUS" \
    PYTHON_BIN="$ENZGFM_PYTHON" \
    ENZGFM_REPO="$ENZGFM_REPO" \
    MODEL_LOCATION="$ENZGFM_MODEL" \
    scripts/run_extract_enzgfm_residue_embeddings_4gpu.sh \
      --run-name reactzyme_reaction_smi_enzgfm_epoch10_test \
      --fasta "$FASTA" \
      --output "$RESIDUE_H5" \
      --batch-size 128 \
      --max-tokens-per-batch 65536 \
      --cleanup-shards
  extract_rc=$?
  set -e
  status extract end "$extract_rc"
  [[ "$extract_rc" -eq 0 ]] || exit "$extract_rc"
fi

"$ENZGFM_PYTHON" - "$RESIDUE_H5" "$FASTA" <<'PY'
import sys
from pathlib import Path

import h5py

h5_path, fasta_path = map(Path, sys.argv[1:])
expected = sum(1 for line in fasta_path.open() if line.startswith(">"))
with h5py.File(h5_path, "r") as handle:
    actual = len(handle["ids"])
    residue_dim = int(handle.attrs["residue_dim"])
    model_type = str(handle.attrs["embedding_model_type"])
if actual != expected or residue_dim != 2048 or model_type != "enzgfm":
    raise RuntimeError(
        f"Invalid EnzGFM test store: proteins={actual}/{expected}, "
        f"residue_dim={residue_dim}, model_type={model_type}"
    )
print(f"Validated EnzGFM test store: {actual} proteins, residue_dim={residue_dim}")
PY

status evaluate start ""
set +e
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$TRAIN_PYTHON" scripts/evaluate_protein_pooling.py \
  --checkpoint "$CHECKPOINT" \
  --config "$CONFIG" \
  --device cuda \
  --direction both \
  --evaluation-protocol paper_test_candidates \
  --batch-size 512 \
  --target-batch-size 2048 \
  --target-embeds-cache "$TARGET_CACHE" \
  --output "$RESULT" \
  > "$RUN_ROOT/logs/evaluation.stdout.log" 2>&1
eval_rc=$?
set -e
status evaluate end "$eval_rc"
[[ "$eval_rc" -eq 0 ]] || exit "$eval_rc"
status controller complete 0
echo "Intermediate test evaluation complete: $RESULT"
