#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/reactzyme_reaction_smi_enzgfm650m_hybrid}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$ROOT/../env/bin/python}"
ENZGFM_PYTHON="${ENZGFM_PYTHON:-$ROOT/.deps/enzgfm-env-uv/bin/python}"
ENZGFM_REPO="${ENZGFM_REPO:-$ROOT/.deps/EnzGFM}"
ENZGFM_MODEL="${ENZGFM_MODEL:-$ROOT/.deps/EnzGFM_650M}"
CONFIG="${CONFIG:-$ROOT/configs/reactzyme_reaction_smi_f3_enzgfm650m_prott5_sleec.yaml}"
FASTA="$ROOT/data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/reaction_smi_train_validation_proteins.fasta"
RESIDUE_H5="$ROOT/data/standardized/retrieval_training_source_collapse/test/horizyn_reactzyme_shared_candidates/reaction_smi_train_validation_proteins_enzgfm_650m_residue.h5"
MASTER_PORT="${MASTER_PORT:-29763}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-30}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-90000}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/locks" "$RUN_ROOT/cache/huggingface" \
  "$RUN_ROOT/cache/torch" "$RUN_ROOT/cache/xdg" "$RUN_ROOT/cache/matplotlib" \
  "$RUN_ROOT/wandb"

exec 9>"$RUN_ROOT/locks/controller.lock"
flock -n 9 || { echo "Another reaction-SMILES EnzGFM controller is active" >&2; exit 1; }

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export MPLCONFIGDIR="$RUN_ROOT/cache/matplotlib"
export WANDB_DIR="$RUN_ROOT/wandb"
export WANDB_CACHE_DIR="$RUN_ROOT/wandb/cache"
export WANDB_CONFIG_DIR="$RUN_ROOT/wandb/config"
export TOKENIZERS_PARALLELISM=false

status() {
  printf '{"time":"%s","stage":"%s","event":"%s","returncode":"%s"}\n' \
    "$(date -Iseconds)" "$1" "$2" "${3:-}" >> "$RUN_ROOT/logs/status.jsonl"
}

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "This run requires four GPU IDs in CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

for required in "$TRAIN_PYTHON" "$ENZGFM_PYTHON" "$FASTA" "$CONFIG" \
  "$ENZGFM_REPO/models/modeling_EnzGFM.py" "$ENZGFM_MODEL/config.json"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

if [[ ! -s "$RESIDUE_H5" ]]; then
  status extract start ""
  set +e
  PYTHON_BIN="$ENZGFM_PYTHON" ENZGFM_REPO="$ENZGFM_REPO" \
    MODEL_LOCATION="$ENZGFM_MODEL" \
    scripts/run_extract_enzgfm_residue_embeddings_4gpu.sh \
      --run-name reactzyme_reaction_smi_enzgfm650m \
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
        f"Invalid EnzGFM store: proteins={actual}/{expected}, "
        f"residue_dim={residue_dim}, model_type={model_type}"
    )
print(f"Validated EnzGFM store: {actual} proteins, residue_dim={residue_dim}")
PY

train_log="$RUN_ROOT/logs/train.stdout.log"
train_cmd=("$TRAIN_PYTHON" scripts/train_protein_pooling.py --config "$CONFIG" --wandb)
last_checkpoint="$RUN_ROOT/checkpoints/last.ckpt"
if [[ -s "$last_checkpoint" ]]; then
  train_cmd+=(--resume "$last_checkpoint")
fi

gpus_have_capacity() {
  local gpu_id
  local free_mib
  for gpu_id in "${gpu_ids[@]}"; do
    free_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"
    if [[ ! "$free_mib" =~ ^[0-9]+$ ]] || (( free_mib < GPU_MIN_FREE_MIB )); then
      return 1
    fi
  done
}

if ! gpus_have_capacity; then
  status gpu_wait start ""
  echo "Waiting for GPUs $CUDA_VISIBLE_DEVICES to each have at least ${GPU_MIN_FREE_MIB} MiB free"
  until gpus_have_capacity; do
    sleep "$GPU_WAIT_SECONDS"
  done
  status gpu_wait end 0
fi

status train start ""
set +e
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" MASTER_PORT="$MASTER_PORT" \
  "${train_cmd[@]}" > "$train_log" 2>&1
train_rc=$?
set -e
status train end "$train_rc"
[[ "$train_rc" -eq 0 ]] || exit "$train_rc"
status controller complete 0
echo "Reaction-SMILES EnzGFM hybrid training complete"
