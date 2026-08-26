#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
DATABASE_DIR="${DATABASE_DIR:-$ROOT/wet_lab/databases/swissprot/current}"
RUN_ROOT="${RUN_ROOT:-$ROOT/wet_lab/runs/swissprot_build}"
GPUS="${GPUS:-1,2,3}"
QUERY_GPU="${QUERY_GPU:-${GPUS%%,*}}"
CONFIG="${CONFIG:-wet_lab/configs/swissprot/all_f4.yaml}"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/nohome" "$RUN_ROOT/cache/huggingface"
export HOME="$RUN_ROOT/nohome"
export HF_HOME="${HF_HOME:-/datastor2/deep-proteins/EnzymeDiscovery/hf_cache}"
export XDG_CACHE_HOME="$RUN_ROOT/cache/xdg"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"

"$PYTHON_BIN" -m wet_lab.prepare_swissprot \
  --output-dir "$DATABASE_DIR"

if [[ ! -s "$DATABASE_DIR/proteins_prott5_residue.h5" ]]; then
  CUDA_VISIBLE_DEVICES="$GPUS" \
    ENV_PATH="$(dirname "$(dirname "$PYTHON_BIN")")" \
    HF_HOME="$HF_HOME" \
    scripts/run_extract_prott5_residue_embeddings_4gpu.sh \
      --run-name swissprot_prott5_residue \
      --fasta "$DATABASE_DIR/proteins.fasta" \
      --output "$DATABASE_DIR/proteins_prott5_residue.h5" \
      --tmp-dir "$DATABASE_DIR/proteins_prott5_residue_shards" \
      --max-sequence-length 1024 \
      --batch-size 8 \
      --max-tokens-per-batch 4096 \
      --dtype float16 \
      --merge-order shard \
      --merge-storage virtual \
      --merged-ids-output "$DATABASE_DIR/candidate_ids_prott5_order.txt"
else
  echo "Reusing Swiss-Prot residue embeddings: $DATABASE_DIR/proteins_prott5_residue.h5"
fi

CUDA_VISIBLE_DEVICES="$QUERY_GPU" \
  "$PYTHON_BIN" -m wet_lab.query_campaign --config "$CONFIG"
