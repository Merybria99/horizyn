#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
REFSEQ_DATABASE_DIR="${REFSEQ_DATABASE_DIR:-$PROJECT_ROOT/wet_lab/databases/refseq/prokaryotes/current}"
REFSEQ_GPUS="${REFSEQ_GPUS:-1,2,3}"
REFSEQ_QUERY_GPU="${REFSEQ_QUERY_GPU:-${REFSEQ_GPUS%%,*}}"
REFSEQ_EMBEDDING_RUN_NAME="${REFSEQ_EMBEDDING_RUN_NAME:-refseq_prokaryotic_prott5_residue}"
REFSEQ_CONFIG="${REFSEQ_CONFIG:-wet_lab/configs/refseq/f4_reactzyme_reaction_smi.yaml}"
REFSEQ_DIVISIONS="${REFSEQ_DIVISIONS:-archaea,bacteria}"
REFSEQ_MAX_FILES_PER_DIVISION="${REFSEQ_MAX_FILES_PER_DIVISION:-1}"
REFSEQ_ALL_FILES="${REFSEQ_ALL_FILES:-false}"
REFSEQ_MIN_PROTEIN_LENGTH="${REFSEQ_MIN_PROTEIN_LENGTH:-50}"
REFSEQ_FORCE_DOWNLOAD="${REFSEQ_FORCE_DOWNLOAD:-false}"
REFSEQ_FORCE_NORMALIZE="${REFSEQ_FORCE_NORMALIZE:-false}"

if [[ "$REFSEQ_DATABASE_DIR" != /* ]]; then
  REFSEQ_DATABASE_DIR="$PROJECT_ROOT/$REFSEQ_DATABASE_DIR"
fi
mkdir -p "$REFSEQ_DATABASE_DIR"
cd "$PROJECT_ROOT"

PREPARE_ARGS=(
  --output-dir "$REFSEQ_DATABASE_DIR"
  --min-length "$REFSEQ_MIN_PROTEIN_LENGTH"
)
IFS=',' read -r -a REFSEQ_DIVISION_ITEMS <<< "$REFSEQ_DIVISIONS"
for division in "${REFSEQ_DIVISION_ITEMS[@]}"; do
  PREPARE_ARGS+=(--division "$division")
done
if [[ "$REFSEQ_ALL_FILES" == "true" ]]; then
  PREPARE_ARGS+=(--all-files)
else
  PREPARE_ARGS+=(--max-files-per-division "$REFSEQ_MAX_FILES_PER_DIVISION")
fi
if [[ "$REFSEQ_FORCE_DOWNLOAD" == "true" ]]; then
  PREPARE_ARGS+=(--force-download)
fi
if [[ "$REFSEQ_FORCE_NORMALIZE" == "true" ]]; then
  PREPARE_ARGS+=(--force-normalize)
fi

"$PYTHON_BIN" -m wet_lab.prepare_refseq "${PREPARE_ARGS[@]}"

RESIDUE_H5="$REFSEQ_DATABASE_DIR/proteins_prott5_residue.h5"
EMBEDDING_ORDER="$REFSEQ_DATABASE_DIR/candidate_ids_prott5_order.txt"
EMBEDDING_SOURCE_HASH="$REFSEQ_DATABASE_DIR/proteins_prott5_residue.source.sha256"
read -r CURRENT_SOURCE_HASH _ < <(sha256sum "$REFSEQ_DATABASE_DIR/candidate_ids.txt")
if [[ -s "$RESIDUE_H5" && -s "$EMBEDDING_ORDER" ]]; then
  if [[ ! -s "$EMBEDDING_SOURCE_HASH" ]]; then
    echo "RefSeq embeddings have no source fingerprint and cannot be safely reused:" >&2
    echo "  $EMBEDDING_SOURCE_HASH" >&2
    exit 1
  fi
  read -r EMBEDDED_SOURCE_HASH _ < "$EMBEDDING_SOURCE_HASH"
  if [[ "$CURRENT_SOURCE_HASH" != "$EMBEDDED_SOURCE_HASH" ]]; then
    echo "RefSeq embeddings were built from a different candidate selection." >&2
    echo "Preserve or remove the old embedding outputs explicitly before rebuilding." >&2
    exit 1
  fi
  echo "Reusing NCBI RefSeq residue embeddings: $RESIDUE_H5"
elif [[ -e "$RESIDUE_H5" || -e "$EMBEDDING_ORDER" || -e "$EMBEDDING_SOURCE_HASH" ]]; then
  echo "Incomplete RefSeq embedding output; preserve or remove it explicitly before retrying:" >&2
  echo "  $RESIDUE_H5" >&2
  echo "  $EMBEDDING_ORDER" >&2
  echo "  $EMBEDDING_SOURCE_HASH" >&2
  exit 1
else
  CUDA_VISIBLE_DEVICES="$REFSEQ_GPUS" \
    ENV_PATH="$(dirname "$(dirname "$PYTHON_BIN")")" \
    scripts/run_extract_prott5_residue_embeddings_4gpu.sh \
      --run-name "$REFSEQ_EMBEDDING_RUN_NAME" \
      --fasta "$REFSEQ_DATABASE_DIR/proteins.fasta" \
      --output "$RESIDUE_H5" \
      --tmp-dir "$REFSEQ_DATABASE_DIR/proteins_prott5_residue_shards" \
      --max-sequence-length 1024 \
      --batch-size 8 \
      --max-tokens-per-batch 4096 \
      --dtype float16 \
      --merge-order shard \
      --merge-storage virtual \
      --merged-ids-output "$EMBEDDING_ORDER"
  EMBEDDING_SOURCE_HASH_PARTIAL="$EMBEDDING_SOURCE_HASH.partial"
  printf '%s  %s\n' "$CURRENT_SOURCE_HASH" "$REFSEQ_DATABASE_DIR/candidate_ids.txt" \
    > "$EMBEDDING_SOURCE_HASH_PARTIAL"
  mv "$EMBEDDING_SOURCE_HASH_PARTIAL" "$EMBEDDING_SOURCE_HASH"
fi

CUDA_VISIBLE_DEVICES="$REFSEQ_QUERY_GPU" \
  WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE="$REFSEQ_DATABASE_DIR" \
  "$PYTHON_BIN" -m wet_lab.query --config "$REFSEQ_CONFIG"
