#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUERY_FASTA="${QUERY_FASTA:-data/sleec_stage1/fasta/mcsa_reference.fasta}"
TARGET_DB="${TARGET_DB:-/datastor2/deep-proteins/databases/mmseqs2/uniref_indexed/databases/0625/UniRef90/UniRef90}"
OUT_DIR="${OUT_DIR:-data/sleec_stage1/msas/mcsa_uniref90}"
RUN_NAME="${RUN_NAME:-mcsa_uniref90}"
THREADS="${THREADS:-32}"
EXPAND_ALN="${EXPAND_ALN:-0}"
LOG_DIR="${LOG_DIR:-logs/data_prep}"
PSEUDO_OUTPUT="${PSEUDO_OUTPUT:-data/sleec_stage1/msa_pseudo_residue_labels_uniref90.csv}"
MANIFEST="${MANIFEST:-data/sleec_stage1/manifests/pseudo_label_manifest_uniref90.json}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/datastor2/deep-proteins/EnzymeDiscovery/.uv-env-msa}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/datastor2/deep-proteins/EnzymeDiscovery/.uv-cache}"
LOG_FILE="$LOG_DIR/sleec_stage1_uniref90_msa_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if [[ ! -f "$QUERY_FASTA" ]]; then
  echo "Missing query FASTA: $QUERY_FASTA" >&2
  exit 1
fi

if [[ ! -f "$TARGET_DB" ]]; then
  echo "Missing MMseqs target DB prefix: $TARGET_DB" >&2
  exit 1
fi

LIGNS_ARGS=()
if [[ "$EXPAND_ALN" == "0" || "$EXPAND_ALN" == "false" ]]; then
  LIGNS_ARGS+=(--dont-expand-aln)
fi

UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" UV_CACHE_DIR="$UV_CACHE_DIR" \
  uv run --python 3.12 --group msa python -u scripts/run_ligns_msa_search.py \
  --query-fasta "$QUERY_FASTA" \
  --target-db "$TARGET_DB" \
  --out-dir "$OUT_DIR" \
  --name "$RUN_NAME" \
  --threads "$THREADS" \
  --cache \
  "${LIGNS_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

UV_PROJECT_ENVIRONMENT="$UV_PROJECT_ENVIRONMENT" UV_CACHE_DIR="$UV_CACHE_DIR" \
  uv run --python 3.12 --group msa python scripts/prepare_sleec_stage1_data.py \
  --mcsa-source skip \
  --output-dir data/sleec_stage1 \
  --msa-dir "$OUT_DIR" \
  --msa-glob "**/*_rm_ins.a3m" \
  --pseudo-output "$PSEUDO_OUTPUT" \
  --pseudo-source uniref90_msa_ligns \
  --balance-pseudo \
  --manifest "$MANIFEST" 2>&1 | tee -a "$LOG_FILE"
