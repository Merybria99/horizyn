#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QUERY_FASTA="${QUERY_FASTA:-data/sleec_stage1/fasta/mcsa_reference.fasta}"
TARGET_DB="${TARGET_DB:-/datastor1/deep-proteins/databases/mmseqs2/databases/0625/UniRef50/UniRef50}"
OUT_DIR="${OUT_DIR:-data/sleec_stage1/msas/mcsa_uniref50}"
RUN_NAME="${RUN_NAME:-mcsa_uniref50}"
THREADS="${THREADS:-32}"
EXPAND_ALN="${EXPAND_ALN:-0}"
LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="$LOG_DIR/sleec_stage1_uniref50_msa_$(date +%Y%m%d_%H%M%S).log"

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

UV_CACHE_DIR="${UV_CACHE_DIR:-/datastor2/deep-proteins/EnzymeDiscovery/.uv-cache}" \
uv run --python 3.12 --group msa python -u scripts/run_ligns_msa_search.py \
  --query-fasta "$QUERY_FASTA" \
  --target-db "$TARGET_DB" \
  --out-dir "$OUT_DIR" \
  --name "$RUN_NAME" \
  --threads "$THREADS" \
  --cache \
  "${LIGNS_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

uv run --python 3.12 python scripts/prepare_sleec_stage1_data.py \
  --mcsa-source skip \
  --output-dir data/sleec_stage1 \
  --msa-dir "$OUT_DIR" \
  --msa-glob "**/*_rm_ins.a3m" \
  --pseudo-output data/sleec_stage1/msa_pseudo_residue_labels.csv \
  --pseudo-source uniref50_msa_ligns \
  --balance-pseudo \
  --manifest data/sleec_stage1/manifests/pseudo_label_manifest.json
