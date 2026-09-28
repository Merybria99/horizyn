#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/datastor2/deep-proteins/EnzymeDiscovery/env/bin/python}"
DATA_ROOT="${DATA_ROOT:-data/standardized/retrieval_training_source_collapse/train_nr90}"
GAP_DIR="${GAP_DIR:-${DATA_ROOT}/multimodal_feature_gaps/top1000}"
OUT_DIR="${OUT_DIR:-${DATA_ROOT}/multimodal_feature_gaps/extracted_top1000}"
mkdir -p "$OUT_DIR"

UNIMOL_TOOLS_PATH="${UNIMOL_TOOLS_PATH:-${ROOT}/.deps/unimol_tools}"
UNIMOL2_SITE="${UNIMOL2_SITE:-/datastor2/deep-proteins/EnzymeDiscovery/env/unimol2_site}"
if [[ -d "$UNIMOL_TOOLS_PATH" ]]; then
  export PYTHONPATH="${UNIMOL_TOOLS_PATH}:${PYTHONPATH:-}"
fi
if [[ -d "$UNIMOL2_SITE" ]]; then
  export PYTHONPATH="${UNIMOL2_SITE}:${PYTHONPATH:-}"
fi

BASE_UNIMOL2="${BASE_UNIMOL2:-${DATA_ROOT}/rxns_unimol2_84m_nr90_plus_clipzyme_eval_readable.h5}"
BASE_CHIRO="${BASE_CHIRO:-${DATA_ROOT}/rxns_chiro_256_nr90_plus_clipzyme_eval.h5}"

EXTRA_UNIMOL2="${EXTRA_UNIMOL2:-${OUT_DIR}/rxns_unimol2_84m_highpair_top1000_missing.h5}"
EXTRA_CHIRO="${EXTRA_CHIRO:-${OUT_DIR}/rxns_chiro_256_highpair_top1000_missing.h5}"
MERGED_UNIMOL2="${MERGED_UNIMOL2:-${DATA_ROOT}/rxns_unimol2_84m_nr90_plus_clipzyme_eval_readable_highpair_top1000_merged.h5}"
MERGED_CHIRO="${MERGED_CHIRO:-${DATA_ROOT}/rxns_chiro_256_nr90_plus_clipzyme_eval_highpair_top1000_merged.h5}"

echo "[$(date --iso-8601=seconds)] Extracting missing high-pair UniMol2 reactions"
"$PYTHON_BIN" scripts/extract_unimol2_reaction_embeddings.py \
  --reactions "${GAP_DIR}/missing_unimol2_reactions.csv" \
  --output "$EXTRA_UNIMOL2" \
  --allow-pseudo-reactions \
  --skip-invalid-molecules \
  --skip-invalid-reactions \
  --force

echo "[$(date --iso-8601=seconds)] Merging UniMol2 features"
"$PYTHON_BIN" scripts/merge_reaction_set_hdf5.py \
  --inputs "$BASE_UNIMOL2" "$EXTRA_UNIMOL2" \
  --output "$MERGED_UNIMOL2" \
  --force

echo "[$(date --iso-8601=seconds)] Extracting missing high-pair ChIRo reactions"
"$PYTHON_BIN" scripts/extract_chiro_reaction_embeddings.py \
  --reactions "${GAP_DIR}/missing_chiro_reactions.csv" \
  --output "$EXTRA_CHIRO" \
  --allow-pseudo-reactions \
  --device "${CHIRO_DEVICE:-cpu}" \
  --num-workers "${CHIRO_NUM_WORKERS:-4}" \
  --batch-size "${CHIRO_BATCH_SIZE:-64}" \
  --force

echo "[$(date --iso-8601=seconds)] Merging ChIRo features"
"$PYTHON_BIN" scripts/merge_reaction_set_hdf5.py \
  --inputs "$BASE_CHIRO" "$EXTRA_CHIRO" \
  --output "$MERGED_CHIRO" \
  --force

echo "[$(date --iso-8601=seconds)] Done"
echo "Merged UniMol2: $MERGED_UNIMOL2"
echo "Merged ChIRo:   $MERGED_CHIRO"
