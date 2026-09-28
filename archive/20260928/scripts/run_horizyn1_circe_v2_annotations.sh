#!/usr/bin/env bash
# One-off, sequence-verified native labels followed by conservative CIRCE-v2 exports.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/data/reconstructed/horizyn1_2023_05}"
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-${PROJECT_ROOT}/.venv-horizyn1-annotations/bin/python}"
NATIVE_PYTHON="${NATIVE_PYTHON:-/usr/bin/python3}"
CACHE_CHAIN="${CACHE_CHAIN:-${PROJECT_ROOT}/outputs/biofp_from_scratch_chains/biofp-fresh-chain-nohome-20260709_172734}"
CACHED_FEATURES="${CACHED_FEATURES:-${CACHE_CHAIN}/data/processed/capability_features/train_exact_rhea_reconstructed/reaction_features.parquet}"
CACHED_REACTIONS="${CACHED_REACTIONS:-${CACHE_CHAIN}/data/standardized/retrieval_training_source_collapse/train_exact/train_rxns_valid_rxn_rhea_reconstructed.csv}"
ANNOTATIONS="${RUN_ROOT}/annotations"
LABEL_DIR="${ANNOTATIONS}/circe_v2"
LOG_DIR="${RUN_ROOT}/logs"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
require_file() { [[ -s "$1" ]] || { log "Missing or empty required file: $1"; exit 2; }; }
require_passed_audit() {
  require_file "$1"
  jq -e '.status == "passed" and (.errors | length == 0)' "$1" >/dev/null || {
    log "Required graph integrity audit has not passed: $1"; exit 2;
  }
}

ACTION="${1:-all}"
case "$ACTION" in extract|export|all) ;; *) echo "Usage: $0 {extract|export|all}" >&2; exit 2 ;; esac
require_file "${RUN_ROOT}/raw/raw_manifest.json"
require_passed_audit "${LOG_DIR}/raw_integrity_audit.json"
require_file "${RUN_ROOT}/downloads/knowledgebase2023_05.tar.gz"
require_file "${RUN_ROOT}/raw/raw_proteins.fasta"
mkdir -p "${ANNOTATIONS}" "${LOG_DIR}"
exec 9>"${LOG_DIR}/circe_v2_annotation_pipeline.lock"
flock -n 9 || { log "Another annotation pipeline holds the lock; not starting a second writer."; exit 3; }

extract_native() {
  log "Extracting native EC/cofactor annotations from UniProtKB 2023_05; one row per raw sequence."
  nice -n 10 "${NATIVE_PYTHON}" "${PROJECT_ROOT}/scripts/extract_horizyn1_uniprot_annotations.py" \
    --input "${RUN_ROOT}/downloads/knowledgebase2023_05.tar.gz" \
    --target-fasta "${RUN_ROOT}/raw/raw_proteins.fasta" \
    --output "${ANNOTATIONS}/native_uniprot.tsv.gz" \
    --manifest "${ANNOTATIONS}/native_uniprot_manifest.json" \
    --source-kind auto --release 2023_05
}

export_labels() {
  require_passed_audit "${LOG_DIR}/clustered_integrity_audit.json"
  require_file "${RUN_ROOT}/clustered/clustered_manifest.json"
  require_file "${ANNOTATIONS}/native_uniprot_manifest.json"
  require_file "${ANNOTATIONS}/native_uniprot.tsv.gz"
  require_file "${CACHED_FEATURES}"
  require_file "${CACHED_REACTIONS}"
  "${ANNOTATION_PYTHON}" -c 'import numpy, pandas, pyarrow, yaml'
  log "Reusing cached reaction descriptors only for exact chemistry matches."
  "${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/build_horizyn1_reaction_annotations.py" \
    --raw-reactions "${RUN_ROOT}/raw/raw_reactions.tsv" \
    --cached-features "${CACHED_FEATURES}" --cached-reactions "${CACHED_REACTIONS}" \
    --output "${ANNOTATIONS}/reaction_annotations.json" \
    --manifest "${ANNOTATIONS}/reaction_annotations_manifest.json"
  log "Exporting representative labels; unsplit inventory profiles are not held-out-safe training targets."
  nice -n 10 "${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/build_horizyn1_circe_v2_labels.py" \
    --representative-fasta "${RUN_ROOT}/clustered/proteins.fasta" \
    --cluster-map "${RUN_ROOT}/clustered/clusters.tsv" \
    --native-annotations "${ANNOTATIONS}/native_uniprot.tsv.gz" \
    --native-manifest "${ANNOTATIONS}/native_uniprot_manifest.json" \
    --reaction-annotations "${ANNOTATIONS}/reaction_annotations.json" \
    --reaction-manifest "${ANNOTATIONS}/reaction_annotations_manifest.json" \
    --association-pairs "${RUN_ROOT}/raw/raw_pairs.tsv" --pair-scope unsplit_inventory \
    --output-dir "${LABEL_DIR}"
  log "Independently auditing every representative's labels and their provenance."
  nice -n 10 "${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/audit_horizyn1_circe_v2_labels.py" \
    --run-root "${RUN_ROOT}" --label-dir "${LABEL_DIR}" \
    --output "${LOG_DIR}/circe_v2_label_integrity_audit.json"
  log "CIRCE-v2 annotation export and full label integrity audit passed."
}

log "Starting CIRCE-v2 annotation stage: ${ACTION}"
case "$ACTION" in
  extract) extract_native ;;
  export) export_labels ;;
  all) extract_native; export_labels ;;
esac
