#!/usr/bin/env bash
# Remap the existing verified native table; never rescan UniProt or start training.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/data/reconstructed/horizyn1_2023_05}"
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-${PROJECT_ROOT}/.venv-horizyn1-annotations/bin/python}"
CACHE_CHAIN="${CACHE_CHAIN:-${PROJECT_ROOT}/outputs/biofp_from_scratch_chains/biofp-fresh-chain-nohome-20260709_172734}"
CACHED_FEATURES="${CACHED_FEATURES:-${CACHE_CHAIN}/data/processed/capability_features/train_exact_rhea_reconstructed/reaction_features.parquet}"
CACHED_REACTIONS="${CACHED_REACTIONS:-${CACHE_CHAIN}/data/standardized/retrieval_training_source_collapse/train_exact/train_rxns_valid_rxn_rhea_reconstructed.csv}"
ANNOTATIONS="${RUN_ROOT}/annotations"
REACTION_DIR="${ANNOTATIONS}/cofactor_v2"
LABEL_DIR="${ANNOTATIONS}/circe_v2_cofactor_v2"
LOG_DIR="${RUN_ROOT}/logs"

log() { printf '[%s] %s\n' "$(date -Iseconds)" "$*"; }
require_file() { [[ -s "$1" ]] || { log "Missing or empty required file: $1"; exit 2; }; }
[[ $# -eq 0 ]] || { echo "Usage: $0 (paths configurable via RUN_ROOT, ANNOTATION_PYTHON, CACHED_FEATURES, CACHED_REACTIONS)" >&2; exit 2; }
for source in "${ANNOTATIONS}/native_uniprot.tsv.gz" "${ANNOTATIONS}/native_uniprot_manifest.json" \
  "${RUN_ROOT}/clustered/proteins.fasta" "${RUN_ROOT}/clustered/clusters.tsv" \
  "${RUN_ROOT}/raw/raw_pairs.tsv" "${RUN_ROOT}/raw/raw_reactions.tsv" \
  "${LOG_DIR}/clustered_integrity_audit.json" "${CACHED_FEATURES}" "${CACHED_REACTIONS}"; do
  require_file "${source}"
done
"${ANNOTATION_PYTHON}" -c 'import numpy, pandas, pyarrow'
mkdir -p "${REACTION_DIR}" "${LABEL_DIR}" "${LOG_DIR}"
exec 9>"${LOG_DIR}/cofactor_v2_annotation_pipeline.lock"
flock -n 9 || { log "Another cofactor-v2 pipeline holds the lock."; exit 3; }

# Cheap freshness gates before the long export; full checksums and row replay
# are verified independently at the end. The archive is not decompressed.
"${ANNOTATION_PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
def signature(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
graph = json.loads((root / 'logs/clustered_integrity_audit.json').read_text())
if graph.get('status') != 'passed' or graph.get('stage') != 'clustered' or graph.get('errors'):
    raise SystemExit('The full clustered graph audit must pass first.')
snapshots = graph.get('artifact_signatures', {})
if not snapshots or any(signature(Path(path)) != expected for path, expected in snapshots.items()):
    raise SystemExit('Clustered graph audit is stale or lacks signatures; rerun it first.')
native = json.loads((root / 'annotations/native_uniprot_manifest.json').read_text())
path = root / 'annotations/native_uniprot.tsv.gz'
keys = ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')
if (native.get('status') != 'complete'
    or native.get('schema_version') != 'horizyn1_native_uniprot_annotations_v1'
    or native.get('output', {}).get('path') != str(path)
    or native.get('output', {}).get('signature') != dict(zip(keys, signature(path)))):
    raise SystemExit('The completed native annotation table is missing or has changed.')
if path.with_name(path.name + '.partial').exists():
    raise SystemExit('Native extraction is still in progress.')
PY

log "Building cofactor-v2 exact-chemistry lookup; legacy artifacts remain untouched."
"${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/build_horizyn1_reaction_annotations_v2.py" \
  --raw-reactions "${RUN_ROOT}/raw/raw_reactions.tsv" \
  --cached-features "${CACHED_FEATURES}" --cached-reactions "${CACHED_REACTIONS}" \
  --output "${REACTION_DIR}/reaction_annotations.json" \
  --manifest "${REACTION_DIR}/reaction_annotations_manifest.json"

log "Exporting 31 cofactor groups plus unknown, separately by evidence source."
nice -n 10 "${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/build_horizyn1_circe_v2_labels_v2.py" \
  --representative-fasta "${RUN_ROOT}/clustered/proteins.fasta" \
  --cluster-map "${RUN_ROOT}/clustered/clusters.tsv" \
  --native-annotations "${ANNOTATIONS}/native_uniprot.tsv.gz" \
  --native-manifest "${ANNOTATIONS}/native_uniprot_manifest.json" \
  --reaction-annotations "${REACTION_DIR}/reaction_annotations.json" \
  --reaction-manifest "${REACTION_DIR}/reaction_annotations_manifest.json" \
  --association-pairs "${RUN_ROOT}/raw/raw_pairs.tsv" --pair-scope unsplit_inventory \
  --output-dir "${LABEL_DIR}"

log "Replaying all source labels independently, including unknown exclusivity and known-only coverage."
nice -n 10 "${ANNOTATION_PYTHON}" "${PROJECT_ROOT}/scripts/audit_horizyn1_circe_v2_labels_v2.py" \
  --run-root "${RUN_ROOT}" --label-dir "${LABEL_DIR}" \
  --output "${LOG_DIR}/circe_v2_cofactor_v2_label_integrity_audit.json"
log "Cofactor-v2 export and independent audit passed. Inventory scope only; no training started."
