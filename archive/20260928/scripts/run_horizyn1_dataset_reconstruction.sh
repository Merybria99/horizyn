#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/data/reconstructed/horizyn1_2023_05}"
DOWNLOAD_DIR="${RUN_ROOT}/downloads"
SOURCE_DIR="${RUN_ROOT}/sources"
RAW_DIR="${RUN_ROOT}/raw"
CLUSTERED_DIR="${RUN_ROOT}/clustered"
LOG_DIR="${RUN_ROOT}/logs"
THREADS="${THREADS:-64}"
SORT_THREADS="${SORT_THREADS:-16}"
RESERVE_TIB="${RESERVE_TIB:-5}"
FORCE="${FORCE:-0}"

UNIPROT_URL="https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2023_05/knowledgebase/knowledgebase2023_05.tar.gz"
UNIPROT_MD5="8d2f41981c4cfbdaf1c9b8932dfe6e91"
UNIPROT_BYTES="186243013906"
RHEA_URL="https://ftp.expasy.org/databases/rhea/old_releases/131.tar.bz2"
RHEA_BYTES="343514968"
UNIPROT_ARCHIVE="${DOWNLOAD_DIR}/knowledgebase2023_05.tar.gz"
RHEA_ARCHIVE="${DOWNLOAD_DIR}/rhea_release_131.tar.bz2"

DEV_FASTA="${PROJECT_ROOT}/data/sota/prots.fasta"
DEV_TRAIN_PAIRS="${PROJECT_ROOT}/data/train_pairs.csv"
DEV_TEST_PAIRS="${PROJECT_ROOT}/data/test_pairs.csv"
DEV_TRAIN_REACTIONS="${PROJECT_ROOT}/data/train_rxns.csv"
DEV_TEST_REACTIONS="${PROJECT_ROOT}/data/test_rxns.csv"
ENZYMEMAP_PROCESSED="${ENZYMEMAP_PROCESSED:-${PROJECT_ROOT}/../sources/enzymemap/data/processed_reactions.csv.gz}"
MMSEQS="${MMSEQS:-${PROJECT_ROOT}/tools/mmseqs/bin/mmseqs}"
PYTHON="${PYTHON:-python3}"
CLI="${PROJECT_ROOT}/scripts/reconstruct_horizyn1_dataset.py"

mkdir -p "${DOWNLOAD_DIR}" "${SOURCE_DIR}" "${RAW_DIR}" "${CLUSTERED_DIR}" "${LOG_DIR}"

run_preflight() {
  "${PYTHON}" "${CLI}" preflight \
    --path "${RUN_ROOT}" \
    --reserve-tib "${RESERVE_TIB}" \
    --manifest "${RUN_ROOT}/preflight.json"
}

download_one() {
  local url="$1"
  local destination="$2"
  local expected_md5="${3:-}"
  local expected_bytes="${4:-}"
  local partial="${destination}.clean.part"
  if [[ -s "${destination}" ]]; then
    if [[ -n "${expected_bytes}" && "$(stat -c '%s' "${destination}")" != "${expected_bytes}" ]]; then
      echo "Existing file has the wrong size: ${destination}" >&2
      echo "Expected ${expected_bytes} bytes; found $(stat -c '%s' "${destination}") bytes." >&2
      exit 3
    fi
    if [[ -n "${expected_md5}" ]]; then
      echo "${expected_md5}  ${destination}" | md5sum --check -
    fi
    echo "Using verified ${destination}"
    return
  fi
  if [[ -s "${partial}" && -n "${expected_bytes}" ]] && \
     (( $(stat -c '%s' "${partial}") > expected_bytes )); then
    echo "Partial file is larger than its authoritative size: ${partial}" >&2
    exit 3
  fi
  wget --continue --tries=0 --timeout=60 --waitretry=5 --progress=bar:force:noscroll \
    --output-document="${partial}" "${url}"
  if [[ -n "${expected_bytes}" && "$(stat -c '%s' "${partial}")" != "${expected_bytes}" ]]; then
    echo "Downloaded file has the wrong size: ${partial}" >&2
    echo "Expected ${expected_bytes} bytes; found $(stat -c '%s' "${partial}") bytes." >&2
    exit 3
  fi
  if [[ -n "${expected_md5}" ]]; then
    echo "${expected_md5}  ${partial}" | md5sum --check -
  fi
  mv "${partial}" "${destination}"
}

run_download() {
  run_preflight
  download_one "${RHEA_URL}" "${RHEA_ARCHIVE}" "" "${RHEA_BYTES}"
  download_one "${UNIPROT_URL}" "${UNIPROT_ARCHIVE}" "${UNIPROT_MD5}" "${UNIPROT_BYTES}"
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "Missing required file: $1" >&2
    echo "Run '$0 download' first, or place the historical archive at that path." >&2
    exit 2
  fi
}

run_prepare() {
  require_file "${RHEA_ARCHIVE}"
  if [[ "${FORCE}" == "1" || ! -s "${SOURCE_DIR}/enzymemap_prepare.json" ]]; then
    "${PYTHON}" "${CLI}" prepare-enzymemap \
      --processed "${ENZYMEMAP_PROCESSED}" \
      --pairs "${SOURCE_DIR}/enzymemap_raw_pairs.tsv.gz" \
      --reactions "${SOURCE_DIR}/enzymemap_reactions.tsv.gz" \
      --wanted-accessions "${SOURCE_DIR}/enzymemap_wanted_accessions.txt" \
      --manifest "${SOURCE_DIR}/enzymemap_prepare.json"
  else
    echo "Skipping completed EnzymeMap preparation"
  fi
  if [[ "${FORCE}" == "1" || ! -s "${SOURCE_DIR}/rhea_direction_map.json" ]]; then
    "${PYTHON}" "${CLI}" rhea-map \
      --input "${RHEA_ARCHIVE}" \
      --output "${SOURCE_DIR}/rhea_direction_to_master.tsv" \
      --manifest "${SOURCE_DIR}/rhea_direction_map.json"
  else
    echo "Skipping completed Rhea mapping"
  fi
}

run_parse() {
  require_file "${UNIPROT_ARCHIVE}"
  run_prepare
  if [[ "${FORCE}" == "1" || ! -s "${SOURCE_DIR}/uniprot_parse.json" ]]; then
    "${PYTHON}" "${CLI}" parse-uniprot \
      --input "${UNIPROT_ARCHIVE}" \
      --rhea-map "${SOURCE_DIR}/rhea_direction_to_master.tsv" \
      --development-reactions "${DEV_TRAIN_REACTIONS}" "${DEV_TEST_REACTIONS}" \
      --wanted-accessions "${SOURCE_DIR}/enzymemap_wanted_accessions.txt" \
      --existing-fasta "${DEV_FASTA}" \
      --pairs "${SOURCE_DIR}/trembl_pairs.tsv.gz" \
      --selected-fasta "${SOURCE_DIR}/trembl_selected.fasta.gz" \
      --extra-fasta "${SOURCE_DIR}/enzymemap_extra.fasta.gz" \
      --accession-map "${SOURCE_DIR}/enzymemap_accession_map.tsv" \
      --manifest "${SOURCE_DIR}/uniprot_parse.json"
  else
    echo "Skipping completed UniProt parsing"
  fi
  if [[ "${FORCE}" == "1" || ! -s "${SOURCE_DIR}/enzymemap_finalize.json" ]]; then
    "${PYTHON}" "${CLI}" finalize-enzymemap \
      --raw-pairs "${SOURCE_DIR}/enzymemap_raw_pairs.tsv.gz" \
      --accession-map "${SOURCE_DIR}/enzymemap_accession_map.tsv" \
      --existing-fasta "${DEV_FASTA}" "${SOURCE_DIR}/trembl_selected.fasta.gz" "${SOURCE_DIR}/enzymemap_extra.fasta.gz" \
      --output "${SOURCE_DIR}/enzymemap_pairs.tsv.gz" \
      --manifest "${SOURCE_DIR}/enzymemap_finalize.json"
  else
    echo "Skipping completed EnzymeMap accession resolution"
  fi
}

run_merge() {
  run_parse
  if [[ "${FORCE}" == "1" || ! -s "${RAW_DIR}/raw_manifest.json" ]]; then
    "${PYTHON}" "${CLI}" merge \
      --development-pairs "${DEV_TRAIN_PAIRS}" "${DEV_TEST_PAIRS}" \
      --development-reactions "${DEV_TRAIN_REACTIONS}" "${DEV_TEST_REACTIONS}" \
      --fasta "${DEV_FASTA}" "${SOURCE_DIR}/trembl_selected.fasta.gz" "${SOURCE_DIR}/enzymemap_extra.fasta.gz" \
      --trembl-pairs "${SOURCE_DIR}/trembl_pairs.tsv.gz" \
      --enzymemap-pairs "${SOURCE_DIR}/enzymemap_pairs.tsv.gz" \
      --enzymemap-reactions "${SOURCE_DIR}/enzymemap_reactions.tsv.gz" \
      --output-dir "${RAW_DIR}" \
      --sort-threads "${SORT_THREADS}"
  else
    echo "Skipping completed raw-source merge"
  fi
}

run_cluster() {
  run_merge
  if [[ "${FORCE}" == "1" || ! -s "${CLUSTERED_DIR}/clustered_manifest.json" ]]; then
    "${PYTHON}" "${CLI}" cluster \
      --raw-fasta "${RAW_DIR}/raw_proteins.fasta" \
      --raw-pairs "${RAW_DIR}/raw_pairs.tsv" \
      --output-dir "${CLUSTERED_DIR}" \
      --mmseqs "${MMSEQS}" \
      --threads "${THREADS}" \
      --min-seq-id 0.8
  else
    echo "Skipping completed MMseqs clustering and pair collapse"
  fi
}

ACTION="${1:-preflight}"
case "${ACTION}" in
  preflight) run_preflight ;;
  download) run_download ;;
  prepare) run_prepare ;;
  parse) run_parse ;;
  merge) run_merge ;;
  cluster) run_cluster ;;
  all) run_download; run_cluster ;;
  *)
    echo "Usage: $0 {preflight|download|prepare|parse|merge|cluster|all}" >&2
    exit 2
    ;;
esac
