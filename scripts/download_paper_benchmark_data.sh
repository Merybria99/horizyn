#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/datastor2/deep-proteins/EnzymeDiscovery/horizyn}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/paper}"

REACTZYME_RECORD="https://zenodo.org/records/11494913/files"
CLIPZYME_RECORD="https://zenodo.org/records/11187895/files"

download_file() {
  local url="$1"
  local output="$2"
  local expected_md5="$3"

  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" ]]; then
    echo "Found existing file: $output"
  else
    echo "Downloading: $url"
    curl -L --fail --retry 5 --retry-delay 10 --continue-at - \
      --output "$output" \
      "$url"
  fi

  if [[ -n "$expected_md5" ]]; then
    echo "${expected_md5}  ${output}" | md5sum -c -
  fi
}

extract_zip() {
  local zip_path="$1"
  local output_dir="$2"
  mkdir -p "$output_dir"
  echo "Extracting: $zip_path -> $output_dir"
  unzip -n "$zip_path" -d "$output_dir"
}

main() {
  cd "$PROJECT_ROOT"
  mkdir -p \
    "$DATA_ROOT/reactzyme/raw" \
    "$DATA_ROOT/reactzyme/processed" \
    "$DATA_ROOT/clipzyme/raw" \
    "$DATA_ROOT/clipzyme/files" \
    "$DATA_ROOT/manifests"

  cat > "$DATA_ROOT/manifests/SOURCES.md" <<'EOF'
# Paper Benchmark Data Sources

Downloaded for reproducing experimental settings, not the paper method.

## ReactZyme

Official ReactZyme Zenodo record:
https://zenodo.org/records/11494913

Files are placed under:
`data/paper/reactzyme/raw`

Split archives are extracted under:
`data/paper/reactzyme/processed`

## CLIPZyme / EnzymeMap Screening

Official CLIPZyme Zenodo record:
https://zenodo.org/records/11187895

The data archive contains:
- `enzymemap.json`
- `cached_enzymemap.p`
- `clipzyme_screening_set.p`
- `uniprot2sequence.p`

Files are extracted under:
`data/paper/clipzyme/files`
EOF

  download_file "$REACTZYME_RECORD/cleaned_uniprot_rhea.tsv" \
    "$DATA_ROOT/reactzyme/raw/cleaned_uniprot_rhea.tsv" \
    "669bdd627c946114e87f06bffb4f33d9"
  download_file "$REACTZYME_RECORD/uniprot_molecules.tsv" \
    "$DATA_ROOT/reactzyme/raw/uniprot_molecules.tsv" \
    "a669647f418bf54dc7c5d0059c2b2a09"
  download_file "$REACTZYME_RECORD/uniprot_rhea.tsv" \
    "$DATA_ROOT/reactzyme/raw/uniprot_rhea.tsv" \
    "5b9d384c96a597680b140c3a333f1600"
  download_file "$REACTZYME_RECORD/rhea_molecules.tsv" \
    "$DATA_ROOT/reactzyme/raw/rhea_molecules.tsv" \
    "cb5a575a08954f6d28311b9a4bef52fe"
  download_file "$REACTZYME_RECORD/deepchem_vocab.txt" \
    "$DATA_ROOT/reactzyme/raw/deepchem_vocab.txt" \
    "95ca5f1d57a4a7a82bb3cca0ad742e9c"
  download_file "$REACTZYME_RECORD/molecule_vocab.pkl" \
    "$DATA_ROOT/reactzyme/raw/molecule_vocab.pkl" \
    "5a64bef090335f884a767006867d64cf"
  download_file "$REACTZYME_RECORD/time_split.zip" \
    "$DATA_ROOT/reactzyme/raw/time_split.zip" \
    "c437435a239326c157e1d20f00d8e00e"
  download_file "$REACTZYME_RECORD/enzyme_smi_split.zip" \
    "$DATA_ROOT/reactzyme/raw/enzyme_smi_split.zip" \
    "e351fdb85830968fc9abe933c39f9eda"
  download_file "$REACTZYME_RECORD/reaction_smi_split.zip" \
    "$DATA_ROOT/reactzyme/raw/reaction_smi_split.zip" \
    "2d9f4e6c78d8daf5752cc2a5ae2bef0d"

  extract_zip "$DATA_ROOT/reactzyme/raw/time_split.zip" \
    "$DATA_ROOT/reactzyme/processed/time"
  extract_zip "$DATA_ROOT/reactzyme/raw/enzyme_smi_split.zip" \
    "$DATA_ROOT/reactzyme/processed/enzyme_smi"
  extract_zip "$DATA_ROOT/reactzyme/raw/reaction_smi_split.zip" \
    "$DATA_ROOT/reactzyme/processed/reaction_smi"

  download_file "$CLIPZYME_RECORD/clipzyme_data.zip" \
    "$DATA_ROOT/clipzyme/raw/clipzyme_data.zip" \
    "1ebd955e83fa480aea198c20c1a66381"
  extract_zip "$DATA_ROOT/clipzyme/raw/clipzyme_data.zip" \
    "$DATA_ROOT/clipzyme/files"

  echo
  echo "Paper benchmark data downloaded into: $DATA_ROOT"
  find "$DATA_ROOT" -maxdepth 4 -type f -printf "%p\t%s bytes\n" | sort > \
    "$DATA_ROOT/manifests/file_manifest.tsv"
  echo "Manifest: $DATA_ROOT/manifests/file_manifest.tsv"
}

main "$@"
