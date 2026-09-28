#!/usr/bin/env python3
"""Extract sequence-verified native CIRCE-v2 annotation evidence from UniProt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "horizyn/datasets/horizyn1_uniprot_annotations.py"
SPEC = importlib.util.spec_from_file_location("horizyn1_uniprot_annotations", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load annotation extractor: {MODULE_PATH}")
implementation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = implementation
SPEC.loader.exec_module(implementation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Frozen UniProt release archive or direct .dat/.dat.gz")
    parser.add_argument("--target-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Gzip TSV; one row per exact target FASTA ID")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-kind", choices=("auto", "sprot", "trembl"), default="auto")
    parser.add_argument("--release", default="2023_05")
    args = parser.parse_args()
    payload = implementation.extract_uniprot_annotations(
        input_path=args.input, target_fasta=args.target_fasta,
        output_path=args.output, manifest_path=args.manifest,
        source_kind=args.source_kind, release=args.release,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
