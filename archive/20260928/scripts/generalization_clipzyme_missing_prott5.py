#!/usr/bin/env python3
"""List EnzymeMap screening sequences that lack reusable ProtT5 residues."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def fasta(path: Path):
    key = None
    sequence = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if key is not None:
                    yield key, "".join(sequence)
                key = line[1:]
                sequence = []
            elif key:
                sequence.append(line)
            else:
                raise ValueError("FASTA starts without an identifier")
    if key is not None:
        yield key, "".join(sequence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prepared = json.loads((args.catalog / "preparation.json").read_text())
    if prepared["candidate_ids"] != 261907 or prepared["candidate_ids_without_sequence"]:
        raise ValueError("Expected complete paper-matched screening pool")
    source = args.catalog / "screening_proteins.fasta"
    if digest(source) != prepared["screening_fasta_sha256"]:
        raise ValueError("Screening sequences changed")
    with h5py.File(args.cache) as handle:
        cached = set(handle["ids"].asstr()[:])
    args.output.mkdir(parents=True)
    missing, reused = 0, 0
    with (args.output / "missing_proteins.fasta").open("w") as output:
        for key, sequence in fasta(source):
            if not sequence:
                raise ValueError("Empty screening sequence")
            if key in cached:
                reused += 1
            else:
                output.write(f">{key}\n{sequence}\n")
                missing += 1
    if missing + reused != prepared["unique_screening_sequences"]:
        raise ValueError("Screening sequence count changed")
    report = {"schema": "clipzyme_f3_missing_prott5_v1",
              "catalog_preparation_sha256": digest(args.catalog / "preparation.json"),
              "cache_path": str(args.cache),
              "cache_container_sha256": digest(args.cache),
              "screening_unique_sequences": missing + reused,
              "cached_unique_sequences": reused,
              "missing_unique_sequences": missing,
              "missing_fasta_sha256": digest(args.output / "missing_proteins.fasta"),
              "source_sha256": digest(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("screening_unique_sequences", "cached_unique_sequences",
                       "missing_unique_sequences")}), flush=True)


if __name__ == "__main__":
    main()
