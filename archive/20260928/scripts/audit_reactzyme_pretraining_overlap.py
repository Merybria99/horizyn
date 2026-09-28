#!/usr/bin/env python3
"""Audit measurable ReactZyme test overlap with SLEEC and ReactionT5 sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


MCSA_ENTRIES_URL = "https://www.ebi.ac.uk/thornton-srv/m-csa/api/entries/?format=json"
UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
REACTIONT5_MODEL_CARD = "https://huggingface.co/sagawa/ReactionT5v2-forward"
REACTIONT5_PAPER = "https://link.springer.com/article/10.1186/s13321-025-01075-4"
SPLITS = ("time", "enzyme_smi", "reaction_smi")
MAX_SEQUENCE_LENGTH = 1022
AMINO_ACID_PATTERN = re.compile(r"[^ACDEFGHIKLMNPQRSTVWYX]")


def _request(url: str, *, retries: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Horizyn-overlap-audit/1.0"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _fetch_mcsa_entries(cache_path: Path, *, refresh: bool) -> list[dict[str, Any]]:
    if cache_path.is_file() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    next_url: str | None = MCSA_ENTRIES_URL
    while next_url:
        payload = json.loads(_request(next_url))
        records.extend(payload.get("results", []))
        next_url = payload.get("next")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def _mcsa_accessions(entries: list[dict[str, Any]]) -> set[str]:
    accessions = set()

    def add_accessions(raw_value: Any) -> None:
        # Some M-CSA records describe multi-chain complexes in one comma-
        # separated field even though the API property is named in singular.
        for accession in re.split(r"[,;\s]+", str(raw_value or "").strip()):
            if accession:
                accessions.add(accession)

    for entry in entries:
        add_accessions(entry.get("reference_uniprot_id"))
        for sequence in (entry.get("protein") or {}).get("sequences", []):
            add_accessions(sequence.get("uniprot_id"))
    return accessions


def _parse_fasta(text: str) -> dict[str, str]:
    records: dict[str, list[str]] = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            token = line[1:].split()[0]
            pieces = token.split("|")
            current = pieces[1] if len(pieces) >= 3 else token
            records.setdefault(current, [])
        elif current is not None:
            records[current].append(line)
    return {accession: "".join(lines) for accession, lines in records.items()}


def _fetch_uniprot_sequences(
    accessions: set[str],
    cache_path: Path,
    *,
    refresh: bool,
    batch_size: int = 40,
) -> dict[str, str]:
    cached: dict[str, str] = {}
    if cache_path.is_file() and not refresh:
        with cache_path.open(newline="", encoding="utf-8") as handle:
            cached = {row["accession"]: row["sequence"] for row in csv.DictReader(handle, delimiter="\t")}
    missing = sorted(accessions - set(cached))
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        query = "(" + " OR ".join(f"accession:{accession}" for accession in batch) + ")"
        url = UNIPROT_STREAM_URL + "?" + urllib.parse.urlencode({"format": "fasta", "query": query})
        cached.update(_parse_fasta(_request(url).decode("utf-8")))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("accession", "sequence"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for accession, sequence in sorted(cached.items()):
            writer.writerow({"accession": accession, "sequence": sequence})
    return {accession: cached[accession] for accession in accessions if accession in cached}


def normalize_sequence(sequence: str) -> str:
    normalized = re.sub(r"\s+", "", sequence.upper())
    normalized = re.sub(r"[UZOB]", "X", normalized)
    return AMINO_ACID_PATTERN.sub("X", normalized)


def truncate_ends_center(sequence: str, max_length: int = MAX_SEQUENCE_LENGTH) -> str:
    if len(sequence) <= max_length:
        return sequence
    first_count = max_length // 4
    last_count = max_length // 4
    middle_count = max_length - first_count - last_count
    middle_start = max((len(sequence) - middle_count) // 2, first_count)
    middle_end = min(middle_start + middle_count, len(sequence) - last_count)
    middle_start = max(middle_end - middle_count, first_count)
    return sequence[:first_count] + sequence[middle_start:middle_end] + sequence[-last_count:]


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def _load_test_sequences(path: Path) -> tuple[dict[str, str], list[str]]:
    by_protein: dict[str, str] = {}
    pair_sequences = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"protein_id", "protein_sequence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} is missing {sorted(required)}")
        for row in reader:
            sequence = normalize_sequence(row["protein_sequence"])
            protein_id = row["protein_id"]
            previous = by_protein.setdefault(protein_id, sequence)
            if previous != sequence:
                raise ValueError(f"Protein ID maps to multiple sequences in {path}: {protein_id}")
            pair_sequences.append(sequence)
    return by_protein, pair_sequences


def _historical_artifacts(root: Path) -> list[dict[str, Any]]:
    paths = (
        "data/SLEEC/catalytic_residues_homologues.json",
        "data/SLEEC/curated_data.csv",
        "data/sleec_stage1/fasta/mcsa_reference.full.fasta",
        "data/sleec_stage1/fasta/mcsa_reference.fasta",
        "data/sleec_stage1/mcsa_residue_labels.csv",
        "data/sleec_stage1/msa_pseudo_residue_labels_uniref90.balanced.csv",
    )
    return [{"path": str((root / path).resolve()), "exists": (root / path).exists()} for path in paths]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    entries = _fetch_mcsa_entries(args.cache_dir / "mcsa_entries_current.json", refresh=args.refresh)
    accessions = _mcsa_accessions(entries)
    mcsa_sequences_by_accession = _fetch_uniprot_sequences(
        accessions,
        args.cache_dir / "mcsa_uniprot_sequences_current.tsv",
        refresh=args.refresh,
    )
    mcsa_full = {
        _sequence_hash(normalize_sequence(sequence)): accession
        for accession, sequence in mcsa_sequences_by_accession.items()
    }
    mcsa_model_view = {
        _sequence_hash(truncate_ends_center(normalize_sequence(sequence))): accession
        for accession, sequence in mcsa_sequences_by_accession.items()
    }

    split_results = {}
    for split in SPLITS:
        test_path = args.protocol_root / split / "test_pairs.csv"
        proteins, pair_sequences = _load_test_sequences(test_path)
        full_matches = {
            protein_id: mcsa_full[_sequence_hash(sequence)]
            for protein_id, sequence in proteins.items()
            if _sequence_hash(sequence) in mcsa_full
        }
        model_view_matches = {
            protein_id: mcsa_model_view[_sequence_hash(truncate_ends_center(sequence))]
            for protein_id, sequence in proteins.items()
            if _sequence_hash(truncate_ends_center(sequence)) in mcsa_model_view
        }
        full_pair_matches = sum(_sequence_hash(sequence) in mcsa_full for sequence in pair_sequences)
        model_view_pair_matches = sum(
            _sequence_hash(truncate_ends_center(sequence)) in mcsa_model_view
            for sequence in pair_sequences
        )
        split_results[split] = {
            "test_pairs": len(pair_sequences),
            "unique_test_protein_ids": len(proteins),
            "unique_test_sequences": len(set(proteins.values())),
            "current_mcsa_proxy": {
                "full_sequence_exact_unique_proteins": len(full_matches),
                "full_sequence_exact_pairs": full_pair_matches,
                "model_view_exact_unique_proteins": len(model_view_matches),
                "model_view_exact_pairs": model_view_pair_matches,
                "model_view_fraction": len(model_view_matches) / len(proteins) if proteins else 0.0,
                "matches": [
                    {"reactzyme_protein_id": protein_id, "mcsa_uniprot_accession": accession}
                    for protein_id, accession in sorted(model_view_matches.items())
                ],
            },
        }

    artifacts = _historical_artifacts(args.root)
    historical_available = all(item["exists"] for item in artifacts)
    return {
        "schema_version": "reactzyme_pretraining_overlap_audit_v1",
        "scope": "exact sequence/reaction exposure; homology is not estimated",
        "sleec": {
            "checkpoint_manifest": str(args.sleec_manifest.resolve()),
            "checkpoint_manifest_exists": args.sleec_manifest.is_file(),
            "historical_source_artifacts_available": historical_available,
            "historical_source_artifacts": artifacts,
            "historical_exact_overlap_status": (
                "measurable" if historical_available else "not_reconstructable_from_retained_artifacts"
            ),
            "current_mcsa_snapshot_proxy": {
                "snapshot_url": MCSA_ENTRIES_URL,
                "entries": len(entries),
                "accessions": len(accessions),
                "sequences_retrieved": len(mcsa_sequences_by_accession),
                "note": "Current M-CSA/UniProt records are a proxy, not proof of the deleted historical training snapshot.",
            },
            "splits": split_results,
            "uniref90_role": (
                "UniRef90 supplied MSA homolog context for pseudo-labels on the M-CSA query proteins; "
                "the local pipeline did not turn every UniRef90 hit into a separate supervised protein."
            ),
        },
        "reactiont5v2": {
            "model_card": REACTIONT5_MODEL_CARD,
            "paper": REACTIONT5_PAPER,
            "reported_training_source": "Open Reaction Database (ORD), with CompoundT5/ZINC initialization",
            "expected_author_artifact": "preprocessed_ord_train.csv",
            "author_preprocessed_train_artifact_available": False,
            "exact_reactzyme_overlap_status": "not_verifiable_from_published_or_retained_artifacts",
            "reason": (
                "The exact author-preprocessed ORD train split is not published locally or with the model. "
                "Reconstructing a current ORD snapshot would not establish membership in that historical 80/10/10 split."
            ),
            "reactzyme_encoding_note": (
                "These ReactZyme files store an undirected molecule-set SMILES. The local extractor encodes it "
                "as a pseudo reaction S>>S, so equality with directional ORD reactions is not a well-defined leakage test."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = []
    for split in SPLITS:
        item = report["sleec"]["splits"][split]
        proxy = item["current_mcsa_proxy"]
        rows.append(
            "| " + " | ".join(
                [
                    split,
                    str(item["unique_test_protein_ids"]),
                    str(proxy["full_sequence_exact_unique_proteins"]),
                    str(proxy["model_view_exact_unique_proteins"]),
                    f"{100 * proxy['model_view_fraction']:.3f}%",
                    str(proxy["model_view_exact_pairs"]),
                ]
            ) + " |"
        )
    sleec = report["sleec"]
    reactiont5 = report["reactiont5v2"]
    markdown = "\n".join(
        [
            "# ReactZyme pretraining-overlap audit",
            "",
            "This audit separates exact exposure that can be measured from provenance that cannot be reconstructed. It does not estimate remote homology.",
            "",
            "## SLEEC",
            "",
            f"Historical exact-overlap status: `{sleec['historical_exact_overlap_status']}`.",
            "",
            f"The retained checkpoint manifest points to deleted stage-1 source artifacts. The table therefore uses a current M-CSA/UniProt snapshot proxy ({sleec['current_mcsa_snapshot_proxy']['entries']} entries, {sleec['current_mcsa_snapshot_proxy']['sequences_retrieved']} sequences retrieved).",
            "",
            "| Split | Test proteins | Full exact | 1022-residue model-view exact | Model-view fraction | Covered test pairs |",
            "| :-- | --: | --: | --: | --: | --: |",
            *rows,
            "",
            "An exact proxy match is evidence of possible direct source exposure; zero proxy matches is not proof against exposure in the deleted historical snapshot.",
            "",
            "## ReactionT5v2",
            "",
            f"Exact-overlap status: `{reactiont5['exact_reactzyme_overlap_status']}`.",
            "",
            reactiont5["reason"],
            "",
            reactiont5["reactzyme_encoding_note"],
        ]
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def _path(value: str) -> Path:
    return Path(value).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=_path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol-root", type=_path, required=True)
    parser.add_argument("--sleec-manifest", type=_path, required=True)
    parser.add_argument("--cache-dir", type=_path, required=True)
    parser.add_argument("--output-json", type=_path, required=True)
    parser.add_argument("--output-markdown", type=_path, required=True)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    report = audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_markdown)
    print(f"Wrote pretraining-overlap audit: {args.output_markdown}")


if __name__ == "__main__":
    main()
