#!/usr/bin/env python3
"""Build one final amino-acid sequence for every protein entry in the workbook.

This is the entry-level companion to ``fetch_protein_sequences.py``.  The
fetcher pools source records; this script turns those source records into the
actual sequence represented by each workbook row.  It:

* uses exact patent sequence listings where a row names a patent SEQ ID;
* reconstructs literature variants from the stated parent and mutations;
* validates every stated reference residue before applying a mutation;
* carries homolog/variant sequences into the Ranked Candidates sheet; and
* records rows whose sequence definition is genuinely incomplete without
  inventing a sequence.

Outputs (in ``sequence_pool`` by default):

* ``final_entry_sequences.fasta`` -- one record per resolved workbook row;
* ``final_entry_sequences.csv`` -- all rows, including the full sequence;
* ``unresolved_final_entries.csv`` -- only unresolved rows and reasons; and
* ``final_entry_summary.txt`` -- per-sheet resolution counts.

The script uses only the Python standard library.  If the source pool is not
present, it first runs ``fetch_protein_sequences.py`` automatically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fetch_protein_sequences as source_pool


SHEET_ORDER = (
    "Ranked Candidates",
    "Homologs",
    "Engineered Variants",
    "Indirect F6P",
)

# These locus/gene names were searchable directly in NCBI even though the
# workbook did not include their accessions.  The two 2026 mining-panel
# proteins are the organism-specific UxaE homologs corresponding to the names
# in source S16.  Source S12 Table S2 identifies its kbaZ donor as MG1655.
EXTRA_NCBI_ACCESSIONS = {
    "H030": "ACY48415.1",   # Rmar_1528
    "H031": "ADH61429.1",   # Tmath_1726
    "H033": "AEJ18237.1",   # spica_0065
    "H034": "SFS20489.1",   # SAMN05421771_3660
    "H067": "ABL78904.1",   # Tpen_1507, Thermofilum pendens
    "H068": "AEH51163.1",   # Theth_1083, Pseudothermotoga thermarum
    "F6P_005": "NP_417601.1",  # kbaZ, E. coli K-12 substr. MG1655
}

CJ_PATENT_SEQUENCE_IDS = {
    "H018": 1,
    "H019": 3,
    "H020": 5,
    "H021": 7,
    "H022": 9,
    "H023": 11,
    "H024": 13,
    "H025": 15,
    "H026": 17,
    "H027": 19,
}

HOMOLOG_VARIANT_LINKS = {
    "H002": "V003",  # Tpet 5V / T4E
    "H004": "V026",  # Tne M3
    "H009": "V076",  # TsT4Ease M4-4
    "H012": "V104",  # TaDt4e S46A/I113M
    "H015": "V120",  # K. olearia C52S
}

FIVE_V_CHANGES = "S125D/N129T/L140P/T181A/H362L"


@dataclass(frozen=True)
class FinalEntry:
    sheet: str
    excel_row: int
    entry_id: str
    name: str
    status: str
    sequence_source: str = ""
    parent_identifier: str = ""
    applied_changes: str = ""
    sequence: str = ""
    reason: str = ""
    evidence_row: str = ""

    @property
    def resolved(self) -> bool:
        return self.status.startswith("resolved") and bool(self.sequence)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest() if self.sequence else ""


def parse_simple_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(parts).upper().rstrip("*")))
            header = line[1:].strip()
            parts = []
        elif header is not None:
            parts.append(line)
    if header is not None:
        records.append((header, "".join(parts).upper().rstrip("*")))
    return records


def load_identifier_sequences(output_dir: Path) -> dict[tuple[str, str], list[str]]:
    canonical_path = output_dir / "protein_sequences.fasta"
    manifest_path = output_dir / "identifier_manifest.csv"
    if not canonical_path.exists() or not manifest_path.exists():
        return {}

    canonical = {
        header.split()[0]: sequence
        for header, sequence in parse_simple_fasta(canonical_path)
    }
    result: dict[tuple[str, str], list[str]] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            canonical_ids = [
                item for item in row["canonical_sequence_ids"].split(";") if item
            ]
            sequences = [canonical[item] for item in canonical_ids if item in canonical]
            if sequences:
                result[(row["provider"].lower(), row["identifier"].upper())] = sequences
    return result


def ensure_source_pool(args: argparse.Namespace) -> None:
    manifest = args.output_dir / "identifier_manifest.csv"
    if manifest.exists() and not args.refresh_source_pool:
        return
    command = [
        str(args.workbook),
        "--output-dir",
        str(args.output_dir),
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
    ]
    if args.email:
        command.extend(["--email", args.email])
    if args.ncbi_api_key:
        command.extend(["--ncbi-api-key", args.ncbi_api_key])
    if args.refresh_source_pool:
        command.append("--refresh")
    return_code = source_pool.main(command)
    if return_code:
        raise RuntimeError(f"Source-sequence pooling failed with exit code {return_code}")


def fetch_extra_ncbi_sequences(
    identifiers: Iterable[str],
    identifier_sequences: dict[tuple[str, str], list[str]],
    args: argparse.Namespace,
) -> None:
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, identifier in enumerate(sorted(set(identifiers))):
        key = ("ncbi", identifier.upper())
        if key in identifier_sequences and not args.refresh_extra:
            continue
        request = source_pool.IdentifierRequest("ncbi", identifier)
        resolution = source_pool.resolve_request(
            request=request,
            cache_dir=cache_dir,
            refresh=args.refresh_extra,
            timeout=args.timeout,
            retries=args.retries,
            email=args.email,
            ncbi_api_key=args.ncbi_api_key,
        )
        if resolution.status.startswith("resolved") and resolution.records:
            identifier_sequences[key] = [record.sequence for record in resolution.records]
        if index and not args.ncbi_api_key:
            time.sleep(0.34)


def load_cj_patent_sequences(script_dir: Path) -> dict[str, str]:
    path = script_dir / "patent_US11180785_proteins.fasta"
    result: dict[str, str] = {}
    for header, sequence in parse_simple_fasta(path):
        identifier = header.split()[0].upper()
        expected_match = re.search(r"\blength=(\d+)\b", header)
        if expected_match and len(sequence) != int(expected_match.group(1)):
            raise ValueError(
                f"Length mismatch in {path.name} for {identifier}: "
                f"expected {expected_match.group(1)}, got {len(sequence)}"
            )
        result[identifier] = sequence
    return result


def unique_sequence(
    sequences: Iterable[tuple[str, str]],
) -> tuple[str, str]:
    """Return one sequence and its identifiers, rejecting real ambiguity."""
    by_sequence: dict[str, list[str]] = {}
    for identifier, sequence in sequences:
        by_sequence.setdefault(sequence, []).append(identifier)
    if not by_sequence:
        raise LookupError("no resolvable parent accession was present")
    if len(by_sequence) > 1:
        details = "; ".join(
            f"{','.join(ids)}={len(sequence)} aa"
            for sequence, ids in by_sequence.items()
        )
        raise ValueError(f"parent accessions have different sequences: {details}")
    sequence, identifiers = next(iter(by_sequence.items()))
    return sequence, ";".join(identifiers)


def pick_parent_sequence(
    row: dict[str, str],
    identifier_sequences: dict[tuple[str, str], list[str]],
) -> tuple[str, str]:
    candidates: list[tuple[str, str]] = []
    for column, value in row.items():
        lowered = column.lower()
        if "genbank" not in lowered and "refseq" not in lowered:
            continue
        for identifier in source_pool.NCBI_PROTEIN_RE.findall(value.upper()):
            for sequence in identifier_sequences.get(("ncbi", identifier), []):
                candidates.append((f"NCBI:{identifier}", sequence))
    if candidates:
        return unique_sequence(candidates)

    for column, value in row.items():
        if "uniprot" not in column.lower():
            continue
        for identifier in source_pool.UNIPROT_RE.findall(value.upper()):
            for sequence in identifier_sequences.get(("uniprot", identifier), []):
                candidates.append((f"UniProt:{identifier}", sequence))
    return unique_sequence(candidates)


def normalize_patent(patent: str) -> str:
    return source_pool._normalize_us_patent(patent)  # Reuse fetcher's normalization.


def row_patent_identifiers(
    row: dict[str, str],
    source_patents: dict[str, set[str]],
) -> list[str]:
    joined = " | ".join(str(value) for value in row.values())
    patents = {normalize_patent(item) for item in source_pool.US_PATENT_RE.findall(joined)}
    if not patents:
        for source_id in re.split(r"\s*;\s*", row.get("Source IDs", "")):
            patents.update(source_patents.get(source_id, set()))
    default_patent = next(iter(patents)) if len(patents) == 1 else ""

    # Patent variant rows commonly contain two SEQ IDs: the final construct in
    # the row's name and its ancestor in the parent-accession column.  The
    # construct field is authoritative for the sequence represented by the row.
    preferred_text = ""
    for column in ("Variant", "Homolog / construct"):
        value = row.get(column, "")
        if source_pool.SEQ_ID_RE.search(value):
            preferred_text = value
            break
    search_text = preferred_text or joined
    identifiers: set[str] = set()
    for match in source_pool.SEQ_ID_RE.finditer(search_text):
        patent = normalize_patent(match.group(1) or default_patent)
        if patent and patent != "UNKNOWN":
            identifiers.add(f"{patent}:SEQ_ID_{int(match.group(2))}")
    return sorted(identifiers)


def exact_patent_sequence(
    row: dict[str, str],
    source_patents: dict[str, set[str]],
    identifier_sequences: dict[tuple[str, str], list[str]],
) -> tuple[str, str] | None:
    identifiers = row_patent_identifiers(row, source_patents)
    candidates: list[tuple[str, str]] = []
    for identifier in identifiers:
        for sequence in identifier_sequences.get(("patent", identifier.upper()), []):
            candidates.append((f"Patent:{identifier}", sequence))
    if not candidates:
        return None
    return unique_sequence(candidates)


def apply_changes(parent: str, specification: str) -> tuple[str, str]:
    """Apply substitutions/deletions using one-based parent coordinates."""
    spec = specification.strip().replace("–", "-").replace("—", "-")
    if spec.startswith("5V +"):
        spec = FIVE_V_CHANGES + "/" + spec.split("+", 1)[1].strip()
    if spec == "Thar-ex":
        # Translation of the Thar-ex mutagenic primer in S16 Table S1.
        spec = "S106A/E115G/K117R/Y118T"

    if re.fullmatch(r"Residues 3-5 replaced with KYQ", spec, flags=re.IGNORECASE):
        if len(parent) < 5:
            raise ValueError("parent is shorter than residue 5")
        return parent[:2] + "KYQ" + parent[5:], "3-5:KYQ replacement"

    substitutions: dict[int, tuple[str, str]] = {}
    for old, position_text, new in re.findall(r"(?<![A-Za-z])([A-Z])(\d+)([A-Z])", spec):
        position = int(position_text)
        if position in substitutions and substitutions[position] != (old, new):
            raise ValueError(f"conflicting substitutions at position {position}")
        substitutions[position] = (old, new)

    deletions: set[int] = set()
    for start_text, end_text in re.findall(r"Δ(\d+)-(\d+)", spec):
        start, end = int(start_text), int(end_text)
        if start > end:
            raise ValueError(f"invalid deletion range Δ{start}-{end}")
        deletions.update(range(start, end + 1))
    for old, position_text in re.findall(r"Δ([A-Z])(\d+)", spec):
        position = int(position_text)
        if position < 1 or position > len(parent):
            raise ValueError(f"deletion position {position} is outside a {len(parent)} aa parent")
        if parent[position - 1] != old:
            raise ValueError(
                f"deletion Δ{old}{position} disagrees with parent residue "
                f"{parent[position - 1]}{position}"
            )
        deletions.add(position)

    if not substitutions and not deletions:
        raise ValueError(f"no fully specified amino-acid change could be parsed from {specification!r}")

    changed = list(parent)
    for position, (old, new) in sorted(substitutions.items()):
        if position < 1 or position > len(parent):
            raise ValueError(
                f"substitution {old}{position}{new} is outside a {len(parent)} aa parent"
            )
        observed = parent[position - 1]
        if observed != old:
            raise ValueError(
                f"substitution {old}{position}{new} disagrees with parent residue "
                f"{observed}{position}"
            )
        changed[position - 1] = new

    for position in deletions:
        if position < 1 or position > len(parent):
            raise ValueError(f"deletion position {position} is outside a {len(parent)} aa parent")
    final = "".join(
        residue for position, residue in enumerate(changed, start=1) if position not in deletions
    )
    normalized_parts = [
        f"{old}{position}{new}"
        for position, (old, new) in sorted(substitutions.items())
    ]
    if deletions:
        ordered = sorted(deletions)
        runs: list[tuple[int, int]] = []
        start = previous = ordered[0]
        for position in ordered[1:]:
            if position == previous + 1:
                previous = position
                continue
            runs.append((start, previous))
            start = previous = position
        runs.append((start, previous))
        normalized_parts.extend(
            f"Δ{start}" if start == end else f"Δ{start}-{end}"
            for start, end in runs
        )
    return final, "/".join(normalized_parts)


def unresolved(
    sheet: str,
    excel_row: int,
    entry_id: str,
    name: str,
    reason: str,
    evidence_row: str = "",
) -> FinalEntry:
    return FinalEntry(
        sheet=sheet,
        excel_row=excel_row,
        entry_id=entry_id,
        name=name,
        status="unresolved_incomplete_definition",
        reason=reason,
        evidence_row=evidence_row,
    )


def resolved_reference(
    sheet: str,
    excel_row: int,
    entry_id: str,
    name: str,
    sequence: str,
    identifier: str,
    status: str = "resolved_reference",
    source: str = "public protein accession",
) -> FinalEntry:
    return FinalEntry(
        sheet=sheet,
        excel_row=excel_row,
        entry_id=entry_id,
        name=name,
        status=status,
        sequence_source=source,
        parent_identifier=identifier,
        sequence=sequence,
    )


def build_variant_entries(
    rows: list[tuple[int, dict[str, str]]],
    source_patents: dict[str, set[str]],
    identifier_sequences: dict[tuple[str, str], list[str]],
) -> dict[str, FinalEntry]:
    result: dict[str, FinalEntry] = {}
    for excel_row, row in rows:
        entry_id = row["Variant evidence ID"]
        name = row["Variant"]
        exact = exact_patent_sequence(row, source_patents, identifier_sequences)
        if exact is not None:
            sequence, identifier = exact
            result[entry_id] = resolved_reference(
                "Engineered Variants",
                excel_row,
                entry_id,
                name,
                sequence,
                identifier,
                status="resolved_exact_patent",
                source="exact USPTO sequence listing",
            )
            continue

        if entry_id in {f"V{number:03d}" for number in range(77, 88)}:
            result[entry_id] = unresolved(
                "Engineered Variants",
                excel_row,
                entry_id,
                name,
                "The error-prone-PCR substitution set is not disclosed in the S06 supporting information.",
            )
            continue
        if entry_id == "V151":
            result[entry_id] = unresolved(
                "Engineered Variants",
                excel_row,
                entry_id,
                name,
                "The workbook states a KYQ insertion but does not define the insertion coordinate; the article text only precisely defines the M1 residues 3-5 replacement.",
            )
            continue
        if entry_id == "V152":
            result[entry_id] = unresolved(
                "Engineered Variants",
                excel_row,
                entry_id,
                name,
                "This is an aggregate mutant panel, not one construct, and the individual substitutions are not listed.",
            )
            continue
        if entry_id == "V153":
            result[entry_id] = unresolved(
                "Engineered Variants",
                excel_row,
                entry_id,
                name,
                "The KoT4E M6 mutation set and its parent accession are not disclosed in the accessible record.",
            )
            continue

        try:
            parent, identifier = pick_parent_sequence(row, identifier_sequences)
            final, normalized_changes = apply_changes(parent, row["Mutation(s)"])
            result[entry_id] = FinalEntry(
                sheet="Engineered Variants",
                excel_row=excel_row,
                entry_id=entry_id,
                name=name,
                status="resolved_reconstructed",
                sequence_source="validated reconstruction from parent accession",
                parent_identifier=identifier,
                applied_changes=normalized_changes,
                sequence=final,
            )
        except (LookupError, ValueError) as exc:
            result[entry_id] = unresolved(
                "Engineered Variants", excel_row, entry_id, name, str(exc)
            )
    return result


def build_homolog_entries(
    rows: list[tuple[int, dict[str, str]]],
    variants: dict[str, FinalEntry],
    source_patents: dict[str, set[str]],
    identifier_sequences: dict[tuple[str, str], list[str]],
    cj_sequences: dict[str, str],
) -> dict[str, FinalEntry]:
    result: dict[str, FinalEntry] = {}
    pending_h069: tuple[int, dict[str, str]] | None = None
    for excel_row, row in rows:
        entry_id = row["Evidence ID"]
        name = row["Homolog / construct"]

        if entry_id == "H069":
            pending_h069 = (excel_row, row)
            continue
        if entry_id == "H017":
            result[entry_id] = unresolved(
                "Homologs",
                excel_row,
                entry_id,
                name,
                "The KoT4E M6 mutation set and parent sequence are not disclosed in the accessible record.",
            )
            continue
        if entry_id in HOMOLOG_VARIANT_LINKS:
            variant_id = HOMOLOG_VARIANT_LINKS[entry_id]
            linked = variants[variant_id]
            if linked.resolved:
                result[entry_id] = FinalEntry(
                    sheet="Homologs",
                    excel_row=excel_row,
                    entry_id=entry_id,
                    name=name,
                    status="resolved_cross_reference",
                    sequence_source=f"workbook construct cross-reference to {variant_id}",
                    parent_identifier=linked.parent_identifier,
                    applied_changes=linked.applied_changes,
                    sequence=linked.sequence,
                    evidence_row=variant_id,
                )
            else:
                result[entry_id] = unresolved(
                    "Homologs", excel_row, entry_id, name, linked.reason, variant_id
                )
            continue
        if entry_id in CJ_PATENT_SEQUENCE_IDS:
            sequence_id = CJ_PATENT_SEQUENCE_IDS[entry_id]
            identifier = f"US11180785:SEQ_ID_{sequence_id}"
            sequence = cj_sequences[identifier]
            result[entry_id] = resolved_reference(
                "Homologs",
                excel_row,
                entry_id,
                name,
                sequence,
                f"Patent:{identifier}",
                status="resolved_exact_patent",
                source="exact US11180785 protein sequence listing",
            )
            continue

        exact = exact_patent_sequence(row, source_patents, identifier_sequences)
        if exact is not None:
            sequence, identifier = exact
            result[entry_id] = resolved_reference(
                "Homologs",
                excel_row,
                entry_id,
                name,
                sequence,
                identifier,
                status="resolved_exact_patent",
                source="exact USPTO sequence listing",
            )
            continue

        if entry_id in EXTRA_NCBI_ACCESSIONS:
            accession = EXTRA_NCBI_ACCESSIONS[entry_id]
            sequences = identifier_sequences.get(("ncbi", accession.upper()), [])
            if sequences:
                status = (
                    "resolved_inferred_reference"
                    if entry_id in {"H067", "H068"}
                    else "resolved_reference"
                )
                source = (
                    "organism/locus-matched NCBI homolog for the S16 mining-panel name"
                    if status == "resolved_inferred_reference"
                    else "NCBI locus-tag match for workbook construct name"
                )
                result[entry_id] = resolved_reference(
                    "Homologs",
                    excel_row,
                    entry_id,
                    name,
                    sequences[0],
                    f"NCBI:{accession}",
                    status=status,
                    source=source,
                )
            else:
                result[entry_id] = unresolved(
                    "Homologs",
                    excel_row,
                    entry_id,
                    name,
                    f"Could not retrieve inferred NCBI accession {accession}.",
                )
            continue

        try:
            sequence, identifier = pick_parent_sequence(row, identifier_sequences)
            result[entry_id] = resolved_reference(
                "Homologs", excel_row, entry_id, name, sequence, identifier
            )
        except (LookupError, ValueError) as exc:
            result[entry_id] = unresolved(
                "Homologs", excel_row, entry_id, name, str(exc)
            )

    if pending_h069 is not None:
        excel_row, row = pending_h069
        entry_id = row["Evidence ID"]
        name = row["Homolog / construct"]
        linked = result.get("H011")
        if linked and linked.resolved:
            result[entry_id] = FinalEntry(
                sheet="Homologs",
                excel_row=excel_row,
                entry_id=entry_id,
                name=name,
                status="resolved_cross_reference",
                sequence_source="S16 Thermoproteales archaeon comparator cross-reference to TaDt4e WT (H011)",
                parent_identifier=linked.parent_identifier,
                sequence=linked.sequence,
                evidence_row="H011",
            )
        else:
            result[entry_id] = unresolved(
                "Homologs",
                excel_row,
                entry_id,
                name,
                "The linked TaDt4e WT sequence (H011) was not resolved.",
                "H011",
            )
    return result


def build_ranked_entries(
    rows: list[tuple[int, dict[str, str]]],
    evidence: dict[str, FinalEntry],
) -> dict[str, FinalEntry]:
    result: dict[str, FinalEntry] = {}
    for excel_row, row in rows:
        rank = int(float(row["Rank"]))
        entry_id = f"RANK_{rank:03d}"
        name = row["Candidate"]
        evidence_id = row["Evidence row"].strip()
        linked = evidence.get(evidence_id)
        if linked and linked.resolved:
            result[entry_id] = FinalEntry(
                sheet="Ranked Candidates",
                excel_row=excel_row,
                entry_id=entry_id,
                name=name,
                status="resolved_cross_reference",
                sequence_source=f"workbook Evidence row {evidence_id}",
                parent_identifier=linked.parent_identifier,
                applied_changes=linked.applied_changes,
                sequence=linked.sequence,
                evidence_row=evidence_id,
            )
        else:
            reason = linked.reason if linked else f"Evidence row {evidence_id} was not found"
            result[entry_id] = unresolved(
                "Ranked Candidates",
                excel_row,
                entry_id,
                name,
                reason,
                evidence_id,
            )
    return result


def build_indirect_entries(
    rows: list[tuple[int, dict[str, str]]],
    identifier_sequences: dict[tuple[str, str], list[str]],
) -> dict[str, FinalEntry]:
    result: dict[str, FinalEntry] = {}
    for index, (excel_row, row) in enumerate(rows, start=1):
        entry_id = f"F6P_{index:03d}"
        name = row["System / enzyme"]
        if name == "E. coli KbaZ":
            accession = EXTRA_NCBI_ACCESSIONS[entry_id]
            sequences = identifier_sequences.get(("ncbi", accession.upper()), [])
            if sequences:
                result[entry_id] = resolved_reference(
                    "Indirect F6P",
                    excel_row,
                    entry_id,
                    name,
                    sequences[0],
                    f"NCBI:{accession}",
                    status="resolved_inferred_reference",
                    source="S12 MG1655 kbaZ donor matched to the MG1655 RefSeq record",
                )
            else:
                result[entry_id] = unresolved(
                    "Indirect F6P",
                    excel_row,
                    entry_id,
                    name,
                    f"Could not retrieve the S12 MG1655 KbaZ accession {accession}.",
                )
            continue
        try:
            parent, identifier = pick_parent_sequence(row, identifier_sequences)
            mutation_match = re.search(r"\b([A-Z]\d+[A-Z])\b", name)
            if mutation_match:
                final, changes = apply_changes(parent, mutation_match.group(1))
                result[entry_id] = FinalEntry(
                    sheet="Indirect F6P",
                    excel_row=excel_row,
                    entry_id=entry_id,
                    name=name,
                    status="resolved_reconstructed",
                    sequence_source="validated reconstruction from parent accession",
                    parent_identifier=identifier,
                    applied_changes=changes,
                    sequence=final,
                )
            else:
                result[entry_id] = resolved_reference(
                    "Indirect F6P", excel_row, entry_id, name, parent, identifier
                )
        except (LookupError, ValueError) as exc:
            result[entry_id] = unresolved(
                "Indirect F6P", excel_row, entry_id, name, str(exc)
            )
    return result


def safe_header_value(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:+-]+", "_", value).strip("_") or "NA"


def wrap_fasta(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


CSV_FIELDS = (
    "sheet",
    "excel_row",
    "entry_id",
    "name",
    "status",
    "sequence_source",
    "parent_identifier",
    "applied_changes",
    "length",
    "sha256",
    "sequence",
    "reason",
    "evidence_row",
)


def entry_as_row(entry: FinalEntry) -> dict[str, str | int]:
    return {
        "sheet": entry.sheet,
        "excel_row": entry.excel_row,
        "entry_id": entry.entry_id,
        "name": entry.name,
        "status": entry.status,
        "sequence_source": entry.sequence_source,
        "parent_identifier": entry.parent_identifier,
        "applied_changes": entry.applied_changes,
        "length": len(entry.sequence) if entry.sequence else "",
        "sha256": entry.sha256,
        "sequence": entry.sequence,
        "reason": entry.reason,
        "evidence_row": entry.evidence_row,
    }


def write_outputs(output_dir: Path, entries: list[FinalEntry]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / "final_entry_sequences.fasta"
    csv_path = output_dir / "final_entry_sequences.csv"
    unresolved_path = output_dir / "unresolved_final_entries.csv"
    summary_path = output_dir / "final_entry_summary.txt"

    with fasta_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            if not entry.resolved:
                continue
            header = (
                f">{safe_header_value(entry.entry_id)}"
                f"|sheet={safe_header_value(entry.sheet)}"
                f"|row={entry.excel_row}"
                f"|name={safe_header_value(entry.name)}"
                f"|status={entry.status}"
                f"|length={len(entry.sequence)}"
                f"|sha256={entry.sha256}"
            )
            handle.write(header + "\n" + wrap_fasta(entry.sequence) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(entry_as_row(entry) for entry in entries)

    with unresolved_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(entry_as_row(entry) for entry in entries if not entry.resolved)

    lines = ["Final entry sequence resolution summary", ""]
    for sheet in SHEET_ORDER:
        sheet_entries = [entry for entry in entries if entry.sheet == sheet]
        resolved_count = sum(entry.resolved for entry in sheet_entries)
        lines.append(
            f"{sheet}: {resolved_count}/{len(sheet_entries)} resolved; "
            f"{len(sheet_entries) - resolved_count} unresolved"
        )
    resolved_count = sum(entry.resolved for entry in entries)
    unique_count = len({entry.sequence for entry in entries if entry.resolved})
    lines.extend(
        [
            "",
            f"All sheets: {resolved_count}/{len(entries)} resolved; "
            f"{len(entries) - resolved_count} unresolved",
            f"Unique resolved amino-acid sequences: {unique_count}",
            "",
            "Statuses:",
        ]
    )
    counts = Counter(entry.status for entry in entries)
    lines.extend(f"  {status}: {count}" for status, count in sorted(counts.items()))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workbook",
        nargs="?",
        type=Path,
        default=script_dir / "T4Ease_homologs_and_variants.xlsx",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "sequence_pool"
    )
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", ""))
    parser.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh-source-pool",
        action="store_true",
        help="Refresh every NCBI/UniProt/PDB/patent source record first.",
    )
    parser.add_argument(
        "--refresh-extra",
        action="store_true",
        help="Refresh the extra NCBI locus-tag/homolog records used here.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any workbook entry lacks a fully defined sequence.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.workbook = args.workbook.resolve()
    args.output_dir = args.output_dir.resolve()
    ensure_source_pool(args)

    workbook_rows = source_pool.read_xlsx_rows(args.workbook)
    source_patents = source_pool._source_patent_map(workbook_rows)
    identifier_sequences = load_identifier_sequences(args.output_dir)
    fetch_extra_ncbi_sequences(EXTRA_NCBI_ACCESSIONS.values(), identifier_sequences, args)
    cj_sequences = load_cj_patent_sequences(Path(__file__).resolve().parent)

    variants = build_variant_entries(
        workbook_rows["Engineered Variants"], source_patents, identifier_sequences
    )
    homologs = build_homolog_entries(
        workbook_rows["Homologs"],
        variants,
        source_patents,
        identifier_sequences,
        cj_sequences,
    )
    evidence = {**variants, **homologs}
    ranked = build_ranked_entries(workbook_rows["Ranked Candidates"], evidence)
    indirect = build_indirect_entries(
        workbook_rows["Indirect F6P"], identifier_sequences
    )

    by_sheet = {
        "Ranked Candidates": ranked,
        "Homologs": homologs,
        "Engineered Variants": variants,
        "Indirect F6P": indirect,
    }
    entries: list[FinalEntry] = []
    for sheet in SHEET_ORDER:
        entries.extend(sorted(by_sheet[sheet].values(), key=lambda entry: entry.excel_row))
    write_outputs(args.output_dir, entries)

    resolved_count = sum(entry.resolved for entry in entries)
    unresolved_count = len(entries) - resolved_count
    unique_count = len({entry.sequence for entry in entries if entry.resolved})
    print(
        f"Wrote {resolved_count} resolved entry records ({unique_count} unique sequences) "
        f"and {unresolved_count} explicitly unresolved rows to {args.output_dir}."
    )
    if args.strict and unresolved_count:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
