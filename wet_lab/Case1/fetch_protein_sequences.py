#!/usr/bin/env python3
"""Build a deduplicated protein-sequence pool from the Case1 workbook.

The script has no third-party Python dependencies. It reads the XLSX container
directly, extracts NCBI/GenBank/RefSeq, UniProt, PDB, and patent SEQ IDs, then
retrieves public sequences from official provider endpoints.

Outputs:
  protein_sequences.fasta   One record per unique amino-acid sequence.
  protein_sequences_by_identifier.fasta
                            One record per source identifier/PDB entity.
  sequence_manifest.csv     Cross-references each unique sequence to source IDs.
  identifier_manifest.csv   Resolution status and workbook provenance per ID.
  unresolved_identifiers.csv
                            Any identifiers that could not be resolved.
  cache/                    Provider responses, reused on subsequent runs.

US patent sequences are retrieved from USPTO PSIPS sequence listings when
available. A documented NCBI accession alias is used for the one short patent
sequence listing that is not hosted by PSIPS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable
from zipfile import BadZipFile, ZipFile


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS, "r": REL_NS}

TARGET_SHEETS = {
    "Ranked Candidates",
    "Homologs",
    "Engineered Variants",
    "Indirect F6P",
}

NCBI_PROTEIN_RE = re.compile(
    r"\b(?:"
    r"(?:AP|NP|XP|YP|WP|ZP)_\d+\.\d+"
    r"|[A-Z]{3}\d{5,9}\.\d+"
    r")\b"
)
UNIPROT_RE = re.compile(
    r"\b(?:"
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2}"
    r")\b"
)
PDB_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")
US_PATENT_RE = re.compile(r"\bUS\d{8,}[A-Z]?\d?\b", re.IGNORECASE)
SEQ_ID_RE = re.compile(
    r"(?:(US\d{8,}[A-Z]?\d?)\s+)?(?:patent\s+)?"
    r"SEQ\s+ID(?:\s+NO\.?)?\s*[:#]?\s*(\d+)",
    re.IGNORECASE,
)

PROVIDER_ORDER = {"ncbi": 0, "uniprot": 1, "rcsb": 2, "patent": 3}
PROVIDER_LABEL = {
    "ncbi": "NCBI",
    "uniprot": "UniProt",
    "rcsb": "PDB",
    "patent": "Patent",
}
VALID_SEQUENCE_RE = re.compile(r"^[A-Z]+$")

AMINO_ACID_3_TO_1 = {
    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V",
    "Asx": "B",
    "Glx": "Z",
    "Xaa": "X",
    "Sec": "U",
    "Pyl": "O",
}

# US12098401 identifies its protein SEQ ID 1 as NCBI OUC05982.1 in the patent
# bibliography. Its short sequence listing is not hosted by USPTO PSIPS.
PATENT_SEQUENCE_ACCESSION_ALIASES = {
    "US12098401:SEQ_ID_1": ("ncbi", "OUC05982.1"),
}


@dataclass(frozen=True, order=True)
class Occurrence:
    sheet: str
    row: int
    label: str
    column: str
    raw_value: str

    def compact(self) -> str:
        label = self.label.replace("|", "/")
        return f"{self.sheet}!{self.row}:{label}:{self.column}"


@dataclass
class IdentifierRequest:
    provider: str
    identifier: str
    occurrences: set[Occurrence] = field(default_factory=set)


@dataclass(frozen=True)
class FastaRecord:
    provider: str
    requested_id: str
    retrieved_id: str
    header: str
    sequence: str


@dataclass
class Resolution:
    request: IdentifierRequest
    status: str
    records: list[FastaRecord] = field(default_factory=list)
    error: str = ""


def _shared_string_text(element: ET.Element) -> str:
    return "".join(
        node.text or "" for node in element.iter(f"{{{MAIN_NS}}}t")
    )


def _column_number(cell_reference: str) -> int:
    match = re.match(r"[A-Z]+", cell_reference)
    if not match:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    number = 0
    for character in match.group(0):
        number = number * 26 + ord(character) - ord("A") + 1
    return number


def _worksheet_path(target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return str(PurePosixPath("xl") / target)


def read_xlsx_rows(path: Path) -> dict[str, list[tuple[int, dict[str, str]]]]:
    """Return visible cell values keyed by sheet and Excel row number."""
    try:
        archive = ZipFile(path)
    except (FileNotFoundError, BadZipFile) as exc:
        raise RuntimeError(f"Cannot open workbook {path}: {exc}") from exc

    with archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = [
                _shared_string_text(element)
                for element in shared_root.findall("m:si", NS)
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            archive.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            element.attrib["Id"]: element.attrib["Target"]
            for element in relationships
        }

        result: dict[str, list[tuple[int, dict[str, str]]]] = {}
        sheets_element = workbook.find("m:sheets", NS)
        if sheets_element is None:
            raise RuntimeError("Workbook contains no sheets")

        for sheet in sheets_element:
            sheet_name = sheet.attrib["name"]
            relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
            xml_path = _worksheet_path(relationship_targets[relationship_id])
            root = ET.fromstring(archive.read(xml_path))
            sheet_data = root.find("m:sheetData", NS)
            if sheet_data is None:
                result[sheet_name] = []
                continue

            raw_rows: list[tuple[int, dict[int, str]]] = []
            for row in sheet_data.findall("m:row", NS):
                excel_row = int(row.attrib["r"])
                values: dict[int, str] = {}
                for cell in row.findall("m:c", NS):
                    column = _column_number(cell.attrib["r"])
                    cell_type = cell.attrib.get("t")
                    value_element = cell.find("m:v", NS)
                    if cell_type == "s":
                        if value_element is None:
                            value = ""
                        else:
                            value = shared_strings[int(value_element.text or "0")]
                    elif cell_type == "inlineStr":
                        inline = cell.find("m:is", NS)
                        value = _shared_string_text(inline) if inline is not None else ""
                    else:
                        value = value_element.text if value_element is not None else ""
                    values[column] = value
                raw_rows.append((excel_row, values))

            if not raw_rows:
                result[sheet_name] = []
                continue

            headers = raw_rows[0][1]
            parsed_rows: list[tuple[int, dict[str, str]]] = []
            for excel_row, values in raw_rows[1:]:
                parsed_rows.append(
                    (
                        excel_row,
                        {
                            headers[column]: value
                            for column, value in values.items()
                            if column in headers and headers[column]
                        },
                    )
                )
            result[sheet_name] = parsed_rows

    return result


def _row_label(row: dict[str, str]) -> str:
    for key in (
        "Candidate",
        "Homolog / construct",
        "Variant",
        "System / enzyme",
        "Evidence ID",
        "Variant evidence ID",
    ):
        if row.get(key):
            return row[key]
    return "unlabelled"


def _add_request(
    requests: dict[tuple[str, str], IdentifierRequest],
    provider: str,
    identifier: str,
    occurrence: Occurrence,
) -> None:
    normalized = identifier.upper()
    key = (provider, normalized)
    if key not in requests:
        requests[key] = IdentifierRequest(provider, normalized)
    requests[key].occurrences.add(occurrence)


def _normalize_us_patent(identifier: str) -> str:
    """Remove an optional US patent kind code (for example B2 or A1)."""
    return re.sub(r"[A-Z]\d$", "", identifier.upper())


def _source_patent_map(
    workbook_rows: dict[str, list[tuple[int, dict[str, str]]]],
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for _, row in workbook_rows.get("Sources", []):
        source_id = row.get("Source ID", "")
        if not source_id:
            continue
        joined_row = " | ".join(str(value) for value in row.values())
        mapping[source_id].update(
            _normalize_us_patent(item) for item in US_PATENT_RE.findall(joined_row)
        )
    return mapping


def extract_identifier_requests(
    workbook_rows: dict[str, list[tuple[int, dict[str, str]]]],
) -> list[IdentifierRequest]:
    requests: dict[tuple[str, str], IdentifierRequest] = {}
    source_patents = _source_patent_map(workbook_rows)

    for sheet_name, rows in workbook_rows.items():
        if sheet_name not in TARGET_SHEETS:
            continue
        for excel_row, row in rows:
            label = _row_label(row)
            for column, raw_value in row.items():
                value = str(raw_value)
                lowered_column = column.lower()
                if "genbank" in lowered_column or "refseq" in lowered_column:
                    for identifier in NCBI_PROTEIN_RE.findall(value.upper()):
                        _add_request(
                            requests,
                            "ncbi",
                            identifier,
                            Occurrence(sheet_name, excel_row, label, column, value),
                        )
                elif "uniprot" in lowered_column:
                    for identifier in UNIPROT_RE.findall(value.upper()):
                        _add_request(
                            requests,
                            "uniprot",
                            identifier,
                            Occurrence(sheet_name, excel_row, label, column, value),
                        )
                elif column.strip().lower() == "pdb":
                    for identifier in PDB_RE.findall(value.upper()):
                        _add_request(
                            requests,
                            "rcsb",
                            identifier,
                            Occurrence(sheet_name, excel_row, label, column, value),
                        )

            joined_row = " | ".join(str(value) for value in row.values())
            row_patents = {
                _normalize_us_patent(item)
                for item in US_PATENT_RE.findall(joined_row)
            }
            if not row_patents:
                source_ids = re.split(r"\s*;\s*", row.get("Source IDs", ""))
                for source_id in source_ids:
                    row_patents.update(source_patents.get(source_id, set()))
            default_patent = next(iter(row_patents)) if len(row_patents) == 1 else "UNKNOWN"
            for column, raw_value in row.items():
                value = str(raw_value)
                for match in SEQ_ID_RE.finditer(value):
                    patent = _normalize_us_patent(match.group(1) or default_patent)
                    identifier = f"{patent}:SEQ_ID_{match.group(2)}"
                    _add_request(
                        requests,
                        "patent",
                        identifier,
                        Occurrence(sheet_name, excel_row, label, column, value),
                    )

    return sorted(
        requests.values(),
        key=lambda item: (PROVIDER_ORDER[item.provider], item.identifier),
    )


def parse_fasta(text: str, provider: str, requested_id: str) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    sequence_parts: list[str] = []

    def finish_record() -> None:
        nonlocal header, sequence_parts
        if header is None:
            return
        sequence = "".join(sequence_parts).replace(" ", "").upper().rstrip("*")
        if not sequence or not VALID_SEQUENCE_RE.fullmatch(sequence):
            raise ValueError(
                f"Provider returned an invalid amino-acid sequence for {requested_id}"
            )
        first_token = header.split()[0]
        if provider == "uniprot" and "|" in first_token:
            token_parts = first_token.split("|")
            retrieved_id = token_parts[1] if len(token_parts) > 1 else first_token
        else:
            retrieved_id = first_token.split("|")[0]
        records.append(
            FastaRecord(
                provider=provider,
                requested_id=requested_id,
                retrieved_id=retrieved_id,
                header=header,
                sequence=sequence,
            )
        )
        header = None
        sequence_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            finish_record()
            header = line[1:].strip()
        elif header is not None:
            sequence_parts.append(line)
    finish_record()

    if not records:
        preview = text[:160].replace("\n", " ")
        raise ValueError(f"No FASTA record returned for {requested_id}: {preview}")
    return records


def _safe_cache_name(provider: str, identifier: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", identifier)
    return f"{provider}_{identifier}.fasta"


def fetch_bytes(
    url: str,
    timeout: float,
    retries: int,
    user_agent: str,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "text/plain"},
    )
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(str(last_error or "unknown HTTP error"))


def fetch_text(
    url: str,
    timeout: float,
    retries: int,
    user_agent: str,
) -> str:
    return fetch_bytes(url, timeout, retries, user_agent).decode("utf-8")


def provider_url(
    request: IdentifierRequest,
    email: str,
    ncbi_api_key: str,
) -> str:
    identifier = urllib.parse.quote(request.identifier, safe="")
    if request.provider == "ncbi":
        parameters = {
            "db": "protein",
            "id": request.identifier,
            "rettype": "fasta",
            "retmode": "text",
            "tool": "EnzymeDiscovery_sequence_pool",
        }
        if email:
            parameters["email"] = email
        if ncbi_api_key:
            parameters["api_key"] = ncbi_api_key
        return (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
            + urllib.parse.urlencode(parameters)
        )
    if request.provider == "uniprot":
        return f"https://rest.uniprot.org/uniprotkb/{identifier}.fasta"
    if request.provider == "rcsb":
        return f"https://www.rcsb.org/fasta/entry/{identifier}/download"
    raise ValueError(f"No public sequence endpoint for provider {request.provider}")


def resolve_request(
    request: IdentifierRequest,
    cache_dir: Path,
    refresh: bool,
    timeout: float,
    retries: int,
    email: str,
    ncbi_api_key: str,
) -> Resolution:
    if request.provider == "patent":
        return Resolution(
            request=request,
            status="unsupported_patent_sequence",
            error=(
                "Patent SEQ IDs require a patent sequence-listing source; no stable "
                "NCBI/UniProt/RCSB accession is present in the workbook."
            ),
        )

    cache_path = cache_dir / _safe_cache_name(
        request.provider, request.identifier
    )
    if request.provider == "uniprot" and not refresh:
        unisave_cache = cache_dir / _safe_cache_name(
            "uniprot_unisave", request.identifier
        )
        direct_cache_available = cache_path.exists() and cache_path.stat().st_size > 0
        if (
            not direct_cache_available
            and unisave_cache.exists()
            and unisave_cache.stat().st_size > 0
        ):
            try:
                archived_records = parse_fasta(
                    unisave_cache.read_text(encoding="utf-8"),
                    request.provider,
                    request.identifier,
                )
                return Resolution(
                    request=request,
                    status="resolved_via_unisave",
                    records=archived_records[:1],
                    error="Used the latest cached archived UniSave version.",
                )
            except Exception:
                pass
    try:
        from_cache = (
            cache_path.exists() and cache_path.stat().st_size > 0 and not refresh
        )
        if from_cache:
            text = cache_path.read_text(encoding="utf-8")
        else:
            url = provider_url(request, email, ncbi_api_key)
            text = fetch_text(
                url=url,
                timeout=timeout,
                retries=retries,
                user_agent="EnzymeDiscovery-sequence-pool/1.0",
            )
        try:
            records = parse_fasta(text, request.provider, request.identifier)
        except ValueError as direct_error:
            if request.provider != "uniprot":
                raise
            unisave_cache = cache_dir / _safe_cache_name(
                "uniprot_unisave", request.identifier
            )
            unisave_from_cache = (
                unisave_cache.exists()
                and unisave_cache.stat().st_size > 0
                and not refresh
            )
            if unisave_from_cache:
                unisave_text = unisave_cache.read_text(encoding="utf-8")
            else:
                identifier = urllib.parse.quote(request.identifier, safe="")
                unisave_text = fetch_text(
                    url=(
                        f"https://rest.uniprot.org/unisave/{identifier}?format=fasta"
                    ),
                    timeout=timeout,
                    retries=retries,
                    user_agent="EnzymeDiscovery-sequence-pool/1.0",
                )
            archived_records = parse_fasta(
                unisave_text, request.provider, request.identifier
            )
            # UniSave orders versions newest first. Retain the latest archived
            # record only; older versions remain available in the cached response.
            records = archived_records[:1]
            if not unisave_from_cache:
                unisave_cache.write_text(unisave_text, encoding="utf-8")
            return Resolution(
                request=request,
                status="resolved_via_unisave",
                records=records,
                error=(
                    f"Current UniProtKB FASTA unavailable ({direct_error}); used "
                    "the latest archived UniSave version."
                ),
            )
        if not from_cache:
            cache_path.write_text(text, encoding="utf-8")
        return Resolution(request=request, status="resolved", records=records)
    except Exception as exc:  # Preserve every failed ID in the output manifest.
        return Resolution(request=request, status="unresolved", error=str(exc))


def parse_st25_protein_sequences(text: str) -> dict[int, tuple[str, str]]:
    """Parse protein entries from a WIPO ST.25 plain-text sequence listing."""
    starts = list(
        re.finditer(
            r"^<210>\s*SEQ ID NO\s*:?[ \t]*(\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
    )
    if not starts:
        raise ValueError("No ST.25 <210> sequence entries were found")

    amino_acid_pattern = re.compile(
        r"\b(?:" + "|".join(AMINO_ACID_3_TO_1) + r")\b"
    )
    proteins: dict[int, tuple[str, str]] = {}
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start.start() : end]
        sequence_number = int(start.group(1))
        type_match = re.search(r"^<212>\s*TYPE:\s*(\S+)", block, re.MULTILINE)
        if type_match is None or type_match.group(1).upper() != "PRT":
            continue
        length_match = re.search(
            r"^<211>\s*LENGTH:\s*(\d+)", block, re.MULTILINE
        )
        sequence_start = re.search(r"^<400>[^\n]*\n", block, re.MULTILINE)
        if length_match is None or sequence_start is None:
            raise ValueError(f"Incomplete ST.25 protein entry {sequence_number}")
        sequence_text = block[sequence_start.end() :]
        sequence = "".join(
            AMINO_ACID_3_TO_1[token]
            for token in amino_acid_pattern.findall(sequence_text)
        )
        expected_length = int(length_match.group(1))
        if len(sequence) != expected_length:
            raise ValueError(
                f"ST.25 SEQ ID {sequence_number} has {len(sequence)} parsed "
                f"residues; expected {expected_length}"
            )
        organism_match = re.search(
            r"^<213>\s*ORGANISM:\s*(.+?)\s*$", block, re.MULTILINE
        )
        organism = organism_match.group(1).strip() if organism_match else "not reported"
        proteins[sequence_number] = (sequence, organism)
    return proteins


def load_uspto_patent_sequences(
    patent: str,
    cache_dir: Path,
    refresh: bool,
    timeout: float,
    retries: int,
) -> dict[int, tuple[str, str]]:
    numeric_match = re.fullmatch(r"US(\d+)", patent)
    if numeric_match is None:
        raise ValueError(f"Unsupported USPTO document identifier: {patent}")
    cache_path = cache_dir / f"uspto_{patent}_sequences.zip"
    from_cache = cache_path.exists() and cache_path.stat().st_size > 0 and not refresh
    if from_cache:
        archive_bytes = cache_path.read_bytes()
    else:
        url = (
            "https://seqdata.uspto.gov/viewservice/api/documents/"
            "download-all-sequences?"
            + urllib.parse.urlencode({"documentId": numeric_match.group(1)})
        )
        archive_bytes = fetch_bytes(
            url=url,
            timeout=timeout,
            retries=retries,
            user_agent="EnzymeDiscovery-sequence-pool/1.0",
        )

    try:
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".txt")
            ]
            if not members:
                raise ValueError("USPTO archive contains no ST.25 text file")
            proteins: dict[int, tuple[str, str]] = {}
            for member in members:
                text = archive.read(member).decode("utf-8", errors="replace")
                for sequence_number, record in parse_st25_protein_sequences(text).items():
                    if sequence_number in proteins and proteins[sequence_number] != record:
                        raise ValueError(
                            f"Conflicting USPTO SEQ ID {sequence_number} records"
                        )
                    proteins[sequence_number] = record
    except BadZipFile as exc:
        preview = archive_bytes[:160].decode("utf-8", errors="replace")
        raise ValueError(f"USPTO response was not a ZIP archive: {preview}") from exc

    if not from_cache:
        cache_path.write_bytes(archive_bytes)
    return proteins


def resolve_patent_requests(
    requests: list[IdentifierRequest],
    cache_dir: Path,
    refresh: bool,
    timeout: float,
    retries: int,
    email: str,
    ncbi_api_key: str,
) -> dict[str, Resolution]:
    """Resolve patent SEQ IDs in one download/parse operation per patent."""
    by_patent: dict[str, list[IdentifierRequest]] = defaultdict(list)
    for request in requests:
        patent, _, _ = request.identifier.partition(":")
        by_patent[patent].append(request)

    results: dict[str, Resolution] = {}
    for patent, patent_requests in sorted(by_patent.items()):
        listing_requests = [
            request
            for request in patent_requests
            if request.identifier not in PATENT_SEQUENCE_ACCESSION_ALIASES
        ]
        proteins: dict[int, tuple[str, str]] = {}
        listing_error = ""
        if listing_requests:
            try:
                proteins = load_uspto_patent_sequences(
                    patent=patent,
                    cache_dir=cache_dir,
                    refresh=refresh,
                    timeout=timeout,
                    retries=retries,
                )
            except Exception as exc:
                listing_error = str(exc)

        for request in patent_requests:
            alias = PATENT_SEQUENCE_ACCESSION_ALIASES.get(request.identifier)
            if alias is not None:
                alias_provider, alias_identifier = alias
                alias_request = IdentifierRequest(
                    provider=alias_provider,
                    identifier=alias_identifier,
                    occurrences=set(request.occurrences),
                )
                alias_resolution = resolve_request(
                    request=alias_request,
                    cache_dir=cache_dir,
                    refresh=refresh,
                    timeout=timeout,
                    retries=retries,
                    email=email,
                    ncbi_api_key=ncbi_api_key,
                )
                if _is_resolved(alias_resolution.status) and alias_resolution.records:
                    source_record = alias_resolution.records[0]
                    results[request.identifier] = Resolution(
                        request=request,
                        status="resolved_via_patent_accession_alias",
                        records=[
                            FastaRecord(
                                provider="patent",
                                requested_id=request.identifier,
                                retrieved_id=alias_identifier,
                                header=(
                                    f"{request.identifier} via "
                                    f"{alias_provider}:{alias_identifier}; "
                                    f"{source_record.header}"
                                ),
                                sequence=source_record.sequence,
                            )
                        ],
                        error=(
                            f"Patent bibliography identifies SEQ ID as "
                            f"{alias_provider}:{alias_identifier}."
                        ),
                    )
                else:
                    results[request.identifier] = Resolution(
                        request=request,
                        status="unresolved",
                        error=(
                            f"Patent accession alias {alias_provider}:"
                            f"{alias_identifier} failed: {alias_resolution.error}"
                        ),
                    )
                continue

            sequence_match = re.search(r":SEQ_ID_(\d+)$", request.identifier)
            sequence_number = int(sequence_match.group(1)) if sequence_match else -1
            if sequence_number in proteins:
                sequence, organism = proteins[sequence_number]
                results[request.identifier] = Resolution(
                    request=request,
                    status="resolved_via_uspto_sequence_listing",
                    records=[
                        FastaRecord(
                            provider="patent",
                            requested_id=request.identifier,
                            retrieved_id=request.identifier,
                            header=f"{request.identifier} organism={organism}",
                            sequence=sequence,
                        )
                    ],
                )
            else:
                error = listing_error or (
                    f"Protein SEQ ID {sequence_number} was not present in the "
                    "USPTO sequence listing"
                )
                results[request.identifier] = Resolution(
                    request=request, status="unresolved", error=error
                )
    return results


def _is_resolved(status: str) -> bool:
    return status.startswith("resolved")


def apply_workbook_alias_fallbacks(resolutions: list[Resolution]) -> None:
    """Resolve retired IDs from a co-listed public accession in the same row.

    The fallback is accepted only when the best available provider yields exactly
    one unique sequence. Ambiguous co-listed sequences stay unresolved.
    """
    by_location: dict[tuple[str, int], list[Resolution]] = defaultdict(list)
    for resolution in resolutions:
        if not _is_resolved(resolution.status):
            continue
        for occurrence in resolution.request.occurrences:
            by_location[(occurrence.sheet, occurrence.row)].append(resolution)

    for resolution in resolutions:
        if resolution.status != "unresolved":
            continue
        if resolution.request.provider == "patent":
            continue
        candidates: list[Resolution] = []
        for occurrence in resolution.request.occurrences:
            candidates.extend(by_location.get((occurrence.sheet, occurrence.row), []))
        candidates = [
            candidate
            for candidate in candidates
            if candidate.request.provider != resolution.request.provider
        ]
        if not candidates:
            continue

        best_priority = min(
            PROVIDER_ORDER[candidate.request.provider] for candidate in candidates
        )
        best_candidates = [
            candidate
            for candidate in candidates
            if PROVIDER_ORDER[candidate.request.provider] == best_priority
        ]
        records_by_hash: dict[str, list[tuple[Resolution, FastaRecord]]] = defaultdict(
            list
        )
        for candidate in best_candidates:
            for record in candidate.records:
                records_by_hash[sequence_hash(record.sequence)].append(
                    (candidate, record)
                )
        if len(records_by_hash) != 1:
            resolution.error += (
                f"; workbook alias fallback was ambiguous across "
                f"{len(records_by_hash)} sequences"
            )
            continue

        candidate, record = next(iter(records_by_hash.values()))[0]
        original_error = resolution.error
        resolution.records = [
            FastaRecord(
                provider=resolution.request.provider,
                requested_id=resolution.request.identifier,
                retrieved_id=record.retrieved_id,
                header=(
                    f"workbook_alias {resolution.request.identifier} via "
                    f"{candidate.request.provider}:{candidate.request.identifier}; "
                    f"{record.header}"
                ),
                sequence=record.sequence,
            )
        ]
        resolution.status = "resolved_via_workbook_alias"
        resolution.error = (
            f"Direct {resolution.request.provider} lookup failed ({original_error}); "
            f"sequence supplied by co-listed "
            f"{candidate.request.provider}:{candidate.request.identifier}."
        )


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def wrap_sequence(sequence: str, width: int = 80) -> Iterable[str]:
    for start in range(0, len(sequence), width):
        yield sequence[start : start + width]


def build_sequence_groups(
    resolutions: list[Resolution],
) -> tuple[dict[str, list[FastaRecord]], dict[str, str]]:
    by_hash: dict[str, list[FastaRecord]] = defaultdict(list)
    for resolution in resolutions:
        for record in resolution.records:
            by_hash[sequence_hash(record.sequence)].append(record)

    def group_sort_key(item: tuple[str, list[FastaRecord]]) -> tuple[int, str, str]:
        digest, records = item
        first = min(
            records,
            key=lambda record: (
                PROVIDER_ORDER[record.provider],
                record.requested_id,
                record.retrieved_id,
            ),
        )
        return PROVIDER_ORDER[first.provider], first.requested_id, digest

    sorted_groups = sorted(by_hash.items(), key=group_sort_key)
    canonical_ids = {
        digest: f"T4ESEQ_{index:04d}"
        for index, (digest, _) in enumerate(sorted_groups, start=1)
    }
    return dict(sorted_groups), canonical_ids


def _joined_occurrences(request: IdentifierRequest) -> str:
    return "; ".join(item.compact() for item in sorted(request.occurrences))


def write_outputs(output_dir: Path, resolutions: list[Resolution]) -> None:
    groups, canonical_ids = build_sequence_groups(resolutions)

    fasta_path = output_dir / "protein_sequences.fasta"
    with fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, records in groups.items():
            canonical_id = canonical_ids[digest]
            sequence = records[0].sequence
            identifiers = sorted(
                {
                    f"{PROVIDER_LABEL[record.provider]}:{record.requested_id}"
                    for record in records
                }
            )
            handle.write(
                f">{canonical_id} length={len(sequence)} sha256={digest} "
                f"ids={','.join(identifiers)}\n"
            )
            for line in wrap_sequence(sequence):
                handle.write(line + "\n")

    by_identifier_fasta_path = output_dir / "protein_sequences_by_identifier.fasta"
    with by_identifier_fasta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for resolution in resolutions:
            if not _is_resolved(resolution.status):
                continue
            seen_records: set[tuple[str, str]] = set()
            for record in resolution.records:
                digest = sequence_hash(record.sequence)
                record_key = (record.retrieved_id, digest)
                if record_key in seen_records:
                    continue
                seen_records.add(record_key)
                handle.write(
                    f">{resolution.request.provider}:{resolution.request.identifier}"
                    f"|canonical={canonical_ids[digest]}"
                    f"|retrieved={record.retrieved_id}"
                    f"|length={len(record.sequence)}\n"
                )
                for line in wrap_sequence(record.sequence):
                    handle.write(line + "\n")

    sequence_manifest_path = output_dir / "sequence_manifest.csv"
    with sequence_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "canonical_sequence_id",
            "length",
            "sha256",
            "ncbi_ids",
            "uniprot_ids",
            "pdb_ids",
            "patent_ids",
            "retrieved_ids",
            "provider_headers",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for digest, records in groups.items():
            ids_by_provider: dict[str, set[str]] = defaultdict(set)
            for record in records:
                ids_by_provider[record.provider].add(record.requested_id)
            writer.writerow(
                {
                    "canonical_sequence_id": canonical_ids[digest],
                    "length": len(records[0].sequence),
                    "sha256": digest,
                    "ncbi_ids": ";".join(sorted(ids_by_provider["ncbi"])),
                    "uniprot_ids": ";".join(sorted(ids_by_provider["uniprot"])),
                    "pdb_ids": ";".join(sorted(ids_by_provider["rcsb"])),
                    "patent_ids": ";".join(sorted(ids_by_provider["patent"])),
                    "retrieved_ids": ";".join(
                        sorted({record.retrieved_id for record in records})
                    ),
                    "provider_headers": " || ".join(
                        sorted({record.header for record in records})
                    ),
                }
            )

    identifier_rows: list[dict[str, str]] = []
    for resolution in resolutions:
        digests = sorted(
            {sequence_hash(record.sequence) for record in resolution.records}
        )
        identifier_rows.append(
            {
                "provider": resolution.request.provider,
                "identifier": resolution.request.identifier,
                "status": resolution.status,
                "canonical_sequence_ids": ";".join(
                    canonical_ids[digest] for digest in digests
                ),
                "retrieved_ids": ";".join(
                    sorted({record.retrieved_id for record in resolution.records})
                ),
                "sequence_lengths": ";".join(
                    str(len(record.sequence)) for record in resolution.records
                ),
                "workbook_occurrences": _joined_occurrences(resolution.request),
                "error": resolution.error,
            }
        )

    identifier_fields = [
        "provider",
        "identifier",
        "status",
        "canonical_sequence_ids",
        "retrieved_ids",
        "sequence_lengths",
        "workbook_occurrences",
        "error",
    ]
    identifier_manifest_path = output_dir / "identifier_manifest.csv"
    with identifier_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=identifier_fields)
        writer.writeheader()
        writer.writerows(identifier_rows)

    unresolved_path = output_dir / "unresolved_identifiers.csv"
    with unresolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=identifier_fields)
        writer.writeheader()
        writer.writerows(
            row for row in identifier_rows if not _is_resolved(row["status"])
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Extract all protein identifiers from the T4Ease workbook and build "
            "a deduplicated FASTA sequence pool."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=script_dir / "T4Ease_homologs_and_variants.xlsx",
        help="Input XLSX workbook (default: workbook next to this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "sequence_pool",
        help="Output directory (default: sequence_pool next to this script)",
    )
    parser.add_argument(
        "--email",
        default=os.environ.get("NCBI_EMAIL", ""),
        help="Contact email sent to NCBI, or set NCBI_EMAIL",
    )
    parser.add_argument(
        "--ncbi-api-key",
        default=os.environ.get("NCBI_API_KEY", ""),
        help="Optional NCBI API key, or set NCBI_API_KEY",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--refresh", action="store_true", help="Ignore cached provider responses"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only extract and print identifier counts; do not access the network",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if any NCBI, UniProt, or RCSB identifier is unresolved",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workbook_rows = read_xlsx_rows(args.workbook.resolve())
    requests = extract_identifier_requests(workbook_rows)
    counts = {
        provider: sum(request.provider == provider for request in requests)
        for provider in PROVIDER_ORDER
    }
    print(
        "Extracted identifiers: "
        + ", ".join(f"{provider}={counts[provider]}" for provider in PROVIDER_ORDER)
    )

    if args.dry_run:
        for request in requests:
            print(f"{request.provider}\t{request.identifier}")
        return 0

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    patent_requests = [request for request in requests if request.provider == "patent"]
    patent_resolutions = resolve_patent_requests(
        requests=patent_requests,
        cache_dir=cache_dir,
        refresh=args.refresh,
        timeout=args.timeout,
        retries=args.retries,
        email=args.email,
        ncbi_api_key=args.ncbi_api_key,
    )
    resolutions: list[Resolution] = []
    public_requests = [request for request in requests if request.provider != "patent"]
    completed_public = 0
    for request in requests:
        if request.provider == "patent":
            resolution = patent_resolutions[request.identifier]
        else:
            resolution = resolve_request(
                request=request,
                cache_dir=cache_dir,
                refresh=args.refresh,
                timeout=args.timeout,
                retries=args.retries,
                email=args.email,
                ncbi_api_key=args.ncbi_api_key,
            )
        resolutions.append(resolution)
        if request.provider != "patent":
            completed_public += 1
            marker = "ok" if _is_resolved(resolution.status) else "FAILED"
            print(
                f"[{completed_public}/{len(public_requests)}] "
                f"{request.provider}:{request.identifier} {marker}"
            )
            if request.provider == "ncbi" and not args.ncbi_api_key:
                time.sleep(0.34)

    apply_workbook_alias_fallbacks(resolutions)
    write_outputs(output_dir, resolutions)
    resolved_public = sum(
        _is_resolved(resolution.status)
        and resolution.request.provider != "patent"
        for resolution in resolutions
    )
    failed_public = sum(
        not _is_resolved(resolution.status)
        and resolution.request.provider != "patent"
        for resolution in resolutions
    )
    resolved_patents = sum(
        _is_resolved(resolution.status)
        and resolution.request.provider == "patent"
        for resolution in resolutions
    )
    failed_patents = sum(
        not _is_resolved(resolution.status)
        and resolution.request.provider == "patent"
        for resolution in resolutions
    )
    groups, _ = build_sequence_groups(resolutions)
    print(
        f"Wrote {len(groups)} unique sequences to {output_dir}; "
        f"public IDs resolved={resolved_public}, failed={failed_public}, "
        f"patent IDs resolved={resolved_patents}, failed={failed_patents}."
    )
    if args.strict and (failed_public or failed_patents):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
