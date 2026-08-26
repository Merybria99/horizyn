"""Enhance train-only enzyme cofactor labels with enzyme-side evidence."""

from __future__ import annotations

import csv
import hashlib
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from horizyn.capability.cofactors import (
    COFACTOR_ARCHITECTURE_BINS,
    COFACTOR_CHEMISTRY_BINS,
    METAL_ION_LABELS,
    NO_COFACTOR_LABEL,
    cofactor_label_for_chebi_term,
    cofactor_architecture_bins,
    cofactor_chemistry_bins,
    cofactor_label_tier,
    extract_cofactor_labels,
    extract_cofactor_labels_from_smiles,
    load_cofactor_aliases,
    split_cofactor_label_tiers,
)
from horizyn.capability.enzyme_labels import intersection_size, union_list_column
from horizyn.capability.io import write_json


UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]|A0A[A-Z0-9]+)(?:-\d+)?$"
)

ENZYME_COFACTOR_QUALITY_FLAGS = {
    "enzyme_cofactor_no_uniprot_accession",
    "enzyme_cofactor_no_uniprot_molecule_record",
    "enzyme_cofactor_no_label_from_molecules",
    "enzyme_cofactor_no_uniprotkb_cofactor_record",
    "enzyme_cofactor_no_label_from_uniprotkb",
    "enzyme_cofactor_no_enzyme_side_label",
    "enzyme_cofactor_explicit_no_cofactor",
    "enzyme_no_cofactor_from_uniprotkb_comment",
    "enzyme_cofactor_from_reaction_train_pair_fallback",
    "enzyme_core_cofactor_from_reaction_train_pair_fallback",
    "enzyme_has_no_cofactor_labels",
    "enzyme_has_no_core_cofactor_labels",
    "enzyme_side_cofactor_labels",
    "enzyme_side_core_cofactor_rescued",
    "enzyme_side_cofactor_rescued",
    "enzyme_side_uniprotkb_cofactor_labels",
}
_CHEBI_LABEL_LOOKUP_CACHE: dict[int, tuple[int, dict[str, set[str]]]] = {}


def normalize_uniprot_accession(value: Any) -> str:
    accession = str(value).strip()
    if "-" in accession:
        accession = accession.split("-", 1)[0]
    return accession


def is_uniprot_accession(value: Any) -> bool:
    return bool(UNIPROT_ACCESSION_RE.match(normalize_uniprot_accession(value)))


def extract_uniprot_accessions_from_source_entries(source_entries: Any) -> list[str]:
    """Extract UniProt accessions from collapsed train-pair provenance strings."""

    accessions: set[str] = set()
    for entry in str(source_entries or "").split("|"):
        for token in entry.split(":"):
            accession = normalize_uniprot_accession(token)
            if is_uniprot_accession(accession):
                accessions.add(accession)
    return sorted(accessions)


def _clean_sequence(sequence: Any) -> str:
    return str(sequence or "").replace(" ", "").replace("\n", "").replace("\r", "").upper()


def protein_uid(sequence: Any) -> str:
    clean = _clean_sequence(sequence)
    return f"uprot_{hashlib.sha1(clean.encode('utf-8')).hexdigest()[:16]}"


def _as_list(value: Any) -> list[str]:
    return union_list_column([value])


def _union(values: Iterable[Iterable[str]]) -> list[str]:
    out: set[str] = set()
    for labels in values:
        out.update(str(label) for label in labels if str(label))
    return sorted(out)


def _dedupe_labels(labels: Iterable[str]) -> list[str]:
    return sorted({str(label) for label in labels if str(label)})


@dataclass(frozen=True)
class _CofactorLabelBundle:
    """Cofactor labels plus the trainable broad bins derived from them."""

    cofactors: list[str]
    core: list[str]
    metal: list[str]
    auxiliary: list[str]
    architecture_bins: list[str]
    chemistry_bins: list[str]


def _empty_cofactor_bundle() -> _CofactorLabelBundle:
    return _CofactorLabelBundle([], [], [], [], [], [])


def _cofactor_bundle_from_lists(
    *,
    cofactors: Iterable[str],
    core: Iterable[str] = (),
    metal: Iterable[str] = (),
    auxiliary: Iterable[str] = (),
    include_unknown_bins: bool = False,
) -> _CofactorLabelBundle:
    """Create a normalized cofactor bundle.

    Source-specific bundles leave bins empty when that source contributed no
    labels. Active training bundles set ``include_unknown_bins=True`` so missing
    cofactors become explicit ``none_or_unknown`` / ``unknown`` targets instead
    of silently disappearing from the supervision.
    """

    cofactor_labels = _dedupe_labels(cofactors)
    has_bins = include_unknown_bins or bool(cofactor_labels)
    return _CofactorLabelBundle(
        cofactors=cofactor_labels,
        core=_dedupe_labels(core),
        metal=_dedupe_labels(metal),
        auxiliary=_dedupe_labels(auxiliary),
        architecture_bins=cofactor_architecture_bins(cofactor_labels) if has_bins else [],
        chemistry_bins=cofactor_chemistry_bins(cofactor_labels) if has_bins else [],
    )


def _cofactor_bundle_from_records(records: Iterable[dict[str, list[str]]]) -> _CofactorLabelBundle:
    record_list = list(records)
    return _cofactor_bundle_from_lists(
        cofactors=_union(record["cofactor_labels"] for record in record_list),
        core=_union(record["core_cofactor_labels"] for record in record_list),
        metal=_union(record["metal_ion_labels"] for record in record_list),
        auxiliary=_union(record["auxiliary_participant_labels"] for record in record_list),
    )


def _merge_cofactor_bundles(
    *bundles: _CofactorLabelBundle,
    include_unknown_bins: bool = False,
) -> _CofactorLabelBundle:
    return _cofactor_bundle_from_lists(
        cofactors=_union(bundle.cofactors for bundle in bundles),
        core=_union(bundle.core for bundle in bundles),
        metal=_union(bundle.metal for bundle in bundles),
        auxiliary=_union(bundle.auxiliary for bundle in bundles),
        include_unknown_bins=include_unknown_bins,
    )


def _active_fallback_and_diagnostic_bundles(
    enzyme_side: _CofactorLabelBundle,
    reaction_train: _CofactorLabelBundle,
) -> tuple[_CofactorLabelBundle, _CofactorLabelBundle, _CofactorLabelBundle]:
    """Return active training labels, reaction fallback labels, and diagnostics.

    UniProt enzyme-side evidence is the preferred supervision. Reaction-derived
    train labels are used only when UniProt gives no cofactor label for that
    enzyme; this avoids overwriting explicit enzyme-side information with
    reaction participant noise. The full reaction+UniProt union is still kept in
    diagnostic columns so the coverage tradeoff remains visible.
    """

    reaction_fallback = reaction_train if not enzyme_side.cofactors else _empty_cofactor_bundle()
    active = _merge_cofactor_bundles(
        enzyme_side,
        reaction_fallback,
        include_unknown_bins=True,
    )
    diagnostic_union = _merge_cofactor_bundles(enzyme_side, reaction_train)
    return active, reaction_fallback, diagnostic_union


def load_train_pair_accession_map(train_pairs_path: str | Path) -> dict[str, set[str]]:
    """Map active enzyme IDs to UniProt accessions found in train-pair provenance."""

    df = pd.read_csv(train_pairs_path, usecols=lambda col: col in {"enzyme_id", "protein_id", "source_entries"})
    enzyme_col = "enzyme_id" if "enzyme_id" in df.columns else "protein_id"
    if enzyme_col not in df.columns or "source_entries" not in df.columns:
        raise ValueError(f"{train_pairs_path} must contain enzyme_id/protein_id and source_entries")
    out: dict[str, set[str]] = {}
    for row in df[[enzyme_col, "source_entries"]].itertuples(index=False):
        enzyme_id = str(row[0])
        out.setdefault(enzyme_id, set()).update(
            extract_uniprot_accessions_from_source_entries(row[1])
        )
    return out


def load_sequence_accession_map(cleaned_uniprot_rhea_path: str | Path | None) -> dict[str, set[str]]:
    """Map active `uprot_<sequence-hash>` IDs to UniProt accessions."""

    if cleaned_uniprot_rhea_path is None:
        return {}
    path = Path(cleaned_uniprot_rhea_path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, sep="\t", usecols=lambda col: col in {"Entry", "Sequence"})
    if not {"Entry", "Sequence"}.issubset(df.columns):
        raise ValueError(f"{path} must contain Entry and Sequence columns")
    out: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        sequence = _clean_sequence(row.Sequence)
        entry = normalize_uniprot_accession(row.Entry)
        if not sequence or not is_uniprot_accession(entry):
            continue
        out.setdefault(protein_uid(sequence), set()).add(entry)
    return out


def _split_molecule_tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[.;|]", text) if token.strip()]


def extract_cofactor_labels_from_molecule_set(
    molecule_set: Any,
    *,
    cofactor_aliases: dict[str, list[str]] | None = None,
    token_cache: dict[str, tuple[str, ...]] | None = None,
) -> tuple[list[str], list[str]]:
    """Extract cofactor labels from UniProt molecule strings.

    ReactZyme stores per-UniProt molecule evidence mostly as dot-separated
    molecule SMILES.  The fast path uses structural/dictionary SMILES matching
    with a token cache.  If that finds nothing, we fall back to conservative
    name/alias matching so fixtures or future metadata with names still work.
    """

    labels: set[str] = set()
    flags: set[str] = set()
    aliases = cofactor_aliases or load_cofactor_aliases(None)
    cache = token_cache if token_cache is not None else {}
    tokens = _split_molecule_tokens(molecule_set)
    smiles_like = any(
        any(marker in token for marker in ("=", "#", "(", ")", "[", "]", "@", "\\", "/"))
        for token in tokens
    )
    for token in tokens:
        cached = cache.get(token)
        if cached is None:
            cached = tuple(extract_cofactor_labels_from_smiles(token, aliases))
            cache[token] = cached
        if cached:
            labels.update(cached)
            flags.add("enzyme_cofactor_from_uniprot_molecule_structure")
    if not labels and not smiles_like:
        name_labels, name_flags, _status = extract_cofactor_labels([str(molecule_set or "")], aliases)
        labels.update(name_labels)
        if name_labels:
            flags.update(f"enzyme_{flag}" for flag in name_flags)
    return sorted(labels), sorted(flags)


def load_uniprot_molecule_cofactors(
    uniprot_molecules_path: str | Path,
    *,
    needed_accessions: set[str] | None = None,
    cofactor_aliases: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Load enzyme-side cofactor labels from a UniProt molecule table."""

    path = Path(uniprot_molecules_path)
    df = pd.read_csv(path, sep="\t")
    required = {"uniprot_id", "molecules"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    aliases = cofactor_aliases or load_cofactor_aliases(None)
    token_cache: dict[str, tuple[str, ...]] = {}
    records: dict[str, dict[str, list[str]]] = {}
    total_rows = 0
    considered_rows = 0
    rows_with_labels = 0
    for row in df.itertuples(index=False):
        total_rows += 1
        accession = normalize_uniprot_accession(row.uniprot_id)
        if needed_accessions is not None and accession not in needed_accessions:
            continue
        considered_rows += 1
        labels, flags = extract_cofactor_labels_from_molecule_set(
            row.molecules,
            cofactor_aliases=aliases,
            token_cache=token_cache,
        )
        tiers = split_cofactor_label_tiers(labels)
        if labels:
            rows_with_labels += 1
        records[accession] = {
            "cofactor_labels": labels,
            "core_cofactor_labels": tiers["core_cofactor_labels"],
            "metal_ion_labels": tiers["metal_ion_labels"],
            "auxiliary_participant_labels": tiers["auxiliary_participant_labels"],
            "quality_flags": flags,
        }
    report = {
        "uniprot_molecule_rows_total": int(total_rows),
        "uniprot_molecule_rows_considered": int(considered_rows),
        "uniprot_molecule_rows_with_cofactor_labels": int(rows_with_labels),
        "unique_molecule_tokens_scanned": int(len(token_cache)),
    }
    return records, report


def _evidence_flags_from_uniprotkb_text(text: str) -> set[str]:
    flags: set[str] = set()
    evidence_codes = set(re.findall(r"ECO:\d+", str(text or "")))
    if "ECO:0000269" in evidence_codes:
        flags.add("enzyme_cofactor_from_uniprotkb_experimental")
    if "ECO:0000250" in evidence_codes:
        flags.add("enzyme_cofactor_from_uniprotkb_by_similarity")
    if evidence_codes - {"ECO:0000269", "ECO:0000250"}:
        flags.add("enzyme_cofactor_from_uniprotkb_other_evidence")
    return flags


def _chebi_label_lookup(cofactor_aliases: dict[str, list[str]]) -> dict[str, set[str]]:
    cache_key = id(cofactor_aliases)
    alias_count = sum(len(aliases) for aliases in cofactor_aliases.values())
    cached = _CHEBI_LABEL_LOOKUP_CACHE.get(cache_key)
    if cached is not None and cached[0] == alias_count:
        return cached[1]
    lookup: dict[str, set[str]] = {}
    for label, aliases in cofactor_aliases.items():
        for alias in aliases:
            alias_text = str(alias).strip().upper()
            if alias_text.startswith("CHEBI:"):
                lookup.setdefault(alias_text, set()).add(label)
    _CHEBI_LABEL_LOOKUP_CACHE[cache_key] = (alias_count, lookup)
    return lookup


def _uniprotkb_comment_declares_no_cofactor(text: str) -> bool:
    folded = str(text or "").casefold()
    patterns = (
        r"does\s+not\s+require\s+(?:a\s+|any\s+)?cofactor",
        r"do\s+not\s+require\s+(?:a\s+|any\s+)?cofactor",
        r"no\s+cofactor\s+(?:is\s+)?(?:required|needed)",
        r"cofactor\s+(?:is\s+)?not\s+(?:required|needed)",
        r"cofactor[-\s]+independent",
        r"does\s+not\s+(?:use|need)\s+(?:a\s+|any\s+)?cofactor",
        r"without\s+(?:a\s+)?cofactor",
    )
    return any(re.search(pattern, folded) is not None for pattern in patterns)


def extract_cofactor_labels_from_uniprotkb_comment(
    cofactor_text: Any,
    *,
    cofactor_aliases: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str]]:
    """Extract controlled cofactor labels from a UniProtKB cofactor comment.

    UniProtKB TSV exports cofactor comments as strings such as:
    ``COFACTOR: Name=Fe(2+); Xref=ChEBI:CHEBI:29033; Evidence=...``.
    We use the explicit ChEBI cross-reference when present and fall back to
    conservative name matching.
    """

    text = str(cofactor_text or "").strip()
    if not text:
        return [], []
    aliases = cofactor_aliases or load_cofactor_aliases(None)
    evidence_flags = _evidence_flags_from_uniprotkb_text(text)
    if _uniprotkb_comment_declares_no_cofactor(text):
        return [NO_COFACTOR_LABEL], sorted(
            {
                "enzyme_cofactor_explicit_no_cofactor",
                "enzyme_no_cofactor_from_uniprotkb_comment",
                *evidence_flags,
            }
        )
    names = [match.strip() for match in re.findall(r"Name=([^;]+)", text) if match.strip()]
    chebi_ids = [
        match.strip()
        for match in re.findall(r"Xref=ChEBI:(CHEBI:\d+)", text)
        if match.strip()
    ]
    labels: set[str] = set()
    chebi_lookup = _chebi_label_lookup(aliases)
    for chebi_id in chebi_ids:
        labels.update(chebi_lookup.get(chebi_id.upper(), set()))
    for chebi_id, name in zip(chebi_ids, names, strict=False):
        if chebi_lookup.get(chebi_id.upper()):
            continue
        inferred = cofactor_label_for_chebi_term(chebi_id, name)
        if inferred:
            labels.add(inferred)
            if inferred in METAL_ION_LABELS or inferred in {"metal"}:
                labels.add("metal")
    note_values = [
        match.strip()
        for match in re.findall(r"Note=([^;]+)", text)
        if match.strip()
    ]
    flags: list[str] = []
    if not labels and not chebi_ids:
        fallback_labels, flags, _status = extract_cofactor_labels(names + chebi_ids + note_values, aliases)
        labels.update(fallback_labels)
    if not labels:
        return [], sorted({"enzyme_cofactor_no_label_from_uniprotkb", *evidence_flags})
    prefixed_flags = {f"enzyme_{flag}" for flag in flags}
    prefixed_flags.add("enzyme_cofactor_from_uniprotkb_comment")
    prefixed_flags.update(evidence_flags)
    return labels, sorted(prefixed_flags)


def _read_uniprotkb_cofactor_cache(path: str | Path) -> dict[str, str]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    records: dict[str, str] = {}
    with cache_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return {}
        if "Entry" not in reader.fieldnames or "Cofactor" not in reader.fieldnames:
            raise ValueError(f"{cache_path} must contain Entry and Cofactor columns")
        for row in reader:
            accession = normalize_uniprot_accession(row.get("Entry"))
            if accession:
                records[accession] = str(row.get("Cofactor") or "")
    return records


def _write_uniprotkb_cofactor_cache(path: str | Path, records: dict[str, str]) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Entry", "Cofactor"], delimiter="\t")
        writer.writeheader()
        for accession in sorted(records):
            writer.writerow({"Entry": accession, "Cofactor": records[accession]})
    tmp_path.replace(cache_path)


def fetch_uniprotkb_cofactor_cache(
    accessions: Iterable[str],
    *,
    cache_path: str | Path,
    batch_size: int = 250,
    request_sleep_seconds: float = 0.05,
    timeout_seconds: int = 60,
    save_every_batches: int = 10,
) -> dict[str, Any]:
    """Fetch UniProtKB cofactor comments into a local TSV cache.

    The cache is intentionally simple (Entry, Cofactor) so annotation runs are
    reproducible and future reruns do not need to hit UniProtKB again.
    """

    import requests

    normalized = sorted(
        {normalize_uniprot_accession(accession) for accession in accessions if is_uniprot_accession(accession)}
    )
    records = _read_uniprotkb_cofactor_cache(cache_path)
    existing_rows = len(records)
    to_fetch = [accession for accession in normalized if accession not in records]
    if not to_fetch:
        return {
            "uniprotkb_cache_path": str(cache_path),
            "uniprotkb_requested_accessions": int(len(normalized)),
            "uniprotkb_cache_existing_rows": int(existing_rows),
            "uniprotkb_accessions_fetched": 0,
            "uniprotkb_batches_fetched": 0,
            "uniprotkb_fetch_failures": [],
        }

    failures: list[dict[str, Any]] = []
    fetched = 0
    batches = 0
    endpoint = "https://rest.uniprot.org/uniprotkb/accessions"
    for start in range(0, len(to_fetch), batch_size):
        batch = to_fetch[start : start + batch_size]
        params = {
            "accessions": ",".join(batch),
            "fields": "accession,cc_cofactor",
            "format": "tsv",
        }
        last_error = None
        response_text = None
        for attempt in range(3):
            try:
                response = requests.get(endpoint, params=params, timeout=timeout_seconds)
                response.raise_for_status()
                response_text = response.text
                break
            except Exception as exc:  # pragma: no cover - network timing dependent
                last_error = str(exc)
                time.sleep(min(2.0 * (attempt + 1), 5.0))
        if response_text is None:
            failures.append({"batch_start": start, "batch_size": len(batch), "error": last_error})
            continue

        returned: set[str] = set()
        reader = csv.DictReader(response_text.splitlines(), delimiter="\t")
        for row in reader:
            accession = normalize_uniprot_accession(row.get("Entry"))
            if not accession:
                continue
            returned.add(accession)
            records[accession] = str(row.get("Cofactor") or "")
        for accession in batch:
            records.setdefault(accession, "")
        fetched += len(batch)
        batches += 1
        if batches % save_every_batches == 0:
            _write_uniprotkb_cofactor_cache(cache_path, records)
        if request_sleep_seconds > 0:
            time.sleep(request_sleep_seconds)

    _write_uniprotkb_cofactor_cache(cache_path, records)
    return {
        "uniprotkb_cache_path": str(cache_path),
        "uniprotkb_requested_accessions": int(len(normalized)),
        "uniprotkb_cache_existing_rows": int(existing_rows),
        "uniprotkb_accessions_fetched": int(fetched),
        "uniprotkb_batches_fetched": int(batches),
        "uniprotkb_fetch_failures": failures,
    }


def load_uniprotkb_cofactor_comments(
    uniprotkb_cofactor_path: str | Path | None,
    *,
    needed_accessions: set[str] | None = None,
    cofactor_aliases: dict[str, list[str]] | None = None,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    """Load enzyme-side cofactor labels from cached UniProtKB comments."""

    if uniprotkb_cofactor_path is None:
        return {}, {
            "uniprotkb_cofactor_cache_path": None,
            "uniprotkb_cofactor_rows_total": 0,
            "uniprotkb_cofactor_rows_considered": 0,
            "uniprotkb_cofactor_rows_with_comments": 0,
            "uniprotkb_cofactor_rows_with_labels": 0,
        }
    path = Path(uniprotkb_cofactor_path)
    if not path.exists():
        return {}, {
            "uniprotkb_cofactor_cache_path": str(path),
            "uniprotkb_cofactor_rows_total": 0,
            "uniprotkb_cofactor_rows_considered": 0,
            "uniprotkb_cofactor_rows_with_comments": 0,
            "uniprotkb_cofactor_rows_with_labels": 0,
        }

    aliases = cofactor_aliases or load_cofactor_aliases(None)
    raw_records = _read_uniprotkb_cofactor_cache(path)
    records: dict[str, dict[str, list[str]]] = {}
    considered = 0
    rows_with_comments = 0
    rows_with_labels = 0
    for accession, text in raw_records.items():
        if needed_accessions is not None and accession not in needed_accessions:
            continue
        considered += 1
        if str(text).strip():
            rows_with_comments += 1
        labels, flags = extract_cofactor_labels_from_uniprotkb_comment(
            text,
            cofactor_aliases=aliases,
        )
        tiers = split_cofactor_label_tiers(labels)
        if labels:
            rows_with_labels += 1
        records[accession] = {
            "cofactor_labels": labels,
            "core_cofactor_labels": tiers["core_cofactor_labels"],
            "metal_ion_labels": tiers["metal_ion_labels"],
            "auxiliary_participant_labels": tiers["auxiliary_participant_labels"],
            "quality_flags": flags,
        }
    return records, {
        "uniprotkb_cofactor_cache_path": str(path),
        "uniprotkb_cofactor_rows_total": int(len(raw_records)),
        "uniprotkb_cofactor_rows_considered": int(considered),
        "uniprotkb_cofactor_rows_with_comments": int(rows_with_comments),
        "uniprotkb_cofactor_rows_with_labels": int(rows_with_labels),
    }


def _top_counts(rows: Iterable[Iterable[str]], n: int = 25) -> dict[str, int]:
    return dict(Counter(label for row in rows for label in row).most_common(n))


def _nonempty_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(lambda value: bool(_as_list(value))).sum())


def _update_vocabs(vocabs_path: Path, labels: pd.DataFrame) -> None:
    vocabs = {}
    if vocabs_path.exists():
        import json

        vocabs = json.loads(vocabs_path.read_text(encoding="utf-8"))
    for column, key in (
        ("enzyme_derived_cofactor_labels_train", "enzyme_derived_cofactor_labels"),
        ("enzyme_derived_core_cofactor_labels_train", "enzyme_derived_core_cofactor_labels"),
        ("enzyme_derived_metal_ion_labels_train", "enzyme_derived_metal_ion_labels"),
        (
            "enzyme_derived_auxiliary_participant_labels_train",
            "enzyme_derived_auxiliary_participant_labels",
        ),
        ("uniprot_cofactor_labels_train", "uniprot_cofactor_labels"),
        ("uniprot_core_cofactor_labels_train", "uniprot_core_cofactor_labels"),
        ("uniprot_metal_ion_labels_train", "uniprot_metal_ion_labels"),
        (
            "uniprot_auxiliary_participant_labels_train",
            "uniprot_auxiliary_participant_labels",
        ),
        ("reaction_fallback_cofactor_labels_train", "reaction_fallback_cofactor_labels"),
        ("reaction_fallback_core_cofactor_labels_train", "reaction_fallback_core_cofactor_labels"),
        ("reaction_fallback_metal_ion_labels_train", "reaction_fallback_metal_ion_labels"),
        (
            "reaction_fallback_auxiliary_participant_labels_train",
            "reaction_fallback_auxiliary_participant_labels",
        ),
        ("enzyme_uniprotkb_cofactor_labels_train", "enzyme_uniprotkb_cofactor_labels"),
        ("combined_cofactor_labels_train", "combined_cofactor_labels"),
        ("combined_core_cofactor_labels_train", "combined_core_cofactor_labels"),
        ("combined_metal_ion_labels_train", "combined_metal_ion_labels"),
        ("combined_auxiliary_participant_labels_train", "combined_auxiliary_participant_labels"),
        ("cofactor_architecture_bins_train", "cofactor_architecture_bins"),
        ("cofactor_chemistry_bins_train", "cofactor_chemistry_bins"),
        ("uniprot_cofactor_architecture_bins_train", "uniprot_cofactor_architecture_bins"),
        ("uniprot_cofactor_chemistry_bins_train", "uniprot_cofactor_chemistry_bins"),
        (
            "reaction_fallback_cofactor_architecture_bins_train",
            "reaction_fallback_cofactor_architecture_bins",
        ),
        (
            "reaction_fallback_cofactor_chemistry_bins_train",
            "reaction_fallback_cofactor_chemistry_bins",
        ),
        (
            "reaction_plus_uniprot_cofactor_architecture_bins_train",
            "reaction_plus_uniprot_cofactor_architecture_bins",
        ),
        (
            "reaction_plus_uniprot_cofactor_chemistry_bins_train",
            "reaction_plus_uniprot_cofactor_chemistry_bins",
        ),
        ("reaction_plus_uniprot_cofactor_labels_train", "reaction_plus_uniprot_cofactor_labels"),
        (
            "reaction_plus_uniprot_core_cofactor_labels_train",
            "reaction_plus_uniprot_core_cofactor_labels",
        ),
        ("reaction_plus_uniprot_metal_ion_labels_train", "reaction_plus_uniprot_metal_ion_labels"),
        (
            "reaction_plus_uniprot_auxiliary_participant_labels_train",
            "reaction_plus_uniprot_auxiliary_participant_labels",
        ),
    ):
        vocabs[key] = sorted({label for labels_value in labels[column] for label in _as_list(labels_value)})
    vocabs["cofactor_architecture_bins"] = sorted(
        set(vocabs.get("cofactor_architecture_bins", [])) | set(COFACTOR_ARCHITECTURE_BINS)
    )
    vocabs["cofactor_chemistry_bins"] = sorted(
        set(vocabs.get("cofactor_chemistry_bins", [])) | set(COFACTOR_CHEMISTRY_BINS)
    )
    write_json(vocabs_path, vocabs)


def _reaction_feature_lookup(reaction_features_path: str | Path) -> dict[str, dict[str, list[str]]]:
    features = pd.read_parquet(
        reaction_features_path,
        columns=[
            "reaction_id",
            "cofactor_labels",
            "core_cofactor_labels",
            "metal_ion_labels",
            "auxiliary_participant_labels",
        ],
    )
    return {
        str(row["reaction_id"]): {
            "cofactor_labels": _as_list(row.get("cofactor_labels")),
            "core_cofactor_labels": _as_list(row.get("core_cofactor_labels")),
            "metal_ion_labels": _as_list(row.get("metal_ion_labels")),
            "auxiliary_participant_labels": _as_list(row.get("auxiliary_participant_labels")),
        }
        for row in features.to_dict("records")
    }


def _write_uniprot_cofactor_label_dictionary(path: Path, labels: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    for column, source_name in (
        ("uniprot_cofactor_labels_train", "uniprot_all"),
        ("uniprot_core_cofactor_labels_train", "uniprot_core"),
        ("uniprot_metal_ion_labels_train", "uniprot_metal_ion"),
        ("uniprot_auxiliary_participant_labels_train", "uniprot_auxiliary"),
        ("enzyme_uniprotkb_cofactor_labels_train", "uniprotkb_comment"),
    ):
        counter = Counter(label for value in labels[column] for label in _as_list(value))
        for label, count in counter.items():
            rows.append(
                {
                    "label": label,
                    "tier": cofactor_label_tier(label),
                    "source_bucket": source_name,
                    "enzyme_count": int(count),
                }
            )
    dictionary = pd.DataFrame(rows)
    if not dictionary.empty:
        dictionary = dictionary.sort_values(
            ["source_bucket", "tier", "enzyme_count", "label"],
            ascending=[True, True, False, True],
        )
    dictionary.to_csv(path, index=False)


def _write_fine_cofactor_label_dictionary(path: Path, labels: pd.DataFrame) -> None:
    """Write fine cofactor evidence labels.

    This is deliberately not the primary cofactor training dictionary. The
    trainable cofactor supervision is the architecture/chemistry bin dictionary;
    fine labels are retained for auditability and reaction-overlap diagnostics.
    """

    rows: list[dict[str, Any]] = []
    for column, source_name in (
        ("combined_cofactor_labels_train", "active_all"),
        ("combined_core_cofactor_labels_train", "active_core"),
        ("combined_metal_ion_labels_train", "active_metal_ion"),
        ("combined_auxiliary_participant_labels_train", "active_auxiliary"),
        ("reaction_fallback_cofactor_labels_train", "reaction_fallback_all"),
        ("reaction_fallback_core_cofactor_labels_train", "reaction_fallback_core"),
        ("reaction_fallback_metal_ion_labels_train", "reaction_fallback_metal_ion"),
        ("reaction_fallback_auxiliary_participant_labels_train", "reaction_fallback_auxiliary"),
    ):
        counter = Counter(label for value in labels[column] for label in _as_list(value))
        for label, count in counter.items():
            rows.append(
                {
                    "label": label,
                    "tier": cofactor_label_tier(label),
                    "source_bucket": source_name,
                    "enzyme_count": int(count),
                }
            )
    dictionary = pd.DataFrame(rows)
    if not dictionary.empty:
        dictionary = dictionary.sort_values(
            ["source_bucket", "tier", "enzyme_count", "label"],
            ascending=[True, True, False, True],
        )
    dictionary.to_csv(path, index=False)


def _write_cofactor_bin_dictionary(path: Path, labels: pd.DataFrame) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for column, source_name, bin_type in (
        ("cofactor_architecture_bins_train", "active", "architecture"),
        ("cofactor_chemistry_bins_train", "active", "chemistry"),
        ("uniprot_cofactor_architecture_bins_train", "uniprot", "architecture"),
        ("uniprot_cofactor_chemistry_bins_train", "uniprot", "chemistry"),
        ("reaction_fallback_cofactor_architecture_bins_train", "reaction_fallback", "architecture"),
        ("reaction_fallback_cofactor_chemistry_bins_train", "reaction_fallback", "chemistry"),
        ("reaction_plus_uniprot_cofactor_architecture_bins_train", "reaction_plus_uniprot", "architecture"),
        ("reaction_plus_uniprot_cofactor_chemistry_bins_train", "reaction_plus_uniprot", "chemistry"),
    ):
        counter = Counter(label for value in labels[column] for label in _as_list(value))
        for label, count in counter.items():
            seen.add((source_name, bin_type, label))
            rows.append(
                {
                    "bin": label,
                    "bin_type": bin_type,
                    "source_bucket": source_name,
                    "enzyme_count": int(count),
                }
            )
    for bin_type, bin_values in (
        ("architecture", COFACTOR_ARCHITECTURE_BINS),
        ("chemistry", COFACTOR_CHEMISTRY_BINS),
    ):
        for label in bin_values:
            key = ("active", bin_type, label)
            if key in seen:
                continue
            rows.append(
                {
                    "bin": label,
                    "bin_type": bin_type,
                    "source_bucket": "active",
                    "enzyme_count": 0,
                }
            )
    dictionary = pd.DataFrame(rows)
    if not dictionary.empty:
        dictionary = dictionary.sort_values(
            ["source_bucket", "bin_type", "enzyme_count", "bin"],
            ascending=[True, True, False, True],
        )
    dictionary.to_csv(path, index=False)


def _write_primary_cofactor_dictionary(path: Path, labels: pd.DataFrame) -> None:
    """Write the redesigned cofactor dictionary used for training targets."""

    rows: list[dict[str, Any]] = []
    for column, bin_type, bin_values in (
        ("cofactor_architecture_bins_train", "architecture", COFACTOR_ARCHITECTURE_BINS),
        ("cofactor_chemistry_bins_train", "chemistry", COFACTOR_CHEMISTRY_BINS),
    ):
        counter = Counter(label for value in labels[column] for label in _as_list(value))
        for label in bin_values:
            rows.append(
                {
                    "bin": label,
                    "bin_type": bin_type,
                    "enzyme_count": int(counter.get(label, 0)),
                    "is_primary_training_target": True,
                }
            )
    dictionary = pd.DataFrame(rows).sort_values(
        ["bin_type", "enzyme_count", "bin"],
        ascending=[True, False, True],
    )
    dictionary.to_csv(path, index=False)


def enhance_enzyme_cofactor_annotations(
    *,
    capability_dir: str | Path,
    train_pairs_path: str | Path,
    uniprot_molecules_path: str | Path,
    cleaned_uniprot_rhea_path: str | Path | None = None,
    reaction_features_path: str | Path | None = None,
    cofactor_dictionary_path: str | Path | None = None,
    uniprotkb_cofactor_path: str | Path | None = None,
    fetch_uniprotkb_cofactors: bool = False,
    uniprotkb_fetch_scope: str = "missing_cofactor",
    uniprotkb_batch_size: int = 250,
    write_csv: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add enzyme-side and combined cofactor labels to capability artifacts."""

    cap_dir = Path(capability_dir)
    labels_path = cap_dir / "enzyme_capability_labels.parquet"
    pairs_path = cap_dir / "pair_capability_training.parquet"
    vocabs_path = cap_dir / "enzyme_label_vocabs.json"
    reaction_path = Path(reaction_features_path) if reaction_features_path else cap_dir / "reaction_features.parquet"
    labels = pd.read_parquet(labels_path)
    pair_training = pd.read_parquet(pairs_path)
    source_accessions = load_train_pair_accession_map(train_pairs_path)
    sequence_accessions = load_sequence_accession_map(cleaned_uniprot_rhea_path)

    enzyme_ids = set(labels["enzyme_id"].astype(str))
    needed_accessions = {
        accession
        for enzyme_id in enzyme_ids
        for accession in (source_accessions.get(enzyme_id, set()) | sequence_accessions.get(enzyme_id, set()))
    }
    aliases = load_cofactor_aliases(cofactor_dictionary_path)
    accession_labels, molecule_report = load_uniprot_molecule_cofactors(
        uniprot_molecules_path,
        needed_accessions=needed_accessions,
        cofactor_aliases=aliases,
    )
    fetch_report: dict[str, Any] = {}
    if fetch_uniprotkb_cofactors:
        if uniprotkb_cofactor_path is None:
            uniprotkb_cofactor_path = cap_dir / "uniprotkb_cofactor_comments.tsv"
        if uniprotkb_fetch_scope not in {"missing_cofactor", "all_accessions"}:
            raise ValueError("uniprotkb_fetch_scope must be 'missing_cofactor' or 'all_accessions'")
        fetch_accessions = set(needed_accessions)
        if uniprotkb_fetch_scope == "missing_cofactor":
            missing_enzyme_ids = {
                str(row["enzyme_id"])
                for row in labels.to_dict("records")
                if not _as_list(row.get("enzyme_derived_cofactor_labels_train"))
                and not _as_list(row.get("enzyme_uniprotkb_cofactor_labels_train"))
            }
            fetch_accessions = {
                accession
                for enzyme_id in missing_enzyme_ids
                for accession in (
                    source_accessions.get(enzyme_id, set())
                    | sequence_accessions.get(enzyme_id, set())
                )
            }
        fetch_report = fetch_uniprotkb_cofactor_cache(
            fetch_accessions,
            cache_path=uniprotkb_cofactor_path,
            batch_size=uniprotkb_batch_size,
        )
    uniprotkb_labels, uniprotkb_report = load_uniprotkb_cofactor_comments(
        uniprotkb_cofactor_path,
        needed_accessions=needed_accessions,
        cofactor_aliases=aliases,
    )

    enhanced_rows: list[dict[str, Any]] = []
    for row in labels.to_dict("records"):
        enzyme_id = str(row["enzyme_id"])
        provenance_accs = set(source_accessions.get(enzyme_id, set()))
        sequence_accs = set(sequence_accessions.get(enzyme_id, set()))
        accessions = sorted(provenance_accs | sequence_accs)
        molecule_accessions = sorted(accession for accession in accessions if accession in accession_labels)
        uniprotkb_accessions = sorted(accession for accession in accessions if accession in uniprotkb_labels)
        molecule_records = [accession_labels[accession] for accession in molecule_accessions]
        uniprotkb_records = [uniprotkb_labels[accession] for accession in uniprotkb_accessions]
        derived_records = molecule_records + uniprotkb_records
        enzyme_side_bundle = _cofactor_bundle_from_records(derived_records)
        derived_flags = _union(record["quality_flags"] for record in derived_records)
        molecule_cofactors = _cofactor_bundle_from_records(molecule_records).cofactors
        uniprotkb_cofactors = _cofactor_bundle_from_records(uniprotkb_records).cofactors

        reaction_bundle = _cofactor_bundle_from_lists(
            cofactors=_as_list(row.get("cofactor_labels_train")),
            core=_as_list(row.get("core_cofactor_labels_train")),
            metal=_as_list(row.get("metal_ion_labels_train")),
            auxiliary=_as_list(row.get("auxiliary_participant_labels_train")),
        )
        active_bundle, reaction_fallback_bundle, reaction_plus_uniprot_bundle = (
            _active_fallback_and_diagnostic_bundles(enzyme_side_bundle, reaction_bundle)
        )
        derived_cofactors = enzyme_side_bundle.cofactors
        derived_core = enzyme_side_bundle.core
        derived_metal = enzyme_side_bundle.metal
        derived_aux = enzyme_side_bundle.auxiliary
        reaction_cofactors = reaction_bundle.cofactors
        reaction_core = reaction_bundle.core
        combined_cofactors = active_bundle.cofactors
        combined_core = active_bundle.core
        combined_metal = active_bundle.metal
        combined_aux = active_bundle.auxiliary
        reaction_fallback_cofactors = reaction_fallback_bundle.cofactors
        reaction_fallback_core = reaction_fallback_bundle.core
        reaction_fallback_metal = reaction_fallback_bundle.metal
        reaction_fallback_aux = reaction_fallback_bundle.auxiliary
        reaction_plus_uniprot_cofactors = reaction_plus_uniprot_bundle.cofactors
        reaction_plus_uniprot_core = reaction_plus_uniprot_bundle.core
        reaction_plus_uniprot_metal = reaction_plus_uniprot_bundle.metal
        reaction_plus_uniprot_aux = reaction_plus_uniprot_bundle.auxiliary

        quality_flags = set(_as_list(row.get("quality_flags"))) - ENZYME_COFACTOR_QUALITY_FLAGS
        quality_flags.update(derived_flags)
        if not accessions:
            quality_flags.add("enzyme_cofactor_no_uniprot_accession")
        if accessions and not molecule_accessions:
            quality_flags.add("enzyme_cofactor_no_uniprot_molecule_record")
        elif molecule_accessions and not molecule_cofactors:
            quality_flags.add("enzyme_cofactor_no_label_from_molecules")
        if uniprotkb_cofactor_path is not None:
            if accessions and not uniprotkb_accessions:
                quality_flags.add("enzyme_cofactor_no_uniprotkb_cofactor_record")
            elif uniprotkb_accessions and not uniprotkb_cofactors:
                quality_flags.add("enzyme_cofactor_no_label_from_uniprotkb")
        if not derived_cofactors:
            quality_flags.add("enzyme_cofactor_no_enzyme_side_label")
        if derived_cofactors:
            quality_flags.add("enzyme_side_cofactor_labels")
        if uniprotkb_cofactors:
            quality_flags.add("enzyme_side_uniprotkb_cofactor_labels")
        if set(derived_core) - set(reaction_core):
            quality_flags.add("enzyme_side_core_cofactor_rescued")
        if set(derived_cofactors) - set(reaction_cofactors):
            quality_flags.add("enzyme_side_cofactor_rescued")
        if reaction_fallback_cofactors:
            quality_flags.add("enzyme_cofactor_from_reaction_train_pair_fallback")
        if reaction_fallback_core:
            quality_flags.add("enzyme_core_cofactor_from_reaction_train_pair_fallback")
        if not combined_cofactors:
            quality_flags.add("enzyme_has_no_cofactor_labels")
        if not combined_core:
            quality_flags.add("enzyme_has_no_core_cofactor_labels")

        source_parts = []
        if molecule_cofactors:
            source_parts.append("uniprot_molecules")
        if uniprotkb_cofactors:
            source_parts.append("uniprotkb_cofactor_comments")
        if reaction_fallback_cofactors:
            source_parts.append("reaction_train_pairs_fallback")
        if source_parts:
            label_source = "+".join(source_parts)
        else:
            label_source = "missing"

        row.update(
            {
                "enzyme_cofactor_accessions": accessions,
                "enzyme_cofactor_source_accessions": sorted(provenance_accs),
                "enzyme_cofactor_sequence_accessions": sorted(sequence_accs),
                "enzyme_cofactor_molecule_accessions": molecule_accessions,
                "enzyme_cofactor_uniprotkb_accessions": uniprotkb_accessions,
                "enzyme_derived_cofactor_labels_train": derived_cofactors,
                "enzyme_derived_core_cofactor_labels_train": derived_core,
                "enzyme_derived_metal_ion_labels_train": derived_metal,
                "enzyme_derived_auxiliary_participant_labels_train": derived_aux,
                "uniprot_cofactor_labels_train": derived_cofactors,
                "uniprot_core_cofactor_labels_train": derived_core,
                "uniprot_metal_ion_labels_train": derived_metal,
                "uniprot_auxiliary_participant_labels_train": derived_aux,
                "reaction_fallback_cofactor_labels_train": reaction_fallback_cofactors,
                "reaction_fallback_core_cofactor_labels_train": reaction_fallback_core,
                "reaction_fallback_metal_ion_labels_train": reaction_fallback_metal,
                "reaction_fallback_auxiliary_participant_labels_train": reaction_fallback_aux,
                "enzyme_uniprotkb_cofactor_labels_train": uniprotkb_cofactors,
                "combined_cofactor_labels_train": combined_cofactors,
                "combined_core_cofactor_labels_train": combined_core,
                "combined_metal_ion_labels_train": combined_metal,
                "combined_auxiliary_participant_labels_train": combined_aux,
                "cofactor_architecture_bins_train": active_bundle.architecture_bins,
                "cofactor_chemistry_bins_train": active_bundle.chemistry_bins,
                "uniprot_cofactor_architecture_bins_train": enzyme_side_bundle.architecture_bins,
                "uniprot_cofactor_chemistry_bins_train": enzyme_side_bundle.chemistry_bins,
                "reaction_fallback_cofactor_architecture_bins_train": reaction_fallback_bundle.architecture_bins,
                "reaction_fallback_cofactor_chemistry_bins_train": reaction_fallback_bundle.chemistry_bins,
                "reaction_plus_uniprot_cofactor_labels_train": reaction_plus_uniprot_cofactors,
                "reaction_plus_uniprot_core_cofactor_labels_train": reaction_plus_uniprot_core,
                "reaction_plus_uniprot_metal_ion_labels_train": reaction_plus_uniprot_metal,
                "reaction_plus_uniprot_auxiliary_participant_labels_train": reaction_plus_uniprot_aux,
                "reaction_plus_uniprot_cofactor_architecture_bins_train": reaction_plus_uniprot_bundle.architecture_bins,
                "reaction_plus_uniprot_cofactor_chemistry_bins_train": reaction_plus_uniprot_bundle.chemistry_bins,
                "num_enzyme_derived_cofactors": len(derived_cofactors),
                "num_enzyme_derived_core_cofactors": len(derived_core),
                "num_combined_cofactors": len(combined_cofactors),
                "num_combined_core_cofactors": len(combined_core),
                "enzyme_cofactor_label_source": label_source,
                "quality_flags": sorted(quality_flags),
            }
        )
        enhanced_rows.append(row)

    enhanced_labels = pd.DataFrame(enhanced_rows)
    label_lookup = {str(row["enzyme_id"]): row for row in enhanced_rows}
    reaction_lookup = _reaction_feature_lookup(reaction_path)
    pair_rows: list[dict[str, Any]] = []
    for row in pair_training.to_dict("records"):
        enzyme_id = str(row["enzyme_id"])
        reaction_id = str(row["reaction_id"])
        enzyme_row = label_lookup[enzyme_id]
        reaction_row = reaction_lookup.get(
            reaction_id,
            {
                "cofactor_labels": [],
                "core_cofactor_labels": [],
                "metal_ion_labels": [],
                "auxiliary_participant_labels": [],
            },
        )
        row.update(
            {
                "enzyme_derived_cofactor_overlap": intersection_size(
                    reaction_row["cofactor_labels"],
                    enzyme_row["enzyme_derived_cofactor_labels_train"],
                ),
                "enzyme_derived_core_cofactor_overlap": intersection_size(
                    reaction_row["core_cofactor_labels"],
                    enzyme_row["enzyme_derived_core_cofactor_labels_train"],
                ),
                "enzyme_derived_metal_ion_overlap": intersection_size(
                    reaction_row["metal_ion_labels"],
                    enzyme_row["enzyme_derived_metal_ion_labels_train"],
                ),
                "enzyme_derived_auxiliary_participant_overlap": intersection_size(
                    reaction_row["auxiliary_participant_labels"],
                    enzyme_row["enzyme_derived_auxiliary_participant_labels_train"],
                ),
                "combined_cofactor_overlap": intersection_size(
                    reaction_row["cofactor_labels"],
                    enzyme_row["combined_cofactor_labels_train"],
                ),
                "combined_core_cofactor_overlap": intersection_size(
                    reaction_row["core_cofactor_labels"],
                    enzyme_row["combined_core_cofactor_labels_train"],
                ),
                "combined_metal_ion_overlap": intersection_size(
                    reaction_row["metal_ion_labels"],
                    enzyme_row["combined_metal_ion_labels_train"],
                ),
                "combined_auxiliary_participant_overlap": intersection_size(
                    reaction_row["auxiliary_participant_labels"],
                    enzyme_row["combined_auxiliary_participant_labels_train"],
                ),
                "reaction_fallback_cofactor_overlap": intersection_size(
                    reaction_row["cofactor_labels"],
                    enzyme_row["reaction_fallback_cofactor_labels_train"],
                ),
                "reaction_fallback_core_cofactor_overlap": intersection_size(
                    reaction_row["core_cofactor_labels"],
                    enzyme_row["reaction_fallback_core_cofactor_labels_train"],
                ),
                "reaction_fallback_metal_ion_overlap": intersection_size(
                    reaction_row["metal_ion_labels"],
                    enzyme_row["reaction_fallback_metal_ion_labels_train"],
                ),
                "reaction_fallback_auxiliary_participant_overlap": intersection_size(
                    reaction_row["auxiliary_participant_labels"],
                    enzyme_row["reaction_fallback_auxiliary_participant_labels_train"],
                ),
                "reaction_plus_uniprot_cofactor_overlap": intersection_size(
                    reaction_row["cofactor_labels"],
                    enzyme_row["reaction_plus_uniprot_cofactor_labels_train"],
                ),
                "reaction_plus_uniprot_core_cofactor_overlap": intersection_size(
                    reaction_row["core_cofactor_labels"],
                    enzyme_row["reaction_plus_uniprot_core_cofactor_labels_train"],
                ),
                "reaction_plus_uniprot_metal_ion_overlap": intersection_size(
                    reaction_row["metal_ion_labels"],
                    enzyme_row["reaction_plus_uniprot_metal_ion_labels_train"],
                ),
                "reaction_plus_uniprot_auxiliary_participant_overlap": intersection_size(
                    reaction_row["auxiliary_participant_labels"],
                    enzyme_row["reaction_plus_uniprot_auxiliary_participant_labels_train"],
                ),
            }
        )
        pair_rows.append(row)
    enhanced_pairs = pd.DataFrame(pair_rows)
    if "source_split" in enhanced_pairs.columns:
        assert set(enhanced_pairs["source_split"].unique()) == {"train"}

    report: dict[str, Any] = {
        "capability_dir": str(cap_dir),
        "labeling_strategy": "uniprot_enzyme_cofactors_with_reaction_fallback",
        "combined_labels_definition": (
            "combined_*_cofactor_labels_train use UniProt enzyme-side labels "
            "from UniProt molecule records plus UniProtKB cofactor comments when available. "
            "For enzymes with no UniProt cofactor label, train reaction-derived labels are used "
            "as a fallback and stored in reaction_fallback_* columns. "
            "reaction_plus_uniprot_* remains the full pooled diagnostic view."
        ),
        "train_pairs_path": str(train_pairs_path),
        "uniprot_molecules_path": str(uniprot_molecules_path),
        "cleaned_uniprot_rhea_path": str(cleaned_uniprot_rhea_path) if cleaned_uniprot_rhea_path else None,
        **molecule_report,
        **fetch_report,
        **uniprotkb_report,
        "num_enzymes_total": int(len(enhanced_labels)),
        "num_enzymes_with_source_entry_accession": int(
            enhanced_labels["enzyme_cofactor_source_accessions"].map(bool).sum()
        ),
        "num_enzymes_with_sequence_hash_accession": int(
            enhanced_labels["enzyme_cofactor_sequence_accessions"].map(bool).sum()
        ),
        "num_enzymes_with_uniprot_molecule_record": int(
            enhanced_labels["enzyme_cofactor_molecule_accessions"].map(bool).sum()
        ),
        "num_enzymes_with_uniprotkb_cofactor_record": int(
            enhanced_labels["enzyme_cofactor_uniprotkb_accessions"].map(bool).sum()
        ),
        "num_enzymes_with_uniprotkb_cofactor_labels": _nonempty_count(
            enhanced_labels,
            "enzyme_uniprotkb_cofactor_labels_train",
        ),
        "num_enzymes_with_explicit_no_cofactor": int(
            enhanced_labels["combined_cofactor_labels_train"].map(
                lambda value: NO_COFACTOR_LABEL in set(_as_list(value))
            ).sum()
        ),
        "reaction_derived_any_cofactor_enzymes": _nonempty_count(enhanced_labels, "cofactor_labels_train"),
        "reaction_derived_core_cofactor_enzymes": _nonempty_count(enhanced_labels, "core_cofactor_labels_train"),
        "enzyme_derived_any_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "enzyme_derived_cofactor_labels_train",
        ),
        "enzyme_derived_core_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "enzyme_derived_core_cofactor_labels_train",
        ),
        "uniprot_only_any_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_cofactor_labels_train",
        ),
        "uniprot_only_core_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_core_cofactor_labels_train",
        ),
        "uniprot_only_metal_ion_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_metal_ion_labels_train",
        ),
        "uniprot_only_auxiliary_participant_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_auxiliary_participant_labels_train",
        ),
        "reaction_fallback_any_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_fallback_cofactor_labels_train",
        ),
        "reaction_fallback_core_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_fallback_core_cofactor_labels_train",
        ),
        "reaction_fallback_metal_ion_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_fallback_metal_ion_labels_train",
        ),
        "reaction_fallback_auxiliary_participant_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_fallback_auxiliary_participant_labels_train",
        ),
        "active_cofactor_architecture_bin_enzymes": _nonempty_count(
            enhanced_labels,
            "cofactor_architecture_bins_train",
        ),
        "active_cofactor_chemistry_bin_enzymes": _nonempty_count(
            enhanced_labels,
            "cofactor_chemistry_bins_train",
        ),
        "uniprot_cofactor_architecture_bin_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_cofactor_architecture_bins_train",
        ),
        "uniprot_cofactor_chemistry_bin_enzymes": _nonempty_count(
            enhanced_labels,
            "uniprot_cofactor_chemistry_bins_train",
        ),
        "combined_any_cofactor_enzymes": _nonempty_count(enhanced_labels, "combined_cofactor_labels_train"),
        "combined_core_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "combined_core_cofactor_labels_train",
        ),
        "reaction_plus_uniprot_any_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_plus_uniprot_cofactor_labels_train",
        ),
        "reaction_plus_uniprot_core_cofactor_enzymes": _nonempty_count(
            enhanced_labels,
            "reaction_plus_uniprot_core_cofactor_labels_train",
        ),
        "new_any_cofactor_enzymes_from_enzyme_side": int(
            (
                ~enhanced_labels["cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
                & enhanced_labels["enzyme_derived_cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
            ).sum()
        ),
        "new_core_cofactor_enzymes_from_enzyme_side": int(
            (
                ~enhanced_labels["core_cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
                & enhanced_labels["enzyme_derived_core_cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
            ).sum()
        ),
        "new_any_cofactor_enzymes_from_uniprotkb": int(
            (
                ~enhanced_labels["cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
                & enhanced_labels["enzyme_uniprotkb_cofactor_labels_train"].map(lambda value: bool(_as_list(value)))
            ).sum()
        ),
        "top_reaction_derived_core_cofactor_labels": _top_counts(
            enhanced_labels["core_cofactor_labels_train"].map(_as_list)
        ),
        "top_enzyme_derived_core_cofactor_labels": _top_counts(
            enhanced_labels["enzyme_derived_core_cofactor_labels_train"].map(_as_list)
        ),
        "top_combined_core_cofactor_labels": _top_counts(
            enhanced_labels["combined_core_cofactor_labels_train"].map(_as_list)
        ),
        "top_uniprot_core_cofactor_labels": _top_counts(
            enhanced_labels["uniprot_core_cofactor_labels_train"].map(_as_list)
        ),
        "top_reaction_fallback_core_cofactor_labels": _top_counts(
            enhanced_labels["reaction_fallback_core_cofactor_labels_train"].map(_as_list)
        ),
        "top_active_cofactor_architecture_bins": _top_counts(
            enhanced_labels["cofactor_architecture_bins_train"].map(_as_list)
        ),
        "top_active_cofactor_chemistry_bins": _top_counts(
            enhanced_labels["cofactor_chemistry_bins_train"].map(_as_list)
        ),
        "top_reaction_plus_uniprot_core_cofactor_labels": _top_counts(
            enhanced_labels["reaction_plus_uniprot_core_cofactor_labels_train"].map(_as_list)
        ),
        "quality_flag_counts": dict(
            Counter(flag for flags in enhanced_labels["quality_flags"].map(_as_list) for flag in flags)
        ),
        "pair_overlap_columns_added": [
            "enzyme_derived_cofactor_overlap",
            "enzyme_derived_core_cofactor_overlap",
            "enzyme_derived_metal_ion_overlap",
            "enzyme_derived_auxiliary_participant_overlap",
            "combined_cofactor_overlap",
            "combined_core_cofactor_overlap",
            "combined_metal_ion_overlap",
            "combined_auxiliary_participant_overlap",
            "reaction_fallback_cofactor_overlap",
            "reaction_fallback_core_cofactor_overlap",
            "reaction_fallback_metal_ion_overlap",
            "reaction_fallback_auxiliary_participant_overlap",
            "reaction_plus_uniprot_cofactor_overlap",
            "reaction_plus_uniprot_core_cofactor_overlap",
            "reaction_plus_uniprot_metal_ion_overlap",
            "reaction_plus_uniprot_auxiliary_participant_overlap",
        ],
    }

    enhanced_labels.to_parquet(labels_path, index=False)
    enhanced_pairs.to_parquet(pairs_path, index=False)
    _update_vocabs(vocabs_path, enhanced_labels)
    _write_uniprot_cofactor_label_dictionary(
        cap_dir / "uniprot_cofactor_label_dictionary.csv",
        enhanced_labels,
    )
    _write_fine_cofactor_label_dictionary(
        cap_dir / "fine_cofactor_label_dictionary.csv",
        enhanced_labels,
    )
    _write_cofactor_bin_dictionary(
        cap_dir / "cofactor_bin_dictionary.csv",
        enhanced_labels,
    )
    _write_primary_cofactor_dictionary(
        cap_dir / "cofactor_dictionary.csv",
        enhanced_labels,
    )
    write_json(cap_dir / "enzyme_cofactor_annotation_report.json", report)
    if write_csv:
        csv_columns = [
            "enzyme_id",
            "enzyme_cofactor_accessions",
            "cofactor_labels_train",
            "core_cofactor_labels_train",
            "uniprot_cofactor_labels_train",
            "uniprot_core_cofactor_labels_train",
            "uniprot_metal_ion_labels_train",
            "uniprot_auxiliary_participant_labels_train",
            "reaction_fallback_cofactor_labels_train",
            "reaction_fallback_core_cofactor_labels_train",
            "reaction_fallback_metal_ion_labels_train",
            "reaction_fallback_auxiliary_participant_labels_train",
            "enzyme_derived_cofactor_labels_train",
            "enzyme_derived_core_cofactor_labels_train",
            "enzyme_uniprotkb_cofactor_labels_train",
            "combined_cofactor_labels_train",
            "combined_core_cofactor_labels_train",
            "combined_metal_ion_labels_train",
            "cofactor_architecture_bins_train",
            "cofactor_chemistry_bins_train",
            "reaction_plus_uniprot_cofactor_labels_train",
            "reaction_plus_uniprot_core_cofactor_labels_train",
            "enzyme_cofactor_label_source",
            "quality_flags",
        ]
        enhanced_labels[csv_columns].to_csv(cap_dir / "enzyme_cofactor_labels_enhanced.csv", index=False)
    return enhanced_labels, enhanced_pairs, report
