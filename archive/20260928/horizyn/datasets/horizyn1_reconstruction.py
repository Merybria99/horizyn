"""Streaming reconstruction utilities for the Horizyn-1 inference dataset.

The paper reports aggregate counts but does not release the complete 2023_05
training graph.  This module reconstructs the closest auditable approximation
from the named public sources.  Every large operation is streaming or delegated
to GNU sort/MMseqs2; no table proportional to TrEMBL is held in Python memory.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

UNIPROT_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/"
    "release-2023_05/knowledgebase/knowledgebase2023_05.tar.gz"
)
UNIPROT_BYTES = 186_243_013_906
UNIPROT_MD5 = "8d2f41981c4cfbdaf1c9b8932dfe6e91"
RHEA_URL = "https://ftp.expasy.org/databases/rhea/old_releases/131.tar.bz2"
RHEA_BYTES = 343_514_968

PAPER_TARGETS = {
    "raw_reactions": 31_101,
    "raw_proteins": 27_099_893,
    "raw_pairs": 34_385_290,
    "clustered_proteins": 7_063_237,
    "clustered_pairs": 8_897_870,
}

RHEA_RE = re.compile(r"RHEA:(\d+)", re.IGNORECASE)
RHEA_ID_RE = re.compile(r"(?:Rh_|RHEA:)?(\d+)$", re.IGNORECASE)
SEQUENCE_RE = re.compile(r"[^A-Za-z]")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def disk_preflight(path: Path, reserve_tib: float = 5.0) -> dict[str, object]:
    """Return a conservative capacity check for the full reconstruction."""
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    reserve_bytes = int(reserve_tib * 1024**4)
    download_bytes = UNIPROT_BYTES + RHEA_BYTES
    return {
        "path": str(path.resolve()),
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes": usage.free,
        "filesystem_free_tib": round(usage.free / 1024**4, 3),
        "known_download_bytes": download_bytes,
        "known_download_gib": round(download_bytes / 1024**3, 3),
        "requested_working_reserve_bytes": reserve_bytes,
        "requested_working_reserve_tib": reserve_tib,
        "enough_space": usage.free >= reserve_bytes,
        "paper_targets": PAPER_TARGETS,
        "note": (
            "The reserve includes downloaded archives, streamed FASTA/TSV outputs, "
            "GNU-sort scratch, and MMseqs2 databases/temp files. Actual use is data- "
            "and MMseqs-parameter-dependent."
        ),
    }


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8", newline="")
    return path.open(mode, encoding=None if "b" in mode else "utf-8", newline="")


@contextmanager
def atomic_text(path: Path, gzip_output: bool | None = None) -> Iterator[TextIO]:
    """Write a text file atomically, optionally using gzip based on its suffix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    use_gzip = path.suffix == ".gz" if gzip_output is None else gzip_output
    if use_gzip:
        handle = gzip.open(temporary, "wt", encoding="utf-8", newline="")
    else:
        handle = temporary.open("w", encoding="utf-8", newline="")
    try:
        yield handle
        handle.close()
        os.replace(temporary, path)
    except BaseException:
        handle.close()
        raise


def normalize_rhea_id(value: str) -> str | None:
    match = RHEA_ID_RE.search(value.strip())
    return match.group(1) if match else None


def development_rhea_ids(reaction_csvs: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in reaction_csvs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                value = normalize_rhea_id(row.get("reaction_id", ""))
                if value:
                    result.add(value)
    return result


def _find_tar_member(archive: Path, basename: str) -> tuple[tarfile.TarFile, tarfile.TarInfo]:
    tar = tarfile.open(archive, "r:*")
    for member in tar:
        if member.isfile() and Path(member.name).name == basename:
            return tar, member
    tar.close()
    raise FileNotFoundError(f"{basename!r} was not found in {archive}")


def build_rhea_direction_map(archive_or_tsv: Path, output_path: Path) -> dict[str, object]:
    """Map master, directional, and bidirectional Rhea IDs to the master ID."""
    tar: tarfile.TarFile | None = None
    if tarfile.is_tarfile(archive_or_tsv):
        tar, member = _find_tar_member(archive_or_tsv, "rhea-directions.tsv")
        raw = tar.extractfile(member)
        if raw is None:
            tar.close()
            raise OSError(f"Could not extract {member.name} from {archive_or_tsv}")
        handle: TextIO = io.TextIOWrapper(raw, encoding="utf-8", newline="")
    else:
        handle = open_text(archive_or_tsv)

    mapping: dict[str, str] = {}
    rows = 0
    try:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Rhea directions table has no header")
        normalized = {name.upper(): name for name in reader.fieldnames}
        master_key = next(
            (normalized[key] for key in normalized if "MASTER" in key),
            reader.fieldnames[0],
        )
        for row in reader:
            master = normalize_rhea_id(row.get(master_key, ""))
            if not master:
                continue
            rows += 1
            mapping[master] = master
            for value in row.values():
                directional = normalize_rhea_id(value or "")
                if directional:
                    previous = mapping.setdefault(directional, master)
                    if previous != master:
                        raise ValueError(
                            f"Rhea ID {directional} maps to both {previous} and {master}"
                        )
    finally:
        handle.close()
        if tar is not None:
            tar.close()

    with atomic_text(output_path, gzip_output=False) as out:
        out.write("rhea_id\tmaster_rhea_id\n")
        for directional, master in sorted(mapping.items(), key=lambda item: int(item[0])):
            out.write(f"{directional}\t{master}\n")
    return {"direction_rows": rows, "mapped_ids": len(mapping), "output": str(output_path)}


def load_rhea_direction_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source = normalize_rhea_id(row.get("rhea_id", ""))
            master = normalize_rhea_id(row.get("master_rhea_id", ""))
            if source and master:
                mapping[source] = master
    return mapping


def parse_protein_refs(value: str) -> list[str]:
    value = (value or "").strip()
    if not value or value == "[]":
        return []
    try:
        refs = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        refs = re.split(r"[,;]", value)
    if isinstance(refs, str):
        refs = [refs]
    result: list[str] = []
    for ref in refs:
        accession = str(ref).strip().split()[0] if str(ref).strip() else ""
        if accession and accession != "-":
            result.append(accession)
    return result


def enzymemap_reaction_id(reaction_smiles: str) -> str:
    digest = hashlib.sha1(reaction_smiles.encode("utf-8")).hexdigest()[:16]
    return f"Em_{digest}"


def prepare_enzymemap(
    processed_csv: Path,
    pairs_path: Path,
    reactions_path: Path,
    wanted_accessions_path: Path,
) -> dict[str, object]:
    """Extract all UniProt-linked EnzymeMap v2 edges without loading its CSV."""
    seen_pairs: set[tuple[str, str]] = set()
    reactions: dict[str, tuple[str, str, str]] = {}
    wanted: set[str] = set()
    raw_rows = eligible_rows = 0
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_text(pairs_path) as pair_out:
        pair_out.write("reaction_id\tprotein_ref\tsource\n")
        with open_text(processed_csv) as handle:
            for raw_rows, row in enumerate(csv.DictReader(handle), start=1):
                if (row.get("protein_db") or "").strip().lower() not in {
                    "uniprot",
                    "swissprot",
                }:
                    continue
                reaction = (row.get("unmapped") or "").strip()
                refs = parse_protein_refs(row.get("protein_refs", ""))
                if not reaction or not refs:
                    continue
                eligible_rows += 1
                reaction_id = enzymemap_reaction_id(reaction)
                reactions.setdefault(
                    reaction_id,
                    (
                        reaction,
                        (row.get("rxn_idx") or "").strip(),
                        (row.get("ec_num") or "").strip(),
                    ),
                )
                for ref in refs:
                    wanted.add(ref)
                    key = (reaction_id, ref)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        pair_out.write(f"{reaction_id}\t{ref}\tEnzymeMap_v2\n")

    with atomic_text(reactions_path) as out:
        out.write("reaction_id\treaction_smiles\tsource_reaction_id\tec\n")
        for reaction_id, values in sorted(reactions.items()):
            out.write("\t".join((reaction_id, *values)) + "\n")
    with atomic_text(wanted_accessions_path, gzip_output=False) as out:
        for accession in sorted(wanted):
            out.write(accession + "\n")
    return {
        "raw_rows": raw_rows,
        "eligible_rows": eligible_rows,
        "unique_reactions": len(reactions),
        "unique_pairs_before_sequence_resolution": len(seen_pairs),
        "wanted_accessions": len(wanted),
    }


@dataclass
class UniProtRecord:
    accessions: list[str]
    sequence: str
    rhea_ids: set[str]


def iter_uniprot_records(lines: Iterable[str]) -> Iterator[UniProtRecord]:
    accessions: list[str] = []
    sequence_parts: list[str] = []
    rhea_ids: set[str] = set()
    in_sequence = False
    for line in lines:
        if line.startswith("//"):
            if accessions and sequence_parts:
                yield UniProtRecord(accessions, "".join(sequence_parts), rhea_ids)
            accessions, sequence_parts, rhea_ids = [], [], set()
            in_sequence = False
            continue
        if line.startswith("AC   "):
            accessions.extend(item.strip() for item in line[5:].split(";") if item.strip())
        for match in RHEA_RE.finditer(line):
            rhea_ids.add(match.group(1))
        if line.startswith("SQ   "):
            in_sequence = True
            continue
        if in_sequence:
            chunk = SEQUENCE_RE.sub("", line).upper()
            if chunk:
                sequence_parts.append(chunk)


@contextmanager
def _inner_member_text(tar: tarfile.TarFile, member: tarfile.TarInfo) -> Iterator[TextIO]:
    raw = tar.extractfile(member)
    if raw is None:
        raise OSError(f"Could not extract {member.name}")
    binary: io.BufferedIOBase | gzip.GzipFile
    if member.name.endswith(".gz"):
        binary = gzip.GzipFile(fileobj=raw)
    else:
        binary = raw  # type: ignore[assignment]
    text = io.TextIOWrapper(binary, encoding="utf-8", errors="replace", newline="")
    try:
        yield text
    finally:
        text.close()


def iter_uniprot_sources(path: Path, source_kind: str = "auto") -> Iterator[tuple[str, TextIO]]:
    """Yield (sprot|trembl, handle) from a flat file or the release tarball."""
    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r|*") as tar:
            matched = 0
            for member in tar:
                basename = Path(member.name).name
                kind = "trembl" if basename.startswith("uniprot_trembl.dat") else None
                kind = "sprot" if basename.startswith("uniprot_sprot.dat") else kind
                if not member.isfile() or kind is None:
                    continue
                matched += 1
                with _inner_member_text(tar, member) as handle:
                    yield kind, handle
            if matched == 0:
                raise FileNotFoundError(f"No UniProtKB flat files found inside {path}")
        return

    if source_kind not in {"sprot", "trembl"}:
        raise ValueError("--source-kind must be sprot or trembl for a direct flat file")
    with open_text(path) as handle:
        yield source_kind, handle


def read_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with open_text(path) as handle:
        return {line.strip().split()[0] for line in handle if line.strip()}


def read_fasta_ids(paths: Iterable[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        with open_text(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    result.add(line[1:].strip().split()[0])
    return result


def parse_uniprot_release(
    input_path: Path,
    rhea_map_path: Path,
    valid_master_ids: set[str],
    wanted_accessions: set[str],
    existing_accessions: set[str],
    pair_output: Path,
    selected_fasta_output: Path,
    extra_fasta_output: Path,
    accession_map_output: Path,
    source_kind: str = "auto",
) -> dict[str, object]:
    """Stream UniProt, retaining TrEMBL Rhea edges and requested EnzymeMap proteins."""
    direction_map = load_rhea_direction_map(rhea_map_path)
    stats: dict[str, int | str] = {
        "records": 0,
        "sprot_records": 0,
        "trembl_records": 0,
        "trembl_records_with_rhea": 0,
        "selected_trembl_proteins": 0,
        "trembl_pairs": 0,
        "wanted_accessions_resolved": 0,
        "enzymemap_extra_proteins": 0,
    }
    resolved_wanted: set[str] = set()
    with (
        atomic_text(pair_output) as pair_out,
        atomic_text(selected_fasta_output) as selected_out,
        atomic_text(extra_fasta_output) as extra_out,
        atomic_text(accession_map_output, gzip_output=False) as map_out,
    ):
        pair_out.write("reaction_id\tprotein_id\tsource\n")
        map_out.write("requested_accession\tprimary_accession\n")
        for kind, handle in iter_uniprot_sources(input_path, source_kind=source_kind):
            for record in iter_uniprot_records(handle):
                stats["records"] = int(stats["records"]) + 1
                stats[f"{kind}_records"] = int(stats[f"{kind}_records"]) + 1
                primary = record.accessions[0]
                matched_wanted = sorted(wanted_accessions.intersection(record.accessions))
                for requested in matched_wanted:
                    if requested not in resolved_wanted:
                        resolved_wanted.add(requested)
                        map_out.write(f"{requested}\t{primary}\n")

                masters = {
                    direction_map[rhea_id]
                    for rhea_id in record.rhea_ids
                    if rhea_id in direction_map and direction_map[rhea_id] in valid_master_ids
                }
                is_selected = kind == "trembl" and bool(masters)
                if kind == "trembl" and record.rhea_ids:
                    stats["trembl_records_with_rhea"] = int(stats["trembl_records_with_rhea"]) + 1
                if is_selected:
                    stats["selected_trembl_proteins"] = int(stats["selected_trembl_proteins"]) + 1
                    selected_out.write(f">{primary}\n{record.sequence}\n")
                    for master in sorted(masters, key=int):
                        pair_out.write(f"Rh_{master}\t{primary}\tUniProtKB_TrEMBL_2023_05\n")
                        stats["trembl_pairs"] = int(stats["trembl_pairs"]) + 1
                if matched_wanted and not is_selected and primary not in existing_accessions:
                    extra_out.write(f">{primary}\n{record.sequence}\n")
                    existing_accessions.add(primary)
                    stats["enzymemap_extra_proteins"] = int(stats["enzymemap_extra_proteins"]) + 1

    stats["wanted_accessions_resolved"] = len(resolved_wanted)
    stats["wanted_accessions_missing"] = len(wanted_accessions - resolved_wanted)
    stats["valid_development_rhea_masters"] = len(valid_master_ids)
    return stats


def finalize_enzymemap_pairs(
    raw_pairs_path: Path,
    accession_map_path: Path,
    existing_accessions: set[str],
    output_path: Path,
) -> dict[str, object]:
    aliases: dict[str, str] = {}
    with open_text(accession_map_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            aliases[row["requested_accession"]] = row["primary_accession"]
    seen: set[tuple[str, str]] = set()
    unresolved: set[str] = set()
    raw = 0
    with atomic_text(output_path) as out:
        out.write("reaction_id\tprotein_id\tsource\n")
        with open_text(raw_pairs_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw += 1
                requested = row["protein_ref"]
                primary = aliases.get(requested, requested)
                if primary not in existing_accessions and requested not in aliases:
                    base = requested.split("-", 1)[0]
                    primary = aliases.get(base, base)
                if primary not in existing_accessions:
                    unresolved.add(requested)
                    continue
                key = (row["reaction_id"], primary)
                if key in seen:
                    continue
                seen.add(key)
                out.write(f"{key[0]}\t{key[1]}\tEnzymeMap_v2\n")
    return {
        "raw_pairs": raw,
        "resolved_unique_pairs": len(seen),
        "unresolved_accessions": len(unresolved),
        "unresolved_examples": sorted(unresolved)[:25],
    }


def _copy_text_files(inputs: Iterable[Path], output: Path) -> int:
    records = 0
    with atomic_text(output, gzip_output=False) as out:
        for path in inputs:
            last = "\n"
            with open_text(path) as handle:
                for line in handle:
                    out.write(line)
                    last = line
                    if line.startswith(">"):
                        records += 1
            if last and not last.endswith("\n"):
                out.write("\n")
    return records


def _sort_file(
    input_path: Path, output_path: Path, temp_dir: Path, keys: list[str], threads: int
) -> None:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    command = [
        "sort",
        f"--temporary-directory={temp_dir}",
        f"--parallel={threads}",
        "-S",
        "25%",
        "-t",
        "\t",
        *keys,
        str(input_path),
        "-o",
        str(output_path),
    ]
    subprocess.run(command, check=True, env=env)


def _aggregate_sorted_pairs(sorted_path: Path, output_path: Path) -> int:
    count = 0
    previous: tuple[str, str] | None = None
    sources: set[str] = set()
    with atomic_text(output_path, gzip_output=False) as out:
        out.write("reaction_id\tprotein_id\tsources\n")
        with sorted_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                reaction_id, protein_id, source = line.rstrip("\n").split("\t", 2)
                key = (reaction_id, protein_id)
                if previous is not None and key != previous:
                    out.write(f"{previous[0]}\t{previous[1]}\t{','.join(sorted(sources))}\n")
                    count += 1
                    sources.clear()
                previous = key
                sources.add(source)
            if previous is not None:
                out.write(f"{previous[0]}\t{previous[1]}\t{','.join(sorted(sources))}\n")
                count += 1
    return count


def merge_sources(
    development_pair_csvs: list[Path],
    development_reaction_csvs: list[Path],
    fasta_inputs: list[Path],
    trembl_pairs: Path,
    enzymemap_pairs: Path,
    enzymemap_reactions: Path,
    output_dir: Path,
    sort_threads: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "sort_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    combined_fasta = output_dir / "raw_proteins.fasta"
    raw_proteins = _copy_text_files(fasta_inputs, combined_fasta)

    unsorted_pairs = output_dir / "raw_pairs.unsorted.tsv"
    with atomic_text(unsorted_pairs, gzip_output=False) as out:
        for path in development_pair_csvs:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    out.write(f"{row['reaction_id']}\t{row['protein_id']}\tdevelopment_release\n")
        for path in (trembl_pairs, enzymemap_pairs):
            with open_text(path) as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    out.write(f"{row['reaction_id']}\t{row['protein_id']}\t{row['source']}\n")
    sorted_pairs = output_dir / "raw_pairs.sorted.tsv"
    _sort_file(
        unsorted_pairs,
        sorted_pairs,
        temp_dir,
        ["-k1,1", "-k2,2", "-k3,3", "-u"],
        sort_threads,
    )
    pair_output = output_dir / "raw_pairs.tsv"
    raw_pairs = _aggregate_sorted_pairs(sorted_pairs, pair_output)
    unsorted_pairs.unlink()
    sorted_pairs.unlink()

    reactions: dict[str, str] = {}
    for path in development_reaction_csvs:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rid, smiles = row["reaction_id"], row["reaction_smiles"]
                if rid in reactions and reactions[rid] != smiles:
                    raise ValueError(f"Conflicting reaction definitions for {rid}")
                reactions[rid] = smiles
    with open_text(enzymemap_reactions) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rid, smiles = row["reaction_id"], row["reaction_smiles"]
            if rid in reactions and reactions[rid] != smiles:
                raise ValueError(f"Conflicting reaction definitions for {rid}")
            reactions[rid] = smiles
    reaction_output = output_dir / "raw_reactions.tsv"
    with atomic_text(reaction_output, gzip_output=False) as out:
        out.write("reaction_id\treaction_smiles\n")
        for rid, smiles in sorted(reactions.items()):
            out.write(f"{rid}\t{smiles}\n")

    stats: dict[str, object] = {
        "raw_reactions": len(reactions),
        "raw_proteins": raw_proteins,
        "raw_pairs": raw_pairs,
        "outputs": {
            "proteins": str(combined_fasta),
            "pairs": str(pair_output),
            "reactions": str(reaction_output),
        },
    }
    stats["target_deltas"] = {
        key: int(stats[key]) - PAPER_TARGETS[key]
        for key in ("raw_reactions", "raw_proteins", "raw_pairs")
    }
    write_json(output_dir / "raw_manifest.json", stats)
    return stats


def _transform_cluster_map(cluster_tsv: Path, member_rep: Path) -> int:
    count = 0
    with atomic_text(member_rep, gzip_output=False) as out:
        with cluster_tsv.open("r", encoding="utf-8") as handle:
            for line in handle:
                representative, member = line.rstrip("\n").split("\t")[:2]
                out.write(f"{member}\t{representative}\n")
                count += 1
    return count


def _pairs_by_protein(pair_tsv: Path, output: Path) -> int:
    count = 0
    with atomic_text(output, gzip_output=False) as out:
        with pair_tsv.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                out.write(f"{row['protein_id']}\t{row['reaction_id']}\t{row['sources']}\n")
                count += 1
    return count


def cluster_and_collapse(
    raw_fasta: Path,
    raw_pairs: Path,
    output_dir: Path,
    mmseqs: Path,
    threads: int,
    min_seq_id: float = 0.8,
    extra_cluster_args: list[str] | None = None,
) -> dict[str, object]:
    """Cluster proteins and collapse graph edges onto representatives."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mmseqs_dir = output_dir / "mmseqs"
    sort_tmp = output_dir / "sort_tmp"
    mmseqs_tmp = output_dir / "mmseqs_tmp"
    mmseqs_dir.mkdir(parents=True, exist_ok=True)
    sort_tmp.mkdir(parents=True, exist_ok=True)
    mmseqs_tmp.mkdir(parents=True, exist_ok=True)
    database = mmseqs_dir / "proteins"
    clusters = mmseqs_dir / "clusters"
    cluster_tsv = output_dir / "clusters.tsv"
    representative_db = mmseqs_dir / "representatives"
    representative_fasta = output_dir / "proteins.fasta"

    subprocess.run([str(mmseqs), "createdb", str(raw_fasta), str(database)], check=True)
    cluster_command = [
        str(mmseqs),
        "cluster",
        str(database),
        str(clusters),
        str(mmseqs_tmp),
        "--min-seq-id",
        str(min_seq_id),
        "--threads",
        str(threads),
        *(extra_cluster_args or []),
    ]
    subprocess.run(cluster_command, check=True)
    subprocess.run(
        [str(mmseqs), "createtsv", str(database), str(database), str(clusters), str(cluster_tsv)],
        check=True,
    )
    subprocess.run(
        [str(mmseqs), "result2repseq", str(database), str(clusters), str(representative_db)],
        check=True,
    )
    subprocess.run(
        [str(mmseqs), "convert2fasta", str(representative_db), str(representative_fasta)],
        check=True,
    )

    member_rep = output_dir / "member_rep.unsorted.tsv"
    clustered_members = _transform_cluster_map(cluster_tsv, member_rep)
    member_rep_sorted = output_dir / "member_rep.sorted.tsv"
    _sort_file(member_rep, member_rep_sorted, sort_tmp, ["-k1,1", "-u"], threads)

    protein_pairs = output_dir / "pairs_by_protein.unsorted.tsv"
    input_pair_count = _pairs_by_protein(raw_pairs, protein_pairs)
    protein_pairs_sorted = output_dir / "pairs_by_protein.sorted.tsv"
    _sort_file(protein_pairs, protein_pairs_sorted, sort_tmp, ["-k1,1", "-k2,2"], threads)

    mapped_unsorted = output_dir / "representative_pairs.unsorted.tsv"
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    with mapped_unsorted.open("w", encoding="utf-8") as out:
        subprocess.run(
            [
                "join",
                "-t",
                "\t",
                "-1",
                "1",
                "-2",
                "1",
                "-o",
                "2.2,1.2,2.3",
                str(member_rep_sorted),
                str(protein_pairs_sorted),
            ],
            check=True,
            env=env,
            stdout=out,
        )
    mapped_count = sum(1 for _ in mapped_unsorted.open("r", encoding="utf-8"))
    if mapped_count != input_pair_count:
        raise RuntimeError(
            f"Only {mapped_count:,}/{input_pair_count:,} pairs mapped to an MMseqs member"
        )
    mapped_sorted = output_dir / "representative_pairs.sorted.tsv"
    _sort_file(mapped_unsorted, mapped_sorted, sort_tmp, ["-k1,1", "-k2,2", "-k3,3", "-u"], threads)
    clustered_pair_output = output_dir / "pairs.tsv"
    clustered_pairs = _aggregate_sorted_pairs(mapped_sorted, clustered_pair_output)
    clustered_proteins = sum(
        1 for line in representative_fasta.open("r", encoding="utf-8") if line.startswith(">")
    )

    for path in (
        member_rep,
        member_rep_sorted,
        protein_pairs,
        protein_pairs_sorted,
        mapped_unsorted,
        mapped_sorted,
    ):
        path.unlink()
    stats: dict[str, object] = {
        "clustered_members": clustered_members,
        "clustered_proteins": clustered_proteins,
        "clustered_pairs": clustered_pairs,
        "min_seq_id": min_seq_id,
        "mmseqs_cluster_command": cluster_command,
        "target_deltas": {
            "clustered_proteins": clustered_proteins - PAPER_TARGETS["clustered_proteins"],
            "clustered_pairs": clustered_pairs - PAPER_TARGETS["clustered_pairs"],
        },
        "outputs": {
            "proteins": str(representative_fasta),
            "pairs": str(clustered_pair_output),
            "clusters": str(cluster_tsv),
        },
    }
    write_json(output_dir / "clustered_manifest.json", stats)
    return stats
