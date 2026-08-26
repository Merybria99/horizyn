#!/usr/bin/env python3
"""Acquire sequence-verified structures for the restricted-setting F3 top 25."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
DEFAULT_RESULTS = SCRIPT_DIR / "runs/tagatose_4_epimerase_e07cc5e4ab31/results.json"
DEFAULT_FASTA = SCRIPT_DIR / "candidate_pool/proteins.fasta"
DEFAULT_SEQUENCE_MANIFEST = SCRIPT_DIR.parent / "sequence_pool/sequence_manifest.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "structures"
AFDB_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
USER_AGENT = "Horizyn-Case1-structure-acquisition/1.0"


def _configure_runtime(output_dir: Path, gpu: int) -> None:
    runtime = output_dir / "runtime"
    directories = {
        "HOME": runtime / "home",
        "HF_HOME": runtime / "cache/huggingface",
        "TORCH_HOME": runtime / "cache/torch",
        "XDG_CACHE_HOME": runtime / "cache/xdg",
        "XDG_CONFIG_HOME": runtime / "config/xdg",
        "XDG_STATE_HOME": runtime / "state/xdg",
        "MPLCONFIGDIR": runtime / "config/matplotlib",
        "TMPDIR": runtime / "tmp",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _load_fasta(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    protein_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if protein_id is not None:
                    sequences[protein_id] = "".join(chunks)
                protein_id = line[1:].split(maxsplit=1)[0]
                chunks = []
            else:
                chunks.append(line)
    if protein_id is not None:
        sequences[protein_id] = "".join(chunks)
    return sequences


def _request_json(url: str, *, attempts: int = 3) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == attempts:
                raise
            time.sleep(attempt)
    raise RuntimeError("unreachable")


def _download(url: str, output: Path, *, force: bool) -> None:
    if output.is_file() and output.stat().st_size > 0 and not force:
        return
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    partial = output.with_suffix(output.suffix + ".partial")
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if partial.stat().st_size == 0:
        raise RuntimeError(f"Downloaded an empty structure from {url}")
    os.replace(partial, output)


def _mean_ca_bfactor(path: Path) -> float:
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        raise ValueError(f"No CA confidence values found in {path}")
    return sum(values) / len(values)


def _uniprot_ids_by_sha(path: Path) -> dict[str, list[str]]:
    identifiers: dict[str, list[str]] = {}
    for row in _load_csv(path):
        identifiers[row["sha256"]] = [
            value for value in row.get("uniprot_ids", "").split(";") if value
        ]
    return identifiers


def _find_afdb_model(accessions: list[str], sequence: str) -> tuple[str, dict[str, Any]] | None:
    for accession in accessions:
        try:
            records = _request_json(AFDB_API.format(accession=accession))
        except urllib.error.HTTPError as error:
            if error.code == 404:
                continue
            raise
        if not records:
            continue
        record = records[0]
        afdb_sequence = "".join(str(record.get("uniprotSequence", "")).split()).upper()
        if afdb_sequence == sequence:
            return accession, record
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_ranked_candidates(
    results_path: Path, fasta_path: Path, *, top_k: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results = json.loads(results_path.read_text(encoding="utf-8"))
    rankings = list(results.get("rankings", []))[:top_k]
    if len(rankings) != top_k:
        raise ValueError(f"Expected {top_k} rankings in {results_path}, found {len(rankings)}")
    sequences = _load_fasta(fasta_path)
    rows: list[dict[str, Any]] = []
    for ranking in rankings:
        protein_id = str(ranking["protein_id"])
        sequence = sequences.get(protein_id, "")
        if not sequence:
            raise ValueError(f"No FASTA sequence found for {protein_id}")
        rows.append(
            {
                **ranking,
                "rank": int(ranking["rank"]),
                "protein_id": protein_id,
                "cosine_similarity": float(ranking["cosine_similarity"]),
                "sequence": sequence,
            }
        )
    return results, rows


def _load_reusable_structures(paths: list[Path]) -> dict[str, dict[str, Any]]:
    reusable: dict[str, dict[str, Any]] = {}
    for manifest_path in paths:
        for row in _load_csv(manifest_path.resolve()):
            sha256 = row.get("sequence_sha256", "")
            structure_path = Path(row.get("unique_structure_path", ""))
            if not structure_path.is_file() or not sha256 or sha256 in reusable:
                continue
            reusable[sha256] = {
                "source": row["structure_source"],
                "source_model": row["source_model"],
                "source_accession": row.get("source_accession", ""),
                "source_url": row.get("source_url", ""),
                "structure_path": structure_path,
                "mean_plddt": float(row["mean_plddt"]),
                "ptm": row.get("ptm", ""),
                "sequence_verified": row.get("sequence_verified", "").lower() == "true",
            }
    return reusable


def build_structures(
    *,
    results_path: Path,
    fasta_path: Path,
    sequence_manifest_path: Path,
    output_dir: Path,
    gpu: int,
    force: bool,
    top_k: int = 25,
    reuse_manifests: list[Path] | None = None,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime(output_dir, gpu)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    results, top_rows = _load_ranked_candidates(
        results_path.resolve(), fasta_path.resolve(), top_k=top_k
    )
    uniprot_by_sha = _uniprot_ids_by_sha(sequence_manifest_path.resolve())
    reusable_structures = _load_reusable_structures(reuse_manifests or [])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in top_rows:
        groups[str(row["sha256"])].append(row)

    unique_dir = output_dir / "unique"
    by_rank_dir = output_dir / "by_rank"
    afdb_metadata_dir = output_dir / "metadata/alphafold_db"
    for directory in (unique_dir, by_rank_dir, afdb_metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    structure_by_sha: dict[str, dict[str, Any]] = {}
    missing_candidates: list[dict[str, Any]] = []
    for sha256, members in sorted(groups.items(), key=lambda item: item[1][0]["rank"]):
        representative = members[0]
        filename = (
            f"rank{representative['rank']:02d}_{representative['protein_id']}_" f"{sha256[:12]}.pdb"
        )
        structure_path = unique_dir / filename
        reusable = reusable_structures.get(sha256)
        if reusable is not None:
            shutil.copy2(Path(reusable["structure_path"]), structure_path)
            structure_by_sha[sha256] = {
                **reusable,
                "structure_path": structure_path,
                "mean_plddt": _mean_ca_bfactor(structure_path),
            }
            print(
                f"Reused {reusable['source']}: rank {representative['rank']} "
                f"{representative['protein_id']}",
                flush=True,
            )
            continue

        accessions = uniprot_by_sha.get(sha256, [])
        match = _find_afdb_model(accessions, representative["sequence"])
        if match is None:
            missing_candidates.append(
                {
                    "rank": representative["rank"],
                    "protein_id": representative["protein_id"],
                    "score": representative["cosine_similarity"],
                    "score_type": "cosine_similarity",
                    "sequence": representative["sequence"],
                    "sequence_length": len(representative["sequence"]),
                    "selenocysteine_positions": [],
                    "sha256": sha256,
                }
            )
            continue

        accession, metadata = match
        _download(str(metadata["pdbUrl"]), structure_path, force=force)
        metadata_path = afdb_metadata_dir / f"{accession}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        structure_by_sha[sha256] = {
            "source": "AlphaFoldDB",
            "source_model": f"AF-{accession}-F1-model_v{metadata.get('latestVersion', '')}",
            "source_accession": accession,
            "source_url": metadata["pdbUrl"],
            "structure_path": structure_path,
            "mean_plddt": _mean_ca_bfactor(structure_path),
            "ptm": "",
            "sequence_verified": True,
        }
        print(
            f"AlphaFold DB: rank {representative['rank']} {representative['protein_id']} "
            f"via {accession}",
            flush=True,
        )

    if missing_candidates:
        from wet_lab.fold import run_esmfold

        print(
            f"Running ESMFold for {len(missing_candidates)} unique sequences without "
            "an exact AlphaFold DB model...",
            flush=True,
        )
        esmfold_output = output_dir / "esmfold_work"
        statuses = run_esmfold(
            missing_candidates,
            esmfold_output,
            {
                "device": "cuda:0",
                "model": "facebook/esmfold_v1",
                "cache_dir": "wet_lab/cache/esmfold",
                "esm_half_precision": True,
                "low_cpu_mem_usage": True,
                "chunk_size": 64,
                "max_sequence_length": 1024,
                "long_sequence_policy": "error",
            },
            force=force,
        )
        status_by_id = {str(status["protein_id"]): status for status in statuses}
        for candidate in missing_candidates:
            status = status_by_id[candidate["protein_id"]]
            if status["status"] not in {"completed", "reused"}:
                raise RuntimeError(
                    f"ESMFold failed for {candidate['protein_id']}: {status['message']}"
                )
            source_path = Path(str(status["structure_path"]))
            filename = (
                f"rank{candidate['rank']:02d}_{candidate['protein_id']}_"
                f"{candidate['sha256'][:12]}.pdb"
            )
            structure_path = unique_dir / filename
            shutil.copy2(source_path, structure_path)
            structure_by_sha[candidate["sha256"]] = {
                "source": "ESMFold",
                "source_model": "facebook/esmfold_v1",
                "source_accession": "",
                "source_url": "",
                "structure_path": structure_path,
                "mean_plddt": _mean_ca_bfactor(structure_path),
                "ptm": status["ptm"],
                "sequence_verified": True,
            }

    manifest_rows: list[dict[str, Any]] = []
    for row in top_rows:
        structure = structure_by_sha[str(row["sha256"])]
        rank_path = by_rank_dir / f"rank{row['rank']:02d}_{row['protein_id']}.pdb"
        shutil.copy2(Path(structure["structure_path"]), rank_path)
        group = groups[str(row["sha256"])]
        manifest_rows.append(
            {
                "rank": row["rank"],
                "protein_id": row["protein_id"],
                "name": row.get("name", ""),
                "cosine_similarity": row["cosine_similarity"],
                "sequence_length": len(row["sequence"]),
                "sequence_sha256": row["sha256"],
                "unique_sequence_representative": group[0]["protein_id"],
                "duplicate_selected_ids": ";".join(member["protein_id"] for member in group),
                "structure_source": structure["source"],
                "source_model": structure["source_model"],
                "source_accession": structure["source_accession"],
                "source_url": structure["source_url"],
                "mean_plddt": f"{float(structure['mean_plddt']):.4f}",
                "ptm": (f"{float(structure['ptm']):.6f}" if structure["ptm"] != "" else ""),
                "sequence_verified": structure["sequence_verified"],
                "structure_path": str(rank_path),
                "unique_structure_path": str(structure["structure_path"]),
            }
        )

    manifest_csv = output_dir / f"top{len(top_rows)}_structure_manifest.csv"
    _write_csv(manifest_csv, manifest_rows)
    machine_manifest = {
        "schema_version": "case1_restricted_ranked_structures_v2",
        "reaction": results.get("reaction", {}),
        "model": results.get("model", {}),
        "ranking_results": str(results_path.resolve()),
        "selected_rows": len(top_rows),
        "unique_sequences": len(groups),
        "alphafold_db_unique_structures": sum(
            structure["source"] == "AlphaFoldDB" for structure in structure_by_sha.values()
        ),
        "esmfold_unique_structures": sum(
            structure["source"] == "ESMFold" for structure in structure_by_sha.values()
        ),
        "experimental_structures": 0,
        "all_sequences_verified": all(
            structure["sequence_verified"] for structure in structure_by_sha.values()
        ),
        "manifest_csv": str(manifest_csv),
    }
    manifest_json = output_dir / "manifest.json"
    manifest_json.write_text(json.dumps(machine_manifest, indent=2) + "\n", encoding="utf-8")

    table = [
        "| Rank | Candidate | Source | Accession | Mean pLDDT | PDB |",
        "|---:|---|---|---|---:|---|",
    ]
    for row in manifest_rows:
        table.append(
            f"| {row['rank']} | {row['protein_id']} | {row['structure_source']} | "
            f"{row['source_accession'] or '-'} | {float(row['mean_plddt']):.2f} | "
            f"[file](by_rank/{Path(row['structure_path']).name}) |"
        )
    report = "\n".join(
        [
            f"# Restricted Setting Top-{len(top_rows)} Structures",
            "",
            f"- Ranked candidates: **{len(top_rows)}**",
            f"- Unique sequences/structures: **{len(groups)}**",
            f"- AlphaFold Database models: "
            f"**{machine_manifest['alphafold_db_unique_structures']}**",
            f"- Local ESMFold models: **{machine_manifest['esmfold_unique_structures']}**",
            "- Exact experimental structures: **0**",
            "",
            "All AlphaFold Database assignments were accepted only when the current UniProt "
            "sequence exactly matched the candidate sequence. Duplicate candidates share the "
            "same unique model but also have a separate rank-addressable PDB copy.",
            "",
            "The PDB B-factor column stores pLDDT. These are predicted monomer structures; "
            "they do not establish catalytic activity, oligomeric state, ligand pose, or "
            "metal occupancy.",
            "",
            *table,
            "",
        ]
    )
    report_path = output_dir / "README.md"
    report_path.write_text(report, encoding="utf-8")

    archive_path = output_dir / f"case1_restricted_top{len(top_rows)}_structures.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in (report_path, manifest_json, manifest_csv, unique_dir, by_rank_dir):
            archive.add(path, arcname=path.relative_to(output_dir))
    print(f"Structure report: {report_path}", flush=True)
    print(f"Structure archive: {archive_path}", flush=True)
    return manifest_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--sequence-manifest", type=Path, default=DEFAULT_SEQUENCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", type=int, default=3, help="Physical GPU used for ESMFold")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        action="append",
        default=[],
        help="Structure manifest whose sequence-identical PDBs may be reused",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_structures(
        results_path=args.results,
        fasta_path=args.fasta,
        sequence_manifest_path=args.sequence_manifest,
        output_dir=args.output_dir,
        gpu=args.gpu,
        force=args.force,
        top_k=args.top_k,
        reuse_manifests=args.reuse_manifest,
    )


if __name__ == "__main__":
    main()
