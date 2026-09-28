#!/usr/bin/env python3
"""Prepare or predict structures for top-ranked wet-lab enzyme candidates."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from wet_lab.query import PROJECT_ROOT, _load_fasta_sequences, _resolve_path


SCHEMA_VERSION = "horizyn_wet_lab_folding_v1"
CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
SUPPORTED_AA = CANONICAL_AA | {"U"}


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing required setting: {section}.{key}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _safe_protein_id(protein_id: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_" for character in protein_id
    )
    safe = safe.strip("._")
    if not safe:
        raise ValueError(f"Invalid protein identifier: {protein_id!r}")
    return safe


def _validate_sequence(sequence: str, protein_id: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise ValueError(f"No sequence found for {protein_id}")
    invalid = sorted(set(normalized) - SUPPORTED_AA)
    if invalid:
        raise ValueError(
            f"Sequence for {protein_id} contains unsupported residues: {', '.join(invalid)}"
        )
    return normalized


def load_ranked_candidates(
    results_path: Path,
    candidate_fasta: Path,
    *,
    top_k: int,
    protein_ids: list[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a campaign or single-model result and attach exact FASTA sequences."""

    if top_k <= 0:
        raise ValueError("ranking.top_k must be positive")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    reaction = dict(results.get("reaction") or {})
    consensus = results.get("consensus")
    if isinstance(consensus, dict) and isinstance(consensus.get("rankings"), list):
        raw_rankings = consensus["rankings"]
        rank_key = "consensus_rank"
        score_key = "rrf_score"
        score_type = "reciprocal_rank_fusion"
    elif isinstance(results.get("rankings"), list):
        raw_rankings = results["rankings"]
        rank_key = "rank"
        score_key = "cosine_similarity"
        score_type = "cosine_similarity"
    else:
        raise ValueError(f"No supported ranking table found in {results_path}")

    if protein_ids is None:
        selected = raw_rankings[:top_k]
        if len(selected) < top_k:
            raise ValueError(
                f"Requested top {top_k}, but {results_path} contains only "
                f"{len(selected)} rankings"
            )
    else:
        if not protein_ids:
            raise ValueError("ranking.protein_ids must not be empty")
        if len(protein_ids) != len(set(protein_ids)):
            raise ValueError("ranking.protein_ids must be unique")
        rows_by_id = {str(row["protein_id"]): row for row in raw_rankings}
        missing_rankings = [
            protein_id for protein_id in protein_ids if protein_id not in rows_by_id
        ]
        if missing_rankings:
            raise ValueError(
                f"{results_path} has no retained ranking for: " f"{', '.join(missing_rankings)}"
            )
        selected = [rows_by_id[protein_id] for protein_id in protein_ids]
    protein_ids = [str(row["protein_id"]) for row in selected]
    if len(protein_ids) != len(set(protein_ids)):
        raise ValueError("Top-ranked protein identifiers must be unique")
    sequences = _load_fasta_sequences(candidate_fasta, set(protein_ids))
    missing = [protein_id for protein_id in protein_ids if protein_id not in sequences]
    if missing:
        raise ValueError(
            f"Candidate FASTA is missing {len(missing)} ranked proteins: {', '.join(missing)}"
        )

    candidates = []
    for row in selected:
        protein_id = str(row["protein_id"])
        sequence = _validate_sequence(sequences[protein_id], protein_id)
        candidates.append(
            {
                "rank": int(row[rank_key]),
                "protein_id": protein_id,
                "score": float(row[score_key]),
                "score_type": score_type,
                "sequence": sequence,
                "sequence_length": len(sequence),
                "selenocysteine_positions": [
                    index for index, residue in enumerate(sequence, start=1) if residue == "U"
                ],
            }
        )
    return reaction, candidates


def write_folding_inputs(
    output_dir: Path,
    reaction: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    model_seeds: list[int],
) -> dict[str, Any]:
    """Write FASTA and one AlphaFold 3 monomer JSON job per candidate."""

    input_dir = output_dir / "inputs"
    af3_dir = input_dir / "alphafold3"
    af3_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = input_dir / f"top_{len(candidates)}.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(
                f">{candidate['protein_id']} rank={candidate['rank']} "
                f"{candidate['score_type']}={candidate['score']:.8f}\n"
            )
            sequence = candidate["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")

    af3_jobs = []
    reaction_id = str(reaction.get("id") or "wet_lab_query")
    for candidate in candidates:
        safe_id = _safe_protein_id(candidate["protein_id"])
        job_path = af3_dir / f"{candidate['rank']:02d}_{safe_id}.json"
        af3_sequence = candidate["sequence"].replace("U", "C")
        protein = {
            "id": "A",
            "sequence": af3_sequence,
        }
        if candidate["selenocysteine_positions"]:
            protein["modifications"] = [
                {
                    "ptmType": "CCD_SEC",
                    "ptmPosition": position,
                }
                for position in candidate["selenocysteine_positions"]
            ]
        job = {
            "name": f"{reaction_id}_rank{candidate['rank']:02d}_{safe_id}",
            "modelSeeds": model_seeds,
            "sequences": [
                {
                    "protein": protein,
                }
            ],
            "dialect": "alphafold3",
            "version": 1,
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        af3_jobs.append(str(job_path))
    return {
        "fasta": str(fasta_path),
        "alphafold3_input_dir": str(af3_dir),
        "alphafold3_jobs": af3_jobs,
    }


def _candidate_status(candidate: dict[str, Any], **updates: Any) -> dict[str, Any]:
    return {
        "rank": candidate["rank"],
        "protein_id": candidate["protein_id"],
        "sequence_length": candidate["sequence_length"],
        "score_type": candidate["score_type"],
        "score": candidate["score"],
        "status": "",
        "structure_path": "",
        "mean_plddt": "",
        "ptm": "",
        "elapsed_seconds": "",
        "message": "",
        **updates,
    }


def _mean_plddt(output: Any) -> float:
    plddt = output["plddt"].float()
    atom_mask = output["atom37_atom_exists"].float()
    denominator = atom_mask.sum().clamp_min(1.0)
    return float((plddt * atom_mask).sum().div(denominator).item())


def _normalize_plddt_scale(output: Any) -> float:
    """Convert Transformers ESMFold confidence and PDB B-factors to 0-100."""

    maximum = float(output["plddt"].detach().max().item())
    scale = 100.0 if maximum <= 1.5 else 1.0
    if scale != 1.0:
        output["plddt"] = output["plddt"] * scale
    return scale


def run_esmfold(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    backend: dict[str, Any],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    """Run Hugging Face Transformers ESMFold once per full-length sequence."""

    from transformers import EsmForProteinFolding

    device = str(backend.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("ESMFold requested CUDA, but torch.cuda.is_available() is false")
    model_name = str(backend.get("model", "facebook/esmfold_v1"))
    cache_dir = _resolve_path(
        backend.get("cache_dir", "wet_lab/cache/esmfold"),
        must_exist=False,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")

    print(f"Loading ESMFold model: {model_name}", flush=True)
    model = EsmForProteinFolding.from_pretrained(
        model_name,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=bool(backend.get("low_cpu_mem_usage", True)),
    )
    model.eval()
    if bool(backend.get("esm_half_precision", True)):
        model.esm = model.esm.half()
    chunk_size = backend.get("chunk_size", 64)
    if chunk_size is not None:
        model.trunk.set_chunk_size(int(chunk_size))
    model = model.to(device)

    structure_dir = output_dir / "structures" / "esmfold"
    metric_dir = output_dir / "metrics" / "esmfold"
    structure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    max_length = int(backend.get("max_sequence_length", 1024))
    long_policy = str(backend.get("long_sequence_policy", "skip"))
    if long_policy not in {"skip", "error"}:
        raise ValueError("backend.long_sequence_policy must be 'skip' or 'error'")

    statuses: list[dict[str, Any]] = []
    for candidate in candidates:
        safe_id = _safe_protein_id(candidate["protein_id"])
        prefix = f"{candidate['rank']:02d}_{safe_id}"
        pdb_path = structure_dir / f"{prefix}.pdb"
        metrics_path = metric_dir / f"{prefix}.json"
        if candidate["sequence_length"] > max_length:
            message = (
                f"Full sequence has {candidate['sequence_length']} residues, exceeding "
                f"the configured ESMFold limit of {max_length}; sequence was not truncated"
            )
            status = _candidate_status(candidate, status="skipped_too_long", message=message)
            statuses.append(status)
            print(f"Skipping {candidate['protein_id']}: {message}", flush=True)
            if long_policy == "error":
                break
            continue
        if pdb_path.is_file() and metrics_path.is_file() and not force:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            statuses.append(
                _candidate_status(
                    candidate,
                    status="reused",
                    structure_path=str(pdb_path),
                    mean_plddt=metrics.get("mean_plddt", ""),
                    ptm=metrics.get("ptm", ""),
                    elapsed_seconds=metrics.get("elapsed_seconds", ""),
                )
            )
            print(f"Reusing existing structure: {pdb_path}", flush=True)
            continue

        print(
            f"Folding rank {candidate['rank']} {candidate['protein_id']} "
            f"({candidate['sequence_length']} aa)...",
            flush=True,
        )
        started = time.monotonic()
        try:
            esmfold_sequence = candidate["sequence"].replace("U", "X")
            with torch.inference_mode():
                output = model.infer(esmfold_sequence)
                plddt_scale = _normalize_plddt_scale(output)
                pdb = model.output_to_pdb(output)[0]
            elapsed = time.monotonic() - started
            mean_plddt = _mean_plddt(output)
            ptm_tensor = output.get("ptm")
            ptm = float(ptm_tensor.reshape(-1)[0].item()) if ptm_tensor is not None else None
            pdb_path.write_text(pdb, encoding="utf-8")
            metrics = {
                "protein_id": candidate["protein_id"],
                "rank": candidate["rank"],
                "sequence_length": candidate["sequence_length"],
                "selenocysteine_positions": candidate["selenocysteine_positions"],
                "esmfold_substitutions": [
                    {
                        "position": position,
                        "original": "U",
                        "model_input": "X",
                    }
                    for position in candidate["selenocysteine_positions"]
                ],
                "model": model_name,
                "mean_plddt": mean_plddt,
                "plddt_scale_applied": plddt_scale,
                "ptm": ptm,
                "elapsed_seconds": elapsed,
                "structure_path": str(pdb_path),
            }
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
            statuses.append(
                _candidate_status(
                    candidate,
                    status="completed",
                    structure_path=str(pdb_path),
                    mean_plddt=mean_plddt,
                    ptm=ptm if ptm is not None else "",
                    elapsed_seconds=elapsed,
                    message=(
                        "ESMFold represented selenocysteine as unknown residue X"
                        if candidate["selenocysteine_positions"]
                        else ""
                    ),
                )
            )
            print(
                f"Saved {pdb_path} (mean pLDDT={mean_plddt:.2f}, "
                f"pTM={ptm if ptm is not None else 'n/a'})",
                flush=True,
            )
            del output
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception as error:
            statuses.append(
                _candidate_status(
                    candidate,
                    status="failed",
                    elapsed_seconds=time.monotonic() - started,
                    message=f"{type(error).__name__}: {error}",
                )
            )
            print(f"Failed {candidate['protein_id']}: {type(error).__name__}: {error}", flush=True)
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return statuses


def run_alphafold3(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    backend: dict[str, Any],
    input_artifacts: dict[str, Any],
) -> list[dict[str, Any]]:
    """Execute an external official AlphaFold 3 checkout over prepared JSON jobs."""

    if bool(backend.get("prepare_only", False)):
        message = "AF3 inputs prepared; execution disabled by backend.prepare_only"
        return [
            _candidate_status(candidate, status="prepared", message=message)
            for candidate in candidates
        ]

    python_path = _resolve_path(_required(backend, "python", "backend"))
    script_path = _resolve_path(_required(backend, "script", "backend"))
    model_dir = _resolve_path(_required(backend, "model_dir", "backend"))
    database_dir = _resolve_path(_required(backend, "database_dir", "backend"))
    structure_dir = output_dir / "structures" / "alphafold3"
    structure_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        str(script_path),
        f"--input_dir={input_artifacts['alphafold3_input_dir']}",
        f"--model_dir={model_dir}",
        f"--db_dir={database_dir}",
        f"--output_dir={structure_dir}",
        *[str(value) for value in backend.get("extra_args", [])],
    ]
    log_path = output_dir / "alphafold3.log"
    print(f"Running AlphaFold 3 over {len(candidates)} jobs...", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=script_path.parent,
            env=os.environ.copy(),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(
            f"AlphaFold 3 failed with return code {completed.returncode}. Log: {log_path}\n"
            + "\n".join(tail)
        )
    message = f"AF3 completed; inspect ranked model files under {structure_dir}"
    return [
        _candidate_status(candidate, status="completed", message=message)
        for candidate in candidates
    ]


def run_folding(config_path: str | Path, *, force: bool = False) -> Path:
    path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    ranking = config.get("ranking")
    backend = config.get("backend")
    if not isinstance(ranking, dict) or not isinstance(backend, dict):
        raise ValueError("Folding config requires ranking and backend mappings")

    results_path = _resolve_path(_required(ranking, "results", "ranking"))
    candidate_fasta = _resolve_path(_required(ranking, "candidate_fasta", "ranking"))
    top_k = int(ranking.get("top_k", 10))
    protein_ids_value = ranking.get("protein_ids")
    protein_ids = (
        [str(protein_id) for protein_id in protein_ids_value]
        if isinstance(protein_ids_value, list)
        else None
    )
    reaction, candidates = load_ranked_candidates(
        results_path,
        candidate_fasta,
        top_k=top_k,
        protein_ids=protein_ids,
    )
    selected_count = len(candidates)
    output_dir = _resolve_path(
        config.get("output", {}).get(
            "directory",
            f"wet_lab/structures/{reaction.get('id', 'query')}_top{selected_count}",
        ),
        must_exist=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in backend.get("model_seeds", [1])]
    if not seeds:
        raise ValueError("backend.model_seeds must contain at least one integer")
    input_artifacts = write_folding_inputs(
        output_dir,
        reaction,
        candidates,
        model_seeds=seeds,
    )

    backend_name = str(_required(backend, "name", "backend")).lower()
    if backend_name == "esmfold":
        statuses = run_esmfold(candidates, output_dir, backend, force=force)
    elif backend_name in {"alphafold3", "af3"}:
        statuses = run_alphafold3(candidates, output_dir, backend, input_artifacts)
    else:
        raise ValueError("backend.name must be 'esmfold' or 'alphafold3'")

    _write_csv(output_dir / "status.csv", statuses)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "configuration": str(path),
        "ranking_results": str(results_path),
        "candidate_fasta": str(candidate_fasta),
        "reaction": reaction,
        "top_k": selected_count,
        "backend": backend_name,
        "input_artifacts": input_artifacts,
        "candidates": [
            {key: value for key, value in candidate.items() if key != "sequence"}
            for candidate in candidates
        ],
        "statuses": statuses,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved folding manifest to: {manifest_path}", flush=True)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Folding YAML configuration")
    parser.add_argument("--force", action="store_true", help="Replace existing structure outputs")
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_folding(args.config, force=args.force)


if __name__ == "__main__":
    main()
