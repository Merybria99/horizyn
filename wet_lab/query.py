#!/usr/bin/env python3
"""Retrieve candidate enzymes for one reaction SMILES."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem

from horizyn.benchmarks.retrieval import (
    BenchmarkTask,
    _target_cache_base_metadata,
    build_reaction_inputs,
    encode_reactions,
    encode_residue_targets,
    filter_candidate_keys_by_score_residue,
    load_candidate_keys_from_residue,
    load_capability_vectors_from_config,
    load_repo_checkpoint,
    load_text_vectors_from_config,
    model_kind_from_config,
    needs_score_residue_embeddings,
    prepare_target_embedding_cache,
    release_target_embedding_cache_lock,
    write_target_embedding_cache,
)
from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOP_K = (1, 5, 10, 20, 50, 100)
SCHEMA_VERSION = "horizyn_wet_lab_query_v1"
CANDIDATE_POOL_ROOT_FILES = {
    "ids": "candidate_ids_prott5_order.txt",
    "residue_embeddings": "proteins_prott5_residue.h5",
    "fasta": "proteins.fasta",
    "metadata_csv": "proteins.csv",
}
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _resolve_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    # Preserve virtual-environment interpreter symlinks so Python can discover
    # the adjacent pyvenv.cfg and environment-specific site-packages.
    resolved = Path(os.path.abspath(resolved))
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    value = mapping.get(key)
    if value in {None, ""}:
        raise ValueError(f"Missing required setting: {section}.{key}")
    return value


def _torch_float_dtype(value: str) -> torch.dtype:
    normalized = str(value).strip().lower()
    dtypes = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtypes[normalized]
    except KeyError as error:
        raise ValueError(
            "candidate_pool.residue_load_dtype must be float16, float32, or bfloat16"
        ) from error


def _configure_cpu_threads(inference: dict[str, Any]) -> None:
    value = inference.get("cpu_threads")
    if value is None:
        return
    if type(value) is not int or value <= 0:
        raise ValueError("inference.cpu_threads must be a positive integer")
    value_text = str(value)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = value_text
    interop_threads = min(value, 4)
    if torch.get_num_interop_threads() != interop_threads:
        try:
            torch.set_num_interop_threads(interop_threads)
        except RuntimeError as error:
            raise RuntimeError(
                "inference.cpu_threads must be configured before PyTorch parallel work starts"
            ) from error
    torch.set_num_threads(value)
    print(
        f"CPU thread pools: intra-op={value}, inter-op={interop_threads}",
        flush=True,
    )


def _safe_query_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not slug:
        raise ValueError("reaction.id must contain at least one alphanumeric character")
    return slug


def _stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def validate_reaction_smiles(reaction_smiles: str) -> list[str]:
    """Validate reaction syntax and return non-fatal chemistry warnings."""

    fields = reaction_smiles.strip().split(">")
    if len(fields) != 3 or not fields[0] or not fields[2]:
        raise ValueError(
            "reaction.smiles must be reaction SMILES in reactants>>products or "
            "reactants>agents>products form"
        )

    warnings: list[str] = []
    for side_name, side in (("reactants", fields[0]), ("products", fields[2])):
        for component_index, component in enumerate(side.split("."), start=1):
            molecule = Chem.MolFromSmiles(component)
            if molecule is None:
                raise ValueError(
                    f"Invalid molecule SMILES in {side_name} component {component_index}: "
                    f"{component}"
                )
            radical_count = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
            if radical_count:
                warnings.append(
                    f"{side_name} component {component_index} contains "
                    f"{radical_count} radical electron(s)"
                )
            unassigned = Chem.FindMolChiralCenters(
                molecule,
                includeUnassigned=True,
                useLegacyImplementation=False,
            )
            if any(configuration == "?" for _, configuration in unassigned):
                warnings.append(
                    f"{side_name} component {component_index} has unassigned stereocenters"
                )
    return warnings


def load_query_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for section in ("reaction", "model", "candidate_pool", "feature_generation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Query config requires a '{section}' mapping")
    config["_config_path"] = str(config_path)
    return config


def candidate_pool_with_root_override(
    candidate_settings: dict[str, Any],
    root: str | Path | None,
) -> dict[str, Any]:
    """Point standard candidate-pool artifacts at one alternate database root."""

    updated = dict(candidate_settings)
    if root in {None, ""}:
        return updated
    root_path = _resolve_path(str(root))
    for setting, filename in CANDIDATE_POOL_ROOT_FILES.items():
        updated[setting] = str(root_path / filename)
    return updated


def bucket_candidate_keys_by_residue_length(
    dataset: ResidueEmbedDataset,
    candidate_keys: list[str],
    *,
    batch_size: int,
    window_size: int,
) -> tuple[list[str], dict[str, int | float | str]]:
    """Locally sort candidates by length to limit padding while retaining I/O locality."""

    if batch_size <= 0:
        raise ValueError("candidate_pool.encoding_batch_size must be positive")
    if window_size < batch_size or window_size % batch_size:
        raise ValueError(
            "candidate_pool.encoding_length_bucket_window must be a positive multiple "
            "of encoding_batch_size"
        )
    if dataset.h5_indices is None:
        raise RuntimeError("Residue HDF5 index mapping is not initialized")

    max_tokens = dataset.max_tokens
    lengths_by_key: dict[str, int] = {}
    for dataset_index, protein_id in enumerate(dataset.keys):
        h5_index = int(dataset.h5_indices[dataset_index].item())
        length = int(dataset.offsets[h5_index + 1] - dataset.offsets[h5_index])
        lengths_by_key[protein_id] = min(length, max_tokens) if max_tokens else length

    missing = [protein_id for protein_id in candidate_keys if protein_id not in lengths_by_key]
    if missing:
        raise ValueError(f"{len(missing)} candidate IDs are absent from the residue length index")

    def padded_tokens(keys: list[str]) -> int:
        total = 0
        for start in range(0, len(keys), batch_size):
            batch = keys[start : start + batch_size]
            if batch:
                total += max(lengths_by_key[protein_id] for protein_id in batch) * len(batch)
        return total

    reordered: list[str] = []
    for start in range(0, len(candidate_keys), window_size):
        window = candidate_keys[start : start + window_size]
        reordered.extend(sorted(window, key=lengths_by_key.__getitem__))

    actual_tokens = sum(lengths_by_key[protein_id] for protein_id in candidate_keys)
    padded_before = padded_tokens(candidate_keys)
    padded_after = padded_tokens(reordered)
    return reordered, {
        "encoding_order": "local_length_bucket",
        "encoding_batch_size": batch_size,
        "encoding_length_bucket_window": window_size,
        "encoding_actual_token_count": actual_tokens,
        "encoding_padded_token_count_before": padded_before,
        "encoding_padded_token_count": padded_after,
        "encoding_padding_factor_before": padded_before / max(actual_tokens, 1),
        "encoding_padding_factor": padded_after / max(actual_tokens, 1),
    }


def _write_reaction_csv(path: Path, reaction_id: str, reaction_smiles: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reaction_id", "reaction_smiles"])
        writer.writeheader()
        writer.writerow({"reaction_id": reaction_id, "reaction_smiles": reaction_smiles})


def _configure_runtime_environment(workspace: Path) -> None:
    runtime_root = workspace / "runtime"
    directories = {
        "HOME": runtime_root / "nohome",
        "HF_HOME": runtime_root / "cache/huggingface",
        "TORCH_HOME": runtime_root / "cache/torch",
        "XDG_CACHE_HOME": runtime_root / "cache/xdg",
        "XDG_CONFIG_HOME": runtime_root / "config/xdg",
        "XDG_STATE_HOME": runtime_root / "state/xdg",
        "MPLCONFIGDIR": runtime_root / "config/matplotlib",
    }
    for variable, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(directory)
    os.environ["TRANSFORMERS_CACHE"] = str(directories["HF_HOME"] / "transformers")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _run_logged(
    *,
    name: str,
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
) -> None:
    print(f"Generating {name} features...", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        raise RuntimeError(
            f"{name} extraction failed with return code {completed.returncode}. "
            f"Log: {log_path}\n" + "\n".join(tail)
        )


def prepare_reaction_features(
    config: dict[str, Any],
    workspace: Path,
    *,
    force: bool,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    reaction = config["reaction"]
    feature_config = config["feature_generation"]
    reaction_id = _safe_query_id(str(_required(reaction, "id", "reaction")))
    reaction_smiles = str(_required(reaction, "smiles", "reaction"))
    warnings = validate_reaction_smiles(reaction_smiles)

    reaction_csv = workspace / "reaction.csv"
    _write_reaction_csv(reaction_csv, reaction_id, reaction_smiles)
    feature_dir = workspace / "features"
    log_dir = workspace / "logs"
    feature_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "reaction_t5": feature_dir / "reactiont5v2.h5",
        "unimol2": feature_dir / "unimol2.h5",
        "chiro": feature_dir / "chiro.h5",
        "chemistry": feature_dir / "reaction_set_features.npz",
    }
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(PROJECT_ROOT), existing_pythonpath) if value
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = str(config.get("inference", {}).get("device", "cuda"))

    if force or not outputs["reaction_t5"].is_file():
        reaction_t5_model = _resolve_path(
            _required(feature_config, "reaction_t5_model", "feature_generation")
        )
        _run_logged(
            name="ReactionT5v2",
            command=[
                sys.executable,
                "scripts/extract_reaction_t5v2_embeddings.py",
                "--reactions",
                str(reaction_csv),
                "--output",
                str(outputs["reaction_t5"]),
                "--model-name",
                str(reaction_t5_model),
                "--batch-size",
                "1",
                "--max-length",
                str(feature_config.get("reaction_t5_max_length", 512)),
                "--pooling",
                str(feature_config.get("reaction_t5_pooling", "mean")),
                "--device",
                device,
                "--dtype",
                "float16",
                "--no-bidirectional",
                "--no-allow-pseudo-reactions",
                "--force",
            ],
            log_path=log_dir / "reaction_t5.log",
            environment=environment,
        )

    if force or not outputs["unimol2"].is_file():
        unimol_environment = dict(environment)
        unimol_pythonpath = [
            str(PROJECT_ROOT / ".deps/unimol_tools"),
            str(PROJECT_ROOT.parent / "env/unimol2_site"),
            str(PROJECT_ROOT),
            existing_pythonpath,
        ]
        unimol_environment["PYTHONPATH"] = os.pathsep.join(
            value for value in unimol_pythonpath if value
        )
        weight_dir = feature_config.get("unimol_weight_dir")
        if weight_dir:
            unimol_environment["UNIMOL_WEIGHT_DIR"] = str(_resolve_path(weight_dir))
        _run_logged(
            name="UniMol2",
            command=[
                sys.executable,
                "scripts/extract_unimol2_reaction_embeddings.py",
                "--reactions",
                str(reaction_csv),
                "--output",
                str(outputs["unimol2"]),
                "--batch-size",
                "16",
                "--dtype",
                "float16",
                "--compression",
                "none",
                "--no-bidirectional",
                "--no-allow-pseudo-reactions",
                "--skip-invalid-molecules",
                "--skip-invalid-reactions",
                "--force",
            ],
            log_path=log_dir / "unimol2.log",
            environment=unimol_environment,
        )

    if force or not outputs["chiro"].is_file():
        chiro_environment = dict(environment)
        chiro_pythonpath = [
            str(PROJECT_ROOT / ".deps/python"),
            str(PROJECT_ROOT / ".deps/ChIRo"),
            str(PROJECT_ROOT),
            existing_pythonpath,
        ]
        chiro_environment["PYTHONPATH"] = os.pathsep.join(
            value for value in chiro_pythonpath if value
        )
        _run_logged(
            name="ChIRo",
            command=[
                sys.executable,
                "scripts/extract_chiro_reaction_embeddings.py",
                "--reactions",
                str(reaction_csv),
                "--output",
                str(outputs["chiro"]),
                "--device",
                device,
                "--num-workers",
                str(feature_config.get("chiro_num_workers", 1)),
                "--batch-size",
                "16",
                "--no-bidirectional",
                "--no-allow-pseudo-reactions",
                "--force",
            ],
            log_path=log_dir / "chiro.log",
            environment=chiro_environment,
        )

    if force or not outputs["chemistry"].is_file():
        chemistry_python = _resolve_path(feature_config.get("chemistry_python", sys.executable))
        _run_logged(
            name="train-schema chemistry",
            command=[
                str(chemistry_python),
                "wet_lab/materialize_chemistry.py",
                "--reactions",
                str(reaction_csv),
                "--schema",
                str(
                    _resolve_path(
                        _required(
                            feature_config,
                            "reaction_chemistry_schema",
                            "feature_generation",
                        )
                    )
                ),
                "--cofactor-dictionary",
                str(
                    _resolve_path(
                        _required(
                            feature_config,
                            "cofactor_dictionary",
                            "feature_generation",
                        )
                    )
                ),
                "--output",
                str(outputs["chemistry"]),
            ],
            log_path=log_dir / "chemistry.log",
            environment=environment,
        )
        chemistry_report = {
            "status": "generated",
            "output": str(outputs["chemistry"]),
        }
    else:
        chemistry_report = {"status": "reused", "output": str(outputs["chemistry"])}

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reaction_id": reaction_id,
        "reaction_smiles": reaction_smiles,
        "warnings": warnings,
        "features": {name: str(path) for name, path in outputs.items()},
        "chemistry": chemistry_report,
    }
    with (workspace / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return reaction_csv, outputs, manifest


def streaming_topk(
    query_embedding: torch.Tensor,
    target_embeddings: torch.Tensor,
    *,
    top_k: int,
    device: str,
    batch_size: int = 32768,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact cosine top-k scores and row indices without a dense score matrix."""

    if query_embedding.ndim == 2:
        if query_embedding.shape[0] != 1:
            raise ValueError("streaming_topk accepts exactly one query embedding")
        query_embedding = query_embedding[0]
    if query_embedding.ndim != 1 or target_embeddings.ndim != 2:
        raise ValueError("Expected query [dim] and target embeddings [num_targets, dim]")
    if query_embedding.shape[0] != target_embeddings.shape[1]:
        raise ValueError("Query and target embedding dimensions differ")
    if top_k <= 0 or batch_size <= 0:
        raise ValueError("top_k and batch_size must be positive")

    top_k = min(top_k, target_embeddings.shape[0])
    query = F.normalize(query_embedding.float().to(device), p=2, dim=0)
    best_scores = torch.empty(0, dtype=torch.float32, device=device)
    best_indices = torch.empty(0, dtype=torch.long, device=device)

    with torch.inference_mode():
        for start in range(0, target_embeddings.shape[0], batch_size):
            end = min(start + batch_size, target_embeddings.shape[0])
            targets = F.normalize(
                target_embeddings[start:end].float().to(device),
                p=2,
                dim=1,
            )
            scores = torch.mv(targets, query)
            local_k = min(top_k, scores.numel())
            local_scores, local_indices = torch.topk(
                scores,
                k=local_k,
                largest=True,
                sorted=True,
            )
            local_indices = local_indices + start
            combined_scores = torch.cat((best_scores, local_scores))
            combined_indices = torch.cat((best_indices, local_indices))
            keep_k = min(top_k, combined_scores.numel())
            best_scores, positions = torch.topk(
                combined_scores,
                k=keep_k,
                largest=True,
                sorted=True,
            )
            best_indices = combined_indices[positions]
    return best_scores.cpu(), best_indices.cpu()


def _load_fasta_sequences(path: Path, protein_ids: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id in protein_ids:
                    sequences[current_id] = "".join(chunks)
                current_id = line[1:].split(maxsplit=1)[0]
                chunks = []
            elif current_id in protein_ids:
                chunks.append(line)
        if current_id in protein_ids:
            sequences[current_id] = "".join(chunks)
    return sequences


def _load_metadata(
    path: Path,
    *,
    id_column: str,
    protein_ids: set[str],
) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if id_column not in (reader.fieldnames or []):
            raise ValueError(f"Metadata CSV {path} has no '{id_column}' column")
        for row in reader:
            protein_id = str(row[id_column])
            if protein_id in protein_ids:
                metadata[protein_id] = {
                    key: str(value)
                    for key, value in row.items()
                    if key != id_column and value not in {None, ""}
                }
    return metadata


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["rank", "protein_id", "cosine_similarity", "sequence_length", "sequence"]
    metadata_fields = sorted({key for row in rows for key in row if key not in set(fieldnames)})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + metadata_fields)
        writer.writeheader()
        writer.writerows(rows)


def write_rankings(
    *,
    workspace: Path,
    reaction: dict[str, Any],
    model: dict[str, Any],
    candidate_stats: dict[str, Any],
    cache_info: dict[str, Any] | None,
    top_k_values: list[int],
    candidate_keys: list[str],
    scores: torch.Tensor,
    indices: torch.Tensor,
    candidate_config: dict[str, Any],
    feature_manifest: dict[str, Any],
) -> Path:
    protein_ids = [candidate_keys[index] for index in indices.tolist()]
    wanted = set(protein_ids)
    fasta_path_value = candidate_config.get("fasta")
    sequences = (
        _load_fasta_sequences(_resolve_path(fasta_path_value), wanted) if fasta_path_value else {}
    )
    metadata_path_value = candidate_config.get("metadata_csv")
    metadata = (
        _load_metadata(
            _resolve_path(metadata_path_value),
            id_column=str(candidate_config.get("metadata_id_column", "protein_id")),
            protein_ids=wanted,
        )
        if metadata_path_value
        else {}
    )

    rows: list[dict[str, Any]] = []
    for rank, (protein_id, score) in enumerate(zip(protein_ids, scores.tolist()), start=1):
        sequence = sequences.get(protein_id, "")
        rows.append(
            {
                "rank": rank,
                "protein_id": protein_id,
                "cosine_similarity": float(score),
                "sequence_length": len(sequence) if sequence else "",
                "sequence": sequence,
                **metadata.get(protein_id, {}),
            }
        )

    _write_csv(workspace / "rankings.csv", rows)
    for cutoff in top_k_values:
        _write_csv(workspace / f"top_{cutoff}.csv", rows[:cutoff])

    if sequences:
        with (workspace / f"top_{max(top_k_values)}.fasta").open("w", encoding="utf-8") as handle:
            for row in rows:
                sequence = row["sequence"]
                if not sequence:
                    continue
                handle.write(
                    f">{row['protein_id']} rank={row['rank']} "
                    f"cosine_similarity={row['cosine_similarity']:.8f}\n"
                )
                for start in range(0, len(sequence), 80):
                    handle.write(sequence[start : start + 80] + "\n")

    result = {
        "schema_version": SCHEMA_VERSION,
        "reaction": {
            "id": reaction["id"],
            "smiles": reaction["smiles"],
            "warnings": feature_manifest["warnings"],
        },
        "model": {
            "name": model.get("name", ""),
            "checkpoint": str(_resolve_path(model["checkpoint"])),
            "config": str(_resolve_path(model["config"])),
        },
        "scoring": {
            "type": "cosine_similarity",
            "calibrated_probability": False,
            "top_k": top_k_values,
        },
        "candidate_pool": {
            **candidate_stats,
            "target_embedding_cache": cache_info,
        },
        "rankings": [
            {key: value for key, value in row.items() if key != "sequence"} for row in rows
        ],
        "artifacts": {
            "rankings_csv": str(workspace / "rankings.csv"),
            "top_k_csv": {
                str(cutoff): str(workspace / f"top_{cutoff}.csv") for cutoff in top_k_values
            },
            "fasta": (str(workspace / f"top_{max(top_k_values)}.fasta") if sequences else None),
            "feature_manifest": str(workspace / "feature_manifest.json"),
        },
    }
    result_path = workspace / "results.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result_path


def run_query(config: dict[str, Any], *, force_features: bool = False) -> Path:
    reaction = config["reaction"]
    model_settings = config["model"]
    candidate_settings = candidate_pool_with_root_override(
        config["candidate_pool"],
        os.environ.get("WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE"),
    )
    inference = config.get("inference", {})
    _configure_cpu_threads(inference)

    query_id = _safe_query_id(str(_required(reaction, "id", "reaction")))
    reaction_smiles = str(_required(reaction, "smiles", "reaction"))
    model_config_path = _resolve_path(_required(model_settings, "config", "model"))
    checkpoint_path = _resolve_path(_required(model_settings, "checkpoint", "model"))
    run_hash = _stable_hash(
        {
            "reaction_smiles": reaction_smiles,
            "feature_generation": config["feature_generation"],
            "model": model_settings,
            "candidate_pool": {
                key: candidate_settings.get(key)
                for key in (
                    "ids",
                    "residue_embeddings",
                    "score_residue_embeddings",
                    "protein_embedding",
                    "score_protein_embedding",
                )
            },
        }
    )
    output_root = _resolve_path(
        config.get("output", {}).get("directory", "wet_lab/runs"),
        must_exist=False,
    )
    workspace = output_root / f"{query_id}_{run_hash}"
    workspace.mkdir(parents=True, exist_ok=True)
    _configure_runtime_environment(workspace)

    reaction_csv, feature_paths, feature_manifest = prepare_reaction_features(
        config,
        workspace,
        force=force_features,
    )

    model_config = load_config(str(model_config_path))
    if model_kind_from_config(model_config) != "residue":
        raise ValueError("wet_lab query currently requires a residue-level enzyme model")
    model_config.data.reaction_t5v2_embeds_path = str(feature_paths["reaction_t5"])
    model_config.data.reaction_unimol2_embeds_path = str(feature_paths["unimol2"])
    model_config.data.reaction_chiro_embeds_path = str(feature_paths["chiro"])
    model_config.data.reaction_chirality_embeds_path = str(feature_paths["chiro"])
    model_config.data.reaction_chienn_embeds_path = str(feature_paths["chiro"])
    model_config.data.reaction_chemistry_vectors_path = str(feature_paths["chemistry"])

    task = BenchmarkTask(
        name="wet_lab_query",
        task_type="retrieval",
        dataset="wet_lab",
        task_label=query_id,
        split="query",
        pairs=reaction_csv,
        reactions=reaction_csv,
        reaction_model_embeds_h5=feature_paths["reaction_t5"],
        reaction_unimol2_embeds_h5=feature_paths["unimol2"],
        reaction_chiro_embeds_h5=feature_paths["chiro"],
        directions=("reaction_to_enzyme",),
        top_k=tuple(DEFAULT_TOP_K),
    )
    reaction_inputs = build_reaction_inputs(task, model_config)
    if query_id not in set(reaction_inputs.keys):
        raise ValueError(f"Reaction '{query_id}' was removed while assembling multimodal features")

    residue_h5_setting = os.environ.get(
        "WET_LAB_RESIDUE_EMBEDDINGS_OVERRIDE",
        _required(candidate_settings, "residue_embeddings", "candidate_pool"),
    )
    residue_h5 = _resolve_path(residue_h5_setting)
    candidate_ids_path = _resolve_path(_required(candidate_settings, "ids", "candidate_pool"))
    residue_load_dtype_name = str(candidate_settings.get("residue_load_dtype", "float32"))
    residue_load_dtype = _torch_float_dtype(residue_load_dtype_name)
    validate_finite_on_access = candidate_settings.get("validate_finite_on_access", True)
    if type(validate_finite_on_access) is not bool:
        raise ValueError("candidate_pool.validate_finite_on_access must be boolean")
    target_dataset = ResidueEmbedDataset(
        str(residue_h5),
        in_memory=False,
        dtype=residue_load_dtype,
        max_tokens=int(model_config.data.get("max_protein_tokens", 1024)),
        truncation=str(model_config.data.get("protein_truncation", "ends_center")),
        validate_finite_on_access=validate_finite_on_access,
    )
    candidate_keys, candidate_stats = load_candidate_keys_from_residue(
        target_dataset,
        candidate_ids_path,
    )
    if not candidate_keys:
        raise ValueError("No valid candidate enzymes remain after filtering")

    score_dataset = None
    score_residue_h5 = None
    if needs_score_residue_embeddings(model_config):
        score_residue_h5 = _resolve_path(
            _required(
                candidate_settings,
                "score_residue_embeddings",
                "candidate_pool",
            )
        )
        score_dataset = ResidueEmbedDataset(
            str(score_residue_h5),
            in_memory=False,
            dtype=residue_load_dtype,
            max_tokens=int(model_config.data.get("max_protein_tokens", 1024)),
            truncation=str(model_config.data.get("protein_truncation", "ends_center")),
            validate_finite_on_access=validate_finite_on_access,
        )
        candidate_keys, score_stats = filter_candidate_keys_by_score_residue(
            candidate_keys,
            score_dataset,
        )
        candidate_stats.update(score_stats)

    encoding_batch_size = int(candidate_settings.get("encoding_batch_size", 2048))
    bucket_window = candidate_settings.get("encoding_length_bucket_window")
    if bucket_window is not None:
        candidate_keys, encoding_order_stats = bucket_candidate_keys_by_residue_length(
            target_dataset,
            candidate_keys,
            batch_size=encoding_batch_size,
            window_size=int(bucket_window),
        )
        candidate_stats.update(encoding_order_stats)
        print(
            "Length-bucketed candidate encoding: "
            f"padding factor {encoding_order_stats['encoding_padding_factor_before']:.3f} "
            f"-> {encoding_order_stats['encoding_padding_factor']:.3f}",
            flush=True,
        )

    capability_dataset = load_capability_vectors_from_config(model_config)
    text_dataset = load_text_vectors_from_config(model_config)
    capability_path = model_config.data.get("protein_capability_vectors_path", None)
    text_path = model_config.data.get("protein_text_vectors_path", None)
    cache_metadata = _target_cache_base_metadata(
        kind="residue",
        checkpoint=checkpoint_path,
        config_path=model_config_path,
        protein_embedding=str(candidate_settings.get("protein_embedding", "prott5")),
        score_protein_embedding=str(candidate_settings.get("score_protein_embedding", "prott5")),
        residue_h5=residue_h5,
        score_residue_h5=score_residue_h5,
        capability_vectors_path=capability_path,
        capability_missing_policy=model_config.data.get("capability_missing_policy", None),
        text_vectors_path=text_path,
        text_vector_missing_policy=model_config.data.get("text_vector_missing_policy", None),
        max_tokens=int(model_config.data.get("max_protein_tokens", 1024)),
        truncation=str(model_config.data.get("protein_truncation", "ends_center")),
    )
    cache_metadata["residue_load_dtype"] = residue_load_dtype_name
    target_cache_dir = _resolve_path(
        _required(candidate_settings, "target_cache_dir", "candidate_pool"),
        must_exist=False,
    )
    device = str(inference.get("device", "cuda"))
    target_embeddings, cache_info = prepare_target_embedding_cache(
        target_cache_dir,
        cache_metadata,
        candidate_keys,
        device=device,
        store_on_device=False,
    )

    module = None
    try:
        print(f"Loading model checkpoint: {checkpoint_path}", flush=True)
        module, loaded_kind = load_repo_checkpoint(checkpoint_path, model_config, device)
        if loaded_kind != "residue":
            raise RuntimeError(f"Unexpected loaded model kind: {loaded_kind}")
        if target_embeddings is None:
            print(
                f"Encoding {len(candidate_keys):,} candidate enzymes; this is a one-time "
                "operation for this checkpoint and candidate pool.",
                flush=True,
            )
            target_embeddings = encode_residue_targets(
                module,
                target_dataset,
                candidate_keys,
                device,
                encoding_batch_size,
                store_on_device=False,
                score_dataset=score_dataset,
                capability_dataset=capability_dataset,
                text_dataset=text_dataset,
                progress_every_batches=int(
                    candidate_settings.get("encoding_progress_every_batches", 25)
                ),
            )
            write_target_embedding_cache(
                cache_info,
                cache_metadata,
                candidate_keys,
                target_embeddings,
            )
        query_embedding = encode_reactions(
            module,
            reaction_inputs,
            [query_id],
            device,
            batch_size=1,
        ).detach()
    except BaseException:
        release_target_embedding_cache_lock(cache_info)
        raise
    finally:
        if module is not None:
            del module
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    top_k_values = sorted(
        {int(value) for value in inference.get("top_k", DEFAULT_TOP_K) if int(value) > 0}
    )
    if not top_k_values:
        raise ValueError("inference.top_k must contain at least one positive integer")
    scores, indices = streaming_topk(
        query_embedding,
        target_embeddings,
        top_k=max(top_k_values),
        device=device,
        batch_size=int(inference.get("score_batch_size", 32768)),
    )
    result_path = write_rankings(
        workspace=workspace,
        reaction={"id": query_id, "smiles": reaction_smiles},
        model=model_settings,
        candidate_stats={
            **candidate_stats,
            "candidate_count": len(candidate_keys),
            "ids": str(candidate_ids_path),
            "residue_embeddings": str(residue_h5),
        },
        cache_info=cache_info,
        top_k_values=top_k_values,
        candidate_keys=candidate_keys,
        scores=scores,
        indices=indices,
        candidate_config=candidate_settings,
        feature_manifest=feature_manifest,
    )
    print(f"Saved ranked enzymes to: {result_path}", flush=True)
    print("Top candidates:", flush=True)
    for rank, (index, score) in enumerate(zip(indices[:10], scores[:10]), start=1):
        print(
            f"{rank:>3}  {candidate_keys[int(index)]:<16} " f"cosine={float(score):.6f}",
            flush=True,
        )
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Wet-lab query YAML configuration")
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Regenerate reaction features even when the query cache exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config = load_query_config(args.config)
    run_query(config, force_features=args.force_features)


if __name__ == "__main__":
    main()
