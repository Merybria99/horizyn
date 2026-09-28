#!/usr/bin/env python3
"""Retrieve candidate enzymes for one reaction SMILES."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from rdkit import Chem
from threadpoolctl import threadpool_limits

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
from horizyn.capability.reaction_set_features import molecule_set_components
from horizyn.chemistry.reaction_recovery import canonical_molecule

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
CPU_THREAD_LIMIT_ENV_VAR = "HORIZYN_CPU_THREADS"
_THREADPOOL_LIMITER: Any | None = None


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


def _configure_cuda_memory(inference: dict[str, Any]) -> None:
    """Limit this process's PyTorch allocator, not other jobs or CUDA overhead."""
    limit = inference.get("cuda_memory_limit_gib")
    if limit is None:
        return
    device = torch.device(str(inference.get("device", "cuda")))
    if device.type != "cuda" or isinstance(limit, bool):
        raise ValueError("cuda_memory_limit_gib requires CUDA and a positive number")
    limit = float(limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("cuda_memory_limit_gib must be finite and positive")
    if torch.cuda.get_allocator_backend() != "native":
        raise ValueError("cuda_memory_limit_gib requires the native PyTorch CUDA allocator")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = limit * (1024**3) / total
    if fraction > 1:
        raise ValueError("Requested CUDA allocation limit exceeds device memory")
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    print(f"PyTorch CUDA allocation limit: {limit:g} GiB on {device} (plus CUDA overhead)", flush=True)


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
    global _THREADPOOL_LIMITER

    configured_value = inference.get("cpu_threads")
    environment_value = os.environ.get(CPU_THREAD_LIMIT_ENV_VAR)
    value: Any = environment_value if environment_value is not None else configured_value
    if value is None:
        return
    if isinstance(value, bool):
        raise ValueError(
            f"{CPU_THREAD_LIMIT_ENV_VAR} or inference.cpu_threads must be a positive integer"
        )
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{CPU_THREAD_LIMIT_ENV_VAR} or inference.cpu_threads must be a positive integer"
        ) from error
    if value <= 0:
        raise ValueError(
            f"{CPU_THREAD_LIMIT_ENV_VAR} or inference.cpu_threads must be a positive integer"
        )
    value_text = str(value)
    for variable in CPU_THREAD_ENV_VARS:
        os.environ[variable] = value_text
    _THREADPOOL_LIMITER = threadpool_limits(limits=value)
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
        f"CPU thread pools: intra-op={value}, inter-op={interop_threads}, "
        f"source={'environment' if environment_value is not None else 'config'}",
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
    normalize_molecule_sets_as_self_reactions: bool = False,
    reaction_input_policy: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    reaction = config["reaction"]
    feature_config = config["feature_generation"]
    reaction_id = _safe_query_id(str(_required(reaction, "id", "reaction")))
    reaction_smiles = str(_required(reaction, "smiles", "reaction"))
    warnings = validate_reaction_smiles(reaction_smiles)
    encoded_smiles = reaction_smiles_for_model(
        reaction_smiles,
        normalize_molecule_sets_as_self_reactions=normalize_molecule_sets_as_self_reactions,
    )
    feature_policy = "participant_self_reaction_v1" if normalize_molecule_sets_as_self_reactions else "physical_reaction"
    if normalize_molecule_sets_as_self_reactions:
        warnings.append("Participant-set model: reaction direction is not encoded.")
    if reaction_input_policy and reaction_input_policy.get("warning"):
        warnings.append(reaction_input_policy["warning"])
    feature_dir = workspace / "features"
    log_dir = workspace / "logs"
    feature_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "reaction_t5": feature_dir / "reactiont5v2.h5",
        "unimol2": feature_dir / "unimol2.h5",
        "chiro": feature_dir / "chiro.h5",
        "chemistry": feature_dir / "reaction_set_features.npz",
    }
    # Never reuse a cache built from another representation or overwrite its
    # provenance. This also protects callers that reuse an explicit workspace.
    previous_path = workspace / "feature_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else None
    has_existing = any(path.is_file() for path in outputs.values())
    compatible = previous is not None and (
        previous.get("reaction_id") == reaction_id
        and previous.get("reaction_smiles") == reaction_smiles
        and previous.get("encoded_reaction_smiles", previous.get("reaction_smiles")) == encoded_smiles
        and previous.get("reaction_feature_policy", "physical_reaction") == feature_policy
    )
    if has_existing and (force or not compatible):
        backup = workspace / "feature_history" / str(time.time_ns())
        backup.mkdir(parents=True)
        for path in [*outputs.values(), previous_path, workspace / "reaction.csv", workspace / "feature_reaction.csv"]:
            if path.is_file():
                shutil.copy2(path, backup / path.name)
        force = True
    _write_reaction_csv(workspace / "reaction.csv", reaction_id, reaction_smiles)
    reaction_csv = workspace / "feature_reaction.csv" if normalize_molecule_sets_as_self_reactions else workspace / "reaction.csv"
    _write_reaction_csv(reaction_csv, reaction_id, encoded_smiles)
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(PROJECT_ROOT), existing_pythonpath) if value
    )
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = str(feature_config.get("device", config.get("inference", {}).get("device", "cuda")))
    if torch.device(device).type == "cpu":
        # UniMol2 selects its device automatically; keep all extractor children
        # off the GPU when the feature-generation device is explicitly CPU.
        environment["CUDA_VISIBLE_DEVICES"] = ""

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
        "encoded_reaction_smiles": encoded_smiles,
        "reaction_feature_policy": feature_policy,
        "reaction_input_policy": reaction_input_policy,
        "warnings": warnings,
        "features": {name: str(path) for name, path in outputs.items()},
        "chemistry": chemistry_report,
    }
    with (workspace / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return reaction_csv, outputs, manifest


def reaction_smiles_for_model(
    reaction_smiles: str, *, normalize_molecule_sets_as_self_reactions: bool
) -> str:
    """Apply the training representation before every reaction extractor.

    A participant model reads one molecular bank, so it must receive products
    there as well as reactants. An existing S>>S representation contributes S
    once, preserving stoichiometric multiplicities and making this idempotent.
    """
    if not normalize_molecule_sets_as_self_reactions:
        return reaction_smiles
    participants = sorted(canonical_molecule(value) for value in molecule_set_components(reaction_smiles))
    if not participants:
        raise ValueError("The participant collection is empty")
    side = ".".join(participants)
    return f"{side}>>{side}"


def resolve_reaction_input_policy(config: dict[str, Any], model_config: Any) -> dict[str, Any]:
    """Infer representation from actual training input, never the wrap flag alone.

    normalize_molecule_sets_as_self_reactions only wraps strings without an
    arrow during training; a directional training catalog remains directional.
    An explicit query override is available for checkpoints with incomplete
    lineage or a documented alternative reaction representation.
    """
    allowed = {"participant_self_reaction", "physical_reaction"}
    explicit = config["feature_generation"].get("reaction_input_policy")
    if explicit is not None:
        if explicit not in allowed:
            raise ValueError(f"feature_generation.reaction_input_policy must be one of {sorted(allowed)}")
        return dict(policy=explicit, source="explicit_query_configuration")
    if not model_config.data.get("normalize_molecule_sets_as_self_reactions", False):
        return dict(policy="physical_reaction", source="training_normalization_disabled")
    training_path = model_config.data.get("train_reactions_path")
    if not training_path:
        return dict(policy="physical_reaction", source="legacy_missing_training_catalog",
                    warning="No training reaction catalog is configured; preserving the physical equation. Set feature_generation.reaction_input_policy explicitly if required.")
    path = _resolve_path(training_path, must_exist=False)
    if not path.is_file():
        return dict(policy="physical_reaction", source="legacy_unavailable_training_catalog",
                    warning="Training reaction catalog is unavailable; preserving the physical equation. Set feature_generation.reaction_input_policy explicitly if required.")
    counts = dict(participant_sets=0, self_reactions=0, directional_reactions=0)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            smiles = row["reaction_smiles"].strip()
            if not smiles:
                raise ValueError(f"Empty training reaction in {path}")
            if ">" not in smiles:
                counts["participant_sets"] += 1
            elif smiles.count(">>") == 1:
                left, right = smiles.split(">>")
                lhs = sorted(canonical_molecule(value) for value in left.split(".") if value)
                rhs = sorted(canonical_molecule(value) for value in right.split(".") if value)
                counts["self_reactions" if lhs == rhs else "directional_reactions"] += 1
            else:
                raise ValueError(f"Ambiguous training reaction syntax in {path}; set feature_generation.reaction_input_policy explicitly")
    if not sum(counts.values()) or (counts["participant_sets"] and counts["directional_reactions"]):
        raise ValueError(f"Ambiguous mixed/empty training reaction catalog {path}; set feature_generation.reaction_input_policy explicitly")
    policy = "physical_reaction" if counts["directional_reactions"] else "participant_self_reaction"
    return dict(policy=policy, source="training_reaction_catalog", training_reactions=str(path),
                training_reactions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), counts=counts)


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
    score_type: str = "cosine_similarity",
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
                score_type: float(score),
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
                    f"{score_type}={row[score_type]:.8f}\n"
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
            "type": score_type,
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


def rank_query_optional_bundle(config: dict[str, Any], **arguments):
    """Retain legacy ranking unless an explicit frozen bundle is configured."""
    if config["model"].get("generalization_bundle"):
        from wet_lab.generalization import rank_generalized_query
        return rank_generalized_query(config=config, **arguments)
    scores, indices = streaming_topk(
        arguments["query_embedding"], arguments["target_embeddings"],
        top_k=arguments["top_k"], device=arguments["device"],
        batch_size=int(config.get("inference", {}).get("score_batch_size", 32768)),
    )
    return scores, indices, None


def run_query(config: dict[str, Any], *, force_features: bool = False) -> Path:
    reaction = config["reaction"]
    model_settings = config["model"]
    candidate_settings = candidate_pool_with_root_override(
        config["candidate_pool"],
        os.environ.get("WET_LAB_CANDIDATE_POOL_ROOT_OVERRIDE"),
    )
    inference = config.get("inference", {})
    if model_settings.get("generalization_bundle"):
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
    _configure_cpu_threads(inference)
    _configure_cuda_memory(inference)

    query_id = _safe_query_id(str(_required(reaction, "id", "reaction")))
    reaction_smiles = str(_required(reaction, "smiles", "reaction"))
    model_config_path = _resolve_path(_required(model_settings, "config", "model"))
    checkpoint_path = _resolve_path(_required(model_settings, "checkpoint", "model"))
    model_config = load_config(str(model_config_path))
    input_policy = resolve_reaction_input_policy(config, model_config)
    participant_model = input_policy["policy"] == "participant_self_reaction"
    representation_identity = {}
    if participant_model:
        representation_identity = {
            "reaction_feature_policy": "participant_self_reaction_v1",
            "encoded_reaction_smiles": reaction_smiles_for_model(
                reaction_smiles, normalize_molecule_sets_as_self_reactions=True
            ),
        }
    run_hash = _stable_hash(
        {
            **representation_identity,
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
        normalize_molecule_sets_as_self_reactions=participant_model,
        reaction_input_policy=input_policy,
    )

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
    allow_uncertified_finite_skip = candidate_settings.get("allow_uncertified_finite_skip", False)
    if type(allow_uncertified_finite_skip) is not bool:
        raise ValueError("candidate_pool.allow_uncertified_finite_skip must be boolean")
    target_dataset = ResidueEmbedDataset(
        str(residue_h5),
        in_memory=False,
        dtype=residue_load_dtype,
        max_tokens=int(model_config.data.get("max_protein_tokens", 1024)),
        truncation=str(model_config.data.get("protein_truncation", "ends_center")),
        validate_finite_on_access=validate_finite_on_access,
        finite_validation_sidecar=candidate_settings.get("finite_validation_sidecar"),
        allow_uncertified_finite_skip=allow_uncertified_finite_skip,
    )
    candidate_keys, candidate_stats = load_candidate_keys_from_residue(
        target_dataset,
        candidate_ids_path,
    )
    candidate_stats["residue_finite_validation"] = {
        "per_access": validate_finite_on_access,
        "certificate": target_dataset.finite_validation_sidecar,
        "uncertified_skip": (
            not validate_finite_on_access and target_dataset.finite_validation_sidecar is None
        ),
        "projected_batches": True,
    }
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
            finite_validation_sidecar=candidate_settings.get("score_finite_validation_sidecar"),
            allow_uncertified_finite_skip=allow_uncertified_finite_skip,
        )
        candidate_keys, score_stats = filter_candidate_keys_by_score_residue(
            candidate_keys,
            score_dataset,
        )
        candidate_stats.update(score_stats)
        candidate_stats["score_residue_finite_validation"] = {
            "per_access": validate_finite_on_access,
            "certificate": score_dataset.finite_validation_sidecar,
            "uncertified_skip": (
                not validate_finite_on_access and score_dataset.finite_validation_sidecar is None
            ),
            "projected_batches": True,
        }

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
                cache_info=cache_info,
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
    scores, indices, generalization_audit = rank_query_optional_bundle(
        config,
        workspace=workspace,
        checkpoint_path=checkpoint_path,
        candidate_keys=candidate_keys,
        residue_h5=residue_h5,
        feature_paths=feature_paths,
        feature_manifest=feature_manifest,
        query_id=query_id,
        query_embedding=query_embedding,
        target_embeddings=target_embeddings,
        top_k=max(top_k_values),
        device=device,
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
        score_type="generalization_similarity" if generalization_audit else "cosine_similarity",
    )
    if generalization_audit:
        result = json.loads(result_path.read_text())
        result["model"]["generalization_bundle"] = generalization_audit["source"]["bundle"]
        result["scoring"]["contract"] = generalization_audit["score"]
        result["artifacts"]["generalization_manifest"] = str(workspace / "generalization.json")
        result["generalization_diagnostics"] = {
            "query": generalization_audit["query_diagnostics"],
            "ranked_candidates": generalization_audit["candidate_diagnostics"],
            "calibrated_probability": False,
        }
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
    print(f"Saved ranked enzymes to: {result_path}", flush=True)
    print("Top candidates:", flush=True)
    for rank, (index, score) in enumerate(zip(indices[:10], scores[:10]), start=1):
        print(
            f"{rank:>3}  {candidate_keys[int(index)]:<16} "
            f"{'score' if generalization_audit else 'cosine'}={float(score):.6f}",
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
