#!/usr/bin/env python3
"""Pretrain a unimodal Lorentz enzyme encoder with EC hierarchy supervision."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.datasets.residue_hdf5 import (  # noqa: E402
    ResidueEmbedDataset,
    truncate_residue_embeddings,
)
from horizyn.hyperbolic_enzyme import (  # noqa: E402
    ECPrefixEntailmentLoss,
    HierarchyRankingLoss,
    LorentzEnzymeProjector,
    NonParametricCentroidConeLoss,
    SLEECGuidedAttentionPool,
    attention_mass_above_threshold,
    compute_ec_shared_depth,
    format_ec_prefixes,
    known_ec_depth,
    mean_attention_entropy,
    parse_ec_prefixes,
    radius_target_loss,
    same_ec4_radius_loss,
)
from horizyn.lorentz import (  # noqa: E402
    lorentz_constraint_error,
    lorentz_distance,
    lorentz_origin,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "residue_embeddings_path": "data/sota/prots_esm2_650m_residue.h5",
    "sleec_logits_path": None,
    "ec_labels_path": "data/sota/uniprot_all_ec_labels.csv",
    "id_column": None,
    "ec_column": None,
    "input_dim": 1280,
    "hyp_dim": 512,
    "curvature": 0.25,
    "p0_sleec_threshold": 0.34,
    "base_margin": 0.05,
    "batch_size": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "tangent_clip": 5.0,
    "tangent_clip_mode": "hard",
    "projector_input_normalization": "none",
    "eps": 1e-6,
    "max_triplets_per_anchor": 32,
    "alpha_radial": 0.0,
    "alpha_radius_target": 0.0,
    "radius_target": 2.0,
    "use_centroid_cone": False,
    "alpha_centroid_cone": 0.0,
    "use_ec_entailment": False,
    "alpha_ec_entailment": 0.05,
    "ec_entailment_warmup_epochs": 2,
    "ec_entailment_eta": 1.0,
    "ec_entailment_min_radius": 0.1,
    "ec_entailment_min_group_size": 2,
    "K": 0.1,
    "eta": 1.0,
    "cone_loss_power": 1.0,
    "centroid_min_group_size": 2,
    "sleec_checkpoint_path": None,
    "sleec_scorer_hidden_dim": 256,
    "attention_bias": True,
    "initial_prior_strength": 1.0,
    "train_prior_strength": True,
    "freeze_sleec": True,
    "max_protein_tokens": None,
    "protein_truncation": "ends_center",
    "output_checkpoint": "checkpoints/hyperbolic_ec/hyperbolic_enzyme_pretrain/last.ckpt",
    "resume_checkpoint": None,
    "device": "auto",
    "seed": 42,
    "num_workers": 0,
    "log_every": 50,
    "max_steps": None,
    "wandb": False,
    "wandb_project": "horizyn-hyperbolic-enzyme",
    "wandb_entity": "omnai",
    "wandb_run_name": None,
    "wandb_mode": "online",
    "wandb_tags": None,
    "wandb_log_model": False,
    "wandb_dir": "logs/wandb",
}

ID_COLUMN_CANDIDATES = (
    "protein_id",
    "enzyme_id",
    "uniprot_accession",
    "Entry",
    "entry",
    "uniprot_id",
)
EC_COLUMN_CANDIDATES = ("ec_number", "EC number", "ec", "EC")
EC_TOKEN_PATTERN = re.compile(r"[1-7](?:\.(?:\d+|-)){0,3}")


def _decode_hdf5_id(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _detect_column(header: list[str], candidates: tuple[str, ...], explicit: str | None) -> str:
    if explicit:
        if explicit not in header:
            raise ValueError(f"Column '{explicit}' not found; available columns: {header}")
        return explicit
    lower_to_column = {column.lower(): column for column in header}
    for candidate in candidates:
        match = lower_to_column.get(candidate.lower())
        if match is not None:
            return match
    raise ValueError(f"Could not detect column from candidates {candidates}; header={header}")


def parse_ec_numbers(value: str | None) -> list[str]:
    """Parse complete or prefix-only EC labels from a CSV cell.

    Incomplete EC labels are kept as normalized four-level labels with unknown
    levels set to ``-``. For example, ``1.1.1.-`` and ``1.1.1`` both normalize
    to ``1.1.1.-``. These labels contribute only their known hierarchy prefix
    to the ranking loss.
    """
    if value is None:
        return []
    output: list[str] = []
    seen: set[str] = set()
    text = str(value).replace(",", ";").replace("EC-", "").replace("EC:", "")
    for raw_ec in text.split(";"):
        raw_ec = raw_ec.strip()
        if not raw_ec:
            continue
        match = EC_TOKEN_PATTERN.search(raw_ec)
        if match is None:
            continue
        ec = match.group(0)
        try:
            normalized = format_ec_prefixes(parse_ec_prefixes(ec))
        except ValueError:
            continue
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def load_ec_rows(
    ec_labels_path: str | Path,
    id_column: str | None = None,
    ec_column: str | None = None,
) -> list[tuple[str, str]]:
    """Load protein IDs and complete or prefix-only EC labels from CSV/TSV."""
    rows: list[tuple[str, str]] = []
    with open(ec_labels_path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        header = reader.fieldnames or []
        protein_column = _detect_column(header, ID_COLUMN_CANDIDATES, id_column)
        detected_ec_column = _detect_column(header, EC_COLUMN_CANDIDATES, ec_column)
        for row in reader:
            protein_id = row.get(protein_column, "").strip()
            if not protein_id:
                continue
            for ec_label in parse_ec_numbers(row.get(detected_ec_column)):
                rows.append((protein_id, ec_label))
    return rows


class RaggedSLEECLogitDataset(Dataset):
    """Load precomputed residue-level SLEEC logits from ragged HDF5."""

    def __init__(
        self,
        file_path: str | Path,
        dtype: torch.dtype = torch.float32,
        max_tokens: int | None = None,
        truncation: str = "ends_center",
    ) -> None:
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"SLEEC logit HDF5 file not found: {file_path}")
        self.file_path = str(file_path_obj)
        self.dtype = dtype
        self.max_tokens = max_tokens
        self.truncation = truncation
        self.file: h5py.File | None = None

        with h5py.File(self.file_path, "r") as h5_file:
            for dataset_name in ("ids", "offsets"):
                if dataset_name not in h5_file:
                    raise KeyError(
                        f"Required dataset '{dataset_name}' not found in {self.file_path}; "
                        f"available datasets: {list(h5_file.keys())}"
                    )
            value_key = None
            for candidate in ("sleec_logits", "logits", "vectors"):
                if candidate in h5_file:
                    value_key = candidate
                    break
            if value_key is None:
                raise KeyError(
                    "SLEEC logit HDF5 must contain one of: sleec_logits, logits, vectors"
                )
            value_shape = h5_file[value_key].shape
            if len(value_shape) not in (1, 2):
                raise ValueError(f"Logit dataset must be rank-1 or rank-2, got {value_shape}")
            if len(value_shape) == 2 and value_shape[1] != 1:
                raise ValueError(
                    f"Rank-2 logit dataset must have second dim 1, got {value_shape}"
                )
            ids_data = h5_file["ids"][:]
            self.keys = [_decode_hdf5_id(value) for value in ids_data]
            offsets_tensor = torch.as_tensor(h5_file["offsets"][:], dtype=torch.long)
            if offsets_tensor.ndim != 1:
                raise ValueError("'offsets' must be rank-1")
            if len(offsets_tensor) != len(self.keys) + 1:
                raise ValueError("'offsets' length must equal len(ids)+1")
            if int(offsets_tensor[0].item()) != 0:
                raise ValueError("'offsets' must start at 0")
            if int(offsets_tensor[-1].item()) != int(value_shape[0]):
                raise ValueError("'offsets' final value must equal number of logits")
            if not torch.all(offsets_tensor[1:] >= offsets_tensor[:-1]):
                raise ValueError("'offsets' must be monotonically non-decreasing")
            self.offsets = offsets_tensor
            self.value_key = value_key
            self.key_to_idx = {key: idx for idx, key in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def _ensure_file(self) -> h5py.File:
        if self.file is None:
            self.file = h5py.File(self.file_path, "r")
        return self.file

    def __getitem__(self, key: str | int) -> torch.Tensor:
        if isinstance(key, int):
            if key < 0 or key >= len(self.keys):
                raise IndexError(f"Index {key} is out of bounds for dataset of length {len(self)}")
            idx = key
        else:
            idx = self.key_to_idx[key]

        start = int(self.offsets[idx].item())
        end = int(self.offsets[idx + 1].item())
        h5_file = self._ensure_file()
        logits = torch.from_numpy(h5_file[self.value_key][start:end]).to(dtype=self.dtype)
        if logits.ndim == 2:
            logits = logits.squeeze(-1)
        logits = truncate_residue_embeddings(
            logits.unsqueeze(-1),
            max_tokens=self.max_tokens,
            strategy=self.truncation,
        ).squeeze(-1)
        return logits

    def __del__(self):
        if hasattr(self, "file") and self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass


class HyperbolicEnzymeResidueDataset(Dataset):
    """EC-labeled enzymes backed by frozen residue embeddings and SLEEC logits."""

    def __init__(
        self,
        residue_dataset: ResidueEmbedDataset,
        rows: list[tuple[str, str]],
        sleec_logits_dataset: RaggedSLEECLogitDataset | None = None,
    ) -> None:
        self.residue_dataset = residue_dataset
        self.sleec_logits_dataset = sleec_logits_dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        enzyme_id, ec_label = self.rows[idx]
        residue_embeddings = self.residue_dataset[enzyme_id]["residue_embeddings"]
        sample: dict[str, Any] = {
            "enzyme_id": enzyme_id,
            "residue_embeddings": residue_embeddings,
            "ec_label": ec_label,
        }
        if self.sleec_logits_dataset is not None:
            sleec_logits = self.sleec_logits_dataset[enzyme_id]
            if sleec_logits.shape[0] != residue_embeddings.shape[0]:
                raise ValueError(
                    f"SLEEC logit length mismatch for {enzyme_id}: "
                    f"logits={sleec_logits.shape[0]}, residues={residue_embeddings.shape[0]}"
                )
            sample["sleec_logits"] = sleec_logits
        return sample


def hyperbolic_enzyme_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    residue_tensors = [item["residue_embeddings"] for item in batch]
    residue_dim = int(residue_tensors[0].shape[-1])
    max_len = max(int(tensor.shape[0]) for tensor in residue_tensors)
    if max_len <= 0:
        raise ValueError("Every enzyme must have at least one residue")

    padded_residues = residue_tensors[0].new_zeros((len(batch), max_len, residue_dim))
    residue_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    has_logits = "sleec_logits" in batch[0]
    padded_logits = (
        torch.zeros((len(batch), max_len), dtype=torch.float32) if has_logits else None
    )

    for row_idx, item in enumerate(batch):
        residues = item["residue_embeddings"]
        if residues.ndim != 2 or residues.shape[-1] != residue_dim:
            raise ValueError("All residue embeddings must have shape [length, residue_dim]")
        length = int(residues.shape[0])
        padded_residues[row_idx, :length] = residues
        residue_mask[row_idx, :length] = True
        if has_logits:
            logits = item["sleec_logits"]
            if logits.shape[0] != length:
                raise ValueError("SLEEC logits and residue embeddings must have equal length")
            padded_logits[row_idx, :length] = logits

    ec_labels = [str(item["ec_label"]) for item in batch]
    output = {
        "enzyme_id": [str(item["enzyme_id"]) for item in batch],
        "residue_embeddings": padded_residues,
        "residue_mask": residue_mask,
        "ec_labels": ec_labels,
        "ec_depth_matrix": compute_ec_shared_depth(ec_labels),
    }
    if padded_logits is not None:
        output["sleec_logits"] = padded_logits
    return output


def build_training_dataset(
    residue_embeddings_path: str | Path,
    ec_labels_path: str | Path,
    input_dim: int,
    *,
    sleec_logits_path: str | Path | None = None,
    id_column: str | None = None,
    ec_column: str | None = None,
    max_protein_tokens: int | None = None,
    protein_truncation: str = "ends_center",
) -> tuple[HyperbolicEnzymeResidueDataset, dict[str, Any]]:
    residue_dataset = ResidueEmbedDataset(
        file_path=str(residue_embeddings_path),
        in_memory=False,
        dtype=torch.float32,
        max_tokens=max_protein_tokens,
        truncation=protein_truncation,
    )
    if residue_dataset.vec_dim != input_dim:
        raise ValueError(
            f"Residue embedding dim mismatch: expected {input_dim}, got {residue_dataset.vec_dim}"
        )

    sleec_logits_dataset = None
    available_ids = set(residue_dataset.keys)
    if sleec_logits_path:
        sleec_logits_dataset = RaggedSLEECLogitDataset(
            sleec_logits_path,
            dtype=torch.float32,
            max_tokens=max_protein_tokens,
            truncation=protein_truncation,
        )
        available_ids &= set(sleec_logits_dataset.keys)

    raw_rows = load_ec_rows(ec_labels_path, id_column=id_column, ec_column=ec_column)
    matched_rows = [(protein_id, ec_label) for protein_id, ec_label in raw_rows if protein_id in available_ids]
    if not matched_rows:
        raise ValueError(
            "No complete EC-labeled enzymes matched the residue embedding/logit files. "
            "Check protein IDs, EC label source, and paths."
        )

    ec_labels = sorted({ec_label for _, ec_label in matched_rows})
    complete_ec4_labels = sorted(
        {ec_label for ec_label in ec_labels if known_ec_depth(ec_label) == 4}
    )
    depth_counts = {
        str(depth): sum(1 for _, ec_label in matched_rows if known_ec_depth(ec_label) == depth)
        for depth in range(1, 5)
    }
    metadata = {
        "raw_ec_rows": len(raw_rows),
        "dataset_samples": len(matched_rows),
        "unique_enzyme_ids": len({protein_id for protein_id, _ in matched_rows}),
        "unique_ec_labels": len(ec_labels),
        "unique_complete_ec4_labels": len(complete_ec4_labels),
        "ec_known_depth_sample_counts": depth_counts,
        "has_precomputed_sleec_logits": sleec_logits_dataset is not None,
        "residue_embedding_dim": int(residue_dataset.vec_dim),
    }
    dataset = HyperbolicEnzymeResidueDataset(
        residue_dataset=residue_dataset,
        rows=matched_rows,
        sleec_logits_dataset=sleec_logits_dataset,
    )
    return dataset, metadata


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            flattened.update(value)
        else:
            flattened[key] = value
    return flattened


def load_config_file(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return _flatten_config(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-path", "--ec-labels-path", dest="ec_labels_path", default=None)
    parser.add_argument("--residue-embeddings-path", default=None)
    parser.add_argument("--sleec-logits-path", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--ec-column", default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--hyp-dim", type=int, default=None)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument(
        "--p0-sleec-threshold",
        "--sleec-threshold",
        dest="p0_sleec_threshold",
        type=float,
        default=None,
    )
    parser.add_argument("--base-margin", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tangent-clip", type=float, default=None)
    parser.add_argument("--tangent-clip-mode", choices=("hard", "soft"), default=None)
    parser.add_argument(
        "--projector-input-normalization",
        choices=("none", "layernorm"),
        default=None,
    )
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--max-triplets-per-anchor", type=int, default=None)
    parser.add_argument("--alpha-radial", type=float, default=None)
    parser.add_argument("--alpha-radius-target", type=float, default=None)
    parser.add_argument("--radius-target", type=float, default=None)
    parser.add_argument("--use-centroid-cone", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha-centroid-cone", type=float, default=None)
    parser.add_argument("--use-ec-entailment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--alpha-ec-entailment", type=float, default=None)
    parser.add_argument("--ec-entailment-warmup-epochs", type=int, default=None)
    parser.add_argument("--ec-entailment-eta", type=float, default=None)
    parser.add_argument("--ec-entailment-min-radius", type=float, default=None)
    parser.add_argument("--ec-entailment-min-group-size", type=int, default=None)
    parser.add_argument("--K", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--cone-loss-power", type=float, default=None)
    parser.add_argument("--centroid-min-group-size", type=int, default=None)
    parser.add_argument("--sleec-checkpoint-path", default=None)
    parser.add_argument("--sleec-scorer-hidden-dim", type=int, default=None)
    parser.add_argument("--attention-bias", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--initial-prior-strength", type=float, default=None)
    parser.add_argument("--train-prior-strength", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-sleec", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-protein-tokens", type=int, default=None)
    parser.add_argument("--protein-truncation", default=None)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument("--wandb-log-model", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb-dir", default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    file_config = load_config_file(args.config)
    if "sleec_threshold" in file_config and "p0_sleec_threshold" not in file_config:
        file_config["p0_sleec_threshold"] = file_config["sleec_threshold"]
    config.update(file_config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            config[key] = value
    return config


def _device_from_config(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _setup_runtime(config: dict[str, Any]) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        local_rank = 0
        rank = 0
        device = _device_from_config(str(config["device"]))
    return {
        "distributed": distributed,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }


def _cleanup_runtime(runtime: dict[str, Any]) -> None:
    if runtime["distributed"] and dist.is_initialized():
        dist.destroy_process_group()


def _is_rank_zero(runtime: dict[str, Any]) -> bool:
    return int(runtime["rank"]) == 0


def _rank_zero_print(runtime: dict[str, Any], message: str) -> None:
    if _is_rank_zero(runtime):
        print(message)


def _module_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(module, DDP):
        return module.module.state_dict()
    return module.state_dict()


def _load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _load_module_state(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    target = module.module if isinstance(module, DDP) else module
    target.load_state_dict(state_dict)


def _move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _reduce_log_dict(
    logs: dict[str, torch.Tensor],
    runtime: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not runtime["distributed"]:
        return {key: value.detach().cpu() for key, value in logs.items()}
    keys = list(logs.keys())
    values = torch.stack(
        [logs[key].detach().to(device=runtime["device"], dtype=torch.float32) for key in keys]
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(runtime["world_size"])
    return {key: value.detach().cpu() for key, value in zip(keys, values)}


def _reduce_epoch_mean(loss_sum: float, steps: int, runtime: dict[str, Any]) -> float:
    values = torch.tensor(
        [loss_sum, float(steps)],
        device=runtime["device"],
        dtype=torch.float64,
    )
    if runtime["distributed"]:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_steps = max(float(values[1].item()), 1.0)
    return float(values[0].item() / total_steps)


def _resolve_wandb_settings(config: dict[str, Any]) -> dict[str, Any]:
    wandb_config = config.get("wandb", False)
    if isinstance(wandb_config, dict):
        requested = bool(wandb_config.get("enabled", False))
        project = (
            config.get("wandb_project")
            or config.get("project")
            or wandb_config.get("project")
        )
        entity = (
            config.get("wandb_entity")
            or config.get("entity")
            or wandb_config.get("entity")
        )
        run_name = (
            config.get("wandb_run_name")
            or config.get("run_name")
            or wandb_config.get("run_name")
        )
        mode = (
            config.get("wandb_mode")
            or config.get("mode")
            or wandb_config.get("mode", "online")
        )
        tags = config.get("wandb_tags")
        if tags is None:
            tags = config.get("tags")
        if tags is None:
            tags = wandb_config.get("tags")
        log_model = bool(
            config.get("wandb_log_model")
            or config.get("log_model")
            or wandb_config.get("log_model", False)
        )
    else:
        requested = bool(wandb_config or config.get("enabled", False))
        project = (
            config.get("wandb_project")
            or config.get("project")
            or config.get("wandb_project_name")
        )
        entity = (
            config.get("wandb_entity")
            or config.get("entity")
            or config.get("wandb_username")
        )
        run_name = (
            config.get("wandb_run_name")
            or config.get("run_name")
        )
        mode = (
            config.get("wandb_mode")
            or config.get("mode")
            or "online"
        )
        tags = config.get("wandb_tags")
        if tags is None:
            tags = config.get("tags")
        log_model = bool(
            config.get("wandb_log_model")
            or config.get("log_model")
            or False
        )
    if isinstance(tags, str):
        tags = [item.strip() for item in tags.split(",") if item.strip()]
    return {
        "enabled": requested and mode != "disabled",
        "project": project or "horizyn-hyperbolic-enzyme",
        "entity": entity,
        "run_name": run_name,
        "mode": mode,
        "tags": tags,
        "log_model": log_model,
        "dir": config.get("wandb_dir", "logs/wandb"),
    }


def _init_wandb(
    config: dict[str, Any],
    runtime: dict[str, Any],
    metadata: dict[str, Any],
) -> Any | None:
    settings = _resolve_wandb_settings(config)
    if not settings["enabled"] or not _is_rank_zero(runtime):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging was requested but wandb is not installed") from exc

    Path(settings["dir"]).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        name=settings["run_name"],
        mode=settings["mode"],
        tags=settings["tags"],
        dir=settings["dir"],
        config={
            "stage": "hyperbolic_enzyme_unimodal_lorentz",
            "config": config,
            "metadata": metadata,
            "distributed": {
                "world_size": runtime["world_size"],
                "rank": runtime["rank"],
                "local_rank": runtime["local_rank"],
            },
        },
    )
    wandb.define_metric("step")
    wandb.define_metric("train/*", step_metric="step")
    wandb.define_metric("epoch/*", step_metric="step")
    return run


def _log_wandb(run: Any | None, payload: dict[str, float], *, step: int) -> None:
    if run is not None:
        run.log({"step": step, **payload})


def _log_checkpoint_artifact(
    run: Any | None,
    checkpoint_paths: list[Path],
    *,
    artifact_name: str,
) -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(artifact_name, type="model")
    added = False
    for path in checkpoint_paths:
        if path.exists():
            artifact.add_file(str(path))
            added = True
    if added:
        run.log_artifact(artifact)


def _format_logs(logs: dict[str, torch.Tensor]) -> str:
    keys = [
        "loss_total",
        "loss_rank",
        "loss_radial",
        "loss_radius_target",
        "loss_centroid_cone",
        "loss_ec_entailment",
        "alpha_ec_entailment_effective",
        "ec_entailment_violation_rate",
        "num_ec_entailment_terms",
        "num_triplets",
        "mean_positive_distance",
        "mean_negative_distance",
        "mean_radius_enzyme",
        "mean_attention_entropy",
        "mean_prior_strength",
        "max_lorentz_constraint_error",
    ]
    return " ".join(
        f"{key}={float(logs[key].item()):.4f}" for key in keys if key in logs
    )


def _save_checkpoint(
    path: Path,
    *,
    attention_pooler: torch.nn.Module,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_epoch_loss: float,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attention_state = _module_state_dict(attention_pooler)
    projector_state = _module_state_dict(projector)
    torch.save(
        {
            "attention_pooler_state_dict": attention_state,
            "sleec_guided_attention_pool_state_dict": attention_state,
            "hyperbolic_projector_state_dict": projector_state,
            "projector_state_dict": projector_state,
            "config": config,
            "curvature": float(config["curvature"]),
            "hyp_dim": int(config["hyp_dim"]),
            "input_dim": int(config["input_dim"]),
            "p0_sleec_threshold": float(config["p0_sleec_threshold"]),
            "normalization_stats": {},
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_epoch_loss": float(best_epoch_loss),
            "metadata": metadata,
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        },
        path,
    )


def _compute_radius_stats(z_hyp: torch.Tensor, curvature: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    origin = lorentz_origin(
        z_hyp.shape[-1] - 1,
        kappa=curvature,
        dtype=z_hyp.dtype,
        device=z_hyp.device,
    )
    radii = lorentz_distance(
        z_hyp,
        origin.unsqueeze(0).expand_as(z_hyp),
        kappa=curvature,
        eps=eps,
    )
    return radii.mean(), radii.std(unbiased=False)


def train(config: dict[str, Any]) -> Path:
    if not bool(config.get("freeze_sleec", True)):
        raise ValueError("SLEEC scorer training is not allowed in this pretraining script")
    if not config.get("sleec_logits_path") and not config.get("sleec_checkpoint_path"):
        raise ValueError(
            "Either sleec_logits_path or sleec_checkpoint_path is required to provide "
            "the frozen SLEEC residue prior"
        )

    runtime = _setup_runtime(config)
    wandb_run = None
    checkpoint_path = Path(config["output_checkpoint"])
    best_checkpoint_path = checkpoint_path.with_name("best.ckpt")
    best_epoch_loss = float("inf")
    metadata: dict[str, Any] = {}

    try:
        random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config["seed"]))
        device = runtime["device"]
        resume_checkpoint = (
            _load_checkpoint(config["resume_checkpoint"], device=torch.device("cpu"))
            if config.get("resume_checkpoint")
            else None
        )
        resume_epoch_offset = 0
        if resume_checkpoint is not None:
            resume_epoch_offset = int(resume_checkpoint.get("epoch", 0))
            if int(resume_checkpoint.get("input_dim", config["input_dim"])) != int(config["input_dim"]):
                raise ValueError("Resume checkpoint input_dim does not match the current config")
            if int(resume_checkpoint.get("hyp_dim", config["hyp_dim"])) != int(config["hyp_dim"]):
                raise ValueError("Resume checkpoint hyp_dim does not match the current config")
            if float(resume_checkpoint.get("curvature", config["curvature"])) != float(config["curvature"]):
                raise ValueError("Resume checkpoint curvature does not match the current config")

        dataset, metadata = build_training_dataset(
            residue_embeddings_path=config["residue_embeddings_path"],
            ec_labels_path=config["ec_labels_path"],
            input_dim=int(config["input_dim"]),
            sleec_logits_path=config.get("sleec_logits_path"),
            id_column=config.get("id_column"),
            ec_column=config.get("ec_column"),
            max_protein_tokens=config.get("max_protein_tokens"),
            protein_truncation=str(config.get("protein_truncation", "ends_center")),
        )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=int(runtime["world_size"]),
                rank=int(runtime["rank"]),
                shuffle=True,
                drop_last=False,
            )
            if runtime["distributed"]
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
            collate_fn=hyperbolic_enzyme_collate_fn,
        )
        if len(loader) == 0:
            raise ValueError("No training batches were created")

        global_batch_size = int(config["batch_size"]) * int(runtime["world_size"])
        metadata.update(
            {
                "residue_embeddings_path": str(config["residue_embeddings_path"]),
                "sleec_logits_path": None
                if config.get("sleec_logits_path") is None
                else str(config["sleec_logits_path"]),
                "sleec_checkpoint_path": None
                if config.get("sleec_checkpoint_path") is None
                else str(config["sleec_checkpoint_path"]),
                "per_gpu_batch_size": int(config["batch_size"]),
                "global_batch_size": global_batch_size,
                "world_size": int(runtime["world_size"]),
            }
        )

        wandb_run = _init_wandb(config, runtime, metadata)

        attention_pooler: torch.nn.Module = SLEECGuidedAttentionPool(
            input_dim=int(config["input_dim"]),
            scorer_hidden_dim=int(config["sleec_scorer_hidden_dim"]),
            p0=float(config["p0_sleec_threshold"]),
            sleec_checkpoint_path=config.get("sleec_checkpoint_path"),
            freeze_sleec=True,
            attention_bias=bool(config["attention_bias"]),
            initial_prior_strength=float(config["initial_prior_strength"]),
            train_prior_strength=bool(config["train_prior_strength"]),
            eps=float(config["eps"]),
        ).to(device)
        projector: torch.nn.Module = LorentzEnzymeProjector(
            input_dim=int(config["input_dim"]),
            hyp_dim=int(config["hyp_dim"]),
            curvature=float(config["curvature"]),
            tangent_clip=config["tangent_clip"],
            tangent_clip_mode=str(config["tangent_clip_mode"]),
            input_normalization=str(config["projector_input_normalization"]),
            eps=float(config["eps"]),
        ).to(device)
        if resume_checkpoint is not None:
            _load_module_state(
                attention_pooler,
                resume_checkpoint["attention_pooler_state_dict"],
            )
            _load_module_state(
                projector,
                resume_checkpoint["hyperbolic_projector_state_dict"],
            )
            best_epoch_loss = float(resume_checkpoint.get("best_epoch_loss", best_epoch_loss))
        if runtime["distributed"]:
            ddp_kwargs: dict[str, Any] = {}
            if device.type == "cuda":
                ddp_kwargs = {
                    "device_ids": [int(runtime["local_rank"])],
                    "output_device": int(runtime["local_rank"]),
                }
            attention_pooler = DDP(attention_pooler, **ddp_kwargs)
            projector = DDP(projector, **ddp_kwargs)

        ranking_loss = HierarchyRankingLoss(
            base_margin=float(config["base_margin"]),
            curvature=float(config["curvature"]),
            max_triplets_per_anchor=int(config["max_triplets_per_anchor"]),
            eps=float(config["eps"]),
        )
        centroid_cone_loss = NonParametricCentroidConeLoss(
            curvature=float(config["curvature"]),
            K=float(config["K"]),
            eta=float(config["eta"]),
            cone_loss_power=float(config["cone_loss_power"]),
            min_group_size=int(config["centroid_min_group_size"]),
            eps=float(config["eps"]),
        )
        ec_entailment_loss = ECPrefixEntailmentLoss(
            curvature=float(config["curvature"]),
            min_radius=float(config["ec_entailment_min_radius"]),
            eta=float(config["ec_entailment_eta"]),
            min_group_size=int(config["ec_entailment_min_group_size"]),
            eps=float(config["eps"]),
        )
        trainable_parameters = [
            parameter
            for module in (attention_pooler, projector)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        if resume_checkpoint is not None and resume_checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)

        _rank_zero_print(runtime, f"Loaded unimodal enzyme dataset: {len(dataset)} samples")
        _rank_zero_print(runtime, f"Unique EC labels: {metadata['unique_ec_labels']}")
        _rank_zero_print(
            runtime,
            f"Unique complete EC4 labels: {metadata['unique_complete_ec4_labels']}",
        )
        _rank_zero_print(
            runtime,
            f"Known-depth sample counts: {metadata['ec_known_depth_sample_counts']}",
        )
        _rank_zero_print(runtime, f"Input dim: {config['input_dim']}")
        _rank_zero_print(runtime, f"Lorentz hyp dim: {config['hyp_dim']}")
        _rank_zero_print(runtime, f"Curvature kappa: {config['curvature']}")
        _rank_zero_print(runtime, f"SLEEC threshold p0: {config['p0_sleec_threshold']}")
        _rank_zero_print(runtime, f"Device: {device}")
        _rank_zero_print(runtime, f"World size: {runtime['world_size']}")
        _rank_zero_print(runtime, f"Per-GPU batch size: {config['batch_size']}")
        _rank_zero_print(runtime, f"Global batch size: {global_batch_size}")
        if resume_checkpoint is not None:
            _rank_zero_print(
                runtime,
                "Resuming from "
                f"{config['resume_checkpoint']} at epoch={resume_epoch_offset} "
                f"global_step={int(resume_checkpoint.get('global_step', 0))}",
            )

        global_step = (
            int(resume_checkpoint.get("global_step", 0))
            if resume_checkpoint is not None
            else 0
        )
        max_steps = config.get("max_steps")
        max_steps = None if max_steps is None else int(max_steps)
        requested_epochs = int(config["epochs"])
        last_completed_epoch = resume_epoch_offset
        for epoch in range(resume_epoch_offset + 1, requested_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in loader:
                global_step += 1
                epoch_steps += 1

                residue_embeddings = batch["residue_embeddings"].to(
                    device=device,
                    non_blocking=True,
                )
                residue_mask = batch["residue_mask"].to(device=device, non_blocking=True)
                sleec_logits = batch.get("sleec_logits")
                if sleec_logits is not None:
                    sleec_logits = sleec_logits.to(device=device, non_blocking=True)
                ec_depth_matrix = batch["ec_depth_matrix"].to(device=device, non_blocking=True)
                ec_labels = batch["ec_labels"]

                pooled, attention_details = attention_pooler(
                    residue_embeddings,
                    residue_mask=residue_mask,
                    sleec_logits=sleec_logits,
                    return_details=True,
                )
                z_hyp, _ = projector(pooled)
                rank_output = ranking_loss(z_hyp, ec_depth_matrix)
                loss_rank = rank_output.loss
                loss_radial = (
                    same_ec4_radius_loss(
                        z_hyp,
                        ec_depth_matrix,
                        curvature=float(config["curvature"]),
                        eps=float(config["eps"]),
                    )
                    if float(config["alpha_radial"]) > 0.0
                    else z_hyp.sum() * 0.0
                )
                loss_radius_target = (
                    radius_target_loss(
                        z_hyp,
                        radius_target=float(config["radius_target"]),
                        curvature=float(config["curvature"]),
                        eps=float(config["eps"]),
                    )
                    if float(config["alpha_radius_target"]) > 0.0
                    else z_hyp.sum() * 0.0
                )
                if bool(config["use_centroid_cone"]) and float(config["alpha_centroid_cone"]) > 0.0:
                    loss_centroid_cone, centroid_logs = centroid_cone_loss(z_hyp, ec_labels)
                else:
                    loss_centroid_cone = z_hyp.sum() * 0.0
                    centroid_logs = {
                        "loss_centroid_cone": loss_centroid_cone.detach(),
                        "num_centroid_cone_terms": torch.zeros(
                            (),
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                    }
                if bool(config["use_ec_entailment"]):
                    loss_ec_entailment, ec_entailment_logs = ec_entailment_loss(z_hyp, ec_labels)
                else:
                    loss_ec_entailment = z_hyp.sum() * 0.0
                    ec_entailment_logs = {
                        "loss_ec_entailment": loss_ec_entailment.detach(),
                        "num_ec_entailment_terms": torch.zeros(
                            (),
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                        "ec_entailment_violation_rate": torch.zeros(
                            (),
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                        "mean_ec_entailment_angle": torch.zeros(
                            (),
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                        "mean_ec_entailment_aperture": torch.zeros(
                            (),
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                    }
                ec_entailment_warmup_epochs = int(config["ec_entailment_warmup_epochs"])
                alpha_ec_entailment_effective = (
                    float(config["alpha_ec_entailment"])
                    if bool(config["use_ec_entailment"]) and epoch > ec_entailment_warmup_epochs
                    else 0.0
                )
                loss_total = (
                    loss_rank
                    + float(config["alpha_radial"]) * loss_radial
                    + float(config["alpha_radius_target"]) * loss_radius_target
                    + float(config["alpha_centroid_cone"]) * loss_centroid_cone
                    + alpha_ec_entailment_effective * loss_ec_entailment
                )

                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                optimizer.step()

                with torch.no_grad():
                    valid_count = residue_mask.sum().clamp_min(1)
                    detail_logits = attention_details["sleec_logits"]
                    mean_sleec_logit = (
                        detail_logits.masked_fill(~residue_mask, 0.0).sum() / valid_count
                    )
                    mean_radius, std_radius = _compute_radius_stats(
                        z_hyp,
                        curvature=float(config["curvature"]),
                        eps=float(config["eps"]),
                    )
                    logs = {
                        "loss_total": loss_total.detach(),
                        "loss_rank": loss_rank.detach(),
                        "loss_radial": loss_radial.detach(),
                        "loss_radius_target": loss_radius_target.detach(),
                        "loss_centroid_cone": loss_centroid_cone.detach(),
                        "loss_ec_entailment": loss_ec_entailment.detach(),
                        "alpha_ec_entailment_effective": torch.tensor(
                            alpha_ec_entailment_effective,
                            dtype=z_hyp.dtype,
                            device=z_hyp.device,
                        ),
                        "mean_radius_enzyme": mean_radius.detach(),
                        "std_radius_enzyme": std_radius.detach(),
                        "mean_attention_entropy": mean_attention_entropy(
                            attention_details["attention_weights"],
                            eps=float(config["eps"]),
                        ).detach(),
                        "mean_prior_strength": attention_details["prior_strength"].detach(),
                        "mean_sleec_logit": mean_sleec_logit.detach(),
                        "mean_attention_mass_above_sleec_threshold": attention_mass_above_threshold(
                            attention_details["attention_weights"],
                            detail_logits,
                            threshold=float(config["p0_sleec_threshold"]),
                            residue_mask=residue_mask,
                        ).detach(),
                        "max_lorentz_constraint_error": lorentz_constraint_error(
                            z_hyp,
                            kappa=float(config["curvature"]),
                        ).max().detach(),
                    }
                    logs.update(rank_output.logs)
                    logs.update(centroid_logs)
                    logs.update(ec_entailment_logs)

                epoch_loss += float(loss_total.item())
                reduced_logs = _reduce_log_dict(logs, runtime)
                if global_step == 1 or global_step % int(config["log_every"]) == 0:
                    _rank_zero_print(
                        runtime,
                        f"epoch={epoch} step={global_step} {_format_logs(reduced_logs)}",
                    )
                    _log_wandb(
                        wandb_run,
                        {f"train/{key}": float(value.item()) for key, value in reduced_logs.items()},
                        step=global_step,
                    )

                if max_steps is not None and global_step >= max_steps:
                    break

            mean_epoch_loss = _reduce_epoch_mean(epoch_loss, epoch_steps, runtime)
            last_completed_epoch = epoch
            _rank_zero_print(runtime, f"epoch={epoch} mean_loss={mean_epoch_loss:.4f}")
            _log_wandb(
                wandb_run,
                {"epoch/mean_loss": mean_epoch_loss, "epoch/index": float(epoch)},
                step=global_step,
            )
            if _is_rank_zero(runtime):
                if mean_epoch_loss < best_epoch_loss:
                    best_epoch_loss = mean_epoch_loss
                    _save_checkpoint(
                        best_checkpoint_path,
                        attention_pooler=attention_pooler,
                        projector=projector,
                        optimizer=optimizer,
                        config=config,
                        epoch=epoch,
                        global_step=global_step,
                        best_epoch_loss=best_epoch_loss,
                        metadata=metadata,
                    )
                    print(f"Saved best checkpoint: {best_checkpoint_path}")
                _save_checkpoint(
                    checkpoint_path,
                    attention_pooler=attention_pooler,
                    projector=projector,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    global_step=global_step,
                    best_epoch_loss=best_epoch_loss,
                    metadata=metadata,
                )
                print(f"Saved last checkpoint: {checkpoint_path}")

            if max_steps is not None and global_step >= max_steps:
                break

        if _is_rank_zero(runtime):
            settings = _resolve_wandb_settings(config)
            if settings["log_model"]:
                artifact_name = settings["run_name"] or "hyperbolic-enzyme-pretrain"
                _log_checkpoint_artifact(
                    wandb_run,
                    [checkpoint_path, best_checkpoint_path],
                    artifact_name=f"{artifact_name}-checkpoints",
                )
            completion_summary = {
                "last_checkpoint": str(checkpoint_path),
                "best_checkpoint": str(best_checkpoint_path),
                "completed_epochs": int(last_completed_epoch),
                "requested_epochs": requested_epochs,
                "global_step": int(global_step),
            }
            checkpoint_path.with_name("pretrain_summary.json").write_text(
                json.dumps(completion_summary, indent=2) + "\n",
                encoding="utf-8",
            )
            _rank_zero_print(runtime, f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _cleanup_runtime(runtime)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = resolve_config(args)
    train(config)


if __name__ == "__main__":
    main()
