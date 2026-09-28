#!/usr/bin/env python3
"""Pretrain a reaction fingerprint encoder with EC hierarchy supervision."""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.config import DotDict  # noqa: E402
from horizyn.hyperbolic_enzyme import (  # noqa: E402
    ECPrefixEntailmentLoss,
    HierarchyRankingLoss,
    LorentzEnzymeProjector,
    NonParametricCentroidConeLoss,
    compute_ec_shared_depth,
    format_ec_prefixes,
    known_ec_depth,
    parse_ec_prefixes,
    radius_target_loss,
    same_ec4_radius_loss,
)
from horizyn.model import MLP  # noqa: E402
from horizyn.reaction_features import build_reaction_feature_dataset  # noqa: E402


DEFAULT_CONFIG: dict[str, Any] = {
    "train_pairs_path": "data/sota/train_pairs.csv",
    "train_reactions_path": "data/sota/train_rxns.csv",
    "ec_labels_path": "data/sota/uniprot_all_ec_labels.csv",
    "pair_reaction_column": None,
    "pair_protein_column": None,
    "id_column": None,
    "ec_column": None,
    "reaction_id_column": "reaction_id",
    "reaction_smiles_column": "reaction_smiles",
    "bidirectional": True,
    "rdkit_fp_dim": 1024,
    "drfp_dim": 1024,
    "standardize_reactions": True,
    "standardize_hypervalent": True,
    "standardize_remove_hs": True,
    "standardize_kekulize": False,
    "standardize_uncharge": True,
    "standardize_metals": True,
    "reaction_encoder_dims": [2048, 4096, 4096, 512],
    "hyp_dim": 512,
    "curvature": 0.25,
    "tangent_clip": 5.0,
    "tangent_clip_mode": "hard",
    "projector_input_normalization": "none",
    "eps": 1e-6,
    "base_margin": 0.05,
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
    "batch_size": 512,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "device": "auto",
    "seed": 42,
    "num_workers": 0,
    "log_every": 50,
    "max_steps": None,
    "output_checkpoint": "checkpoints/hyperbolic_ec/hyperbolic_reaction_pretrain/last.ckpt",
    "resume_checkpoint": None,
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
PAIR_REACTION_COLUMN_CANDIDATES = ("reaction_id", "query_id", "rxn_id")
PAIR_PROTEIN_COLUMN_CANDIDATES = (
    "protein_id",
    "target_id",
    "enzyme_id",
    "uniprot_accession",
)
EC_TOKEN_PATTERN = re.compile(r"[1-7](?:\.(?:\d+|-)){0,3}")


def parse_ec_numbers(value: str | None) -> list[str]:
    """Parse complete or prefix-only EC labels from a CSV cell."""
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
        try:
            normalized = format_ec_prefixes(parse_ec_prefixes(match.group(0)))
        except ValueError:
            continue
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


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


def load_ec_label_map(
    ec_labels_path: str | Path,
    id_column: str | None = None,
    ec_column: str | None = None,
) -> dict[str, set[str]]:
    """Load protein ID -> normalized EC label set."""
    labels_by_protein: dict[str, set[str]] = {}
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
            labels = parse_ec_numbers(row.get(detected_ec_column))
            if labels:
                labels_by_protein.setdefault(protein_id, set()).update(labels)
    return labels_by_protein


def load_reaction_ec_label_sets(
    train_pairs_path: str | Path,
    ec_labels_path: str | Path,
    *,
    pair_reaction_column: str | None = None,
    pair_protein_column: str | None = None,
    id_column: str | None = None,
    ec_column: str | None = None,
) -> dict[str, set[str]]:
    """Derive reaction ID -> EC label set from train reaction-protein pairs."""
    protein_to_ec = load_ec_label_map(
        ec_labels_path,
        id_column=id_column,
        ec_column=ec_column,
    )
    reaction_to_ec: dict[str, set[str]] = {}
    with open(train_pairs_path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        header = reader.fieldnames or []
        reaction_column = _detect_column(
            header,
            PAIR_REACTION_COLUMN_CANDIDATES,
            pair_reaction_column,
        )
        protein_column = _detect_column(
            header,
            PAIR_PROTEIN_COLUMN_CANDIDATES,
            pair_protein_column,
        )
        for row in reader:
            reaction_id = row.get(reaction_column, "").strip()
            protein_id = row.get(protein_column, "").strip()
            if not reaction_id or not protein_id:
                continue
            labels = protein_to_ec.get(protein_id)
            if labels:
                reaction_to_ec.setdefault(reaction_id, set()).update(labels)
    return reaction_to_ec


def base_reaction_id(reaction_feature_key: str) -> str:
    """Map bidirectional keys like ``rxn_f``/``rxn_r`` back to ``rxn``."""
    if reaction_feature_key.endswith(("_f", "_r")):
        return reaction_feature_key[:-2]
    return reaction_feature_key


class ReactionECFingerprintDataset(Dataset):
    """Reaction fingerprint rows with one EC label per training sample."""

    def __init__(
        self,
        reaction_features,
        reaction_ec_labels: dict[str, set[str]],
    ) -> None:
        self.reaction_features = reaction_features
        self.reaction_ec_labels = reaction_ec_labels
        rows: list[tuple[str, str]] = []
        for feature_key in reaction_features.keys:
            labels = sorted(reaction_ec_labels.get(base_reaction_id(feature_key), set()))
            for ec_label in labels:
                rows.append((feature_key, ec_label))
        if not rows:
            raise ValueError(
                "No EC-labeled reaction samples were created. Check train pairs, "
                "reaction IDs, protein IDs, EC labels, and reaction feature keys."
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        reaction_id, ec_label = self.rows[idx]
        reaction_vec = self.reaction_features[reaction_id]
        if isinstance(reaction_vec, dict):
            raise ValueError(
                "Reaction hyperbolic pretraining currently expects fingerprint tensors. "
                "Use reaction_representation='fingerprint'."
            )
        return {
            "reaction_id": reaction_id,
            "reaction_vec": reaction_vec,
            "ec_label": ec_label,
        }


def reaction_ec_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    reaction_vecs = torch.stack([item["reaction_vec"] for item in batch])
    ec_labels = [str(item["ec_label"]) for item in batch]
    return {
        "reaction_id": [str(item["reaction_id"]) for item in batch],
        "reaction_vec": reaction_vecs,
        "ec_labels": ec_labels,
        "ec_depth_matrix": compute_ec_shared_depth(ec_labels),
    }


def build_reaction_config(config: dict[str, Any]) -> DotDict:
    return DotDict(
        {
            "data": {
                "reaction_representation": "fingerprint",
                "rdkit_fp_dim": int(config["rdkit_fp_dim"]),
                "drfp_dim": int(config["drfp_dim"]),
                "standardize_reactions": bool(config["standardize_reactions"]),
                "standardize_hypervalent": bool(config["standardize_hypervalent"]),
                "standardize_remove_hs": bool(config["standardize_remove_hs"]),
                "standardize_kekulize": bool(config["standardize_kekulize"]),
                "standardize_uncharge": bool(config["standardize_uncharge"]),
                "standardize_metals": bool(config["standardize_metals"]),
            }
        }
    )


def build_training_dataset(config: dict[str, Any]) -> tuple[ReactionECFingerprintDataset, dict[str, Any]]:
    reaction_features = build_reaction_feature_dataset(
        reactions_path=config["train_reactions_path"],
        config=build_reaction_config(config),
        key_column=str(config["reaction_id_column"]),
        smiles_column=str(config["reaction_smiles_column"]),
        bidirectional=bool(config["bidirectional"]),
    )
    reaction_ec_labels = load_reaction_ec_label_sets(
        train_pairs_path=config["train_pairs_path"],
        ec_labels_path=config["ec_labels_path"],
        pair_reaction_column=config.get("pair_reaction_column"),
        pair_protein_column=config.get("pair_protein_column"),
        id_column=config.get("id_column"),
        ec_column=config.get("ec_column"),
    )
    dataset = ReactionECFingerprintDataset(
        reaction_features=reaction_features,
        reaction_ec_labels=reaction_ec_labels,
    )
    labeled_reaction_ids = {
        base_reaction_id(feature_key) for feature_key, _ in dataset.rows
    }
    unique_ec_labels = sorted({ec_label for _, ec_label in dataset.rows})
    metadata = {
        "reaction_feature_count": len(reaction_features),
        "reaction_ec_label_sets": len(reaction_ec_labels),
        "dataset_samples": len(dataset),
        "unique_labeled_reactions": len(labeled_reaction_ids),
        "unique_ec_labels": len(unique_ec_labels),
        "unique_complete_ec4_labels": sum(
            1 for ec_label in unique_ec_labels if known_ec_depth(ec_label) == 4
        ),
        "bidirectional": bool(config["bidirectional"]),
    }
    return dataset, metadata


def build_reaction_encoder(dims: list[int]) -> MLP:
    if len(dims) < 2:
        raise ValueError("reaction_encoder_dims must include at least [input_dim, output_dim]")
    return MLP(
        input_dim=int(dims[0]),
        output_dim=int(dims[-1]),
        num_layers=max(len(dims) - 2, 0),
        widths=[int(value) for value in dims[1:-1]],
        normalise_output=True,
    )


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
    parser.add_argument("--train-pairs-path", default=None)
    parser.add_argument("--train-reactions-path", default=None)
    parser.add_argument("--ec-labels-path", "--data-path", dest="ec_labels_path", default=None)
    parser.add_argument("--pair-reaction-column", default=None)
    parser.add_argument("--pair-protein-column", default=None)
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--ec-column", default=None)
    parser.add_argument("--reaction-id-column", default=None)
    parser.add_argument("--reaction-smiles-column", default=None)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rdkit-fp-dim", type=int, default=None)
    parser.add_argument("--drfp-dim", type=int, default=None)
    parser.add_argument("--standardize-reactions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reaction-encoder-dims", type=int, nargs="+", default=None)
    parser.add_argument("--hyp-dim", type=int, default=None)
    parser.add_argument("--curvature", type=float, default=None)
    parser.add_argument("--tangent-clip", type=float, default=None)
    parser.add_argument("--tangent-clip-mode", choices=("hard", "soft"), default=None)
    parser.add_argument(
        "--projector-input-normalization",
        choices=("none", "layernorm"),
        default=None,
    )
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--base-margin", type=float, default=None)
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
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config.update(load_config_file(args.config))
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


def _format_logs(logs: dict[str, torch.Tensor]) -> str:
    keys = [
        "loss_total",
        "loss_rank",
        "loss_radial",
        "loss_radius_target",
        "loss_centroid_cone",
        "loss_ec_entailment",
        "alpha_ec_entailment_effective",
        "num_triplets",
        "mean_positive_distance",
        "mean_negative_distance",
        "mean_radius_reaction",
        "max_lorentz_constraint_error",
    ]
    return " ".join(
        f"{key}={float(logs[key].item()):.4f}" for key in keys if key in logs
    )


def _save_checkpoint(
    path: Path,
    *,
    reaction_encoder: torch.nn.Module,
    projector: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    best_epoch_loss: float,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reaction_encoder_state = _module_state_dict(reaction_encoder)
    projector_state = _module_state_dict(projector)
    torch.save(
        {
            "reaction_encoder_state_dict": reaction_encoder_state,
            "query_encoder_state_dict": reaction_encoder_state,
            "hyperbolic_projector_state_dict": projector_state,
            "projector_state_dict": projector_state,
            "config": config,
            "curvature": float(config["curvature"]),
            "hyp_dim": int(config["hyp_dim"]),
            "input_dim": int(config["reaction_encoder_dims"][0]),
            "reaction_embedding_dim": int(config["reaction_encoder_dims"][-1]),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_epoch_loss": float(best_epoch_loss),
            "metadata": metadata,
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
        },
        path,
    )


def _compute_radius_stats(z_hyp: torch.Tensor, curvature: float, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    origin = z_hyp.new_zeros(z_hyp.shape[-1])
    origin[0] = 1.0 / (curvature**0.5)
    from horizyn.lorentz import lorentz_distance

    radii = lorentz_distance(
        z_hyp,
        origin.unsqueeze(0).expand_as(z_hyp),
        kappa=curvature,
        eps=eps,
    )
    return radii.mean(), radii.std(unbiased=False)


def train(config: dict[str, Any]) -> Path:
    runtime = _setup_runtime(config)
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
            if int(resume_checkpoint.get("reaction_embedding_dim", config["reaction_encoder_dims"][-1])) != int(
                config["reaction_encoder_dims"][-1]
            ):
                raise ValueError("Resume checkpoint reaction embedding dim does not match config")
            if float(resume_checkpoint.get("curvature", config["curvature"])) != float(config["curvature"]):
                raise ValueError("Resume checkpoint curvature does not match config")

        dataset, metadata = build_training_dataset(config)
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
            collate_fn=reaction_ec_collate_fn,
        )
        if len(loader) == 0:
            raise ValueError("No training batches were created")

        global_batch_size = int(config["batch_size"]) * int(runtime["world_size"])
        metadata.update(
            {
                "train_pairs_path": str(config["train_pairs_path"]),
                "train_reactions_path": str(config["train_reactions_path"]),
                "ec_labels_path": str(config["ec_labels_path"]),
                "per_gpu_batch_size": int(config["batch_size"]),
                "global_batch_size": global_batch_size,
                "world_size": int(runtime["world_size"]),
            }
        )

        reaction_encoder: torch.nn.Module = build_reaction_encoder(
            [int(value) for value in config["reaction_encoder_dims"]]
        ).to(device)
        projector: torch.nn.Module = LorentzEnzymeProjector(
            input_dim=int(config["reaction_encoder_dims"][-1]),
            hyp_dim=int(config["hyp_dim"]),
            curvature=float(config["curvature"]),
            tangent_clip=config["tangent_clip"],
            tangent_clip_mode=str(config["tangent_clip_mode"]),
            input_normalization=str(config["projector_input_normalization"]),
            eps=float(config["eps"]),
        ).to(device)
        if resume_checkpoint is not None:
            _load_module_state(
                reaction_encoder,
                resume_checkpoint["query_encoder_state_dict"],
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
            reaction_encoder = DDP(reaction_encoder, **ddp_kwargs)
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
        optimizer = torch.optim.AdamW(
            [
                parameter
                for module in (reaction_encoder, projector)
                for parameter in module.parameters()
                if parameter.requires_grad
            ],
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        if resume_checkpoint is not None and resume_checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)

        _rank_zero_print(runtime, f"Loaded reaction EC dataset: {len(dataset)} samples")
        _rank_zero_print(runtime, f"Unique labeled reactions: {metadata['unique_labeled_reactions']}")
        _rank_zero_print(runtime, f"Unique EC labels: {metadata['unique_ec_labels']}")
        _rank_zero_print(runtime, f"Reaction encoder dims: {config['reaction_encoder_dims']}")
        _rank_zero_print(runtime, f"Lorentz hyp dim: {config['hyp_dim']}")
        _rank_zero_print(runtime, f"Curvature kappa: {config['curvature']}")
        _rank_zero_print(runtime, f"Device: {device}")
        _rank_zero_print(runtime, f"World size: {runtime['world_size']}")
        _rank_zero_print(runtime, f"Per-GPU batch size: {config['batch_size']}")
        _rank_zero_print(runtime, f"Global batch size: {global_batch_size}")

        global_step = (
            int(resume_checkpoint.get("global_step", 0))
            if resume_checkpoint is not None
            else 0
        )
        max_steps = config.get("max_steps")
        max_steps = None if max_steps is None else int(max_steps)
        for local_epoch in range(1, int(config["epochs"]) + 1):
            epoch = resume_epoch_offset + local_epoch
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch_loss = 0.0
            epoch_steps = 0
            for batch in loader:
                global_step += 1
                epoch_steps += 1

                reaction_vec = batch["reaction_vec"].to(device=device, non_blocking=True)
                ec_depth_matrix = batch["ec_depth_matrix"].to(device=device, non_blocking=True)
                ec_labels = batch["ec_labels"]

                reaction_embedding = reaction_encoder(reaction_vec)
                z_hyp, _ = projector(reaction_embedding)
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
                        "num_centroid_cone_terms": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                    }
                if bool(config["use_ec_entailment"]):
                    loss_ec_entailment, ec_entailment_logs = ec_entailment_loss(z_hyp, ec_labels)
                else:
                    loss_ec_entailment = z_hyp.sum() * 0.0
                    ec_entailment_logs = {
                        "loss_ec_entailment": loss_ec_entailment.detach(),
                        "num_ec_entailment_terms": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                        "ec_entailment_violation_rate": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                        "mean_ec_entailment_angle": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
                        "mean_ec_entailment_aperture": torch.zeros((), dtype=z_hyp.dtype, device=z_hyp.device),
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
                    mean_radius, std_radius = _compute_radius_stats(
                        z_hyp,
                        curvature=float(config["curvature"]),
                        eps=float(config["eps"]),
                    )
                    from horizyn.lorentz import lorentz_constraint_error

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
                        "mean_radius_reaction": mean_radius.detach(),
                        "std_radius_reaction": std_radius.detach(),
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

                if max_steps is not None and global_step >= max_steps:
                    break

            mean_epoch_loss = _reduce_epoch_mean(epoch_loss, epoch_steps, runtime)
            _rank_zero_print(runtime, f"epoch={epoch} mean_loss={mean_epoch_loss:.4f}")
            if _is_rank_zero(runtime):
                if mean_epoch_loss < best_epoch_loss:
                    best_epoch_loss = mean_epoch_loss
                    _save_checkpoint(
                        best_checkpoint_path,
                        reaction_encoder=reaction_encoder,
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
                    reaction_encoder=reaction_encoder,
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
    finally:
        _cleanup_runtime(runtime)

    return checkpoint_path


def main() -> None:
    args = build_arg_parser().parse_args()
    config = resolve_config(args)
    train(config)


if __name__ == "__main__":
    main()
