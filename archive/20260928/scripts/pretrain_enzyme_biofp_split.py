#!/usr/bin/env python3
"""Pretrain sequence-derived biological enzyme branches from residue embeddings."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.enzyme_capability_dataset import (  # noqa: E402
    BioFPTargetDataset,
    TargetWithBioFPTargetDataset,
)
from horizyn.config import DotDict  # noqa: E402
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset  # noqa: E402
from horizyn.model import ProteinPooledDualModel  # noqa: E402
from horizyn.utils import residue_collate_fn  # noqa: E402
from horizyn.wandb_utils import build_wandb_logger  # noqa: E402


class KeySubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.dataset[self.indices[idx]]


class EnzymeBioFPLitModule(pl.LightningModule):
    def __init__(
        self,
        model: ProteinPooledDualModel,
        *,
        learning_rate: float,
        weight_decay: float,
        family_weights: dict[str, float],
        confidence_cap: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.family_weights = {
            str(name): float(weight) for name, weight in family_weights.items()
        }
        self.confidence_cap = float(confidence_cap)

    def _step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        _, details = self.model.encode_targets(
            batch["residue_embeddings"],
            residue_padding_mask=batch["residue_padding_mask"],
            return_pooling_details=True,
        )
        total = torch.zeros((), device=self.device)
        active_families = 0
        for family, family_weight in self.family_weights.items():
            logits = details.get(f"biofp_logits_{family}")
            targets = batch.get(f"biofp_{family}_targets")
            mask = batch.get(f"biofp_{family}_mask")
            denominator = batch.get(f"biofp_{family}_denominator")
            explicit_confidence = batch.get(f"biofp_{family}_confidence")
            if logits is None or targets is None or mask is None or denominator is None:
                continue
            targets = targets.to(device=logits.device, dtype=logits.dtype)
            mask = mask.to(device=logits.device, dtype=torch.bool)
            denominator = denominator.to(device=logits.device, dtype=logits.dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(1).expand_as(logits)
            if denominator.ndim == 1:
                denominator = denominator.unsqueeze(1).expand_as(logits)
            if explicit_confidence is None:
                confidence = (
                    denominator.clamp(min=1.0, max=self.confidence_cap)
                    / self.confidence_cap
                )
            else:
                confidence = explicit_confidence.to(
                    device=logits.device,
                    dtype=logits.dtype,
                )
                if confidence.ndim == 1:
                    confidence = confidence.unsqueeze(1).expand_as(logits)
            if mask.shape != logits.shape or denominator.shape != logits.shape:
                raise ValueError(f"{family} target masks must be row-wise or per-label")
            if confidence.shape != logits.shape:
                raise ValueError(f"{family} confidence must be row-wise or per-label")
            valid = mask & denominator.gt(0) & confidence.gt(0)
            active = valid.any(dim=1).sum()
            self.log(f"{stage}/biofp_{family}_active", active.float(), prog_bar=False, sync_dist=True)
            if active.item() == 0:
                continue
            per_label = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            weights = confidence.clamp_min(0.0) * valid.to(dtype=logits.dtype)
            family_loss = (per_label * weights).sum() / weights.sum().clamp_min(1e-12)
            total = total + float(family_weight) * family_loss
            active_families += 1
            self.log(f"{stage}/biofp_{family}", family_loss, prog_bar=False, sync_dist=True)

        if active_families == 0:
            raise RuntimeError("No active BioFP targets were present in the batch")
        self.log(f"{stage}/loss", total, prog_bar=True, sync_dist=True)
        if "biofp_sequence_gate_raw_mean" in details:
            self.log(
                f"{stage}/gate_raw_mean",
                details["biofp_sequence_gate_raw_mean"].mean(),
                prog_bar=False,
                sync_dist=True,
            )
            self.log(
                f"{stage}/gate_pooled",
                details["biofp_sequence_gate_pooled"].mean(),
                prog_bar=False,
                sync_dist=True,
            )
        return total

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume Lightning state from a checkpoint; max_epochs remains the total target.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return DotDict(payload)


def active_biofp_keys(
    dataset: BioFPTargetDataset,
    family_weights: dict[str, float],
) -> set[str]:
    families = [name for name, weight in family_weights.items() if float(weight) > 0]
    if not families:
        raise ValueError("BioFP pretraining requires at least one positive family weight")
    unknown = set(families) - set(dataset.FAMILIES)
    if unknown:
        raise ValueError(f"BioFP target artifact has no families: {sorted(unknown)}")

    active = torch.zeros(len(dataset), dtype=torch.bool)
    for family in families:
        mask = dataset.masks[family]
        denominator = dataset.denominators[family]
        if mask.ndim == 1:
            mask = mask.unsqueeze(1).expand_as(dataset.targets[family])
        if denominator.ndim == 1:
            denominator = denominator.unsqueeze(1).expand_as(dataset.targets[family])
        family_active = (mask & denominator.gt(0)).any(dim=1)
        active |= family_active
    return {dataset.keys[index] for index in active.nonzero(as_tuple=False).flatten().tolist()}


def configured_family_weights(config: dict[str, Any]) -> dict[str, float]:
    training = config["training"]
    configured = training.get("biofp_family_weights", None)
    if configured is not None:
        return {str(name): float(weight) for name, weight in configured.items()}
    return {
        "center": float(training.get("biofp_center_weight", 0.45)),
        "transition": float(training.get("biofp_transition_weight", 0.35)),
        "cofactor": float(training.get("biofp_cofactor_weight", 0.20)),
    }


def build_dataset(config: dict[str, Any]) -> TargetWithBioFPTargetDataset:
    data = config["data"]
    print("Loading ProT5 residue HDF5 for BioFP pretraining...", flush=True)
    residues = ResidueEmbedDataset(
        data["protein_residue_embeds_path"],
        in_memory=bool(data.get("residue_in_memory", False)),
        dtype=torch.float32,
        max_tokens=data.get("max_protein_tokens", 1022),
        truncation=data.get("protein_truncation", "ends_center"),
    )
    print(f"Loaded residue embeddings for {len(residues)} proteins", flush=True)
    print("Loading enzyme BioFP soft targets...", flush=True)
    biofp = BioFPTargetDataset(data["protein_biofp_targets_path"])
    print(f"Loaded BioFP targets for {len(biofp)} proteins", flush=True)
    dataset = TargetWithBioFPTargetDataset(
        residues,
        biofp,
        missing_policy=data.get("biofp_missing_policy", "zero_with_mask"),
    )
    biofp_key_set = active_biofp_keys(
        biofp,
        configured_family_weights(config),
    )
    keep_indices = [
        idx
        for idx, key in enumerate(dataset.keys)
        if key in biofp_key_set
    ]
    if not keep_indices:
        raise ValueError("No residue HDF5 proteins have BioFP targets")
    print(
        f"Keeping {len(keep_indices)} proteins with active supervised BioFP targets "
        f"out of {len(dataset)} residue-embedded proteins",
        flush=True,
    )
    return KeySubsetDataset(dataset, keep_indices)


def build_model(config: dict[str, Any]) -> ProteinPooledDualModel:
    data = config["data"]
    model_cfg = config["model"]
    biofp = model_cfg["biofp"]
    sleec = model_cfg.get("sleec_pooling", {})
    hyperbolic = model_cfg.get("hyperbolic_encoder", {})
    embedding_dim = int(model_cfg.get("embedding_dim", 512))
    return ProteinPooledDualModel(
        query_encoder_kwargs={
            "input_dim": embedding_dim,
            "output_dim": embedding_dim,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        target_encoder_kwargs={
            "input_dim": embedding_dim,
            "output_dim": embedding_dim,
            "num_layers": 0,
            "widths": [],
            "normalise_output": True,
        },
        residue_dim=int(data.get("residue_dim", 1024)),
        pooling=model_cfg.get("pooling", "sleec_guided_attention"),
        sleec_threshold=sleec.get("threshold", 0.34),
        sleec_scorer_hidden_dim=sleec.get("scorer_hidden_dim", 256),
        sleec_checkpoint_path=sleec.get("checkpoint_path", None),
        sleec_freeze_scorer=sleec.get("freeze_scorer", True),
        sleec_guided_initial_bias_scale=sleec.get("initial_bias_scale", 1.0),
        sleec_guided_train_bias_scale=sleec.get("train_bias_scale", False),
        hyperbolic_checkpoint_path=hyperbolic.get("checkpoint_path", None),
        hyperbolic_freeze_projector=hyperbolic.get("freeze_projector", False),
        hyperbolic_use_tangent=hyperbolic.get("use_tangent", True),
        hyperbolic_hyp_dim=hyperbolic.get("hyp_dim", 128),
        enzyme_input_mode=model_cfg.get(
            "enzyme_input_mode",
            "raw_mean_sleec_biofp_split",
        ),
        enzyme_block_dims=model_cfg.get("enzyme_block_fusion", {}).get("dims", None),
        enzyme_block_weights=model_cfg.get("enzyme_block_fusion", {}).get("weights", None),
        enzyme_block_dropout=model_cfg.get("enzyme_block_fusion", {}).get("dropout", 0.0),
        biofp_center_dim=int(biofp.get("center_dim", 0)),
        biofp_cofactor_dim=int(biofp.get("cofactor_dim", 0)),
        biofp_transition_dim=int(biofp.get("transition_dim", 0)),
        biofp_family_dims=biofp.get("family_dims", None),
        biofp_seq_dim=int(biofp.get("seq_dim", 384)),
        biofp_dim=int(biofp.get("dim", 128)),
        biofp_hidden_dim=int(biofp.get("hidden_dim", 512)),
        biofp_seq_weight=float(biofp.get("seq_weight", 0.75)),
        biofp_dropout=float(biofp.get("dropout", 0.1)),
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(config["logging"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config["logging"]["log_dir"]).mkdir(parents=True, exist_ok=True)

    seed = int(config.get("seed", 42))
    pl.seed_everything(seed, workers=True)
    dataset = build_dataset(config)
    training = config["training"]
    val_fraction = float(training.get("validation_fraction", 0.05))
    val_size = max(1, int(math.floor(len(dataset) * val_fraction)))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("BioFP pretraining dataset is too small after validation split")
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    print(
        f"BioFP pretraining split: train={train_size}, val={val_size}, "
        f"batch_size={int(training.get('batch_size', 64))}",
        flush=True,
    )

    num_workers = int(training.get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(training.get("batch_size", 64)),
        "num_workers": num_workers,
        "pin_memory": bool(training.get("pin_memory", False)),
        "collate_fn": residue_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(training.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(config)
    print("Built enzyme BioFP split stack", flush=True)
    module = EnzymeBioFPLitModule(
        model,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        family_weights=configured_family_weights(config),
        confidence_cap=float(training.get("biofp_confidence_cap", 8.0)),
    )

    logger = build_wandb_logger(argparse.Namespace(**config.get("wandb_args", {})), config)
    print("Initialized logging; starting Lightning trainer setup", flush=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="enzyme-biofp-pretrain-{epoch:02d}",
        monitor=training.get("checkpoint_monitor", "val/loss"),
        mode=training.get("checkpoint_mode", "min"),
        save_last=True,
        save_top_k=1,
    )
    callbacks = [
        checkpoint_callback,
        pl.callbacks.EarlyStopping(
            monitor=training.get("checkpoint_monitor", "val/loss"),
            mode=training.get("checkpoint_mode", "min"),
            patience=int(training.get("early_stopping_patience", 4)),
            min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        ),
    ]
    trainer = pl.Trainer(
        max_epochs=int(training.get("max_epochs", 10)),
        accelerator=training.get("accelerator", "gpu"),
        devices=training.get("devices", 1),
        strategy=training.get("strategy", "auto"),
        precision=training.get("precision", "32-true"),
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=int(training.get("log_every_n_steps", 10)),
        enable_progress_bar=bool(training.get("enable_progress_bar", True)),
        gradient_clip_val=float(training.get("gradient_clip_val", 0.0)),
    )
    print("Starting enzyme BioFP pretraining", flush=True)
    resume_checkpoint = args.resume or training.get("resume_from_checkpoint")
    if resume_checkpoint is not None:
        resume_checkpoint = str(Path(resume_checkpoint).resolve())
        if not Path(resume_checkpoint).is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        print(f"Resuming enzyme BioFP pretraining from {resume_checkpoint}", flush=True)
    trainer.fit(module, train_loader, val_loader, ckpt_path=resume_checkpoint)

    if trainer.is_global_zero:
        selected_checkpoint = Path(checkpoint_callback.best_model_path)
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(
                f"Validation-selected BioFP checkpoint is missing: {selected_checkpoint}"
            )
        stable_best_checkpoint = output_dir / "best.ckpt"
        temporary_best_checkpoint = output_dir / ".best.ckpt.tmp"
        shutil.copy2(selected_checkpoint, temporary_best_checkpoint)
        temporary_best_checkpoint.replace(stable_best_checkpoint)
        best_score = checkpoint_callback.best_model_score
        summary = {
            "checkpoint_dir": str(output_dir),
            "last_checkpoint": str(output_dir / "last.ckpt"),
            "best_checkpoint": str(stable_best_checkpoint),
            "source_best_checkpoint": str(selected_checkpoint),
            "best_score": None if best_score is None else float(best_score.item()),
            "train_size": train_size,
            "val_size": val_size,
            "completed_epochs": int(trainer.current_epoch),
        }
        (output_dir / "pretrain_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    trainer.strategy.barrier("biofp-pretrain-summary")


if __name__ == "__main__":
    main()
