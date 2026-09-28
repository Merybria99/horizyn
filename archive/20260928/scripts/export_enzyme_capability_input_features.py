#!/usr/bin/env python3
"""Export static enzyme branch features for capability pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.utils import residue_collate_fn


def _read_id_file(path: str | Path | None) -> list[str] | None:
    if path is None:
        return None
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _subset_for_ids(dataset: ResidueEmbedDataset, ids: list[str] | None) -> tuple[Subset, list[str]]:
    if ids is None:
        return Subset(dataset, list(range(len(dataset)))), list(dataset.keys)
    keep = []
    kept_ids = []
    for enzyme_id in ids:
        try:
            keep.append(dataset.key_to_idx[enzyme_id])
            kept_ids.append(enzyme_id)
        except KeyError:
            continue
    if not keep:
        raise ValueError("None of the requested enzyme IDs are present in the residue HDF5")
    return Subset(dataset, keep), kept_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Protein pooling checkpoint")
    parser.add_argument("--config", required=True, help="Protein pooling config")
    parser.add_argument("--protein-id-file", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    module = ProteinPooledLitModule.load_from_checkpoint(
        args.checkpoint,
        map_location=args.device,
    )
    module.eval()
    module.to(args.device)
    model = module.model
    if model.raw_mean_pooling is None or model.hyperbolic_projector is None:
        raise ValueError(
            "Checkpoint/config must use raw_mean_sleec_hyperbolic_gated or "
            "raw_mean_sleec_hyperbolic_capability_gated"
        )

    residue_path = config.data.get("protein_residue_embeds_path")
    if residue_path is None:
        raise ValueError("config.data.protein_residue_embeds_path is required")
    dataset = ResidueEmbedDataset(
        file_path=residue_path,
        in_memory=False,
        max_tokens=config.data.get("max_protein_tokens", 1024),
        truncation=config.data.get("protein_truncation", "ends_center"),
    )
    requested_ids = _read_id_file(args.protein_id_file)
    subset, ordered_ids = _subset_for_ids(dataset, requested_ids)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=residue_collate_fn,
    )

    raw_vectors = []
    sleec_vectors = []
    tangent_vectors = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            residues = batch["residue_embeddings"].to(args.device)
            padding_mask = batch["residue_padding_mask"].to(args.device)
            attention_mask = ~padding_mask.to(torch.bool)
            raw_mean = model.raw_mean_pooling(residues, attention_mask=attention_mask)
            pooled = model.pool_residues(
                residues,
                residue_padding_mask=padding_mask,
                return_attention=False,
            )
            if isinstance(pooled, tuple):
                pooled = pooled[0]
            _hyp, tangent = model.hyperbolic_projector(pooled)
            raw_vectors.append(raw_mean.detach().cpu().numpy().astype(np.float32))
            sleec_vectors.append(pooled.detach().cpu().numpy().astype(np.float32))
            tangent_vectors.append(tangent.detach().cpu().numpy().astype(np.float32))
            offset += residues.shape[0]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(ordered_ids)
    np.savez_compressed(
        out / "prot5_mean.npz",
        ids=ids,
        vectors=np.concatenate(raw_vectors, axis=0),
    )
    np.savez_compressed(
        out / "prot5_sleec.npz",
        ids=ids,
        vectors=np.concatenate(sleec_vectors, axis=0),
    )
    np.savez_compressed(
        out / "lorentz_tangent.npz",
        ids=ids,
        vectors=np.concatenate(tangent_vectors, axis=0),
    )


if __name__ == "__main__":
    main()
