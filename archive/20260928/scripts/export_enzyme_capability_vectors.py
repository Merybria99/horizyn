#!/usr/bin/env python3
"""Export static enzyme capability vectors from a pretrained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils.rnn import pad_sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from horizyn.capability.enzyme_capability_model import EnzymeCapabilityEncoder
from horizyn.capability.io import load_vector_file
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset


def _infer_biological_family_names(hparams: dict, state: dict) -> tuple[str, ...] | None:
    configured = hparams.get("biological_family_names")
    if configured:
        return tuple(str(name) for name in configured)
    names: list[str] = []
    prefix = "enzyme_encoder.family_heads."
    for key in state:
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        name = key[len(prefix) : -len(".weight")]
        if "." in name:
            continue
        if name not in names:
            names.append(name)
    return tuple(names) if names else None


def _resolve_feature_path(feature_root: Path, name: str) -> Path:
    for suffix in (".npz", ".h5", ".hdf5", ".pt", ".npy"):
        path = feature_root / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name} vectors in {feature_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--enzyme-features", required=True)
    parser.add_argument(
        "--residue-h5",
        default=None,
        help=(
            "Optional ProT5 residue HDF5 for checkpoints trained with "
            "multi-query residue pooling. If omitted, prot5_residue.* is "
            "resolved under --enzyme-features."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vector-filename", default="enzyme_capability_vectors.npz")
    parser.add_argument(
        "--enzyme-ids",
        default=None,
        help="Optional newline/CSV file of enzyme IDs to include in metadata.",
    )
    parser.add_argument("--split", default="all")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--truncation", default="ends_center")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for encoder inference.",
    )
    args = parser.parse_args()
    import pandas as pd

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})
    state = checkpoint.get("state_dict", checkpoint)
    biological_family_names = _infer_biological_family_names(hparams, state)
    input_dims = hparams.get("enzyme_input_dims")
    if not isinstance(input_dims, dict):
        raise ValueError("Checkpoint is missing hyper_parameters.enzyme_input_dims")
    encoder = EnzymeCapabilityEncoder(
        input_dims={key: int(value) for key, value in input_dims.items()},
        capability_dim=int(hparams.get("capability_dim", 256)),
        hidden_dim=int(hparams.get("hidden_dim", 512)),
        dropout=float(hparams.get("dropout", 0.1)),
        capability_mode=str(hparams.get("capability_mode", "single")),
        family_dim=int(hparams.get("family_dim", 64)),
        residue_input_dim=hparams.get("residue_input_dim", None),
        use_residue_multiquery_pooling=bool(
            hparams.get("use_residue_multiquery_pooling", False)
        ),
        residue_pool_dropout=float(hparams.get("residue_pool_dropout", 0.0)),
        residue_pool_scale=float(hparams.get("residue_pool_scale", 0.1)),
        biological_family_names=biological_family_names,
    )
    encoder_state = {
        key.removeprefix("enzyme_encoder."): value
        for key, value in state.items()
        if key.startswith("enzyme_encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()

    feature_root = Path(args.enzyme_features)
    feature_maps: dict[str, tuple[list[str], torch.Tensor]] = {}
    common_ids: set[str] | None = None
    for name in input_dims:
        ids, vectors = load_vector_file(_resolve_feature_path(feature_root, name))
        feature_maps[name] = (ids, vectors)
        common_ids = set(ids) if common_ids is None else common_ids & set(ids)
    residue_dataset = None
    if bool(hparams.get("use_residue_multiquery_pooling", False)):
        residue_path = Path(args.residue_h5) if args.residue_h5 else _resolve_feature_path(
            feature_root,
            "prot5_residue",
        )
        residue_dataset = ResidueEmbedDataset(
            str(residue_path),
            in_memory=False,
            max_tokens=args.max_tokens,
            truncation=args.truncation,
        )
        common_ids = set(residue_dataset.keys) if common_ids is None else common_ids & set(residue_dataset.keys)
    if not common_ids:
        raise ValueError("No common enzyme IDs across required feature files")
    if args.enzyme_ids is not None:
        ids_path = Path(args.enzyme_ids)
        if ids_path.suffix.lower() == ".csv":
            frame = pd.read_csv(ids_path)
            id_col = (
                "enzyme_id"
                if "enzyme_id" in frame.columns
                else ("protein_id" if "protein_id" in frame.columns else frame.columns[0])
            )
            requested_ids = [str(value) for value in frame[id_col].dropna().astype(str)]
        else:
            requested_ids = [
                line.strip()
                for line in ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    else:
        requested_ids = sorted(common_ids)
    ordered_ids = [enzyme_id for enzyme_id in requested_ids if enzyme_id in common_ids]
    index_maps = {name: {enzyme_id: idx for idx, enzyme_id in enumerate(ids)} for name, (ids, _) in feature_maps.items()}
    vectors = []
    total_batches = (len(ordered_ids) + args.batch_size - 1) // args.batch_size
    print(
        f"Exporting {len(ordered_ids)} enzyme capability vectors "
        f"in {total_batches} batches on {device}",
        flush=True,
    )
    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, len(ordered_ids), args.batch_size), start=1):
            batch_ids = ordered_ids[start : start + args.batch_size]
            batch = {
                name: tensor[
                    torch.tensor([index_maps[name][enzyme_id] for enzyme_id in batch_ids], dtype=torch.long)
                ]
                for name, (_ids, tensor) in feature_maps.items()
            }
            batch = {name: value.to(device) for name, value in batch.items()}
            residue_embeddings = None
            residue_mask = None
            if residue_dataset is not None:
                residues = [
                    residue_dataset[enzyme_id]["residue_embeddings"]
                    for enzyme_id in batch_ids
                ]
                lengths = torch.tensor([int(value.shape[0]) for value in residues], dtype=torch.long)
                residue_embeddings = pad_sequence(residues, batch_first=True).to(device)
                max_len = int(residue_embeddings.shape[1])
                residue_mask = (
                    torch.arange(max_len, device=device).unsqueeze(0)
                    < lengths.to(device).unsqueeze(1)
                )
            vectors.append(
                encoder(
                    batch,
                    residue_embeddings=residue_embeddings,
                    residue_mask=residue_mask,
                ).cpu().numpy().astype(np.float32)
            )
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % 20 == 0:
                print(f"Exported batch {batch_idx}/{total_batches}", flush=True)
    matrix = (
        np.concatenate(vectors, axis=0)
        if vectors
        else np.zeros((0, int(hparams.get("capability_dim", 256))), dtype=np.float32)
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / args.vector_filename,
        ids=np.asarray(ordered_ids),
        vectors=matrix,
    )
    vector_index = {enzyme_id: idx for idx, enzyme_id in enumerate(ordered_ids)}
    metadata = pd.DataFrame(
        {
            "enzyme_id": requested_ids,
            "vector_index": [vector_index.get(enzyme_id, -1) for enzyme_id in requested_ids],
            "split": [args.split] * len(requested_ids),
            "has_vector": [enzyme_id in vector_index for enzyme_id in requested_ids],
            "source": ["enzyme_side_only"] * len(requested_ids),
            "checkpoint_path": [str(args.checkpoint)] * len(requested_ids),
        }
    )
    metadata.to_parquet(out / "enzyme_capability_metadata.parquet", index=False)
    with (out / "enzyme_capability_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "checkpoint": args.checkpoint,
                "enzyme_features": args.enzyme_features,
                "residue_h5": args.residue_h5,
                "enzyme_ids": args.enzyme_ids,
                "split": args.split,
                "vector_filename": args.vector_filename,
            },
            handle,
            sort_keys=False,
        )


if __name__ == "__main__":
    main()
