#!/usr/bin/env python3
"""Build factorized capability-vector artifacts from a flat capability NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FAMILIES = ("cofactor", "center", "transition")


def _decode_ids(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def _window(vectors: np.ndarray, start: int, width: int) -> np.ndarray:
    dim = vectors.shape[1]
    if width <= dim:
        end = min(start + width, dim)
        out = vectors[:, start:end]
        if out.shape[1] == width:
            return out
        return np.concatenate([out, vectors[:, : width - out.shape[1]]], axis=1)
    repeats = int(np.ceil(width / dim))
    tiled = np.tile(vectors, (1, repeats))
    return tiled[:, :width]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flat-vectors", required=True, type=Path)
    parser.add_argument("--out-npz", required=True, type=Path)
    parser.add_argument("--out-metadata", type=Path, default=None)
    parser.add_argument("--family-dim", type=int, default=128)
    args = parser.parse_args()

    payload = np.load(args.flat_vectors, allow_pickle=True)
    if "ids" not in payload or "vectors" not in payload:
        raise KeyError("--flat-vectors must contain 'ids' and 'vectors'")
    ids = np.asarray(_decode_ids(payload["ids"]), dtype=object)
    vectors = np.asarray(payload["vectors"], dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be rank-2, got shape={vectors.shape}")
    if args.family_dim <= 0:
        raise ValueError("--family-dim must be positive")

    dim = vectors.shape[1]
    starts = {
        "cofactor": 0,
        "center": max(0, (dim - args.family_dim) // 2),
        "transition": max(0, dim - args.family_dim),
    }
    mask = (
        np.asarray(payload["mask"], dtype=bool)
        if "mask" in payload
        else np.linalg.norm(vectors, axis=1) > 0
    )
    arrays = {"ids": ids}
    for family in FAMILIES:
        arrays[f"{family}_vectors"] = _window(vectors, starts[family], args.family_dim).astype(
            np.float32,
            copy=False,
        )
        arrays[f"{family}_mask"] = mask.astype(bool, copy=False)

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **arrays)

    metadata = {
        "source": str(args.flat_vectors),
        "num_ids": int(len(ids)),
        "source_dim": int(dim),
        "family_dim": int(args.family_dim),
        "families": list(FAMILIES),
        "windows": starts,
        "note": (
            "Deterministic overlapping windows derived from the flat capability "
            "latent. Replace with exported family-specific latents when available."
        ),
    }
    metadata_path = args.out_metadata or args.out_npz.with_suffix(".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
