"""Shared I/O helpers for capability preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_id(value: Any) -> str:
    return str(value).strip()


def decode_ids(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def load_npz_vectors(path: str | Path):
    import torch

    data = np.load(path, allow_pickle=True)
    if "ids" not in data or "vectors" not in data:
        raise KeyError(f"{path} must contain 'ids' and 'vectors' arrays")
    ids = decode_ids(data["ids"])
    vectors = torch.as_tensor(data["vectors"], dtype=torch.float32)
    if vectors.ndim != 2:
        raise ValueError(f"'vectors' must be rank-2, got shape={tuple(vectors.shape)}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(
            f"Mismatch between ids ({len(ids)}) and vectors ({vectors.shape[0]}) in {path}"
        )
    return ids, vectors


def load_vector_file(path: str | Path):
    import torch

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_vectors(path)
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5:
            if "ids" not in h5 or "vectors" not in h5:
                raise KeyError(f"{path} must contain HDF5 datasets 'ids' and 'vectors'")
            ids = decode_ids(h5["ids"][:])
            vectors = torch.as_tensor(h5["vectors"][:], dtype=torch.float32)
        return ids, vectors
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict) or "ids" not in payload or "vectors" not in payload:
            raise KeyError(f"{path} must be a torch dict with 'ids' and 'vectors'")
        ids = decode_ids(payload["ids"])
        vectors = torch.as_tensor(payload["vectors"], dtype=torch.float32)
        return ids, vectors
    if suffix == ".npy":
        vectors = torch.as_tensor(np.load(path), dtype=torch.float32)
        ids_path = path.with_name(f"{path.stem}_ids.json")
        if not ids_path.exists():
            ids_path = path.with_name(f"{path.stem}_ids.txt")
        if not ids_path.exists():
            raise FileNotFoundError(
                f"{path} is a raw .npy vector matrix; expected sidecar "
                f"{path.stem}_ids.json or {path.stem}_ids.txt"
            )
        if ids_path.suffix == ".json":
            ids = [str(value) for value in read_json(ids_path)]
        else:
            ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
        return ids, vectors
    raise ValueError(f"Unsupported vector file extension: {path.suffix}")
