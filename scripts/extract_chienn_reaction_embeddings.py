#!/usr/bin/env python3
"""
Extract ChiENN molecule embeddings grouped by reaction side.

The output HDF5 schema matches ``UniMol2ReactionEmbedDataset`` so it can be
used as the chirality branch in ``multimodal_reaction_attention``:

    /ids
    /reactant_vectors, /reactant_offsets
    /product_vectors, /product_offsets

ChiENN is an architecture rather than a released pretrained encoder in the
upstream repository. This script can load a user-provided checkpoint, otherwise
it uses a deterministic frozen ChiENN initialization controlled by ``--seed``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.chemistry.standardizer import Standardizer


@dataclass(frozen=True)
class ReactionSetRecord:
    reaction_id: str
    reactants: tuple[str, ...]
    products: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reactions",
        nargs="+",
        required=True,
        help="Reaction CSV files to encode. Use train + validation/test reactions for a shared HDF5.",
    )
    parser.add_argument("--output", required=True, help="Output ragged HDF5 path")
    parser.add_argument(
        "--chienn-root",
        default=".deps/ChiENN",
        help="Path to cloned gmum/ChiENN repository",
    )
    parser.add_argument(
        "--extra-pythonpath",
        default=".deps/python",
        help="Path containing optional local Python dependencies such as torch-geometric",
    )
    parser.add_argument("--checkpoint", default=None, help="Optional ChiENN checkpoint path")
    parser.add_argument("--key-column", default="reaction_id")
    parser.add_argument("--smiles-column", default="reaction_smiles")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel RDKit/ChiENN preprocessing workers. Model encoding stays in the main process.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-number-of-atoms", type=int, default=512)
    parser.add_argument("--max-number-of-attempts", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="HDF5 compression for vector datasets",
    )
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-hypervalent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-remove-hs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-kekulize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--standardize-uncharge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize-metals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allow-pseudo-reactions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Treat molecule-set strings with no reaction arrow as self reactions.",
    )
    parser.add_argument(
        "--skip-invalid-molecules",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip molecules that ChiENN preprocessing cannot embed.",
    )
    parser.add_argument(
        "--skip-invalid-reactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip reaction directions left with an empty side.",
    )
    parser.add_argument("--limit-reactions", type=int, default=None, help="Debug limit")
    parser.add_argument(
        "--log-skipped-molecules",
        action="store_true",
        help="Log every molecule that ChiENN preprocessing skips.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    return parser.parse_args()


def add_runtime_paths(chienn_root: str, extra_pythonpath: str | None) -> None:
    for path_str in (extra_pythonpath, chienn_root):
        if not path_str:
            continue
        path = Path(path_str).resolve()
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def normalize_smiles(
    smiles: str,
    standardizer: Standardizer | None,
    allow_pseudo_reactions: bool,
) -> str:
    if smiles.count(">") < 2:
        if not allow_pseudo_reactions:
            return smiles
        smiles = f"{smiles}>>{smiles}"
    return standardizer.standardize_reaction(smiles) if standardizer is not None else smiles


def split_reaction_smiles(
    reaction_id: str,
    smiles: str,
    allow_pseudo_reactions: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parts = smiles.split(">>")
    if len(parts) != 2:
        if not allow_pseudo_reactions or ">" in smiles:
            raise ValueError(f"Reaction {reaction_id} is not a two-sided reaction SMILES: {smiles}")
        molecules = tuple(mol for mol in smiles.split(".") if mol)
        if not molecules:
            raise ValueError(f"Reaction {reaction_id} has no molecules: {smiles}")
        return molecules, molecules
    reactants = tuple(mol for mol in parts[0].split(".") if mol)
    products = tuple(mol for mol in parts[1].split(".") if mol)
    if not reactants:
        raise ValueError(f"Reaction {reaction_id} has no reactant molecules: {smiles}")
    if not products:
        raise ValueError(f"Reaction {reaction_id} has no product molecules: {smiles}")
    return reactants, products


def read_reactions(args: argparse.Namespace) -> list[ReactionSetRecord]:
    standardizer = (
        Standardizer(
            standardize_hypervalent=args.standardize_hypervalent,
            standardize_remove_hs=args.standardize_remove_hs,
            standardize_kekulize=args.standardize_kekulize,
            standardize_uncharge=args.standardize_uncharge,
            standardize_metals=args.standardize_metals,
        )
        if args.standardize
        else None
    )
    records: list[ReactionSetRecord] = []
    seen: set[str] = set()

    for path_str in args.reactions:
        path = Path(path_str)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if args.key_column not in reader.fieldnames or args.smiles_column not in reader.fieldnames:
                raise KeyError(
                    f"{path} must contain columns {args.key_column!r} and "
                    f"{args.smiles_column!r}; found {reader.fieldnames}"
                )
            for row_idx, row in enumerate(reader):
                if args.limit_reactions is not None and len(records) >= args.limit_reactions:
                    return records
                base_id = str(row[args.key_column])
                try:
                    smiles = normalize_smiles(
                        str(row[args.smiles_column]),
                        standardizer=standardizer,
                        allow_pseudo_reactions=args.allow_pseudo_reactions,
                    )
                    reactants, products = split_reaction_smiles(
                        base_id,
                        smiles,
                        allow_pseudo_reactions=args.allow_pseudo_reactions,
                    )
                except Exception as exc:
                    if args.skip_invalid_reactions:
                        print(f"Warning: skipping invalid reaction {base_id}: {exc}")
                        continue
                    raise

                directions = (
                    ((base_id, reactants, products),)
                    if not args.bidirectional
                    else (
                        (f"{base_id}_f", reactants, products),
                        (f"{base_id}_r", products, reactants),
                    )
                )
                for reaction_id, side_a, side_b in directions:
                    if reaction_id in seen:
                        continue
                    seen.add(reaction_id)
                    records.append(ReactionSetRecord(reaction_id, side_a, side_b))
    return records


def ordered_unique_molecules(records: list[ReactionSetRecord]) -> list[str]:
    seen: set[str] = set()
    molecules: list[str] = []
    for record in records:
        for smiles in (*record.reactants, *record.products):
            if smiles not in seen:
                seen.add(smiles)
                molecules.append(smiles)
    return molecules


def smiles_to_chienn_data(
    smiles: str,
    max_number_of_atoms: int,
    max_number_of_attempts: int,
):
    from chienn.data.edge_graph.to_edge_graph import to_edge_graph
    from chienn.data.featurization.mol_to_data import mol_to_data
    from chienn.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol

    mol = smiles_to_3d_mol(
        smiles,
        max_number_of_atoms=max_number_of_atoms,
        max_number_of_attempts=max_number_of_attempts,
    )
    if mol is None:
        return None
    data = mol_to_data(mol)
    if data.edge_index.numel() == 0:
        return None
    data = to_edge_graph(data)
    data.pos = None
    return data


def _preprocess_worker(task: tuple[int, str, int, int]) -> tuple[int, str, object | None, str | None]:
    idx, smiles, max_number_of_atoms, max_number_of_attempts = task
    try:
        data = smiles_to_chienn_data(
            smiles,
            max_number_of_atoms=max_number_of_atoms,
            max_number_of_attempts=max_number_of_attempts,
        )
        return idx, smiles, data, None if data is not None else "empty_or_unsupported_graph"
    except Exception as exc:
        return idx, smiles, None, str(exc)


class ChiENNEmbedder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        k_neighbors: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        from chienn.model.chienn_model import ChiENNModel
        from torch_geometric.utils import to_dense_batch

        self.model = ChiENNModel(
            k_neighbors=k_neighbors,
            hidden_dim=hidden_dim,
            n_layers=num_layers,
            dropout=dropout,
            out_dim=1,
        )
        self.to_dense_batch = to_dense_batch

    def forward(self, batch) -> torch.Tensor:
        batch.x = self.model.embedding_layer(batch.x)
        for layer in self.model.gps_layers:
            batch.x = layer(batch)
        dense_x, mask = self.to_dense_batch(batch.x, batch.batch)
        mask_f = mask.unsqueeze(-1).to(dtype=dense_x.dtype)
        return (dense_x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)


def load_checkpoint_if_needed(model: torch.nn.Module, checkpoint_path: str | None) -> None:
    if checkpoint_path is None:
        return
    payload = torch.load(checkpoint_path, map_location="cpu")
    state = payload
    for key in ("state_dict", "model_state_dict", "model", "chienn_state_dict"):
        if isinstance(payload, dict) and key in payload:
            state = payload[key]
            break
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported ChiENN checkpoint format: {checkpoint_path}")
    cleaned = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "chienn.", "encoder."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("model."):
            cleaned[new_key] = value
        else:
            cleaned[f"model.{new_key}"] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(
        f"Loaded ChiENN checkpoint {checkpoint_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def encode_molecules(
    molecules: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[str]]:
    from chienn import collate_with_circle_index

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = ChiENNEmbedder(
        hidden_dim=args.hidden_dim,
        k_neighbors=args.k_neighbors,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    load_checkpoint_if_needed(model, args.checkpoint)
    model.eval().to(args.device)

    cache: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    data_batch = []
    smiles_batch = []

    def flush() -> None:
        nonlocal data_batch, smiles_batch
        if not data_batch:
            return
        batch = collate_with_circle_index(data_batch, k_neighbors=args.k_neighbors).to(args.device)
        with torch.inference_mode():
            embeddings = model(batch).detach().cpu().float().numpy()
        for smiles, vector in zip(smiles_batch, embeddings):
            cache[smiles] = vector
        data_batch = []
        smiles_batch = []

    def handle_preprocessed(idx: int, smiles: str, data, error: str | None) -> None:
        nonlocal data_batch, smiles_batch
        if data is None:
            if not args.skip_invalid_molecules:
                raise ValueError(
                    f"ChiENN preprocessing failed for molecule {smiles}: {error}"
                )
            if args.log_skipped_molecules and error is not None:
                logging.warning(
                    "Skipping molecule after ChiENN preprocessing failure: %s (%s)",
                    smiles,
                    error,
                )
            skipped.append(smiles)
        else:
            data_batch.append(data)
            smiles_batch.append(smiles)
        if len(data_batch) >= args.batch_size:
            flush()

    if args.num_workers > 0:
        tasks = (
            (idx, smiles, args.max_number_of_atoms, args.max_number_of_attempts)
            for idx, smiles in enumerate(molecules, start=1)
        )
        with get_context("spawn").Pool(processes=args.num_workers) as pool:
            for idx, smiles, data, error in pool.imap_unordered(_preprocess_worker, tasks, chunksize=16):
                handle_preprocessed(idx, smiles, data, error)
                if idx % 1000 == 0:
                    print(
                        f"Prepared/encoded {idx}/{len(molecules)} molecules; "
                        f"skipped={len(skipped)}"
                    )
    else:
        for idx, smiles in enumerate(molecules, start=1):
            try:
                data = smiles_to_chienn_data(
                    smiles,
                    max_number_of_atoms=args.max_number_of_atoms,
                    max_number_of_attempts=args.max_number_of_attempts,
                )
                error = None if data is not None else "empty_or_unsupported_graph"
            except Exception as exc:
                data = None
                error = str(exc)
            handle_preprocessed(idx, smiles, data, error)
            if idx % 1000 == 0:
                print(
                    f"Prepared/encoded {idx}/{len(molecules)} molecules; "
                    f"skipped={len(skipped)}"
                )

    flush()
    return cache, skipped


def filter_records(
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    skip_invalid_molecules: bool,
    skip_invalid_reactions: bool,
) -> list[ReactionSetRecord]:
    filtered = []
    for record in records:
        reactants = tuple(smiles for smiles in record.reactants if smiles in molecule_cache)
        products = tuple(smiles for smiles in record.products if smiles in molecule_cache)
        if (
            not skip_invalid_molecules
            and (len(reactants) != len(record.reactants) or len(products) != len(record.products))
        ):
            missing = sorted(set((*record.reactants, *record.products)) - set(molecule_cache))
            raise KeyError(f"Missing ChiENN embeddings for reaction {record.reaction_id}: {missing}")
        if not reactants or not products:
            if skip_invalid_reactions:
                continue
            raise ValueError(f"Reaction {record.reaction_id} has an empty side after ChiENN filtering")
        filtered.append(ReactionSetRecord(record.reaction_id, reactants, products))
    return filtered


def flatten_side(
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    side: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    offsets = [0]
    for record in records:
        for smiles in getattr(record, side):
            vectors.append(molecule_cache[smiles])
        offsets.append(offsets[-1] + len(getattr(record, side)))
    return np.asarray(vectors, dtype=dtype), np.asarray(offsets, dtype=np.int64)


def write_hdf5(
    output: str,
    records: list[ReactionSetRecord],
    molecule_cache: dict[str, np.ndarray],
    skipped_molecules: list[str],
    args: argparse.Namespace,
) -> None:
    output_path = Path(output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    dtype = np.float16 if args.dtype == "float16" else np.float32
    reactant_vectors, reactant_offsets = flatten_side(records, molecule_cache, "reactants", dtype)
    product_vectors, product_offsets = flatten_side(records, molecule_cache, "products", dtype)
    h5_compression = None if args.compression == "none" else args.compression
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_path, "w") as handle:
        handle.create_dataset(
            "ids",
            data=np.asarray([record.reaction_id for record in records], dtype=object),
            dtype=string_dtype,
        )
        handle.create_dataset("reactant_vectors", data=reactant_vectors, compression=h5_compression)
        handle.create_dataset("reactant_offsets", data=reactant_offsets)
        handle.create_dataset("product_vectors", data=product_vectors, compression=h5_compression)
        handle.create_dataset("product_offsets", data=product_offsets)
        handle.attrs["embedding_dim"] = int(args.hidden_dim)
        handle.attrs["reaction_representation"] = "chienn_chirality"
        handle.attrs["model_name"] = "gmum/ChiENN"
        handle.attrs["checkpoint"] = "" if args.checkpoint is None else str(args.checkpoint)
        handle.attrs["deterministic_seed"] = int(args.seed)
        handle.attrs["k_neighbors"] = int(args.k_neighbors)
        handle.attrs["num_layers"] = int(args.num_layers)
        handle.attrs["hidden_dim"] = int(args.hidden_dim)
        handle.attrs["num_skipped_molecules"] = int(len(skipped_molecules))
        handle.attrs["num_reactions"] = int(len(records))


def main() -> None:
    args = parse_args()
    add_runtime_paths(args.chienn_root, args.extra_pythonpath)
    records = read_reactions(args)
    print(f"Loaded {len(records)} reaction directions")
    molecules = ordered_unique_molecules(records)
    print(f"Encoding {len(molecules)} unique molecules with ChiENN")
    molecule_cache, skipped = encode_molecules(molecules, args)
    print(f"Encoded {len(molecule_cache)} molecules; skipped {len(skipped)}")
    records = filter_records(
        records,
        molecule_cache=molecule_cache,
        skip_invalid_molecules=args.skip_invalid_molecules,
        skip_invalid_reactions=args.skip_invalid_reactions,
    )
    print(f"Keeping {len(records)} reaction directions after ChiENN molecule filtering")
    write_hdf5(args.output, records, molecule_cache, skipped, args)
    print(f"Wrote ChiENN reaction embeddings to {args.output}")


if __name__ == "__main__":
    main()
