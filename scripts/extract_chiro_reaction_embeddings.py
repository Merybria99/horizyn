#!/usr/bin/env python3
"""
Extract ChIRo molecule embeddings grouped by reaction side.

The output HDF5 schema matches ``UniMol2ReactionEmbedDataset`` so it can be
used as the chirality branch in ``multimodal_reaction_attention``:

    /ids
    /reactant_vectors, /reactant_offsets
    /product_vectors, /product_offsets

ChIRo's public checkpoints were trained with an older PyTorch Geometric API.
This extractor remaps those checkpoint keys to the current PyG names and uses
the internal ChIRo embedding before the task prediction head. The default
checkpoint is the published R/S classification ChIRo checkpoint, which yields a
256-dimensional representation when using ``--embedding-kind both``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sqlite3
import sys
import types
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import h5py
import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from horizyn.chemistry.standardizer import Standardizer

DEFAULT_CHIRO_ROOT = ".deps/ChIRo"
DEFAULT_PARAMS = ".deps/ChIRo/paper_results/RS_experiment/ChIRo/params_RS_ChIRo.json"
DEFAULT_CHECKPOINT = (
    ".deps/ChIRo/paper_results/RS_experiment/ChIRo/" "results_RS_ChIRo_seed1/best_model.pt"
)
_WORKER_ARGS: argparse.Namespace | None = None
CACHE_SCHEMA_VERSION = "1"
T = TypeVar("T")
R = TypeVar("R")


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
        help="Reaction CSV files to encode. Use train + validation reactions for a shared HDF5.",
    )
    parser.add_argument("--output", help="Output ragged HDF5 path")
    parser.add_argument("--chiro-root", default=DEFAULT_CHIRO_ROOT)
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--extra-pythonpath", default=".deps/python")
    parser.add_argument("--key-column", default="reaction_id")
    parser.add_argument("--smiles-column", default="reaction_smiles")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel RDKit/ChIRo graph preprocessing workers. Model inference stays in the main process.",
    )
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=0,
        help="Maximum submitted preprocessing jobs. Zero uses twice --num-workers.",
    )
    parser.add_argument(
        "--molecule-cache",
        help="Persistent SQLite molecule-embedding cache used to resume and deduplicate extraction.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Populate --molecule-cache without writing a reaction HDF5.",
    )
    parser.add_argument(
        "--cache-read-only",
        action="store_true",
        help="Require all molecules to exist in --molecule-cache; never run ChIRo inference.",
    )
    parser.add_argument(
        "--skip-uncached-molecules",
        action="store_true",
        help="With --cache-read-only, omit uncached molecules from ChIRo features.",
    )
    parser.add_argument(
        "--embedding-kind",
        choices=("both", "molecule", "conformer", "z_alpha"),
        default="both",
        help="Which internal ChIRo embedding to export.",
    )
    parser.add_argument("--max-number-of-atoms", type=int, default=512)
    parser.add_argument("--max-embed-attempts", type=int, default=10)
    parser.add_argument("--uff-iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "lzf"),
        default="none",
        help="HDF5 compression for vector datasets.",
    )
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--standardize-hypervalent", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--standardize-remove-hs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--standardize-kekulize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--standardize-uncharge", action=argparse.BooleanOptionalAction, default=True
    )
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
        help="Skip molecules that ChIRo preprocessing cannot embed.",
    )
    parser.add_argument(
        "--skip-invalid-reactions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip reaction directions left with an empty side.",
    )
    parser.add_argument("--limit-reactions", type=int, default=None, help="Debug limit")
    parser.add_argument("--log-skipped-molecules", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()
    if args.cache_only and not args.molecule_cache:
        parser.error("--cache-only requires --molecule-cache")
    if args.cache_read_only and not args.molecule_cache:
        parser.error("--cache-read-only requires --molecule-cache")
    if args.skip_uncached_molecules and not args.cache_read_only:
        parser.error("--skip-uncached-molecules requires --cache-read-only")
    if not args.cache_only and not args.output:
        parser.error("--output is required unless --cache-only is used")
    if args.max_pending_tasks < 0:
        parser.error("--max-pending-tasks must be non-negative")
    return args


def add_runtime_paths(chiro_root: str, extra_pythonpath: str | None) -> None:
    for path_str in (extra_pythonpath, chiro_root):
        if not path_str:
            continue
        path = Path(path_str).resolve()
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def install_chiro_compatibility(device: str) -> None:
    if not hasattr(nx, "from_numpy_matrix") and hasattr(nx, "from_numpy_array"):
        nx.from_numpy_matrix = nx.from_numpy_array

    if "torch_scatter" not in sys.modules:
        scatter_mod = types.ModuleType("torch_scatter")
        composite = types.ModuleType("torch_scatter.composite")

        def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim: int = 0) -> torch.Tensor:
            if dim != 0:
                raise NotImplementedError("ChIRo compatibility scatter_softmax only supports dim=0")
            out = torch.empty_like(src)
            for value in torch.unique(index):
                mask = index == value
                out[mask] = torch.softmax(src[mask], dim=0)
            return out

        composite.scatter_softmax = scatter_softmax
        scatter_mod.composite = composite
        sys.modules["torch_scatter"] = scatter_mod
        sys.modules["torch_scatter.composite"] = composite

    # ChIRo's old code creates a CUDA scalar whenever CUDA is globally
    # available. For CPU extraction we force that specific branch to CPU.
    if str(device).lower() == "cpu":
        torch.cuda.is_available = lambda: False  # type: ignore[method-assign]


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


def read_reactions(
    args: argparse.Namespace,
    deduplicate_ids: bool = True,
) -> list[ReactionSetRecord]:
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
    raw_rows = 0

    for path_str in args.reactions:
        path = Path(path_str)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if (
                args.key_column not in reader.fieldnames
                or args.smiles_column not in reader.fieldnames
            ):
                raise KeyError(
                    f"{path} must contain columns {args.key_column!r} and "
                    f"{args.smiles_column!r}; found {reader.fieldnames}"
                )
            for row in reader:
                raw_rows += 1
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
                    if deduplicate_ids and reaction_id in seen:
                        continue
                    seen.add(reaction_id)
                    records.append(ReactionSetRecord(reaction_id, side_a, side_b))
                if raw_rows % 5000 == 0:
                    print(
                        f"Parsed {raw_rows} reaction rows into {len(records)} "
                        "reaction directions"
                    )
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


def load_chiro_params(params_path: str | Path) -> dict:
    with Path(params_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def molecule_cache_signature(args: argparse.Namespace) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model_name": "keiradams/ChIRo",
        "params_sha256": file_sha256(args.params),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "embedding_kind": str(args.embedding_kind),
        "seed": int(args.seed),
        "max_number_of_atoms": int(args.max_number_of_atoms),
        "max_embed_attempts": int(args.max_embed_attempts),
        "uff_iters": int(args.uff_iters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class MoleculeEmbeddingCache:
    """Small transactional store for resumable molecule-level ChIRo inference."""

    def __init__(self, path: str | Path, signature: str, read_only: bool = False):
        self.path = Path(path)
        self.signature = signature
        self.read_only = read_only
        if read_only:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.connection = sqlite3.connect(uri, uri=True, timeout=60.0)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=60.0)
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS molecules (
                    smiles TEXT PRIMARY KEY,
                    status INTEGER NOT NULL,
                    embedding_dim INTEGER,
                    embedding BLOB
                )
                """)
            self.connection.commit()
        self._validate_signature()

    def _validate_signature(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key = 'signature'"
            ).fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError(f"Invalid ChIRo molecule cache schema: {self.path}") from exc
        if row is None:
            if self.read_only:
                raise ValueError(f"ChIRo molecule cache has no signature: {self.path}")
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES('signature', ?)",
                (self.signature,),
            )
            self.connection.commit()
        elif row[0] != self.signature:
            raise ValueError(
                "ChIRo molecule cache was produced with incompatible model or conformer settings: "
                f"{self.path}"
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MoleculeEmbeddingCache":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def load(self, molecules: Iterable[str]) -> tuple[dict[str, np.ndarray], set[str]]:
        requested = list(dict.fromkeys(molecules))
        embeddings: dict[str, np.ndarray] = {}
        skipped: set[str] = set()
        for start in range(0, len(requested), 500):
            chunk = requested[start : start + 500]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = self.connection.execute(
                f"SELECT smiles, status, embedding_dim, embedding "
                f"FROM molecules WHERE smiles IN ({placeholders})",
                chunk,
            )
            for smiles, status, embedding_dim, blob in rows:
                if int(status) == 0:
                    skipped.add(str(smiles))
                    continue
                if blob is None or embedding_dim is None:
                    raise ValueError(f"Corrupt ChIRo cache entry for {smiles!r}")
                vector = np.frombuffer(blob, dtype="<f4").copy()
                if vector.size != int(embedding_dim):
                    raise ValueError(f"Corrupt ChIRo cache embedding for {smiles!r}")
                embeddings[str(smiles)] = vector
        return embeddings, skipped

    def put_embeddings(self, embeddings: dict[str, np.ndarray]) -> None:
        if self.read_only:
            raise PermissionError(f"ChIRo molecule cache is read-only: {self.path}")
        rows = []
        for smiles, vector in embeddings.items():
            array = np.asarray(vector, dtype="<f4").reshape(-1)
            rows.append((smiles, 1, int(array.size), sqlite3.Binary(array.tobytes())))
        self.connection.executemany(
            """
            INSERT INTO molecules(smiles, status, embedding_dim, embedding)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(smiles) DO UPDATE SET
                status=excluded.status,
                embedding_dim=excluded.embedding_dim,
                embedding=excluded.embedding
            """,
            rows,
        )
        self.connection.commit()

    def put_skipped(self, molecules: Iterable[str]) -> None:
        if self.read_only:
            raise PermissionError(f"ChIRo molecule cache is read-only: {self.path}")
        rows = [(smiles, 0, None, None) for smiles in molecules]
        self.connection.executemany(
            """
            INSERT INTO molecules(smiles, status, embedding_dim, embedding)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(smiles) DO UPDATE SET
                status=excluded.status,
                embedding_dim=NULL,
                embedding=NULL
            """,
            rows,
        )
        self.connection.commit()


def bounded_executor_results(
    executor: Executor,
    function: Callable[[T], R],
    items: Iterable[T],
    max_pending: int,
) -> Iterator[R]:
    """Yield executor results while retaining at most ``max_pending`` futures."""

    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    iterator = iter(items)
    pending: set[Future[R]] = set()

    def submit_one() -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(function, item))
        return True

    for _ in range(max_pending):
        if not submit_one():
            break
    while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
            pending.remove(future)
            result = future.result()
            submit_one()
            yield result


def build_chiro_model(
    params: dict, checkpoint_path: str | Path, device: torch.device
) -> torch.nn.Module:
    from model.alpha_encoder import Encoder
    from model.params_interpreter import string_to_object

    activation_dict = {
        key: string_to_object[value] for key, value in params["activation_dict"].items()
    }
    model = Encoder(
        F_z_list=params["F_z_list"],
        F_H=params["F_H"],
        F_H_embed=52,
        F_E_embed=14,
        F_H_EConv=params["F_H_EConv"],
        layers_dict=dict(params["layers_dict"]),
        activation_dict=activation_dict,
        GAT_N_heads=params["GAT_N_heads"],
        chiral_message_passing=params["chiral_message_passing"],
        CMP_EConv_MLP_hidden_sizes=params["CMP_EConv_MLP_hidden_sizes"],
        CMP_GAT_N_layers=params["CMP_GAT_N_layers"],
        CMP_GAT_N_heads=params["CMP_GAT_N_heads"],
        c_coefficient_normalization=params["c_coefficient_normalization"],
        encoder_reduction=params["encoder_reduction"],
        output_concatenation_mode=params["output_concatenation_mode"],
        EConv_bias=params["EConv_bias"],
        GAT_bias=params["GAT_bias"],
        encoder_biases=params["encoder_biases"],
        dropout=params["dropout"],
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(convert_chiro_state_dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ChIRo checkpoint did not load cleanly after compatibility conversion: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval().to(device)
    return model


def convert_chiro_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, value in state.items():
        if key == "Graph_Embedder.EConv.root":
            converted["Graph_Embedder.EConv.lin.weight"] = value.T.contiguous()
            continue
        new_key = key.replace(".att_l", ".att_src").replace(".att_r", ".att_dst")
        if ".lin_l.weight" in new_key:
            new_key = new_key.replace(".lin_l.weight", ".lin.weight")
        elif ".lin_r.weight" in new_key:
            continue
        converted[new_key] = value
    return converted


def smiles_to_chiro_data(smiles: str, args: argparse.Namespace):
    from torch_geometric.data import Data
    from model.embedding_functions import embedConformerWithAllPaths

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() > args.max_number_of_atoms:
        return None
    mol = Chem.AddHs(mol)
    embed_params = AllChem.ETKDGv3()
    embed_params.randomSeed = int(args.seed)
    embed_params.useRandomCoords = False
    rc = -1
    for attempt in range(max(1, int(args.max_embed_attempts))):
        embed_params.randomSeed = int(args.seed) + attempt
        try:
            rc = AllChem.EmbedMolecule(mol, embed_params)
        except Exception:
            rc = -1
        if rc == 0:
            break
        embed_params.useRandomCoords = True
    if rc != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=int(args.uff_iters))
    except Exception:
        pass
    try:
        (
            _,
            edge_index,
            edge_features,
            node_features,
            bond_distances,
            bond_distance_index,
            bond_angles,
            bond_angle_index,
            dihedral_angles,
            dihedral_angle_index,
        ) = embedConformerWithAllPaths(mol, repeats=False)
    except Exception:
        return None

    data = Data(
        x=torch.as_tensor(node_features),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_features),
    )
    data.bond_distances = torch.as_tensor(bond_distances)
    data.bond_distance_index = torch.as_tensor(bond_distance_index, dtype=torch.long).T
    data.bond_angles = torch.as_tensor(bond_angles)
    data.bond_angle_index = torch.as_tensor(bond_angle_index, dtype=torch.long).T
    data.dihedral_angles = torch.as_tensor(dihedral_angles)
    data.dihedral_angle_index = torch.as_tensor(dihedral_angle_index, dtype=torch.long).T

    if args.params_data.get("stereoMask", True):
        data.x[:, -9:] = 0.0
        data.edge_attr[:, -7:] = 0.0
    if args.params_data.get("mask_coordinates", False):
        data.bond_distances[:] = 0.0
        data.bond_angles[:] = 0.0
        data.dihedral_angles[:] = 0.0
    return data


def chiro_data_to_numpy_payload(data) -> dict[str, np.ndarray]:
    keys = (
        "x",
        "edge_index",
        "edge_attr",
        "bond_distances",
        "bond_distance_index",
        "bond_angles",
        "bond_angle_index",
        "dihedral_angles",
        "dihedral_angle_index",
    )
    return {key: getattr(data, key).detach().cpu().numpy() for key in keys}


def numpy_payload_to_chiro_data(payload: dict[str, np.ndarray]):
    from torch_geometric.data import Data

    data = Data(
        x=torch.as_tensor(payload["x"]),
        edge_index=torch.as_tensor(payload["edge_index"], dtype=torch.long),
        edge_attr=torch.as_tensor(payload["edge_attr"]),
    )
    data.bond_distances = torch.as_tensor(payload["bond_distances"])
    data.bond_distance_index = torch.as_tensor(payload["bond_distance_index"], dtype=torch.long)
    data.bond_angles = torch.as_tensor(payload["bond_angles"])
    data.bond_angle_index = torch.as_tensor(payload["bond_angle_index"], dtype=torch.long)
    data.dihedral_angles = torch.as_tensor(payload["dihedral_angles"])
    data.dihedral_angle_index = torch.as_tensor(payload["dihedral_angle_index"], dtype=torch.long)
    return data


def _init_chiro_worker(args: argparse.Namespace) -> None:
    global _WORKER_ARGS
    add_runtime_paths(args.chiro_root, args.extra_pythonpath)
    install_chiro_compatibility(args.device)
    _WORKER_ARGS = args


def _smiles_to_chiro_data_worker(smiles: str):
    if _WORKER_ARGS is None:
        raise RuntimeError("ChIRo worker was not initialized")
    try:
        data = smiles_to_chiro_data(smiles, _WORKER_ARGS)
        if data is None:
            return smiles, None
        return smiles, chiro_data_to_numpy_payload(data)
    except Exception:
        return smiles, None


def get_local_structure_map(psi_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local_structures: OrderedDict[tuple[int, int], int] = OrderedDict()
    ls_map = torch.zeros(psi_indices.shape[1], dtype=torch.long)
    next_idx = 0
    for i, indices in enumerate(psi_indices.T):
        key = (int(indices[1]), int(indices[2]))
        if key not in local_structures:
            local_structures[key] = next_idx
            next_idx += 1
        ls_map[i] = local_structures[key]
    alpha_indices = torch.zeros((2, len(local_structures)), dtype=torch.long)
    for i, key in enumerate(local_structures):
        alpha_indices[:, i] = torch.LongTensor(key)
    return ls_map, alpha_indices


def select_embedding(
    embedding_kind: str,
    latent_vector: torch.Tensor,
    z_alpha: torch.Tensor,
    mol_embedding: torch.Tensor,
) -> torch.Tensor:
    if embedding_kind == "both":
        return torch.cat([mol_embedding, latent_vector], dim=1)
    if embedding_kind == "molecule":
        return mol_embedding
    if embedding_kind == "conformer":
        return latent_vector
    if embedding_kind == "z_alpha":
        start = (latent_vector.shape[1] // 3) * 2
        return latent_vector[:, start:]
    raise ValueError(f"Unsupported embedding_kind: {embedding_kind}")


def encode_molecules(
    molecules: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[str], int]:
    from torch_geometric.loader import DataLoader

    cache: dict[str, np.ndarray] = {}
    skipped_set: set[str] = set()
    cache_path = getattr(args, "molecule_cache", None)
    cache_read_only = bool(getattr(args, "cache_read_only", False))
    cache_signature = molecule_cache_signature(args) if cache_path else None
    if cache_path:
        with MoleculeEmbeddingCache(
            cache_path,
            cache_signature,
            read_only=cache_read_only,
        ) as persistent_cache:
            cache, skipped_set = persistent_cache.load(molecules)
        print(
            f"Loaded {len(cache)} embeddings and {len(skipped_set)} skipped molecules "
            f"from {cache_path}",
            flush=True,
        )

    pending_molecules = [
        smiles for smiles in molecules if smiles not in cache and smiles not in skipped_set
    ]
    if cache_read_only and pending_molecules and not args.skip_uncached_molecules:
        preview = ", ".join(repr(smiles) for smiles in pending_molecules[:3])
        raise RuntimeError(
            f"Read-only ChIRo cache is missing {len(pending_molecules)} molecules; "
            f"first missing entries: {preview}"
        )

    embedding_dims = {int(vector.size) for vector in cache.values()}
    if len(embedding_dims) > 1:
        raise ValueError(
            f"ChIRo molecule cache contains mixed embedding dimensions: {embedding_dims}"
        )
    embedding_dim: int | None = next(iter(embedding_dims), None)
    if cache_read_only and args.skip_uncached_molecules:
        skipped_set.update(pending_molecules)
        if embedding_dim is None:
            raise RuntimeError("No cached ChIRo molecule embeddings were produced")
        print(f"Omitting {len(pending_molecules)} uncached ChIRo molecules", flush=True)
        return cache, [smiles for smiles in molecules if smiles in skipped_set], embedding_dim
    if not pending_molecules:
        if embedding_dim is None:
            raise RuntimeError("No ChIRo molecule embeddings were produced")
        skipped = [smiles for smiles in molecules if smiles in skipped_set]
        return cache, skipped, embedding_dim

    device = torch.device(args.device)
    model = build_chiro_model(args.params_data, args.checkpoint, device=device)
    data_batch = []
    smiles_batch = []

    def persist_embeddings(new_embeddings: dict[str, np.ndarray]) -> None:
        if not cache_path:
            return
        with MoleculeEmbeddingCache(cache_path, cache_signature) as persistent_cache:
            persistent_cache.put_embeddings(new_embeddings)

    def persist_skipped(smiles: str) -> None:
        if not cache_path:
            return
        with MoleculeEmbeddingCache(cache_path, cache_signature) as persistent_cache:
            persistent_cache.put_skipped((smiles,))

    def flush() -> None:
        nonlocal data_batch, smiles_batch, embedding_dim
        if not data_batch:
            return
        loader = DataLoader(data_batch, batch_size=len(data_batch), shuffle=False)
        batch = next(iter(loader)).to(device)
        ls_map, alpha_indices = get_local_structure_map(batch.dihedral_angle_index)
        ls_map = ls_map.to(device)
        alpha_indices = alpha_indices.to(device)
        with torch.inference_mode():
            result = model(batch, ls_map, alpha_indices)
            if len(result) == 10:
                _, latent_vector, _, z_alpha, mol_embedding, *_ = result
            else:
                latent_vector, _, z_alpha, mol_embedding, *_ = result
            embeddings = select_embedding(
                args.embedding_kind,
                latent_vector=latent_vector,
                z_alpha=z_alpha,
                mol_embedding=mol_embedding,
            )
        embeddings_np = embeddings.detach().cpu().float().numpy()
        embedding_dim = int(embeddings_np.shape[1])
        new_embeddings = {}
        for smiles, vector in zip(smiles_batch, embeddings_np):
            cache[smiles] = vector
            new_embeddings[smiles] = vector
        persist_embeddings(new_embeddings)
        data_batch = []
        smiles_batch = []

    resolved_before_run = len(cache) + len(skipped_set)

    def handle_preprocessed(idx: int, smiles: str, data) -> None:
        nonlocal data_batch, smiles_batch
        if data is None:
            if not args.skip_invalid_molecules:
                raise ValueError(f"ChIRo preprocessing failed for molecule {smiles}")
            if args.log_skipped_molecules:
                logging.warning("Skipping molecule after ChIRo preprocessing failure: %s", smiles)
            skipped_set.add(smiles)
            persist_skipped(smiles)
        else:
            if isinstance(data, dict):
                data = numpy_payload_to_chiro_data(data)
            data_batch.append(data)
            smiles_batch.append(smiles)
        if len(data_batch) >= args.batch_size:
            flush()
        if idx % 1000 == 0:
            completed = resolved_before_run + idx
            print(
                f"Prepared/encoded {completed}/{len(molecules)} molecules; "
                f"skipped={len(skipped_set)}",
                flush=True,
            )

    if int(args.num_workers) > 0:
        max_pending = int(getattr(args, "max_pending_tasks", 0)) or max(
            2 * int(args.num_workers), 1
        )
        print(
            f"Preprocessing {len(pending_molecules)} uncached molecules with "
            f"{args.num_workers} workers and at most {max_pending} pending tasks",
            flush=True,
        )
        with ProcessPoolExecutor(
            max_workers=int(args.num_workers),
            initializer=_init_chiro_worker,
            initargs=(args,),
        ) as executor:
            results = bounded_executor_results(
                executor,
                _smiles_to_chiro_data_worker,
                pending_molecules,
                max_pending=max_pending,
            )
            for idx, (smiles, data) in enumerate(results, start=1):
                handle_preprocessed(idx, smiles, data)
    else:
        for idx, smiles in enumerate(pending_molecules, start=1):
            data = smiles_to_chiro_data(smiles, args)
            handle_preprocessed(idx, smiles, data)

    flush()
    if embedding_dim is None:
        raise RuntimeError("No ChIRo molecule embeddings were produced")
    skipped = [smiles for smiles in molecules if smiles in skipped_set]
    return cache, skipped, embedding_dim


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
        if not skip_invalid_molecules and (
            len(reactants) != len(record.reactants) or len(products) != len(record.products)
        ):
            missing = sorted(set((*record.reactants, *record.products)) - set(molecule_cache))
            raise KeyError(f"Missing ChIRo embeddings for reaction {record.reaction_id}: {missing}")
        if not reactants or not products:
            if skip_invalid_reactions:
                continue
            raise ValueError(
                f"Reaction {record.reaction_id} has an empty side after ChIRo filtering"
            )
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
    embedding_dim: int,
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
        handle.attrs["embedding_dim"] = int(embedding_dim)
        handle.attrs["reaction_representation"] = "chiro_chirality"
        handle.attrs["model_name"] = "keiradams/ChIRo"
        handle.attrs["params"] = str(args.params)
        handle.attrs["checkpoint"] = str(args.checkpoint)
        handle.attrs["embedding_kind"] = str(args.embedding_kind)
        handle.attrs["deterministic_seed"] = int(args.seed)
        handle.attrs["num_skipped_molecules"] = int(len(skipped_molecules))
        handle.attrs["num_reactions"] = int(len(records))
        if args.molecule_cache:
            handle.attrs["molecule_cache"] = str(Path(args.molecule_cache).resolve())
            handle.attrs["molecule_cache_signature"] = molecule_cache_signature(args)


def main() -> None:
    args = parse_args()
    if args.output and Path(args.output).exists() and not args.force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {args.output}")
    add_runtime_paths(args.chiro_root, args.extra_pythonpath)
    install_chiro_compatibility(args.device)
    args.params_data = load_chiro_params(args.params)
    records = read_reactions(args, deduplicate_ids=not args.cache_only)
    print(f"Loaded {len(records)} reaction directions", flush=True)
    molecules = ordered_unique_molecules(records)
    print(f"Encoding {len(molecules)} unique molecules with ChIRo", flush=True)
    molecule_cache, skipped, embedding_dim = encode_molecules(molecules, args)
    print(
        f"Encoded {len(molecule_cache)} molecules; skipped {len(skipped)}",
        flush=True,
    )
    if args.cache_only:
        print(f"ChIRo molecule cache is complete: {args.molecule_cache}", flush=True)
        return
    records = filter_records(
        records,
        molecule_cache=molecule_cache,
        skip_invalid_molecules=args.skip_invalid_molecules,
        skip_invalid_reactions=args.skip_invalid_reactions,
    )
    print(
        f"Keeping {len(records)} reaction directions after ChIRo molecule filtering",
        flush=True,
    )
    write_hdf5(args.output, records, molecule_cache, skipped, embedding_dim, args)
    print(f"Wrote ChIRo reaction embeddings to {args.output}", flush=True)


if __name__ == "__main__":
    main()
