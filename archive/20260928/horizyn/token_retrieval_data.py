"""Released ReactZyme inputs and deterministic, shared frozen-feature caches."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ESMC_REPO = "biohub/esmc-600m-2024-12"
ESMC_REVISION = "e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def truncate_sequence(sequence, limit=1022):
    sequence = re.sub(r"\s+", "", sequence.upper())
    sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "X", sequence)
    if not sequence:
        raise ValueError("Empty protein sequence")
    if len(sequence) <= limit:
        return sequence
    first = last = limit // 4
    middle = limit - first - last
    start = max((len(sequence) - middle) // 2, first)
    end = min(start + middle, len(sequence) - last)
    start = max(end - middle, first)
    return sequence[:first] + sequence[start:end] + (sequence[-last:] if last else "")


class PairRows:
    """Metadata facade accepted by the existing TypedNegativeBatchSampler."""
    def __init__(self, rows):
        self.rows = rows
        self.keys = list(range(len(rows)))
        self.tuple_dataset = self

    def __getitem__(self, index):
        return self.rows[index]

    def __len__(self):
        return len(self.rows)


class ReactionSMIProtocol:
    def __init__(self, split_dir, negative_pool, include_test=False):
        self.split_dir = Path(split_dir)
        self.sequences, self.smiles, self.pairs, self.candidates = {}, {}, {}, {}
        for split in ["train", "validation"] + (["test"] if include_test else []):
            with (self.split_dir / f"{split}_pairs.csv").open() as handle:
                pairs = list(csv.DictReader(handle))
            self.pairs[split] = []
            for row in pairs:
                q, e = row["reaction_id"], row["protein_id"]
                if e in self.sequences and self.sequences[e] != row["protein_sequence"]:
                    raise ValueError(f"Conflicting sequences for {e}")
                self.sequences[e] = row["protein_sequence"]
                self.pairs[split].append((q, e))
            with (self.split_dir / f"{split}_rxns.csv").open() as handle:
                for row in csv.DictReader(handle):
                    q, smi = row["reaction_id"], row["reaction_smiles"]
                    if q in self.smiles and self.smiles[q] != smi:
                        raise ValueError(f"Conflicting chemistry for {q}")
                    self.smiles[q] = smi
            ids = (self.split_dir / f"{split}_candidate_ids.txt").read_text().splitlines()
            self.candidates[split] = [x.strip() for x in ids if x.strip()]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {split} candidates")
        for split, ids in self.candidates.items():
            missing = set(ids) - self.sequences.keys()
            if missing:
                raise ValueError(f"Missing sequences for {len(missing)} {split} candidates")
        self.positives = defaultdict(set)
        for q, e in self.pairs["train"]:
            self.positives[q].add(e)

        from horizyn.reaction_conditioned_data_module import _load_typed_negative_map
        pools = _load_typed_negative_map(negative_pool)
        self.typed_pools = {}
        rows = [{"query_id": q, "target_id": e, "pair_type": "positive"}
                for q, e in self.pairs["train"]]
        for q, pool in pools.items():
            if q not in self.positives:
                continue
            self.typed_pools[q] = pool
            for category, kind in [("biological", "biological_negative"), ("random", "random_negative")]:
                for e in pool[category]:
                    if e not in self.sequences:
                        raise ValueError(f"Negative {e} has no sequence; do not silently change the pool")
                    if e in self.positives[q]:
                        raise ValueError(f"Known positive in negative pool: {q}, {e}")
                    rows.append({"query_id": q, "target_id": e, "pair_type": kind})
        self.training_rows = PairRows(rows)

    def masks(self, queries, targets, device):
        """Match CIRCE: all known positives and all typed pool entries in batch."""
        lookup = {e: i for i, e in enumerate(targets)}
        masks = [torch.zeros((len(queries), len(targets)), dtype=torch.bool) for _ in range(3)]
        for i, q in enumerate(queries):
            sets = [self.positives[q], self.typed_pools.get(q, {}).get("biological", []),
                    self.typed_pools.get(q, {}).get("random", [])]
            for mask, values in zip(masks, sets):
                cols = [lookup[e] for e in values if e in lookup]
                if cols:
                    mask[i, cols] = True
        if bool((masks[0] & (masks[1] | masks[2])).any()) or bool((masks[1] & masks[2]).any()):
            raise ValueError("Positive and typed-negative masks overlap")
        return tuple(x.to(device) for x in masks)

    def manifest(self):
        return {
            "split": "reaction_smi", "pairs": {k: len(v) for k, v in self.pairs.items()},
            "candidates": {k: len(v) for k, v in self.candidates.items()},
            "training_rows_with_negative_pool": len(self.training_rows),
            "max_protein_tokens": 1022, "protein_truncation": "ends_center",
            "truncated_proteins": sum(len(s) > 1022 for s in self.sequences.values()),
        }


class MoleculeStore:
    def __init__(self, graph_builder=None):
        self.graphs, self.reactions = {}, {}
        self.graph_builder = graph_builder

    def reaction(self, smiles):
        from rdkit import Chem

        if smiles in self.reactions:
            return self.reactions[smiles]
        if ">" in smiles:
            raise ValueError("This benchmark adapter expects released molecule sets, not directed reactions")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Cannot parse released molecular input: {smiles[:160]}")
        components = []
        for component in Chem.GetMolFrags(mol, asMols=True):
            for atom in component.GetAtoms():
                atom.SetAtomMapNum(0)
            canonical = Chem.MolToSmiles(component, canonical=True, isomericSmiles=True)
            if canonical not in self.graphs:
                component = Chem.MolFromSmiles(canonical)
                if self.graph_builder is not None:
                    graph = self.graph_builder(component, canonical)
                else:
                    from rxnzyme.data.reaction import build_mol_graph, get_mol_features
                    graph = build_mol_graph([component], *get_mol_features([component]))
                    graph.spatial_pos = torch.as_tensor(Chem.GetDistanceMatrix(component).copy(), dtype=torch.long)
                    # Direct bond bias does not use the optional edge-path tensor.
                    graph.path_indices = torch.empty((5, 0), dtype=torch.long)
                    graph.canonical_smiles = canonical
                self.graphs[canonical] = graph
            components.append(canonical)
        if not components:
            raise ValueError("Empty molecular set")
        # Sorting makes enumeration deterministic while retaining multiplicity.
        result = [self.graphs[key] for key in sorted(components)]
        self.reactions[smiles] = result
        return result


class FrozenResidueStore:
    """On-demand FP32 ESM-C extraction with atomic files shared by both jobs.

    Frozen weights live outside the trainable model and its checkpoints. Cache
    keys include the exact normalized input and pinned foundation revision.
    Concurrent identical computations may occur, but readers never see a
    partial file. No learned features or association labels enter this cache.
    """
    def __init__(self, cache_dir, checkpoint_path, device, extraction_tokens=8192):
        self.root = Path(cache_dir) / f"esmc600_{ESMC_REVISION[:12]}_fp32_endscenter1022_v1"
        self.root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.extraction_tokens = extraction_tokens
        self.model = None
        self.extracted, self.cache_hits = 0, 0

    def _path(self, sequence):
        key = hashlib.sha256(sequence.encode()).hexdigest()
        return self.root / key[:2] / f"{key}.npy"

    def _load_model(self):
        if self.model is not None:
            return
        from esm.models.esmc import ESMC
        from esm.tokenization import get_esmc_model_tokenizers
        print(f"Loading frozen ESM-C 600M on {self.device}", flush=True)
        # Model initialization must not consume the trainable model's dropout RNG.
        with torch.random.fork_rng(devices=[self.device.index]):
            model = ESMC(d_model=1152, n_heads=18, n_layers=36,
                         tokenizer=get_esmc_model_tokenizers(), use_flash_attn=False)
        weights = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        self.model = model.to(self.device, dtype=torch.float32).eval().requires_grad_(False)

    def get_many(self, sequences):
        normalized = [truncate_sequence(s) for s in sequences]
        missing = sorted({s for s in normalized if not self._path(s).is_file()}, key=len)
        self.cache_hits += len(normalized) - len(missing)
        if missing:
            self._load_model()
        cursor = 0
        reported = 0
        while cursor < len(missing):
            stop = cursor + 1
            while stop < len(missing) and (stop - cursor + 1) * (len(missing[stop]) + 2) <= self.extraction_tokens:
                stop += 1
            batch = missing[cursor:stop]
            tokens = self.model.tokenizer(batch, padding=True, return_tensors="pt")["input_ids"].to(self.device)
            with torch.inference_mode():
                embeddings = self.model(tokens).embeddings
                for i, sequence in enumerate(batch):
                    array = embeddings[i, 1:len(sequence) + 1].float().cpu().numpy().copy()
                    if array.shape != (len(sequence), 1152) or not np.isfinite(array).all():
                        raise ValueError("Invalid frozen residue features")
                    path = self._path(sequence)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
                    try:
                        with os.fdopen(fd, "wb") as handle:
                            np.save(handle, array, allow_pickle=False)
                        os.replace(temporary, path)
                    finally:
                        if os.path.exists(temporary):
                            os.unlink(temporary)
                    self.extracted += 1
            del embeddings, tokens
            cursor = stop
            if cursor - reported >= 256 or cursor == len(missing):
                print(f"[{self.device}] frozen features {cursor}/{len(missing)} new proteins", flush=True)
                reported = cursor
        return [torch.from_numpy(np.load(self._path(s), allow_pickle=False)) for s in normalized]
