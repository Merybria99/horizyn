"""Pinned PCQM4Mv2 Graphormer and Microsoft's original OGB preprocessing.

The HF checkpoint is a conversion of Microsoft's molecular-property model.
Only its graph encoder supplies atom representations; the property head is
discarded after a strict load of the complete released checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

GRAPHORMER_REPO = "clefourrier/graphormer-base-pcqm4mv2"
GRAPHORMER_REVISION = "662d78215c8c8b2313837765b4b7f0d98b594a7e"
GRAPHORMER_WEIGHTS_SHA256 = "fc92119f232e1918df3d342a51f4523d6354c9b7f1d4819e2915cc66e0ad73cc"
GRAPHORMER_CONFIG_SHA256 = "fe33c24a69f16f6198320f1633e30b2cfaee605df800cfb4e1dd7e1327c3d80c"
PREPROCESSING_ID = "microsoft_ogb136_bounded_paths_v1"
UPSTREAM_REVISION = "a04573c40705fb174db261bb746a8258d00992f5"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PretrainedMolecularGraphormer(nn.Module):
    def __init__(self, checkpoint_dir):
        super().__init__()
        from transformers.models.deprecated.graphormer.configuration_graphormer import GraphormerConfig
        from transformers.models.deprecated.graphormer.modeling_graphormer import GraphormerForGraphClassification

        if checkpoint_dir is None:
            raise ValueError("Pretrained Graphormer requires an explicit local checkpoint directory")
        directory = Path(checkpoint_dir)
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        for filename, expected in [("config.json", GRAPHORMER_CONFIG_SHA256),
                                   ("pytorch_model.bin", GRAPHORMER_WEIGHTS_SHA256)]:
            if file_sha256(directory / filename) != expected:
                raise ValueError(f"Unexpected pretrained Graphormer file: {directory / filename}")
        self.config = GraphormerConfig.from_json_file(directory / "config.json")
        complete = GraphormerForGraphClassification(self.config)
        weights = torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
        complete.load_state_dict(weights, strict=True)
        self.encoder = complete.encoder.graph_encoder
        # Bitwise equality catches accidental initialization after loading.
        for name, value in self.encoder.state_dict().items():
            if not torch.equal(value, weights["encoder.graph_encoder." + name]):
                raise ValueError(f"Pretrained tensor was not preserved: {name}")
        self.output_dim = self.config.embedding_dim
        self.finetuning = True
        self.provenance = {
            "repo": GRAPHORMER_REPO, "revision": GRAPHORMER_REVISION,
            "weights_sha256": GRAPHORMER_WEIGHTS_SHA256,
            "config_sha256": GRAPHORMER_CONFIG_SHA256,
            "checkpoint_dir": str(directory.resolve()), "strict_load": True,
            "encoder_tensors_loaded": len(self.encoder.state_dict()),
            "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "discarded_property_head_keys": [k for k in weights if not k.startswith("encoder.graph_encoder.")],
            "preprocessing": PREPROCESSING_ID, "upstream_revision": UPSTREAM_REVISION,
            "layers": self.config.num_hidden_layers, "width": self.output_dim,
            "heads": self.config.num_attention_heads,
            "edge_type": self.config.edge_type, "multi_hop_max_dist": self.config.multi_hop_max_dist,
            "spatial_pos_max": self.config.spatial_pos_max,
        }

    def set_trainable(self, enabled):
        self.finetuning = bool(enabled)
        self.requires_grad_(self.finetuning)
        self.train(self.training)

    def train(self, mode=True):
        return super().train(bool(mode) and self.finetuning)

    def forward(self, batch):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.finetuning):
            states, graph = self.encoder(**batch, last_state_only=True)
        # The leading graph token is not an atom and must not enter the token bank.
        return states[-1].transpose(0, 1)[:, 1:, :], graph


@dataclass
class PretrainedMoleculeGraph:
    canonical_smiles: str
    num_nodes: int
    features: dict[str, np.ndarray]


@lru_cache(maxsize=1)
def graph_algorithms():
    import pyximport
    pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)
    from . import _graphormer_algos
    return _graphormer_algos


class PretrainedGraphBuilder:
    """Label-free graph features; shared cache holds no learned representations."""
    def __init__(self, config, cache_dir=None):
        self.config = config
        self.cache_dir = None
        if cache_dir is not None:
            from rdkit import rdBase
            tag = f"{PREPROCESSING_ID}_rdkit{rdBase.rdkitVersion}_h{config.multi_hop_max_dist}"
            self.cache_dir = Path(cache_dir) / tag
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, mol, canonical_smiles):
        n = mol.GetNumAtoms()
        if not 0 < n <= self.config.max_nodes:
            raise ValueError(f"Graphormer requires 1..{self.config.max_nodes} atoms per component; got {n}")
        path = None
        if self.cache_dir is not None:
            key = hashlib.sha256(canonical_smiles.encode()).hexdigest()
            path = self.cache_dir / key[:2] / (key + ".npz")
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    features = {k: saved[k] for k in saved.files}
                if features["x"].shape != (n, 9):
                    raise ValueError(f"Invalid Graphormer feature cache: {path}")
                return PretrainedMoleculeGraph(canonical_smiles, n, features)

        from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
        node_features = np.asarray([atom_to_feature_vector(a) for a in mol.GetAtoms()], dtype=np.int64)
        # Exactly one feature-field offset here and one padding shift in collate.
        x = node_features + 1 + 512 * np.arange(9, dtype=np.int64)
        adjacency = np.zeros((n, n), dtype=bool)
        edge_type = np.zeros((n, n, 3), dtype=np.int64)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[i, j] = adjacency[j, i] = True
            values = np.asarray(bond_to_feature_vector(bond), dtype=np.int64) + 2 + 512 * np.arange(3)
            edge_type[i, j] = edge_type[j, i] = values
        algos = graph_algorithms()
        distance, paths = algos.floyd_warshall(adjacency)
        if int(distance.max()) + 1 >= self.config.num_spatial:
            raise ValueError("Molecular graph exceeds the pretrained spatial vocabulary")
        # One zero-filled hop makes isolated atoms valid alone as well as in a
        # mixed batch. Other paths match upstream's full tensor sliced to five.
        hops = max(1, min(int(distance.max()), self.config.multi_hop_max_dist))
        edges = algos.gen_edge_input(hops, paths, edge_type)
        degree = adjacency.sum(axis=1).astype(np.int64)
        features = {"x": x, "attn_edge_type": edge_type,
                    "spatial_pos": distance, "in_degree": degree, "edge_input": edges}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as handle:
                    np.savez_compressed(handle, **features)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return PretrainedMoleculeGraph(canonical_smiles, n, features)


def collate_pretrained_graphs(graphs, config):
    """Microsoft padding/masking semantics, without silently filtering graphs."""
    if not graphs:
        raise ValueError("Cannot collate an empty molecular batch")
    if any(not 0 < g.num_nodes <= config.max_nodes for g in graphs):
        raise ValueError("A molecular component exceeds the pretrained node limit")
    count, n = len(graphs), max(g.num_nodes for g in graphs)
    hops = max(g.features["edge_input"].shape[2] for g in graphs)
    batch = {
        "input_nodes": torch.zeros((count, n, 9), dtype=torch.long),
        "input_edges": torch.zeros((count, n, n, hops, 3), dtype=torch.long),
        "attn_bias": torch.full((count, n + 1, n + 1), float("-inf")),
        "attn_edge_type": torch.zeros((count, n, n, 3), dtype=torch.long),
        "spatial_pos": torch.zeros((count, n, n), dtype=torch.long),
        "in_degree": torch.zeros((count, n), dtype=torch.long),
    }
    for index, graph in enumerate(graphs):
        f, size = graph.features, graph.num_nodes
        batch["input_nodes"][index, :size] = torch.from_numpy(f["x"] + 1)
        batch["spatial_pos"][index, :size, :size] = torch.from_numpy(f["spatial_pos"] + 1)
        batch["attn_edge_type"][index, :size, :size] = torch.from_numpy(f["attn_edge_type"])
        batch["in_degree"][index, :size] = torch.from_numpy(f["in_degree"] + 1)
        edge_hops = f["edge_input"].shape[2]
        batch["input_edges"][index, :size, :size, :edge_hops] = torch.from_numpy(f["edge_input"] + 1)
        bias = batch["attn_bias"][index]
        bias[:size + 1, :size + 1] = 0
        bias[size + 1:, :size + 1] = 0
        bias[1:size + 1, 1:size + 1].masked_fill_(
            torch.from_numpy(f["spatial_pos"] >= config.spatial_pos_max), float("-inf"))
    batch["out_degree"] = batch["in_degree"]
    return batch
