#!/usr/bin/env python3
"""Check every released molecular input and prime label-free graph caches."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import yaml

from horizyn.pretrained_graphormer import PretrainedGraphBuilder, file_sha256
from horizyn.token_retrieval import TokenModelConfig, TokenRetrievalModel
from horizyn.token_retrieval_data import MoleculeStore, atomic_json
from horizyn.token_retrieval_evaluation import evaluator_provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/reactzyme_reaction_smi_b1_esmc_graphormer.yaml"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])
    model = TokenRetrievalModel(TokenModelConfig(**cfg["model"]), cfg["variant"]).eval()
    graph_cfg = model.graph_encoder.config
    store = MoleculeStore(PretrainedGraphBuilder(graph_cfg, cfg["graph_feature_cache"]))
    inputs, by_split, hashes = {}, {}, {}
    for split in ["train", "validation", "test"]:
        path = ROOT / cfg["data"]["split_dir"] / f"{split}_rxns.csv"
        hashes[str(path)] = file_sha256(path)
        rows = list(csv.DictReader(path.open()))
        by_split[split] = len(rows)
        for row in rows:
            identifier, smiles = row["reaction_id"], row["reaction_smiles"]
            if identifier in inputs and inputs[identifier] != smiles:
                raise ValueError(f"Conflicting chemistry for {identifier}")
            inputs[identifier] = smiles
    started = time.monotonic()
    for index, smiles in enumerate(inputs.values(), 1):
        store.reaction(smiles)
        if index % 250 == 0:
            print(json.dumps({"checked_reactions": index, "total": len(inputs),
                              "molecules": len(store.graphs), "elapsed_seconds": time.monotonic() - started}), flush=True)
    for graph in store.graphs.values():
        f = graph.features
        if int(f["x"].max()) + 1 > graph_cfg.num_atoms:
            raise ValueError("Atom vocabulary exceeded")
        if int(f["edge_input"].max()) + 1 > graph_cfg.num_edges:
            raise ValueError("Bond vocabulary exceeded")
        if int(f["in_degree"].max()) + 1 >= graph_cfg.num_in_degree:
            raise ValueError("Degree vocabulary exceeded")
    sizes = sorted(g.num_nodes for g in store.graphs.values())
    largest = max(store.graphs.values(), key=lambda g: g.num_nodes)
    model.to(args.device)
    with torch.no_grad():
        representations = model.encode_reactions([[largest], store.reaction("[Na+]"), store.reaction("CCO.O")])
    if not all(torch.isfinite(x).all() for x in representations):
        raise FloatingPointError("Non-finite pretrained reaction features")
    initial_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().numpy().tobytes())
    report = {"pretrained_graphormer": model.graph_encoder.provenance,
              "reaction_rows": by_split, "unique_reactions": len(inputs),
              "unique_molecular_components": len(sizes), "max_component_atoms": max(sizes),
              "p99_component_atoms": int(np.quantile(sizes, .99)), "total_atoms": sum(sizes),
              "dropped_inputs": 0, "all_feature_vocabularies_valid": True,
              "largest_component_and_isolated_atom_forward_finite": True,
              "initial_model_sha256": initial_hash.hexdigest(),
              "total_retrieval_parameters": sum(p.numel() for p in model.parameters()),
              "pooling_parameters": sum(p.numel() for p in model.protein_pool.parameters())
                                    + sum(p.numel() for p in model.reaction_pool.parameters()),
              "input_sha256": hashes, "evaluation": evaluator_provenance(),
              "graph_cache": str(store.graph_builder.cache_dir),
              "elapsed_seconds": time.monotonic() - started,
              "scope": "Input/weight checks only; no retrieval test predictions or test-label tuning"}
    atomic_json(output / "input_and_weights_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
