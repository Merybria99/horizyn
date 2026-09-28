#!/usr/bin/env python3
"""Replay the released CLIPZyme checkpoint on the official screening queries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from cyp_external_common import safe_torch, source


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--screening-pickle", type=Path, required=True)
    parser.add_argument("--cached-enzymemap", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch = safe_torch()
    source("clipzyme_pretrained")
    from clipzyme.models.protmol import EnzymeReactionCLIP
    from clipzyme.utils.screening import process_mapped_reaction
    from torch_geometric.data import Batch
    from torch_geometric.inspector import Inspector

    if not hasattr(Inspector, "distribute"):
        Inspector.distribute = Inspector.collect_param_data
    torch.set_num_threads(4)

    manifest = json.loads(args.manifest.read_text())
    source_info = manifest["associations"]["test"]
    test_path = Path(source_info["path"])
    if (sha256(test_path) != source_info["sha256"]
            or sha256(args.cached_enzymemap) != manifest["source_cache_sha256"]):
        raise ValueError("Official EnzymeMap test source changed")
    with test_path.open(newline="") as handle:
        test_rows = list(csv.DictReader(handle))
    first_index = {}
    for row in test_rows:
        first_index.setdefault(row["reaction"], int(row["source_index"]))
    with (args.protocol / "query_inputs.csv").open(newline="") as handle:
        queries = list(csv.DictReader(handle))
    if len(queries) != 1521 or any(row["reaction"] not in first_index for row in queries):
        raise ValueError("Official screening queries do not match the test associations")
    with args.cached_enzymemap.open("rb") as handle:
        cached = pickle.load(handle)
    if len(cached) != 46356:
        raise ValueError("Official cached EnzymeMap count changed")
    with args.screening_pickle.open("rb") as handle:
        screening = pickle.load(handle)
    candidate_ids = screening["uniprots"]
    protein = screening["hiddens"]
    if (len(candidate_ids) != 261907 or len(set(candidate_ids)) != len(candidate_ids)
            or tuple(protein.shape) != (261907, 1280)
            or not torch.isfinite(protein).all()):
        raise ValueError("Released screening protein features changed")
    with (args.protocol.parent / "clipzyme_f3_catalog_v1/screening_candidate_map.csv").open(newline="") as handle:
        catalog_ids = [row["uniprot_id"] for row in csv.DictReader(handle)]
    if candidate_ids != catalog_ids:
        raise ValueError("Released CLIPZyme/F3 candidate ordering differs")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_args = checkpoint["hyper_parameters"]["args"]
    if checkpoint.get("epoch") != 26 or checkpoint_args.experiment_name != "bf6b607124c5cca3430fc0c2ee1148dd":
        raise ValueError("Expected released CLIPZyme EnzymeMap checkpoint")
    model = EnzymeReactionCLIP(checkpoint_args).eval()
    model.load_state_dict({key.removeprefix("model."): value
                           for key, value in checkpoint["state_dict"].items()}, strict=True)
    model.args.use_as_mol_encoder = True
    model.to(args.device)
    del checkpoint

    args.output.mkdir(parents=True)
    query_vectors = np.empty((1521, 1280), dtype=np.float32)
    with torch.inference_mode():
        for i, row in enumerate(queries):
            item = cached[first_index[row["reaction"]]]
            if item["reaction_string"] != row["reaction"]:
                raise ValueError(f"Mapped test reaction mismatch at query {i}")
            mapped = ".".join(item["reactants"]) + ">>" + ".".join(item["products"])
            reactants, products = process_mapped_reaction(
                mapped, bond_changes=item["bond_changes"],
                use_one_hot_mol_features=checkpoint_args.use_one_hot_mol_features)
            reaction_nodes = torch.zeros(reactants.x.shape[0])
            for left, right, _ in reactants.bond_changes:
                reaction_nodes[left] = 1
                reaction_nodes[right] = 1
            reactants.reaction_nodes = reaction_nodes
            batch = {"reactants": Batch.from_data_list([reactants]).to(args.device),
                     "products": Batch.from_data_list([products]).to(args.device)}
            hidden = model(batch)["hidden"]
            if hidden.shape != (1, 1280) or not torch.isfinite(hidden).all():
                raise ValueError(f"Invalid released CLIPZyme reaction embedding at {i}")
            query_vectors[i] = hidden[0].float().cpu().numpy()
            if (i + 1) % 100 == 0:
                print(json.dumps({"encoded_queries": i + 1}), flush=True)
    np.save(args.output / "reaction_embeddings.npy", query_vectors)
    (args.output / "query_ids.txt").write_text(
        "\n".join(row["reaction_id"] for row in queries) + "\n")
    (args.output / "candidate_ids.txt").write_text("\n".join(candidate_ids) + "\n")
    target = protein.to(args.device, dtype=torch.float32)
    scores_path = args.output / "scores.npy"
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32,
                                       shape=(1521, 261907))
    with torch.inference_mode():
        for start in range(0, 1521, args.score_batch_size):
            end = min(start + args.score_batch_size, 1521)
            query = torch.from_numpy(query_vectors[start:end]).to(args.device)
            scores[start:end] = (query @ target.T).float().cpu().numpy()
            scores.flush()
    del scores
    receipt = {"schema": "clipzyme_released_checkpoint_screen_v1",
               "checkpoint_sha256": sha256(args.checkpoint),
               "screening_pickle_sha256": sha256(args.screening_pickle),
               "cached_enzymemap_sha256": sha256(args.cached_enzymemap),
               "manifest_sha256": sha256(args.manifest),
               "query_inputs_sha256": sha256(args.protocol / "query_inputs.csv"),
               "query_ids_sha256": sha256(args.output / "query_ids.txt"),
               "candidate_ids_sha256": sha256(args.output / "candidate_ids.txt"),
               "scores_sha256": sha256(scores_path), "scores_shape": [1521, 261907],
               "labels_read": False, "source_sha256": sha256(Path(__file__))}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"scores": str(scores_path), "shape": receipt["scores_shape"]}), flush=True)


if __name__ == "__main__":
    main()
