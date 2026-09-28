#!/usr/bin/env python3
"""Check distributed representation/score ordering with CPU-only ID-coded inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("token_trainer", Path(__file__).with_name("train_token_retrieval.py"))
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    cfg = SimpleNamespace(global_dim=2, local_dim=2)

    def represent(ids):
        numbers = torch.tensor([int(identifier[1:]) for identifier in ids], dtype=torch.float32)
        global_vectors = torch.stack([numbers, torch.ones_like(numbers)], dim=1)
        tokens = numbers[:, None, None].expand(-1, 3, 2).clone()
        return global_vectors, tokens

    # Unequal shard sizes, stable ID restoration, and complete score-grid
    # assembly follow the exact helper calls used by the live evaluator.
    queries = [f"q{i}" for i in range(7)]
    enzymes = [f"e{i}" for i in range(11)]
    query, qids = trainer.global_representations(represent(queries[rank::world]), queries[rank::world], 3, cfg, False)
    enzyme, eids = trainer.global_representations(represent(enzymes[rank::world]), enzymes[rank::world], 3, cfg, False)
    qorder = torch.tensor([qids.index(i) for i in queries])
    eorder = torch.tensor([eids.index(i) for i in enzymes])
    query = tuple(x[qorder] for x in query)
    enzyme = tuple(x[eorder] for x in enzyme)
    torch.testing.assert_close(query[0], represent(queries)[0], rtol=0, atol=0)
    torch.testing.assert_close(enzyme[1], represent(enzymes)[1], rtol=0, atol=0)
    local_scores = query[0][rank::world] @ enzyme[0].T
    rows = trainer.gather_tensor(local_scores, [len(queries[r::world]) for r in range(world)])
    gathered_order = [i for r in range(world) for i in range(r, len(queries), world)]
    scores = rows[torch.argsort(torch.tensor(gathered_order))]
    expected = torch.arange(7).float()[:, None] * torch.arange(11).float()[None, :] + 1
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores.T, expected.T, rtol=0, atol=0)

    # Also exercise cross-rank deduplication in the training helper.
    local_ids = [["q3", "q1", "q2"], ["q4", "q3", "q0"]][rank]
    merged, merged_ids = trainer.global_representations(represent(local_ids), local_ids, 3, cfg, False)
    assert merged_ids == ["q3", "q1", "q2", "q4", "q0"]
    torch.testing.assert_close(merged[0], represent(merged_ids)[0], rtol=0, atol=0)
    if rank == 0:
        result = {"backend": "gloo_cpu", "world_size": 2, "uneven_shard_gather": "passed",
                  "representation_id_restoration": "passed", "bidirectional_score_grid_restoration": "passed",
                  "training_cross_rank_deduplication": "passed"}
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
