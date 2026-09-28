#!/usr/bin/env python3
"""Exercise actual two-rank retrieval gradients across Graphormer unfreezing."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "VenusRXN")]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from horizyn.losses import SampledMultiPositiveInfoNCELoss
from horizyn.pretrained_graphormer import PretrainedGraphBuilder
from horizyn.token_retrieval import TokenModelConfig, TokenRetrievalModel
from horizyn.token_retrieval_data import MoleculeStore, atomic_json
from scripts.train_token_retrieval import configure_graph_training, build_optimizer, global_representations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["B1", "B2"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 2:
        raise ValueError("This smoke test requires two GPUs")
    config = yaml.safe_load((ROOT / f"configs/reactzyme_reaction_smi_{args.variant.lower()}_esmc_graphormer.yaml").read_text())
    torch.manual_seed(config["seed"])
    model = TokenRetrievalModel(TokenModelConfig(**config["model"]), args.variant).to(device)
    optimizer = build_optimizer(model, config["training"])
    store = MoleculeStore(PretrainedGraphBuilder(model.graph_encoder.config, config["graph_feature_cache"]))
    residues = [torch.randn(12 + rank, model.cfg.residue_dim), torch.randn(18 + rank, model.cfg.residue_dim)]
    chemistry = [store.reaction(s) for s in (["CCO.O", "c1ccccc1"] if rank == 0 else ["CC(=O)O", "C[C@H](O)C(=O)O"])]
    ids = [f"{rank}_{i}" for i in range(2)]
    watched = next(model.graph_encoder.parameters())
    original = watched.detach().clone()
    wrapped = None
    records = []
    for epoch in [0, 2]:
        active = configure_graph_training(model, config["training"], epoch)
        dist.barrier()
        del wrapped
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        wrapped.train()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            reaction, enzyme = wrapped(residues, chemistry)
            reaction, qids = global_representations(reaction, ids, model.cfg.reaction_tokens, model.cfg, True)
            enzyme, eids = global_representations(enzyme, ids, model.cfg.protein_tokens, model.cfg, True)
            indices = torch.cartesian_prod(torch.arange(4, device=device), torch.arange(4, device=device))
            scores = model.score_pairs(reaction, enzyme, indices).reshape(4, 4)
            positive = torch.eye(4, dtype=torch.bool, device=device)
            loss = SampledMultiPositiveInfoNCELoss(beta=10)(
                1 - scores, *positive.nonzero(as_tuple=True), ~positive, torch.zeros_like(positive))
            loss.backward()
            gradients = [p.grad for p in model.graph_encoder.parameters() if p.grad is not None]
            if bool(gradients) != active or not all(torch.isfinite(g).all() for g in gradients):
                raise AssertionError("Incorrect distributed Graphormer gradient state")
            norm = sum(float(g.abs().sum()) for g in gradients)
            if active and norm == 0:
                raise AssertionError("Graphormer received no retrieval learning signal")
            optimizer.step()
            if torch.equal(original, watched) != (not active):
                raise AssertionError("Frozen/pretrained encoder update state is incorrect")
            # Actual DDP synchronization, checked across all encoder/pool weights.
            for parameter in model.parameters():
                reference = parameter.detach().clone()
                dist.broadcast(reference, src=0)
                if not torch.equal(reference, parameter):
                    raise AssertionError("Model parameters differ across DDP ranks")
            records.append({"epoch": epoch, "step": step, "graph_finetuning": active,
                            "loss": loss.item(), "graph_gradient_l1": norm})
            del reaction, enzyme, scores, loss
    if rank == 0:
        atomic_json(args.output, {"variant": args.variant, "ranks": 2, "checks_passed": True,
                                 "warmup_preserved_weights": True, "unfreeze_received_gradients": True,
                                 "all_parameters_synchronized": True, "steps": records})
        print(json.dumps(records), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
