#!/usr/bin/env python3
"""Train B1/B2 with CIRCE's sampler/loss on exactly two GPUs per run.

Run with torch.distributed.run. Frozen ESM-C features are cached on demand,
allowing training to start without a separate full-corpus extraction job.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "VenusRXN")]

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import yaml
from torch.nn.parallel import DistributedDataParallel

from horizyn.losses import SampledMultiPositiveInfoNCELoss
from horizyn.reaction_conditioned_data_module import TypedNegativeBatchSampler
from horizyn.token_retrieval import TokenModelConfig, TokenRetrievalModel
from horizyn.token_retrieval_data import (
    ESMC_REPO, ESMC_REVISION, FrozenResidueStore, MoleculeStore, ReactionSMIProtocol, atomic_json,
)
from horizyn.token_retrieval_evaluation import (
    EVALUATOR_ID, CirceEvaluationPlan, evaluate_circe_score_matrix, evaluator_provenance,
)


def gather_objects(value):
    if not dist.is_initialized():
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def gather_tensor(value, sizes, with_grad=False):
    if not dist.is_initialized():
        return value
    maximum = max(sizes)
    padding = value.new_zeros((maximum - len(value), *value.shape[1:]))
    padded = torch.cat([value, padding]).contiguous()
    if with_grad:
        gathered = dist_nn.all_gather(padded)
    else:
        gathered = [torch.empty_like(padded) for _ in sizes]
        dist.all_gather(gathered, padded)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes)])


def pack(representation):
    global_vector, tokens = representation
    return torch.cat([global_vector, tokens.flatten(1)], dim=1)


def unpack(value, count, cfg):
    return value[:, :cfg.global_dim], value[:, cfg.global_dim:].reshape(-1, count, cfg.local_dim)


def global_representations(local, ids, token_count, cfg, with_grad):
    gathered_ids = gather_objects(ids)
    value = gather_tensor(pack(local), [len(part) for part in gathered_ids], with_grad)
    unique, keep, seen = [], [], set()
    for i, identifier in enumerate(x for part in gathered_ids for x in part):
        if identifier not in seen:
            seen.add(identifier)
            unique.append(identifier)
            keep.append(i)
    keep = torch.tensor(keep, device=value.device)
    return unpack(value[keep], token_count, cfg), unique


def configure_graph_training(model, training, epoch):
    """Freeze pretrained weights and dropout during pooling warmup."""
    if model.cfg.graph_backbone != "pcqm4mv2_pretrained":
        return True
    finetuning = epoch >= int(training.get("graph_warmup_epochs", 0))
    model.graph_encoder.set_trainable(finetuning)
    return finetuning


def build_optimizer(model, training):
    if model.cfg.graph_backbone == "pcqm4mv2_pretrained":
        groups = [
            {"params": list(model.protein_pool.parameters()) + list(model.reaction_pool.parameters()),
             "lr": training["learning_rate"], "name": "pooling"},
            {"params": list(model.graph_encoder.parameters()),
             "lr": training["graph_learning_rate"], "name": "pretrained_graphormer"},
        ]
    else:
        groups = model.parameters()
    return torch.optim.AdamW(groups, lr=training["learning_rate"], weight_decay=training["weight_decay"])


def check_checkpoint_architecture(saved, config):
    if TokenModelConfig(**saved["config"]["model"]) != TokenModelConfig(**config["model"]):
        raise ValueError("Checkpoint architecture does not match this run; start a new pretrained experiment")


@torch.no_grad()
def evaluate(model, protocol, store, molecules, split, rank, world, output_dir, epoch, save_scores=False):
    model.eval()
    cfg, device = model.cfg, model.device
    plan = CirceEvaluationPlan.from_protocol(protocol, split)
    qids, eids = plan.query_ids, plan.candidate_ids
    # Each process encodes a disjoint slice; gathering below restores IDs.
    local_eids, local_qids = eids[rank::world], qids[rank::world]
    ep, qp = [], []
    for start in range(0, len(local_eids), 64):
        ids = local_eids[start:start + 64]
        residues = store.get_many([protocol.sequences[e] for e in ids])
        ep.append(pack(model.encode_proteins(residues)))
        if start % 1024 == 0:
            print(f"[{split} rank {rank}] proteins {start + len(ids)}/{len(local_eids)}", flush=True)
    for start in range(0, len(local_qids), 64):
        ids = local_qids[start:start + 64]
        qp.append(pack(model.encode_reactions([
            molecules.reaction(protocol.smiles[plan.raw_query_ids[q]]) for q in ids])))
    enzyme, gathered_eids = global_representations(
        unpack(torch.cat(ep), cfg.protein_tokens, cfg), local_eids, cfg.protein_tokens, cfg, False)
    reaction, gathered_qids = global_representations(
        unpack(torch.cat(qp), cfg.reaction_tokens, cfg), local_qids, cfg.reaction_tokens, cfg, False)
    elookup = {e: i for i, e in enumerate(gathered_eids)}
    qlookup = {q: i for i, q in enumerate(gathered_qids)}
    eorder = torch.tensor([elookup[e] for e in eids], device=device)
    qorder = torch.tensor([qlookup[q] for q in qids], device=device)
    enzyme = tuple(x[eorder] for x in enzyme)
    reaction = tuple(x[qorder] for x in reaction)
    local_scores = model.score_pairs(tuple(x[rank::world] for x in reaction), enzyme)
    rows = gather_tensor(local_scores, [len(qids[r::world]) for r in range(world)])
    gathered_row_order = [i for r in range(world) for i in range(r, len(qids), world)]
    inverse = torch.argsort(torch.tensor(gathered_row_order, device=device))
    scores = rows[inverse]
    result = None
    if rank == 0:
        result = evaluate_circe_score_matrix(scores, plan, output_dir / f"{split}_epoch{epoch}")
        if save_scores:
            torch.save({"scores": scores.cpu(), "query_ids": qids, "candidate_ids": eids,
                        "query_to_targets": plan.query_to_targets, "target_to_queries": plan.target_to_queries,
                        "epoch": epoch, "split": split, "evaluation_implementation": EVALUATOR_ID},
                       output_dir / f"{split}_epoch{epoch}_scores.pt")
        result.update(epoch=epoch, split=split, variant=model.variant,
                      score="global_cosine" if model.variant == "B1" else "0.5_global_plus_0.5_local")
        atomic_json(output_dir / f"{split}_epoch{epoch}.json", result)
        print(f"[{split} epoch {epoch} {EVALUATOR_ID}] first-positive mean={result['mean_bidirectional_mrr']:.6f} "
              f"all-positive mean={result['mean_bidirectional_reactzyme_mrr']:.6f}", flush=True)
    return gather_objects(result)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="Explicit smoke/profile limit; not a benchmark run")
    parser.add_argument("--batch-size", type=int, default=None, help="Explicit profile override")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--evaluate-only", default=None)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    parser.add_argument("--save-evaluation-scores", action="store_true", help="Retain a real score matrix for an evaluator audit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    run_dir = Path(args.run_dir).resolve()
    rank, world, local_rank = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("LOCAL_RANK", "0"))
    if args.max_steps is None and not args.evaluate_only:
        if world != config["training"]["devices"]:
            raise ValueError("Launch the benchmark with its configured number of distributed workers")
        if args.batch_size is not None:
            raise ValueError("Batch-size overrides are reserved for explicit smoke/profile runs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.set_num_threads(config.get("cpu_threads", 8))
    torch.set_float32_matmul_precision("high")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = TokenModelConfig(**config.get("model", {}))
    model = TokenRetrievalModel(cfg, config["variant"]).to(device)
    configure_graph_training(model, config["training"], 0)
    init_hash = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        init_hash.update(name.encode())
        init_hash.update(tensor.detach().cpu().numpy().tobytes())
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(run_dir / "status.json", {"state": "checking_inputs", "variant": model.variant})
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    data = config["data"]
    protocol = ReactionSMIProtocol(ROOT / data["split_dir"], ROOT / data["negative_pool"],
                                   include_test=bool(args.evaluate_only and args.evaluation_split == "test"))
    store = FrozenResidueStore(config["feature_cache"], ROOT / config["foundation_checkpoint"], device,
                               config.get("extraction_tokens", 8192))
    graph_builder = None
    if cfg.graph_backbone == "pcqm4mv2_pretrained":
        from horizyn.pretrained_graphormer import PretrainedGraphBuilder
        graph_builder = PretrainedGraphBuilder(model.graph_encoder.config, config.get("graph_feature_cache"))
    molecules = MoleculeStore(graph_builder)
    # Check every required molecular input before the first optimizer update.
    for smiles in protocol.smiles.values():
        molecules.reaction(smiles)
    batch_size = args.batch_size or config["training"]["batch_size_per_gpu"]
    sampler = TypedNegativeBatchSampler(protocol.training_rows, batch_size=batch_size,
                                        positive_fraction=0.5, biological_negative_fraction=0.5, seed=seed)
    if rank == 0:
        metadata = {"config": config, "model": model.description(), "initial_state_sha256": init_hash.hexdigest(),
                    "protocol": protocol.manifest(), "world_size": world, "batch_size_per_gpu": batch_size,
                    "effective_global_pair_rows": batch_size * world, "steps_per_epoch": len(sampler),
                    "foundation_repo": ESMC_REPO, "foundation_revision": ESMC_REVISION,
                    "feature_cache": str(store.root),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                    "total_retrieval_parameters": sum(p.numel() for p in model.parameters()),
                    "molecular_components": len(molecules.graphs),
                    "max_component_atoms": max(g.num_nodes for g in molecules.graphs.values()),
                    "start_utc": datetime.now(timezone.utc).isoformat(), "smoke_max_steps": args.max_steps,
                    "torch_version": torch.__version__, "python": sys.executable,
                    "evaluation": evaluator_provenance()}
        if args.resume and (run_dir / "run_manifest.json").exists():
            prior = json.loads((run_dir / "run_manifest.json").read_text())
            prior.setdefault("resumes", []).append({"checkpoint": str(Path(args.resume).resolve()),
                                                    "utc": metadata["start_utc"], "evaluation": metadata["evaluation"]})
            prior["evaluation"] = metadata["evaluation"]
            atomic_json(run_dir / "run_manifest.json", prior)
        else:
            atomic_json(run_dir / "run_manifest.json", metadata)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        print(json.dumps(metadata, indent=2), flush=True)
    if args.evaluate_only:
        saved = torch.load(args.evaluate_only, map_location=device, weights_only=False)
        check_checkpoint_architecture(saved, config)
        model.load_state_dict(saved["model"])
        result = evaluate(model, protocol, store, molecules, args.evaluation_split, rank, world,
                          evaluation_dir, saved["epoch"], save_scores=args.save_evaluation_scores)
        if rank == 0:
            atomic_json(run_dir / "status.json", {"state": "evaluation_complete", "result": result})
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    optimizer = build_optimizer(model, config["training"])
    loss_fn = SampledMultiPositiveInfoNCELoss(beta=10.0, learn_beta=False).to(device)
    best, early_best, bad_epochs, start_epoch, global_step = float("-inf"), float("-inf"), 0, 0, 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("evaluation_implementation") != EVALUATOR_ID:
            raise ValueError("Re-evaluate the saved checkpoint with CIRCE before resuming; old selection metrics are incompatible")
        check_checkpoint_architecture(saved, config)
        if saved["config"]["training"].get("graph_warmup_epochs", 0) != config["training"].get("graph_warmup_epochs", 0):
            raise ValueError("Resume must preserve the pretrained Graphormer warmup schedule")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        best, early_best, bad_epochs = saved["best"], saved["early_best"], saved["bad_epochs"]
        start_epoch, global_step = saved["epoch"] + 1, saved["global_step"]
        state = saved["rng_states"][rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu())
        torch.cuda.set_rng_state(state["cuda"].cpu())
    else:
        torch.manual_seed(seed + rank)
    started = time.monotonic()
    wrapped, previous_graph_stage = None, None
    for epoch in range(start_epoch, config["training"]["max_epochs"]):
        graph_stage = configure_graph_training(model, config["training"], epoch)
        if graph_stage != previous_graph_stage:
            # DDP registers the parameters requiring gradients at construction.
            # Rebuild it once when the pretrained encoder is unfrozen.
            if dist.is_initialized():
                dist.barrier()
            del wrapped
            wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else model
            previous_graph_stage = graph_stage
            if rank == 0:
                print(json.dumps({"epoch": epoch, "graph_finetuning": graph_stage,
                                  "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                  "learning_rates": [g["lr"] for g in optimizer.param_groups]}), flush=True)
        sampler.set_epoch(epoch)
        wrapped.train()
        for batch_index, indices in enumerate(sampler):
            tick = time.monotonic()
            rows = [protocol.training_rows[i] for i in indices]
            qids = list(dict.fromkeys(row["query_id"] for row in rows))
            eids = list(dict.fromkeys(row["target_id"] for row in rows))
            if rank == 0:
                atomic_json(run_dir / "status.json", {"state": "preparing_batch", "epoch": epoch,
                                                      "batch": batch_index, "global_step": global_step,
                                                      "local_proteins": len(eids)})
            residues = store.get_many([protocol.sequences[e] for e in eids])
            chemistry = [molecules.reaction(protocol.smiles[q]) for q in qids]
            optimizer.zero_grad(set_to_none=True)
            reaction, enzyme = wrapped(residues, chemistry)
            reaction, global_qids = global_representations(reaction, qids, cfg.reaction_tokens, cfg, True)
            enzyme, global_eids = global_representations(enzyme, eids, cfg.protein_tokens, cfg, True)
            positive, biological, random_negative = protocol.masks(global_qids, global_eids, device)
            allowed_indices = (positive | biological | random_negative).nonzero()
            allowed_scores = model.score_pairs(reaction, enzyme, allowed_indices)
            scores = allowed_scores.new_zeros(positive.shape).index_put(tuple(allowed_indices.T), allowed_scores)
            qi, ei = positive.nonzero(as_tuple=True)
            loss = loss_fn(1.0 - scores, qi, ei, biological, random_negative)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            loss.backward()
            if global_step < 2 and not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Non-finite model gradient")
            optimizer.step()
            torch.cuda.synchronize()
            global_step += 1
            if rank == 0:
                record = {"epoch": epoch, "batch": batch_index, "global_step": global_step,
                          "loss": loss.item(), "step_seconds": time.monotonic() - tick,
                          "unique_reactions": len(global_qids), "unique_enzymes": len(global_eids),
                          "positive_pairs": int(positive.sum()), "biological_pairs": int(biological.sum()),
                          "random_pairs": int(random_negative.sum()), "scored_pairs": len(allowed_indices),
                          "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                          "graph_finetuning": graph_stage,
                          "cache_extracted_rank0": store.extracted, "elapsed_seconds": time.monotonic() - started}
                with (run_dir / "steps.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                atomic_json(run_dir / "status.json", {"state": "training", **record})
                if global_step <= 5 or global_step % 10 == 0:
                    print(json.dumps(record), flush=True)
            del residues, chemistry, reaction, enzyme, scores, loss, allowed_scores
            if args.max_steps is not None and global_step >= args.max_steps:
                if rank == 0:
                    torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, run_dir / "smoke_checkpoint.pt")
                    atomic_json(run_dir / "status.json", {"state": "smoke_complete", "global_step": global_step})
                if dist.is_initialized():
                    dist.barrier()
                    dist.destroy_process_group()
                return
        validation = evaluate(model, protocol, store, molecules, "validation", rank, world, evaluation_dir, epoch)
        metric = validation["mean_bidirectional_mrr"]
        improved = metric > best
        best = max(best, metric)
        if metric > early_best + config["training"]["min_delta"]:
            early_best, bad_epochs = metric, 0
        else:
            bad_epochs += 1
        rng_states = gather_objects({"python": random.getstate(), "numpy": np.random.get_state(),
                                     "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})
        if rank == 0:
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                     "global_step": global_step, "best": best, "early_best": early_best,
                     "bad_epochs": bad_epochs, "rng_states": rng_states, "config": config,
                     "validation": validation, "evaluation_implementation": EVALUATOR_ID}
            temporary = run_dir / "checkpoints/last.tmp"
            torch.save(state, temporary)
            os.replace(temporary, run_dir / "checkpoints/last.pt")
            if improved:
                temporary = run_dir / "checkpoints/best.tmp"
                torch.save(state, temporary)
                os.replace(temporary, run_dir / "checkpoints/best.pt")
            atomic_json(run_dir / "status.json", {"state": "validated", "epoch": epoch,
                                                  "global_step": global_step, "best_validation_mrr": best,
                                                  "bad_epochs": bad_epochs})
        if dist.is_initialized():
            dist.barrier()
        if bad_epochs >= config["training"]["patience"]:
            break
    saved = torch.load(run_dir / "checkpoints/best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    protocol = ReactionSMIProtocol(ROOT / data["split_dir"], ROOT / data["negative_pool"], include_test=True)
    result = evaluate(model, protocol, store, molecules, "test", rank, world, evaluation_dir, saved["epoch"])
    if rank == 0:
        result["checkpoint"] = str(run_dir / "checkpoints/best.pt")
        atomic_json(evaluation_dir / "test_both.json", result)
        reference = json.loads((ROOT / config["reference_metrics"]).read_text())
        atomic_json(evaluation_dir / "comparison_with_circe_v2.json", {
            "candidate": result, "circe_v2": reference,
            "all_positive_mean_delta": result["balanced_reactzyme_mrr"] - reference["balanced_reactzyme_mrr"],
            "first_positive_mean_delta": result["balanced_first_positive_mrr"] - reference["balanced_first_positive_mrr"],
        })
        atomic_json(run_dir / "status.json", {"state": "complete", "global_step": global_step,
                                              "best_epoch": saved["epoch"], "test": result})
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--run-dir" in sys.argv:
            import traceback
            directory = Path(sys.argv[sys.argv.index("--run-dir") + 1])
            failure = {"state": "failed", "rank": int(os.getenv("RANK", "0")),
                       "error": str(error), "traceback": traceback.format_exc()}
            atomic_json(directory / f"failure_rank{failure['rank']}.json", failure)
            atomic_json(directory / "status.json", failure)
        raise
