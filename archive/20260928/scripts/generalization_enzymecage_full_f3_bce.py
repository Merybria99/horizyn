#!/usr/bin/env python3
"""Train both F3 towers on every original EnzymeCAGE label with BCE.

Protein-grouped minibatches encode each protein once while retaining duplicate
and conflicting source rows. Run with ``torchrun --nproc_per_node=4``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler

from horizyn.config import load_config
from horizyn.datasets.residue_hdf5 import ResidueEmbedDataset
from horizyn.reaction_features import build_reaction_feature_dataset
from horizyn.utils.collate import residue_collate_fn, unimol2_reaction_collate_fn
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.training_options import protein_pooling_model_kwargs


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


class LabeledProteinDataset(Dataset):
    def __init__(self, pairs: np.ndarray, protein_ids: list[str], primary: str,
                 missing: str, max_tokens: int, truncation: str):
        self.primary = ResidueEmbedDataset(primary, max_tokens=max_tokens,
                                           truncation=truncation)
        self.missing = ResidueEmbedDataset(missing, max_tokens=max_tokens,
                                           truncation=truncation)
        self.ids = protein_ids
        self.pairs = pairs
        self.order = np.argsort(pairs[:, 1], kind="stable")
        self.proteins, self.start, self.count = np.unique(
            pairs[self.order, 1], return_index=True, return_counts=True)
        available = set(self.primary.keys) | set(self.missing.keys)
        absent = [protein_ids[i] for i in self.proteins if protein_ids[i] not in available]
        if absent:
            raise ValueError(f"Missing {len(absent)} labeled proteins: {absent[:3]}")
        self.primary_ids = set(self.primary.keys)
        # VDS reads become nearly sequential when neighboring records share a
        # batch. Shuffle batches, not individual proteins, for SSD throughput.
        physical = np.asarray([
            self.primary.key_to_idx[key] if key in self.primary_ids
            else len(self.primary) + self.missing.key_to_idx[key]
            for key in (protein_ids[i] for i in self.proteins)
        ])
        physical_order = np.argsort(physical, kind="stable")
        self.proteins = self.proteins[physical_order]
        self.start = self.start[physical_order]
        self.count = self.count[physical_order]

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, position: int):
        protein_index = int(self.proteins[position])
        key = self.ids[protein_index]
        source = self.primary if key in self.primary_ids else self.missing
        sample = source[key]
        lo = int(self.start[position])
        hi = lo + int(self.count[position])
        return sample, self.pairs[self.order[lo:hi]], protein_index


def collate_protein_groups(samples):
    residues = residue_collate_fn([sample for sample, _, _ in samples])
    rows = np.concatenate([rows for _, rows, _ in samples], axis=0)
    protein_row = torch.repeat_interleave(torch.arange(len(samples)),
                                          torch.tensor([len(rows) for _, rows, _ in samples]))
    return (residues["residue_embeddings"], residues["residue_padding_mask"],
            torch.from_numpy(rows), protein_row)


class ProteinBatchSampler(Sampler[list[int]]):
    def __init__(self, size: int, batch_size: int, rank: int, world: int,
                 seed: int, shuffle: bool):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world, self.seed, self.shuffle = rank, world, seed, shuffle
        self.epoch = 0

    def __iter__(self):
        local = np.arange(self.rank * self.size // self.world,
                          (self.rank + 1) * self.size // self.world)
        batches = [local[start:start + self.batch_size].tolist()
                   for start in range(0, len(local), self.batch_size)]
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(batches)
        yield from batches

    def __len__(self):
        return math.ceil(((self.rank + 1) * self.size // self.world
                          - self.rank * self.size // self.world) / self.batch_size)


class GPUResidueCache:
    """Stage a rank-local contiguous HDF5 residue range once per run."""
    def __init__(self, train_data, train_sampler, valid_data, val_sampler, device):
        self.device = device
        self.max_tokens = train_data.primary.max_tokens
        self.locations = {}
        local = []
        for dataset, sampler in ((train_data, train_sampler), (valid_data, val_sampler)):
            begin = sampler.rank * len(dataset) // sampler.world
            end = (sampler.rank + 1) * len(dataset) // sampler.world
            local.append((dataset, np.arange(begin, end)))
        primary_keys = train_data.primary.key_to_idx
        used_primary = [primary_keys[dataset.ids[int(dataset.proteins[position])]]
                        for dataset, positions in local for position in positions
                        if dataset.ids[int(dataset.proteins[position])] in primary_keys]
        first, last = min(used_primary), max(used_primary)
        offsets = train_data.primary.offsets.numpy()
        vector_first, vector_last = int(offsets[first]), int(offsets[last + 1])
        missing_offsets = train_data.missing.offsets.numpy()
        extra = int(missing_offsets[-1])
        total = vector_last - vector_first + extra
        self.flat = torch.empty((total, train_data.primary.vec_dim),
                                dtype=torch.float16, device=device)
        if dist.get_rank() == 0:
            print(json.dumps({"gpu_residue_cache_gib": round(self.flat.numel() * 2 / 2**30, 2),
                              "primary_protein_range": [first, last],
                              "primary_residues": vector_last - vector_first}), flush=True)
        chunk = 32768  # 64 MiB; avoids host allocations proportional to the whole store.
        with h5py.File(train_data.primary.file_path, "r") as handle:
            vectors = handle["vectors"]
            for start in range(vector_first, vector_last, chunk):
                stop = min(start + chunk, vector_last)
                block = torch.from_numpy(np.asarray(vectors[start:stop]))
                self.flat[start-vector_first:stop-vector_first].copy_(block.to(device))
        with h5py.File(train_data.missing.file_path, "r") as handle:
            vectors = handle["vectors"]
            for start in range(0, extra, chunk):
                stop = min(start + chunk, extra)
                block = torch.from_numpy(np.asarray(vectors[start:stop]))
                self.flat[vector_last-vector_first+start:vector_last-vector_first+stop].copy_(block.to(device))
        for dataset, positions in local:
            starts = np.zeros(len(dataset), dtype=np.int64)
            lengths = np.zeros(len(dataset), dtype=np.int32)
            for position in positions:
                key = dataset.ids[int(dataset.proteins[position])]
                if key in primary_keys:
                    source_idx = primary_keys[key]
                    starts[position] = int(offsets[source_idx]) - vector_first
                    lengths[position] = int(offsets[source_idx+1] - offsets[source_idx])
                else:
                    source_idx = dataset.missing.key_to_idx[key]
                    starts[position] = vector_last - vector_first + int(missing_offsets[source_idx])
                    lengths[position] = int(missing_offsets[source_idx+1] - missing_offsets[source_idx])
            self.locations[id(dataset)] = starts, lengths
        dist.barrier()
        if dist.get_rank() == 0:
            print(json.dumps({"gpu_cache_ready": True,
                              "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}), flush=True)

    def batch(self, dataset, positions):
        starts_np, lengths_np = self.locations[id(dataset)]
        starts = torch.as_tensor(starts_np[positions], device=self.device)
        lengths = torch.as_tensor(lengths_np[positions], device=self.device)
        max_length = min(int(lengths.max()), self.max_tokens)
        pos = torch.arange(max_length, device=self.device)[None, :]
        raw_len = lengths[:, None]
        first = self.max_tokens // 4
        last = self.max_tokens // 4
        middle = self.max_tokens - first - last
        middle_start = torch.maximum((raw_len - middle) // 2,
                                     torch.full_like(raw_len, first))
        middle_end = torch.minimum(middle_start + middle, raw_len - last)
        middle_start = torch.maximum(middle_end - middle,
                                     torch.full_like(raw_len, first))
        long_rel = torch.where(pos < first, pos,
                    torch.where(pos < first + middle,
                                middle_start + pos - first,
                                raw_len - last + pos - first - middle))
        relative = torch.where(raw_len > self.max_tokens, long_rel, pos)
        relative = torch.minimum(relative, raw_len - 1)
        indices = starts[:, None] + relative
        residues = self.flat.index_select(0, indices.reshape(-1)).view(
            len(positions), max_length, self.flat.shape[1])
        mask = pos >= torch.clamp(raw_len, max=self.max_tokens)
        groups = [dataset.pairs[dataset.order[int(dataset.start[p]):
                                             int(dataset.start[p] + dataset.count[p])]]
                  for p in positions]
        rows = torch.from_numpy(np.concatenate(groups, axis=0))
        protein_row = torch.repeat_interleave(torch.arange(len(positions)),
                                              torch.tensor([len(group) for group in groups]))
        return residues, mask, rows, protein_row


class PairBCE(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-1.6))

    def forward(self, reaction_inputs, residues, mask, reaction_row, protein_row,
                return_embeddings=False):
        reaction = self.base.encode_queries(reaction_inputs)
        enzyme = self.base.encode_targets(residues, residue_padding_mask=mask)
        reaction = F.normalize(reaction.float(), dim=-1)
        enzyme = F.normalize(enzyme.float(), dim=-1)
        logits = self.logit_scale.exp().clamp(max=100) * (
            reaction[reaction_row] * enzyme[protein_row]).sum(-1) + self.bias
        if return_embeddings:
            return logits, reaction, enzyme
        return logits


def labeled_retrieval_loss(reaction, enzyme, reaction_row, protein_row, label,
                           temperature: float):
    """Bidirectional multi-positive retrieval over observed labels in a minibatch."""
    similarity = (enzyme.float() @ reaction.float().T) / temperature
    known = torch.zeros_like(similarity, dtype=torch.bool)
    positive = torch.zeros_like(similarity, dtype=torch.bool)
    known[protein_row, reaction_row] = True
    positive[protein_row[label > 0.5], reaction_row[label > 0.5]] = True
    masked = similarity.masked_fill(~known, -torch.inf)
    positive_masked = similarity.masked_fill(~positive, -torch.inf)
    eligible_protein = positive.any(dim=1) & (known.sum(dim=1) > positive.sum(dim=1))
    eligible_reaction = positive.any(dim=0) & (known.sum(dim=0) > positive.sum(dim=0))
    terms = []
    if eligible_protein.any():
        terms.append((torch.logsumexp(masked[eligible_protein], dim=1)
                      - torch.logsumexp(positive_masked[eligible_protein], dim=1)).mean())
    if eligible_reaction.any():
        terms.append((torch.logsumexp(masked[:, eligible_reaction], dim=0)
                      - torch.logsumexp(positive_masked[:, eligible_reaction], dim=0)).mean())
    if not terms:
        return similarity.sum() * 0.0
    return torch.stack(terms).mean()


def reaction_cache(config, split: str):
    path = config.data.train_reactions_path if split == "train" else config.data.validation_reactions_path
    return build_reaction_feature_dataset(path, config, bidirectional=True,
                                          split_name=split)


def query_batch(rows: torch.Tensor, protein_row: torch.Tensor, reaction_ids: list[str],
                cache, device: torch.device):
    rxn_unique, rxn_inverse = torch.unique(rows[:, 0], sorted=True, return_inverse=True)
    keys = [reaction_ids[int(i)] + "_f" for i in rxn_unique]
    samples = [cache[key] for key in keys]
    query = unimol2_reaction_collate_fn(samples)
    query = {key: value.to(device, non_blocking=True) for key, value in query.items()
             if torch.is_tensor(value)}
    return query, rxn_inverse.to(device, non_blocking=True), protein_row.to(device, non_blocking=True)


def evaluate(model, loader, reaction_ids, cache, device):
    model.eval()
    scores, labels = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for residues, mask, rows, protein_row in loader:
            query, r_idx, p_idx = query_batch(rows, protein_row, reaction_ids, cache, device)
            residues = residues.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            score = model(query, residues, mask, r_idx, p_idx)
            scores.append(score.float().cpu().numpy())
            labels.append(rows[:, 2].numpy())
            del query, residues, mask, rows, protein_row, r_idx, p_idx, score
    score = np.concatenate(scores)
    label = np.concatenate(labels)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (score, label))
    if dist.get_rank() == 0:
        score = np.concatenate([item[0] for item in gathered])
        label = np.concatenate([item[1] for item in gathered])
        return float(roc_auc_score(label, score)), len(label)
    return None, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--missing-residues", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-proteins", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--optimizer", choices=("adam", "adamw"), default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-decay", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=0,
                        help="Validation-AUROC early-stop patience; zero disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--gpu-residue-cache", action="store_true")
    parser.add_argument("--init-from-scratch", action="store_true",
                        help="Initialize F3 towers from config, retaining only configured frozen SLEEC prior")
    parser.add_argument("--retrieval-weight", type=float, default=0.0,
                        help="Add bidirectional observed-label multi-positive retrieval loss")
    parser.add_argument("--retrieval-temperature", type=float, default=0.1)
    args = parser.parse_args()
    if args.retrieval_weight < 0 or args.retrieval_temperature <= 0:
        parser.error("Retrieval weight must be nonnegative and temperature positive")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    config = load_config(str(args.config))
    with np.load(args.pairs) as source:
        train, valid = source["train"], source["validation"]
    with np.load(args.features) as source:
        protein_ids = source["protein_ids"].tolist()
        reaction_ids = source["reaction_ids"].tolist()
    if len(train) != 1445915 or len(valid) != 67677:
        raise ValueError("Original labeled EnzymeCAGE split changed")
    primary = str(config.data.protein_residue_embeds_path)
    data_opts = (protein_ids, primary, str(args.missing_residues),
                 config.data.max_protein_tokens, config.data.protein_truncation)
    train_data = LabeledProteinDataset(train, *data_opts)
    valid_data = LabeledProteinDataset(valid, *data_opts)
    sampler = ProteinBatchSampler(len(train_data), args.batch_proteins, rank, world,
                                  args.seed, True)
    val_sampler = ProteinBatchSampler(len(valid_data), args.batch_proteins, rank, world,
                                      args.seed, False)
    loader_opts = dict(num_workers=args.workers, pin_memory=True,
                       collate_fn=collate_protein_groups,
                       persistent_workers=args.workers > 0)
    if args.workers:
        loader_opts["prefetch_factor"] = 1
    train_loader = DataLoader(train_data, batch_sampler=sampler, **loader_opts)
    val_loader = DataLoader(valid_data, batch_sampler=val_sampler, **loader_opts)
    if args.gpu_residue_cache:
        cache = GPUResidueCache(train_data, sampler, valid_data, val_sampler, device)
        train_loader = (cache.batch(train_data, positions) for positions in sampler)
        val_loader = None
    train_reactions = reaction_cache(config, "train")
    valid_reactions = reaction_cache(config, "validation")
    if not set(train_reactions.keys).issubset({key + "_f" for key in reaction_ids}):
        raise ValueError("Train reaction catalog is incomplete")
    if args.init_from_scratch:
        lit = ProteinPooledLitModule(**protein_pooling_model_kwargs(config))
    else:
        lit = ProteinPooledLitModule.load_from_checkpoint(str(args.checkpoint),
                                                          map_location="cpu")
    base = lit.model
    del lit
    model = PairBCE(base).to(device)
    model = DistributedDataParallel(model, device_ids=[device.index],
                                    find_unused_parameters=True)
    optimizer_type = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_type([p for p in model.parameters() if p.requires_grad],
                               lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    if rank == 0:
        receipt = {"schema": "enzymecage_full_f3_labeled_bce_v1",
                   "config": str(args.config), "config_sha256": digest(args.config),
                   "architecture_reference": str(args.checkpoint),
                   "architecture_reference_sha256": digest(args.checkpoint),
                   "warm_start": None if args.init_from_scratch else str(args.checkpoint),
                   "warm_start_sha256": None if args.init_from_scratch else digest(args.checkpoint),
                   "initialization": "F3_config_random_trainable_towers" if args.init_from_scratch
                                     else "F3_positive_only_EnzymeCAGE_checkpoint",
                   "pairs": str(args.pairs), "pairs_sha256": digest(args.pairs),
                   "features": str(args.features), "features_sha256": digest(args.features),
                   "missing_residues": str(args.missing_residues),
                   "train_rows": len(train), "validation_rows": len(valid),
                   "train_positives": int(train[:, 2].sum()),
                   "train_negatives": int(len(train) - train[:, 2].sum()),
                   "batch_proteins_per_gpu": args.batch_proteins, "world_size": world,
                   "gpu_residue_cache": args.gpu_residue_cache,
                   "learning_rate": args.learning_rate, "seed": args.seed,
                   "precision": "bfloat16_mixed", "optimizer": args.optimizer,
                   "weight_decay": args.weight_decay, "lr_decay": args.lr_decay,
                   "patience": args.patience,
                   "loss": "unweighted_BCEWithLogitsLoss_cosine_trainable_scale_bias",
                   "retrieval_weight": args.retrieval_weight,
                   "retrieval_temperature": args.retrieval_temperature,
                   "retrieval_negative_policy": "observed_labeled_rows_only",
                   "selection": "original_validation_global_AUROC",
                   "source_sha256": digest(Path(__file__))}
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (args.output / "trainer_source.py").write_bytes(Path(__file__).read_bytes())
        print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()),
                          "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
                          "train_batches": len(sampler), "validation_batches": len(val_sampler),
                          "receipt": str(args.output / "receipt.json")}), flush=True)
    best = -float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        sampler.epoch = epoch
        if args.gpu_residue_cache:
            train_loader = (cache.batch(train_data, positions) for positions in sampler)
        model.train()
        started = time.monotonic()
        running = torch.zeros(2, device=device)
        for step, (residues, mask, rows, protein_row) in enumerate(train_loader):
            query, r_idx, p_idx = query_batch(rows, protein_row, reaction_ids,
                                              train_reactions, device)
            residues = residues.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            label = rows[:, 2].to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model(query, residues, mask, r_idx, p_idx,
                                return_embeddings=args.retrieval_weight > 0)
                if args.retrieval_weight:
                    logits, reaction, enzyme = encoded
                    retrieval_loss = labeled_retrieval_loss(
                        reaction, enzyme, r_idx, p_idx, label,
                        args.retrieval_temperature)
                else:
                    logits = encoded
                bce_loss = F.binary_cross_entropy_with_logits(logits.float(), label)
                loss = bce_loss + args.retrieval_weight * retrieval_loss if args.retrieval_weight else bce_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch} step {step}")
            loss.backward()
            optimizer.step()
            running[0] += loss.detach() * len(label)
            running[1] += len(label)
            if step == 0 or (step + 1) % 10 == 0:
                torch.cuda.synchronize()
                if rank == 0:
                    print(json.dumps({"epoch": epoch, "step": step + 1,
                                      "loss": float(loss.item()),
                                      "elapsed_s": round(time.monotonic() - started, 1),
                                      "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                      "gpu_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}), flush=True)
            # The next GPU-resident batch is materialized before Python assigns
            # loop variables, so release prior activations and inputs explicitly.
            del query, residues, mask, label, logits, loss, bce_loss, encoded, r_idx, p_idx, rows, protein_row
            if args.retrieval_weight:
                del reaction, enzyme, retrieval_loss
            if args.smoke_steps and step + 1 >= args.smoke_steps:
                break
        dist.all_reduce(running)
        if args.smoke_steps:
            break
        if args.gpu_residue_cache:
            val_loader = (cache.batch(valid_data, positions) for positions in val_sampler)
        auroc, n_val = evaluate(model.module, val_loader, reaction_ids,
                                valid_reactions, device)
        if rank == 0:
            row = {"epoch": epoch, "train_bce": float(running[0] / running[1]),
                   "validation_auroc": auroc, "validation_rows": n_val,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "elapsed_s": round(time.monotonic() - started, 1),
                   "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if auroc > best:
                best = auroc
                stale_epochs = 0
                out = args.output / "best.pt"
                temp = args.output / "best.pt.tmp"
                torch.save({"model": {key: value.cpu() for key, value in model.module.state_dict().items()},
                            "epoch": epoch, "validation_auroc": auroc,
                            "receipt_sha256": digest(args.output / "receipt.json")}, temp)
                os.replace(temp, out)
            else:
                stale_epochs += 1
        scheduler.step()
        dist.barrier()
        stop = torch.tensor([int(args.patience > 0 and stale_epochs >= args.patience)],
                            device=device)
        dist.broadcast(stop, src=0)
        if bool(stop.item()):
            if rank == 0:
                print(json.dumps({"early_stopped_after_epoch": epoch,
                                  "best_validation_auroc": best}), flush=True)
            break
    if rank == 0 and not args.smoke_steps:
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        selected = max(history, key=lambda row: row["validation_auroc"])
        complete = {"schema": "enzymecage_full_f3_labeled_bce_complete_v1",
                    "epochs_run": len(history), "selected_epoch": selected["epoch"],
                    "selected_validation_auroc": selected["validation_auroc"],
                    "checkpoint_sha256": digest(args.output / "best.pt"),
                    "training_receipt_sha256": digest(args.output / "receipt.json"),
                    "source_snapshot_sha256": digest(args.output / "trainer_source.py")}
        (args.output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
