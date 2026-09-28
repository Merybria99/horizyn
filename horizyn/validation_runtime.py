"""Exact-ownership retrieval validation for the opt-in fast-I/O launcher.

The validation-loss loader is deliberately unchanged. Retrieval is run in the
epoch-end hook, where unequal local shard lengths require no per-batch DDP
collectives. Only completed lookup tables and metric sums/counts are reduced.
"""
import math
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from horizyn.benchmarks.chunked_retrieval import chunked_positive_ranks, rank_metrics
from horizyn.protein_pooling_lightning_module import ProteinPooledLitModule
from horizyn.utils.collate import dict_collate_fn


class WholeBatchShardSampler(Sampler):
    """Assign each original batch to one rank, without padding or duplicates.

Keeping entire batches also preserves encoder padding/batch composition.
Ranks may have zero batches; no collective may be placed inside this loop.
"""
    def __init__(self, size, batch_size, rank=0, world_size=1):
        if size < 0 or batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("Invalid validation shard dimensions")
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size = rank, world_size

    def __len__(self):
        batches = math.ceil(self.size / self.batch_size)
        return max(0, math.ceil((batches - self.rank) / self.world_size))

    def __iter__(self):
        for start in range(self.rank * self.batch_size, self.size,
                           self.world_size * self.batch_size):
            yield list(range(start, min(start + self.batch_size, self.size)))


class ShardedValidationDataMixin:
    supports_sharded_validation = True

    def val_dataloader(self):
        loaders = super().val_dataloader()
        module = getattr(getattr(self, "trainer", None), "lightning_module", None)
        if self.validation_retrieval_metrics and getattr(module, "sharded_retrieval_validation", False):
            # Keep exactly the original validation-loss batches and gather
            # semantics. The module handles all retrieval loaders separately.
            return loaders[:1]
        return loaders


class FastValidationLitModule(ProteinPooledLitModule):
    sharded_retrieval_validation = True

    def _fast_validation_enabled(self):
        return self.validation_retrieval_metrics and getattr(
            self.trainer.datamodule, "supports_sharded_validation", False)

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self._validation_started = time.perf_counter()

    def _log_loss_components(self, prefix, components, batch_size, on_step, on_epoch):
        if prefix != "val" or not self._fast_validation_enabled():
            return super()._log_loss_components(prefix, components, batch_size, on_step, on_epoch)
        # There is now only one Lightning loader. Retain the old diagnostic
        # keys, which Lightning previously suffixed for the five-loader path.
        for name, value in components.items():
            self.log(f"val/loss_{name}/dataloader_idx_0", value,
                     on_step=on_step, on_epoch=on_epoch, batch_size=batch_size,
                     sync_dist=True, add_dataloader_idx=False)

    def _validation_shard(self, size, batch_size):
        rank, world = (dist.get_rank(), dist.get_world_size()) if self._distributed_enabled() else (0, 1)
        return WholeBatchShardSampler(size, batch_size, rank, world)

    def _validation_loader(self, dataset, batch_size, *, target):
        data = self.trainer.datamodule
        sampler = self._validation_shard(len(dataset), batch_size)
        if not len(sampler):
            return ()  # No idle worker pool on an empty rank.
        return DataLoader(
            dataset, batch_sampler=sampler,
            collate_fn=data._base_collate_fn if target else dict_collate_fn,
            generator=getattr(self, "_validation_loader_generator", None),
            **data._loader_worker_kwargs(),
        )

    def _validation_encode(self, batch, *, target, direction):
        # Epoch-end hooks are outside Lightning's per-step precision context.
        # Use the same transfer hooks and AMP policy as ordinary validation.
        batch = self.trainer.strategy.batch_to_device(batch, dataloader_idx=1 if target else 2)
        with self.trainer.precision_plugin.forward_context():
            value = (self._encode_target_batch if target else self._encode_query_batch)(
                batch, retrieval_direction=direction)
        value = value.detach().float()
        if value.ndim != 2 or value.shape[1] != int(self.hparams.embedding_dim):
            raise ValueError("Unexpected validation embedding shape")
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite validation embedding")
        self._validation_work[0 if target else 1] += len(value)
        return batch, value

    def _validation_lookup(self, dataset, ids, batch_size, *, target):
        raw = torch.zeros((len(ids), int(self.hparams.embedding_dim)), device=self.device,
                          dtype=torch.float32)
        seen = torch.zeros(len(ids), device=self.device, dtype=torch.int32)
        row_key = "target_lookup_row_idx" if target else "query_lookup_row_idx"
        direction = "reaction_to_enzyme" if target else "enzyme_to_reaction"
        for batch in self._validation_loader(dataset, batch_size, target=target):
            batch, values = self._validation_encode(batch, target=target, direction=direction)
            rows = batch[row_key].to(device=self.device, dtype=torch.long)
            raw[rows] = values
            seen[rows] += 1
        # All ranks participate, including those with no batches. Each row
        # has exactly one owner; summing zeros plus its embedding is sufficient.
        if self._distributed_enabled():
            dist.all_reduce(raw, op=dist.ReduceOp.SUM)
            dist.all_reduce(seen, op=dist.ReduceOp.SUM)
        if not bool((seen == 1).all()):
            raise RuntimeError("Validation catalog has missing or duplicated owners")
        return raw

    def _validation_anchor_batches(self, dataset, batch_size, raw, lookup, *, target):
        # Directional adapters deliberately change embeddings; in that case
        # re-encode anchors instead of reusing the opposite-direction table.
        adapter = "r2e_adapter" if target else "e2r_adapter"
        direction = "enzyme_to_reaction" if target else "reaction_to_enzyme"
        id_key = "target_id" if target else "query_id"
        if getattr(self.model, adapter, None) is not None:
            for batch in self._validation_loader(dataset, batch_size, target=target):
                batch, values = self._validation_encode(batch, target=target, direction=direction)
                yield list(batch[id_key]), values
            return
        keys = dataset.keys
        data = self.trainer.datamodule
        collate = data._base_collate_fn if target else dict_collate_fn
        for rows in self._validation_shard(len(dataset), batch_size):
            ids = [keys[row] for row in rows]
            positions = [lookup.get(value) for value in ids]
            values = raw.new_empty((len(rows), raw.shape[1]))
            hits = [i for i, position in enumerate(positions) if position is not None]
            if hits:
                values[hits] = raw[[positions[i] for i in hits]]
                self._validation_work[2 if target else 3] += len(hits)
            missing = [i for i, position in enumerate(positions) if position is None]
            if missing:
                # Custom candidate catalogs need not contain every anchor.
                # Read only the missing features, never silently drop queries.
                batch = collate([dataset[rows[i]] for i in missing])
                _, encoded = self._validation_encode(batch, target=target, direction=direction)
                values[missing] = encoded
            yield ids, values

    def _validation_metric_names(self):
        names = ["mrr", "mean_rank", "reactzyme_mrr"]
        if self.validation_retrieval_candidate_chunk_size:
            names.extend(["r_precision", "avg_precision"])
            for k in self.retrieval_metric_top_k:
                names.extend([f"top_{k}", f"recall_{k}"])
        else:
            names.extend(name for name in ("r_precision", "avg_precision") if name in self.metric_functionals)
            names.extend(f"top_{k}" for k in self.retrieval_metric_top_k)
        return list(dict.fromkeys(names))

    def _validation_score_batches(self, batches, candidates, positive_lookup, candidate_lookup,
                                  scorer, names):
        totals = torch.zeros(len(names) + 1, device=self.device, dtype=torch.float64)
        for ids, anchors in batches:
            if self.validation_retrieval_candidate_chunk_size:
                blocks = range(0, len(ids), 32)
            else:
                blocks = (0,)
            for start in blocks:
                stop = start + 32 if self.validation_retrieval_candidate_chunk_size else len(ids)
                block, block_ids = anchors[start:stop], ids[start:stop]
                if self.validation_retrieval_candidate_chunk_size:
                    positives = [[candidate_lookup[p] for p in positive_lookup.get(q, ())
                                  if p in candidate_lookup] for q in block_ids]
                    with torch.autocast(device_type=self.device.type, enabled=False):
                        ranks = chunked_positive_ranks(
                            block.float(), candidates, positives,
                            chunk_size=self.validation_retrieval_candidate_chunk_size, score_fn=scorer)
                    metrics = rank_metrics(ranks, self.retrieval_metric_top_k)
                    count = sum(bool(row.numel()) for row in ranks)
                else:
                    count, metrics = self._batched_retrieval_metric_values(
                        scorer(block, candidates), block_ids, positive_lookup, candidate_lookup)
                totals[-1] += count
                for i, name in enumerate(names):
                    if name in metrics:
                        totals[i] += metrics[name].double().sum()
        return totals

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        if not self._fast_validation_enabled():
            return super().on_validation_epoch_end()
        if self.trainer.sanity_checking:
            return  # Do not run a full catalog during a short loss sanity check.
        data = self.trainer.datamodule
        retrieval_start = time.perf_counter()
        # DataLoader iterator construction consumes one CPU RNG seed even
        # with sequential sampling. Reserve the same draws as the legacy
        # retrieval loaders, independent of empty ranks and cache hits, then
        # isolate our actual I/O iterators from the subsequent training RNG.
        self._validation_loader_generator = torch.Generator()
        self._validation_loader_generator.set_state(torch.random.get_rng_state())
        for _ in range(2 + len(self.validation_retrieval_directions)):
            torch.empty((), dtype=torch.int64).random_()
        self._validation_work = torch.zeros(4, device=self.device, dtype=torch.long)
        batch_size = max(1, int(data.validation_retrieval_batch_size))
        # These raw embeddings are local to THIS validation check. Keep raw and
        # normalized forms separate, preserving the legacy normalization count.
        target_raw = self._validation_lookup(data._val_metric_target_lookup_data,
                                            data._val_retrieval_target_candidate_ids,
                                            data.train_batch_size, target=True)
        query_raw = self._validation_lookup(data._val_metric_query_lookup_data,
                                           data._val_retrieval_query_candidate_ids,
                                           batch_size, target=False)
        self.val_target_lookup_table = F.normalize(target_raw, p=2, dim=-1, eps=1e-12)
        self.val_query_lookup_table = F.normalize(query_raw, p=2, dim=-1, eps=1e-12)
        names = self._validation_metric_names()
        totals = torch.zeros((2, len(names) + 1), device=self.device, dtype=torch.float64)
        for i, direction in enumerate(self._BALANCED_DIRECTIONS):
            if direction not in self.validation_retrieval_directions:
                continue
            target = direction == "enzyme_to_reaction"
            dataset = data._val_metric_target_query_data if target else data._val_metric_query_data
            batches = self._validation_anchor_batches(
                dataset, batch_size, target_raw if target else query_raw,
                self.val_target_id_to_idx if target else self.val_query_id_to_idx, target=target)
            totals[i] = self._validation_score_batches(
                batches, self.val_query_lookup_table if target else self.val_target_lookup_table,
                data._target_to_queries if target else data._query_to_targets,
                self.val_query_id_to_idx if target else self.val_target_id_to_idx,
                self._compute_enzyme_to_reaction_scores if target else self._compute_retrieval_scores,
                names)
        if self._distributed_enabled():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(self._validation_work, op=dist.ReduceOp.SUM)
        means = (totals[:, :-1] / totals[:, -1:].clamp_min(1)).float()
        metrics = {}
        for i, direction in enumerate(self._BALANCED_DIRECTIONS):
            if direction in self.validation_retrieval_directions:
                metrics[f"val/{direction}/num_queries"] = totals[i, -1].float()
                metrics.update({f"val/{direction}/{name}": means[i, j] for j, name in enumerate(names)})
        if set(self.validation_retrieval_directions) == set(self._BALANCED_DIRECTIONS):
            for name in self._BALANCED_METRICS:
                if name in names:
                    a, b = means[:, names.index(name)]
                    metrics[f"val/mean_bidirectional_{name}"] = (a + b) * .5
                    metrics[f"val/balanced_{name}"] = self._harmonic_mean(a, b)
        elapsed = torch.tensor([time.perf_counter() - retrieval_start,
                                time.perf_counter() - self._validation_started], device=self.device)
        if self._distributed_enabled():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        metrics["val/retrieval_seconds"], metrics["val/total_seconds"] = elapsed
        # All ranks already have identical global means. Lightning's final
        # mean reduction is harmless and keeps its checkpoint/logger contract
        # satisfied without per-query or unequal-tail synchronization.
        self.log_dict(metrics, on_step=False, on_epoch=True, batch_size=1,
                      add_dataloader_idx=False, sync_dist=True)
        self.fast_validation_report = {
            "target_encodes": int(self._validation_work[0]),
            "query_encodes": int(self._validation_work[1]),
            "target_anchor_reuses": int(self._validation_work[2]),
            "query_anchor_reuses": int(self._validation_work[3]),
            "retrieval_seconds": float(elapsed[0]),
        }
        if self.trainer.is_global_zero:
            print(f"Sharded retrieval validation complete: {self.fast_validation_report}", flush=True)
