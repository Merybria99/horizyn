"""Training-edge batches drawn from uniform reaction and enzyme anchors."""
from __future__ import annotations

from collections import defaultdict
import math
import random

import torch.distributed as dist
from torch.utils.data import BatchSampler


class BalancedAnchorBatchSampler(BatchSampler):
    """Alternate uniformly drawn endpoint anchors, retaining singleton nodes.

    Uniform anchor choice does not imply uniform incidental node exposure:
    neighbors of an opposite-side anchor still follow the observed graph.
    No labels or candidates outside the supplied training dataset are used.
    """

    def __init__(self, dataset, *, batch_size, positives_per_anchor=4, seed=42):
        if batch_size < 1 or positives_per_anchor < 1 or len(dataset) < 1:
            raise ValueError("Positive batch/anchor sizes and a nonempty dataset are required")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.positives_per_anchor = int(positives_per_anchor)
        self.seed, self.epoch = int(seed), 0
        self.reactions, self.enzymes = defaultdict(list), defaultdict(list)
        for i, key in enumerate(dataset.keys):
            pair = dataset.tuple_dataset[key]
            if pair.get("pair_type", "positive") != "positive":
                raise ValueError("Balanced anchors require training positive pairs only")
            self.reactions[str(pair["query_id"])].append(i)
            self.enzymes[str(pair["target_id"])].append(i)
        self.reaction_ids, self.enzyme_ids = list(self.reactions), list(self.enzymes)

    @staticmethod
    def _rank_info():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        _, world = self._rank_info()
        return math.ceil(len(self.dataset) / (self.batch_size * world))

    def __iter__(self):
        rank, world = self._rank_info()
        size = self.batch_size * world
        rng = random.Random(self.seed + 100003 * self.epoch)
        for step in range(len(self)):
            wanted = min(size, len(self.dataset) - step * size)
            batch, seen, attempts = [], set(), 0
            while len(batch) < wanted and attempts < 20 * wanted:
                groups, keys = ((self.reactions, self.reaction_ids) if attempts % 2 == 0
                                else (self.enzymes, self.enzyme_ids))
                rows = groups[rng.choice(keys)]
                for i in rng.sample(rows, min(len(rows), self.positives_per_anchor)):
                    if i not in seen:
                        seen.add(i)
                        batch.append(i)
                        if len(batch) == wanted:
                            break
                attempts += 1
            # Only relevant when a requested batch nearly exhausts a tiny graph.
            if len(batch) < wanted:
                remaining = [i for i in range(len(self.dataset)) if i not in seen]
                batch.extend(rng.sample(remaining, wanted - len(batch)))
            rng.shuffle(batch)
            # Equal nonempty local batches keep distributed contrastive gathers
            # aligned. Only the final global batch can repeat a padding row.
            if world > 1:
                padding = (-len(batch)) % world
                batch.extend(batch[i % len(batch)] for i in range(padding))
            yield batch[rank::world]
        self.epoch += 1
