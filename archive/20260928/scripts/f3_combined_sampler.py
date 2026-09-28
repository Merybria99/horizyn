"""Reaction-balanced homolog competition for an isolated F3 training snapshot.

Only real positive training rows are sampled. A competitor enters through one
of its own positive associations; its unannotated cross-pair is never relabelled
positive. The all-known mask protects every known in-batch association.
"""
import random


def extend_sampler(base):
    class ReactionBalancedHomologSampler(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.direction != 'reaction_to_enzyme':
                raise ValueError('Combined pilot supports reaction-to-enzyme anchors only')
            # Include reactions without homolog competitors, rather than training
            # only on the subset with a nonempty hard pool.
            self.anchor_queries = sorted(self.query_to_indices)
            self.last_statistics = {}

        def __iter__(self):
            rank, world_size = self._rank_info()
            rng = random.Random(self.seed + self.epoch)
            order = list(self.anchor_queries)
            rng.shuffle(order)
            cursor = 0
            for batch_number in range(self._ddp_padded_total_batches()):
                anchors = []
                while len(anchors) < self.anchor_queries_per_batch:
                    if cursor == len(order):
                        rng.shuffle(order); cursor = 0
                    anchors.append(order[cursor]); cursor += 1
                batch, seen = [], set()
                for query in anchors:
                    for row in self._build_anchor_rows(query, rng):
                        if row not in seen:
                            batch.append(row); seen.add(row)
                        if len(batch) == self.batch_size:
                            break
                    if len(batch) == self.batch_size:
                        break
                structured_rows = len(batch)
                attempts = 0
                # Reaction-uniform filler, not edge-uniform filler. Cap attempts
                # to avoid nontermination for pathological tiny test datasets.
                while len(batch) < self.batch_size and attempts < self.batch_size * 100:
                    query = rng.choice(self.anchor_queries)
                    row = rng.choice(self.query_to_indices[query])
                    attempts += 1
                    if row not in seen:
                        batch.append(row); seen.add(row)
                if len(batch) < self.batch_size:
                    remaining = [i for i in self.all_indices if i not in seen]
                    rng.shuffle(remaining)
                    batch.extend(remaining[:self.batch_size-len(batch)])
                if len(batch) != self.batch_size or len(set(batch)) != self.batch_size:
                    raise ValueError('Cannot construct a complete distinct-row batch')
                rng.shuffle(batch)
                self.last_statistics = dict(batch_number=batch_number, anchors=anchors,
                                            structured_rows=structured_rows, total_rows=len(batch))
                if batch_number % world_size == rank:
                    yield batch
            self.epoch += 1
    return ReactionBalancedHomologSampler
