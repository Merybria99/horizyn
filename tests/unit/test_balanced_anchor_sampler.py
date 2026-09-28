from types import SimpleNamespace
from unittest.mock import patch

import pytest

from horizyn.balanced_anchor_sampler import BalancedAnchorBatchSampler
from horizyn.config import DotDict
from horizyn.training_options import reaction_data_module_kwargs


class Graph:
    def __init__(self):
        edges = [("singleton", "singleton")] + [("hub", f"e{i}") for i in range(30)]
        edges += [(f"r{i}", "common") for i in range(30)]
        self.keys = list(range(len(edges)))
        self.tuple_dataset = {i: dict(query_id=r, target_id=e, pair_type="positive")
                              for i, (r, e) in enumerate(edges)}

    def __len__(self):
        return len(self.keys)


def test_singletons_retained_and_batches_valid():
    graph = Graph()
    sampler = BalancedAnchorBatchSampler(graph, batch_size=16)
    assert sampler.reactions["singleton"] == [0]
    assert sampler.enzymes["singleton"] == [0]
    batches = list(sampler)
    assert [len(b) for b in batches] == [16, 16, 16, 13]
    assert all(len(b) == len(set(b)) for b in batches)
    assert all(0 <= i < len(graph) for b in batches for i in b)
    observed = set()
    for _ in range(10):
        observed.update(i for b in sampler for i in b)
    assert 0 in observed


def test_seed_epoch_and_distributed_partition():
    a = BalancedAnchorBatchSampler(Graph(), batch_size=8)
    b = BalancedAnchorBatchSampler(Graph(), batch_size=8)
    first = list(a)
    assert first == list(b)
    assert first != list(a)
    a.set_epoch(0)
    assert first == list(a)
    ranks = []
    for rank in range(4):
        a.set_epoch(2)
        with patch.object(a, "_rank_info", return_value=(rank, 4)):
            ranks.append(list(a))
    assert all([len(batch) for batch in r] == [8, 8] for r in ranks)
    assert len({i for rank in ranks for i in rank[0]}) == 32
    assert len({i for rank in ranks for i in rank[1]}) == 29


def test_rejects_negative_edges():
    graph = Graph()
    graph.tuple_dataset[0]["pair_type"] = "negative"
    with pytest.raises(ValueError, match="positive pairs"):
        BalancedAnchorBatchSampler(graph, batch_size=8)
