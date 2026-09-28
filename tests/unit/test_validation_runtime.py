import importlib.util
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from horizyn.training_io import training_io_adapters
from horizyn.validation_runtime import FastValidationLitModule, WholeBatchShardSampler

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("size", [0, 1, 3, 7, 25, 128])
@pytest.mark.parametrize("batch_size", [1, 3, 100])
@pytest.mark.parametrize("world", [1, 4, 8])
def test_whole_batches_have_exactly_one_owner(size, batch_size, world):
    batches = []
    for rank in range(world):
        sampler = WholeBatchShardSampler(size, batch_size, rank, world)
        local = list(sampler)
        assert len(local) == len(sampler)
        batches.extend(local)
    assert sorted(i for batch in batches for i in batch) == list(range(size))
    assert sorted(batches) == [list(range(i, min(i + batch_size, size))) for i in range(0, size, batch_size)]


def make_validation_fixture(path, fast, *, chunk_size=3, model_options=None):
    spec = importlib.util.spec_from_file_location("validation_fixture", ROOT / "tests/fixtures/multimodal_training.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    if fast:
        with training_io_adapters(fixture):
            data, module = fixture.make_multimodal_fixture(path, True, model_options=model_options)
    else:
        data, module = fixture.make_multimodal_fixture(path, True, model_options=model_options)
    data.validation_enabled = data.validation_retrieval_metrics = True
    data.validation_retrieval_candidate_set = "validation"
    # No singleton contrastive-loss batch; retrieval still has a 3+1 tail
    # and empty ranks when world_size=4.
    data.train_batch_size = 2
    data.validation_retrieval_batch_size = 3
    module.validation_retrieval_metrics = True
    module.validation_retrieval_candidate_chunk_size = chunk_size
    return data, module


def merged_metrics(rows):
    result = {}
    for row in rows:
        result.update(row)
    return result


@pytest.mark.parametrize("chunk_size", [0, 3])
def test_real_multimodal_validation_matches_and_rebuilds_cache(tmp_path, chunk_size):
    before_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        baseline_path, fast_path = tmp_path / "baseline", tmp_path / "fast"
        baseline_path.mkdir(); fast_path.mkdir()
        torch.manual_seed(42)
        baseline_data, baseline = make_validation_fixture(baseline_path, False, chunk_size=chunk_size)
        torch.manual_seed(42)
        fast_data, fast = make_validation_fixture(fast_path, True, chunk_size=chunk_size)
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, fast.state_dict()[key], rtol=0, atol=0)
        options = dict(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False,
                       enable_progress_bar=False, use_distributed_sampler=False)
        before_validation_rng = torch.random.get_rng_state()
        expected = merged_metrics(pl.Trainer(**options).validate(baseline, datamodule=baseline_data, verbose=False))
        expected_rng = torch.random.get_rng_state()
        torch.random.set_rng_state(before_validation_rng)
        trainer = pl.Trainer(**options)
        actual = merged_metrics(trainer.validate(fast, datamodule=fast_data, verbose=False))
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
        for key, value in expected.items():
            if key.endswith("/num_queries"):
                continue  # New diagnostic is total valid queries, not mean batch size.
            assert actual[key] == pytest.approx(value, abs=2e-6), key
        assert actual["val/reaction_to_enzyme/num_queries"] == 4
        assert actual["val/enzyme_to_reaction/num_queries"] == 4
        assert len(fast_data.val_dataloader()) == 1
        assert len(baseline_data.val_dataloader()) == 5
        report = fast.fast_validation_report
        assert [report[k] for k in ("target_encodes", "query_encodes", "target_anchor_reuses", "query_anchor_reuses")] == [4, 4, 4, 4]
        old_table = fast.val_target_lookup_table.clone()
        with torch.no_grad():
            for p in fast.model.parameters():
                if p.requires_grad:
                    p.add_(.1)
        trainer.validate(fast, datamodule=fast_data, verbose=False)
        assert not torch.allclose(old_table, fast.val_target_lookup_table)
        assert fast.fast_validation_report["target_encodes"] == 4  # Recomputed, not stale.
        assert set(fast.state_dict()) == set(baseline.state_dict())
    finally:
        torch.set_num_threads(before_threads)


class ToyAnchors(Dataset):
    keys = ["a", "b", "missing"]

    def __init__(self):
        self.reads = []

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        self.reads.append(i)
        return {"target_id": self.keys[i], "query_id": self.keys[i], "query_vec": torch.tensor([i + 1., 0.])}


@pytest.mark.parametrize("target", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
def test_anchor_reuse_missing_fallback_and_directional_guard(target, adapter):
    from horizyn.utils.collate import dict_collate_fn
    source = ToyAnchors()
    data = SimpleNamespace(_base_collate_fn=dict_collate_fn, _loader_worker_kwargs=lambda: {"num_workers": 0})
    harness = SimpleNamespace(
        trainer=SimpleNamespace(datamodule=data),
        model=SimpleNamespace(r2e_adapter=object() if target and adapter else None,
                              e2r_adapter=object() if not target and adapter else None),
        _validation_work=torch.zeros(4, dtype=torch.long),
        _validation_shard=lambda size, batch: WholeBatchShardSampler(size, batch),
    )
    harness._validation_loader = lambda *args, **kwargs: FastValidationLitModule._validation_loader(harness, *args, **kwargs)
    def encode(batch, *, target, direction):
        assert direction == ("enzyme_to_reaction" if target else "reaction_to_enzyme")
        return batch, batch["query_vec"]
    harness._validation_encode = encode
    raw = torch.tensor([[1., 0.], [2., 0.]])
    batches = list(FastValidationLitModule._validation_anchor_batches(
        harness, source, 3, raw, {"a": 0, "b": 1}, target=target))
    assert batches[0][0] == source.keys
    torch.testing.assert_close(batches[0][1], torch.tensor([[1., 0.], [2., 0.], [3., 0.]]))
    assert source.reads == ([0, 1, 2] if adapter else [2])


def test_chunked_shards_keep_ties_and_skip_unknown_positive_rows():
    from horizyn.benchmarks.chunked_retrieval import chunked_positive_ranks, rank_metrics
    queries = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 1.]])
    candidates = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    ids = list("abcde")
    positive = {"a": ["p1", "p2"], "b": ["not_in_catalog"], "c": ["p0"], "d": ["p2"], "e": []}
    lookup = {f"p{i}": i for i in range(3)}
    scorer = lambda q, c: q @ c.T
    names = ["mrr", "mean_rank", "reactzyme_mrr", "r_precision", "avg_precision", "top_1", "recall_1"]
    harness = SimpleNamespace(device=torch.device("cpu"), validation_retrieval_candidate_chunk_size=2,
                              retrieval_metric_top_k=[1])
    totals = torch.zeros(len(names) + 1, dtype=torch.float64)
    for rank in range(8):
        batches = [([ids[i] for i in rows], queries[rows]) for rows in WholeBatchShardSampler(5, 2, rank, 8)]
        totals += FastValidationLitModule._validation_score_batches(harness, batches, candidates, positive, lookup, scorer, names)
    expected = rank_metrics(chunked_positive_ranks(queries, candidates, [[lookup[p] for p in positive[q] if p in lookup] for q in ids], chunk_size=2), [1])
    assert totals[-1] == 3
    for i, name in enumerate(names):
        torch.testing.assert_close(totals[i] / totals[-1], expected[name].double().mean())
