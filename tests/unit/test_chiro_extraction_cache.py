import importlib.util
import sys
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "extract_chiro_reaction_embeddings.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("extract_chiro_reaction_embeddings", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chiro = _load_script()


class _TrackingFuture(Future):
    def __init__(self, executor, value):
        super().__init__()
        self.executor = executor
        self.set_result(value)

    def result(self, timeout=None):
        if not getattr(self, "_counted", False):
            self.executor.retained_results -= 1
            self._counted = True
        return super().result(timeout)


class _ImmediateExecutor:
    def __init__(self):
        self.retained_results = 0
        self.max_retained_results = 0

    def submit(self, function, item):
        self.retained_results += 1
        self.max_retained_results = max(self.max_retained_results, self.retained_results)
        return _TrackingFuture(self, function(item))


def test_bounded_executor_never_retains_more_than_configured_limit():
    executor = _ImmediateExecutor()
    results = list(
        chiro.bounded_executor_results(
            executor,
            lambda value: value * 2,
            range(100),
            max_pending=7,
        )
    )

    assert sorted(results) == [value * 2 for value in range(100)]
    assert executor.max_retained_results <= 7
    assert executor.retained_results == 0


def test_molecule_cache_round_trip_and_incremental_resume(tmp_path):
    cache_path = tmp_path / "chiro.sqlite3"
    signature = "test-signature"
    first = {
        "C": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "CC": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
    }

    with chiro.MoleculeEmbeddingCache(cache_path, signature) as cache:
        cache.put_embeddings(first)
        cache.put_skipped(["invalid"])

    with chiro.MoleculeEmbeddingCache(cache_path, signature) as cache:
        cache.put_embeddings({"CCC": np.asarray([7.0, 8.0, 9.0], dtype=np.float32)})

    with chiro.MoleculeEmbeddingCache(cache_path, signature, read_only=True) as cache:
        embeddings, skipped = cache.load(["C", "CC", "CCC", "invalid", "missing"])

    assert set(embeddings) == {"C", "CC", "CCC"}
    np.testing.assert_array_equal(embeddings["C"], first["C"])
    np.testing.assert_array_equal(embeddings["CCC"], np.asarray([7.0, 8.0, 9.0], dtype=np.float32))
    assert skipped == {"invalid"}


def test_molecule_cache_rejects_incompatible_signature(tmp_path):
    cache_path = tmp_path / "chiro.sqlite3"
    with chiro.MoleculeEmbeddingCache(cache_path, "signature-a") as cache:
        cache.put_embeddings({"C": np.ones(2, dtype=np.float32)})

    with pytest.raises(ValueError, match="incompatible"):
        chiro.MoleculeEmbeddingCache(cache_path, "signature-b")


def test_bounded_executor_rejects_zero_pending_limit():
    with pytest.raises(ValueError, match="positive"):
        list(chiro.bounded_executor_results(_ImmediateExecutor(), lambda value: value, [1], 0))
