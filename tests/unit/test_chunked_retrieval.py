import pytest
import torch

from horizyn.benchmarks.chunked_retrieval import chunked_positive_ranks, rank_metrics


@pytest.mark.parametrize("chunk_size", [1, 3, 8, 100])
def test_chunked_ranks_match_dense_with_ties_and_multiple_positives(chunk_size):
    generator = torch.Generator().manual_seed(42)
    q = torch.randint(-2, 3, (4, 5), generator=generator).float()
    c = torch.randint(-2, 3, (19, 5), generator=generator).float()
    c[3] = c[1]
    c[8] = c[1]
    positives = [[1, 3, 8, 18], [0, 4, 4], [], list(range(19))]
    actual = chunked_positive_ranks(q, c, positives, chunk_size=chunk_size)
    ordering = (q @ c.T).argsort(descending=True, stable=True)
    inverse = ordering.argsort() + 1
    for i, row in enumerate(positives):
        assert torch.equal(actual[i], inverse[i, sorted(set(row))])
    metrics = rank_metrics(actual, [1, 5, 100])
    assert len(metrics["mrr"]) == 3
    assert metrics["recall_100"].tolist() == [1, 1, 1]
    assert metrics["avg_precision"][-1].item() == 1


def test_chunked_rejects_bad_catalog_and_nonfinite():
    with pytest.raises(ValueError, match="outside"):
        chunked_positive_ranks(torch.ones(1, 2), torch.ones(3, 2), [[3]])
    with pytest.raises(ValueError, match="Non-finite"):
        chunked_positive_ranks(torch.ones(1, 2), torch.full((3, 2), float("nan")), [[0]])


def test_zero_queries_and_zero_positives():
    assert chunked_positive_ranks(torch.empty(0, 2), torch.empty(0, 2), []) == []
    assert rank_metrics([torch.empty(0, dtype=torch.long)], [1]) == {}
