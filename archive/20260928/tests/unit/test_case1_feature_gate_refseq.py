import pytest

from scripts.run_case1_feature_gate_refseq import partition_ranges


@pytest.mark.parametrize("count,workers", [(4, 4), (11, 3), (3944613, 4)])
def test_partition_is_complete_disjoint_and_balanced(count, workers):
    ranges = partition_ranges(count, workers)
    assert ranges[0][0] == 0 and ranges[-1][1] == count
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    lengths = [end - start for start, end in ranges]
    assert sum(lengths) == count and max(lengths) - min(lengths) <= 1


@pytest.mark.parametrize("count,workers", [(0, 1), (2, 3), (5, 0)])
def test_reject_invalid_partition(count, workers):
    with pytest.raises(ValueError):
        partition_ranges(count, workers)
