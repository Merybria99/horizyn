from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from generalization_clipzyme_f3_validation import evaluate_query
from generalization_clipzyme_screening_evaluate import notebook_metrics


def test_parallel_metric_parity_with_ties_and_exclusions():
    rng = np.random.default_rng(42)
    values = rng.integers(-4, 5, size=(12, 1207)).astype(np.float32)
    kept = np.arange(0, 1207, 2)
    arguments, expected = [], []
    for i, row in enumerate(values):
        positives = np.unique(rng.integers(0, 1207, size=30))
        labels = np.zeros(1207, dtype=bool)
        labels[positives] = True
        reduced = labels[kept]
        expected.append(dict(query_index=i, reaction_id=str(i),
            positives_table1=int(labels.sum()), positives_table2=int(reduced.sum()),
            table1=notebook_metrics(labels[np.argsort(-row)]),
            table2=notebook_metrics(reduced[np.argsort(-row[kept])]) if reduced.any() else None))
        arguments.append((i, str(i), row, positives, kept))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(evaluate_query, arguments)) == expected


def test_no_positive_after_training_enzyme_exclusion():
    result = evaluate_query((0, 'q', np.arange(100, dtype=np.float32),
                             np.array([1, 3]), np.arange(0, 100, 2)))
    assert result['table2'] is None
    assert result['positives_table2'] == 0


def test_missing_positives_and_nonfinite_scores():
    assert evaluate_query((0, 'q', np.ones(5), [], np.arange(5))) is None
    with pytest.raises(ValueError):
        evaluate_query((0, 'q', np.array([1., float('nan'), 2.]), np.array([0]), np.arange(3)))
