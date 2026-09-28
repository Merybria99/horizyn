#!/usr/bin/env python3
"""Secondary numerical-sensitivity audit; does not replace official metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generalization_full_graph import atomic_json, sha


def bounds(scores, reactions, enzymes, tolerance):
    optimistic, pessimistic, near = [], [], []
    for start in range(0, len(reactions), 256):
        r, e = reactions[start:start+256], enzymes[start:start+256]
        values = scores[:, e].T
        positive = scores[r, e][:, None]
        optimistic.extend((1 + (values > positive + tolerance).sum(1)).tolist())
        pessimistic.extend((values >= positive - tolerance).sum(1).tolist())
        near.extend(((np.abs(values - positive) <= tolerance).sum(1) - 1).tolist())
    count = np.bincount(enzymes, minlength=scores.shape[1]); valid = count > 0
    def macro(rank):
        return float((np.bincount(enzymes, weights=1. / np.asarray(rank), minlength=len(count))[valid] / count[valid]).mean())
    return dict(tolerance=tolerance, optimistic_mrr=macro(optimistic), pessimistic_mrr=macro(pessimistic),
        positive_edges_with_near_competitor=int(np.count_nonzero(near)), positive_edges=len(reactions),
        mean_near_competitors=float(np.mean(near)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--variant', action='append')
    p.add_argument('--followup-layout', action='store_true')
    a = p.parse_args(); results = []
    for target in ('reaction_smi', 'enzyme_smi', 'time'):
        for variant in (a.variant or ('control', 'all_0p03', 'all_0p1', 'all_0p3', 'shuffled_0p1')):
            parent = a.campaign / 'followup' / target if a.followup_layout else a.campaign
            run = parent / target / variant
            scores = np.load(run / 'test_scores.npy', mmap_mode='r')
            with np.load(run / 'test_ranks.npz') as z:
                r = z['enzyme_to_reaction_reaction_index']; e = z['enzyme_to_reaction_enzyme_index']
            record = dict(target=target, variant=variant, score_sha256=sha(run / 'test_scores.npy'),
                bounds=[bounds(scores, r, e, epsilon) for epsilon in (0., 1e-7, 1e-6, 1e-5)])
            results.append(record); print(json.dumps(record), flush=True)
    atomic_json(a.campaign / 'tie_sensitivity.json', dict(records=results,
        interpretation='Tolerance bands around each positive score give diagnostic rank bounds, not a new official metric or a consistent global tie ordering.',
        official_metric_unchanged=True))


if __name__ == '__main__':
    main()
