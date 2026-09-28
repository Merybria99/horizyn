#!/usr/bin/env python3
"""Paired screening deltas, with query and reaction-rule cluster bootstrap."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from generalization_full_graph import atomic_json, sha
from generalization_clipzyme_screening_protocol import identifier

METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def rule_clusters(path):
    rules = defaultdict(set)
    with path.open() as stream:
        for row in csv.DictReader(stream):
            rules[identifier(row['reaction'])].add(row['rule_id'])
    parent = {r: r for group in rules.values() for r in group}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for group in rules.values():
        values = sorted(group)
        for r in values[1:]:
            parent[root(r)] = root(values[0])
    # A query annotated with two rules connects those rules. Never split the
    # same query across nominally independent bootstrap clusters.
    return {key: root(sorted(group)[0]) for key, group in rules.items()}


def compare(base_path, candidate_path, rules_path, samples=2000):
    def read(path):
        rows = [json.loads(s) for s in path.read_text().splitlines() if s]
        result = {r['reaction_id']: r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate query identifier')
        return result
    base, candidate = read(base_path), read(candidate_path)
    if set(base) != set(candidate):
        raise ValueError('Compared query sets differ')
    cluster_map = rule_clusters(rules_path)
    results = {}
    for table in ('table1', 'table2'):
        keys = sorted(k for k in base if base[k][table] is not None)
        if set(keys) != {k for k in candidate if candidate[k][table] is not None}:
            raise ValueError('Compared table denominators differ')
        if not set(keys) <= cluster_map.keys():
            raise ValueError('Query lacks an official reaction-rule assignment')
        b = np.array([[base[k][table][m] for m in METRICS] for k in keys])
        c = np.array([[candidate[k][table][m] for m in METRICS] for k in keys])
        d = c - b
        unique = sorted({cluster_map[k] for k in keys}); index = {k: i for i, k in enumerate(unique)}
        group = np.array([index[cluster_map[k]] for k in keys])
        counts = np.bincount(group); sums = np.zeros((len(unique), len(METRICS)))
        np.add.at(sums, group, d)
        rng = np.random.default_rng(1701)
        query_samples, rule_samples = [], []
        for start in range(0, samples, 64):
            size = min(64, samples - start)
            query_samples.append(d[rng.integers(len(keys), size=(size, len(keys)))].mean(1))
            chosen = rng.integers(len(unique), size=(size, len(unique)))
            rule_samples.append(sums[chosen].sum(1) / counts[chosen].sum(1)[:, None])
        qlo, qhi = np.quantile(np.concatenate(query_samples), [.025, .975], axis=0)
        rlo, rhi = np.quantile(np.concatenate(rule_samples), [.025, .975], axis=0)
        results[table] = dict(queries=len(keys), rule_clusters=len(unique), metrics={m: dict(
            baseline_mean=float(b[:, i].mean()), candidate_mean=float(c[:, i].mean()),
            paired_delta=float(d[:, i].mean()), query_bootstrap_95=[float(qlo[i]), float(qhi[i])],
            rule_cluster_bootstrap_95=[float(rlo[i]), float(rhi[i])],
            improved_queries=int((d[:, i] > 0).sum()), worsened_queries=int((d[:, i] < 0).sum()),
            tied_queries=int((d[:, i] == 0).sum())) for i, m in enumerate(METRICS)})
    return dict(results=results, baseline=str(base_path), candidate=str(candidate_path),
        baseline_sha256=sha(base_path), candidate_sha256=sha(candidate_path),
        rules_sha256=sha(rules_path), bootstrap_samples=samples, seed=1701,
        interpretation='Descriptive paired uncertainty for fixed trained models, not variability across training seeds. '
            'Rule clusters join reaction rules sharing a query; repeated experimental comparisons are not multiplicity-adjusted.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--rules', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.baseline, a.candidate, a.rules)
    atomic_json(a.output, result)
    print(json.dumps(result['results'], indent=2))


if __name__ == '__main__':
    main()
