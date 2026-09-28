#!/usr/bin/env python3
"""Descriptive paired screening uncertainty with query and reaction-rule resampling."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ('bedroc85', 'bedroc20', 'ef0.05', 'ef0.1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def calculate(method, reference, queries, associations, replicates, seed):
    summaries = [json.loads((p / 'summary.json').read_text()) for p in (method, reference)]
    for key in ('protocol_receipt_sha256', 'query_ids_sha256', 'candidate_ids_sha256', 'notebook_sha256'):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f'Comparison protocol mismatch: {key}')
    rows = [[json.loads(line) for line in (p / 'per_query.jsonl').read_text().splitlines()]
            for p in (method, reference)]
    if len(rows[0]) != len(rows[1]):
        raise ValueError('Different numbers of query records')
    for a, b in zip(*rows):
        for key in ('query_index', 'reaction_id', 'positives_table1', 'positives_table2'):
            if a[key] != b[key]:
                raise ValueError(f'Query/positive alignment mismatch: {key}')

    rules = defaultdict(set)
    for row in read_rows(associations):
        rules[row['reaction']].add(row['rule_id'])
    query_rules = {}
    for row in read_rows(queries):
        query_rules[row['reaction_id']] = rules[row['reaction']]
    # A query may belong to multiple rules. Merge those rules into components
    # so a query is never duplicated or arbitrarily assigned to one rule.
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    for row in rows[0]:
        keys = sorted(query_rules[row['reaction_id']])
        if not keys:
            raise ValueError('A screening query has no test reaction-rule metadata')
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
        find(keys[0])
    groups = {rid: find(sorted(keys)[0]) for rid, keys in query_rules.items() if keys}
    result = {}
    rng = np.random.default_rng(seed)
    for table in ('table1', 'table2'):
        kept = [i for i, row in enumerate(rows[0]) if row[table] is not None]
        if any(rows[1][i][table] is None for i in kept):
            raise ValueError('Compared models exclude different queries')
        arrays = [np.array([[records[i][table][metric] for metric in METRICS] for i in kept], dtype=np.float64)
                  for records in rows]
        for array, summary in zip(arrays, summaries):
            expected = np.array([summary['summary'][table][metric] for metric in METRICS])
            if not np.allclose(array.mean(0), expected, rtol=0, atol=1e-10):
                raise ValueError('Per-query metrics do not reproduce the reported summary')
        delta = arrays[0] - arrays[1]
        labels = [groups[rows[0][i]['reaction_id']] for i in kept]
        unique = sorted(set(labels))
        index = {key: i for i, key in enumerate(unique)}
        group_index = np.array([index[key] for key in labels])
        sums = np.zeros((len(unique), len(METRICS)))
        np.add.at(sums, group_index, delta)
        counts = np.bincount(group_index, minlength=len(unique))
        query_draws, rule_draws = [], []
        for start in range(0, replicates, 100):
            n = min(100, replicates - start)
            q = rng.integers(0, len(kept), size=(n, len(kept)))
            query_draws.append(delta[q].mean(1))
            g = rng.integers(0, len(unique), size=(n, len(unique)))
            rule_draws.append(sums[g].sum(1) / counts[g].sum(1)[:, None])
        intervals = [np.quantile(np.concatenate(draws), [.025, .975], axis=0)
                     for draws in (query_draws, rule_draws)]
        result[table] = dict(queries=len(kept), rule_components=len(unique),
            component_query_counts=dict(zip(unique, counts.tolist())), metrics={
                metric: dict(delta=float(delta[:, j].mean()), query_ci95=intervals[0][:, j].tolist(),
                             rule_component_ci95=intervals[1][:, j].tolist())
                for j, metric in enumerate(METRICS)})
    sources = [method / 'summary.json', method / 'per_query.jsonl', reference / 'summary.json',
               reference / 'per_query.jsonl', queries, associations, Path(__file__)]
    return dict(replicates=replicates, seed=seed, comparison=result,
        interpretation='Descriptive, unadjusted paired percentile intervals after repeated test inspection. Rule components merge rules sharing a query; resampled component sums/counts preserve query-weighted means. Few rule groups, protein homology, seed variation and multiple comparisons limit inference. No uncertainty claim against published-only FGW-CLIP.',
        inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in sources])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('method', 'reference', 'queries', 'associations', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--replicates', type=int, default=5000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.replicates < 100:
        p.error('At least 100 bootstrap replicates are required')
    result = calculate(a.method, a.reference, a.queries, a.associations, a.replicates, a.seed)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparison'], indent=2))
