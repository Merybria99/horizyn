#!/usr/bin/env python3
"""Evaluate predeclared frozen predictions on literature Case1 and P450.

Manifest schema (paths are relative to manifest directory unless absolute)::

  {"freeze": {"path": "frozen_recipe.json", "sha256": "..."},
   "baseline_method": "f3_epoch29_corrected",
   "methods": [
     {"label": "f3_epoch29_corrected", "primary": false,
      "panels": {"case1": {"path": "case1/scores.npz", "score_key": "baseline"},
                 "p450": {"path": "p450/scores.npz", "score_key": "baseline"}}},
     {"label": "combined_seed42", "primary": true,
      "panels": {"case1": {"path": "case1/scores.npz", "score_key": "selected"},
                 "p450": {"path": "p450/scores.npz", "score_key": "selected"}}}],
   "historical_case1": {"path": "case1_audit/features/f3_epoch29/features.npz"}}

Each score entry must have an adjacent complete.json prediction receipt or an
explicit sha256 plus catalog_sha256 (optionally receipt path). Native prediction receipts also bind
the feature catalog and frozen bundle. Exactly one nonbaseline primary method
is required. Additional seeds/ensembles are diagnostics, never selected here.
Optional case1_embedding_controls entries contain label, path and query_key
(query_matched or query_historical). These native F3/CIRCE checkpoint controls
are nonprimary and need not supply P450 scores.
No real-score evaluation is allowed without a frozen held-out recipe manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generalization_metrics import evaluate_scores


CUTS = (1, 5, 10, 20)
CASE_CUTS = (5, 10, 25)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(base, value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def checked_identity(path, expected=None):
    value = identity(path)
    if expected is not None and value['sha256'] != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return value


def rank_vector(scores):
    """Stable candidate-catalog order resolves exact score ties."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def conditional_auc_ap(scores, positive):
    """AUC gives ties half credit; AP groups tied scores as one threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    npos, nneg = int(positive.sum()), int((~positive).sum())
    if not npos or not nneg:
        return dict(auc=None, average_precision=None, positive=npos, negative=nneg)
    auc = np.mean((scores[positive, None] > scores[~positive][None, :]) +
                  .5 * (scores[positive, None] == scores[~positive][None, :]))
    order = np.argsort(-scores, kind='stable')
    labels, ranked = positive[order], scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked) - 1]
    true = np.cumsum(labels)[endpoints]
    precision = true / (endpoints + 1)
    ap = np.sum(np.diff(np.r_[0, true]) * precision) / npos
    return dict(auc=float(auc), average_precision=float(ap), positive=npos, negative=nneg)


def random_hit_probability(n, positives, k):
    k = min(k, n)
    if k > n - positives:
        return 1.0
    return 1.0 - math.prod((n - positives - i) / (n - i) for i in range(k))


def recall_summary(ranks, indices):
    indices = np.asarray(sorted(indices), dtype=np.int64)
    selected = ranks[indices]
    out = {'positive_count': len(indices), 'best_positive_rank': int(selected.min()) if len(indices) else None}
    for k in CASE_CUTS:
        hit = int((selected <= k).sum())
        out[f'recovered_at_{k}'] = hit
        out[f'recall_at_{k}'] = hit / len(indices) if len(indices) else None
        out[f'random_expected_recovered_at_{k}'] = min(k, len(ranks)) * len(indices) / len(ranks)
        out[f'random_expected_recall_at_{k}'] = min(k, len(ranks)) / len(ranks) if len(indices) else None
    return out


def case1_metadata(directory):
    catalog = json.loads((directory / 'features/catalog.json').read_text())
    evidence = json.loads((directory / 'candidate_evidence.json').read_text())
    sets = json.loads((directory / 'positive_sets.json').read_text())
    sources = json.loads((directory / 'source_verification.json').read_text())
    group = {v['representative_id']: v for v in catalog['groups']}
    records = {r['entry_id']: r for r in evidence['records']}
    sha_index = {group[p]['sequence_sha256']: i for i, p in enumerate(catalog['proteins'])}
    if len(sha_index) != 123 or len(catalog['proteins']) != 123:
        raise ValueError('Case1 must retain the fixed 123 unique sequences')
    primary = set(sets['primary_rechecked_positive_unique_sequences'])
    paper = {r['sequence_sha256'] for r in evidence['records']
             if r['primary_source_rechecked'] and r['primary_source_rechecked'] not in ('S10', 'S19', 'S20')}
    broad = set(sets['workbook_all_reported_active_unique_sequences'])
    nondetect = set(sets['workbook_non_detect_only_unique_sequences'])
    if (len(paper), len(primary), len(broad), len(nondetect)) != (12, 24, 81, 42):
        raise ValueError('Frozen Case1 label counts changed')
    if broad & nondetect or broad | nondetect != set(sha_index):
        raise ValueError('Case1 broad labels must partition the unique candidate pool')
    conflict = {x['sequence_sha256'] for x in sets['cross_assay_conflicts']}
    if conflict & nondetect:
        raise ValueError('Cross-assay conflicts cannot enter the negative class')
    labels = {'primary_papers': paper, 'primary_papers_and_patents': primary, 'broad_workbook_active': broad}
    indices = {key: {sha_index[s] for s in values} for key, values in labels.items()}
    studies = defaultdict(set)
    families = defaultdict(set)
    for r in evidence['records']:
        if r['sequence_sha256'] not in primary or not r['primary_source_rechecked']:
            continue
        i = sha_index[r['sequence_sha256']]
        studies[r['primary_source_rechecked']].add(i)
        # Recorded parent accession identifies WT/mutant construct lineage, not
        # a phylogenetically inferred enzyme family.
        family = r['parent_identifier'] or ('organism:' + r['organism'])
        families[family].add(i)
    return dict(catalog=catalog, group=group, records=records, indices=indices,
                positive=indices['broad_workbook_active'], conflict=conflict,
                studies=studies, families=families,
                sources={x['Source ID']: x for x in sources})


def evaluate_case1(scores, metadata, label):
    scores = scores.reshape(-1)
    ranks = rank_vector(scores)
    summary = {name: recall_summary(ranks, ids) for name, ids in metadata['indices'].items()}
    y = np.array([i in metadata['positive'] for i in range(len(scores))])
    summary['broad_assay_conditional_discrimination'] = conditional_auc_ap(scores, y)
    summary['broad_assay_conditional_discrimination'].update(
        interpretation='81 workbook-reported active vs 42 non-detect-only sequences; conditional and incompletely independently verified assays; conflicts excluded from negative class.',
        random_auc=.5, random_large_sample_ap_prevalence=float(y.mean()),
        random_expected_ap_untied=(sum(1 / i for i in range(1, len(y) + 1)) / len(y) +
            (int(y.sum()) - 1) * (len(y) - sum(1 / i for i in range(1, len(y) + 1))) / (len(y) * (len(y) - 1))))
    rows = []
    for i, p in enumerate(metadata['catalog']['proteins']):
        group = metadata['group'][p]
        evidence = [metadata['records'][x] for x in group['all_entry_ids']]
        row = dict(method=label, representative_id=p, sequence_sha256=group['sequence_sha256'],
                   entry_ids=';'.join(group['all_entry_ids']), rank=int(ranks[i]), score=float(scores[i]),
                   primary_paper=i in metadata['indices']['primary_papers'],
                   primary_paper_or_patent=i in metadata['indices']['primary_papers_and_patents'],
                   broad_reported_active=bool(y[i]), non_detect_only=not bool(y[i]),
                   cross_assay_conflict=group['sequence_sha256'] in metadata['conflict'],
                   names=';'.join(dict.fromkeys(x['name'] for x in evidence)),
                   organisms=';'.join(dict.fromkeys(x['organism'] for x in evidence)),
                   source_ids=';'.join(sorted({s for x in evidence for s in x['source_ids']})),
                   primary_source_ids=';'.join(sorted({x['primary_source_rechecked'] for x in evidence if x['primary_source_rechecked']})))
        rows.append(row)
    group_rows = []
    for kind, groups in [('study', metadata['studies']), ('construct_lineage', metadata['families'])]:
        for name, indices in sorted(groups.items()):
            row = dict(method=label, grouping=kind, group=name, **recall_summary(ranks, indices))
            if kind == 'study':
                row.update(citation=metadata['sources'][name]['Citation'], url=metadata['sources'][name]['URL'])
            else:
                row.update(citation='', url='')
            group_rows.append(row)
    summary['uncertainty'] = 'One reaction; related constructs and reused studies are dependent. No candidate-IID confidence intervals are reported.'
    return summary, rows, group_rows


def p450_metadata(directory, split):
    catalog = json.loads((directory / 'features/catalog.json').read_text())
    labels = json.loads((directory / 'labels.json').read_text())
    strata = json.loads((directory / 'evaluation_strata.json').read_text())['splits'][split]
    if labels['catalog_sha256'] != sha256(directory / 'features/catalog.json'):
        raise ValueError('P450 label catalog checksum mismatch')
    qids, pids = catalog['query_ids'], catalog['proteins']
    qi, pi = ({q: i for i, q in enumerate(qids)}, {p: i for i, p in enumerate(pids)})
    edges = [(qi[q], pi[p]) for q, positives in labels['known_positive_ids'].items() for p in positives]
    if len(qids) != 191 or len(pids) != 490 or len(edges) != 318 or len(set(edges)) != len(edges):
        raise ValueError('P450 fixed panel dimensions/edges changed')
    absent = set(strata['molecule_pair_not_contained_in_any_training_set'])
    both = set(strata['chemistry_and_all_known_positive_sequences_absent'])
    if len(absent) != 163 or (split == 'reaction_smi' and len(both) != 126):
        raise ValueError('P450 declared overlap strata changed')
    subsets = {'all': set(qids), 'molecule_pair_absent': absent,
               'molecule_pair_present': set(qids) - absent, 'both_absent': both,
               'not_both_absent': set(qids) - both}
    return dict(catalog=catalog, reaction=np.asarray([x[0] for x in edges]),
                enzyme=np.asarray([x[1] for x in edges]),
                subsets={name: {qi[q] for q in values} for name, values in subsets.items()})


def evaluate_p450(scores, metadata):
    """Restrict stratum positive edges, never either candidate dimension."""
    matrix = torch.as_tensor(scores)
    results, all_ranks = {}, None
    for name, qset in metadata['subsets'].items():
        mask = np.asarray([int(q) in qset for q in metadata['reaction']])
        truth = dict(reaction_index=metadata['reaction'][mask], enzyme_index=metadata['enzyme'][mask])
        result = evaluate_scores(matrix, truth)
        results[name] = {}
        if name == 'all':
            all_ranks = result['per_positive']
        for direction in result['summary']:
            summary = result['summary'][direction]['all']
            per = result['per_query'][direction]['all']
            for k in CUTS:
                per[f'top_{k}'] = (per['first_rank'] <= k).astype(np.float64)
                summary[f'top_{k}'] = float(per[f'top_{k}'].mean()) if len(per['first_rank']) else None
            candidate_count = scores.shape[1] if direction == 'reaction_to_enzyme' else scores.shape[0]
            summary['candidate_count'] = candidate_count
            summary['random_expected_all_positive_mrr'] = sum(1 / i for i in range(1, candidate_count + 1)) / candidate_count
            summary['random_expected_hits'] = {
                str(k): float(np.mean([random_hit_probability(candidate_count, int(n), k) for n in per['positive_count']]))
                if len(per['positive_count']) else None for k in CUTS}
            results[name][direction] = dict(summary=summary, per_query=per)
    return results, all_ranks


def paired_bootstrap(method, baseline, replicates, seed):
    """Direction/stratum query resamples are shared across all method contrasts."""
    if not np.array_equal(method['query_index'], baseline['query_index']):
        raise ValueError('Paired bootstrap query identities differ')
    n = len(method['query_index'])
    if not n:
        return {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(replicates, n))
    out = {}
    for key in ['reactzyme_mrr', *[f'top_{k}' for k in CUTS]]:
        delta = np.asarray(method[key], dtype=np.float64) - np.asarray(baseline[key], dtype=np.float64)
        samples = delta[draws].mean(axis=1)
        low, high = np.quantile(samples, [.025, .975])
        out[key] = dict(delta=float(delta.mean()), lower_95=float(low), upper_95=float(high), queries=n,
                        replicates=replicates, seed=int(seed), bootstrap_unit='retrieval query',
                        interpretation='Paired percentile query bootstrap; query independence approximation, no multiplicity adjustment, not candidate-IID uncertainty.')
    return out


def reaction_permutation_control(scores, metadata, replicates=1000, seed=20260919):
    """Fixed gold graph and candidate pool; randomly replace query score rows.

    One shared permutation over ALL reaction rows is used for each draw, also
    when averaging a strict query stratum. Fixed points are allowed (uniform
    permutations, not derangements). No new negatives or candidate filtering.
    """
    nquery, ncandidate = scores.shape
    order = np.argsort(-scores, axis=1, kind='stable')
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(1, ncandidate + 1)[None, :], axis=1)
    donor_metrics = {key: np.empty((nquery, nquery), dtype=np.float64)
                     for key in ('reactzyme_mrr', 'top_5', 'top_10')}
    for q in range(nquery):
        positives = metadata['enzyme'][metadata['reaction'] == q]
        if not len(positives):
            raise ValueError('Permutation control requires known positives for every reaction')
        selected = ranks[:, positives]
        donor_metrics['reactzyme_mrr'][:, q] = np.mean(1. / selected, axis=1)
        first = selected.min(axis=1)
        donor_metrics['top_5'][:, q] = first <= 5
        donor_metrics['top_10'][:, q] = first <= 10
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(nquery) for _ in range(replicates)])
    output = {}
    for stratum in ('all', 'both_absent'):
        queries = np.asarray(sorted(metadata['subsets'][stratum]), dtype=np.int64)
        output[stratum] = {}
        for metric, table in donor_metrics.items():
            correct = float(table[queries, queries].mean())
            null = table[permutations[:, queries], queries[None, :]].mean(axis=1)
            low, high = np.quantile(null, [.025, .975])
            output[stratum][metric] = dict(correct=correct, null_mean=float(null.mean()),
                null_lower_95=float(low), null_upper_95=float(high),
                correct_minus_null_mean=correct-float(null.mean()),
                one_sided_p_null_ge_correct=float((1 + np.count_nonzero(null >= correct)) / (replicates + 1)),
                empirical_fraction_correct_ge_null=float(np.mean(correct >= null)),
                queries=len(queries), replicates=replicates, seed=seed,
                interpretation='Query-conditioning diagnostic, uniform full-panel reaction permutations with fixed gold edges and candidate universe. Null 95% range is not a confidence interval; exchangeability and dependence limit formal inference.')
    return output


def load_scores(entry, manifest_dir, catalog_path, shape, provenance, freeze_identity):
    path = resolve(manifest_dir, entry['path'])
    score_identity = checked_identity(path, entry.get('sha256'))
    receipt_path = resolve(manifest_dir, entry['receipt']) if 'receipt' in entry else path.parent / 'complete.json'
    receipt = None
    catalog_bound = entry.get('catalog_sha256') == sha256(catalog_path)
    if 'catalog_sha256' in entry and not catalog_bound:
        raise ValueError(f'Explicit score catalog checksum mismatch: {path}')
    if receipt_path.exists():
        provenance[str(receipt_path)] = identity(receipt_path)
        receipt = json.loads(receipt_path.read_text())
        expected = receipt.get('output_sha256') or receipt.get('scores_sha256')
        if expected and expected != score_identity['sha256']:
            raise ValueError(f'Prediction receipt checksum mismatch: {path}')
        if not expected and 'sha256' not in entry:
            raise ValueError(f'Receipt lacks score checksum: {receipt_path}')
        source_catalog = receipt.get('inputs', {}).get('catalog')
        if source_catalog:
            if source_catalog['sha256'] != sha256(catalog_path):
                raise ValueError(f'Prediction catalog does not match evaluation order: {path}')
            catalog_bound = True
        bundle = receipt.get('bundle')
        if bundle:
            bundle_path = resolve(receipt_path.parent, bundle['path'])
            provenance[str(bundle_path)] = checked_identity(bundle_path, bundle['sha256'])
            bundle_data = json.loads(bundle_path.read_text())
            if bundle_data['frozen_recipe']['sha256'] != freeze_identity['sha256']:
                raise ValueError('Prediction bundle is tied to a different frozen recipe')
        if receipt.get('labels_used', False) or receipt.get('activity_labels_used', False):
            raise ValueError('Prediction receipt reports activity/association labels used')
    elif 'sha256' not in entry:
        raise ValueError(f'Score checksum or complete prediction receipt required: {path}')
    if not catalog_bound:
        raise ValueError(f'Score order must be bound by a native receipt or explicit catalog_sha256: {path}')
    with np.load(path, allow_pickle=False) as values:
        key = entry.get('score_key', 'scores')
        if key not in values:
            raise ValueError(f'{path} lacks score key {key}')
        scores = np.asarray(values[key])
        if scores.shape == (shape[1],) and shape[0] == 1:
            scores = scores[None, :]
        for key, expected in [('protein_ids', json.loads(catalog_path.read_text())['proteins']),
                              ('reaction_ids', json.loads(catalog_path.read_text())['query_ids'])]:
            if key in values and values[key].astype(str).tolist() != expected:
                raise ValueError(f'Score ID order mismatch: {path}:{key}')
    if scores.shape != shape or not np.isfinite(scores).all():
        raise ValueError(f'Expected finite scores {shape}; got {scores.shape}: {path}')
    used_keys = set(provenance.get(str(path), {}).get('score_keys', [])) | {entry.get('score_key', 'scores')}
    provenance[str(path)] = {**score_identity, 'score_keys': sorted(used_keys),
                             'catalog': identity(catalog_path), 'receipt': str(receipt_path) if receipt else None}
    return scores


def case1_embedding_control(entry, manifest_dir, output, catalog_path, provenance):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from horizyn.generalization_retrieval import canonical_dot
    score_source = Path(__file__).resolve().parents[1] / 'horizyn/generalization_retrieval.py'
    provenance[str(score_source)] = identity(score_source)
    path = resolve(manifest_dir, entry['path'])
    provenance[str(path)] = checked_identity(path, entry.get('sha256'))
    receipt_path = path.parent / 'complete.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('reference_labels_used', False) or receipt.get('activity_labels_used', False):
        raise ValueError('Native control export used external evaluation labels')
    if receipt['catalog']['sha256'] != sha256(catalog_path):
        raise ValueError('Native Case1 control catalog differs')
    if receipt['output']['sha256'] != sha256(path):
        raise ValueError('Native Case1 control learned feature checksum differs')
    provenance[str(receipt_path)] = identity(receipt_path)
    with np.load(path) as value:
        proteins = torch.from_numpy(value['proteins']).float()
        query_key = entry.get('query_key', 'query_historical')
        if query_key not in ('query_historical', 'query_matched'):
            raise ValueError('Case1 control query_key must be historical or matched')
        query = torch.from_numpy(value[query_key]).float()
    with torch.inference_mode():
        score = canonical_dot(F.normalize(query, dim=1), F.normalize(proteins, dim=1)).numpy()
    if score.shape != (1, 123) or not np.isfinite(score).all():
        raise ValueError('Native Case1 control score shape/nonfinite error')
    label = entry.get('label', 'F3_epoch29_historical_inputs_diagnostic')
    if entry.get('primary') or not label or any(not (c.isalnum() or c in '_-') for c in label):
        raise ValueError('Embedding controls must be nonprimary with a safe method label')
    target = output / (label + '_scores.npz')
    np.savez(target, scores=score)
    return score, dict(source=identity(path), receipt=identity(receipt_path), output=identity(target),
                      method=label, primary=False, query_key=query_key, checkpoint=receipt['checkpoint'], score_contract='One FP32 normalization of each raw cached endpoint; shared canonical_dot FP64 accumulation rounded to FP32.',
                      interpretation='Frozen native checkpoint, identical unique candidate sequences; input protocol control only. Prior saved epoch28 metrics remain separate and are not recomputed or relabelled.')


def write_readout(path, summary, comparisons):
    lines = ['# Frozen external evaluation', '',
             f"Primary method: **{summary['primary_method']}**. Corrected-input reference: **{summary['baseline_method']}**.",
             'All declared methods are reported; this evaluation makes no model choice.', '',
             '## Case1: one literature-derived reaction, 123 unique sequences', '',
             '| Method | Paper positives @5 /10 /25 (of12) | All primary @5 /10 /25 (of24) | Conditional AUC | Conditional AP |',
             '|---|---|---|---:|---:|']
    for method, values in summary['case1'].items():
        paper = values['primary_papers']; allprimary = values['primary_papers_and_patents']
        counts = lambda x: ' / '.join(str(x[f'recovered_at_{k}']) for k in CASE_CUTS)
        diagnostic = values['broad_assay_conditional_discrimination']
        lines.append(f"| {method} | {counts(paper)} | {counts(allprimary)} | {diagnostic['auc']:.4f} | {diagnostic['average_precision']:.4f} |")
    lines += ['', 'Uniform random ranking recovers on average 0.49 / 0.98 / 2.44 of the 12 paper positives and 0.98 / 1.95 / 4.88 of the 24 primary positives at 5 / 10 / 25.',
              'Conditional AUC/AP use 81 workbook-reported active versus 42 non-detect-only sequences. Seven cross-assay conflicts are excluded from the negative class. These are condition-dependent literature labels; source-screen negatives have not all been independently rechecked.',
              'Case1 supplies one reaction with related constructs and overlapping studies. No candidate-IID confidence intervals are reported. Study and recorded-parent construct-lineage summaries are in `case1_study_and_lineage_summary.csv`; these are not phylogenetic family tests. All ranks and source identifiers are retained in the companion CSVs.', '',
              '## P450: full candidate universe in every stratum', '',
              '| Method | Stratum | Direction | Queries | All-positive MRR | H1 | H5 | H10 | H20 |',
              '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for method, strata in summary['p450'].items():
        for stratum, directions in strata.items():
            for direction, metrics in directions.items():
                short = 'R2E' if direction == 'reaction_to_enzyme' else 'E2R'
                values = ' | '.join(f"{metrics[key]:.4f}" if metrics[key] is not None else 'NA'
                                    for key in ['reactzyme_mrr', 'top_1', 'top_5', 'top_10', 'top_20'])
                lines.append(f"| {method} | {stratum} | {short} | {metrics['num_queries']} | {values} |")
    lines += ['', 'All-positive MRR averages reciprocal ranks over every known positive within a query, then averages queries. H@k means at least one known positive. Reaction strata restrict truth edges while retaining all 490 enzyme and 191 reaction candidates. Enzyme queries without any stratum-positive edge are omitted from that stratum average.',
              'Zero-labelled pairs are unlisted associations, not experimentally demonstrated inactivity. Scores are raw model scores; the historical EnzymeCAGE homology prior is not applied. Random retrieval expectations are stored with each summary.', '',
              '## Paired contrasts against corrected F3', '',
              '| Method | Stratum | Direction | Metric | Difference | Paired 95% interval |',
              '|---|---|---|---|---:|---|']
    for row in comparisons:
        if not row['primary'] or row['metric'] not in ('reactzyme_mrr', 'top_20'):
            continue
        direction = 'R2E' if row['direction'] == 'reaction_to_enzyme' else 'E2R'
        lines.append(f"| {row['method']} | {row['stratum']} | {direction} | {row['metric']} | {row['delta']:.4f} | [{row['lower_95']:.4f}, {row['upper_95']:.4f}] |")
    lines += ['', 'Query-bootstrap draws are paired with the reference and shared across methods within each direction/stratum. These intervals assume query independence and do not account for all reaction/protein dependence or multiple comparisons. All seed and diagnostic contrasts are preserved in `p450_paired_bootstrap.csv`.',
              'Input hashes, frozen-recipe identity, score receipts and evaluator hashes are recorded in `complete.json`. Historical Case1 input scores use the same epoch29 checkpoint; the older saved epoch28 analysis remains separate.']
    lines += ['', '## Reaction-conditioning control on P450', '',
              '| Method | Stratum | Metric | Correct | Permuted mean | Null 95% range | One-sided null ≥ correct p |',
              '|---|---|---|---:|---:|---|---:|']
    for method, strata in summary['p450_reaction_permutation'].items():
        for stratum, metrics in strata.items():
            for metric, value in metrics.items():
                lines.append(f"| {method} | {stratum} | {metric} | {value['correct']:.4f} | {value['null_mean']:.4f} | [{value['null_lower_95']:.4f}, {value['null_upper_95']:.4f}] | {value['one_sided_p_null_ge_correct']:.4f} |")
    lines += ['', 'All methods use the same 1,000 fixed-seed permutations of complete reaction-score rows; gold associations and both candidate universes stay fixed. The strict subset is averaged after the full-panel permutation. This diagnoses conditioning on the requested reaction, including comparison with a generic enzyme prior. It introduces no alternative negative-label definition and makes no model choice. Null ranges are permutation reference ranges, not confidence intervals; tail probabilities are exploratory under an exchangeability assumption. No Case1 permutation is possible for its single query.']
    Path(path).write_text('\n'.join(lines) + '\n')


def run(args):
    manifest_dir = args.manifest.resolve().parent
    manifest = json.loads(args.manifest.read_text())
    freeze_entry = manifest['freeze']
    freeze_path = resolve(manifest_dir, freeze_entry['path'])
    freeze_identity = checked_identity(freeze_path, freeze_entry.get('sha256'))
    freeze = json.loads(freeze_path.read_text())
    if not freeze.get('frozen_before_held_out_evaluation'):
        raise ValueError('A recipe frozen before held-out evaluation is required')
    methods = manifest['methods']
    labels = [m['label'] for m in methods]
    baseline_label = manifest['baseline_method']
    if len(labels) != len(set(labels)) or baseline_label not in labels:
        raise ValueError('Method labels must be unique and include the corrected F3 baseline')
    primary = [m for m in methods if m.get('primary') and m['label'] != baseline_label]
    if len(primary) != 1 or next(m for m in methods if m['label'] == baseline_label).get('primary'):
        raise ValueError('Exactly one nonbaseline primary method is required')
    if any(set(m['panels']) != {'case1', 'p450'} for m in methods):
        raise ValueError('Every declared main method must supply both complete panels')
    if args.bootstrap_replicates < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Evaluation outputs are immutable; choose an empty directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    case_dir, p450_dir = args.audit_root / 'case1_audit', args.audit_root / 'p450_audit'
    case = case1_metadata(case_dir)
    p450 = p450_metadata(p450_dir, args.training_split)
    provenance = {str(args.manifest.resolve()): identity(args.manifest), str(freeze_path): freeze_identity}
    for path in [case_dir / 'features/catalog.json', case_dir / 'candidate_evidence.json',
                 case_dir / 'positive_sets.json', case_dir / 'source_verification.json',
                 case_dir / 'features/raw/manifest.json', case_dir / 'features/f3_epoch29/complete.json',
                 p450_dir / 'features/catalog.json', p450_dir / 'labels.json',
                 p450_dir / 'evaluation_strata.json', p450_dir / 'f3_train_overlap.json',
                 p450_dir / 'features/raw/manifest.json', p450_dir / 'features/f3_epoch29/complete.json']:
        provenance[str(path.resolve())] = identity(path)
    loaded = {}
    for method in methods:
        loaded[method['label']] = {
            'case1': load_scores(method['panels']['case1'], manifest_dir, case_dir / 'features/catalog.json', (1, 123), provenance, freeze_identity),
            'p450': load_scores(method['panels']['p450'], manifest_dir, p450_dir / 'features/catalog.json', (191, 490), provenance, freeze_identity)}
    # All files are validated before any ranking is computed.
    case_results, rank_rows, group_rows, p450_results, positive_rows = {}, [], [], {}, []
    for label, panels in loaded.items():
        summary, rows, groups = evaluate_case1(panels['case1'], case, label)
        case_results[label] = summary
        rank_rows += rows
        group_rows += groups
        result, ranks = evaluate_p450(panels['p450'], p450)
        p450_results[label] = result
        for direction, values in ranks.items():
            positive_rows += [dict(method=label, direction=direction,
                reaction_id=p450['catalog']['query_ids'][int(q)], protein_id=p450['catalog']['proteins'][int(p)],
                rank=int(rank)) for q, p, rank in zip(values['reaction_index'], values['enzyme_index'], values['rank'])]
    embedding_controls = []
    controls = list(manifest.get('case1_embedding_controls', []))
    if manifest.get('historical_case1'):
        controls.insert(0, manifest['historical_case1'])
    for control in controls:
        label = control.get('label', 'F3_epoch29_historical_inputs_diagnostic')
        if label in case_results:
            raise ValueError('Case1 embedding-control method label collided')
        score, control_receipt = case1_embedding_control(control, manifest_dir, args.output,
                                                       case_dir / 'features/catalog.json', provenance)
        embedding_controls.append(control_receipt)
        summary, rows, groups = evaluate_case1(score, case, label)
        case_results[label] = summary
        rank_rows += rows
        group_rows += groups
    permutation_results = {label: reaction_permutation_control(panels['p450'], p450, 1000, args.seed)
                           for label, panels in loaded.items()}
    p450_summaries, per_query_rows, comparisons = {}, [], []
    for label, strata in p450_results.items():
        p450_summaries[label] = {}
        for stratum, directions in strata.items():
            p450_summaries[label][stratum] = {}
            for direction, value in directions.items():
                p450_summaries[label][stratum][direction] = value['summary']
                per = value['per_query']
                query_ids = p450['catalog']['query_ids'] if direction == 'reaction_to_enzyme' else p450['catalog']['proteins']
                for row_index, query_index in enumerate(per['query_index']):
                    per_query_rows.append(dict(method=label, stratum=stratum, direction=direction,
                        query_id=query_ids[int(query_index)], first_rank=int(per['first_rank'][row_index]),
                        positive_count=int(per['positive_count'][row_index]),
                        reactzyme_mrr=float(per['reactzyme_mrr'][row_index]),
                        **{f'top_{k}': int(per[f'top_{k}'][row_index]) for k in CUTS}))
                if label != baseline_label:
                    stable_offset = int.from_bytes(hashlib.sha256((stratum + '/' + direction).encode()).digest()[:4], 'big')
                    contrast = paired_bootstrap(per, p450_results[baseline_label][stratum][direction]['per_query'],
                                                args.bootstrap_replicates, args.seed + stable_offset)
                    for metric, values in contrast.items():
                        comparisons.append(dict(method=label, baseline=baseline_label,
                            primary=next(bool(m.get('primary')) for m in methods if m['label'] == label),
                            stratum=stratum, direction=direction, metric=metric, **values))
    summary = dict(schema='frozen_external_evaluation_v1', baseline_method=baseline_label,
        primary_method=primary[0]['label'], methods=methods, case1=case_results, p450=p450_summaries,
        case1_embedding_controls=embedding_controls, p450_reaction_permutation=permutation_results,
        protocol=dict(score_orientation='reaction rows, enzyme columns; larger score ranks first',
            tie_policy='stable feature-catalog candidate order', score_contract='Prediction artifacts use canonical_dot FP64 accumulation rounded to FP32; baseline endpoints each normalized once in FP32.', p450_candidates=dict(reactions=191, enzymes=490),
            p450_mrr='For each retrieval query, mean reciprocal rank of every known-positive candidate; macro-average over queries with at least one stratum-positive edge.',
            p450_hits='At least one stratum-positive candidate at or above cutoff; H1/5/10/20.',
            p450_strata='Restrict positive edges by reaction overlap strata; retain all reaction and enzyme candidates, including candidates outside the stratum.',
            p450_prior='None: raw model ranking; historical official-prior baseline metrics are separate.',
            p450_zero_labels='Unlisted catalytic associations, not experimentally assayed inactivity.',
            p450_permutation='1000 fixed-seed full-panel reaction-row permutations, same permutations for every method; all and strict both-absent strata; no Case1 permutation.',
            p450_uncertainty='Paired query bootstrap within each direction and stratum; shared resamples across methods. Query dependence and multiplicity are not corrected.',
            case1_evidence='12 primary-paper, 24 primary paper/patent, 81 workbook-active, 42 conditional non-detect-only unique sequences. Seven cross-assay conflicts are excluded from negative class.',
            case1_grouping='Studies may overlap. Construct-lineage summaries use recorded parent accession, falling back to organism; they are not phylogenetic family tests.',
            case1_uncertainty='Single query; no candidate-IID intervals.', model_selection='None; all declared methods reported; primary method supplied by frozen manifest.',
            training_overlap_split=args.training_split))
    write_json(args.output / 'summary.json', summary)
    write_readout(args.output / 'readout.md', summary, comparisons)
    write_csv(args.output / 'case1_all_candidates.csv', sorted(rank_rows, key=lambda x: (x['method'], x['rank'])))
    write_csv(args.output / 'case1_positive_ranks.csv', [r for r in rank_rows if r['broad_reported_active']])
    write_csv(args.output / 'case1_study_and_lineage_summary.csv', group_rows)
    write_csv(args.output / 'p450_per_query.csv', per_query_rows)
    write_csv(args.output / 'p450_positive_ranks.csv', positive_rows)
    write_csv(args.output / 'p450_paired_bootstrap.csv', comparisons)
    write_json(args.output / 'p450_paired_bootstrap.json', comparisons)
    write_json(args.output / 'p450_reaction_permutation.json', permutation_results)
    write_csv(args.output / 'p450_reaction_permutation.csv', [dict(method=label, stratum=stratum, metric=metric, **values)
        for label, strata in permutation_results.items() for stratum, metrics in strata.items() for metric, values in metrics.items()])
    receipt = dict(schema='frozen_external_evaluation_receipt_v1', frozen_before_held_out_evaluation=True,
        freeze=freeze_identity, methods_manifest=identity(args.manifest), source_inputs=provenance,
        evaluator=identity(__file__), metric_helper=identity(Path(__file__).with_name('generalization_metrics.py')),
        bootstrap=dict(replicates=args.bootstrap_replicates, seed=args.seed),
        reaction_permutation=dict(replicates=1000, seed=args.seed),
        outputs={p.name: identity(p) for p in sorted(args.output.iterdir()) if p.is_file()})
    write_json(args.output / 'complete.json', receipt)
    print(json.dumps(dict(complete=True, output=str(args.output.resolve()), primary=primary[0]['label'], methods=labels)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-split', choices=['reaction_smi', 'enzyme_smi', 'time'], default='reaction_smi')
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260919)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
