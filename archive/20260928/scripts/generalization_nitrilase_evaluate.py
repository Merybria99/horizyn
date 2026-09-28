#!/usr/bin/env python3
"""Evaluate the preregistered complete nitrilase assay panel after model freeze.

All score files, their frozen recipes, and the preregistration must authenticate
before the first pairwise label is read. No candidate or assay filtering and no
model selection occur here. Exact score ties have uniform-order expectations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generalization_external_evaluate import (checked_identity, conditional_auc_ap,
    identity, load_scores, resolve, sha256, write_csv, write_json)

CUTS = (1, 3, 5)
RETRIEVAL = ('all_positive_mrr', 'first_positive_mrr', 'hit_at_1', 'hit_at_3', 'hit_at_5')
DISCRIMINATION = ('auroc', 'average_precision')


def query_metrics(labels, scores):
    """Exact expected ranks/hits for uniformly permuted equal-score candidates."""
    y, s = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or not len(y) or y.shape != s.shape:
        raise ValueError('Invalid query shapes')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError('Labels must be binary and scores finite')
    ptotal = int(y.sum())
    out = dict(positive_count=ptotal, candidate_count=len(y),
        no_positive=int(ptotal == 0), all_positive=int(ptotal == len(y)),
        all_positive_mrr=0., first_positive_mrr=0.,
        **{f'hit_at_{k}': 0. for k in CUTS})
    order = np.argsort(-s, kind='stable')
    ranked, gold = s[order], y[order]
    edges = np.r_[0, np.flatnonzero(np.diff(ranked)) + 1, len(y)]
    earlier_positive = 0
    tie_candidates = 0
    for start, stop in zip(edges[:-1], edges[1:]):
        size, positive = int(stop - start), int(gold[start:stop].sum())
        tie_candidates += size if size > 1 else 0
        if ptotal:
            out['all_positive_mrr'] += positive / size * sum(1. / j for j in range(start + 1, stop + 1)) / ptotal
        if positive and not earlier_positive:
            den = math.comb(size, positive)
            out['first_positive_mrr'] = sum(
                math.comb(size - j, positive - 1) / den / (start + j)
                for j in range(1, size - positive + 2))
        for k in CUTS:
            take = max(0, min(size, min(k, len(y)) - int(start)))
            if take and earlier_positive:
                out[f'hit_at_{k}'] = 1.
            elif take and positive:
                out[f'hit_at_{k}'] = (1. if take > size - positive else
                    1. - math.comb(size - positive, take) / math.comb(size, take))
        earlier_positive += positive
    discrimination = conditional_auc_ap(s, y.astype(bool))
    out.update(auroc=discrimination['auc'], average_precision=discrimination['average_precision'],
        unique_scores=len(edges) - 1, candidates_in_ties=tie_candidates,
        all_scores_tied=bool(len(edges) == 2))
    return out


def evaluate_matrix(scores, labels):
    if scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Score and complete assay matrix shapes disagree')
    result = {}
    for direction, s, y in [('reaction_to_enzyme', scores, labels),
                            ('enzyme_to_reaction', scores.T, labels.T)]:
        per = [dict(query_index=i, **query_metrics(row_y, row_s))
               for i, (row_y, row_s) in enumerate(zip(y, s))]
        positive = [r for r in per if r['positive_count']]
        mixed = [r for r in per if 0 < r['positive_count'] < r['candidate_count']]
        mean = lambda rows, keys: {k: float(np.mean([r[k] for r in rows])) if rows else None for k in keys}
        summary = dict(query_count=len(per), candidate_count=s.shape[1],
            positive_query_count=len(positive), no_positive_query_count=len(per)-len(positive),
            mixed_class_query_count=len(mixed), omitted_discrimination_query_count=len(per)-len(mixed),
            all_positive_query_count=sum(r['all_positive'] for r in per),
            all_queries=mean(per, RETRIEVAL), positive_queries_only=mean(positive, RETRIEVAL),
            mixed_class_queries_only=mean(mixed, DISCRIMINATION),
            all_scores_tied_queries=sum(r['all_scores_tied'] for r in per))
        random = [query_metrics(row, np.zeros(len(row))) for row in y]
        summary['random_expected_all_queries'] = mean(random, RETRIEVAL)
        summary['random_expected_positive_queries'] = mean([r for r in random if r['positive_count']], RETRIEVAL)
        summary['constant_score_mixed_class_reference'] = mean([r for r in random if r['auroc'] is not None], DISCRIMINATION)
        result[direction] = dict(summary=summary, per_query=per)
    return result


def paired_bootstrap(method, baseline, replicates=10000, seed=20260920):
    """Queries are shared across methods; no candidate-wise pseudo-replication."""
    out = {}
    for direction in method:
        left, right = method[direction]['per_query'], baseline[direction]['per_query']
        if [r['query_index'] for r in left] != [r['query_index'] for r in right]:
            raise ValueError('Paired query IDs disagree')
        out[direction] = {}
        for subset, keys, allowed in [
            ('all_queries', RETRIEVAL, lambda r: True),
            ('positive_queries_only', RETRIEVAL, lambda r: r['positive_count'] > 0),
            ('mixed_class_queries_only', DISCRIMINATION, lambda r: r['auroc'] is not None)]:
            indices = [i for i, r in enumerate(left) if allowed(r)]
            if indices != [i for i, r in enumerate(right) if allowed(r)]:
                raise ValueError('Bootstrap eligibility differs between methods')
            stream = seed + int(hashlib.sha256((direction + subset).encode()).hexdigest()[:8], 16)
            draw = np.random.default_rng(stream).integers(0, len(indices), size=(replicates, len(indices))) if indices else None
            out[direction][subset] = {}
            for key in keys:
                if not indices:
                    out[direction][subset][key] = None
                    continue
                delta = np.asarray([left[i][key] - right[i][key] for i in indices])
                samples = delta[draw].mean(axis=1)
                low, high = np.quantile(samples, [.025, .975])
                out[direction][subset][key] = dict(delta=float(delta.mean()), lower_95=float(low),
                    upper_95=float(high), query_count=len(indices), replicates=replicates, seed=stream,
                    interpretation='Paired percentile query bootstrap; substrate/enzyme dependence limits independence, no candidate-IID interval and no multiplicity correction.')
    return out


def permutation_control(scores, labels, replicates=1000, seed=20260920):
    """Permute all reaction score rows, retaining the measured matrix and pools."""
    observed = evaluate_matrix(scores, labels)
    rng = np.random.default_rng(seed)
    permutations = np.stack([rng.permutation(len(scores)) for _ in range(replicates)])
    # R2E donor lookup preserves exact tie expectations and is independent of draws.
    lookup = [[query_metrics(labels[q], scores[donor]) for donor in range(len(scores))]
              for q in range(len(scores))]
    keys = list(RETRIEVAL + DISCRIMINATION)
    null = {direction: {key: [] for key in keys} for direction in observed}
    for permutation in permutations:
        rows = [lookup[q][int(permutation[q])] for q in range(len(scores))]
        columns = [query_metrics(labels[:, p], scores[permutation, p]) for p in range(scores.shape[1])]
        for direction, records in [('reaction_to_enzyme', rows), ('enzyme_to_reaction', columns)]:
            for key in keys:
                values = [r[key] for r in records if r[key] is not None]
                null[direction][key].append(float(np.mean(values)) if values else np.nan)
    out = {}
    for direction, metrics in null.items():
        out[direction] = {}
        for key, values in metrics.items():
            subset = 'all_queries' if key in RETRIEVAL else 'mixed_class_queries_only'
            correct = observed[direction]['summary'][subset][key]
            if correct is None:
                out[direction][key] = None
                continue
            values = np.asarray(values)
            low, high = np.quantile(values, [.025, .975])
            out[direction][key] = dict(correct=correct, null_mean=float(values.mean()),
                null_lower_95=float(low), null_upper_95=float(high),
                correct_minus_null_mean=correct-float(values.mean()),
                one_sided_p_null_ge_correct=float((1 + np.count_nonzero(values >= correct)) / (replicates + 1)),
                replicates=replicates, seed=seed, subset=subset,
                interpretation='Reaction-conditioning diagnostic; uniform full-panel reaction-row permutations with fixed measured labels, candidates and possible fixed points. Null range is not a confidence interval. Exchangeability and shared enzyme/substrate dependence limit inference.')
    return out


def load_labels(catalog, path, expected_count):
    qi = {q: i for i, q in enumerate(catalog['query_ids'])}
    pi = {p: i for i, p in enumerate(catalog['proteins'])}
    labels = np.full((len(qi), len(pi)), -1, dtype=np.int8)
    with Path(path).open() as handle:
        for r in csv.DictReader(handle):
            if r['query_id'] not in qi or r['protein_id'] not in pi or r['label'] not in ('0', '1'):
                raise ValueError('Unknown assay ID or nonbinary label')
            index = qi[r['query_id']], pi[r['protein_id']]
            if labels[index] != -1:
                raise ValueError('Duplicate assay cell')
            labels[index] = int(r['label'])
    if (labels < 0).any() or labels.size != expected_count:
        raise ValueError('Every assay cell must be measured exactly once')
    return labels


def validate_underlying_release(root, catalog, labels, lock):
    """Check source binary and finite workbook without profile-based ID guesses."""
    from activity_panels import canonical_molecule, sequence_hash
    import openpyxl
    binary = root / lock['source_binary_locked']['path']
    workbook = root / lock['source_workbook_locked']['path']
    source_profiles = {}
    source_labels = {}
    with binary.open() as handle:
        for r in csv.DictReader(handle):
            if r['Conversion'] not in ('0', '1'):
                raise ValueError('Invalid released binary label')
            substrate = canonical_molecule(r['SUBSTRATES'])
            p = 'nit_p_' + sequence_hash(r['SEQ'])[:16]
            q = 'nit_r_' + hashlib.sha256(substrate.encode()).hexdigest()[:16]
            if (q, p) in source_labels:
                raise ValueError('Duplicate released assay')
            source_labels[q, p] = int(r['Conversion'])
            source_profiles.setdefault(p, {})[substrate] = int(r['Conversion'])
    expected = {(q, p): int(labels[i, j]) for i, q in enumerate(catalog['query_ids']) for j, p in enumerate(catalog['proteins'])}
    if source_labels != expected:
        raise ValueError('Locked assay labels disagree with released binary source')
    wb = openpyxl.load_workbook(workbook, data_only=True, read_only=True)
    values = list(wb.active.values)
    wb.close()
    accessions = values[0][3:]
    workbook_profiles = {a: {} for a in accessions}
    count = 0
    for row in values[1:]:
        if not isinstance(row[1], str):
            continue
        substrate = canonical_molecule(row[2])
        if len(row[3:]) != len(accessions):
            raise ValueError('Ragged underlying assay row')
        for accession, value in zip(accessions, row[3:]):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Missing/invalid workbook value cannot become a negative')
            if substrate in workbook_profiles[accession]:
                raise ValueError('Duplicate workbook substrate')
            workbook_profiles[accession][substrate] = int(value > 0)
            count += 1
    substrates = sorted(next(iter(source_profiles.values())))
    profile = lambda d: tuple(d[s] for s in substrates)
    if count != 684 or Counter(map(profile, source_profiles.values())) != Counter(map(profile, workbook_profiles.values())):
        raise ValueError('Released binary profiles and finite underlying workbook disagree')
    return dict(measured_cells=count, finite_nonnegative_workbook=True,
        released_binary_matches_locked_pairs=True, binary_profile_multiset_matches_workbook=True,
        accession_policy='No accession/sequence mapping inferred from indistinguishable activity profiles.',
        alternate_threshold_sensitivity='Not preregistered; not computed.')


def load_phase2_scores(entry, base, catalog_path, provenance, freeze, shape=(38, 18)):
    """Phase2 bundles retain the phase1 input lineage and pin a second freeze."""
    path = resolve(base, entry['path'])
    receipt_path = resolve(base, entry['receipt']) if 'receipt' in entry else path.parent / 'complete.json'
    receipt = json.loads(receipt_path.read_text())
    record = receipt.get('bundle')
    if not record:
        raise ValueError('Phase2 scores require a complete prediction receipt and frozen bundle')
    bundle_path = resolve(receipt_path.parent, record['path'])
    checked_identity(bundle_path, record['sha256'])
    bundle = json.loads(bundle_path.read_text())
    phase2 = bundle.get('phase2_frozen_recipe')
    if not phase2 or phase2['sha256'] != freeze['sha256']:
        raise ValueError('Prediction bundle belongs to a different phase2 freeze')
    phase2_path = resolve(bundle_path.parent, phase2['path'])
    checked_identity(phase2_path, phase2['sha256'])
    if str(phase2_path) != freeze['path']:
        raise ValueError('Prediction phase2 freeze path differs from declared freeze')
    original = bundle['frozen_recipe']
    original_path = resolve(bundle_path.parent, original['path'])
    original_identity = checked_identity(original_path, original['sha256'])
    frozen = json.loads(phase2_path.read_text())
    if frozen['original_frozen_recipe']['sha256'] != original_identity['sha256']:
        raise ValueError('Phase2 and original feature-lineage freezes disagree')
    provenance[str(original_path)] = original_identity
    return load_scores(entry, base, catalog_path, shape, provenance, original_identity)


def run(args):
    root = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    freeze = checked_identity(resolve(base, manifest['freeze']['path']), manifest['freeze']['sha256'])
    frozen = json.loads(Path(freeze['path']).read_text())
    if frozen.get('schema') != 'phase2_generalization_frozen_recipe_v1' or not frozen.get('frozen_before_new_external_evaluation'):
        raise ValueError('Require the completed phase2 model freeze before label access')
    prereg_path = resolve(base, manifest['preregistration']['path'])
    prereg_identity = checked_identity(prereg_path, manifest['preregistration']['sha256'])
    prereg = json.loads(prereg_path.read_text())
    if prereg['schema'] != 'nitrilase_phase2_preregistration_v1':
        raise ValueError('Unrecognized preregistration')
    if sha256(__file__) != prereg['evaluator_sha256']:
        raise ValueError('Evaluator changed after preregistration')
    for record in prereg['source_dependencies']:
        checked_identity(record['path'], record['sha256'])
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Preserve existing evaluation outputs; use a fresh directory')
    audit = resolve(base, prereg['audit_directory'])
    catalog_path = audit / 'features/catalog.json'
    checked_identity(catalog_path, prereg['catalog_sha256'])
    catalog = json.loads(catalog_path.read_text())
    if len(catalog['proteins']) != 18 or len(catalog['query_ids']) != 38:
        raise ValueError('The full 38x18 candidate panel is required')
    if len(set(catalog['protein_sequence_sha256'].values())) != 18:
        raise ValueError('Unexpected duplicate exact sequences: preregistration must precede any dedup change')
    if len(set(catalog['query_ids'])) != 38 or len(catalog['reactions']) != 38:
        raise ValueError('Expected one fixed variant per unique assay query')
    selection_path = audit / 'selection_lock.json'
    checked_identity(selection_path, prereg['selection_lock_sha256'])
    lock = json.loads(selection_path.read_text())
    provenance = {str(p): identity(p) for p in [manifest_path, prereg_path, selection_path, catalog_path, Path(__file__)]}
    provenance[freeze['path']] = freeze
    for key in ['labels_locked', 'source_binary_locked', 'source_workbook_locked', 'panel_manifest']:
        record = lock[key]
        p = root / record['path']
        provenance[str(p)] = checked_identity(p, record['sha256'])
    methods = manifest['methods']
    labels = [m['label'] for m in methods]
    if len(set(labels)) != len(labels) or sum(bool(m.get('primary')) for m in methods) != 1:
        raise ValueError('Require distinct methods and one preselected primary method')
    baseline_name = manifest['baseline_method']
    if baseline_name not in labels or any(m.get('primary') and m['label'] == baseline_name for m in methods):
        raise ValueError('Require a separate frozen F3 baseline')
    if frozen.get('canonical_reference_method') != baseline_name:
        raise ValueError('Paired reference differs from the frozen recipe')
    scores = {m['label']: load_phase2_scores(m['scores'], base, catalog_path, provenance, freeze) for m in methods}
    args.output.mkdir(parents=True, exist_ok=True)
    # This receipt is written only after every model/score/catalog is authenticated.
    write_json(args.output / 'label_open_receipt.json', dict(opened_at_utc=datetime.now(timezone.utc).isoformat(),
        freeze=freeze, preregistration=prereg_identity, manifest=identity(manifest_path),
        labels=lock['labels_locked'], all_model_scores_authenticated_before_labels=True))
    y = load_labels(catalog, root / lock['labels_locked']['path'], 684)
    if int(y.sum()) != 85 or int((y.sum(1) == 0).sum()) != 3 or int((y.sum(0) == 0).sum()) != 8:
        raise ValueError('Locked aggregate label counts changed')
    source_validation = validate_underlying_release(root, catalog, y, lock)
    results = {label: evaluate_matrix(s, y) for label, s in scores.items()}
    boot = {label: paired_bootstrap(result, results[baseline_name], prereg['bootstrap_replicates'], prereg['seed'])
            for label, result in results.items() if label != baseline_name}
    permutations = {label: permutation_control(s, y, prereg['permutation_replicates'], prereg['seed'])
                    for label, s in scores.items()}
    per_query = []
    pairs = []
    for label, result in results.items():
        for direction, records in result.items():
            ids = catalog['query_ids'] if direction == 'reaction_to_enzyme' else catalog['proteins']
            per_query.extend(dict(method=label, direction=direction, query_id=ids[r['query_index']], **r) for r in records['per_query'])
        pairs.extend(dict(method=label, query_id=q, protein_id=p, label=int(y[i,j]), score=float(scores[label][i,j]))
                     for i,q in enumerate(catalog['query_ids']) for j,p in enumerate(catalog['proteins']))
    summary = dict(schema='nitrilase_frozen_evaluation_v1', primary_method=next(m['label'] for m in methods if m.get('primary')),
        baseline_method=baseline_name, panel=dict(reactions=38, enzymes=18, pairs=684, active=85, non_detect=599,
            no_positive_reactions=3, no_positive_enzymes=8, exact_sequence_duplicates=0),
        label_policy=lock['label_policy'], prior_exposure=lock['prior_exposure'], source_validation=source_validation,
        metrics={label:{d:r['summary'] for d,r in value.items()} for label,value in results.items()},
        method_roles={m['label']:('primary' if m.get('primary') else 'baseline' if m['label']==baseline_name else 'diagnostic') for m in methods},
        limitations=prereg['limitations'], provenance=provenance)
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'paired_bootstrap.json', boot)
    write_json(args.output/'reaction_permutation.json', permutations)
    write_csv(args.output/'per_query.csv', per_query)
    write_csv(args.output/'all_assay_scores.csv', pairs)
    text = ['# Frozen nitrilase assay evaluation', '', f"Primary method: `{summary['primary_method']}`; baseline: `{baseline_name}`.", '',
        'All 684 measured enzyme–substrate cells are retained. The 599 zero labels are assay-condition non-detects. This panel was previously evaluated in the workspace; it was sequestered for this phase, not globally untouched.', '',
        'Exact-score ties use uniform-order expected retrieval metrics. All-query retrieval averages assign zero to queries without positives; positive-query averages are reported separately. AUROC/AP exclude single-class queries. No global AUC is a primary endpoint.', '',
        '| Method | Direction | All-positive MRR | First-positive MRR | H1 | H3 | H5 | Mixed-query AUROC | Mixed-query AP |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, result in results.items():
        for direction, value in result.items():
            a=value['summary']['all_queries']; b=value['summary']['mixed_class_queries_only']
            row=[a[k] for k in RETRIEVAL]+[b[k] for k in DISCRIMINATION]
            text.append('| '+label+' | '+direction+' | '+' | '.join('NA' if x is None else f'{x:.4f}' for x in row)+' |')
    text.extend(['', 'Uncertainty is a paired query bootstrap, with dependence across substrates and enzymes limiting formal inference. Reaction-row permutation assesses query conditioning; its null range is not a confidence interval. Replicates and diagnostic methods are not selected on this panel.', '',
        'Primary source: [Black et al., 2015](https://doi.org/10.1039/C4CC06021K). Binary positives use the released curated ammonia signal > 0, which differs from the paper’s ≥20% conversion discussion. Template products are not experimentally established product-selectivity labels.', ''])
    (args.output/'readout.md').write_text('\n'.join(text))
    write_json(args.output/'complete.json', dict(schema='nitrilase_evaluation_receipt_v1', freeze=freeze,
        preregistration=prereg_identity, outputs={p.name:identity(p) for p in sorted(args.output.iterdir()) if p.is_file()}))
    print(json.dumps({'complete':str(args.output/'complete.json'),'primary_method':summary['primary_method']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())


if __name__=='__main__':
    main()
