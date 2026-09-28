#!/usr/bin/env python3
"""Audit multi-equation query hypotheses; never promote them to training labels."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import recover_reactzyme_cgr as single
from horizyn.chemistry.reaction_recovery import participant_key, read_reactions


def audit_group(smiles, row, equations):
    if row['status'] != 'decomposition_hypothesis':
        return 'not_single_decomposition'
    if row.get('search_limited'):
        return 'search_limited'
    groups = row.get('candidate_groups', [])
    if len(groups) != 1 or len(groups[0]) < 2:
        return 'invalid_group'
    group = groups[0]
    if len(set(group)) != len(group):
        return 'duplicate_equations'
    sets = []
    for eid in group:
        eq = equations[eid]
        if eq['checks']['status'] != 'eligible':
            return 'ineligible_equation'
        if not eq.get('rhea_ids'):
            return 'missing_rhea_provenance'
        sets.append(set(participant_key(eq['reaction_smiles'].replace('>>', '.'), 'no_water_proton_set')))
    query = set(participant_key(smiles, 'no_water_proton_set'))
    if set().union(*sets) != query:
        return 'participant_mismatch'
    if any(set().union(*(s for j, s in enumerate(sets) if j != i)) == query for i in range(len(sets))):
        return 'redundant_group'
    return 'chemistry_consistent_hypothesis'


def read_mapping_records(directory):
    settings_path = directory / 'mapping_settings.json'
    if not settings_path.exists():
        return {}, None
    settings = single.load(settings_path)
    checksum = single.digest(settings_path)
    records = {}
    for path in sorted((directory / 'mapped').glob('*.json')):
        row = single.load(path)
        if row['settings_sha256'] != checksum or path.stem != row['equation_id']:
            raise ValueError(f'Invalid mapping provenance: {path}')
        records[row['equation_id']] = row
    return records, settings


def prepare(args):
    single.verify(SimpleNamespace(output=args.baseline))
    settings = single.load(args.baseline / 'mapping_settings.json')
    args.min_confidence = settings['min_confidence']
    args.max_atoms = settings['max_atoms']
    recovery_args = SimpleNamespace(**vars(args))
    recovery_args.output = args.output / 'recovery'
    recovery_args.output.mkdir(exist_ok=True)
    single.recover(recovery_args)
    recovery = single.load(recovery_args.output / 'recovery.json')
    equations = single.load(recovery_args.output / 'equations.json')
    previous, _ = read_mapping_records(args.baseline)
    audits = {}
    wanted = set()
    for part, table in recovery.items():
        queries = read_reactions(args.source_root / args.split / f'{part}_rxns.csv')
        if set(queries) != set(table):
            raise ValueError('Query ID mismatch')
        audits[part] = {}
        for query, row in table.items():
            status = audit_group(queries[query], row, equations)
            group = row['candidate_groups'][0] if status == 'chemistry_consistent_hypothesis' else []
            audits[part][query] = dict(audit_status=status, equation_ids=group,
                                      recovery_status=row['status'])
            wanted.update(group)
    for eid in wanted & previous.keys():
        if previous[eid]['reaction_smiles'] != equations[eid]['reaction_smiles']:
            raise ValueError(f'Cached equation changed: {eid}')
    missing = wanted - previous.keys()
    mapping = args.output / 'constituent_mapping'
    mapping.mkdir(exist_ok=True)
    # This internal worklist is at equation grain, NOT a query assignment table.
    worklist = {'equation_worklist': {eid: dict(status='unique_candidate', mode='constituent_only', equation_ids=[eid])
                                    for eid in sorted(missing)}}
    inputs = [args.baseline / 'manifest.json', args.baseline / 'mapping_settings.json',
              recovery_args.output / 'manifest.json', recovery_args.output / 'recovery.json',
              recovery_args.output / 'equations.json']
    inputs += [args.baseline / 'mapped' / f'{eid}.json' for eid in sorted(wanted & previous.keys())]
    signature = dict(schema=1, split=args.split, inputs={str(p): single.digest(p) for p in inputs},
                     code={str(Path(__file__).resolve()): single.digest(__file__)},
                     search=dict(max_candidates=args.max_candidates, max_equations=args.max_equations,
                                 max_nodes=args.max_nodes))
    manifest = mapping / 'manifest.json'
    if manifest.exists():
        if single.load(manifest)['signature'] != signature:
            raise ValueError('Inputs/code changed; use a new output directory')
        single.verify(SimpleNamespace(output=mapping))
    else:
        single.atomic_json(mapping / 'recovery.json', worklist)
        single.atomic_json(mapping / 'equations.json', {e: equations[e] for e in sorted(missing)})
        single.atomic_json(manifest, dict(signature=signature,
            policy='Internal missing-constituent worklist; all query decompositions remain hypotheses',
            outputs={name: single.digest(mapping / name) for name in ('recovery.json', 'equations.json')}))
    single.atomic_json(args.output / 'query_audit.json', audits)
    print(f'Constituent equations: {len(wanted)}; reuse {len(wanted)-len(missing)}; map {len(missing)}', flush=True)
    return mapping, audits


def report(args, mapping, audits):
    single.verify(SimpleNamespace(output=mapping))
    previous, old_settings = read_mapping_records(args.baseline)
    new, new_settings = read_mapping_records(mapping)
    if new_settings and any(new_settings[k] != old_settings[k] for k in ('model', 'max_atoms', 'min_confidence')):
        raise ValueError('Baseline and new mapper settings differ; refusing to combine')
    combined = {**previous, **new}
    summary = dict(split=args.split, partitions={}, new_mapping_status=dict(Counter(r['status'] for r in new.values())),
                   policy='Hypothesis availability is not certified coverage; baseline masks unchanged; no training integration')
    all_wanted = set()
    for part, table in audits.items():
        baseline = single.load(args.baseline / f'{part}_cgr_index.json')
        if set(baseline) != set(table):
            raise ValueError('Baseline and audited query IDs differ')
        rows = {}; counts = Counter(); ready = 0; baseline_count = sum(r['mask'] for r in baseline.values())
        for query, row in table.items():
            ids = row['equation_ids']; all_wanted.update(ids)
            states = [combined.get(e, {}).get('status', 'pending') for e in ids]
            available = bool(ids) and all(s == 'structurally_valid_cgr' for s in states)
            # Separate availability of constituent graphs from trust in their query assignment.
            rows[query] = dict(**row, baseline_mask=baseline[query]['mask'],
                multi_cgr_hypothesis_mask=available, assignment_verified=False,
                mapping_statuses=states,
                graph_records=[str((args.baseline if e in previous else mapping) / 'mapped' / f'{e}.json') for e in ids])
            counts[row['audit_status']] += 1
            ready += available and not baseline[query]['mask']
        single.atomic_json(args.output / f'{part}_multi_cgr_index.json', rows)
        summary['partitions'][part] = dict(queries=len(rows), baseline_usable=baseline_count,
            audit_counts=dict(counts), additional_graph_complete_hypotheses=ready,
            baseline_plus_hypotheses=baseline_count+ready,
            exploratory_fraction=(baseline_count+ready)/len(rows))
    summary['pending_equations'] = len(all_wanted-combined.keys())
    summary['mapping_complete'] = not summary['pending_equations']
    single.atomic_json(args.output / 'report.json', summary)
    print(__import__('json').dumps(summary, indent=2), flush=True)


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/reactzyme_cgr_recovery_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-candidates', type=int, default=128)
    parser.add_argument('--max-equations', type=int, default=6)
    parser.add_argument('--max-nodes', type=int, default=20000)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if min(args.max_candidates, args.max_equations, args.max_nodes, args.threads) <= 0 or (args.limit is not None and args.limit < 0):
        parser.error('Invalid limits')
    args.baseline = args.baseline.resolve(); args.output = args.output.resolve()
    if args.output == args.baseline or args.baseline in args.output.parents:
        parser.error('Output must be separate from baseline')
    baseline = single.load(args.baseline/'manifest.json')['signature']
    args.split = baseline['split']; args.partitions = baseline['partitions']
    inputs = [Path(p) for p in baseline['inputs']]
    args.rhea = next(p for p in inputs if p.name == 'rhea_molecules.tsv')
    args.source_root = next(p for p in inputs if p.name.endswith('_rxns.csv')).parent.parent
    args.retry_failed = False
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        mapping, audits = prepare(args)
        report(args, mapping, audits)
        mapper_args = SimpleNamespace(**vars(args)); mapper_args.output = mapping
        single.map_equations(mapper_args)
        report(args, mapping, audits)


if __name__ == '__main__':
    main()
