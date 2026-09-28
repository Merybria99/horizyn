"""Training-only targets for the frozen, directly scored biology experiment."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from horizyn.capability.biological_targets import canonical_protein_key, mechanism_groups
from horizyn.capability.positive_biological_targets import _positive_labels, _confidence


# Conservative exact names. Unknowns and generic 'metal' do not become classes.
COFACTORS = {
    'NAD', 'NADP', 'FAD', 'FMN', 'PLP', 'TPP', 'CoA', 'SAM', 'FeS_cluster',
    'heme', 'quinone', 'lipoate', 'Mg2+', 'Mn2+', 'Zn2+', 'Ca2+', 'Fe2+',
    'Fe3+', 'Cu2+', 'Cu+', 'Co2+', 'Ni2+', 'Mo', 'W',
}


def positive_pairs(path):
    with Path(path).open() as handle:
        rows = csv.DictReader(handle)
        if not {'protein_id', 'reaction_id'} <= set(rows.fieldnames or []):
            raise ValueError(f'Missing pair columns: {path}')
        pairs = set()
        for row in rows:
            if any(name in row and float(row[name]) != 1 for name in ('label', 'Label')):
                continue
            q, p = row['reaction_id'].strip(), row['protein_id'].strip()
            if not q or not p:
                raise ValueError('Empty pair ID')
            pairs.add((q, p))
    if not pairs:
        raise ValueError(f'No positive pairs: {path}')
    return sorted(pairs)


def sparse_rows(ids, evidence, vocabulary):
    lookup = {label: i for i, label in enumerate(vocabulary)}
    width = max(1, max((len(evidence.get(key, {})) for key in ids), default=0))
    indices = np.full((len(ids), width), -1, dtype=np.int64)
    confidence = np.zeros(indices.shape, dtype=np.float32)
    for i, key in enumerate(ids):
        for j, (label, value) in enumerate(sorted(evidence.get(key, {}).items())):
            indices[i, j], confidence[i, j] = lookup[label], value
    return indices, confidence


def build_targets(config):
    data = config['data']
    original = json.loads(Path(data['protein_biofp_vocab_path']).read_text())
    sources = original['sources']
    pairs = positive_pairs(data['train_pairs_path'])
    proteins = sorted({p for _, p in pairs})
    reactions = sorted({q for q, _ in pairs})
    canonical = {canonical_protein_key(p): p for p in proteins}
    if len(canonical) != len(proteins):
        raise ValueError('Canonical training protein collision')
    edges = {(q, canonical_protein_key(p)) for q, p in pairs}
    if Path(sources['train_pairs']).resolve() != Path(data['train_pairs_path']).resolve():
        raise ValueError('Existing target bundle is from different training pairs')
    evidence = {side: {family: {} for family in ('ec', 'cofactor', 'mechanism')}
                for side in ('enzyme', 'reaction')}
    def add(side, family, key, labels, confidence):
        if confidence <= 0:
            return
        row = evidence[side][family].setdefault(key, {})
        for label in labels:
            row[label] = max(row.get(label, 0), confidence)

    # EC and transformation targets were already filtered to retained edges.
    # Check the exact protein universe; never silently join by row position.
    with np.load(data['protein_biofp_targets_path'], allow_pickle=False) as old:
        old_ids = old['ids'].astype(str).tolist()
        if set(old_ids) != set(proteins):
            raise ValueError('Existing biological targets do not match training proteins')
        for family in ('ec', 'mechanism'):
            labels = original['families'][family]
            for p, indices, weights in zip(old_ids, old[f'{family}_positive_indices'], old[f'{family}_confidence']):
                for idx, weight in zip(indices, weights):
                    if idx >= 0:
                        add('enzyme', family, p, [labels[idx]], float(weight))

    source_counts = Counter()
    # Do not promote every label to experimental based on a row-level flag.
    columns = {
        'enzyme_derived_core_cofactor_labels_train': .4,
        'uniprot_core_cofactor_labels_train': .4,
        'uniprot_metal_ion_labels_train': .4,
        'enzyme_uniprotkb_cofactor_labels_train': .7,
    }
    with Path(sources['enzyme_cofactors']).open() as handle:
        for row in csv.DictReader(handle):
            p = canonical.get(canonical_protein_key(row['enzyme_id']))
            if p is None:
                continue
            for column, weight in columns.items():
                labels = _positive_labels(row.get(column)) & COFACTORS
                add('enzyme', 'cofactor', p, labels,
                    weight * _confidence(row, ('confidence', 'annotation_confidence', 'label_confidence')))
                source_counts[column] += len(labels)

    features = pd.read_parquet(sources['directional_features'])
    if features['reaction_id'].duplicated().any():
        raise ValueError('Duplicate directional feature IDs')
    features = {str(row['reaction_id']): row for row in features.to_dict('records')}
    with Path(sources['matched_members']).open() as handle:
        for row in csv.DictReader(handle):
            q, pkey = row['source_reaction_id'], canonical_protein_key(row['source_protein_id'])
            if (q, pkey) not in edges:
                continue
            feature = features.get(row['reaction_id'])
            if feature is None:
                continue
            match = _confidence(row, ('match_confidence',))
            transformation = mechanism_groups(feature)
            mapping = _confidence(feature, ('mapping_confidence', 'reaction_center_mapping_confidence'))
            add('reaction', 'mechanism', q, transformation, match * mapping)
            cofactors = (_positive_labels(feature.get('core_cofactor_labels')) |
                         _positive_labels(feature.get('cofactor_labels'))) & COFACTORS
            add('reaction', 'cofactor', q, cofactors, .4 * match)
            add('enzyme', 'cofactor', canonical[pkey], cofactors, .4 * match)
    # No reaction EC inferred from a promiscuous enzyme's union of activities.
    vocabulary = dict(original['families'])
    vocabulary['cofactor'] = sorted(COFACTORS)
    arrays = {}
    coverage = {}
    for side, ids in (('enzyme', proteins), ('reaction', reactions)):
        arrays[f'{side}_ids'] = np.asarray(ids)
        coverage[side] = {}
        for family, labels in vocabulary.items():
            indices, confidence = sparse_rows(ids, evidence[side][family], labels)
            arrays[f'{side}_{family}_indices'] = indices
            arrays[f'{side}_{family}_confidence'] = confidence
            coverage[side][family] = int((indices >= 0).any(1).sum())
    metadata = dict(families=vocabulary, coverage=coverage, sources=sources,
        num_proteins=len(proteins), num_reactions=len(reactions), cofactor_source_counts=dict(source_counts),
        confidence_policy='absolute weights / observed label count; maximum per repeated label',
        limitations=original['limitations'] + [
            'Specific cofactors are exact whitelisted names, not inferred requirements.',
            'UniProt-derived columns may include indirect evidence; weights are heuristics, not probabilities.',
            'Reaction EC is unsupervised; no enzyme-label union transferred to reactions.',
            'Base checkpoint already received biological supervision; lambda=0 means no NEW supervision.',
        ])
    return arrays, metadata
