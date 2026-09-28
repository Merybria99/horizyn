#!/usr/bin/env python3
"""Post-fit diagnostic of biological neighborhoods in unseen validation rules.

Validation annotations are read only for this analysis, never for training or
checkpoint selection. Compare observed category geometry with fixed shuffled
assignments to separate annotation alignment from global embedding contraction.
"""
from collections import defaultdict
import argparse
import csv
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from rdkit import RDLogger
from horizyn.biological_geometry import category_pair_distance
from generalization_biological_geometry_prepare import (
    add, ec_labels, empty, compact, mechanism_groups, cofactor_name_groups,
    extract_cofactor_labels_from_smiles, coarse_reaction_center_labels, extract_reaction_center_raw_labels, CROSS)
from generalization_clipzyme_screening_protocol import identifier
from generalization_full_graph import atomic_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); out = a.campaign.resolve()
    torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    acquisition = json.loads((CROSS / 'clipzyme_data_acquisition.json').read_text())
    native_path = Path(acquisition['files']['cached_enzymemap.p']['path'])
    with native_path.open('rb') as stream:
        native = pickle.load(stream)
    path = CROSS / 'clipzyme_manifests_v2/dev_associations.csv'
    rows = defaultdict(empty); parsed = {}; molecules = {}
    with path.open() as stream:
        for item in csv.DictReader(stream):
            source = native[int(item['source_index'])]
            if source['reaction_string'] != item['reaction'] or source['protein_id'] != item['protein_id']:
                raise ValueError('Validation source indices changed')
            key = identifier(item['reaction']); target = rows[key]
            add(target, 'ec', ec_labels(source['ec']), 1.)
            if key not in parsed:
                cofactors = set()
                for mol in item['reaction'].replace('>', '.').split('.'):
                    if mol:
                        if mol not in molecules:
                            molecules[mol] = cofactor_name_groups(extract_cofactor_labels_from_smiles(mol))
                        cofactors.update(molecules[mol])
                mapped = '.'.join(source['reactants']) + '>>' + '.'.join(source['products'])
                try:
                    labels = coarse_reaction_center_labels(extract_reaction_center_raw_labels(mapped))
                    mechanism = mechanism_groups({'reaction_center_coarse_labels': labels})
                except (ValueError, RuntimeError):
                    mechanism = set()
                parsed[key] = (cofactors, mechanism)
            cofactors, mechanism = parsed[key]
            add(target, 'cofactor', cofactors, .4)
            quality = float(source.get('quality', 0))
            if quality >= .5:
                add(target, 'mechanism', mechanism, min(quality, 1.))
    paths = {
        'v4': CROSS / 'v4_biological_geometry_20260921_v1/enzymemap/validation_cache.pt',
        'biological_f3': out / 'followup/enzymemap/enzymemap/validation_cache.pt'}
    caches = {name: torch.load(path, map_location='cpu', weights_only=False) for name, path in paths.items()}
    keys = caches['v4']['ids']
    if caches['biological_f3']['ids'] != keys or not set(keys) <= rows.keys():
        raise ValueError('Validation reaction axes differ')
    train = set(json.loads((out / 'enzymemap/phase2/annotations.json').read_text())['reaction_ids'])
    # Match screening validation: exclude reactions that also occur in training.
    kept = [i for i, key in enumerate(keys) if key not in train]
    labels = compact([rows[keys[i]] for i in kept])
    records = []
    for family, block in labels.items():
        memberships = torch.tensor(block['rows'], dtype=torch.long)
        groups = torch.tensor(block['groups'], dtype=torch.long)
        confidence = torch.tensor(block['confidence'], dtype=torch.float32)
        weights = torch.tensor(block['category_weight'], dtype=torch.float32)
        generator = torch.Generator().manual_seed(1701)
        permutations = [torch.randperm(len(kept), generator=generator)[memberships] for _ in range(50)]
        for name, payload in caches.items():
            z = payload['base_r'][kept]
            observed = float(category_pair_distance(z, memberships, groups, confidence, weights))
            shuffled = np.array([float(category_pair_distance(z, perm, groups, confidence, weights)) for perm in permutations])
            mean = float(shuffled.mean())
            records.append(dict(model=name, family=family, queries=len(kept), coverage=block['coverage'],
                categories=len(block['labels']), observed_distance=observed,
                shuffled_mean_distance=mean, shuffled_distance_std=float(shuffled.std()),
                relative_category_separation=(mean - observed) / mean if mean else None))
    result = dict(records=records, native_sha256=sha(native_path), validation_manifest_sha256=sha(path),
        caches={name: dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
        labels_scope='validation only; excluded training reactions; post-fit analysis',
        training_or_selection_used=False, shuffled_assignments=50,
        interpretation='Positive relative separation means observed biological categories are closer than random '
            'category assignments. It is a representation diagnostic, not activity accuracy or proof of slot specialization.')
    atomic_json(out / 'validation_biological_geometry.json', result)
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
