#!/usr/bin/env python3
"""Prepare audited training-only annotation neighborhoods for unchanged V4."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from rdkit import RDLogger
from horizyn.capability.biological_targets import _labels, mechanism_groups
from horizyn.capability.cofactor_vocabulary_v2 import cofactor_name_groups
from horizyn.capability.cofactors import extract_cofactor_labels_from_smiles
from horizyn.capability.reaction_center import extract_reaction_center_raw_labels, coarse_reaction_center_labels
from horizyn.capability.reaction_features import canonicalize_reaction_smiles
from horizyn.positive_bio import ec_prefixes
from generalization_full_graph import sha, atomic_json

FAMILIES = ('ec', 'cofactor', 'mechanism')
CROSS = ROOT / 'runs/generalization_20260919_2251/cross_paper_retraining'


def add(row, family, labels, confidence):
    if not np.isfinite(confidence) or confidence <= 0:
        return
    for label in labels:
        if label not in ('unknown', 'none', 'no_cofactor'):
            row[family][label] = max(row[family].get(label, 0.), min(float(confidence), 1.))


def ec_labels(value):
    out = set()
    for label in _labels(value):
        try:
            # Only specified hierarchy levels; no imputation of missing digits.
            out.update(ec_prefixes(label))
        except ValueError:
            pass
    return out


def empty():
    return {f: {} for f in FAMILIES}


def compact(rows):
    result = {}
    for family in FAMILIES:
        vocabulary = sorted({label for row in rows for label in row[family]})
        lookup = {k: i for i, k in enumerate(vocabulary)}
        ri, gi, confidence = [], [], []
        for i, row in enumerate(rows):
            for label, value in sorted(row[family].items()):
                ri.append(i); gi.append(lookup[label]); confidence.append(value)
        depth_weights = {1: .125, 2: .25, 3: .5, 4: 1.}
        result[family] = dict(labels=vocabulary, rows=ri, groups=gi, confidence=confidence,
            category_weight=[depth_weights[len(k.split('.'))] if family == 'ec' else 1. for k in vocabulary],
            coverage=len(set(ri)), annotated_memberships=len(ri))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    a = p.parse_args(); a.campaign.mkdir(parents=True, exist_ok=True)
    if (a.campaign / 'protocol.json').exists():
        raise ValueError('Choose a new immutable campaign')
    RDLogger.DisableLog('rdApp.*')
    freeze_path = CROSS / 'shared_recipe_alpha04_cap05_v1/case1_freeze.json'
    freeze = json.loads(freeze_path.read_text())
    native_path = ROOT / 'data/paper/reactzyme/raw/cleaned_uniprot_rhea.tsv'
    native = pd.read_csv(native_path, sep='\t', usecols=['Sequence', 'EC number'], dtype=str).fillna('')
    native_ec = defaultdict(set)
    for sequence, ec in native[['Sequence', 'EC number']].itertuples(index=False, name=None):
        key = 'prot_' + hashlib.sha1(sequence.encode()).hexdigest()[:16]
        native_ec[key].update(ec_labels(ec))
    del native
    cached_path = ROOT / 'data/processed/capability_features/train_exact_rhea_reconstructed/reaction_features.parquet'
    # Reaction-only EC lookup: no protein associations from this resource.
    cached = pd.read_parquet(cached_path, columns=['raw_reaction_smiles', 'ec_numbers'])
    chemistry_ec = defaultdict(set)
    for smiles, ecs in cached.itertuples(index=False, name=None):
        labels = ec_labels(ecs)
        if labels:
            key = canonicalize_reaction_smiles(smiles)
            if key:
                chemistry_ec[key].update(labels)
    acquisition = json.loads((CROSS / 'clipzyme_data_acquisition.json').read_text())
    clip_cache_path = Path(acquisition['files']['cached_enzymemap.p']['path'])
    with clip_cache_path.open('rb') as handle:
        clip_cache = pickle.load(handle)
    molecule_cache = {}
    def cofactors(smiles):
        labels = set()
        for molecule in smiles.replace('>', '.').split('.'):
            if not molecule:
                continue
            if molecule not in molecule_cache:
                molecule_cache[molecule] = sorted(cofactor_name_groups(extract_cofactor_labels_from_smiles(molecule)))
            labels.update(molecule_cache[molecule])
        return labels
    tasks = []
    for gpu, model in enumerate(freeze['models']):
        name = model['name']; source = Path(model['source_phase2']); out = a.campaign / name
        out.mkdir(exist_ok=True)
        if (out / 'annotation_audit.json').exists():
            tasks.append(dict(name=name, gpu=gpu, source_phase2=str(source), model=model,
                annotations=str(out / 'annotations.json'), annotations_sha256=sha(out / 'annotations.json')))
            continue
        catalog = json.loads((source / 'features/catalog.json').read_text())
        with np.load(source / 'features/pairs.npz') as z:
            edges = z['train']
        r_global, e_global = np.unique(edges[:, 0]), np.unique(edges[:, 1])
        reactions = [catalog['reactions'][i] for i in r_global]
        enzymes = [catalog['proteins'][i] for i in e_global]
        assert reactions == catalog['train_reactions']
        rxn_index = {r.removesuffix('_f'): i for i, r in enumerate(reactions)}
        enzyme_index = {e: i for i, e in enumerate(enzymes)}
        reaction_rows = [empty() for _ in reactions]; enzyme_rows = [empty() for _ in enzymes]
        sources = []
        def source_record(path, role):
            sources.append(dict(path=str(path.resolve()), sha256=sha(path), role=role))
        if name != 'enzymemap':
            pairs_path = ROOT / f'data/revised_protocols/reactzyme_paper/{name}/train_pairs.csv'
            rxns_path = pairs_path.with_name('train_rxns.csv')
            src = ROOT / f'runs/reactzyme_reaction_features_v1/data/{name}/reaction_directional/source/train'
            mapping_path = src / 'rhea_directional_reactions.csv'
            mechanisms_path = src / 'reaction_centers/reaction_features.parquet'
            matching = pd.read_csv(mapping_path).set_index('reaction_id')
            for row in pd.read_parquet(mechanisms_path).to_dict('records'):
                rid = row['reaction_id']
                if rid not in rxn_index or rid not in matching.index:
                    continue
                match = matching.loc[rid]
                match_confidence = float(match['match_confidence'])
                # Reject weak atom maps rather than treating every stereotag change as a mechanism.
                confidence = row.get('reaction_center_mapping_confidence')
                if confidence is not None and np.isfinite(confidence) and confidence >= .5:
                    add(reaction_rows[rxn_index[rid]], 'mechanism', mechanism_groups(row), confidence * match_confidence)
                key = canonicalize_reaction_smiles(match['reaction_smiles'])
                add(reaction_rows[rxn_index[rid]], 'ec', chemistry_ec.get(key, set()), match_confidence)
            for row in pd.read_csv(rxns_path).itertuples(index=False):
                if row.reaction_id in rxn_index:
                    add(reaction_rows[rxn_index[row.reaction_id]], 'cofactor', cofactors(row.reaction_smiles), .4)
            for key, i in enzyme_index.items():
                add(enzyme_rows[i], 'ec', native_ec.get(key, set()), 1.)
            for path, role in [(pairs_path, 'permitted training associations'), (rxns_path, 'permitted training chemistry'),
                    (mapping_path, 'training reaction-only Rhea mapping'), (mechanisms_path, 'cached training bond-change descriptors'),
                    (native_path, 'native ReactZyme EC annotations, restricted to training sequences'),
                    (cached_path, 'reaction-only curated EC lookup, exact canonical chemistry')]:
                source_record(path, role)
        else:
            pairs_path = CROSS / 'clipzyme_f3_catalog_v1/train_pairs.csv'
            rxns_path = CROSS / 'clipzyme_f3_catalog_v1/train_rxns.csv'
            native_train_path = CROSS / 'clipzyme_manifests_v2/train_associations.csv'
            candidate_map_path = CROSS / 'clipzyme_f3_catalog_v1/screening_candidate_map.csv'
            candidate_map = pd.read_csv(candidate_map_path)
            rescued_ids = dict(zip(candidate_map.uniprot_id, candidate_map.protein_id))
            rxn_strings = dict(zip(pd.read_csv(rxns_path).reaction_smiles, pd.read_csv(rxns_path).reaction_id))
            parsed = {}
            permitted_indices = []
            for row in pd.read_csv(native_train_path).itertuples(index=False):
                item = clip_cache[int(row.source_index)]; permitted_indices.append(int(row.source_index))
                if item['reaction_string'] != row.reaction or item['protein_id'] != row.protein_id:
                    raise ValueError('Native training source index mismatch')
                rid = rxn_strings[row.reaction]
                if rid not in rxn_index:
                    raise ValueError('Native training reaction missing from F3')
                annotation = reaction_rows[rxn_index[rid]]
                add(annotation, 'ec', ec_labels(item['ec']), 1.)
                eid = ('p_' + hashlib.sha256(row.sequence.encode()).hexdigest()[:24]
                       if isinstance(row.sequence, str) and row.sequence else rescued_ids[row.protein_id])
                if eid not in enzyme_index:
                    raise ValueError('Native training enzyme missing from F3')
                add(enzyme_rows[enzyme_index[eid]], 'ec', ec_labels(item['ec']), 1.)
                if rid not in parsed:
                    mapped = '.'.join(item['reactants']) + '>>' + '.'.join(item['products'])
                    try:
                        coarse = coarse_reaction_center_labels(extract_reaction_center_raw_labels(mapped))
                        mechanism = mechanism_groups({'reaction_center_coarse_labels': coarse})
                    except (ValueError, RuntimeError):
                        mechanism = set()
                    parsed[rid] = (mechanism, cofactors(row.reaction))
                mechanism, cofactor = parsed[rid]
                quality = float(item.get('quality', 0))
                if quality >= .5:
                    add(annotation, 'mechanism', mechanism, min(quality, 1.))
                add(annotation, 'cofactor', cofactor, .4)
            atomic_json(out / 'permitted_source_indices.json', dict(indices=permitted_indices, scope='official train only'))
            for path, role in [(pairs_path, 'permitted training associations'), (rxns_path, 'permitted training chemistry'),
                    (native_train_path, 'official training-only source indices'),
                    (candidate_map_path, 'existing sequence-rescue ID mapping; no activity labels'),
                    (clip_cache_path, 'native EC and atom mapping; only permitted source indices accessed')]:
                source_record(path, role)
        ri = {int(g): i for i, g in enumerate(r_global)}
        ei = {int(g): i for i, g in enumerate(e_global)}
        for r, e in edges:
            for family in FAMILIES:
                for label, confidence in reaction_rows[ri[int(r)]][family].items():
                    add(enzyme_rows[ei[int(e)]], family, [label], confidence)
        payload = dict(schema='v4_training_biological_geometry_v1', reaction_ids=reactions, enzyme_ids=enzymes,
            feature_manifest_sha256=sha(source / 'features/manifest.json'),
            endpoints=dict(reaction=compact(reaction_rows), enzyme=compact(enzyme_rows)), sources=sources,
            train_edges=len(edges), heldout_associations_used=False, heldout_annotations_required_at_inference=False,
            limitations=['EC is functional classification, not positional encoding.',
                'Native training-sequence EC annotations may describe multiple activities; no held-out association table was used.',
                'Cofactor presence is a reaction-associated descriptor, not an experimentally established requirement.',
                'Mechanism labels are coarse bond-change proxies, not complete catalytic mechanisms.',
                'Missing labels are neutral. Weak atom maps below 0.5 are omitted.',
                'Extra biological annotations differ from the unannotated V4 control; total supervision is not identical to all competitors.'])
        atomic_json(out / 'annotations.json', payload)
        for artifact in ('features', 'anchors.pt'):
            (out / artifact).symlink_to(source / artifact)
        tasks.append(dict(name=name, gpu=gpu, source_phase2=str(source), model=model,
            annotations=str(out / 'annotations.json'), annotations_sha256=sha(out / 'annotations.json')))
        coverage = {endpoint: {family: block['coverage'] for family, block in payload['endpoints'][endpoint].items()}
                    for endpoint in ('reaction', 'enzyme')}
        atomic_json(out / 'annotation_audit.json', dict(coverage=coverage, sources=sources,
            train_reactions=len(reactions), train_enzymes=len(enzymes), train_edges=len(edges),
            annotations_sha256=sha(out / 'annotations.json'), limitations=payload['limitations']))
        print(json.dumps(dict(task=name, coverage=coverage)), flush=True)
    variants = [dict(name='control', weights=dict.fromkeys(FAMILIES, 0.))]
    variants += [dict(name=f'all_{str(w).replace(".", "p")}', weights=dict.fromkeys(FAMILIES, w)) for w in (.03, .1, .3)]
    variants += [dict(name='without_' + family, weights={f: 0. if f == family else .1 for f in FAMILIES}) for family in FAMILIES]
    variants += [dict(name='shuffled_0p1', weights=dict.fromkeys(FAMILIES, .1), shuffle_seed=1701)]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), tasks=tasks, variants=variants,
        parent_freeze=str(freeze_path), parent_freeze_sha256=sha(freeze_path),
        architecture='Unchanged V4 with SLEEC; zero new trainable parameters and no new inference components',
        training='Fresh phase-2 heads on each existing target-specific fresh F3 fit; same 100-update recipe as V4',
        selection='Report all predeclared variants; prioritize shared validation gains; test inspections exploratory',
        inference=dict(alpha=.4, cap=.5, fusion_multiplier=3),
        contribution='Matched control, leave-one-family-out and shuffled-label control at weight 0.1',
        case1='Frozen final benchmark-qualified variants only; retrospective literature panel',
        stop_policy='Do not stop useful live runs at the three-hour boundary')
    atomic_json(a.campaign / 'protocol.json', protocol)


if __name__ == '__main__':
    main()
