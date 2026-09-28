#!/usr/bin/env python3
"""Describe chemistry behind near-tied ReactZyme E-to-R candidates."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from rdkit import Chem, RDLogger
from generalization_full_graph import atomic_json, sha


def canonical_participants(smiles):
    parts = []
    for token in smiles.split('.'):
        mol = Chem.MolFromSmiles(token)
        if mol is None:
            return None
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
    return '.'.join(sorted(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--variant', default='f3_biology')
    parser.add_argument('--tolerance', type=float, default=1e-6)
    args = parser.parse_args(); campaign = args.campaign.resolve()
    RDLogger.DisableLog('rdApp.*'); records = []
    for split in ('reaction_smi', 'enzyme_smi', 'time'):
        followup = campaign / 'followup' / split
        task, = json.loads((followup / 'protocol.json').read_text())['tasks']
        config = yaml.safe_load(Path(task['model']['test_config']).read_text())
        path = Path(config['data']['test_reactions_path'])
        path = path if path.is_absolute() else ROOT / path
        with path.open() as stream:
            reaction_smiles = {r['reaction_id']: r['reaction_smiles'] for r in csv.DictReader(stream)}
        catalog = json.loads((Path(task['test_features']) / 'catalog.json').read_text())
        ids = catalog['reactions']
        if any('>' in reaction_smiles[k] for k in ids):
            raise ValueError('This diagnostic expects the released participant-set representation')
        canonical = [canonical_participants(reaction_smiles[k]) for k in ids]
        run = followup / split / args.variant
        scores = np.load(run / 'test_scores.npy', mmap_mode='r')
        with np.load(run / 'test_ranks.npz') as rank:
            r = rank['enzyme_to_reaction_reaction_index']; e = rank['enzyme_to_reaction_enzyme_index']
        with np.load(Path(task['test_features']) / 'f3_features.npz') as vectors:
            features = vectors['reactions']
        if scores.shape[0] != len(ids) or len(features) != len(ids):
            raise ValueError('Reaction axes disagree')
        pairs = Counter(); near_edges = alias_edges = 0
        for start in range(0, len(r), 256):
            rs, es = r[start:start+256], e[start:start+256]
            near = np.abs(scores[:, es].T - scores[rs, es][:, None]) <= args.tolerance
            near[np.arange(len(rs)), rs] = False
            near_edges += int(near.any(1).sum())
            for row, ri in enumerate(rs):
                others = np.flatnonzero(near[row])
                alias_edges += any(canonical[ri] is not None and canonical[ri] == canonical[j] for j in others)
                for other in others:
                    pairs[tuple(sorted((int(ri), int(other))))] += 1
        details = []
        for (left, right), count in pairs.most_common():
            details.append(dict(reaction_ids=[ids[left], ids[right]], positive_edge_occurrences=count,
                same_canonical_participants=canonical[left] is not None and canonical[left] == canonical[right],
                f3_vector_max_abs_difference=float(np.max(np.abs(features[left] - features[right]))),
                raw_smiles=[reaction_smiles[ids[left]], reaction_smiles[ids[right]]]))
        record = dict(split=split, variant=args.variant, tolerance=args.tolerance, positive_edges=len(r),
            near_competitor_edges=near_edges, edges_with_canonical_participant_alias=int(alias_edges),
            unique_near_pairs=len(pairs), unique_canonical_alias_pairs=sum(x['same_canonical_participants'] for x in details),
            source_scores_sha256=sha(run / 'test_scores.npy'), reactions_sha256=sha(path), pairs=details)
        records.append(record)
        print({k: v for k, v in record.items() if k not in ('pairs', 'source_scores_sha256', 'reactions_sha256')}, flush=True)
    atomic_json(campaign / 'tie_chemistry_audit.json', dict(records=records,
        interpretation='Secondary post-test diagnostic. Canonical participant equality preserves stereochemistry and '
            'multiplicity but cannot restore missing physical reaction sides. No metrics, labels or qualification rules are changed.'))


if __name__ == '__main__':
    main()
