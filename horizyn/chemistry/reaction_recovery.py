"""Chemistry-only equation recovery and conservative atom-mapped CGR validation.

No enzyme IDs, sequences, EC labels or positive-pair tables are used here.
Relaxed lookup identifies candidates; it never edits the recovered equation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
from pathlib import Path

from rdkit import Chem

MATCH_MODES = ('exact_multiset', 'no_water_multiset',
               'no_water_proton_multiset', 'no_water_proton_set')


@lru_cache(maxsize=100_000)
def canonical_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Invalid molecule: {smiles[:100]}')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=40_000)
def components(smiles):
    if '>>' in smiles:
        raise ValueError('Expected one molecular side/participant collection')
    return tuple(canonical_molecule(s) for s in smiles.split('.') if s.strip())


def participant_key(smiles, mode='exact_multiset'):
    if mode not in MATCH_MODES:
        raise ValueError(f'Unknown matching mode: {mode}')
    values = components(smiles)
    if mode.startswith('no_water'):
        values = tuple(s for s in values if s != 'O')
    if 'proton' in mode:
        values = tuple(s for s in values if s != '[H+]')
    return tuple(sorted(set(values) if mode.endswith('_set') else values))


def read_reactions(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        key, smiles = row['reaction_id'].strip(), row['reaction_smiles'].strip()
        if not key or not smiles or key in result:
            raise ValueError(f'Empty/duplicate reaction ID in {path}: {key}')
        # Deliberately discard every other column, including possible annotations.
        result[key] = smiles
    if not result:
        raise ValueError(f'Empty reaction table: {path}')
    return result


def equation_identity(left, right):
    sides = ['.'.join(sorted(components(s))) for s in (left, right)]
    # ReactZyme does not specify an orientation. Reverse-equivalent Rhea entries
    # share an equation, but genuinely different partitions remain ambiguous.
    return '>>'.join(sorted(sides))


def equation_checks(smiles):
    """Check full formula (including H/isotopes), charge, and concrete structure."""
    left, right = smiles.split('>>')
    stats = []
    for side in (left, right):
        mol = Chem.MolFromSmiles(side)
        if mol is None or not mol.GetNumAtoms():
            return {'status': 'invalid_structure'}
        if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
            return {'status': 'generic_atoms'}
        heavy = Counter((a.GetAtomicNum(), a.GetIsotope()) for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
        full = Counter((a.GetAtomicNum(), a.GetIsotope()) for a in Chem.AddHs(mol).GetAtoms())
        stats.append((heavy, full, sum(a.GetFormalCharge() for a in mol.GetAtoms())))
    if not stats[0][0] or not stats[1][0]:
        status = 'no_heavy_atoms'
    elif stats[0][0] != stats[1][0]:
        status = 'heavy_atom_imbalance'
    elif stats[0][1] != stats[1][1]:
        status = 'hydrogen_or_isotope_imbalance'
    elif stats[0][2] != stats[1][2]:
        status = 'charge_imbalance'
    elif participant_key(left) == participant_key(right):
        status = 'identical_sides'
    else:
        status = 'eligible'
    return dict(status=status, heavy_atoms=sum(stats[0][0].values()),
                charge_before=stats[0][2], charge_after=stats[1][2])


class ReactionIndex:
    def __init__(self, rhea_path):
        self.equations = {}
        self.index = {mode: defaultdict(set) for mode in MATCH_MODES}
        self.invalid_sources = []
        seen = set()
        with Path(rhea_path).open(newline='') as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                source_id = row['Rhea ID'].strip()
                if not source_id or source_id in seen:
                    raise ValueError(f'Duplicate/empty Rhea ID: {source_id}')
                seen.add(source_id)
                try:
                    left, right = row['substrate'].strip(), row['product'].strip()
                    if not left or not right:
                        raise ValueError('Empty side')
                    smiles = equation_identity(left, right)
                    key = 'eq_' + hashlib.sha256(smiles.encode()).hexdigest()[:24]
                    if key not in self.equations:
                        self.equations[key] = dict(equation_id=key, reaction_smiles=smiles,
                            rhea_ids=[], checks=equation_checks(smiles),
                            orientation='canonical unordered sides; not physiological direction')
                    self.equations[key]['rhea_ids'].append(source_id)
                    for mode in MATCH_MODES:
                        lookup = participant_key(left+'.'+right, mode)
                        if lookup: self.index[mode][lookup].add(key)
                except ValueError as error:
                    self.invalid_sources.append(dict(rhea_id=source_id, error=str(error)))
        self.sets = {eid: frozenset(participant_key(e['reaction_smiles'].replace('>>','.'),
                          'no_water_proton_set')) for eid,e in self.equations.items()}
        self.inverted = defaultdict(set)
        for eid, values in self.sets.items():
            if self.equations[eid]['checks']['status'] == 'identical_sides': continue
            for value in values: self.inverted[value].add(eid)

    def recover(self, smiles, *, max_candidates=32, max_equations=4, max_nodes=3000):
        try:
            if '>>' in smiles:
                if smiles.count('>>') != 1:
                    raise ValueError('Unsupported reaction syntax')
                left, right = smiles.split('>>')
                normalized = equation_identity(left, right)
                eid = 'eq_' + hashlib.sha256(normalized.encode()).hexdigest()[:24]
                self.equations.setdefault(eid, dict(equation_id=eid, reaction_smiles=normalized,
                    rhea_ids=[], checks=equation_checks(normalized), orientation='canonical unordered sides'))
                return dict(status='unique_candidate', mode='explicit_sides', equation_ids=[eid])
            for mode in MATCH_MODES:
                key = participant_key(smiles, mode)
                matches = sorted(self.index[mode].get(key, ())) if key else []
                if matches:
                    return dict(status='unique_candidate' if len(matches)==1 else 'ambiguous',
                                mode=mode, equation_ids=matches)
            values = frozenset(participant_key(smiles, 'no_water_proton_set'))
        except ValueError as error:
            return dict(status='invalid_query', error=str(error), equation_ids=[])
        candidate_ids = set().union(*(self.inverted.get(s, set()) for s in values)) if values else set()
        candidates = {eid: self.sets[eid] for eid in sorted(candidate_ids)
                      if self.sets[eid] and self.sets[eid] <= values}
        if len(candidates) > max_candidates:
            return dict(status='decomposition_search_limited', equation_ids=[],
                        candidate_count=len(candidates), reason='candidate_limit')
        covers, limited = bounded_covers(values, candidates, max_equations, max_nodes)
        if covers:
            # Even a unique bounded cover is only a hypothesis, NOT a recovered
            # experimental assignment and never an automatic usable query feature.
            return dict(status='decomposition_hypothesis' if len(covers)==1 else 'ambiguous_decomposition',
                        equation_ids=sorted(set().union(*(set(c) for c in covers))),
                        candidate_groups=covers, search_limited=limited,
                        max_equations=max_equations, mode='no_water_proton_set')
        return dict(status='decomposition_search_limited' if limited else 'unresolved',
                    equation_ids=[], candidate_count=len(candidates))


def bounded_covers(query, candidates, max_equations=4, max_nodes=3000):
    """Enumerate at most two irredundant covers. Never certify a truncated search."""
    solutions, visited = set(), set()
    nodes, limited = 0, False
    def visit(selected, covered):
        nonlocal nodes, limited
        if len(solutions) >= 2: return
        signature = tuple(sorted(selected))
        if signature in visited: return
        visited.add(signature)
        nodes += 1
        if nodes > max_nodes:
            limited = True; return
        if covered == query:
            if all(set().union(*(candidates[j] for j in selected if j != i)) != query for i in selected):
                solutions.add(signature)
            return
        if len(selected) >= max_equations:
            limited = True; return
        uncovered = query-covered
        choices = {s: [i for i,v in candidates.items() if s in v and i not in selected] for s in uncovered}
        pivot = min(choices, key=lambda s: (len(choices[s]), s))
        for i in choices[pivot]:
            visit(selected+(i,), covered|candidates[i])
            if len(solutions)>=2 or nodes>max_nodes: break
    if query: visit((), frozenset())
    return [list(c) for c in sorted(solutions)], limited


def _stereo_orientation(atom):
    """Tetrahedral orientation relative to map-sorted neighbours, not SMILES order."""
    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_UNSPECIFIED: return 0
    if tag not in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
        raise ValueError('Unsupported non-tetrahedral stereo')
    neighbors = [n.GetAtomMapNum() if n.GetAtomicNum()!=1 else 0 for n in atom.GetNeighbors()]
    if len(neighbors)==3: neighbors.append(0)
    parity = sum(a>b for i,a in enumerate(neighbors) for b in neighbors[i+1:]) % 2
    sign = 1 if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else -1
    return sign * (-1 if parity else 1)


def _mapped_state(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError('Invalid mapped SMILES')
    atoms, bonds = {}, {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()==1: continue
        key = atom.GetAtomMapNum()
        if key<=0 or key in atoms: raise ValueError('Missing/duplicate heavy-atom map IDs')
        atoms[key] = [atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
                      atom.GetTotalNumHs(includeNeighbors=True), int(atom.GetIsAromatic()),
                      _stereo_orientation(atom)]
    for bond in mol.GetBonds():
        a,b = bond.GetBeginAtom(),bond.GetEndAtom()
        if a.GetAtomicNum()==1 or b.GetAtomicNum()==1: continue
        bonds[tuple(sorted((a.GetAtomMapNum(),b.GetAtomMapNum())))] = [str(bond.GetBondType()),str(bond.GetStereo())]
    return atoms,bonds


def build_cgr(original, mapped):
    """Strict structural gate; successful validation is not a chemistry oracle."""
    if equation_checks(original)['status']!='eligible':
        raise ValueError('Source equation is not concrete and balanced')
    if mapped.count('>>') != 1: raise ValueError('Invalid mapped equation syntax')
    source_sides, mapped_sides = original.split('>>'),mapped.split('>>')
    for expected,actual in zip(source_sides,mapped_sides):
        if participant_key(expected) != participant_key(actual):
            raise ValueError('Atom mapper changed the chemical equation')
    before, rb = _mapped_state(mapped_sides[0])
    after, pb = _mapped_state(mapped_sides[1])
    if not before or set(before)!=set(after): raise ValueError('Incomplete atom correspondence')
    if any(before[i][:2]!=after[i][:2] for i in before):
        raise ValueError('Atom mapping changes elements/isotopes')
    atom_ids = sorted(before)
    index = {key:i for i,key in enumerate(atom_ids)}
    center = {i for i in before if before[i]!=after[i]}
    edges = []
    for pair in sorted(set(rb)|set(pb)):
        changed = rb.get(pair)!=pb.get(pair)
        if changed: center.update(pair)
        edges.append(dict(atoms=[index[i] for i in pair], before=rb.get(pair),
                          after=pb.get(pair), changed=changed))
    if not center: raise ValueError('No net atom/bond change after mapping')
    neighborhood = set(center)
    for pair in set(rb)|set(pb):
        if set(pair)&center: neighborhood.update(pair)
    return dict(atom_feature_names=['atomic_number','isotope','formal_charge','hydrogens','aromatic','stereo_orientation'],
        atoms_before=[before[i] for i in atom_ids], atoms_after=[after[i] for i in atom_ids],
        edges=edges, center_mask=[i in center for i in atom_ids],
        center_context_mask=[i in neighborhood for i in atom_ids],
        validation='structural checks only; atom mapping remains a model prediction')
