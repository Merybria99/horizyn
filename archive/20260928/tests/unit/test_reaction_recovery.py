import csv
import json
from types import SimpleNamespace

import pytest
from rdkit import Chem

from horizyn.chemistry.reaction_recovery import (
    ReactionIndex, participant_key, bounded_covers, equation_checks,
    build_cgr, _mapped_state, equation_identity, read_reactions,
)


def write_rhea(tmp_path, equations):
    path=tmp_path/'rhea.tsv'
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle,delimiter='\t');writer.writerow(['Rhea ID','substrate','product'])
        for i,(left,right) in enumerate(equations): writer.writerow([f'RHEA:{i}',left,right])
    return path


def test_keys_preserve_multiplicity_stereo_and_isotopes():
    assert participant_key('CC.CC')!=participant_key('CC')
    assert participant_key('CC.CC','no_water_proton_set')==participant_key('CC','no_water_proton_set')
    assert participant_key('O.[H+].CC','no_water_proton_multiset')==('CC',)
    assert participant_key('N[C@H](C)O')!=participant_key('N[C@@H](C)O')
    assert participant_key('[13CH4]')!=participant_key('C')


def test_dative_bonds_are_not_reaction_separators(tmp_path):
    assert participant_key('N->[Cu+2]')
    index=ReactionIndex(write_rhea(tmp_path,[('N.[Cu+2]','N->[Cu+2]')]))
    assert index.recover('N.[Cu+2].N->[Cu+2]')['status']=='unique_candidate'


def test_recovery_relaxes_lookup_not_equation(tmp_path):
    index=ReactionIndex(write_rhea(tmp_path,[('CCO.O','COC.O')]))
    hit=index.recover('COC.CCO')
    assert hit['status']=='unique_candidate' and hit['mode']=='no_water_multiset'
    equation=index.equations[hit['equation_ids'][0]]['reaction_smiles']
    assert all('O' in side.split('.') for side in equation.split('>>'))


def test_reverse_equations_deduplicate_but_different_partitions_are_ambiguous(tmp_path):
    index=ReactionIndex(write_rhea(tmp_path,[('CCO','COC'),('COC','CCO')]))
    assert len(index.equations)==1
    assert index.recover('CCO.COC')['status']=='unique_candidate'
    index=ReactionIndex(write_rhea(tmp_path,[('C.O','CO'),('C','O.CO')]))
    assert index.recover('C.O.CO')['status']=='ambiguous'


def test_multiple_equations_stay_hypotheses(tmp_path):
    index=ReactionIndex(write_rhea(tmp_path,[('CCO','COC'),('CC=O','C=CO')]))
    hit=index.recover('CCO.COC.CC=O.C=CO')
    assert hit['status']=='decomposition_hypothesis'
    assert len(hit['candidate_groups'][0])==2
    limited=index.recover('CCO.COC.CC=O.C=CO',max_candidates=1)
    assert limited['status']=='decomposition_search_limited'


def test_cover_ambiguity_and_limits():
    candidates={'a':frozenset('ab'),'b':frozenset('bc'),'c':frozenset('ac')}
    covers,limited=bounded_covers(frozenset('abc'),candidates)
    assert len(covers)==2
    covers,limited=bounded_covers(frozenset('abc'),candidates,max_nodes=1)
    assert limited and not covers


@pytest.mark.parametrize('reaction,status',[
    ('CCO>>COC','eligible'),('C>>CC','heavy_atom_imbalance'),
    ('CCO>>CC=O','hydrogen_or_isotope_imbalance'),
    ('[Na+]>>[Na]','charge_imbalance'),('*CO>>CO*','generic_atoms'),
    ('CCO>>OCC','identical_sides'),('O>>O','identical_sides'),
])
def test_equation_gates(reaction,status):
    assert equation_checks(reaction)['status']==status


def test_cgr_preserves_context_and_bond_edits():
    graph=build_cgr('CCO>>COC','[CH3:1][CH2:2][OH:3]>>[CH3:1][O:3][CH3:2]')
    assert len(graph['atoms_before'])==3
    assert all(graph['center_mask'])
    assert sum(e['changed'] for e in graph['edges'])==2
    assert any(e['before'] is None for e in graph['edges'])
    assert any(e['after'] is None for e in graph['edges'])


@pytest.mark.parametrize('mapped',[
    '[CH3:1][CH2:2][OH:3]>>[CH3:1][O:3][CH3:1]',
    '[CH3:1][CH2:2][OH:3]>>[CH3:1][O:2][CH3:3]',
    '[CH3:1][CH2:2][OH:3]>>[CH3:1][O:3]C',
    '[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]',
])
def test_invalid_mapping_never_becomes_cgr(mapped):
    with pytest.raises(ValueError): build_cgr('CCO>>COC',mapped)


def test_stereo_representation_independent_of_smiles_neighbor_order():
    text='[F:2][C@H:1]([Cl:3])[Br:4]'
    mol=Chem.MolFromSmiles(text)
    before,_=_mapped_state(text)
    for root in range(mol.GetNumAtoms()):
        reordered=Chem.MolToSmiles(mol,rootedAtAtom=root,canonical=False)
        after,_=_mapped_state(reordered)
        assert before==after


def test_stereo_change_is_detected():
    left='[F:2][C@H:1]([Cl:3])[Br:4]'
    right='[F:2][C@@H:1]([Cl:3])[Br:4]'
    graph=build_cgr('F[C@H](Cl)Br>>F[C@@H](Cl)Br',left+'>>'+right)
    assert sum(graph['center_mask'])==1


def test_invalid_query_and_duplicate_ids(tmp_path):
    index=ReactionIndex(write_rhea(tmp_path,[('CCO','COC')]))
    assert index.recover('notasmiles')['status']=='invalid_query'
    path=tmp_path/'queries.csv';path.write_text('reaction_id,reaction_smiles\nq,CCO\nq,CCO\n')
    with pytest.raises(ValueError): read_reactions(path)


def test_pipeline_preserves_queries_reports_overlap_and_rejects_tampering(tmp_path):
    from scripts.recover_reactzyme_cgr import recover, report, verify, digest
    from horizyn.generalization_diagnostics import atomic_json
    split=tmp_path/'source/reaction_smi';split.mkdir(parents=True)
    for part in ('train','validation'):
        (split/f'{part}_rxns.csv').write_text('reaction_id,reaction_smiles\nq,CCO.COC\nunresolved,CCCC\n')
    args=SimpleNamespace(source_root=tmp_path/'source',split='reaction_smi',partitions=['train','validation'],
        rhea=write_rhea(tmp_path,[('CCO','COC')]),output=tmp_path/'out',
        max_candidates=32,max_equations=4,max_nodes=3000)
    args.output.mkdir();recover(args);recover(args)
    summary=report(args,quiet=True)
    assert summary['pending_equations']==1
    assert summary['partitions']['validation']['unique_candidate_equation_overlap_with_train']==1
    assert summary['partitions']['train']['queries']==2
    assert summary['partitions']['train']['usable_cgr_queries']==0
    settings=args.output/'mapping_settings.json';atomic_json(settings,{'test':True})
    eid=next(iter(json.loads((args.output/'equations.json').read_text())))
    atomic_json(args.output/f'mapped/{eid}.json',dict(equation_id=eid,settings_sha256=digest(settings),status='structurally_valid_cgr'))
    summary=report(args,quiet=True)
    assert summary['partitions']['train']['usable_cgr_queries']==1
    assert not json.loads((args.output/'train_cgr_index.json').read_text())['unresolved']['mask']
    (split/'train_rxns.csv').write_text('changed')
    with pytest.raises(ValueError,match='changed'): verify(args)
