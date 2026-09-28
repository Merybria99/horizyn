import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[2] / 'scripts/recover_reactzyme_multi_cgr.py'
spec = importlib.util.spec_from_file_location('multi_recovery', path)
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


@pytest.fixture
def equations():
    return {
        'a': dict(reaction_smiles='CCO>>COC', checks={'status': 'eligible'}, rhea_ids=['1']),
        'b': dict(reaction_smiles='CC=O>>C=CO', checks={'status': 'eligible'}, rhea_ids=['2']),
    }


def hypothesis(**kwargs):
    return dict(status='decomposition_hypothesis', candidate_groups=[['a', 'b']], **kwargs)


def test_full_chemistry_cover_is_only_hypothesis(equations):
    assert multi.audit_group('CCO.COC.CC=O.C=CO', hypothesis(), equations) == 'chemistry_consistent_hypothesis'


def test_bounded_or_ambiguous_is_not_approved(equations):
    assert multi.audit_group('CCO.COC.CC=O.C=CO', hypothesis(search_limited=True), equations) == 'search_limited'
    assert multi.audit_group('CCO', {'status':'ambiguous_decomposition'}, equations) == 'not_single_decomposition'


def test_mismatch_and_provenance_fail_closed(equations):
    assert multi.audit_group('CCO.COC', hypothesis(), equations) == 'participant_mismatch'
    equations['b']['rhea_ids'] = []
    assert multi.audit_group('CCO.COC.CC=O.C=CO', hypothesis(), equations) == 'missing_rhea_provenance'


def test_imbalance_and_duplicate_fail_closed(equations):
    equations['b']['checks']['status'] = 'heavy_atom_imbalance'
    assert multi.audit_group('CCO.COC.CC=O.C=CO', hypothesis(), equations) == 'ineligible_equation'
    row = hypothesis(); row['candidate_groups'] = [['a', 'a']]
    assert multi.audit_group('CCO.COC', row, equations) == 'duplicate_equations'


def test_report_keeps_masks_and_unverified_assignment_separate(tmp_path, monkeypatch):
    base = tmp_path/'base'; base.mkdir()
    output = tmp_path/'output'; output.mkdir()
    mapping = output/'mapping'; mapping.mkdir()
    (base/'train_cgr_index.json').write_text(json.dumps({'q': {'mask':False}, 'p': {'mask':True}}))
    args = SimpleNamespace(output=output, baseline=base, split='reaction_smi')
    settings = dict(model='test', max_atoms=200, min_confidence=0)
    old = {'a': {'status':'structurally_valid_cgr'}}
    new = {'b': {'status':'structurally_valid_cgr'}}
    monkeypatch.setattr(multi.single, 'verify', lambda args: None)
    monkeypatch.setattr(multi, 'read_mapping_records', lambda directory: (old if directory == base else new, settings))
    audits = {'train': {
        'q':dict(audit_status='chemistry_consistent_hypothesis', equation_ids=['a','b'], recovery_status='decomposition_hypothesis'),
        'p':dict(audit_status='not_single_decomposition', equation_ids=[], recovery_status='unique_candidate')}}
    multi.report(args, mapping, audits)
    rows = json.loads((output/'train_multi_cgr_index.json').read_text())
    assert not rows['q']['baseline_mask'] and rows['q']['multi_cgr_hypothesis_mask']
    assert not rows['q']['assignment_verified']
    assert rows['p']['baseline_mask'] and not rows['p']['multi_cgr_hypothesis_mask']
    del new['b']
    multi.report(args, mapping, audits)
    rows = json.loads((output/'train_multi_cgr_index.json').read_text())
    assert not rows['q']['multi_cgr_hypothesis_mask']
    assert json.loads((output/'report.json').read_text())['pending_equations'] == 1
