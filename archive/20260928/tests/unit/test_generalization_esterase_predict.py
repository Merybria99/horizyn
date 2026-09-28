"""Fail closed before panel values or features are opened."""
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE=Path(__file__).resolve().parents[2]/'scripts/generalization_esterase_predict.py'
spec=importlib.util.spec_from_file_location('esterase_predict',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_changed_source_receipt_rejected(tmp_path):
    target=tmp_path/'receipt.json';target.write_text('{}')
    identity=module.record(target);target.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Changed authenticated artifact'):module.checked(identity)

@pytest.mark.parametrize('change',[{}, {'unknown':True}, {'phase2_csr_dense':False}])
def test_partial_or_failed_numerical_audit_rejected(change):
    complete={key:True for key in module.NUMERICAL_CHECKS}
    if not change:complete.pop('morgan_singleton_scores')
    else:complete.update(change)
    with pytest.raises(ValueError,match='numerical acceptance'):module.check_numerical_receipt(complete)

def test_complete_numerical_contract_accepted():
    module.check_numerical_receipt({key:True for key in module.NUMERICAL_CHECKS})

def test_reaction_csv_must_be_exact_native_export_identity(tmp_path):
    target=tmp_path/'full_participants.csv';target.write_text('id,smiles\nR,O.C>>O.C\n')
    identity=module.record(target)
    native={'outputs':{'feature_reactions.csv':identity}}
    assert module.contains_record([native],identity)
    assert not module.contains_record([native],dict(identity,sha256='0'*64))
    assert not module.contains_record([native],dict(identity,path=str(tmp_path/'molecular.csv')))

@pytest.mark.parametrize('field,value',[('primary','phase2'),('expected_shape',[86,144]),('methods',module.METHODS[::-1])])
def test_wrong_panel_or_role_rejected_before_any_input_access(tmp_path,field,value):
    plan=dict(schema='esterase_frozen_evaluation_protocol_v1',frozen_before_predictions=True,
              methods=module.METHODS,expected_shape=[86,145],primary='morgan',baseline='F3_fp64',predecessor='phase2')
    plan[field]=value;path=tmp_path/'protocol.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):module.checked_protocol(path)
