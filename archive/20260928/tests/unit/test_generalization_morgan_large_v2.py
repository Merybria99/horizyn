import ast,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('v2',ROOT/'scripts/generalization_morgan_large_case1_evaluate_v2.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_scientific_ast_unchanged():
 old=ast.parse((ROOT/'scripts/generalization_morgan_large_case1_evaluate.py').read_text());new=ast.parse((ROOT/'scripts/generalization_morgan_large_case1_evaluate_v2.py').read_text())
 for name in ['run','validate_plan','validate_original_gate']:
  before=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name);after=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
  assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False)
def test_amendment_binds_original_source(tmp_path):
 helper=ROOT/'scripts/generalization_large_case1_evaluate.py';source=m.identity(helper)
 protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps({'evaluation_source':source}));record=m.identity(protocol)
 freeze=dict(path=str(tmp_path/'freeze.json'),sha256='freeze')
 amendment=dict(schema='morgan_large_evaluation_source_amendment_v1',original_freeze=freeze,original_scan_protocol=record,approved_original_evaluation_source=source,reason='Missing source-list entry already pinned by original protocol',implementation_sources=[source])
 assert m.validate_amendment(amendment,freeze,dict(original_scan_protocol=record))==source
 for key in ('original_freeze','original_scan_protocol','approved_original_evaluation_source'):
  bad=dict(amendment);bad[key]=dict(amendment[key],sha256='changed')
  with pytest.raises(ValueError):m.validate_amendment(bad,freeze,dict(original_scan_protocol=record))
