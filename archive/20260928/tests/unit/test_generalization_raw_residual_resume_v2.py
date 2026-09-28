"""The guard amendment changes one serialized key and no training statements."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def function(path,name):
    return next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef) and node.name==name)

def test_training_and_selection_ast_identical_except_rank_key():
    original=ROOT/'scripts/generalization_raw_residual.py'
    text=original.read_text();old="f[d+'_'+k]";new="f['positive__'+d+'__'+k]"
    assert ast.unparse(function(original,'train_one')).count("f[d + '_' + k]")==1
    patched=ast.parse(text.replace(old,new))
    expected=next(node for node in patched.body if isinstance(node,ast.FunctionDef) and node.name=='train_one')
    actual=function(ROOT/'scripts/generalization_raw_residual_resume_v2.py','train_one')
    assert ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)

def test_amended_finalizer_and_original_serializer_agree():
    serializer=function(ROOT/'scripts/generalization_composed_calibration.py','save_result')
    finalizer=function(ROOT/'scripts/generalization_raw_residual_resume_v2.py','finalize_existing')
    assert 'positive__' in ast.unparse(serializer)
    assert "'positive__' + d + '__' + k" in ast.unparse(finalizer)
