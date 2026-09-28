import ast,copy,importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/generalization_calibration_external_evaluate_v2.py'
s=importlib.util.spec_from_file_location('calibration_external_v2_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class CalibrationAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.f={'path':str(ROOT/'frozen_recipe.json'),'sha256':'original'}
        self.a={'schema':'calibration_evaluation_guard_amendment_v1','original_freeze':self.f,
                'reason':'Prior two feature receipts intentionally omit a later freeze identity.',
                'implementation_sources':[{'path':str(P),'sha256':'source'}],
                'approved_missing_freeze_receipts':[{'path':str(m.RUN/(p+'_audit/features/feature_bundle_receipt.json')),'sha256':p} for p in ['nitrilase','aminotransferase']]}
    def test_only_two_exact_external_receipts_allowed(self):
        with patch.object(m,'checked',side_effect=lambda r:Path(r['path'])) as check:
            m.validate_amendment(self.a,self.f);self.assertEqual(check.call_count,2)
        a=copy.deepcopy(self.a);a['approved_missing_freeze_receipts'][0]['path']=str(m.RUN/'case1_audit/features/feature_bundle_receipt.json')
        with self.assertRaises(ValueError):m.validate_amendment(a,self.f)
    def test_wrong_freeze_rejected(self):
        for field,value in [('sha256','changed'),('path',str(ROOT/'other.json'))]:
            a=copy.deepcopy(self.a);a['original_freeze'][field]=value
            with self.assertRaises(ValueError):m.validate_amendment(a,self.f)
    def test_no_implicit_or_unexplained_amendment(self):
        for key,value in [('schema','other'),('reason',''),('implementation_sources',[]),('approved_missing_freeze_receipts',[])]:
            a=copy.deepcopy(self.a);a[key]=value
            with self.assertRaises(ValueError):m.validate_amendment(a,self.f)
    def test_metrics_and_rectangles_are_ast_identical(self):
        old=ast.parse((ROOT/'scripts/generalization_calibration_external_evaluate.py').read_text());new=ast.parse(P.read_text())
        for name in ['validate_contract','rank_invariance','rectangles','evaluate_universe']:
            a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(a),ast.dump(b))

if __name__=='__main__':unittest.main()
