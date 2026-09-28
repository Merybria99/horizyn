import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[1]/'scripts/generalization_calibration_external_evaluate.py'
spec=importlib.util.spec_from_file_location('calibration_external_test',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CalibrationExternalTests(unittest.TestCase):
    def setUp(self):
        self.freeze={'schema':'post_evaluation_calibration_transfer_freeze_v1',
            'frozen_before_new_predictions':True,'development_exposed':True,
            'rule':{'gamma_enzyme':0.,'gamma_reaction':1.,'enzyme_mean_scheme':'uniform_proteins'}}
        self.plan={'schema':'calibration_external_evaluation_plan_v1','methods':list(m.METHODS),
            'panels':list(m.SHAPES),'primary':'calibrated_seed42','training_split':'reaction_smi',
            'development_exposed':True,'bootstrap_replicates':10000,'permutation_replicates':1000,'seed':20260920}
    def test_fixed_contract(self):m.validate_contract(self.freeze,self.plan)
    def test_cannot_relabel_primary_or_drop_seed(self):
        for change in [{'primary':'calibrated_seed17'},{'methods':list(m.METHODS)[:-1]}]:
            plan=dict(self.plan,**change)
            with self.assertRaises(ValueError):m.validate_contract(self.freeze,plan)
    def test_cannot_change_training_rule_or_exposure(self):
        for rule in [{'gamma_enzyme':1.,'gamma_reaction':1.,'enzyme_mean_scheme':'uniform_proteins'},
                     {'gamma_enzyme':0.,'gamma_reaction':1.,'enzyme_mean_scheme':'edge_weighted'}]:
            with self.assertRaises(ValueError):m.validate_contract(dict(self.freeze,rule=rule),self.plan)
        with self.assertRaises(ValueError):m.validate_contract(dict(self.freeze,development_exposed=False),self.plan)
    def test_rank_constant_shift(self):
        a=np.array([[3.,1.,1.],[0.,2.,-1.]])
        r=m.rank_invariance(a,a-np.array([[1.],[7.]]))
        self.assertTrue(r['all_stable_orders_identical']);self.assertTrue(r['all_weak_orders_and_ties_identical'])
        self.assertEqual([x['score_shift_spread'] for x in r['rows']],[0.,0.])
    def test_tie_creation_is_reported(self):
        a=np.array([[3.,2.,1.]])
        r=m.rank_invariance(a,np.array([[3.,2.,2.]]))
        self.assertTrue(r['all_stable_orders_identical']);self.assertFalse(r['all_weak_orders_and_ties_identical'])
    def test_reversal_and_nonfinite_rejected(self):
        a=np.array([[3.,1.,2.]])
        self.assertFalse(m.rank_invariance(a,-a)['all_stable_orders_identical'])
        with self.assertRaises(ValueError):m.rank_invariance(a,np.array([[np.nan,1.,2.]]))
    def test_fixed_rectangles_preserve_named_candidate_counts(self):
        c={'query_ids':[f'r{i}' for i in range(18)],'proteins':[f'p{i}' for i in range(25)]}
        overlap={'matches':{'contains_pair_neutralized':c['query_ids'][:15]},'exact_sequence_matches':c['proteins'][:5]}
        rectangles=m.rectangles(c,np.array([True]*24+[False]),overlap)
        self.assertEqual([(int(r.sum()),int(p.sum())) for _,r,p in rectangles],[(18,25),(18,24),(3,25),(18,20),(3,20)])
        with self.assertRaises(ValueError):m.rectangles(c,np.ones(25,bool),overlap)
    def test_method_seed_key_bindings_are_fixed(self):
        self.assertEqual(m.METHODS['calibrated_seed17'],(17,'selected'))
        self.assertEqual(m.METHODS['phase2_seed73'],(73,'parent'))
        self.assertEqual(m.METHODS['F3_fp64'],(42,'baseline_fp64'))

if __name__=='__main__':unittest.main()
