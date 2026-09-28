import importlib.util
from pathlib import Path
import unittest
import numpy as np

p=Path(__file__).resolve().parents[1]/'scripts/generalization_large_case1_length_audit.py'
s=importlib.util.spec_from_file_location('length_audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class LengthAuditTests(unittest.TestCase):
    def test_uniform_cutoff_tie(self):
        w,stable,b=m.weights(np.array([3.,2.,2.,2.,0.]),2)
        np.testing.assert_allclose(w,[1,1/3,1/3,1/3,0]);self.assertEqual(stable.sum(),2)
        self.assertEqual(b['boundary_tie_size'],3)
    def test_constant_scores_do_not_prefer_order(self):
        w,_,_=m.weights(np.zeros(8),2);np.testing.assert_array_equal(w,np.full(8,.25))
    def test_length_boundaries_and_capped_full_length(self):
        f=np.array([50,99,1022,1023,1024,2000]);s=np.minimum(f,1024)
        d=m.describe(f,s,np.zeros(6,bool),np.ones(6))
        self.assertEqual(d['counts']['full_aa_length>1024']['count'],1)
        self.assertEqual(d['counts']['stored_residue_length==1024']['count'],2)
        self.assertEqual(d['counts']['stored_residue_length>1022']['count'],3)
        self.assertEqual(d['lengths']['native_effective']['maximum_positive_weight'],1022)
        self.assertEqual(sum(x['count'] for x in d['full_aa_bins'].values()),6)
    def test_zero_weight_candidates_do_not_affect_summary(self):
        d=m.describe(np.array([50,10000]),np.array([50,1024]),np.array([False,True]),np.array([1.,0.]))
        self.assertEqual(d['lengths']['full_aa']['weighted_mean'],50)
        self.assertEqual(d['lengths']['full_aa']['maximum_positive_weight'],50)
    def test_invalid_cutoff_or_scores(self):
        for k in [0,3]:
            with self.assertRaises(ValueError):m.weights(np.ones(2),k)
        with self.assertRaises(ValueError):m.weights(np.array([1.,np.nan]),1)

if __name__=='__main__':unittest.main()
