import hashlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'scripts/generalization_large_case1_training_exposure.py'
s=importlib.util.spec_from_file_location('training_exposure',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class TrainingExposureTests(unittest.TestCase):
    def test_digest_hit_requires_real_string_equality(self):
        seq=b'ACDE';h=hashlib.sha256(seq).digest()
        self.assertEqual(m.exact_matches(seq,{h:[(b'WRONG','false'),(seq,'true')]}),['true'])
    def test_ambiguous_residues_not_wildcards(self):
        self.assertEqual(m.exact_matches(b'ACDX',{hashlib.sha256(b'ACDX').digest():[(b'ACDE','p')]}),[])
    def test_uniform_boundary_ties(self):
        r=m.top_exposure(np.array([3.,2.,2.,2.]),np.array([1,1,0,0]),2)
        self.assertAlmostEqual(r['uniform_tie_expected_seen_count'],4/3)
        self.assertEqual(r['stable_order_seen_count'],2)
    def test_constant_scores_recover_population_fraction(self):
        r=m.top_exposure(np.zeros(8),np.array([1,0,0,1,0,0,0,0]),3)
        self.assertEqual(r['uniform_tie_expected_seen_fraction'],.25)
    def test_streaming_fasta_semantics_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.fasta';p.write_bytes(b'>id description\nac d\nEX\n>p\nMKW\n')
            record=m.identity(p)
            self.assertEqual(list(m.fasta_records(record)),[('id',b'ACDEX'),('p',b'MKW')])
            record['sha256']='wrong'
            with self.assertRaises(ValueError):list(m.fasta_records(record))
    def test_no_activity_result_before_gate(self):
        with self.assertRaises(ValueError):m.gate({'schema':'wrong'})

if __name__=='__main__':unittest.main()
