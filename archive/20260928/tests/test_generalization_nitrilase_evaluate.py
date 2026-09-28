"""Synthetic-only checks; no external panel labels or scores are loaded."""
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/generalization_nitrilase_evaluate.py'
spec = importlib.util.spec_from_file_location('nitrilase_eval', PATH)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class NitrilaseEvaluationTest(unittest.TestCase):
    def test_exact_tie_expectations_match_exhaustive_orders(self):
        labels = np.array([0, 1, 1, 0, 1, 0])
        scores = np.array([2, 2, 2, 1, 1, 0.])
        all_values = []
        for a, b in itertools.product(itertools.permutations(range(3)), itertools.permutations(range(3,5))):
            order = list(a)+list(b)+[5]
            ranks = np.flatnonzero(labels[order])+1
            all_values.append([np.mean(1/ranks), 1/min(ranks), *[float(min(ranks)<=k) for k in evaluator.CUTS]])
        actual = evaluator.query_metrics(labels, scores)
        np.testing.assert_allclose([actual[k] for k in evaluator.RETRIEVAL], np.mean(all_values,axis=0))

    def test_all_zero_scores_are_random_not_catalog_success(self):
        actual=evaluator.query_metrics([1,1,0,0,0],np.zeros(5))
        self.assertAlmostEqual(actual['all_positive_mrr'],sum(1/i for i in range(1,6))/5)
        self.assertAlmostEqual(actual['hit_at_1'],.4)
        self.assertAlmostEqual(actual['hit_at_3'],.9)
        self.assertAlmostEqual(actual['auroc'],.5)
        self.assertAlmostEqual(actual['average_precision'],.4)
        self.assertTrue(actual['all_scores_tied'])

    def test_no_positive_queries_retained_but_discrimination_omitted(self):
        labels=np.array([[1,0,0],[0,0,0],[1,1,1]])
        result=evaluator.evaluate_matrix(np.arange(9).reshape(3,3),labels)
        r=result['reaction_to_enzyme']['summary']
        self.assertEqual(r['query_count'],3)
        self.assertEqual(r['candidate_count'],3)
        self.assertEqual(r['no_positive_query_count'],1)
        self.assertEqual(r['all_positive_query_count'],1)
        self.assertEqual(r['mixed_class_query_count'],1)
        self.assertEqual(r['omitted_discrimination_query_count'],2)
        self.assertAlmostEqual(r['all_queries']['all_positive_mrr'],r['positive_queries_only']['all_positive_mrr']*2/3)

    def test_bootstrap_is_paired_and_reproducible(self):
        y=np.eye(4,dtype=int)
        baseline=evaluator.evaluate_matrix(np.zeros((4,4)),y)
        method=evaluator.evaluate_matrix(y.astype(float),y)
        a=evaluator.paired_bootstrap(method,baseline,50,11)
        b=evaluator.paired_bootstrap(method,baseline,50,11)
        self.assertEqual(a,b)
        self.assertGreater(a['reaction_to_enzyme']['all_queries']['all_positive_mrr']['lower_95'],0)
        same=evaluator.paired_bootstrap(method,method,50,11)
        self.assertEqual(same['enzyme_to_reaction']['mixed_class_queries_only']['auroc']['upper_95'],0)

    def test_generic_enzyme_prior_has_no_permutation_advantage(self):
        y=np.array([[1,0,0],[0,1,0],[0,0,0],[1,1,0]])
        s=np.tile([.9,.2,.1],(4,1))
        result=evaluator.permutation_control(s,y,30,12)
        for direction in result.values():
            for item in direction.values():
                if item:
                    self.assertAlmostEqual(item['correct_minus_null_mean'],0)
                    self.assertEqual(item['one_sided_p_null_ge_correct'],1)

    def test_reaction_conditioning_detected_synthetically(self):
        y=np.eye(8,dtype=int)
        result=evaluator.permutation_control(y.astype(float),y,200,12)
        for direction in result.values():
            self.assertGreater(direction['all_positive_mrr']['correct_minus_null_mean'],.5)
            self.assertLess(direction['auroc']['one_sided_p_null_ge_correct'],.02)

    def test_complete_label_matrix_and_duplicate_failures(self):
        catalog={'query_ids':['q1','q2'],'proteins':['p1','p2']}
        rows=[dict(query_id=q,protein_id=p,label=int(q=='q1')) for q in catalog['query_ids'] for p in catalog['proteins']]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'labels.csv'
            def write(values):
                with p.open('w') as h:
                    writer=csv.DictWriter(h,fieldnames=['query_id','protein_id','label']);writer.writeheader();writer.writerows(values)
            write(rows)
            np.testing.assert_array_equal(evaluator.load_labels(catalog,p,4),[[1,1],[0,0]])
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError,'Every assay'):evaluator.load_labels(catalog,p,4)
            write(rows+rows[:1])
            with self.assertRaisesRegex(ValueError,'Duplicate'):evaluator.load_labels(catalog,p,4)

    def test_nonfinite_and_nonbinary_rejected(self):
        with self.assertRaises(ValueError):evaluator.query_metrics([1,0],[1,np.nan])
        with self.assertRaises(ValueError):evaluator.query_metrics([1,-1],[1,0])

    def test_dual_freeze_score_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def dump(name,value):
                p=root/name;p.write_text(json.dumps(value));return p
            old=dump('phase1.json',{'schema':'phase1'})
            old_id=evaluator.identity(old)
            new=dump('phase2.json',{'original_frozen_recipe':old_id})
            new_id=evaluator.identity(new)
            catalog=dump('catalog.json',{'proteins':['p1','p2'],'query_ids':['q1','q2']})
            bundle=dump('bundle.json',{'frozen_recipe':old_id,'phase2_frozen_recipe':new_id})
            scores=root/'scores.npz';np.savez(scores,selected=np.eye(2))
            dump('complete.json',{'bundle':evaluator.identity(bundle),'inputs':{'catalog':evaluator.identity(catalog)},'labels_used':False,'output_sha256':evaluator.sha256(scores)})
            entry={'path':'scores.npz','score_key':'selected'}
            actual=evaluator.load_phase2_scores(entry,root,catalog,{},new_id,(2,2))
            np.testing.assert_array_equal(actual,np.eye(2))
            with self.assertRaisesRegex(ValueError,'different phase2'):
                evaluator.load_phase2_scores(entry,root,catalog,{},old_id,(2,2))
            np.savez(scores,selected=np.ones((2,2)))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):
                evaluator.load_phase2_scores(entry,root,catalog,{},new_id,(2,2))


if __name__=='__main__':
    unittest.main()
