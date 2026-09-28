"""Scientific metric contracts, using synthetic scores only."""
import importlib.util
from pathlib import Path
import unittest
import json
import tempfile

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/generalization_external_evaluate.py'
SPEC = importlib.util.spec_from_file_location('external_eval', SCRIPT)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class ExternalMetricContracts(unittest.TestCase):
    def test_auc_and_ap_ties_are_threshold_grouped(self):
        got = evaluator.conditional_auc_ap([1., 1., 0.], [True, False, True])
        self.assertAlmostEqual(got['auc'], .25)
        self.assertAlmostEqual(got['average_precision'], 7 / 12)
        tied = evaluator.conditional_auc_ap([0., 0., 0., 0.], [True, False, True, False])
        self.assertEqual(tied['auc'], .5)
        self.assertEqual(tied['average_precision'], .5)

    def test_stable_ties_use_catalog_order(self):
        np.testing.assert_array_equal(evaluator.rank_vector([.1, .8, .8, .2]), [4, 1, 2, 3])

    def test_all_positive_mrr_and_full_stratum_candidates(self):
        # R0 positives E0/E1 rank 1/3; first-positive MRR is 1, all-positive is 2/3.
        # E0's positive R0 ranks 2 among all three reactions. Restricting truth to
        # R0 must keep the other two reaction candidates and retain rank 2.
        matrix = np.asarray([[3., 1., 2.], [4., 5., 0.], [2., 3., 6.]], dtype=np.float32)
        meta = dict(reaction=np.asarray([0, 0, 1]), enzyme=np.asarray([0, 1, 2]),
                    subsets={'all': {0, 1, 2}, 'restricted': {0}})
        value, _ = evaluator.evaluate_p450(matrix, meta)
        r2e = value['restricted']['reaction_to_enzyme']
        self.assertAlmostEqual(r2e['summary']['reactzyme_mrr'], 2 / 3, places=6)
        self.assertEqual(r2e['summary']['first_positive_mrr'], 1.)
        self.assertEqual(r2e['summary']['candidate_count'], 3)
        e2r = value['restricted']['enzyme_to_reaction']
        np.testing.assert_array_equal(e2r['per_query']['first_rank'], [2., 3.])
        self.assertEqual(e2r['summary']['candidate_count'], 3)
        self.assertEqual(e2r['summary']['num_queries'], 2)
        self.assertEqual(e2r['summary']['top_20'], 1.)

    def test_bootstrap_is_paired_and_reproducible(self):
        a = {'query_index': np.asarray([0, 3, 7]), 'reactzyme_mrr': np.asarray([.5, .6, .7])}
        b = {'query_index': np.asarray([0, 3, 7]), 'reactzyme_mrr': np.asarray([.4, .5, .6])}
        for k in evaluator.CUTS:
            a[f'top_{k}'] = np.asarray([1., 0., 1.])
            b[f'top_{k}'] = np.asarray([0., 0., 1.])
        one = evaluator.paired_bootstrap(a, b, 500, 42)
        two = evaluator.paired_bootstrap(a, b, 500, 42)
        self.assertEqual(one, two)
        self.assertAlmostEqual(one['reactzyme_mrr']['delta'], .1)
        self.assertAlmostEqual(one['reactzyme_mrr']['lower_95'], .1)
        self.assertAlmostEqual(one['reactzyme_mrr']['upper_95'], .1)
        b['query_index'] = np.asarray([0, 2, 7])
        with self.assertRaises(ValueError):
            evaluator.paired_bootstrap(a, b, 500, 42)

    def test_permutation_preserves_generic_enzyme_prior(self):
        scores = np.tile(np.arange(8., 0., -1.), (8, 1))
        meta = dict(reaction=np.arange(8), enzyme=np.arange(8),
                    subsets={'all': set(range(8)), 'both_absent': {0, 1, 2, 3}})
        got = evaluator.reaction_permutation_control(scores, meta, 1000, 9)
        for stratum in got.values():
            for metric in stratum.values():
                self.assertAlmostEqual(metric['correct'], metric['null_mean'])
                self.assertAlmostEqual(metric['correct'], metric['null_lower_95'])
                self.assertEqual(metric['one_sided_p_null_ge_correct'], 1.)
        # Reusing an identical ranking model uses the same seeded null draws.
        self.assertEqual(got, evaluator.reaction_permutation_control(scores * 2, meta, 1000, 9))

    def test_permutation_detects_correct_query_conditioning(self):
        scores = np.eye(8, dtype=np.float32) * 10 + np.arange(8)[None, :] * .001
        meta = dict(reaction=np.arange(8), enzyme=np.arange(8),
                    subsets={'all': set(range(8)), 'both_absent': {0, 1, 2, 3}})
        got = evaluator.reaction_permutation_control(scores, meta, 1000, 9)
        result = got['all']['reactzyme_mrr']
        self.assertEqual(result['correct'], 1.)
        self.assertLess(result['null_mean'], .5)
        self.assertLess(result['one_sided_p_null_ge_correct'], .01)
        self.assertEqual(got['both_absent']['reactzyme_mrr']['queries'], 4)

    def test_score_receipt_binds_array_and_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            catalog = directory / 'catalog.json'
            catalog.write_text(json.dumps({'proteins': ['a', 'b'], 'query_ids': ['q']}))
            scores = directory / 'scores.npz'
            np.savez(scores, selected=np.asarray([[1., 2.]]), baseline=np.asarray([[2., 1.]]))
            receipt = {'output_sha256': evaluator.sha256(scores),
                       'inputs': {'catalog': {'sha256': evaluator.sha256(catalog)}}, 'labels_used': False}
            (directory / 'complete.json').write_text(json.dumps(receipt))
            provenance = {}
            for key in ('baseline', 'selected'):
                got = evaluator.load_scores({'path': 'scores.npz', 'score_key': key}, directory,
                                            catalog, (1, 2), provenance, {'sha256': 'frozen'})
                self.assertEqual(got.shape, (1, 2))
            self.assertEqual(provenance[str(scores)]['score_keys'], ['baseline', 'selected'])
            receipt['inputs']['catalog']['sha256'] = 'wrong'
            (directory / 'complete.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'catalog'):
                evaluator.load_scores({'path': 'scores.npz', 'score_key': 'selected'}, directory,
                                      catalog, (1, 2), {}, {'sha256': 'frozen'})

    def test_explicit_score_hash_alone_cannot_establish_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            catalog = directory / 'catalog.json'
            catalog.write_text(json.dumps({'proteins': ['a', 'b'], 'query_ids': ['q']}))
            scores = directory / 'scores.npz'
            np.savez(scores, scores=np.asarray([[1., 2.]]))
            entry = {'path': 'scores.npz', 'sha256': evaluator.sha256(scores)}
            with self.assertRaisesRegex(ValueError, 'order'):
                evaluator.load_scores(entry, directory, catalog, (1, 2), {}, {'sha256': 'frozen'})
            entry['catalog_sha256'] = evaluator.sha256(catalog)
            evaluator.load_scores(entry, directory, catalog, (1, 2), {}, {'sha256': 'frozen'})

    def test_random_hit_and_recall_count(self):
        self.assertAlmostEqual(evaluator.random_hit_probability(5, 2, 2), .7)
        self.assertEqual(evaluator.random_hit_probability(5, 2, 4), 1.)
        r = evaluator.recall_summary(np.arange(1, 124), {0, 6, 90})
        self.assertEqual(r['recovered_at_5'], 1)
        self.assertAlmostEqual(r['recall_at_5'], 1 / 3)
        self.assertAlmostEqual(r['random_expected_recovered_at_5'], 15 / 123)


if __name__ == '__main__':
    unittest.main()
