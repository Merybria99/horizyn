"""Scientific-contract checks, including parity with the full released evaluator.

P450_OFFICIAL_SOURCE points at the pinned source. P450_OFFICIAL_WORK points at
the prepared official_evaluation/scripts directory (for its RHEA relative path).
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
from p450_protocol import (KEYS, align_scores, apply_official_prior, score_metrics, validate_panel)


def fixture():
    return pd.DataFrame([dict(CANO_RXN_SMILES=r, UniprotID=f'u{i:03}', sequence=f'SEQ{i}',
        Label=int(i in ([3, 100] if r == 'r1' else [24, 25]))) for r in ['r1', 'r2'] for i in range(490)])


class ProtocolTests(unittest.TestCase):
    def test_reject_partial_and_duplicate_scores(self):
        panel = fixture(); scores = panel.assign(pred=1.)
        for bad in [scores.iloc[:-1], pd.concat([scores, scores.iloc[:1]])]:
            with self.assertRaises(ValueError):
                align_scores(panel, bad)

    def test_reject_nonfinite_and_changed_labels(self):
        panel = fixture()
        for value in [np.nan, np.inf, -np.inf]:
            scores = panel.assign(pred=1.); scores.loc[0, 'pred'] = value
            with self.assertRaises(ValueError):
                align_scores(panel, scores)
        scores = panel.assign(pred=1.); scores.loc[0, 'Label'] = 1
        with self.assertRaises(ValueError):
            align_scores(panel, scores)

    def test_restores_released_order(self):
        panel = fixture(); scores = panel.assign(pred=np.arange(len(panel)))
        actual = align_scores(panel, scores.sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(actual, scores.astype({'pred': float}))

    def test_pool_and_sequence_integrity(self):
        panel = fixture(); validate_panel(panel)
        for bad in [panel.iloc[:-1], panel.assign(sequence=['wrong'] + panel.sequence.tolist()[1:])]:
            with self.assertRaises(ValueError):
                validate_panel(bad)

    def test_prior_zeroes_half_without_reducing_denominator(self):
        panel = fixture().assign(pred=.5)
        prior = dict(zip(panel.CANO_RXN_SMILES + '_' + panel.UniprotID, np.tile(np.arange(490), 2)))
        result = apply_official_prior(panel, prior)
        self.assertEqual(len(result), 980)
        self.assertEqual(int((result.pred == 0).sum()), 490)
        self.assertEqual(result.groupby(KEYS[0]).size().tolist(), [490, 490])
        prior.pop(next(iter(prior)))
        with self.assertRaises(ValueError):
            apply_official_prior(panel, prior)

    def test_positive_cosine_adapter_preserves_order_and_gate(self):
        x = np.array([-.9, -.5, -.1, 0., .2, 1.])
        y = (x + 2.) / 3.
        np.testing.assert_array_equal(np.argsort(x), np.argsort(y))
        self.assertTrue((y > 0).all())

    def test_floor_cutoffs_and_multiple_positives(self):
        source = os.environ['P450_OFFICIAL_SOURCE']
        panel = fixture()
        # Positive at rank 4: Top1% hit; second query best positive at rank25:
        # Top5% (floor 24) must be a miss. All known positives count.
        scores = panel.assign(pred=np.tile(-np.arange(490), 2))
        result = score_metrics(scores, panel, source)
        self.assertEqual(result, {'Top 1.0%': .5, 'Top 3.0%': .5, 'Top 5.0%': .5})

    def test_full_released_evaluator_parity_including_ties(self):
        import sys
        source = Path(os.environ['P450_OFFICIAL_SOURCE']).resolve()
        old = Path.cwd()
        os.chdir(os.environ['P450_OFFICIAL_WORK'])
        sys.path.insert(0, str(source))
        try:
            spec = importlib.util.spec_from_file_location('upstream_p450_test', source / 'scripts/evaluate_external-test.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            panel = fixture()
            rng = np.random.default_rng(42)
            scores = panel.assign(pred=rng.integers(0, 10, size=len(panel)) / 10.)
            prior = dict(zip(panel.CANO_RXN_SMILES + '_' + panel.UniprotID, rng.integers(0, 4, size=len(panel))))
            ours = apply_official_prior(scores, prior)
            expected_metrics = score_metrics(ours, panel, source)
            original = module.eval_top_rank_result
            called = []
            def capture(frame, *args, **kwargs):
                pd.testing.assert_frame_equal(frame[ours.columns], ours)
                result = original(frame, *args, **kwargs)
                self.assertEqual(result[1], expected_metrics)
                called.append(True)
                return result
            module.eval_top_rank_result = capture
            with tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                panel.to_csv(directory / 'test.csv', index=False)
                scores.to_csv(directory / 'pred.csv', index=False)
                with open(directory / 'corr_score_map.pkl', 'wb') as handle:
                    pickle.dump(prior, handle)
                with contextlib.redirect_stdout(io.StringIO()):
                    module.evaluate_external_test(str(directory / 'pred.csv'), str(directory / 'test.csv'), 'unused')
            self.assertEqual(called, [True])
        finally:
            os.chdir(old)


if __name__ == '__main__':
    unittest.main(verbosity=2)
