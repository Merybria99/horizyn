"""Strict input checks and a thin adapter to the released P450 evaluator."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = 'e6887c4109e7c4a861b9cb1a3e33beede90103dc'
DATA_SHA = 'dd516780485ff1c2be9255071189f459adec49bccfa795a49cd8ca6571cb672d'
KEYS = ['CANO_RXN_SMILES', 'UniprotID']
PERCENTS = [.01, .03, .05]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_panel(frame, released=False):
    if frame[KEYS + ['sequence', 'Label']].isna().any().any():
        raise ValueError('Missing panel fields')
    if frame.duplicated(KEYS).any():
        raise ValueError('Duplicate reaction/enzyme pairs')
    if not set(frame.Label) <= {0, 1}:
        raise ValueError('Nonbinary labels')
    if frame.groupby('UniprotID').sequence.nunique().max() != 1:
        raise ValueError('Conflicting enzyme sequences')
    pools = [set(g.UniprotID) for _, g in frame.groupby(KEYS[0])]
    if not pools or any(pool != pools[0] for pool in pools):
        raise ValueError('Candidate pools differ')
    if (frame.groupby(KEYS[0]).Label.sum() < 1).any():
        raise ValueError('Reaction without a known positive')
    counts = dict(rows=len(frame), reactions=len(pools), enzymes=len(pools[0]), positives=int(frame.Label.sum()))
    if released and counts != dict(rows=93590, reactions=191, enzymes=490, positives=318):
        raise ValueError(f'Released panel changed: {counts}')
    return counts


def align_scores(panel, predictions):
    """Reject partial/duplicate/nonfinite scores; restore released row order."""
    if predictions.duplicated(KEYS).any():
        raise ValueError('Duplicate prediction pairs')
    expected = pd.MultiIndex.from_frame(panel[KEYS])
    actual = pd.MultiIndex.from_frame(predictions[KEYS])
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError('Prediction coverage differs from the complete panel')
    aligned = predictions.set_index(KEYS).loc[expected].reset_index()
    if not np.isfinite(aligned['pred'].to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite predictions')
    for column in ['sequence', 'Label']:
        if column in aligned and not np.array_equal(aligned[column].to_numpy(), panel[column].to_numpy()):
            raise ValueError(f'Prediction {column} differs from released panel')
    result = panel.copy()
    result['pred'] = aligned['pred'].to_numpy(dtype=float)
    return result


def apply_official_prior(frame, prior):
    frame = frame.copy()
    frame['corr_score'] = (frame[KEYS[0]] + '_' + frame[KEYS[1]]).map(prior)
    if not np.isfinite(frame.corr_score).all():
        raise ValueError('Missing/nonfinite prior values')
    groups = []
    for _, group in frame.groupby(KEYS[0]):
        # These three lines deliberately match evaluate_external-test.py,
        # including pandas default tie handling and zeroing (not deleting).
        group = group.sort_values('corr_score', ascending=False).reset_index(drop=True)
        group.loc[group[len(group)//2:].index, 'pred'] = 0
        groups.append(group)
    return pd.concat(groups)


def official_rank_function(source):
    """Execute the unchanged standalone function; avoid unrelated CLI imports.

    The AST body is taken directly from the pinned evaluate.py, not reimplemented.
    Full external-evaluator parity is also checked by the integration tests.
    """
    path = Path(source) / 'evaluate.py'
    parsed = ast.parse(path.read_text())
    function = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_top_rank_result')
    namespace = {'pd': pd, 'np': np, 'UID_COL': 'UniprotID'}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function.name]


def score_metrics(frame, panel, source):
    positives = {r: set(g.UniprotID) for r, g in panel[panel.Label == 1].groupby(KEYS[0])}
    _, result = official_rank_function(source)(frame, set(panel[KEYS[0]]), true_enz_dict=positives,
                                               top_percent=PERCENTS, to_print=False)
    return result
