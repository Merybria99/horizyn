#!/usr/bin/env python3
"""Evaluation-only compatibility for two exact, pre-freeze input receipts.

The frozen checker remains unchanged. The only amended rule permits a missing
optional freeze_sha256 field for the two exact approved external receipts,
while authenticating their checkpoint and source receipt chain. Scoring, model
state, labels and metric computation are not changed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from generalization_calibration_transfer import validate_freeze,same
from horizyn.generalization_retrieval import checked_artifact,sha256


def validate_amendment(amendment,freeze_record):
    freeze=validate_freeze(freeze_record)
    if amendment.get('schema')!='calibration_evaluation_guard_amendment_v1' or not same(amendment['original_freeze'],freeze_record):
        raise ValueError('Wrong evaluation amendment lineage')
    for item in amendment['implementation_sources']:checked_artifact(item)
    pinned={str(Path(x['path']).resolve()):x['sha256'] for x in amendment['implementation_sources']}
    if pinned.get(str(Path(__file__).resolve()))!=sha256(__file__):raise ValueError('Guard source not pinned')
    permitted={str(Path(row['input_receipt']['path']).resolve()):row['input_receipt']['sha256'] for row in freeze['approved_predictions'] if row['panel'] in ('nitrilase','aminotransferase')}
    supplied={str(Path(row['path']).resolve()):row['sha256'] for row in amendment['approved_missing_freeze_receipts']}
    if len(permitted)!=2 or supplied!=permitted:raise ValueError('Compatibility must name exactly the two approved receipts')
    for row in amendment['approved_missing_freeze_receipts']:checked_artifact(row)
    return freeze


def validate_input_lineage(inp,input_record,parent,freeze,amendment):
    if inp.get('schema')!='generalization_feature_bundle_receipt_v1':raise ValueError('Invalid input receipt schema')
    supplied=inp.get('freeze_sha256')
    if supplied is not None:
        if supplied!=freeze['original_frozen_recipe']['sha256']:raise ValueError('Input freeze mismatch')
        return
    if not any(same(input_record,item) for item in amendment['approved_missing_freeze_receipts']):
        raise ValueError('Unapproved missing-freeze receipt')
    if inp.get('labels_used') is not False or not same(inp['checkpoint'],parent['base_checkpoint']):
        raise ValueError('Missing-freeze checkpoint/label-use mismatch')
    checked_artifact(inp['checkpoint'])
    for item in inp['source_receipts']:checked_artifact(item)


def checked_prediction(freeze_record,score_path,catalog_path,split,seed,score_key,*,amendment):
    """Authenticate a complete prediction without reading any assay labels."""
    freeze=validate_amendment(amendment,freeze_record);score_path=Path(score_path);receipt_path=score_path.parent/'complete.json';receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='generalization_predictions_v1' or receipt.get('phase')!='post_evaluation_calibration' or receipt.get('labels_used') is not False:raise ValueError('Wrong prediction phase or label use')
    if not same(receipt['calibration_freeze'],freeze_record) or receipt['output_sha256']!=sha256(score_path):raise ValueError('Prediction freeze/output mismatch')
    if receipt['inputs']['catalog']['sha256']!=sha256(catalog_path):raise ValueError('Prediction/catalog mismatch')
    if score_key not in ['selected','parent','baseline','baseline_fp64']:raise ValueError('Unregistered score key')
    rows=[row for row in freeze['approved_models'] if row['split']==split and row['seed']==seed and same(row['parent_bundle'],receipt['bundle']) and same(row['calibration_state'],receipt['calibration_state'])]
    if len(rows)!=1:raise ValueError('Unapproved model/mean tuple')
    inputs=[row for row in freeze['approved_predictions'] if row['split']==split and row['seed']==seed and same(row['input_receipt'],receipt['input_receipt']) and same(row['parent_scores'],receipt['parent_reference']['scores'])]
    if len(inputs)!=1:raise ValueError('Unapproved prediction input/control tuple')
    for key in ['bundle','calibration_state','input_receipt']:checked_artifact(receipt[key])
    from horizyn.generalization_phase2 import validate_phase2_bundle
    parent=json.loads(checked_artifact(receipt['bundle']).read_text());validate_phase2_bundle(parent,Path(receipt['bundle']['path']).parent)
    if parent['split']!=split or parent['seed']!=seed:raise ValueError('Parent split/seed mismatch')
    for key in ['phase2_frozen_recipe','frozen_recipe']:
        reference='original_frozen_recipe' if key=='frozen_recipe' else key
        if not same(receipt[key],freeze[reference]):raise ValueError('Prediction parent freeze mismatch')
    inp=json.loads(checked_artifact(receipt['input_receipt']).read_text())
    validate_input_lineage(inp,receipt['input_receipt'],parent,freeze,amendment)
    for key in ['catalog','base','protein_means','reaction_features']:
        if not same(inp['inputs'][key],receipt['inputs'][key]):raise ValueError('Receipt input mismatch: '+key)
    if not receipt.get('parent_scores_exact') or not receipt.get('zero_scores_exact') or not all(receipt['subset_parity'].values()):raise ValueError('Numerical replay failed')
    for item in receipt['parent_reference'].values():checked_artifact(item)
    with np.load(score_path) as f:values=f[score_key]
    catalog=json.loads(Path(catalog_path).read_text());shape=(len(catalog.get('reactions',catalog.get('query_ids'))),len(catalog['proteins']))
    if values.dtype!=np.float32 or values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid calibrated scores')
    return values,receipt

