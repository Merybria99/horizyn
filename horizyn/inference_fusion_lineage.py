"""Authenticate the sole permitted checkpoint change in a fusion calibration."""
import hashlib
import json
import math
from pathlib import Path

import torch


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_refiner_base(training_sha, inference_sha, receipt_path=None, checkpoint=None):
    if training_sha == inference_sha:
        return None
    if receipt_path is None or checkpoint is None:
        raise ValueError('Refiner training features belong to a different base checkpoint; explicit fusion lineage required')
    receipt = json.loads(Path(receipt_path).read_text())
    original = Path(receipt['source_checkpoint'])
    if (receipt['source_checkpoint_sha256'] != training_sha or digest(original) != training_sha
            or receipt['calibrated_checkpoint_sha256'] != inference_sha or digest(checkpoint) != inference_sha):
        raise ValueError('Fusion calibration checkpoint hashes do not match training/inference inputs')
    before = torch.load(original, map_location='cpu', weights_only=False)['state_dict']
    after = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
    key = receipt['parameter']
    allowed = [k for k in before if k.endswith('multiview_encoder.raw_residual_scale')]
    if allowed != [key] or before.keys() != after.keys():
        raise ValueError('Fusion calibration changed the model schema')
    if any(not torch.equal(before[k], after[k]) for k in before if k != key):
        raise ValueError('Fusion calibration changed parameters outside the single fusion scalar')
    if before[key].numel() != 1 or after[key].numel() != 1 or before[key].dtype != after[key].dtype:
        raise ValueError('Fusion scalar shape/dtype changed')
    old, new = float(.5 * before[key].sigmoid()), float(.5 * after[key].sigmoid())
    if not (0 < old < .5 and 0 < new < .5 and math.isfinite(new)):
        raise ValueError('Fusion strength is outside its positive bounded domain')
    if not math.isclose(new, old * receipt['requested_multiplier'], rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError('Fusion checkpoint does not implement the declared multiplier')
    return dict(receipt_sha256=digest(receipt_path), training_base_checkpoint_sha256=training_sha,
                inference_base_checkpoint_sha256=inference_sha, changed_parameter=key,
                original_scale=old, inference_scale=new, all_other_state_tensors_identical=True,
                phase2_retrained=False)
