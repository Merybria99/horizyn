import json
import math

import pytest
import torch

from horizyn.inference_fusion_lineage import digest, verify_refiner_base


def fixtures(tmp_path):
    key = 'model.multiview_encoder.raw_residual_scale'
    before = {key: torch.tensor(math.log(.2/.8)), 'weight': torch.eye(3)}
    after = {k: v.clone() for k, v in before.items()}
    after[key] = torch.tensor(math.log(.6/.4))
    source, target, receipt = [tmp_path / name for name in ('source.pt', 'target.pt', 'receipt.json')]
    torch.save(dict(state_dict=before), source)
    torch.save(dict(state_dict=after), target)
    record = dict(source_checkpoint=str(source), source_checkpoint_sha256=digest(source),
                  calibrated_checkpoint_sha256=digest(target), parameter=key, requested_multiplier=3.)
    receipt.write_text(json.dumps(record))
    return source, target, receipt, record


def test_only_declared_fusion_scalar_can_change(tmp_path):
    source, target, receipt, _ = fixtures(tmp_path)
    result = verify_refiner_base(digest(source), digest(target), receipt, target)
    assert result['all_other_state_tensors_identical']
    assert not result['phase2_retrained']


def test_unrelated_weight_change_is_rejected_even_with_updated_receipt(tmp_path):
    source, target, receipt, record = fixtures(tmp_path)
    state = torch.load(target, weights_only=False)
    state['state_dict']['weight'][0, 0] = 2.
    torch.save(state, target)
    record['calibrated_checkpoint_sha256'] = digest(target)
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='outside the single fusion scalar'):
        verify_refiner_base(digest(source), digest(target), receipt, target)


def test_mismatch_without_explicit_lineage_is_rejected(tmp_path):
    source, target, _, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match='explicit fusion lineage required'):
        verify_refiner_base(digest(source), digest(target))
    assert verify_refiner_base(digest(source), digest(source)) is None


def test_stale_inference_digest_is_rejected(tmp_path):
    source, target, receipt, _ = fixtures(tmp_path)
    with pytest.raises(ValueError, match='hashes'):
        verify_refiner_base(digest(source), 'wrong', receipt, target)
