"""Narrow compatibility tests; no model, assay or score fixtures required."""
import importlib.util
from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import generalization_calibration_evaluation_guard as guard

@pytest.fixture
def lineage(monkeypatch):
    monkeypatch.setattr(guard,'checked_artifact',lambda x:Path(x['path']))
    record={'path':'/receipt.json','sha256':'receipt'}
    checkpoint={'path':'/checkpoint.ckpt','sha256':'checkpoint'}
    inp=dict(schema='generalization_feature_bundle_receipt_v1',labels_used=False,checkpoint=checkpoint,source_receipts=[])
    parent={'base_checkpoint':checkpoint}
    freeze={'original_frozen_recipe':{'sha256':'original'}}
    amendment={'approved_missing_freeze_receipts':[record]}
    return inp,record,parent,freeze,amendment

def test_exact_approved_missing_freeze_with_same_checkpoint(lineage):
    guard.validate_input_lineage(*lineage)

def test_existing_freeze_still_requires_exact_identity(lineage):
    lineage[0]['freeze_sha256']='wrong'
    with pytest.raises(ValueError,match='freeze mismatch'):guard.validate_input_lineage(*lineage)

def test_unlisted_missing_field_rejected(lineage):
    lineage[-1]['approved_missing_freeze_receipts']=[]
    with pytest.raises(ValueError,match='Unapproved'):guard.validate_input_lineage(*lineage)

def test_wrong_checkpoint_rejected(lineage):
    lineage[0]['checkpoint']={'path':'/checkpoint.ckpt','sha256':'wrong'}
    with pytest.raises(ValueError,match='checkpoint'):guard.validate_input_lineage(*lineage)

def test_labels_used_rejected(lineage):
    lineage[0]['labels_used']=True
    with pytest.raises(ValueError,match='label-use'):guard.validate_input_lineage(*lineage)
