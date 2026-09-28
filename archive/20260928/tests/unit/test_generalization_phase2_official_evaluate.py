import json
import numpy as np
import pytest

from scripts.generalization_export import atomic_json, digest
from scripts.generalization_phase2_official_evaluate import checked_scores


def fixture_scores(tmp_path, original_sha="old", new_sha="new"):
    catalog=tmp_path/"catalog.json"
    atomic_json(catalog,dict(reactions=["R"],proteins=["P0","P1"]))
    np.savez(tmp_path/"scores.npz",selected=np.array([[.7,.2]],np.float32))
    bundle=tmp_path/"bundle.json"
    atomic_json(bundle,dict(phase2_frozen_recipe=dict(sha256=new_sha),frozen_recipe=dict(sha256=original_sha)))
    atomic_json(tmp_path/"complete.json",dict(output_sha256=digest(tmp_path/"scores.npz"),labels_used=False,
        bundle=dict(path=str(bundle),sha256=digest(bundle)),inputs=dict(catalog=dict(sha256=digest(catalog)))))
    return dict(path="scores.npz",score_key="selected"),catalog


def test_phase2_scores_bind_new_recipe_and_original_feature_lineage(tmp_path):
    entry,catalog=fixture_scores(tmp_path)
    result=checked_scores(entry,tmp_path,catalog,(1,2),"new","old",{})
    assert np.allclose(result,[[.7,.2]])
    with pytest.raises(ValueError,match="another phase2 recipe"):
        checked_scores(entry,tmp_path,catalog,(1,2),"different_new","old",{})
    with pytest.raises(ValueError,match="original feature-freeze lineage"):
        checked_scores(entry,tmp_path,catalog,(1,2),"new","different_old",{})


def test_phase2_rejects_unreceipted_scores_even_with_explicit_hash(tmp_path):
    entry,catalog=fixture_scores(tmp_path)
    (tmp_path/"complete.json").unlink()
    entry["sha256"]=digest(tmp_path/"scores.npz")
    entry["catalog_sha256"]=digest(catalog)
    with pytest.raises(ValueError,match="native prediction receipt"):
        checked_scores(entry,tmp_path,catalog,(1,2),"new","old",{})
