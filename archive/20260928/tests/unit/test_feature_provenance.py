import h5py
import numpy as np
import pandas as pd
import pytest

from horizyn.chemistry.feature_provenance import (
    stamp_h5_reaction_feature_provenance,
    validate_h5_reaction_feature_provenance,
)


def test_h5_provenance_round_trip_and_rejects_wrong_source(tmp_path):
    source = tmp_path / "reactions.csv"
    other = tmp_path / "other.csv"
    artifact = tmp_path / "features.h5"
    pd.DataFrame(
        {"reaction_id": ["r1", "r2"], "reaction_smiles": ["CC>>CC", "CO>>CO"]}
    ).to_csv(source, index=False)
    pd.DataFrame(
        {"reaction_id": ["r1"], "reaction_smiles": ["CC>>CC"]}
    ).to_csv(other, index=False)
    with h5py.File(artifact, "w") as handle:
        handle.create_dataset(
            "ids",
            data=np.asarray(["r1_f", "r1_r", "r2_f", "r2_r"], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
    stamp_h5_reaction_feature_provenance(
        artifact,
        source,
        extractor_name="test",
        extractor_version="v1",
    )
    report = validate_h5_reaction_feature_provenance(artifact, source)
    assert report["smiles_mode"] == "canonical_isomeric"
    with pytest.raises(ValueError, match="was not extracted"):
        validate_h5_reaction_feature_provenance(artifact, other)


def test_unstamped_h5_is_rejected(tmp_path):
    source = tmp_path / "reactions.csv"
    artifact = tmp_path / "features.h5"
    pd.DataFrame({"reaction_id": ["r1"], "reaction_smiles": ["CC"]}).to_csv(
        source, index=False
    )
    with h5py.File(artifact, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(["r1"], dtype="S2"))
    with pytest.raises(ValueError, match="lacks required"):
        validate_h5_reaction_feature_provenance(artifact, source)
