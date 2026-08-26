import json

import h5py
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("rdkit")

from horizyn.capability.reaction_directional_features import (
    CENTER_VOCAB_CAPACITY,
    build_directional_vector_splits,
    build_reaction_only_rhea_map,
    participant_multiset,
    symmetric_center_label,
)


def test_participant_multiset_collapses_self_reaction_and_preserves_counts():
    assert participant_multiset("CCO.CCO>>CCO.CCO") == participant_multiset("CCO.CCO")
    assert participant_multiset("CCO.CCO") != participant_multiset("CCO")


def test_reaction_only_rhea_map_resolves_unique_and_rejects_ambiguous(tmp_path):
    reactions = tmp_path / "reactions.csv"
    rhea = tmp_path / "rhea.tsv"
    mapping = tmp_path / "mapping.csv"
    directional = tmp_path / "directional.csv"
    report_path = tmp_path / "report.json"
    pd.DataFrame(
        [
            ("unique", "CCO.CC=O"),
            ("ambiguous", "CCN.CC=N"),
            ("missing", "CCC.CC=C"),
        ],
        columns=["reaction_id", "reaction_smiles"],
    ).to_csv(reactions, index=False)
    rhea.write_text(
        "Rhea ID\tsubstrate\tproduct\n"
        "RHEA:1\tCCO\tCC=O\n"
        "RHEA:2\tCCN\tCC=N\n"
        "RHEA:3\tCC=N\tCCN\n",
        encoding="utf-8",
    )

    report = build_reaction_only_rhea_map(
        reactions_path=reactions,
        rhea_molecules_path=rhea,
        output_mapping_path=mapping,
        output_directional_reactions_path=directional,
        output_report_path=report_path,
    )

    mapped = pd.read_csv(mapping, dtype={"reaction_id": str}).set_index("reaction_id")
    assert mapped.loc["unique", "match_status"] == "exact_unique"
    assert mapped.loc["ambiguous", "match_status"] == "exact_ambiguous"
    assert mapped.loc["missing", "match_status"] == "unmatched"
    assert pd.read_csv(directional)["reaction_id"].tolist() == ["unique"]
    assert report["num_resolved"] == 1
    assert "protein" in report["leakage_policy"]
    assert json.loads(report_path.read_text(encoding="utf-8"))["num_resolved"] == 1


def test_center_labels_remove_formed_broken_direction():
    assert symmetric_center_label("bond_formed_C_O") == "bond_changed_C_O"
    assert symmetric_center_label("bond_broken_C_O") == "bond_changed_C_O"
    assert symmetric_center_label("charge_change_N") == "charge_change_N"


def _write_fixed_h5(path, ids, dim):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(ids, dtype="S"))
        handle.create_dataset(
            "vectors",
            data=np.arange(len(ids) * dim, dtype=np.float32).reshape(len(ids), dim),
        )
        handle.attrs["embedding_dim"] = dim


def _write_ragged_h5(path, ids, dim):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ids", data=np.asarray(ids, dtype="S"))
        vectors = np.arange(2 * len(ids) * dim, dtype=np.float32).reshape(2 * len(ids), dim)
        handle.create_dataset("reactant_vectors", data=vectors[::2])
        handle.create_dataset("product_vectors", data=vectors[1::2])
        handle.create_dataset("reactant_offsets", data=np.arange(len(ids) + 1))
        handle.create_dataset("product_offsets", data=np.arange(len(ids) + 1))
        handle.attrs["embedding_dim"] = dim


def test_directional_bundle_has_fixed_dimensions_and_masked_fallback(tmp_path):
    split_inputs = {}
    for split in ("train", "validation", "test"):
        root = tmp_path / split
        root.mkdir()
        reactions = root / "reactions.csv"
        mapping = root / "mapping.csv"
        t5 = root / "t5.h5"
        unimol = root / "unimol.h5"
        chiro = root / "chiro.h5"
        centers = root / "centers.parquet"
        pd.DataFrame(
            [("r1", "CCO.CC=O"), ("r2", "CCC.CC=C")],
            columns=["reaction_id", "reaction_smiles"],
        ).to_csv(reactions, index=False)
        pd.DataFrame(
            [
                ("r1", "RHEA:1", "exact_unique", 1.0, 1),
                ("r2", "", "unmatched", 0.0, 0),
            ],
            columns=[
                "reaction_id",
                "rhea_id",
                "match_status",
                "match_confidence",
                "candidate_count",
            ],
        ).to_csv(mapping, index=False)
        ids = ["r1_f", "r1_r"]
        _write_fixed_h5(t5, ids, 2)
        _write_ragged_h5(unimol, ids, 3)
        _write_ragged_h5(chiro, ids, 4)
        pd.DataFrame(
            [
                {
                    "reaction_id": "r1",
                    "reaction_center_raw_labels": ["bond_formed_C_O"],
                    "reaction_center_coarse_labels": ["bond_formation"],
                }
            ]
        ).to_parquet(centers, index=False)
        split_inputs[split] = {
            "reactions": reactions,
            "mapping": mapping,
            "reaction_t5": t5,
            "unimol2": unimol,
            "chiro": chiro,
            "center_features": centers,
        }

    report = build_directional_vector_splits(
        split_inputs=split_inputs,
        out_dir=tmp_path / "out",
    )

    f5 = np.load(tmp_path / "out/train_reaction_directional_f5.npz", allow_pickle=True)
    f6 = np.load(tmp_path / "out/train_reaction_directional_f6.npz", allow_pickle=True)
    assert f5["vectors"].shape == (2, 2 + 2 * 3 + 2 * 4 + 5)
    assert f6["vectors"].shape == (2, f5["vectors"].shape[1] + CENTER_VOCAB_CAPACITY + 1)
    assert f5["mask"].tolist() == [True, False]
    assert np.count_nonzero(f5["vectors"][1]) == 0
    assert report["splits"]["train"]["num_resolved"] == 1
