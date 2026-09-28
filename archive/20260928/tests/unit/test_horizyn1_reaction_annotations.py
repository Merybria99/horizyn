import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/build_horizyn1_reaction_annotations.py"
SPEC = importlib.util.spec_from_file_location("horizyn1_reaction_annotations", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def bridge(reaction_id="rxn_a", chemistry="C>>CO", sources="Rh_100|RHEA:101"):
    return {"reaction_id": reaction_id, "reaction_smiles": chemistry, "source_reaction_ids": sources}


def feature(reaction_id="rxn_a", chemistry="C>>CO", mechanism=None, cofactor=None):
    return {
        "reaction_id": reaction_id, "raw_reaction_smiles": chemistry,
        "reaction_type_labels": ["oxidoreduction"] if mechanism is None else mechanism,
        "core_cofactor_labels": ["NAD"] if cofactor is None else cofactor,
    }


def test_rhea_exact_labels_and_order():
    rows, report = module.build_lookup({"Rh_100": "C>>CO"}, [bridge()], [feature()])
    assert rows["Rh_100"]["mechanism"] == ["redox_carbonyl_interconversion"]
    assert rows["Rh_100"]["cofactor"] == ["NAD_NADP"]
    assert rows["Rh_100"]["source_reaction_id"] == "rxn_a"
    assert rows["Rh_100"]["chemistry_match"] == "exact"
    assert rows["Rh_100"]["evidence_type"] == "reaction_associated_descriptor"
    assert report["counts"] == {"matched": 1, "unknown": 0, "ambiguous": 0, "chemistry_mismatch": 0}


def test_rhea_colon_source_matches_and_no_partial_token_matches():
    rows, report = module.build_lookup(
        {"Rh_101": "C>>CO", "Rh_10": "C>>CO"}, [bridge()], [feature()]
    )
    assert set(rows) == {"Rh_101"}
    assert report["unknown_reasons"]["no_source_id_match"] == ["Rh_10"]


def test_same_chemistry_conflicting_descriptor_is_ambiguous():
    rows, report = module.build_lookup(
        {"Rh_100": "C>>CO"}, [bridge(), bridge("rxn_b")],
        [feature(), feature("rxn_b", cofactor=["FAD"])],
    )
    assert not rows
    assert report["reaction_ids_by_status"]["ambiguous"] == ["Rh_100"]


def test_duplicate_feature_id_conflict_is_also_ambiguous():
    rows, report = module.build_lookup(
        {"Rh_100": "C>>CO"}, [bridge()], [feature(), feature(mechanism=["hydrolysis"])],
    )
    assert not rows and report["counts"]["ambiguous"] == 1


def test_same_descriptor_has_stable_source_id_and_preserves_all_ids():
    rows, _ = module.build_lookup(
        {"Rh_100": "C>>CO"}, [bridge("rxn_b"), bridge()], [feature("rxn_b"), feature()],
    )
    assert rows["Rh_100"]["source_reaction_id"] == "rxn_a"
    assert rows["Rh_100"]["source_reaction_ids"] == ["rxn_a", "rxn_b"]


def test_chemistry_disambiguates_source_id():
    rows, _ = module.build_lookup(
        {"Rh_100": "C>>CO"}, [bridge(), bridge("rxn_b", "CO>>C")],
        [feature(), feature("rxn_b", "CO>>C", cofactor=["FAD"])],
    )
    assert rows["Rh_100"]["source_reaction_ids"] == ["rxn_a"]


@pytest.mark.parametrize("bridge_chemistry,feature_chemistry", [("CO>>C", "C>>CO"), ("C>>CO", "CO>>C")])
def test_both_bridge_and_feature_chemistry_must_match(bridge_chemistry, feature_chemistry):
    rows, report = module.build_lookup(
        {"Rh_100": "C>>CO"}, [bridge(chemistry=bridge_chemistry)], [feature(chemistry=feature_chemistry)],
    )
    assert not rows and report["counts"]["chemistry_mismatch"] == 1


def test_raw_feature_chemistry_cannot_be_overridden_by_canonical_field():
    row = feature(chemistry="CO>>C")
    row["canonical_reaction_smiles"] = "C>>CO"
    rows, report = module.build_lookup({"Rh_100": "C>>CO"}, [bridge()], [row])
    assert not rows and report["counts"]["chemistry_mismatch"] == 1


def test_canonical_field_allowed_only_when_raw_is_missing():
    row = feature()
    row["raw_reaction_smiles"] = float("nan")
    row["canonical_reaction_smiles"] = "C>>CO"
    rows, _ = module.build_lookup({"Rh_100": "C>>CO"}, [bridge()], [row])
    assert "Rh_100" in rows


def test_enzymemap_requires_hash_id_and_exact_chemistry():
    rows, report = module.build_lookup(
        {"Em_a": "C>>CO", "Em_b": "C>>CO", "Em_c": "CO>>C"},
        [bridge(), bridge("rxn_c")], [feature(), feature("rxn_c")],
    )
    assert set(rows) == {"Em_a"}
    assert report["counts"]["unknown"] == 1
    assert report["counts"]["chemistry_mismatch"] == 1


def test_missing_features_and_unsupported_labels_are_unknown():
    rows, report = module.build_lookup(
        {"Rh_100": "C>>CO", "Rh_200": "C>>CO"},
        [bridge(), bridge("rxn_b", sources="Rh_200")],
        [feature(mechanism=["something_else"], cofactor=[])],
    )
    assert not rows and report["counts"]["unknown"] == 2
    assert report["unknown_reasons"]["no_supported_positive_descriptor"] == ["Rh_100"]
    assert report["unknown_reasons"]["no_cached_feature_row"] == ["Rh_200"]


def test_nan_arrays_strings_and_empty_family_are_sparse_unknown():
    row = feature(mechanism=np.nan, cofactor=np.array(["NAD", "PLP"]))
    row["cofactor_labels"] = "['FAD', 'NADP']"
    row["substrate_product_transition_labels"] = np.nan
    rows, report = module.build_lookup({"Rh_100": "C>>CO"}, [bridge()], [row])
    assert rows["Rh_100"]["mechanism"] == []
    assert rows["Rh_100"]["cofactor"] == ["NAD_NADP", "FAD_FMN", "PLP"]
    assert report["family_positive_coverage"] == {"mechanism": 0, "cofactor": 1}


def test_vocab_is_exact_current_minimal_contract():
    assert module.FAMILIES["mechanism"] == list(module.MECHANISM_LABELS)
    assert module.FAMILIES["cofactor"] == list(module.COFACTOR_LABELS)
    assert len(module.FAMILIES["mechanism"]) == 8
    assert len(module.FAMILIES["cofactor"]) == 10


def test_conflicting_bridge_chemistry_rejected():
    with pytest.raises(ValueError, match="Conflicting bridge"):
        module.build_lookup({}, [bridge(), bridge(chemistry="CO>>C")], [])


@pytest.fixture
def files(tmp_path):
    raw = tmp_path / "raw.tsv"
    raw.write_text("reaction_id\treaction_smiles\nRh_100\tC>>CO\n")
    cache = tmp_path / "cache.parquet"
    pd.DataFrame([feature()]).to_parquet(cache, index=False)
    mapping = tmp_path / "bridge.csv"
    pd.DataFrame([bridge()]).to_csv(mapping, index=False)
    return {"raw_reactions": raw, "cached_features": cache, "cached_reactions": mapping,
            "output": tmp_path / "annotations.json", "manifest": tmp_path / "manifest.json"}


def test_real_parquet_end_to_end_hashes_resume_and_manifest(files):
    report = module.run_build(**files)
    output = json.loads(files["output"].read_text())
    assert output["label_semantics"] == "positive_only_unknown_not_negative"
    assert set(output["reactions"]) == {"Rh_100"}
    assert "ec" not in output["reactions"]["Rh_100"]
    assert report["output"]["sha256"] == module._sha256(files["output"])
    before = {key: path.stat().st_mtime_ns for key, path in files.items()}
    assert module.run_build(**files)["skipped"]
    assert before == {key: path.stat().st_mtime_ns for key, path in files.items()}


def test_changed_inputs_are_not_silently_reused(files):
    module.run_build(**files)
    files["raw_reactions"].write_text("reaction_id\treaction_smiles\nRh_100\tCO>>C\n")
    with pytest.raises(FileExistsError):
        module.run_build(**files)
    report = module.run_build(**files, force=True)
    assert report["coverage"]["counts"]["chemistry_mismatch"] == 1


def test_tampered_output_is_not_reused(files):
    module.run_build(**files)
    files["output"].write_text("{}\n")
    with pytest.raises(FileExistsError):
        module.run_build(**files)


def test_malformed_completed_manifest_is_not_reused(files):
    module.run_build(**files)
    report = json.loads(files["manifest"].read_text())
    report["output"] = ["invalid metadata"]
    files["manifest"].write_text(json.dumps(report))
    with pytest.raises(FileExistsError):
        module.run_build(**files)


@pytest.mark.parametrize("target", ["output", "manifest"])
@pytest.mark.parametrize("link_kind", ["same_path", "symlink", "hardlink"])
def test_output_input_aliases_are_rejected(files, target, link_kind):
    if link_kind == "same_path":
        files[target] = files["raw_reactions"]
    elif link_kind == "symlink":
        files[target].symlink_to(files["raw_reactions"])
    else:
        files[target].hardlink_to(files["raw_reactions"])
    with pytest.raises(ValueError, match="aliases"):
        module.run_build(**files, force=True)


def test_output_manifest_alias_rejected(files):
    files["manifest"] = files["output"]
    with pytest.raises(ValueError, match="aliases"):
        module.run_build(**files)


def test_duplicate_raw_ids_rejected_without_output(files):
    with files["raw_reactions"].open("a") as handle:
        handle.write("Rh_100\tC>>CO\n")
    with pytest.raises(ValueError, match="duplicate"):
        module.run_build(**files)
    assert not files["output"].exists() and not files["manifest"].exists()


def test_input_mutation_during_loading_prevents_completion(files, monkeypatch):
    original = module.pd.read_parquet

    def mutate(path):
        frame = original(path)
        with files["raw_reactions"].open("a") as handle:
            handle.write("Rh_101\tC>>CO\n")
        return frame

    monkeypatch.setattr(module.pd, "read_parquet", mutate)
    with pytest.raises(RuntimeError, match="input changed"):
        module.run_build(**files)
    assert not files["output"].exists() and not files["manifest"].exists()
