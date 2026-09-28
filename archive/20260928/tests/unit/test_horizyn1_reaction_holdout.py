import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_horizyn1_reaction_holdout as preparation
from scripts import project_horizyn1_training_labels as labels
from scripts import horizyn1_circe_v2_reaction_holdout as launcher
from scripts.build_horizyn1_circe_v2_labels_v2 import signature as legacy_signature


def table(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as out:
        writer = csv.writer(out, delimiter="\t" if path.suffix == ".tsv" else ",")
        writer.writerow(fields)
        writer.writerows(rows)


@pytest.fixture
def graph(tmp_path):
    pytest.importorskip("rdkit")
    source, output = tmp_path / "source", tmp_path / "prepared"
    smiles = ["CCO>>CC=O", "CCN>>CC=N", "CC(=O)O>>CCO", "CC(C)O>>CC(C)=O",
              "CCC(=O)O>>CCCO", "c1ccccc1O>>c1ccccc1", "CCl>>CO", "CBr>>CCl",
              "C#N>>NC=O", "CCS>>CC=O", "N=C=O>>NC(=O)O", "CC(=O)N>>CC(=O)O"]
    reactions = [(f"r{i:02}", smi) for i, smi in enumerate(smiles)] + [("alias", "CC=O>>OCC")]
    table(source / "raw/raw_reactions.tsv", ["reaction_id", "reaction_smiles"], reactions)
    pairs = sorted([(rid, f"p{i % 3}") for i, (rid, _) in enumerate(reactions)] + [("r00", "p3"), ("r03", "p3"), ("r08", "p3")])
    table(source / "clustered/pairs.tsv", ["reaction_id", "protein_id"], pairs)
    table(source / "raw/raw_pairs.tsv", ["reaction_id", "protein_id"],
          [(r, "member" if p == "p3" else p) for r, p in pairs])
    (source / "clustered/proteins.fasta").write_text(
        ">p0\nACDE\n>p1\nACDF\n>p2\nACDG\n>p3\nACDH\n")
    (source / "clustered/clustered_manifest.json").write_text(json.dumps(
        {"min_seq_id": .8, "clustered_proteins": 4, "clustered_pairs": len(pairs)}))
    preparation.prepare(source, output, validation_fraction=.2, test_fraction=.2, panel_reactions=1, panel_proteins=1)
    return source, output, pairs


def test_fingerprint_is_paper_binary_ecfp6_not_change_fingerprint():
    pytest.importorskip("rdkit")
    _, key = preparation.describe_reaction("CCO.O>>CC=O.O")
    assert len(key) == 512  # 4096 bits, not the 2048-bit training composite.
    assert key == preparation.describe_reaction("O.CC=O>>O.OCC")[1]
    assert key != preparation.describe_reaction("CCO>>CC=O")[1]  # Spectators are NOT cancelled.
    assert preparation.describe_reaction("CCO.CCO>>CC=O")[1] == preparation.describe_reaction("CCO>>CC=O")[1]


def test_paper_standardization_and_invalid_chemistry():
    pytest.importorskip("rdkit")
    assert preparation.describe_reaction("CC(=O)[O-]>>CCO")[1] == preparation.describe_reaction("CC(=O)O>>CCO")[1]
    for invalid in ("bad>>CCO", "CCO", "CCO>O>CC=O", ">>CCO"):
        with pytest.raises(ValueError):
            preparation.describe_reaction(invalid)


def test_existing_clustered_graph_is_preserved_and_directions_do_not_leak(graph):
    source, output, original = graph
    assignments = json.loads((output / "reaction_split.json").read_text())
    assert assignments["alias"] == assignments["r00"]
    recovered = []
    for split in ("train", "validation", "test"):
        rows = list(preparation.csv_rows(output / f"{split}_pairs.csv"))
        assert rows
        expected = {(preparation.reaction_id(r, d), p) for r, p in original if assignments[r] == split for d in preparation.DIRECTIONS}
        assert {(r["reaction_id"], r["protein_id"]) for r in rows} == expected
        recovered.extend(rows)
    assert len(recovered) == 2 * len(original)
    assert any(r["protein_id"] == "p3" for r in recovered)  # member-transferred positives survive.
    assert not list(output.rglob("*.sqlite"))
    assert not list(output.rglob("*mmseqs*"))
    own = list(preparation.csv_rows(output / "train_own_raw_associations.csv"))
    assert all(assignments[r["reaction_id"]] == "train" and r["protein_id"] != "p3" for r in own)
    augmented = {r["reaction_id"]: r["reaction_smiles"] for r in preparation.csv_rows(output / "reactions.csv")}
    for rid in assignments:
        forward = augmented[preparation.reaction_id(rid, "forward")]
        assert augmented[preparation.reaction_id(rid, "reverse")] == ">>".join(reversed(forward.split(">>")))


def test_validation_panel_retains_all_gold_for_both_query_directions(graph):
    _, output, _ = graph
    queries = json.loads((output / "panels/validation_reaction_cold_query_ids.json").read_text())
    pairs = list(preparation.csv_rows(output / "validation_pairs.csv"))
    panel = {(r["reaction_id"], r["protein_id"]) for r in preparation.csv_rows(output / "panels/validation_reaction_cold_pairs.csv")}
    for direction, anchors in queries.items():
        column = "reaction_id" if direction == "reaction_to_enzyme" else "protein_id"
        assert {(r["reaction_id"], r["protein_id"]) for r in pairs if r[column] in anchors} <= panel
    candidates = set((output / "panels/validation_reaction_cold_protein_ids.txt").read_text().splitlines())
    assert {p for r, p in panel if r in queries["reaction_to_enzyme"]} <= candidates


def test_preparation_refuses_existing_or_source_output(graph):
    source, output, _ = graph
    before = (output / "preparation_manifest.json").read_bytes()
    with pytest.raises(ValueError, match="nonempty"):
        preparation.prepare(source, output)
    assert (output / "preparation_manifest.json").read_bytes() == before
    with pytest.raises(ValueError, match="non-nested"):
        preparation.prepare(source, source / "new")


def test_unpaired_inventory_entries_are_reported_and_never_split(graph, tmp_path):
    source, baseline, original = graph
    reactions = list(preparation.csv_rows(source / "raw/raw_reactions.tsv"))
    unpaired = [("unpaired_new_group", "CF>>CO"), ("unpaired_reverse_alias", "CC=O>>OCC")]
    table(source / "raw/raw_reactions.tsv", ["reaction_id", "reaction_smiles"],
          [(r["reaction_id"], r["reaction_smiles"]) for r in reactions] + unpaired)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    output = tmp_path / "with_unpaired"
    result = preparation.prepare(source, output, validation_fraction=.2, test_fraction=.2,
                                 panel_reactions=1, panel_proteins=1)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert result["source_pairs"] == len(original)
    assert sum(result["pairs_after_augmentation"].values()) == 2 * len(original)
    inventory = result["reaction_inventory"]
    assert inventory["source_reactions"] == len(reactions) + 2
    assert inventory["paired_reactions"] == len(reactions)
    assert inventory["excluded_unpaired_reactions"] == 2
    assert inventory["excluded_raw_pair_count"] == 0
    assert inventory["excluded_ids"] == sorted(r for r, _ in unpaired)
    report = list(preparation.csv_rows(output / inventory["excluded_table"]))
    assert [(r["reaction_id"], r["reaction_smiles"]) for r in report] == unpaired
    assert {r["reason"] for r in report} == {"no_observed_enzyme_reaction_pairs"}
    # Neither an unpaired new group nor an unpaired reverse alias changes the
    # supervised split, candidates, zero-gold query policy, pairs or panels.
    for path in baseline.rglob("*"):
        if path.is_file() and path.name not in ("preparation_manifest.json", "excluded_unpaired_reactions.csv"):
            assert (output / path.relative_to(baseline)).read_bytes() == path.read_bytes()
    # Cached annotations may still contain unused inventory reactions.
    cache_labels(source, output)
    lookup_path = source / "annotations/cofactor_v2/reaction_annotations.json"
    lookup = json.loads(lookup_path.read_text())
    lookup["reactions"][unpaired[0][0]] = next(iter(lookup["reactions"].values()))
    lookup_path.write_text(json.dumps(lookup))
    labels.project(source, output, tmp_path / "projected_with_unpaired")


def test_reaction_lost_during_clustering_is_not_silently_excluded(graph, tmp_path):
    source, _, original = graph
    kept = [(r, p) for r, p in original if r != "r00"]
    table(source / "clustered/pairs.tsv", ["reaction_id", "protein_id"], kept)
    manifest_path = source / "clustered/clustered_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["clustered_pairs"] = len(kept)
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "lost_reaction"
    with pytest.raises(ValueError, match="r00 has raw pairs but no clustered pairs"):
        preparation.prepare(source, output, validation_fraction=.2, test_fraction=.2)
    assert not (output / "preparation_manifest.json").exists()
    assert not (output / "excluded_unpaired_reactions.csv").exists()


@pytest.mark.parametrize("reactions,pairs,error", [
    ([{"reaction_id": "r"}, {"reaction_id": "r"}], [], "duplicate reaction ID"),
    ([{"reaction_id": ""}], [], "Empty or duplicate reaction ID"),
    ([{"reaction_id": "r"}], [{"reaction_id": "unknown"}], "Unknown clustered reaction ID"),
])
def test_paired_inventory_rejects_invalid_ids(reactions, pairs, error):
    with pytest.raises(ValueError, match=error):
        preparation.paired_reaction_inventory(reactions, pairs)


def test_paired_inventory_can_report_an_empty_graph():
    selected, excluded, count = preparation.paired_reaction_inventory(
        [{"reaction_id": "unpaired", "reaction_smiles": "CCO>>CC=O"}], [])
    assert selected == [] and count == 0
    assert excluded[0]["reaction_id"] == "unpaired"


def cache_labels(source, prepared):
    cache = source / "annotations/circe_v2_cofactor_v2"
    cache.mkdir(parents=True)
    ids = np.asarray(["p0", "p1", "p2", "p3"])
    native = np.zeros((4, 32), dtype=np.float32)
    native[:, labels.UNKNOWN_INDEX] = 1
    native[0, labels.UNKNOWN_INDEX] = 0
    native[0, 0] = 1
    np.savez_compressed(cache / "enzyme_biofp_targets.npz", ids=ids,
        cofactor_labels=np.asarray(labels.COFACTOR_LABELS), pair_scope=np.asarray("unsplit_inventory"),
        annotation_semantics=np.asarray(labels.ANNOTATION_SEMANTICS),
        cofactor_vocabulary_version=np.asarray(labels.COFACTOR_VOCABULARY_VERSION),
        cofactor_unknown_index=np.asarray(labels.UNKNOWN_INDEX, dtype=np.int64),
        native_cofactor_targets=native, native_cofactor_has_annotation=np.array([1, 0, 0, 0], dtype=bool),
        native_cofactor_has_unmapped=np.zeros(4, dtype=bool))
    table(cache / "enzyme_ec_labels.csv", ["protein_id", "ec_number", "known_depth"],
          [(f"p{i}", f"{i+1}.1.1.1", 4) for i in range(4)])
    table(cache / "candidate_eligibility.csv", ["protein_id", "ec_negative_candidate_eligible",
          "biological_negative_candidate_eligible", "reason", "member_count", "members_with_ec"],
          [(f"p{i}", 1, 1, "eligible", 1, 1) for i in range(4)])
    (cache / "enzyme_biofp_vocab.json").write_text("{}")
    paths = {"biofp": cache / "enzyme_biofp_targets.npz", "ec": cache / "enzyme_ec_labels.csv",
             "eligibility": cache / "candidate_eligibility.csv", "vocab": cache / "enzyme_biofp_vocab.json"}
    manifest = {"schema_version": labels.SCHEMA, "pair_scope": "unsplit_inventory", "representative_count": 4,
                "cofactor_unknown_index": labels.UNKNOWN_INDEX,
                "families": {"mechanism": list(labels.MECHANISM_LABELS), "cofactor": list(labels.COFACTOR_LABELS)},
                "output_signatures": {k: legacy_signature(p) for k, p in paths.items()}}
    (cache / "label_manifest.json").write_text(json.dumps(manifest))
    split = json.loads((prepared / "reaction_split.json").read_text())
    lookup = {"schema_version": labels.REACTION_SCHEMA, "families": manifest["families"],
              "label_semantics": "positive_only_unknown_not_negative",
              "cofactor_vocabulary_version": labels.COFACTOR_VOCABULARY_VERSION,
              "reactions": {r: {"mechanism": [labels.MECHANISM_LABELS[0 if s == "train" else 7]],
                  "cofactor": ["PLP" if s == "train" else "SAM"],
                  "evidence_type": "reaction_associated_descriptor", "chemistry_match": "exact"} for r, s in split.items()}}
    path = source / "annotations/cofactor_v2/reaction_annotations.json"
    path.parent.mkdir()
    path.write_text(json.dumps(lookup))
    return cache


def test_projected_labels_reuse_native_but_exclude_heldout_and_member_transfers(graph, tmp_path):
    source, prepared, _ = graph
    cache = cache_labels(source, prepared)
    original = {p.name: p.read_bytes() for p in cache.iterdir()}
    output = tmp_path / "projected"
    labels.project(source, prepared, output)
    with np.load(output / "enzyme_biofp_targets.npz", allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
        assert str(payload["pair_scope"]) == "train"
        assert not payload["mechanism_targets"][:, 7].any()
        assert not payload["reaction_cofactor_targets"][:, labels.COFACTOR_LABELS.index("SAM")].any()
        assert payload["native_cofactor_targets"][0, 0] == 1
        assert not payload["mechanism_targets"][3].any()  # no weak member-to-representative transfer
        assert payload["reaction_cofactor_targets"][3, labels.UNKNOWN_INDEX] == 1
        for family in ("cofactor", "native_cofactor", "reaction_cofactor"):
            assert not payload[f"{family}_confidence"][:, labels.UNKNOWN_INDEX].any()
    assert all(p.read_bytes() == original[p.name] for p in cache.iterdir())
    # The existing index builder validates our complete annotation contract.
    from scripts.build_annotation_negative_pools import _validate_v2_arrays, _v2_metadata
    _validate_v2_arrays(arrays, output / "enzyme_biofp_targets.npz")
    with np.load(output / "enzyme_biofp_targets.npz") as payload:
        assert _v2_metadata(payload, output) == "train"
    from scripts.build_indexed_training_pairs import build
    result = build(train_pairs=prepared / "train_pairs.csv", train_reactions=prepared / "train_rxns.csv",
                   ec_labels=output / "enzyme_ec_labels.csv", biofp_targets=output / "enzyme_biofp_targets.npz",
                   candidate_eligibility=output / "candidate_eligibility.csv", output_dir=tmp_path / "index")
    assert result["num_pairs"] == len(list(preparation.csv_rows(prepared / "train_pairs.csv")))


def test_projection_fails_on_heldout_association(graph, tmp_path):
    source, prepared, _ = graph
    cache_labels(source, prepared)
    split = json.loads((prepared / "reaction_split.json").read_text())
    heldout = next(r for r, s in split.items() if s == "test")
    with (prepared / "train_own_raw_associations.csv").open("a") as out:
        out.write(f"{heldout},p0\n")
    with pytest.raises(ValueError, match="Held-out reaction leaked"):
        labels.project(source, prepared, tmp_path / "labels")


def test_projection_does_not_truncate_unknown_long_protein_ids(graph, tmp_path):
    source, prepared, _ = graph
    cache_labels(source, prepared)
    split = json.loads((prepared / "reaction_split.json").read_text())
    train = next(r for r, s in split.items() if s == "train")
    with (prepared / "train_own_raw_associations.csv").open("a") as out:
        out.write(f"{train},p0_extra\n")
    with pytest.raises(ValueError, match="unknown representative"):
        labels.project(source, prepared, tmp_path / "labels")


def test_separate_launcher_keeps_circe_settings_and_changes_only_data_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_ROOT", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.setenv("GPU_COUNT", "4")
    assert launcher.parse_args(["--profile", "h200"]).run_root.endswith("reaction_holdout_h200")
    args = launcher.parse_args(["--profile", "h200", "--run-root", str(tmp_path / "run")])
    runner = launcher.Pipeline(args)
    output = runner.run / "config.yaml"
    runner.config(output)
    import yaml
    config = yaml.safe_load(output.read_text())
    assert config["data"]["typed_negative_positive_fraction"] == .85
    assert config["training"]["loss"]["biofp_aux_weight"] == 0
    assert config["training"]["precision"] == "bf16-mixed"
    assert "reaction_cold" in config["data"]["validation_pairs_path"]
    assert "both_cold" not in output.read_text()
    assert config["data"]["direction_augmentation"] == "materialized_forward_and_reverse"
    first = output.read_bytes()
    runner.config(output)
    assert output.read_bytes() == first


def test_new_preparation_command_does_not_launch_mmseqs(tmp_path, monkeypatch):
    args = launcher.parse_args(["--run-root", str(tmp_path / "run")])
    runner = launcher.Pipeline(args)
    monkeypatch.setattr(runner, "require_stage", lambda *a: None)
    commands = []
    monkeypatch.setattr(runner, "command", lambda command, *a: commands.append(list(map(str, command))))
    monkeypatch.setattr(runner, "step", lambda name, inputs, outputs, action, *args: action())
    runner.prepare()
    assert len(commands) == 1
    assert commands[0][1].endswith("prepare_horizyn1_reaction_holdout.py")
    assert not any("mmseqs" in word or "--min-seq-id" == word for word in commands[0])


def test_launcher_refuses_an_old_run_without_protocol_binding(tmp_path):
    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    (run / "state/prepare.json").write_text("{}")
    with pytest.raises(SystemExit, match="NEW RUN_ROOT"):
        launcher.main(["prepare", "--run-root", str(run)])
    assert not (run / "pipeline.pid").exists()
