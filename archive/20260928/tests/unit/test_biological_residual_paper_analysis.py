"""Regression tests for harmonization, audit joins and resumable evaluation."""

import json
import subprocess

import numpy as np
import pytest

from scripts import biological_residual_paper_analysis as analysis
from scripts import run_biological_residual_paper_analysis as runner


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "unknown"),
        ("", "unknown"),
        (np.nan, "unknown"),
        ("unknown", "unknown"),
        ("1.2.-.-", "partial_EC_only"),
        ("1.2.-.-|2.3.4.5", "complete_EC4_present"),
    ],
)
def test_unknown_annotation_is_not_absence(value, expected):
    assert analysis.ec_coverage(value) == expected


@pytest.mark.parametrize(
    "identity,qcov,tcov,expected",
    [
        (0.2, 0.9, 0.9, "<30%"),
        (0.3, 0.9, 0.9, "30-50%"),
        (0.5, 0.8, 0.8, "50-80%"),
        (0.8, 1, 1, ">=80%"),
        (0.95, 0.5, 1, "coverage<80%"),
        (0.95, 1, 0.5, "coverage<80%"),
    ],
)
def test_homology_bins_require_both_coverages(identity, qcov, tcov, expected):
    assert (
        analysis.sequence_bin(
            dict(local_sequence_identity=identity, query_coverage=qcov, target_coverage=tcov)
        )
        == expected
    )
    assert analysis.sequence_bin(None) == "no_reported_hit"


def test_invalid_similarities_and_identifier_normalization():
    for bad in (-0.1, 1.1, np.nan):
        with pytest.raises(ValueError):
            analysis.similarity_bin(bad)
    assert analysis.similarity_bin(None) == "unknown"
    assert analysis.similarity_bin(1) == "1.0 (fingerprint-equal)"
    assert {analysis.protein_key(x) for x in ("uprot_abc", "prot_abc", "nr90_abc", "abc")} == {
        "abc"
    }


def alignment_fixture(tmp_path):
    directory = tmp_path / "mmseqs/time"
    directory.mkdir(parents=True)
    (directory / "train.fasta").write_text(">train\nAA\nAA\n")
    (directory / "test.fasta").write_text(">test\nAAGG\n>nohit\nKKKK\n")
    (directory / "test_to_train.tsv").write_text("test\ttrain\t0.5\t4\t1\t1\t1e-9\t50\n")
    analysis.write_csv(
        tmp_path / "protein_nearest_train_mmseqs.csv",
        [
            dict(
                protocol="time",
                protein_id="test",
                nearest_train_protein_id="train",
                local_sequence_identity=0.5,
                alignment_length=4,
                query_coverage=1,
                target_coverage=1,
                evalue=1e-9,
                bits=50,
            )
        ],
    )
    return (
        directory,
        analysis.fasta_hashes(directory / "train.fasta"),
        analysis.fasta_hashes(directory / "test.fasta"),
    )


def test_sequence_audit_checks_entire_reference_and_raw_output(tmp_path):
    directory, train, test = alignment_fixture(tmp_path)
    hits = analysis.verified_sequence_hits(tmp_path, "time", train, test)
    assert set(hits) == {"test"}
    with pytest.raises(ValueError, match="FASTA differs"):
        analysis.verified_sequence_hits(tmp_path, "time", {"wrong": "hash"}, test)
    (directory / "test_to_train.tsv").write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        analysis.verified_sequence_hits(tmp_path, "time", train, test)


def test_missing_alignment_is_unknown_not_no_hit(tmp_path):
    assert analysis.verified_sequence_hits(tmp_path, "time", {}, {}) is None


def test_chemistry_participant_union_and_bad_reference():
    pytest.importorskip("rdkit")
    result = analysis.nearest_chemistry({"q": "CCO.O>>CC=O.O"}, {"r": "O.CC=O.CCO"})
    assert result["q"] == 1
    assert analysis.nearest_chemistry({"q": "not-smiles"}, {"r": "CC"})["q"] is None
    with pytest.raises(ValueError, match="unparsable"):
        analysis.nearest_chemistry({"q": "CC"}, {"r": "not-smiles"})


def test_metadata_parquet_join_and_source_exposure(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pytest.importorskip("rdkit")
    audit, output = tmp_path / "audit", tmp_path / "out"
    audit.mkdir()
    output.mkdir()
    annotations = []
    for subset, p, r, sequence, smiles in (
        ("train", "prot_a", "r1", "AAAA", "CC"),
        ("validation", "prot_b", "r2", "BBBB", "CO"),
        ("test", "prot_c", "r3", "CCCC", "CCO"),
    ):
        analysis.write_csv(
            tmp_path / "data/revised_protocols/reactzyme_paper/time" / f"{subset}_pairs.csv",
            [dict(protein_id=p, reaction_id=r, protein_sequence=sequence, reaction_smiles=smiles)],
        )
        annotations.append(dict(protocol="time", subset=subset, protein_id=p, reaction_id=r))
    pq.write_table(pa.Table.from_pylist(annotations), audit / "pair_annotations.parquet")
    pq.write_table(
        pa.Table.from_pylist(
            [dict(protein_id="prot_c", ec_numbers="1.2.3.4", enzyme_core_cofactor_labels="")]
        ),
        audit / "enzyme_annotations.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [dict(reaction_id="r3", associated_ec_numbers="", source_had_generic_wildcard=True)]
        ),
        audit / "reaction_annotations.parquet",
    )
    source = tmp_path / "runs/biological_residual_time/data/source_pretrain"
    analysis.write_csv(
        source / "train_pairs.csv",
        [dict(protein_id="uprot_c", protein_uid="uprot_c", reaction_id="s1")],
    )
    analysis.write_csv(source / "train_rxns.csv", [dict(reaction_id="s1", reaction_smiles="CCO")])
    (source / "manifest.json").write_text(
        json.dumps(
            dict(
                output_pairs_sha256=analysis.digest(source / "train_pairs.csv"),
                output_reactions_sha256=analysis.digest(source / "train_rxns.csv"),
            )
        )
    )
    analysis.metadata(tmp_path, output, audit, splits=("time",))
    enzyme, reaction = list(analysis.rows(output / "time/query_metadata.csv"))
    assert enzyme["stratum_source_entity_id_exposure"] == "present"
    assert enzyme["stratum_reactzyme_sequence_similarity"] == "unknown"
    assert enzyme["stratum_source_sequence_similarity"] == "unknown"
    assert enzyme["stratum_native_cofactor_annotation"] == "unknown"
    assert reaction["stratum_combined_chemistry_similarity"] == "1.0 (fingerprint-equal)"
    assert reaction["query_id"] == "r3_f"
    annotations.pop()
    pq.write_table(pa.Table.from_pylist(annotations), audit / "pair_annotations.parquet")
    with pytest.raises(ValueError, match="different pair partition"):
        analysis.metadata(tmp_path, output, audit, splits=("time",))


def test_commands_use_same_rule_and_never_select_on_test(tmp_path):
    for split, job in runner.plans().items():
        assert split in analysis.SPLITS
        for subset in ("validation", "test"):
            command = runner.command_for(job, subset, tmp_path / split)
            assert command[1].endswith("evaluate_biological_residual.py")
            assert command[command.index("--selection-metric") + 1] == "reactzyme_mrr"
            if subset == "test":
                assert "--alphas" not in command
                assert command[command.index("--selection-from") + 1].endswith("validation.json")
            else:
                assert "--selection-from" not in command
    assert runner.plans()["time"]["parent"].name == "protein-pooling-epoch=29.ckpt"


def test_frozen_parent_identity_is_checked(tmp_path):
    torch = pytest.importorskip("torch")
    parent, residual = tmp_path / "parent.ckpt", tmp_path / "residual.ckpt"
    torch.save({"state_dict": {"model.weight": torch.tensor([1.0])}}, parent)
    torch.save(
        {
            "state_dict": {
                "model.weight": torch.tensor([1.0]),
                "model.biological_residual.weight": torch.tensor([2.0]),
            }
        },
        residual,
    )
    assert runner.validate_parent_weights(parent, residual) == 1
    torch.save({"state_dict": {"model.weight": torch.tensor([3.0])}}, residual)
    with pytest.raises(ValueError, match="differs"):
        runner.validate_parent_weights(parent, residual)


def test_completed_jobs_bind_selection_and_outputs(tmp_path, monkeypatch):
    (tmp_path / "validation.json").write_text("selection")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        (tmp_path / "test.json").write_text("metrics")
        (tmp_path / "test.per_query.csv").write_text("queries")

    monkeypatch.setattr(runner.subprocess, "run", run)
    runner.checked_job(["fake"], tmp_path, "test")
    runner.checked_job(["fake"], tmp_path, "test")
    assert len(commands) == 1
    (tmp_path / "validation.json").write_text("different selection")
    with pytest.raises(ValueError, match="changed"):
        runner.checked_job(["fake"], tmp_path, "test")


def test_failed_child_is_not_marked_complete(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "fake")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        runner.checked_job(["fake"], tmp_path, "validation")
    assert not (tmp_path / "validation.complete.json").exists()


def report_fixture(tmp_path, alpha=0.1):
    directory = tmp_path / "time"
    directory.mkdir()
    records, metadata_rows = [], []
    metrics = ("reactzyme_mrr", "first_positive_mrr", "top_1", "precision_at_10", "mean_rank")
    scores = {}
    for a in (0, 0.1):
        scores[f"{a:g}"] = {}
        for direction in analysis.DIRECTIONS:
            scores[f"{a:g}"].update({f"{direction}/{m}": 0.5 + a for m in metrics})
            scores[f"{a:g}"][f"{direction}/num_queries"] = 3
            for i in range(3):
                records.append(
                    dict(
                        alpha=a,
                        query_id=f"q{i}",
                        direction=direction,
                        known_positive_count=i + 1,
                        train_known_association_count=i,
                        has_unimol2=True,
                        has_chiro=False,
                        **{m: 0.5 + a for m in metrics},
                    )
                )
                if a == 0:
                    metadata_rows.append(
                        dict(direction=direction, query_id=f"q{i}", stratum_ec_annotation="unknown")
                    )
    common = dict(
        checkpoint="frozen",
        best_alpha=alpha,
        selection_metric="balanced_reactzyme_mrr",
        alpha_results=scores,
    )
    for subset in ("validation", "test"):
        (directory / f"{subset}.json").write_text(
            json.dumps(dict(**common, evaluation_split=subset))
        )
    analysis.write_csv(directory / "test.per_query.csv", records)
    analysis.write_csv(directory / "query_metadata.csv", metadata_rows)
    return directory


@pytest.mark.parametrize("alpha,expected_delta", [(0, 0), (0.1, 0.1)])
def test_paired_reports_preserve_unknown_and_alpha_zero(tmp_path, alpha, expected_delta):
    report_fixture(tmp_path, alpha)
    analysis.report(tmp_path, splits=("time",))
    groups = list(analysis.rows(tmp_path / "reports/biological_strata.csv"))
    unknown = [r for r in groups if r["stratum"] == "stratum_ec_annotation"]
    assert len(unknown) == 2 and all(r["group"] == "unknown" for r in unknown)
    assert all(float(r["delta"]) == pytest.approx(expected_delta) for r in groups)
    assert "TIGER ESM2Text (published)" in (tmp_path / "reports/summary.md").read_text()


def test_report_refuses_incomplete_join_and_wrong_aggregates(tmp_path):
    directory = report_fixture(tmp_path)
    records = list(analysis.rows(directory / "query_metadata.csv"))
    analysis.write_csv(directory / "query_metadata.csv", records[:-1])
    with pytest.raises(ValueError, match="query sets differ"):
        analysis.report(tmp_path, splits=("time",))
    analysis.write_csv(directory / "query_metadata.csv", records)
    test = json.loads((directory / "test.json").read_text())
    test["alpha_results"]["0"]["enzyme_to_reaction/reactzyme_mrr"] = 0.9
    (directory / "test.json").write_text(json.dumps(test))
    with pytest.raises(ValueError, match="metrics disagree"):
        analysis.report(tmp_path, splits=("time",))


def test_gpu_check_filters_other_users_gpus(monkeypatch):
    responses = iter(
        ["0, gpu0, 140000\n1, gpu1, 140000\n2, gpu2, 140000\n3, gpu3, 1000\n", "gpu3, 123\n"]
    )
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: next(responses))
    runner.check_gpus(["0", "1", "2"])
    responses = iter(["0, gpu0, 140000\n", "gpu0, 456\n"])
    with pytest.raises(RuntimeError, match="occupied"):
        runner.check_gpus(["0"])
