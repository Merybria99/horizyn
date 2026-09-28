import io
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pytest

from scripts import cyp_baselines as core
from scripts import run_cyp_baselines as runner
from tests.unit.test_cyp_specificity import bundle, predictions, release


def test_exact_expected_ties_not_reciprocal_mean_rank():
    m = core.tie_metrics(1, 3, 3)
    assert m["mrr"] == pytest.approx((1 + .5 + 1 / 3) / 3)
    assert m["mrr"] != .5
    assert m["hit_at_1"] == pytest.approx(1 / 3)
    assert m["hit_at_10pct"] == 0
    assert m["hit_at_50pct"] == pytest.approx(1 / 3)
    assert m["percentile"] == pytest.approx(100 * 2 / 3)


def test_partial_tie_and_top_fraction_boundary():
    m = core.tie_metrics(4, 7, 50)
    assert m["hit_at_5"] == .5
    assert m["hit_at_10pct"] == .5
    assert m["mean_rank"] == 5.5
    assert m["hit_at_1"] == 0
    assert core.tie_metrics(1, 1, 6)["hit_at_10pct"] == 0
    with pytest.raises(ValueError):
        core.tie_metrics(2, 1, 5)


def test_evaluation_preserves_strict_pools_and_order_invariance(bundle):
    scores = predictions(bundle, tied=True)
    a = core.evaluate(bundle, scores)
    assert a == core.evaluate(bundle, reversed(scores))
    assert a[0]["mrr"] == pytest.approx((1 + .5 + 1 / 3) / 3)
    assert a[0]["positive_ties"] == 2
    assert len(a[2]) == 6
    assert a[1][0]["legacy_id_tiebreak_mrr"] == 1
    with pytest.raises(ValueError, match="Missing"):
        core.evaluate(bundle, scores[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        core.evaluate(bundle, scores + scores[:1])
    scores[0]["score"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        core.evaluate(bundle, scores)


def test_bootstrap_pairs_by_id_and_keeps_whole_organisms(bundle):
    _, a, _ = core.evaluate(bundle, predictions(bundle))
    _, b, _ = core.evaluate(bundle, predictions(bundle, tied=True))
    result = core.paired_intervals(a, b, samples=100, seed=13)
    assert result == core.paired_intervals(list(reversed(a)), b, samples=100, seed=13)
    assert result["mrr"]["delta"] == pytest.approx(np.mean([r["mrr"] for r in b]) - np.mean([r["mrr"] for r in a]))
    assert result["mrr"]["organism_clusters"] == 2
    assert core.paired_intervals(a, a, samples=100)["mrr"]["ci95"] == [0., 0.]
    with pytest.raises(ValueError, match="unique"):
        core.paired_intervals(a, b[:1])
    b[0]["organism"] = "different"
    with pytest.raises(ValueError, match="organisms"):
        core.paired_intervals(a, b)


def test_archive_rejects_missing_link_duplicate_and_bad_hash(tmp_path):
    archive = release(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        list(core.archive_members(archive, names=[], expected_sha="bad"))
    with pytest.raises(ValueError, match="Missing"):
        list(core.archive_members(archive, names=["absent.csv"]))
    for kind in ("link", "duplicate", "traversal"):
        target = tmp_path / f"{kind}.tar.gz"
        name = "data/../unsafe" if kind == "traversal" else "data/test"
        with tarfile.open(target, "w:gz") as tar:
            m = tarfile.TarInfo(name)
            if kind == "link":
                m.type, m.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(m)
            else:
                m.size = 1
                tar.addfile(m, io.BytesIO(b"x"))
                if kind == "duplicate":
                    tar.addfile(m, io.BytesIO(b"x"))
        with pytest.raises(ValueError, match="Unsafe"):
            list(core.archive_members(target, prefix="data/"))


def test_prepare_uses_only_train_positives_and_original_molecule_order(tmp_path):
    original = release(tmp_path)
    extended = tmp_path / "extended.tar.gz"
    with tarfile.open(original) as source, tarfile.open(extended, "w:gz") as target:
        for member in source:
            target.addfile(member, source.extractfile(member))
        for fold, seq in (("train_pos", "ACDEFG"), ("val_pos", "AAAAAA"), ("train_neg", "CCCCCC")):
            raw = f">A|protein|ignored\n{seq}\n>B|ccd\nHEM\n>C|smiles\nCCO\n".encode()
            member = tarfile.TarInfo(f"data_dir/retrain/train_inputs/{fold}/REF_1_0.fasta")
            member.size = len(raw)
            target.addfile(member, io.BytesIO(raw))
    from scripts.cyp_specificity import prepare
    directory, output = tmp_path / "bundle", tmp_path / "inputs"
    prepare(extended, directory, expected_queries=2)
    info = core.prepare_inputs(extended, directory, output, expected_sha=core.digest(extended))
    assert info["reference_samples"] == 1
    assert list(core.read_rows(output / "references.csv"))[0]["sequence"] == "ACDEFG"
    assert len(list(core.read_rows(output / "mapped_inputs.csv"))) == 6
    assert list(core.read_rows(output / "substrates.csv"))[0]["unmapped_canonical"] == "CCO"


def test_controls_follow_global_alignment_and_mean_record_weighting(bundle, tmp_path):
    prepared = tmp_path / "prepared"
    refs = [dict(sample_id=str(i), protein_id=f"p{i}", sequence=seq, substrate=sub)
            for i, (seq, sub) in enumerate((
                ("ACDEFGHIK", "CCO"), ("CDEFGHIKL", "CCO"), ("CDEFGHIKL", "CCO"),
                ("DEFGHIKLM", "CCN")))]
    core.write_rows(prepared / "references.csv", refs)
    core.write_rows(prepared / "substrates.csv", [dict(query_id="r_A_0_0", substrate="CCO"),
                                                  dict(query_id="r_B_1_0", substrate="CCN")])
    outputs = core.alignment_controls(bundle, prepared, tmp_path / "controls", workers=1)
    proteins = {r["protein_id"]: r["sequence"] for r in core.read_rows(bundle / "proteins.csv")}
    aligner = core.make_aligner()
    for method, rows in outputs.items():
        core.evaluate(bundle, rows)
        for row in rows:
            seq = proteins[row["protein_id"]]
            relevant = refs if method == "alignment_sequence" else (refs[:3] if row["query_id"] == "r_A_0_0" else refs[3:])
            values = [aligner.score(seq, r["sequence"]) for r in relevant]
            expected = max(values) if method == "alignment_sequence" else np.mean(values)
            assert row["score"] == expected


def test_nearest_substrate_uses_first_reference_tie():
    # CCO and OCC are equivalent fingerprints but distinct original strings.
    refs = [dict(substrate="OCC"), dict(substrate="CCO")]
    rows = core.nearest_substrates([dict(query_id="q", substrate="CCO")], refs)
    assert rows[0]["substrate"] == "OCC"
    assert rows[0]["tied_nearest_substrates"] == 2
    assert rows[0]["tanimoto"] == 1


def settings_for(bundle):
    return dict(benchmark=str(bundle), methods={"clipzyme_pretrained": dict(group="pretrained", import_only=True)})


def test_scores_reject_changes_and_tampering(bundle, tmp_path):
    settings, run = settings_for(bundle), tmp_path / "run"
    rows = predictions(bundle)
    runner.save_scores(settings, run, "test", rows, {"id": "x"}, "protocol")
    runner.save_scores(settings, run, "test", list(reversed(rows)), {"id": "x"}, "protocol")
    with pytest.raises(ValueError, match="provenance"):
        runner.save_scores(settings, run, "test", rows, {"id": "y"}, "protocol")
    rows[0]["score"] += 1
    with pytest.raises(ValueError, match="Changed scores"):
        runner.save_scores(settings, run, "test", rows, {"id": "x"}, "protocol")
    with pytest.raises(ValueError, match="Stale"):
        runner.existing_score(run, "test", "changed-protocol")
    (run / "scores/test.csv").write_text("broken")
    with pytest.raises(ValueError, match="tampered"):
        runner.existing_score(run, "test", "protocol")


def test_external_import_needs_complete_scores_and_concrete_provenance(bundle, tmp_path):
    scores, metadata = tmp_path / "scores.csv", tmp_path / "provenance.json"
    core.write_rows(scores, predictions(bundle))
    provenance = dict(method="clipzyme_pretrained", variant="pretrained", checkpoint_sha256="a" * 64,
        source_revision="b" * 40, input_manifest_sha256=core.digest(bundle / "manifest.json"),
        input_scope="reaction+sequence+structure", upstream_exposure="UNKNOWN",
        checkpoint_selection="fixed_without_cyp_selection", generator_command="official inference command",
        scores_sha256=core.digest(scores), score_direction="higher_is_better")
    core.write_json(metadata, provenance)
    runner.import_scores(settings_for(bundle), tmp_path / "good", "clipzyme_pretrained", scores, metadata, "p")
    for key, value, message in (("variant", "cyp_adapted", "variant"),
                                ("checkpoint_sha256", "UNKNOWN", "checkpoint"),
                                ("source_revision", "main", "immutable"),
                                ("scores_sha256", "0" * 64, "checksum"),
                                ("input_manifest_sha256", "0" * 64, "different benchmark"),
                                ("score_direction", "lower_is_better", "higher-is-better"),
                                ("checkpoint_selection", "best CYP MRR", "CYP-selected")):
        core.write_json(metadata, {**provenance, key: value})
        with pytest.raises(ValueError, match=message):
            runner.import_scores(settings_for(bundle), tmp_path / "bad", "clipzyme_pretrained", scores, metadata, "p")


def test_frozen_protocol_rejects_changes(bundle, tmp_path):
    settings = settings_for(bundle)
    sha = runner.freeze(settings, tmp_path / "run")
    assert sha == runner.freeze(settings, tmp_path / "run")
    with pytest.raises(ValueError, match="NEW"):
        runner.freeze({**settings, "seed": 2}, tmp_path / "run")


def test_config_preserves_virtualenv_interpreter_symlink(tmp_path):
    import yaml
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to("/usr/bin/python3")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(dict(benchmark="bundle", archive="data.tar.gz",
        fusion_checkpoints="model.tar.gz", fusion_python=str(executable), existing_run="old")))
    assert runner.load_config(config)["fusion_python"] == str(executable)


def test_report_marks_missing_external_methods_pending(bundle, tmp_path):
    settings = dict(**settings_for(bundle), existing_run=str(tmp_path / "missing"),
                    seed=42, bootstrap_samples=10)
    settings["methods"]["random"] = dict(group="control", label="Random")
    run = tmp_path / "reporting"
    runner.save_scores(settings, run, "random", predictions(bundle, tied=True), {}, "p")
    runner.report(settings, run, "p")
    summary = runner.read_json(run / "report/summary.json")
    assert not summary["complete"]
    assert summary["pending"] == ["clipzyme_pretrained"]
    assert summary["methods"]["random"]["metrics"]["queries"] == 2


def test_fusion_forward_is_independent_of_batch_composition():
    import torch
    from scripts.cyp_fusionesp import Contrastive_learning_layer
    torch.manual_seed(42)
    model = Contrastive_learning_layer().eval()
    e, s = torch.randn(3, 2560), torch.randn(3, 768)
    with torch.inference_mode():
        whole = model(e, s)
        first = model(e[:1], s[:1])
    for a, b in zip(whole, first):
        torch.testing.assert_close(a[:1], b)
        torch.testing.assert_close(a.norm(dim=1), torch.ones(3))


def test_restricted_loader_in_isolated_torch_environment():
    interpreter = runner.ROOT / ".deps/cyp-checkpoint-converter/bin/python"
    if not interpreter.is_file():
        pytest.skip("Optional isolated torch>=2.6 environment not installed")
    program = '''
import io, os, pickle
import numpy as np
import torch
from scripts.cyp_fusionesp import safe_load
buf = io.BytesIO()
torch.save({"feature": np.ones(3, dtype=np.float32)}, buf)
assert np.array_equal(safe_load(buf.getvalue())["feature"], np.ones(3))
class Forbidden:
    def __reduce__(self):
        return os.system, ("exit 0",)
buf = io.BytesIO()
torch.save(Forbidden(), buf)
try:
    safe_load(buf.getvalue())
except pickle.UnpicklingError:
    pass
else:
    raise AssertionError("Unrestricted pickle loading was permitted")
'''
    subprocess.run([str(interpreter), "-c", program], cwd=runner.ROOT, check=True, timeout=30)
