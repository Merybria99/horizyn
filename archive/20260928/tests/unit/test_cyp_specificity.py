import csv
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from scripts import cyp_specificity_audit as audit
from scripts import run_cyp_specificity as runner
from scripts.cyp_specificity import (
    POOL_PREFIX, canonical_reaction, compare_reports, digest, evaluate_scores, prepare,
    reaction_key, read_rows, verify_bundle, write_json, write_rows,
)


def release(tmp_path, mutate=None):
    sequences = {"A": "ACDEFGHIK", "B": "CDEFGHIKL", "C": "ACDEFGHIK", "D": "DEFGHIKLM"}
    pools = []
    for i, (positive, ids, reaction) in enumerate((
        ("A", "ABC", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"),
        ("B", "ABD", "CCN>>CC=N"),
    )):
        rows = [dict(reaction=reaction, protein_id=p, sequence=sequences[p],
                     cif=f"data/organism{i}/{p}.cif") for p in ids]
        pools.append((f"{POOL_PREFIX}r_{positive}_{i}_0.csv", rows))
    if mutate:
        mutate(pools)
    archive = tmp_path / "release.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, rows in pools:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            data = stream.getvalue().encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return archive


@pytest.fixture
def bundle(tmp_path):
    path = tmp_path / "benchmark"
    prepare(release(tmp_path), path, expected_queries=2)
    return path


def predictions(bundle, tied=False):
    rows = list(read_rows(bundle / "candidates.csv"))
    for row in rows:
        row["score"] = 0 if tied else ({"A": -1, "B": 0, "C": 1, "D": -2}[row["protein_id"]])
    return rows


def test_prepare_keeps_original_ids_and_all_unknown_candidates(bundle):
    manifest = verify_bundle(bundle)
    assert (manifest["query_count"], manifest["candidate_count"], manifest["candidate_pairs"]) == (2, 4, 6)
    queries = list(read_rows(bundle / "queries.csv"))
    assert queries[0]["rxn"] == "CCO>>CC=O"
    assert queries[0]["unique_sequences"] == "2"
    assert queries[0]["candidate_count"] == "3"
    assert len(list(read_rows(bundle / "encoding_pairs.csv"))) == 4
    assert "UNKNOWN" in manifest["labels"]
    # The second pass is checksum-verified, not a silent redefinition of the test.
    assert prepare(bundle.parent / "release.tar.gz", bundle, expected_queries=2) == manifest


def test_charge_preserving_features_reuse_verified_proteins(bundle, tmp_path, monkeypatch):
    import h5py
    import numpy as np
    cached = tmp_path / "cached"
    cached.mkdir()
    source = cached / "proteins.h5"
    ids = [row["protein_id"] for row in read_rows(bundle / "proteins.csv")]
    with h5py.File(source, "w") as handle:
        handle.create_dataset("ids", data=ids, dtype=h5py.string_dtype())
        handle.create_dataset("vectors", data=np.zeros((len(ids), 1024), dtype=np.float16))
    signature = dict(benchmark=digest(bundle / "manifest.json"),
                     backbone_files={"prott5": {}},
                     code={"extract_prott5_residue_embeddings.py": "fixed"})
    write_json(cached / "prott5.complete.json", dict(signature=signature, outputs=[runner.identity(source)]))
    settings = dict(benchmark=str(bundle), features=dict(prott5=str(cached),
                    reaction_t5=str(cached), unimol_weights=str(cached),
                    protein_cache=str(source), preserve_reaction_charges=True))
    args = SimpleNamespace(run_root=tmp_path / "run", models=[], device="cpu", batch_size=16)
    commands = []
    monkeypatch.setattr(runner, "feature_signature", lambda *a: signature)
    monkeypatch.setattr(runner, "stage", lambda command, *a, **kw: commands.append(command))
    runner.features(settings, args)
    assert (args.run_root / "features/proteins.h5").resolve() == source
    assert len(commands) == 3
    assert all("--no-standardize" in command for command in commands)
    assert all("extract_prott5_residue_embeddings.py" not in str(command) for command in commands)
    with h5py.File(source, "r+") as handle:
        handle.attrs["changed"] = True
    with pytest.raises(ValueError, match="provenance"):
        runner.features(settings, args)


@pytest.mark.parametrize("mutation", [
    lambda pools: pools[0][1].pop(0),  # Designated positive absent.
    lambda pools: pools[0][1].append(pools[0][1][0]),  # Duplicate candidate ID.
    lambda pools: pools[1][1][0].update(sequence="AAAA"),  # Conflicting protein ID.
    lambda pools: pools[0][1][0].update(reaction="C>>O"),  # Mixed query reactions.
    lambda pools: pools[0][1][0].update(cif="other/species/A.cif"),
    lambda pools: pools.__setitem__(0, (POOL_PREFIX + "../r_A_0_0.csv", pools[0][1])),
])
def test_bad_release_fails_closed(tmp_path, mutation):
    with pytest.raises(ValueError):
        prepare(release(tmp_path, mutation), tmp_path / "bad", expected_queries=2)
    assert not (tmp_path / "bad/manifest.json").exists()


def test_tampered_bundle_is_rejected(bundle):
    with (bundle / "candidates.csv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="changed"):
        evaluate_scores(bundle, [])


def test_chemistry_preserves_stereo_stoichiometry_and_direction():
    assert canonical_reaction("O.[CH3:1][OH:2]>>C=O.O") == canonical_reaction("CO.O>>O.C=O")
    assert canonical_reaction("O.O>>O") != canonical_reaction("O>>O")
    assert canonical_reaction("N[C@H](C)O>>CC=O") != canonical_reaction("N[C@@H](C)O>>CC=O")
    assert canonical_reaction("C>>O") != canonical_reaction("O>>C")
    assert reaction_key("C>>O") == reaction_key("O>>C")
    with pytest.raises(ValueError):
        canonical_reaction("C.*>>O")
    dative = canonical_reaction("[NH3]->[Cu+2]>>[NH3]->[Cu+2]")
    assert canonical_reaction(dative) == dative


def test_designated_positive_metrics_and_separate_sequence_diagnostic(bundle):
    scores = predictions(bundle)
    metrics, queries, ranks = evaluate_scores(bundle, scores)
    assert metrics["first_positive_mrr"] == pytest.approx((1 / 3 + 1) / 2)
    assert metrics["hit_at_1"] == .5
    assert metrics["hit_at_5"] == 1
    assert metrics["sequence_mrr"] == 1
    assert queries[0]["first_positive_rank"] == 3
    assert queries[0]["sequence_rank"] == 1
    assert len(ranks) == 6
    assert evaluate_scores(bundle, reversed(scores)) == (metrics, queries, ranks)


def test_ties_have_explicit_bounds(bundle):
    metrics, _, _ = evaluate_scores(bundle, predictions(bundle, tied=True))
    assert metrics["first_positive_mrr"] == .75  # Deterministic protein-ID order.
    assert metrics["optimistic_mrr"] == 1
    assert metrics["pessimistic_mrr"] == pytest.approx(1 / 3)
    assert metrics["positive_tie_size"] == 3


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(),
    lambda rows: rows.append(rows[0].copy()),
    lambda rows: rows[0].update(score=float("nan")),
    lambda rows: rows[0].update(score=float("inf")),
    lambda rows: rows[0].update(protein_id="not_a_candidate"),
])
def test_invalid_predictions_never_shrink_candidate_pool(bundle, mutation):
    rows = predictions(bundle)
    mutation(rows)
    with pytest.raises(ValueError):
        evaluate_scores(bundle, rows)


def test_paired_report_is_reproducible(bundle, tmp_path):
    base, full = tmp_path / "f3.csv", tmp_path / "residual.csv"
    write_rows(base, predictions(bundle))
    write_rows(full, predictions(bundle, tied=True))
    compare_reports(bundle, {"f3": base, "residual": full}, tmp_path / "report", bootstrap_samples=50)
    result = json.loads((tmp_path / "report/summary.json").read_text())
    difference = result["differences"][0]
    assert difference["delta"] == pytest.approx(.75 - (1 / 3 + 1) / 2)
    assert difference["organism_clusters"] == 2
    compare_reports(bundle, {"f3": base, "residual": full}, tmp_path / "again", bootstrap_samples=50)
    assert json.loads((tmp_path / "again/summary.json").read_text()) == result
    with pytest.raises(ValueError):
        compare_reports(bundle, {}, tmp_path / "empty")


def source_files(tmp_path):
    spec = {key: tmp_path / filename for key, filename in (
        ("pairs", "pairs.csv"), ("reactions", "reactions.csv"), ("proteins", "training.fasta"))}
    write_rows(spec["pairs"], [
        dict(reaction_id="train1", protein_id="prot_t1", label=1),
        dict(reaction_id="train2", protein_id="uprot_t2", label=1),
        dict(reaction_id="not_training", protein_id="ignored", label=0),
    ])
    write_rows(spec["reactions"], [
        dict(reaction_id="train1", rxn="CCO>>CC=O"),
        dict(reaction_id="train2", rxn="CC=O>>CCO"),
        dict(reaction_id="not_training", rxn="CCN>>CC=N"),
    ])
    spec["proteins"].write_text(">sp|t1|one\nACDEFGHIK\n>t2\nCDEFGHIKL\n>ignored\nDEFGHIKLM\n")
    return spec


def test_audit_uses_train_positives_only_and_keeps_unknowns(bundle, tmp_path, monkeypatch):
    spec = source_files(tmp_path)
    def hits(*args):
        return {"A": [dict(target="t1", bits=100, identity=1, qcov=1, tcov=1)],
                "D": [dict(target="t2", bits=20, identity=.9, qcov=.3, tcov=1)]}
    monkeypatch.setattr(audit, "align", hits)
    output = tmp_path / "audit"
    result = audit.audit_source(bundle, spec, output, mmseqs=__file__)
    assert result["training_proteins"] == 2
    assert result["valid_reactions"] == 2
    rows = {r["query_id"]: r for r in read_rows(output / "queries.csv")}
    assert rows["r_A_0_0"]["exact_pair_match"] == "True"
    assert rows["r_A_0_0"]["exact_reaction_matches"] == "train1;train2"
    assert rows["r_A_0_0"]["nearest_reaction_ids"] == "train1"
    assert rows["r_B_1_0"]["exact_reaction_matches"] == ""
    proteins = {r["protein_id"]: r for r in read_rows(output / "proteins.csv")}
    assert proteins["C"]["exact_sequence_matches"] == "t1"  # Alias shares same sequence.
    assert proteins["D"]["homology_status"] == "no_reported_coverage80_hit"
    assert proteins["D"]["best_reported_identity"] == ""
    assert result["upstream_supervised_exposure"] == "UNKNOWN"
    assert audit.audit_source(bundle, spec, output, mmseqs=__file__) == result
    with (output / "homology_baseline.csv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="artifact changed"):
        audit.audit_source(bundle, spec, output, mmseqs=__file__)


def test_audit_rejects_conflicting_reactions(bundle, tmp_path):
    spec = source_files(tmp_path)
    rows = list(read_rows(spec["reactions"])) + [dict(reaction_id="train1", rxn="C>>O")]
    write_rows(spec["reactions"], rows)
    with pytest.raises(ValueError, match="Conflicting source reaction"):
        audit.audit_source(bundle, spec, tmp_path / "audit", mmseqs=__file__)


def test_reactzyme_participant_sets_are_not_claimed_exact_transformations(bundle, tmp_path, monkeypatch):
    spec = source_files(tmp_path)
    write_rows(spec["reactions"], [dict(reaction_id="train1", rxn="CCO.CC=O"),
                                   dict(reaction_id="train2", rxn="CCO.CCN")])
    monkeypatch.setattr(audit, "align", lambda *args: {})
    output = tmp_path / "audit"
    result = audit.audit_source(bundle, spec, output, mmseqs=__file__, homology=False)
    assert result["reaction_representation"] == "participant_set"
    rows = list(read_rows(output / "queries.csv"))
    assert rows[0]["exact_reaction_matches"] == ""
    assert rows[0]["exact_pair_match"] == "UNKNOWN"
    assert rows[0]["participant_set_matches"] == "train1"
    assert rows[0]["participant_set_pair_match"] == "True"
    assert rows[0]["nearest_reaction_similarity"] == "1.0"


def test_audit_rejects_missing_training_sequences(bundle, tmp_path):
    spec = source_files(tmp_path)
    spec["proteins"].write_text(">t1\nACDEFGHIK\n")
    with pytest.raises(ValueError, match="Missing 1 training sequences"):
        audit.audit_source(bundle, spec, tmp_path / "audit", mmseqs=__file__)


def test_interrupted_alignment_is_not_reused(tmp_path, monkeypatch):
    output = tmp_path / "hits.tsv"
    def fail(command, **kwargs):
        Path(command[4]).write_text("partial\n")
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(audit.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        audit.align("mmseqs", "query", "reference", output, 1)
    assert not output.exists()
    def success(command, **kwargs):
        Path(command[4]).write_text("A\tt1\t1.0\t1.0\t1.0\t100\n")
    monkeypatch.setattr(audit.subprocess, "run", success)
    assert audit.align("mmseqs", "query", "reference", output, 1)["A"][0]["bits"] == 100


def test_feature_stage_reuses_only_successful_matching_output(tmp_path, monkeypatch):
    output, receipt = tmp_path / "features.h5", tmp_path / "stage.complete.json"
    calls = []
    def fake(command, *args):
        calls.append(command)
        output.write_text("features")
    monkeypatch.setattr(runner, "run_command", fake)
    runner.stage(["extract"], [output], {"input": "one"}, receipt)
    runner.stage(["extract"], [output], {"input": "one"}, receipt)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="Changed stage"):
        runner.stage(["extract"], [output], {"input": "two"}, receipt)
    output.write_text("tampered")
    with pytest.raises(ValueError, match="Changed stage"):
        runner.stage(["extract"], [output], {"input": "one"}, receipt)


def test_training_csv_sequences_match_fasta_normalization(tmp_path):
    path = tmp_path / "training.csv"
    write_rows(path, [dict(protein_id="prot_A", protein_sequence=" acd ef\n")])
    assert list(audit.sequences_from(path)) == [("prot_A", "ACDEF")]


def test_feature_commands_preserve_pools_and_fixed_recipe(bundle, tmp_path, monkeypatch):
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    settings = dict(benchmark=str(bundle), features={
        "prott5": str(tmp_path), "reaction_t5": str(tmp_path),
        "unimol_weights": str(tmp_path), "cofactor_dictionary": str(schema),
        "sleec_checkpoint": str(schema)},
        models={name: {"chemistry_schema": str(schema)} for name in ("f3", "residual")})
    args = SimpleNamespace(run_root=tmp_path / "run", models=["f3", "residual"], device="cuda", batch_size=16)
    commands = []
    def fake_stage(command, outputs, *rest, after_command=None):
        commands.append(list(map(str, command)))
        if after_command:
            commands.append(list(map(str, after_command)))
        for path in outputs:
            path.write_text("fake")
    monkeypatch.setattr(runner, "stage", fake_stage)
    runner.features(settings, args)
    by_script = {Path(command[1]).name: command for command in commands}
    for script in ("extract_unimol2_reaction_embeddings.py", "extract_chiro_reaction_embeddings.py"):
        assert "--no-skip-invalid-reactions" in by_script[script]
        assert "--no-bidirectional" in by_script[script]
    protein_commands = [c for c in commands if Path(c[1]).name == "extract_prott5_residue_embeddings.py"]
    assert len(protein_commands) == 2
    assert "--merge-only" not in protein_commands[0]
    assert "--merge-only" in protein_commands[1]
    assert protein_commands[1][protein_commands[1].index("--merge-order") + 1] == "shard"
    functional = by_script["cache_sleec_functional_tokens.py"]
    assert "--bf16" in functional
    assert functional[functional.index("--top-k") + 1] == "48"
    assert functional[functional.index("--context-k") + 1] == "16"
    assert functional[functional.index("--pairs") + 1] == str(bundle / "encoding_pairs.csv")


def test_score_adapter_checks_chemistry_mask_and_writes_complete_pool(bundle, tmp_path, monkeypatch):
    import torch
    from horizyn.config import DotDict
    from horizyn.benchmarks import retrieval
    from horizyn.datasets import residue_hdf5
    import horizyn.config

    queries = list(read_rows(bundle / "queries.csv"))
    pids = [r["protein_id"] for r in read_rows(bundle / "proteins.csv")]
    chemistry_present = [False]
    class ReactionData:
        keys = [q["reaction_id"] for q in queries]
        def __getitem__(self, key):
            return {"has_unimol2": True, "has_chiro": True,
                    "has_reaction_chemistry": chemistry_present[0]}
    class TargetData:
        keys = pids
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
    monkeypatch.setattr(runner, "prediction_signature", lambda *a: {"test": True})
    monkeypatch.setattr(horizyn.config, "load_config", lambda *a: DotDict({"data": {}}))
    monkeypatch.setattr(retrieval, "build_reaction_inputs", lambda *a: ReactionData())
    monkeypatch.setattr(retrieval, "load_repo_checkpoint", lambda *a: (SimpleNamespace(), "residue"))
    monkeypatch.setattr(retrieval, "encode_residue_targets", lambda *a, **kw: torch.tensor([[1., 0.], [0., 1.], [1., 0.], [-1., 0.]]))
    monkeypatch.setattr(retrieval, "encode_reactions", lambda *a: torch.eye(2))
    monkeypatch.setattr(residue_hdf5, "ResidueEmbedDataset", TargetData)
    settings = dict(benchmark=str(bundle), models={"f3": {"config": "unused", "checkpoint": "unused"}})
    args = SimpleNamespace(run_root=tmp_path / "run", device="cpu", batch_size=2)
    with pytest.raises(ValueError, match="Missing reaction modality"):
        runner.score_model(settings, args, "f3")
    chemistry_present[0] = True
    runner.score_model(settings, args, "f3")
    scores = args.run_root / "scores/f3.csv"
    metrics, _, _ = evaluate_scores(bundle, read_rows(scores))
    assert metrics["first_positive_mrr"] == 1
    runner.score_model(settings, args, "f3")  # Verified cache reuse.


def test_report_checks_receipts_and_marks_missing_audits_unknown(bundle, tmp_path, monkeypatch):
    args = SimpleNamespace(run_root=tmp_path / "run", models=["f3"], external_scores=[],
                           device="cpu", batch_size=16)
    settings = dict(benchmark=str(bundle), models={"f3": {"sources": ["training"]}})
    path = args.run_root / "scores/f3.csv"
    write_rows(path, predictions(bundle))
    signature = dict(device="cuda", batch_size=16, checkpoint="fixed")
    write_json(path.with_suffix(".json"), dict(signature=signature, scores_sha256=digest(path)))
    monkeypatch.setattr(runner, "prediction_signature", lambda *a: dict(signature, device="cpu"))
    runner.report(settings, args)
    provenance = json.loads((args.run_root / "reports/provenance.json").read_text())
    assert "UNKNOWN" in provenance["exposure_summary"]["training"]["status"]
    assert provenance["models"]["f3"]["signature"]["device"] == "cuda"
    with path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Stale or modified"):
        runner.report(settings, args)


def test_unsafe_transformers_bin_loading_is_caught_before_audits(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"placeholder")
    with pytest.raises(RuntimeError, match="PyTorch 2.4"):
        runner.check_transformer_checkpoint(tmp_path, "2.4.0+cu121")
    runner.check_transformer_checkpoint(tmp_path, "2.6.0+cpu")
    (tmp_path / "model.safetensors").write_bytes(b"placeholder")
    runner.check_transformer_checkpoint(tmp_path, "2.4.0+cu121")


def test_empty_transformers_snapshot_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="No Transformers model weights"):
        runner.check_transformer_checkpoint(tmp_path, "2.6.0+cpu")


def test_checkpoint_conversion_refuses_unpatched_torch(tmp_path, monkeypatch):
    import torch
    from scripts.prepare_cyp_prott5_safetensors import convert
    monkeypatch.setattr(torch, "__version__", "2.4.0+cu121")
    with pytest.raises(RuntimeError, match="requires PyTorch >=2.6"):
        convert(tmp_path / "source", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.fixture
def completed_prott5_attempt(tmp_path):
    import h5py
    import numpy as np
    from scripts import extract_prott5_residue_embeddings as extractor

    def different_lengths(pools):
        for _, rows in pools:
            for row in rows:
                row["sequence"] = row["sequence"][:{"A": 9, "B": 6, "C": 9, "D": 4}[row["protein_id"]]]
    bundle = tmp_path / "benchmark"
    prepare(release(tmp_path, different_lengths), bundle, expected_queries=2)
    model = tmp_path / "model"
    model.mkdir()
    schema = tmp_path / "schema.json"
    schema.write_text("{}")
    settings = dict(benchmark=str(bundle), features={
        "prott5": str(model), "reaction_t5": str(model), "unimol_weights": str(model),
        "cofactor_dictionary": str(schema), "sleec_checkpoint": str(schema)},
        models={name: {"chemistry_schema": str(schema)} for name in ("f3", "residual")})
    args = SimpleNamespace(run_root=tmp_path / "run", models=["f3", "residual"], device="cuda", batch_size=16)
    output = args.run_root / "features"
    shard = extractor.get_shard_path(output / "proteins.h5", None, 0, 1)
    shard.parent.mkdir(parents=True)
    records = sorted(extractor.iter_fasta_records(bundle / "proteins.fasta"),
                     key=lambda r: (len(r.sequence), r.index))
    assert [r.index for r in records] == [3, 1, 0, 2]  # Not FASTA-contiguous.
    offsets = np.cumsum([0] + [len(r.sequence) for r in records], dtype=np.int64)
    with h5py.File(shard, "w") as handle:
        handle["ids"] = np.asarray([r.protein_id.encode() for r in records])
        handle["indices"] = np.array([r.index for r in records], dtype=np.int64)
        handle["offsets"] = offsets
        handle["vectors"] = np.concatenate([np.full((len(r.sequence), 1024), r.index, dtype="float16") for r in records])
        handle.attrs.update(model_name=str(model), source_fasta=str((bundle / "proteins.fasta").resolve()),
                            max_sequence_length=1022, sequence_truncation="ends_center", rank=0, world_size=1,
                            length_sort=True, padded_token_budget=True, residue_dim=1024, processed_count=4)
    previous = dict(runner.feature_signature(settings, args),
                    runner_sha256="58864d5d2aacbc54e08764a131f4bdd350429942b8f88b72793da62c7d05d377")
    write_json(output / "prott5.complete.inputs.json", previous)
    return settings, args, shard


@pytest.mark.parametrize("merge_only", [False, True])
def test_actual_extractor_merge_reuses_completed_sorted_shard(completed_prott5_attempt, monkeypatch, merge_only):
    import sys
    import h5py
    import numpy as np
    from scripts import extract_prott5_residue_embeddings as extractor

    settings, args, shard = completed_prott5_attempt
    original_sha = digest(shard)
    calls = []
    def never_load_model(*a, **kw):
        raise AssertionError("Completed shard must not reload ProtT5 or use a GPU")
    monkeypatch.setattr(extractor, "load_model_and_tokenizer", never_load_model)
    def actual_extractor(command, *a):
        calls.append(command)
        with monkeypatch.context() as cli:
            cli.setattr(sys, "argv", list(map(str, command[1:])))
            extractor.main()
    monkeypatch.setattr(runner, "run_command", actual_extractor)
    actual_stage = runner.stage
    def only_prott5_stage(command, outputs, *a, **kw):
        if Path(command[1]).name == "extract_prott5_residue_embeddings.py":
            return actual_stage(command, outputs, *a, **kw)
        for path in outputs:
            path.write_text("Other feature stages are out of scope for this regression")
    monkeypatch.setattr(runner, "stage", only_prott5_stage)
    runner.features(settings, args, merge_only=merge_only)
    assert len(calls) == (1 if merge_only else 2)
    assert "--merge-only" in calls[-1]
    if not merge_only:
        assert "--resume" in calls[0] and "--merge-only" not in calls[0]
    with h5py.File(shard) as source, h5py.File(args.run_root / "features/proteins.h5") as merged:
        for key in ("ids", "offsets", "vectors"):
            np.testing.assert_array_equal(source[key][:], merged[key][:])
        assert not merged["vectors"].is_virtual
        assert merged.attrs["merge_order"] == "shard"
    assert digest(shard) == original_sha
    output = args.run_root / "features"
    assert (output / "prott5.complete.inputs.before-merge-fix.json").exists()
    assert (output / "prott5.merge_recovery.json").exists()
    assert (output / "prott5.complete.json").exists()
    runner.features(settings, args, merge_only=merge_only)
    assert len(calls) == (1 if merge_only else 2)  # Nothing re-extracted or re-merged.


def test_merge_recovery_rejects_changed_inputs(completed_prott5_attempt):
    settings, args, _ = completed_prott5_attempt
    args.batch_size = 32
    with pytest.raises(ValueError, match="inputs or runtime changed"):
        runner.features(settings, args, merge_only=True)
    assert not (args.run_root / "features/prott5.complete.inputs.before-merge-fix.json").exists()


def test_merge_recovery_rejects_unfinished_shard(completed_prott5_attempt):
    import h5py
    settings, args, shard = completed_prott5_attempt
    with h5py.File(shard, "r+") as handle:
        handle.attrs["processed_count"] = 0
    with pytest.raises(ValueError, match="Shard is not complete"):
        runner.features(settings, args, merge_only=True)


def test_failed_merge_does_not_mark_feature_stage_complete(tmp_path, monkeypatch):
    output, receipt = tmp_path / "proteins.h5", tmp_path / "prott5.complete.json"
    def command(*args):
        if args[0] == ["merge"]:
            raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(runner, "run_command", command)
    with pytest.raises(subprocess.CalledProcessError):
        runner.stage(["extract"], [output], {"input": "fixed"}, receipt, after_command=["merge"])
    assert not receipt.exists()
    assert receipt.with_suffix(".inputs.json").exists()
