import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts import evaluate_horizyn1_parallel as parallel


def test_partitions_preserve_serial_batches_and_cover_tail():
    chunks = parallel.blocks(209, 64)
    owned = [x for rank in range(4) for x in parallel.owned_blocks(209, 64, rank, 4)]
    assert [(a, b) for _, a, b in sorted(owned)] == chunks
    assert [x for a, b in chunks for x in range(a, b)] == list(range(209))
    assert all(a % 16 == 0 for a, _ in chunks)
    assert parallel.owned_blocks(1, 64, 3, 4) == []


def test_atomic_json_accepts_cli_paths(tmp_path):
    path = tmp_path / "job.json"
    parallel.atomic_json(path, {"args": {"checkpoint": tmp_path / "model.ckpt"}})
    assert json.loads(path.read_text())["args"]["checkpoint"] == str(tmp_path / "model.ckpt")


def test_tensor_chunk_resume_and_validation(tmp_path):
    path = parallel.tensor_path(tmp_path, 0, 3)
    value = torch.randn(3, 512)
    parallel.save_tensor(path, "sig", 0, 3, value)
    assert torch.equal(parallel.read_tensor(path, "sig", 0, 3), value)
    with pytest.raises(ValueError, match="Invalid"):
        parallel.read_tensor(path, "other", 0, 3)
    with pytest.raises(ValueError, match="Invalid"):
        parallel.read_tensor(path, "sig", 1, 4)
    with pytest.raises(ValueError, match="Invalid"):
        parallel.save_tensor(path, "sig", 0, 3, value.half())
    value[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Invalid"):
        parallel.save_tensor(path, "sig", 0, 3, value)


def test_streamed_merge_matches_catalog_and_detects_tampering(tmp_path):
    expected = torch.randn(9, 512)
    for a, b in parallel.blocks(9, 4):
        parallel.save_tensor(parallel.tensor_path(tmp_path, a, b), "sig", a, b, expected[a:b])
    path = parallel.merge_catalog(tmp_path, "sig", 9, 4)
    np.testing.assert_array_equal(np.load(path), expected.numpy())
    assert parallel.merge_catalog(tmp_path, "sig", 9, 4) == path
    with path.open("r+b") as handle:
        handle.seek(-4, 2)
        handle.write(b"xxxx")
    with pytest.raises(ValueError, match="changed"):
        parallel.merge_catalog(tmp_path, "sig", 9, 4)


def test_missing_chunk_cannot_complete_merge(tmp_path):
    parallel.save_tensor(parallel.tensor_path(tmp_path, 0, 4), "sig", 0, 4, torch.zeros(4, 512))
    with pytest.raises(FileNotFoundError):
        parallel.merge_catalog(tmp_path, "sig", 9, 4)
    assert not (tmp_path / "proteins.complete.json").exists()


def test_parallel_query_scoring_matches_serial_with_ties_and_unequal_tail(tmp_path):
    torch.manual_seed(9)
    candidates = torch.randn(13, 5)
    candidates[2] = candidates[0]  # exercise canonical-row tie ordering
    queries = torch.cat([candidates[:3], torch.randn(4, 5)])
    ids = [f"r{i}" for i in range(7)]
    lookup = {f"p{i}": i for i in range(13)}
    truth = {q: {f"p{i % 13}", f"p{(i + 4) % 13}"} for i, q in enumerate(ids)}
    options = dict(batch_size=2, chunk_size=4, device="cpu", label="test")
    reference = parallel.serial.evaluate_direction(queries, candidates, ids, truth, lookup, **options)
    direction = "reaction_to_enzyme"
    for rank in range(4):
        for _, a, b in parallel.owned_blocks(len(ids), 4, rank, 4):
            result = parallel.serial.evaluate_direction(queries[a:b], candidates, ids[a:b], truth, lookup, **options)
            parallel.atomic_json(parallel.score_path(tmp_path, direction, a, b),
                                 dict(signature="sig", direction=direction, start=a, stop=b, metrics=result))
    combined = parallel.merge_scores(tmp_path, "sig", direction, 7, 13, 4)
    assert combined.keys() == reference.keys()
    assert combined == pytest.approx(reference, abs=1e-12)
    with pytest.raises(ValueError, match="Invalid score"):
        parallel.merge_scores(tmp_path, "wrong", direction, 7, 13, 4)


def test_encoding_chunks_reuse_existing_serial_encoder_and_resume(tmp_path, monkeypatch):
    proteins = [f"p{i}" for i in range(9)]
    class Model:
        enzyme_prototype_count = 1
    class Module:
        model = Model()
        embedding_similarity = "cosine"
        def eval(self):
            return self
        def to(self, _):
            return self
    class Residues:
        keys = proteins
    class Config:
        data = type("Data", (dict,), {"protein_residue_embeds_path": "ignored"})()
    monkeypatch.setattr(parallel, "catalog_ids", lambda _: (proteins, ["r"]))
    monkeypatch.setattr(parallel.serial, "load_config", lambda _: Config())
    monkeypatch.setattr(parallel.serial.ProteinPooledLitModule, "load_from_checkpoint", lambda *a, **k: Module())
    monkeypatch.setattr(parallel.serial, "ResidueEmbedDataset", lambda *a, **k: Residues())
    calls = []
    def encode(module, dataset, device, batch_size, store_on_device):
        calls.append(dataset.keys)
        assert batch_size == 2 and not store_on_device
        return torch.tensor([[float(p[1:])] * 512 for p in dataset.keys])
    monkeypatch.setattr(parallel.serial, "encode_targets", encode)
    # Reaction encoding is independent and already committed in this fixture.
    parallel.save_tensor(tmp_path / "reactions.pt", "sig", 0, 1, torch.zeros(1, 512))
    args = argparse.Namespace(encoding_chunk_size=4, target_batch_size=2, config="x", checkpoint="x")
    for rank in range(4):
        parallel.encode_worker(args, tmp_path, "sig", rank, 4, "cpu")
    assert sorted(p for batch in calls for p in batch) == proteins
    assert len(calls) == 3
    for rank in range(4):
        parallel.encode_worker(args, tmp_path, "sig", rank, 4, "cpu")
    assert len(calls) == 3


def test_gpu_guard_only_allows_verified_old_test(monkeypatch):
    def output(command, **kwargs):
        if "--query-gpu=index,uuid,memory.free" in command:
            return "0, GPU-a, 100000\n1, GPU-b, 100000\n"
        return "GPU-a, 77\n"
    monkeypatch.setattr(parallel.subprocess, "check_output", output)
    parallel.check_gpus(["0", "1"], {77})
    parallel.check_gpus(["1"])
    with pytest.raises(RuntimeError, match="occupied"):
        parallel.check_gpus(["0", "1"])
    with pytest.raises(ValueError, match="distinct"):
        parallel.check_gpus(["0", "0"])
    with pytest.raises(ValueError, match="does not exist"):
        parallel.check_gpus(["2"])


def test_nan_score_cannot_be_merged(tmp_path):
    value = {k: 0.1 for k in parallel.METRICS}
    value.update(queries=1, candidates=2, mrr=float("nan"))
    path = parallel.score_path(tmp_path, "enzyme_to_reaction", 0, 1)
    parallel.atomic_json(path, dict(signature="sig", direction="enzyme_to_reaction", start=0, stop=1, metrics=value))
    with pytest.raises(ValueError, match="Invalid score"):
        parallel.merge_scores(tmp_path, "sig", "enzyme_to_reaction", 1, 2, 1)


def test_scoring_workers_end_to_end_on_cpu(tmp_path):
    torch.manual_seed(3)
    proteins = [f"p{i}" for i in range(9)]
    reactions = [f"r{i}" for i in range(5)]
    for name, ids in (("proteins", proteins), ("reactions", reactions)):
        (tmp_path / f"{name}.txt").write_text("\n".join(ids) + "\n")
    (tmp_path / "gold.csv").write_text("reaction_id,protein_id\n" + "".join(
        f"{r},{p}\n" for i, p in enumerate(proteins) for r in {reactions[i % 5], reactions[(i + 2) % 5]}))
    pvalues, rvalues = torch.randn(9, 512), torch.randn(5, 512)
    pvalues[1] = pvalues[0]
    np.save(tmp_path / "proteins.npy", pvalues.numpy())
    parallel.save_tensor(tmp_path / "reactions.pt", "sig", 0, 5, rvalues)
    args = argparse.Namespace(protein_candidates=tmp_path / "proteins.txt",
                              reaction_candidates=tmp_path / "reactions.txt",
                              reaction_query_ids=tmp_path / "reactions.txt",
                              enzyme_query_ids=tmp_path / "proteins.txt", pairs=tmp_path / "gold.csv",
                              protocol="reaction_holdout_clustered", catalog_on_gpu=True,
                              batch_size=2, candidate_chunk_size=3, score_block_size=4)
    for rank in range(4):
        parallel.score_worker(args, tmp_path, "sig", rank, 4, "cpu")
    truth_r, truth_e = parallel.serial.query_truth(args.pairs, set(reactions), set(proteins))
    for direction, ids, queries, candidates, truth, cids in (
        (parallel.DIRECTIONS[0], reactions, rvalues, pvalues, truth_r, proteins),
        (parallel.DIRECTIONS[1], proteins, pvalues, rvalues, truth_e, reactions),
    ):
        expected = parallel.serial.evaluate_direction(queries, candidates, ids, truth,
                    {k: i for i, k in enumerate(cids)}, batch_size=2, chunk_size=3, device="cpu", label="serial")
        actual = parallel.merge_scores(tmp_path, "sig", direction, len(ids), len(cids), 4)
        assert actual == pytest.approx(expected, abs=1e-12)
    # Finished score chunks are reused; corrupting a count is not silently accepted.
    path = parallel.score_path(tmp_path, parallel.DIRECTIONS[0], 0, 4)
    data = json.loads(path.read_text())
    data["metrics"]["queries"] = 3
    parallel.atomic_json(path, data)
    with pytest.raises(ValueError, match="Invalid score"):
        parallel.score_worker(args, tmp_path, "sig", 0, 4, "cpu")


def test_legacy_stop_matches_only_exact_output_and_checkpoint(tmp_path, monkeypatch):
    output, checkpoint = tmp_path / "old.json", tmp_path / "model.ckpt"
    evaluator = tmp_path / "evaluate_horizyn1_circe_v2.py"
    evaluator.write_text("import time; print('ready',flush=True); time.sleep(60)\n")
    monkeypatch.setattr(parallel.serial, "__file__", str(evaluator))
    def spawn(path):
        process = subprocess.Popen([sys.executable, "-u", str(evaluator), "--output", str(path),
                                    "--checkpoint", str(checkpoint)], stdout=subprocess.PIPE, text=True)
        assert process.stdout.readline().strip() == "ready"
        return process
    target, decoy = spawn(output), spawn(tmp_path / "unrelated.json")
    try:
        matches = parallel.legacy_processes(output, checkpoint)
        assert set(matches) == {target.pid}
        with pytest.raises(RuntimeError, match="different checkpoint"):
            parallel.legacy_processes(output, tmp_path / "wrong.ckpt")
        parallel.stop_verified(matches, output, checkpoint)
        target.wait(timeout=5)
        assert decoy.poll() is None
    finally:
        for child in (target, decoy):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.mark.parametrize("prefix", [
    ["tmux", "new-session", "python"],
    ["flock", "lock", "python"],
    ["bash", "-c", "python"],
    ["env", "python"],
    [sys.executable, "-c", "print('not an evaluator')"],
    [sys.executable, "-m", "some_launcher"],
    [sys.executable, "different.py"],
])
def test_script_operand_rejects_command_mentions(tmp_path, prefix):
    evaluator = tmp_path / "evaluate.py"
    assert parallel.script_arguments(prefix + [str(evaluator), "--output", "x"],
                                      sys.executable, tmp_path, evaluator) is None


@pytest.mark.parametrize("options", [[], ["-u"], ["-B", "-u"], ["--"]])
def test_script_operand_accepts_actual_python_invocation(tmp_path, options):
    evaluator = tmp_path / "evaluate.py"
    assert parallel.script_arguments([sys.executable, *options, "evaluate.py", "--output", "x"],
                                      sys.executable, tmp_path, evaluator) == ["--output", "x"]
    assert parallel.script_arguments([sys.executable, *options, str(evaluator)],
                                      "/bin/bash", tmp_path, evaluator) is None


def test_launcher_reference_and_its_child_are_not_stop_targets(tmp_path):
    output, checkpoint = tmp_path / "old.json", tmp_path / "model.ckpt"
    code = ("import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            "print(p.pid,flush=True); p.wait()")
    parent = subprocess.Popen([sys.executable, "-c", code, "tmux-launcher",
                               str(parallel.serial.__file__), "--output", str(output),
                               "--checkpoint", str(checkpoint)], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        assert parallel.legacy_processes(output, checkpoint) == {}
        assert parent.poll() is None
        assert parallel.read_process(child) is not None
    finally:
        os.kill(child, parallel.signal.SIGTERM)  # Only the fixture-owned child.
        parent.wait(timeout=5)


def test_current_controller_and_ancestors_are_protected(monkeypatch, tmp_path):
    protected = parallel.protected_processes()
    assert os.getpid() in protected and os.getppid() in protected
    monkeypatch.setattr(parallel, "process_arguments", lambda *a: pytest.fail("Must not inspect protected targets"))
    for pid in protected:
        assert parallel.verify_legacy_process(pid, tmp_path / "out", tmp_path / "ckpt") is None


def test_stop_revalidates_identity_immediately_before_signal(tmp_path, monkeypatch):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        process = parallel.read_process(child.pid)
        monkeypatch.setattr(parallel, "verify_legacy_process", lambda *a: None)
        monkeypatch.setattr(parallel, "send_signal", lambda *a: pytest.fail("No signal permitted"))
        with pytest.raises(RuntimeError, match="identity changed"):
            parallel.stop_verified({child.pid: process}, tmp_path / "out", tmp_path / "ckpt")
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_existing_parallel_controller_blocks_new_launch(tmp_path, monkeypatch):
    script = tmp_path / "parallel.py"
    script.write_text("import time; print('ready',flush=True); time.sleep(60)\n")
    monkeypatch.setattr(parallel, "__file__", str(script))
    child = subprocess.Popen([sys.executable, "-u", str(script), "run"], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="Another parallel test controller"):
            parallel.refuse_other_controllers()
    finally:
        child.terminate()
        child.wait(timeout=5)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is unavailable")
def test_real_tmux_server_and_other_session_survive_legacy_stop(tmp_path, monkeypatch):
    # A dedicated socket isolates this regression test from every user session.
    socket = f"circe-stop-audit-{os.getpid()}-{tmp_path.name}"
    command = ["tmux", "-L", socket, "-f", "/dev/null"]
    evaluator = tmp_path / "evaluate_horizyn1_circe_v2.py"
    evaluator.write_text("import time; time.sleep(60)\n")
    monkeypatch.setattr(parallel.serial, "__file__", str(evaluator))
    output, checkpoint = tmp_path / "old.json", tmp_path / "model.ckpt"
    try:
        subprocess.run([*command, "new-session", "-d", "-s", "old", "-c", str(tmp_path),
                        sys.executable, "-u", str(evaluator), "--output", str(output),
                        "--checkpoint", str(checkpoint)], check=True)
        subprocess.run([*command, "new-session", "-d", "-s", "replacement", "-c", str(tmp_path),
                        sys.executable, "-u", str(evaluator), "--output", str(tmp_path / "new.json")], check=True)
        matches = parallel.legacy_processes(output, checkpoint)
        assert len(matches) == 1
        parallel.stop_verified(matches, output, checkpoint)
        subprocess.run([*command, "has-session", "-t", "=replacement"], check=True)
    finally:
        subprocess.run([*command, "kill-server"], check=False, capture_output=True)


def test_failed_worker_is_reported_without_committing_results(tmp_path, monkeypatch):
    script = tmp_path / "fail.py"
    script.write_text("raise SystemExit(3)\n")
    monkeypatch.setattr(parallel, "__file__", str(script))
    job = tmp_path / "job.json"
    job.write_text("{}")
    with pytest.raises(RuntimeError, match="worker failed"):
        parallel.run_workers(job, "encode", ["0"])
    assert (tmp_path / "encode.gpu0.log").exists()
    assert not list(tmp_path.glob("proteins-*.pt"))
