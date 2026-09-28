"""Small synthetic checks; never read production embeddings or use a GPU."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/rebuild_reactzyme_prott5_cache.py"
spec = importlib.util.spec_from_file_location("rebuild_cache", SCRIPT)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def arguments(root):
    return SimpleNamespace(run_root=root / "new_cache", gpus=["0", "1"],
                           python=Path(sys.executable), model=root / "model", expected_proteins=3,
                           batch_size=256, max_tokens=65536, full_scan=False, expected_host="slurm-node-014")


def make_sources(root):
    for split in cache.PROTOCOLS:
        candidate_root = root / "data/revised_protocols/reactzyme_paper" / split
        candidate_root.mkdir(parents=True)
        for part, pid in zip(cache.PARTS, ("p1", "p2", "p3")):
            (candidate_root / f"{part}_candidate_ids.txt").write_text(pid + "\n")
        fasta = root / "data/revised_protocols/reactzyme_official" / split / "proteins.fasta"
        fasta.parent.mkdir(parents=True)
        fasta.write_text(">p1\nAAAA\n>p2\nCCCCCC\n>p3\nMMMMMMMM\n>unrelated\nVV\n")


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = arguments(self.root)
        self.args.run_root.mkdir()
        make_sources(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_union_no_extra_proteins(self):
        records, groups, sources = cache.collect_sequences(self.root)
        self.assertEqual(set(records), {"p1", "p2", "p3"})
        self.assertEqual(len(groups), 9)
        self.assertEqual(len(sources), 12)

    def test_conflicting_ids_fail_before_normalization(self):
        fasta = self.root / "data/revised_protocols/reactzyme_official/time/proteins.fasta"
        fasta.write_text(">p1\nAAAB\n>p2\nCCCCCC\n>p3\nMMMMMMMM\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            cache.collect_sequences(self.root)

    def test_missing_sequence_fails(self):
        for split in cache.PROTOCOLS:
            path = self.root / "data/revised_protocols/reactzyme_official" / split / "proteins.fasta"
            path.write_text(">p1\nAAAA\n>p2\nCCCCCC\n")
        with self.assertRaisesRegex(ValueError, "Missing 1"):
            cache.collect_sequences(self.root)

    def test_duplicate_candidate_id_fails(self):
        path = self.root / "data/revised_protocols/reactzyme_paper/time/train_candidate_ids.txt"
        path.write_text("p1\np1\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cache.collect_sequences(self.root)

    def test_prepare_resume_and_changed_source(self):
        with patch.object(cache, "ROOT", self.root):
            report = cache.prepare(self.args)
            self.assertEqual(report["fp16_payload_bytes"], 18 * 1024 * 2)
            self.assertEqual(cache.prepare(self.args), report)
            source = self.root / "data/revised_protocols/reactzyme_paper/time/train_candidate_ids.txt"
            source.write_text("p1\np2\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                cache.prepare(self.args)

    def test_modified_prepared_sequence_is_rejected(self):
        with patch.object(cache, "ROOT", self.root):
            cache.prepare(self.args)
            (self.args.run_root / "proteins.fasta").write_text(">p1\nXXXX\n")
            with self.assertRaisesRegex(ValueError, "Prepared file changed"):
                cache.prepare(self.args)

    def test_pilot_is_unique_and_contains_longest_per_rank(self):
        records = {f"p{i:05}": "M" * (20 + i) for i in range(2000)}
        ids = cache.pilot_ids(records, 4)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(len(records[key]) >= 1022 for key in ids[:512]))
        self.assertIn("p00000", ids)

    def test_empty_fasta_sequence_rejected(self):
        path = self.root / "bad.fasta"
        path.write_text(">p1\n>p2\nM\n")
        with self.assertRaisesRegex(ValueError, "Empty sequence"):
            list(cache.fasta_records(path))

    def test_storage_probe_and_space_gate(self):
        cache.storage_probe(self.args)
        self.assertFalse(list(self.args.run_root.glob("storage_probe_*")))
        cache.atomic_json(self.args.run_root / "preparation.json", {"fp16_payload_bytes": 1024})
        with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=1)):
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                cache.storage_probe(self.args)

    def test_extraction_command_never_force_or_delete(self):
        for tag in ("pilot", "full"):
            command = cache.extraction_command(self.args, tag, rank=0)
            self.assertIn("--resume", command)
            self.assertIn("--padded-token-budget", command)
            self.assertNotIn("--force", command)
            self.assertNotIn("--cleanup-shards", command)
            merge = cache.extraction_command(self.args, tag)
            self.assertIn("--merge-only", merge)
            self.assertEqual(merge[merge.index("--merge-storage") + 1], "virtual")

    def test_gpu_check_rejects_occupied_devices(self):
        with patch.object(cache.subprocess, "run", return_value=SimpleNamespace(stdout="1234\n")):
            with self.assertRaisesRegex(RuntimeError, "compute processes"):
                cache.gpu_check(self.args)

    def test_gpu_check_requires_sufficient_memory(self):
        responses = [SimpleNamespace(stdout=""), SimpleNamespace(stdout="0, NVIDIA H200 NVL, 100000\n1, NVIDIA H200 NVL, 140000\n")]
        with patch.object(cache.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(RuntimeError, "125 GiB"):
                cache.gpu_check(self.args)

    def test_wrong_host_does_not_create_controller_lock(self):
        with patch.object(cache.socket, "gethostname", return_value="slurm-node-013"):
            with self.assertRaisesRegex(RuntimeError, "slurm-node-014"):
                cache.run(self.args)
        self.assertFalse((self.args.run_root / ".controller.lock").exists())

    def test_runner_failure_stops_owned_children(self):
        (self.args.run_root / "logs").mkdir()
        runner = cache.Runner(self.args)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            runner.stage("bad", [("", [sys.executable, "-c", "raise SystemExit(3)"], {})], timeout=5)
        self.assertTrue(all(child.poll() is not None for child in runner.children))

    def test_runner_timeout_stops_owned_children(self):
        (self.args.run_root / "logs").mkdir()
        runner = cache.Runner(self.args)
        with self.assertRaises(TimeoutError):
            runner.stage("timeout", [("", [sys.executable, "-c", "import time; time.sleep(20)"], {})], timeout=0.1)
        self.assertTrue(all(child.poll() is not None for child in runner.children))


try:
    import h5py
    import numpy as np
    import torch
    HAVE_ML = True
except ImportError:
    HAVE_ML = False


@unittest.skipUnless(HAVE_ML, "Run with the ML environment for real HDF5/reader tests")
class HDFValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.args = arguments(self.root)
        self.args.run_root.mkdir()
        self.records = [("p1", "MMMM"), ("p2", "AAA"), ("p3", "CCC")]
        cache.write_fasta(self.args.run_root / "pilot.fasta", self.records)
        self.output, self.shards = cache.paths(self.args, "pilot")
        self.shards.mkdir(parents=True)
        all_ids, offsets, base = [], [0], 0
        layout = h5py.VirtualLayout(shape=(10, 1024), dtype=np.float16)
        self.shard_paths = []
        for rank in range(2):
            path = self.shards / f"proteins_prott5_residue.shard{rank:02d}-of-02.h5"
            expected = sorted([(i, key, len(seq)) for i, (key, seq) in enumerate(self.records) if i % 2 == rank], key=lambda row: (row[2], row[0]))
            rank_offsets = np.concatenate(([0], np.cumsum([row[2] for row in expected])))
            with h5py.File(path, "w") as handle:
                attrs = dict(model_name=str(self.args.model), max_sequence_length=1022,
                             sequence_truncation="ends_center", rank=rank, world_size=2,
                             processed_count=len(expected), source_fasta=str(self.args.run_root / "pilot.fasta"),
                             length_sort=True, padded_token_budget=True)
                handle.attrs.update(attrs)
                handle["ids"] = np.asarray([row[1] for row in expected], dtype=h5py.string_dtype())
                handle["indices"] = [row[0] for row in expected]
                handle["offsets"] = rank_offsets
                handle["vectors"] = np.full((int(rank_offsets[-1]), 1024), rank + 1, dtype=np.float16)
            size = int(rank_offsets[-1])
            layout[base:base + size] = h5py.VirtualSource(str(path), "vectors", shape=(size, 1024))
            base += size
            all_ids.extend(row[1] for row in expected)
            for row in expected:
                offsets.append(offsets[-1] + row[2])
            self.shard_paths.append(path)
        with h5py.File(self.output, "w") as handle:
            handle["ids"] = np.asarray(all_ids, dtype=h5py.string_dtype())
            handle["offsets"] = offsets
            handle.create_virtual_dataset("vectors", layout)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_virtual_cache_actual_training_reader(self):
        cache.verify(self.args, "pilot")
        report = json.loads((self.args.run_root / "pilot_validation.json").read_text())
        self.assertEqual(report["proteins"], 3)
        self.assertEqual(report["sampled_training_reads"], 3)
        self.assertTrue(report["full_stored_finite_scan"])

    def test_missing_vds_source_is_not_accepted_as_zeros(self):
        self.shard_paths[0].unlink()
        with self.assertRaises((FileNotFoundError, OSError)):
            cache.verify(self.args, "pilot")

    def test_incomplete_shard_rejected(self):
        with h5py.File(self.shard_paths[0], "r+") as handle:
            handle.attrs["processed_count"] = 0
        with self.assertRaisesRegex(ValueError, "processed_count"):
            cache.verify(self.args, "pilot")

    def test_wrong_model_rejected(self):
        with h5py.File(self.shard_paths[0], "r+") as handle:
            handle.attrs["model_name"] = "different-backbone"
        with self.assertRaisesRegex(ValueError, "model_name"):
            cache.verify(self.args, "pilot")

    def test_wrong_lengths_rejected(self):
        with h5py.File(self.output, "r+") as handle:
            handle["offsets"][1] = 1
        with self.assertRaisesRegex(ValueError, "lengths/offsets"):
            cache.verify(self.args, "pilot")

    def test_nonfinite_stored_values_rejected(self):
        with h5py.File(self.shard_paths[0], "r+") as handle:
            handle["vectors"][1, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            cache.verify(self.args, "pilot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
