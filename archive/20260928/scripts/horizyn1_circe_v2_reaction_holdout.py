#!/usr/bin/env python3
"""CIRCE-v2 using the existing Horizyn-style graph and a reaction-held-out test.

Reuses 80% representatives and all collapsed positives. No new protein MMseqs
run. Uses paper ECFP6 equality groups and chemical forward/reverse augmentation.
Native annotations are reused; only cached weak profiles are scoped to train.
The model, 85/15 sampler, disabled auxiliary loss and H200 options are unchanged.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import horizyn1_circe_v2_pipeline as base

PROTOCOL = "horizyn80_clustered_reaction_holdout_v1"
IMPLEMENTATIONS = [Path(__file__).resolve(), ROOT / "scripts/prepare_horizyn1_reaction_holdout.py",
                   ROOT / "scripts/project_horizyn1_training_labels.py",
                   ROOT / "horizyn/chemistry/standardizer.py"]


class Pipeline(base.Pipeline):
    def work_directory(self, *, check_space=True):
        # There is no preparation SQLite database, and no compatibility marker
        # that could let the old both-cold preparation masquerade as this split.
        return self.run / "work/reaction_holdout"

    def step(self, name, inputs, outputs, action, parameters=None):
        parameters = {**(parameters or {}), "dataset_protocol": PROTOCOL,
                      "validation_fraction": self.args.validation_fraction,
                      "test_fraction": self.args.test_fraction}
        if name == "prepare" and not (self.state / "prepare.json").exists() and any(self.data.iterdir()):
            raise RuntimeError("Use a new RUN_ROOT; existing preparation data must not be overwritten")
        return super().step(name, [*inputs, *IMPLEMENTATIONS], outputs, action, parameters)

    def preflight(self):
        cache = self.source / "annotations/circe_v2_cofactor_v2"
        required = [cache / name for name in ("label_manifest.json", "enzyme_ec_labels.csv",
                    "enzyme_biofp_targets.npz", "candidate_eligibility.csv", "enzyme_biofp_vocab.json")]
        for path in required:
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing already computed CIRCE-v2 annotation cache: {path}")
        # Preserve all existing model/environment/storage checks. The inherited
        # check merely verifies the installed MMseqs binary; we never execute it.
        super().preflight()
        marker = self.state / "preflight.json"
        value = json.loads(marker.read_text())
        value["required_inputs"].extend(base.signature(p) for p in [*required, *IMPLEMENTATIONS])
        value["dataset_protocol"] = PROTOCOL
        base.atomic_json(marker, value)
        self.log("Using cached Horizyn80 graph + reaction holdout; no protein clustering/alignment will run")

    def prepare(self):
        self.require_stage("preflight")
        inputs = [self.source / name for name in ("clustered/clustered_manifest.json",
                  "clustered/proteins.fasta", "clustered/pairs.tsv", "raw/raw_pairs.tsv", "raw/raw_reactions.tsv")]
        def build():
            self.command([self.setup_python, ROOT / "scripts/prepare_horizyn1_reaction_holdout.py",
                          "--source", self.source, "--output", self.data, "--seed", self.args.seed,
                          "--validation-fraction", self.args.validation_fraction,
                          "--test-fraction", self.args.test_fraction], "prepare")
        self.step("prepare", inputs, [self.data / "preparation_manifest.json"],
                  lambda: self.rebuild_directory(self.data, build))

    def labels(self):
        self.require_stage("prepare")
        cache = self.source / "annotations/circe_v2_cofactor_v2"
        inputs = [self.data / name for name in ("preparation_manifest.json", "reaction_split.json",
                  "train_own_raw_associations.csv")]
        inputs += [cache / name for name in ("label_manifest.json", "enzyme_biofp_targets.npz",
                   "enzyme_ec_labels.csv", "candidate_eligibility.csv", "enzyme_biofp_vocab.json")]
        inputs.append(self.source / "annotations/cofactor_v2/reaction_annotations.json")
        def build():
            self.command([self.setup_python, ROOT / "scripts/project_horizyn1_training_labels.py",
                          "--source", self.source, "--prepared", self.data,
                          "--output", self.data / "labels"], "labels")
        self.step("labels", inputs, [self.data / "labels/label_manifest.json"],
                  lambda: self.rebuild_directory(self.data / "labels", build))

    def config(self, output, *, pilot=None):
        import yaml
        # Let the existing controller resolve all CIRCE/H200 settings, then
        # change only data paths/protocol before the immutable publication.
        with tempfile.TemporaryDirectory(prefix=".reaction-config-", dir=self.run) as temporary:
            path = Path(temporary) / "train.yaml"
            super().config(path, pilot=pilot)
            value = yaml.safe_load(path.read_text())
        value["data"]["dataset_protocol"] = PROTOCOL
        value["data"]["direction_augmentation"] = "materialized_forward_and_reverse"
        if pilot is None:
            panel = self.data / "panels/validation_reaction_cold"
            value["data"]["validation_pairs_path"] = str(panel) + "_pairs.csv"
            for section in ("data", "training"):
                value[section]["validation_retrieval_candidate_ids_path"] = str(panel) + "_protein_ids.txt"
                value[section]["validation_retrieval_query_ids_path"] = str(panel) + "_query_ids.json"
        serialized = yaml.safe_dump(value, sort_keys=False)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and output.read_text() != serialized:
            raise RuntimeError(f"Training configuration changed at {output}; use a new RUN_ROOT")
        if not output.exists():
            output.write_text(serialized)

    def test(self):
        self.require_stage("train")
        self.require_free_gpus()
        selected = self.args.test_checkpoint or json.loads((self.state / "train.json").read_text())["best_checkpoint"]["path"]
        self.command([self.python, ROOT / "scripts/evaluate_horizyn1_circe_v2.py",
            "--checkpoint", selected, "--config", self.run / "configs/train.yaml",
            "--pairs", self.data / "test_query_gold.csv", "--reactions", self.data / "reactions.csv",
            "--protein-candidates", self.data / "all_candidate_ids.txt",
            "--reaction-candidates", self.data / "test_reaction_ids.txt",
            "--enzyme-query-ids", self.data / "test_enzyme_query_ids.txt",
            "--reaction-query-ids", self.data / "test_reaction_query_ids.txt",
            "--protocol", "reaction_holdout_clustered", "--output", self.run / "results/test.json",
            "--embedding-cache", self.run / "results/embedding_cache.pt",
            "--batch-size", "128", "--candidate-chunk-size", "32768", "--device", "cuda"], "test",
            env={"CUDA_VISIBLE_DEVICES": self.gpus[0]})
        base.atomic_json(self.state / "test.json", {"status": "complete", "dataset_protocol": PROTOCOL,
                         "report": base.signature(self.run / "results/test.json")})


def parse_args(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--validation-fraction", type=float, default=.05)
    parser.add_argument("--test-fraction", type=float, default=.05)
    specific, remaining = parser.parse_known_args(argv)
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        parser.print_help()
    args = base.parse_args(remaining)
    if not any(x == "--run-root" or x.startswith("--run-root=") for x in remaining) and not os.environ.get("RUN_ROOT"):
        args.run_root = str(ROOT / "runs" / f"horizyn1_circe_v2_reaction_holdout_{args.profile}")
    args.validation_fraction, args.test_fraction = specific.validation_fraction, specific.test_fraction
    if not (0 < args.validation_fraction < 1 and 0 < args.test_fraction < 1
            and args.validation_fraction + args.test_fraction < 1):
        parser.error("Validation/test fractions must be positive and sum to less than one")
    return args


def main(argv=None):
    args = parse_args(argv)
    pipeline = Pipeline(args)
    with (pipeline.run / "pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another pipeline holds this RUN_ROOT lock")
        # Refuse using an old run even when only a later stage was requested.
        binding = pipeline.state / "dataset_protocol.json"
        if not binding.exists() and any(pipeline.state.glob("*.json")):
            raise SystemExit("Existing run has no reaction-holdout binding; use a NEW RUN_ROOT")
        protocol = {"dataset_protocol": PROTOCOL, "validation_fraction": args.validation_fraction,
                    "test_fraction": args.test_fraction, "seed": args.seed}
        if binding.exists() and json.loads(binding.read_text()) != protocol:
            raise SystemExit("Reaction split parameters changed; use a NEW RUN_ROOT")
        if not binding.exists():
            base.atomic_json(binding, protocol)
        (pipeline.run / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, pipeline.terminate)
        for stage in base.STAGES if args.stage == "all" else [args.stage]:
            (pipeline.run / "current_stage.txt").write_text(stage + "\n")
            getattr(pipeline, "features_stage" if stage == "features" else stage)()
        pipeline.log(f"Requested stage '{args.stage}' completed")


if __name__ == "__main__":
    main()
