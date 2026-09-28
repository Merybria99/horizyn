#!/usr/bin/env python3
"""Isolated fast-I/O training or bounded baseline/fast throughput benchmark.

Uses the original training entrypoint, model, optimizer and sampler. Outputs
must go to a new directory, never into a running pipeline's output paths.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from contextlib import nullcontext

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import lightning.pytorch as pl
import torch

import train_protein_pooling as training
from horizyn.config import load_config
from horizyn.training_io import training_io_adapters


class StepTiming(pl.Callback):
    """Per-rank wall-clock timings including inter-batch data waits."""
    def __init__(self, output, warmup=5):
        self.output = Path(output)
        self.warmup = warmup
        self.samples = []
        self.previous_end = None

    def on_train_start(self, trainer, pl_module):
        if pl_module.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(pl_module.device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.start = time.perf_counter()
        self.wait = None if self.previous_end is None else self.start - self.previous_end

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is None or not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError("Benchmark produced a missing/non-finite training loss")
        if pl_module.device.type == "cuda":
            torch.cuda.synchronize(pl_module.device)
        end = time.perf_counter()
        self.samples.append(dict(step=int(trainer.global_step),
                                 compute_seconds=end - self.start,
                                 between_batches_seconds=self.wait,
                                 total_seconds=None if self.wait is None else self.wait + end - self.start))
        self.previous_end = end
        if len(self.samples) % 10 == 0:
            self.report(trainer, pl_module, complete=False)

    def report(self, trainer, pl_module, complete):
        measured = self.samples[self.warmup:]
        totals = [row["total_seconds"] for row in measured if row["total_seconds"] is not None]
        classification = {}
        for name in ("weighted_biofp", "biofp_mechanism_active", "biofp_cofactor_active"):
            for suffix in ("_epoch", "_step", ""):
                value = trainer.callback_metrics.get(f"train/loss_{name}{suffix}")
                if value is not None:
                    classification[name] = float(value.detach().cpu())
                    break
        result = dict(complete=complete, rank=trainer.global_rank,
                      classification=classification,
                      world_size=trainer.world_size, warmup_steps=self.warmup,
                      measured_steps=len(totals), samples=self.samples,
                      mean_step_seconds=statistics.mean(totals) if totals else None,
                      median_step_seconds=statistics.median(totals) if totals else None,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(pl_module.device)
                      if pl_module.device.type == "cuda" else 0)
        path = self.output / f"timing_rank{trainer.global_rank}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(path)
        if trainer.is_global_zero and totals:
            print(f"I/O benchmark: {len(self.samples)} steps; post-warmup mean "
                  f"{result['mean_step_seconds']:.3f}s/step (includes data waits)", flush=True)

    def on_train_end(self, trainer, pl_module):
        self.report(trainer, pl_module, complete=True)


def output_overrides(args, config):
    output = Path(args.io_output_dir).resolve()
    for key in ("log_dir", "checkpoint_dir"):
        original = Path(config.logging[key]).resolve()
        if output == original or output.is_relative_to(original) or original.is_relative_to(output):
            raise ValueError(f"I/O output overlaps the source run's {key}: {original}")
    overrides = {"logging.log_dir": str(output / "logs"),
                 "logging.checkpoint_dir": str(output / "checkpoints")}
    recovery_interval = getattr(args, "io_recovery_every_n_train_steps", None)
    if recovery_interval is not None:
        overrides["logging.recovery_every_n_train_steps"] = recovery_interval
    if args.io_mode == "fast":
        if not config.data.get("indexed_pairs_dir"):
            raise ValueError("Fast-I/O launcher requires an indexed training graph")
    if args.io_mode in ("fast", "residue"):
        overrides["data.prefetch_factor"] = args.io_prefetch
    if args.io_benchmark_steps:
        if args.resume:
            raise ValueError("Benchmark from the common configured initialization, not --resume")
        if args.io_benchmark_steps <= 5:
            raise ValueError("Benchmark must exceed the five warmup steps")
        overrides.update({"training.max_steps": args.io_benchmark_steps,
                          "training.validation_enabled": False,
                          "training.validation_retrieval_metrics": False,
                          "training.early_stopping.enabled": False,
                          "logging.screen_selection_every_n_epochs": None,
                          "logging.log_every_n_steps": 10})
    return overrides


def main():
    parser = training.build_arg_parser()
    parser.add_argument("--io-mode", choices=("fast", "residue", "baseline"), default="fast",
                        help="residue preserves CSV sampling and original validation; only optimizes residue transport")
    parser.add_argument("--io-output-dir", required=True)
    parser.add_argument("--io-prefetch", type=int, default=4)
    parser.add_argument("--io-benchmark-steps", type=int, default=0)
    parser.add_argument("--io-recovery-every-n-train-steps", type=int, default=None,
                        help="Save resumable recovery checkpoints independently of validation")
    args, unknown = parser.parse_known_args()
    if args.io_prefetch < 1 or args.io_benchmark_steps < 0:
        parser.error("prefetch must be positive and benchmark steps nonnegative")
    if args.io_recovery_every_n_train_steps is not None and args.io_recovery_every_n_train_steps <= 0:
        parser.error("recovery checkpoint interval must be positive")
    if unknown:
        parser.error("Use the source YAML for model/training settings; unknown options: " + " ".join(unknown))
    source = load_config(args.config)
    overrides = output_overrides(args, source)
    output = Path(args.io_output_dir).resolve()
    # Lightning re-executes this wrapper for child ranks with the original argv.
    primary = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
    if primary:
        output.mkdir(parents=True, exist_ok=False)
        manifest = dict(source_config=str(Path(args.config).resolve()),
                        source_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                        mode=args.io_mode, overrides=overrides,
                        benchmark_steps=args.io_benchmark_steps,
                        code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (Path(__file__).resolve(), ROOT / "horizyn/training_io.py",
                                                  ROOT / "horizyn/datasets/indexed_pairs.py",
                                                  ROOT / "horizyn/validation_runtime.py",
                                                  ROOT / "horizyn/losses.py",
                                                  ROOT / "horizyn/protein_pooling_lightning_module.py",
                                                  ROOT / "horizyn/model.py",
                                                  ROOT / "horizyn/training_checkpoints.py",
                                                  ROOT / "horizyn/config.py",
                                                  ROOT / "horizyn/config_validation.py",
                                                  ROOT / "horizyn/training_options.py",
                                                  ROOT / "horizyn/training_runtime.py",
                                                  ROOT / "horizyn/training_warm_start.py",
                                                  ROOT / "scripts/train_protein_pooling.py")})
        (output / "io_run.json").write_text(json.dumps(manifest, indent=2) + "\n")

    original_parser = training.build_arg_parser
    original_loader = training.load_config
    original_trainer = pl.Trainer

    def configured_loader(path, overrides=None, **kwargs):
        merged = dict(overrides or {})
        merged.update(output_overrides(args, source))
        return original_loader(path, overrides=merged, **kwargs)

    class TimedTrainer(original_trainer):
        def __init__(self, **kwargs):
            if args.io_benchmark_steps:
                kwargs["callbacks"] = [callback for callback in kwargs["callbacks"]
                                       if not isinstance(callback, (pl.callbacks.ModelCheckpoint, pl.callbacks.EarlyStopping))]
                kwargs["callbacks"].append(StepTiming(output))
                kwargs["enable_checkpointing"] = False
            super().__init__(**kwargs)

    training.build_arg_parser = lambda: parser
    training.load_config = configured_loader
    pl.Trainer = TimedTrainer
    try:
        print(f"I/O mode={args.io_mode}; isolated output={output}; "
              "sampler/model/loss unchanged", flush=True)
        with training_io_adapters(training, residue_only=args.io_mode == "residue") if args.io_mode != "baseline" else nullcontext():
            training.main()
    finally:
        training.build_arg_parser = original_parser
        training.load_config = original_loader
        pl.Trainer = original_trainer


if __name__ == "__main__":
    main()
