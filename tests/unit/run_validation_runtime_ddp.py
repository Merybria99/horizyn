"""Explicit small real-model DDP verification; never touches production data.

Run with --accelerator gpu --devices 4 on verified free GPUs. --fit additionally
checks that training resumes after validation and checkpoint selection works.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightning.pytorch as pl
import torch
from test_validation_runtime import make_validation_fixture
from horizyn.training_checkpoints import recovery_checkpoint


class Capture(pl.Callback):
    def __init__(self):
        self.checks = []

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.retrieval_encoder_calls = [0, 0]

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.checks.append({
                "metrics": {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("val/")},
                "local_encoder_rows": pl_module.retrieval_encoder_calls.copy(),
                "global_report": getattr(pl_module, "fast_validation_report", None),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "baseline"), required=True)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--degenerate-validation", action="store_true",
                        help="All validation pairs share one reaction: no contrastive negatives")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    fixture_dir = args.output / f"fixture_rank{rank}"
    fixture_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(2)
    data, module = make_validation_fixture(
        fixture_dir, args.mode == "fast",
        model_options={"contrastive_fp32": True, "fp32_sensitive_modules": True})
    if args.degenerate_validation:
        with data.test_pairs_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        destination = fixture_dir / "validation_star_pairs.csv"
        with destination.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pr_id", "reaction_id", "protein_id"])
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row, reaction_id=rows[0]["reaction_id"]))
        data.test_pairs_path = destination
    for index, name in enumerate(("_encode_target_batch", "_encode_query_batch")):
        original = getattr(module, name)
        def counted(batch, retrieval_direction=None, _original=original, _index=index):
            result = (_original(batch) if retrieval_direction is None
                      else _original(batch, retrieval_direction=retrieval_direction))
            module.retrieval_encoder_calls[_index] += len(result)
            return result
        setattr(module, name, counted)
    capture = Capture()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=args.output / "checkpoints", monitor="val/mean_bidirectional_mrr",
        mode="max", save_top_k=1, save_last=True)
    recovery = recovery_checkpoint({"checkpoint_dir": str(args.output / "checkpoints"),
                                    "recovery_every_n_train_steps": 1})
    trainer = pl.Trainer(
        accelerator=args.accelerator, devices=args.devices,
        strategy="ddp_find_unused_parameters_true", logger=False,
        enable_checkpointing=args.fit, callbacks=[capture, checkpoint, recovery] if args.fit else [capture],
        enable_progress_bar=False, use_distributed_sampler=False,
        precision="bf16-mixed" if args.accelerator == "gpu" else "32-true",
        max_steps=2, max_epochs=2, num_sanity_val_steps=0, check_val_every_n_epoch=1,
        deterministic=True)
    if args.fit:
        data.train_batch_size = 20  # Exact 85/15 sampler; one batch per epoch.
        trainer.fit(module, datamodule=data)
        assert trainer.global_step == 2 and len(capture.checks) == 2
        assert checkpoint.best_model_path and Path(checkpoint.best_model_path).is_file()
        assert Path(recovery.last_model_path).is_file()
    else:
        trainer.validate(module, datamodule=data, verbose=False)
        assert len(capture.checks) == 1
    if args.degenerate_validation:
        for check in capture.checks:
            assert check["metrics"]["val/contrastive_batch_valid_fraction"] == 0.0
            assert "val/loss" not in check["metrics"]
            assert "val/mean_bidirectional_mrr" in check["metrics"]
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().float().numpy().tobytes())
    result = dict(mode=args.mode, rank=trainer.global_rank, world_size=trainer.world_size,
                  checks=capture.checks, model_sha256=digest.hexdigest(),
                  cpu_rng_sha256=hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
                  global_step=trainer.global_step, best_checkpoint=checkpoint.best_model_path)
    (args.output / f"result_rank{trainer.global_rank}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
