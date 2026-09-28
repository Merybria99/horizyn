# CIRCE-v2: opt-in training I/O

`scripts/train_protein_pooling_fast_io.py` uses the existing training entrypoint
with process-local data adapters. It does not edit the existing run, its config,
the extraction code or any preparation/feature receipt.

## What changes

- Load the compact indexed graph into immutable RAM arrays once per DDP rank
  (714 MiB per rank for the current reaction-held-out dataset; 2 GiB limit).
  Fork workers share these pages. Spawn workers reopen read-only memmaps
  instead of receiving a serialized copy of the entire resident index.
- Keep **already FP16** HDF5 residues in FP16 during CPU loading, padding,
  worker IPC, pinning and device transfer. Restore FP32 on the destination
  device, before the model sees the batch. FP32 source stores are not quantized.
  Finite-value validation remains enabled.
- Retain each worker's HDF5 vector-dataset handle so the virtual dataset and
  its source-file caches can survive consecutive reads. Handles are discarded
  on serialization and reopened after a fork; no feature files are modified.
- Use prefetch factor 4 instead of 2; worker count remains as configured.
  With four workers, batch 100, length 1022 and width 1024, queued residue
  payload alone can reach about 3.1 GiB per rank, excluding bookkeeping and
  temporary/shared/pinned copies. This is host RAM, not GPU memory.

Batch size, RNG, per-epoch edge coverage, support-positive rows, the 85/15
positive/negative ratio, train-only annotation policy, masks, tower weights,
loss, optimizer and validation cadence are unchanged in normal training.
The model still receives FP32 residues; existing mixed-precision settings
still govern computation.

## Retrieval-validation optimization

The same `--io-mode fast` launcher now also enables
`horizyn/validation_runtime.py`; no additional flag or data extraction is needed.
The original training entrypoint and fingerprinted preparation files remain
unchanged. `--io-mode baseline` retains the original five validation loaders.

- Assign each original retrieval batch to exactly one GPU. Whole-batch
  ownership preserves encoder padding/batch composition, and never pads the
  catalog with duplicate queries. Ranks may own zero batches.
- Keep the validation-loss loader and its original contrastive batch/gather
  behavior unchanged. Run retrieval in the validation epoch-end hook; reduce
  completed candidate tables and metric sums/counts, not every query batch.
  Checkpoint and early-stopping metrics are logged before validation finishes.
- Reuse candidate encodings for the overlapping enzyme/reaction query sets
  **within that check**. Preserve both raw and normalized embeddings to avoid
  adding an extra normalization when reusing an anchor. Missing anchors are
  encoded separately. Direction-specific adapters disable incompatible reuse.
- Allocate new lookup tables each validation check. Nothing is reused across
  tower updates; final embeddings are not mistaken for frozen input features.
- Preserve chunked versus dense ranking behavior, candidate order, tie rules,
  and macro-query weighting. Empty ranks contribute zero sums/counts. Queries
  without an in-catalog positive remain excluded from metric denominators.
- Keep the legacy CPU loader-seed draws so a different number of I/O iterators
  does not perturb subsequent training randomness. Actual retrieval loaders
  use a separate generator. Encoding uses the original transfer and AMP hooks.

Diagnostic `val/*/num_queries` now reports the total valid query count rather
than an average batch/block count. Selection metrics and validation-loss keys
are unchanged. New `val/retrieval_seconds` and `val/total_seconds` metrics, plus
the `Sharded retrieval validation complete` log message, expose runtime and
encoder/reuse counts.

This implements the first two recommended validation changes. The optional
96.7-GiB raw-residue cache is **not** allocated or enabled: this implementation
only retains the much smaller embeddings for the duration of one check.
It does not reduce the validation frequency or candidate universe.

For an explicit small four-GPU integration check on **free** GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  ../env/bin/python tests/unit/run_validation_runtime_ddp.py \
  --mode fast --accelerator gpu --devices 4 --fit \
  --output runs/validation_runtime_fit_NEW
```

This uses synthetic HDF5/NPZ inputs and the actual multimodal factorized model,
runs two optimizer steps with two validation checks, and checks checkpoint
selection. It does not touch production data. Omit `--fit` for uneven-tail
and empty-rank validation only; `--mode baseline` permits a comparison with
the original path. Use a new output directory for each invocation.

### Validation verification, September 10, 2026

The selected regression suite passed **152 tests**, with one pre-existing
opt-in CUDA unit test skipped. Separate four-H200, bf16-mixed integration
runs compared the baseline and fast paths on the small real-model fixture:

- Uneven-tail validation completed with empty ranks; all four ranks agreed.
- Two optimizer steps, two validation checks and best-checkpoint saving
  completed in both modes. Final model and CPU RNG hashes were identical.
- The largest difference in validation loss/retrieval metrics was
  `5.960464477539063e-08`, excluding the intentional `num_queries` diagnostic
  change and the new timing metrics.
- Each check encoded 4 proteins and 4 reactions globally, versus 32 of each
  in the duplicated baseline. This fixture has complete query/catalog overlap;
  the reduction in encoded rows is **not** a wall-clock speedup measurement.

Per-rank results are in `runs/validation_runtime_check_fast_20260910/`,
`runs/validation_runtime_check_baseline_v2_20260910/`,
`runs/validation_runtime_fit_fast_20260910/`, and
`runs/validation_runtime_fit_baseline_20260910/`. Full production-catalog
validation runtime has not yet been measured with the new path. Production
processes were neither stopped nor restarted for these checks.

## Bounded comparison

Use a **new output directory for every invocation**, on GPUs verified free.
The output directory must not overlap source log or checkpoint directories.
No benchmark checkpoints are saved and no validation/test panels are evaluated.
The full graph and real residue/reaction features are used.

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  ../env/bin/python scripts/train_protein_pooling_fast_io.py \
  --config runs/horizyn1_circe_v2_reaction_holdout_h200_paired/configs/train.yaml \
  --io-mode fast --io-benchmark-steps 60 \
  --io-output-dir runs/io_benchmark_fast_NEW --wandb-mode disabled
```

Repeat with `--io-mode baseline` and a different output directory. Baseline
uses the source prefetch factor and original loaders. Both modes start from
the same configured initialization/seed; `--resume` is prohibited in benchmarks.
Compare all four `timing_rank*.json` reports. The first five steps are excluded;
timing includes inter-batch waits and synchronizes CUDA at batch end in both
modes. Startup is excluded. This synchronization itself adds overhead.
Filesystem-cache warmth, sequence lengths and concurrent NFS traffic affect
results; a short benchmark is not a full-epoch or validation-time guarantee.

### Measured on tyrosine, September 10, 2026

Real model, real indexed graph/features, four H200 GPUs, global batch 400.
The final comparison ran 20 steps per mode and excluded the first five.
Fast ran before baseline; previous diagnostic reads had warmed the same data.

| Post-warmup metric (rank 0; other ranks agree) | Baseline | Fast I/O |
| --- | ---: | ---: |
| Mean step time, including waits | 0.4572 s | 0.2155 s |
| Median step time | 0.2145 s | 0.1817 s |
| Logged loss at step 9 | 3.307337760925293 | 3.307337760925293 |
| Logged loss at step 19 | 2.6615285873413086 | 2.6615285873413086 |

The mean improvement is 2.12x; the median improvement is 1.18x. This small,
warm-cache sample does **not** support a 2.12x full-epoch claim. An earlier
version without persistent HDF5 dataset handles was stopped after its first
20 reported steps because shared-NFS waits dominated (~9.4 s/step); it is
not a completed or controlled baseline comparison. No production processes
were stopped and no feature files were rewritten.

Reports are in `runs/io_benchmark_20260910_cached_handles/` and
`runs/io_benchmark_20260910_baseline/`, including four per-rank timing files,
mode/code receipts and CSV metrics. The selected CPU test suite passed
79 tests; one explicitly opt-in CUDA unit test was skipped. The actual
four-GPU training comparisons above were run separately and both completed.

## Normal training / checkpoint continuation

Omit `--io-benchmark-steps` to retain the source training budget, validation and
checkpoint behavior. `--resume /absolute/path/to/last.ckpt` loads the original
Lightning training state, not just model weights. Outputs must still go to a
new directory. The original pipeline is **not** stopped, retargeted, or marked
complete by this standalone launcher; do not run a duplicate production job.
Keep its automatic test controller in mind when planning a switch.

Before switching a live run, verify a usable checkpoint exists and the old
controller/ranks have exited. Without a checkpoint, stopping loses training
progress. Lightning does not guarantee exact mid-epoch dataloader continuation,
and changing the checkpoint directory can reset the best-checkpoint bookkeeping.
Do not present such a restart as bitwise uninterrupted training.

`io_run.json` records source config hash, adapter/launcher hashes, mode and
overrides. No source YAML is rewritten. The CPU tests compare arrays, sampled
rows at several epochs/ranks, positive/negative masks, FP16 transport, FP32
fallback, finite guards, gradients and full multimodal optimizer steps.
