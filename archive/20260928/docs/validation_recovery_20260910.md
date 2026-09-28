# Validation and recovery repair (2026-09-10)

The Tyrosine run `horizyn1_circe_v2_reaction_holdout_tyrosine_fast_io`
failed during its first validation after step 2000. No checkpoint was present.
Its files are preserved. The replacement run is
`horizyn1_circe_v2_reaction_holdout_tyrosine_validation_fixed`, in detached
session `circe_v2_tyrosine` on `slurm-node-013`.

## Changes

- Sampled losses still reject batches without a supported positive/negative
  anchor. They now raise a specific `NoContrastiveAnchorsError` in that case.
- Evaluation alone catches that exception and excludes the undefined loss
  from the validation-loss average. It does not insert a zero, drop retrieval
  queries, invent negative labels, or suppress other validation errors.
- `val/contrastive_batch_valid_fraction` records the fraction of usable loss
  batches. If every batch is degenerate, there is no `val/loss`; retrieval MRR
  remains the selection metric for these runs.
- The DDP path makes this decision after gathering the global batch, so all
  ranks take the same branch. Training's bidirectional support checks remain
  strict, and the training loss/sampling/optimizer settings are unchanged.
- Independent, unmonitored full-state recovery checkpoints are saved on
  training-batch end, before validation. Only the newest recovery checkpoint
  is retained; `checkpoints/recovery/last.ckpt` links to it. Metric-selected
  checkpoints remain separate in `checkpoints/`.

The replacement uses the original paired H200 configuration and existing
features, with `--io-mode fast --io-prefetch 1` and
`--io-recovery-every-n-train-steps 100`. It starts from configured initialization,
not from the lost step-2000 state. Validation remains every 2000 steps.

Recovery includes model, optimizer, and Lightning loop state. Exact
mid-epoch data-order replay is not guaranteed by the existing dataloader.

## Verification

- 123 unit/integration tests passed; one CUDA-only test was skipped in the
  CPU test invocation.
- A separate four-H200, bf16 DDP smoke test completed two optimizer steps and
  two full retrieval validations with deliberately all-positive validation
  batches. All ranks agreed on metrics and the zero valid-contrast fraction.
  Both recovery and metric-selected checkpoints were written.
- A deliberate validation-crash test confirmed that a step-1 recovery file
  already contained optimizer state and could resume to step 2.

The existing Glutamine process was not stopped or hot-patched: a running
Python process must be restarted to load these code changes. Newly prepared
Glutamine configurations also select a 100-step recovery interval.
