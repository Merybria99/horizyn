# B1/B2 reaction_smi campaign

The default B1/B2 configs now select **pretrained PCQM4Mv2 Graphormer**. See
[the pretrained replacement record](README_B12_PRETRAINED.md) for its checkpoint,
preprocessing, runtime, warmup schedule and new launches.

The September 8, 2026 scratch campaign completed two models against the retained
CIRCE-V2 epoch-23 reference. B1 uses global cosine scoring. B2 uses an equal
mixture of global cosine and token-level chemical-to-protein coverage. Both use
the same frozen ESM-C 600M initializer, trainable molecular Graphormer, token
banks, seed, and initial trainable weights.

## Preserved scratch campaign

Campaign directory:

    runs/b12_reaction_smi_train_20260908_230718/

| Run | Physical GPUs | Torchrun supervisor PID |
|---|---|---:|
| B1 | 0, 1 | See current `launches.json` |
| B2 | 2, 3 | See current `launches.json` |

These are independent detached torchrun processes, each with two workers. The
launch commands and process IDs are recorded in `launches.json`. The per-run
directories `b1/` and `b2/` contain:

- `train.log`: startup, extraction, training, and validation output.
- `status.json`: current phase and last completed training step, or a failure.
- `steps.jsonl`: losses, candidate counts, step time, and peak GPU memory.
- `run_manifest.json`: initialization hash, model, data, and runtime settings.
- `checkpoints/best.pt` and `checkpoints/last.pt`: written after validation.
- `evaluation/`: directional validation ranks and metrics for each epoch;
  after training, test metrics and `comparison_with_circe_v2.json`.

A separate read-only monitor (PID recorded in `monitor/monitor.pid`) checks both jobs every 30 seconds
until they finish or exit. Its live report is
`runs/b12_reaction_smi_train_20260908_230718/monitor/status.md`; the same directory
contains `latest.json`, `history.jsonl`, and `monitor.log`. It records process
health, losses, validation metrics, GPU activity, free cache space, and alerts
for failures or extended inactivity. It does not restart or modify training.

For example, from the Horizyn directory:

```bash
tail -f runs/b12_reaction_smi_train_20260908_230718/b1/train.log
```

## Training and comparison

Each worker samples 1,000 pair rows, giving 2,000 global rows before entity
deduplication. The existing `TypedNegativeBatchSampler` supplies 50% positives,
with biological/random negatives balanced where pools permit. The new trainer
uses `SampledMultiPositiveInfoNCELoss` with fixed beta 10. All known training
positives and all typed-pool entries present in the global batch enter their
respective masks; other cells are ignored. Scoring only allowed cells in
training is mathematically equivalent to scoring the full masked matrix.

Training uses FP32, AdamW at 1e-4 with weight decay 0.01, at most 30 epochs,
and validation every epoch. Checkpoint selection uses arithmetic mean
bidirectional first-positive MRR, matching CIRCE-V2. Early stopping uses five
epochs and min_delta 0.0001. ReactZyme all-positive MRR is reported separately.
The two-GPU sampler need not reproduce the historical four-GPU run's exact
random draws; both new runs use the same two-GPU construction.

The fixed test universe has 386 reaction candidates and 14,688 enzyme
candidates, with test-only ground truth and canonical forward reaction queries.
Both retrieval directions use exactly the same pair score. Each run evaluates
its validation-selected checkpoint on test once when training ends, then writes
a comparison with the existing CIRCE-V2 epoch-23 JSON. No baseline retraining
is scheduled.

### Shared CIRCE-V2 evaluation

`horizyn/token_retrieval_evaluation.py` adapts the B1/B2 score matrix to the
existing CIRCE code. Validation directly calls
`ProteinPooledLitModule._batched_retrieval_metric_values`, with its rank
`1 + number of strictly higher scores`, top-K settings, and macro averaging.
Final testing calls `scripts/evaluate_protein_pooling.py`'s
`append_retrieval_metrics`, including its descending `torch.argsort`, average
precision, R-precision, and first/all-positive MRR. These two CIRCE ranking
routines handle ties differently; the adapter preserves each routine.

The adapter also reuses CIRCE's forward query expansion, validation positive
lookup builder, and candidate selection. Validation uses the ordered custom
candidate file (2,448 reactions and 16,374 proteins). Paper testing uses unique
test proteins in pair-file order. Enzyme-to-reaction query anchors are sorted
proteins with positives, as in CIRCE. Missing required inputs raise an error.

For validation, `mean_bidirectional_mrr` is the arithmetic checkpoint selector;
`balanced_mrr` and `balanced_reactzyme_mrr` are harmonic means, matching CIRCE's
validation logs. The monitor displays arithmetic means explicitly. The test
JSON retains CIRCE's standalone names, where `balanced_*` are arithmetic.
Every result records the shared evaluator identifier and function hashes.

The initial six epochs used a separate stable-sort evaluator. On September 8,
both jobs were paused at step 888 and their checkpoints and results preserved
under `evaluation_migration/b{1,2}/original/`. Those original metrics are retired
from the live comparison. The saved epoch-5 weights were re-evaluated using the
shared CIRCE routines and both runs resumed at epoch index 6. Historical epoch-0 through epoch-4
weights were not retained and cannot be re-scored; selection is re-established
from the retained checkpoint, with optimizer and RNG state preserved.
The corrected mean first-positive MRR was 0.096457917 for B1 and 0.099469485
for B2; this change was negligible. `evaluation_migration/README.md` and
`real_score_evaluator_audit.json` document the real-score comparison and
candidate-eligibility verification.

## Frozen features and runtime

The foundation checkpoint is pinned to
`biohub/esmc-600m-2024-12`, revision
`e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`. It was downloaded into
`.deps/b12_hf_cache/`. All foundation parameters stay frozen. FP32 residue
features are computed on demand and cached atomically under
`/tmp/enzymediscovery_b12_residue_cache/`, shared by both jobs. The first epoch
therefore includes extraction work; subsequent accesses reuse the cache.
Features contain no pair labels or learned task pooling. Cache keys identify
the normalized, ends-center-truncated sequence and the pinned encoder setup.

The runtime is `../.b12-run-py/bin/python`, a separate package overlay on the
existing Torch 2.4 environment. ESM 3.2.1 and Transformers 4.48.1 are pinned for
the ESM-C tokenizer and weight format. Installed package versions, source
snapshots, and input SHA-256 values are saved in the campaign directory.

Molecule graphs preserve available atom/bond stereochemistry, component
multiplicity, and unknown participant roles. All released inputs parsed in
preflight: 7,726 unique reaction IDs and 9,514 unique molecular components;
no examples were dropped. The full validation scoring grid was profiled with
synthetic token banks; that profile measures scoring only, excluding encoding.

## Validation and restart

Twenty-six targeted tests cover the model/loss checks, the existing CIRCE test
protocol, and the adapter's ties, duplicate positives, candidate ordering,
partial batches, and query eligibility. Both models also completed two real two-GPU smoke steps with finite
losses and gradients. Smoke checkpoints do not initialize the full runs.

The launcher is `scripts/launch_token_retrieval_pair.py`. A new invocation creates
new runs and checks that all four GPUs are available; it is not a resume command.
To resume one interrupted run from its last completed validation checkpoint,
use the same GPU allocation and run directory, for example:

```bash
CUDA_VISIBLE_DEVICES=0,1 ../.b12-run-py/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/train_token_retrieval.py \
  --config runs/b12_reaction_smi_train_20260908_230718/b1/launch_config.yaml \
  --run-dir runs/b12_reaction_smi_train_20260908_230718/b1 \
  --resume runs/b12_reaction_smi_train_20260908_230718/b1/checkpoints/last.pt
```

An interrupted epoch is replayed from the last completed epoch. The shared
frozen-feature cache remains reusable. Do not start a resume process while the
original job is still running.
