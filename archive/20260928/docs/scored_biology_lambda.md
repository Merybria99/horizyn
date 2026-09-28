# Frozen, directly scored biology: matched lambda experiment

This is an isolated experiment, not a replacement of the active CIRCE trainer.
The default source is the completed feature-gate/multiview epoch-29 checkpoint
(30 completed epochs) in `runs/circe_v3_feature_gate_vybz7n4w`.

## What is trained

The existing towers, SLEEC scorer, attention and gates stay frozen. Export their
shared features once from existing backbone caches: 1,536 enzyme features and
1,539 reaction features (three modality tokens plus availability). No new
ProT5, UniMol2, ChIRo or ReactionT5 backbone extraction is requested.

Two independent LayerNorm → Linear(256) → GELU → Linear(192) branches produce
three normalized 64-dimensional blocks: EC, cofactor, transformation. The
supervised blocks themselves participate in the retrieval score:

`score = (1-alpha) * baseline_cosine + alpha * mean(block_cosines)`.

This score is also a dot product of concatenated, square-root-weighted vectors;
it does not require a cross-encoder or labels at inference. `alpha=0` recovers
the frozen baseline exactly. Fixed anchors encode category identity and EC
ancestry; they are not measured biological distances. Specific cofactor names
and ions remain separate categories. Transformation labels are coarse inferred
chemistry, not experimentally verified enzyme mechanisms.

## Objective and controls

- The existing decoupled all-positive bidirectional InfoNCE, fixed beta=10,
  equal direction weights, unknown-negative denominator weight=0.5.
- Every known training positive in the batch is protected; shared annotations
  never manufacture enzyme–reaction positives.
- Added positive-anchor loss weights: lambda=0, 0.02, 0.05. Warmup is zero at
  epoch index 0, then one-third, two-thirds, and full weight from index 3.
- Confidence multiplies distance and the average divides by observed label
  count, not confidence sum. A lone confidence-0.4 label has 40% of the gradient
  of the same label with confidence 1. Missing labels contribute zero.
- Each family averages annotated entities; active families then active sides
  are averaged. Unannotated reaction EC remains neutral and is learned through
  retrieval, not inferred from the union of an enzyme's EC labels.
- Source-specific cofactor weights are heuristic evidence strengths, not
  calibrated probabilities. Row-level experimental flags do not upgrade every
  label to experimental evidence.
- Global pair batch 1536, AdamW lr=1e-4, weight decay=0.01, gradient clip=1,
  10 epochs, train alpha=0.1. Identical seed, batches and initialization across
  lambda runs. No dropout in the new branches.

Lambda=0 means **no new biological supervision**. The frozen base already
received biological supervision. The new loss's confidence normalization also
means lambda values are not directly comparable to the legacy trainer's.

## Validation and interpretation

Full validation candidates, both directions, stable candidate-ID tie breaking,
query-macro all-positive MRR, first-positive MRR, recall@10 and hit@10. Rescore
alpha=0, 0.05, 0.1, 0.2 from the same embeddings. Checkpoints/alpha are selected
on validation only. No held-out test is automatically consulted.

The configured split has 2,085 reaction IDs shared between train and validation,
but no shared positive pairs. A separate unseen-reaction R→E metric uses the
same full enzyme candidate catalog. This is descriptive validation, not an
independent untouched-test claim. The base checkpoint itself was selected using
this validation set. Historical metrics with another tie policy may differ.

Outputs:

- `split_audit.json`, `annotations.json`, `manifest.json`: coverage and provenance.
- `lambda_*/history.json`: losses by side/family, weighted biological loss,
  first-batch weighted-biology/retrieval gradient-norm ratio, block variance,
  validation for every alpha.
- `lambda_*/best.pt`, `last.pt`: small heads, optimizer and base identity.
- `comparison.csv`: all alphas at each run's validation-selected epoch. An
  alpha=0 winner means the baseline won, not that biology improved retrieval.

Compare lambda variants at matched epochs and fixed alpha=0.1 in history before
interpreting each run's selected best. The gradient ratio is a diagnostic from
one batch, not proof of usefulness or harm. Promotion requires improvement in
retrieval, especially unseen-reaction validation; lower biological loss alone
is insufficient. A larger lambda can over-cluster broad EC/cofactor categories.

## Detached launch on the GPU server

### Higher-lambda follow-up without another feature export

After the original sweep completes, use a new run root and explicitly reuse
its frozen snapshot. The source checkpoint/config, model code, chunk layout,
data and cache-file identities are checked. Existing features/results are
read only; each new head starts from the same seeded initialization, not the
lambda=0.05 checkpoint. Lambda=0.5 and 1.0 are stress tests for over-weighting
annotations, not presumed improvements.

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../.capability-run-py/bin/python scripts/run_circe_scored_biology.py launch \
  --run-root runs/circe_scored_biology_lambda_high_v1 \
  --feature-cache-root runs/circe_scored_biology_lambda_v1 \
  --gpus 0,1,2,3 --allow-shared-gpus \
  --lambdas 0.1,0.2,0.5,1.0 --epochs 10
```

Four independent head runs execute concurrently, one per selected GPU. All
other training parameters and the validation alpha sweep stay unchanged.
Compare both fixed epoch-10/alpha=0.05 outcomes and validation-selected outcomes
against the original sweep; do not select from the test set.

### Original sweep

From a Mac first connect with `ssh arginine`; run the following on that server:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../.capability-run-py/bin/python scripts/run_circe_scored_biology.py launch \
  --run-root runs/circe_scored_biology_lambda_v1 \
  --gpus 0,1,2,3 --lambdas 0,0.02,0.05 --epochs 10
```

By default all selected GPUs must be free; no existing job is stopped.
To explicitly share occupied GPUs, add
`--allow-shared-gpus` to the launch command. This bypasses only occupancy
rejection at both launch and later stage checks; GPU-ID validation and the
duplicate-run lock remain. Existing jobs are left running. This is not a VRAM
reservation or memory cap: simultaneous jobs can still exhaust device memory.
With either mode, four GPUs export
features, then three GPUs train the three independent heads (the fourth is
idle). Small heads do not benefit from artificially filling VRAM. Shared-storage
reading can dominate the one-time export. Per-batch frozen exports are atomic.

Repeat the same command to reuse completed exports and resume saved epochs.
Changed settings, code or source identities require a new run root. Concurrent
controllers are prevented by a lock inherited by their workers. The launcher
prints the detached session and pipeline-log path; worker progress is in
`export_worker_*.log` and `train_weight_*.log` in the run directory.

Verified locally: real target preparation, CPU exports through the actual
checkpoint for one enzyme and train/validation reaction, and synthetic
training/resume plus gradient/metric/provenance tests. A full multi-GPU run was
not launched or throughput-benchmarked here.
