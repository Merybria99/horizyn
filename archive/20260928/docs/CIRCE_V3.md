# CIRCE-v3: train-only, promiscuity-aware F3 training

Branch: `codex/CIRCE-v3`. This is a training-recipe change, not a new backbone or
a claim of improved accuracy. Existing runs, checkpoints and caches are untouched.

## Architecture and objective

- Reuse the F3 architecture and cached ProT5 residue, ReactionT5v2, UniMol2,
  ChIRo and chemistry features. Keep the pretrained SLEEC scorer frozen.
- Initialize the trainable reaction and enzyme towers from scratch, jointly train
  them, and retain the source F3 optimizer, batch size and numerical precision.
  This does **not** fine-tune the cached foundation-model backbones.
- Use `DecoupledAllPositiveInfoNCELoss`, equal R→E/E→R weights, fixed beta 10,
  and `all_known_in_batch` from retained training associations only.
- For each anchor, contrast each known positive against non-edges, without
  competing against other known positives; average positives within anchor.
  Non-edge exponential mass is multiplied by 0.5. This is a weak-negative
  heuristic, **not nnPU, experimental inactivity supervision, or a 50/50 batch**.
  Existing random pair batching supplies in-batch competitors; no hard-negative
  miner is enabled. It cannot establish that an unobserved activity is positive.
- Use up to 30 epochs and patience 5 on
  `val/mean_bidirectional_reactzyme_mrr` (all-positive MRR, not first-hit MRR).
  Validate every epoch. Save best/last checkpoints and recovery checkpoints every
  50 training steps. Recovery does not promise exact mid-epoch sampler replay.

## Classification heads

The F3 mechanism/cofactor blocks remain part of retrieval even when their
classification losses are disabled. Labels are training targets, never required
query inputs. No generic “unknown” target is optimized as a biological class.

`prepare --aux-evidence evidence.csv` accepts:

```csv
protein_id,family,label_index,value,confidence,source,evidence_kind,support_reaction_id
proteinA,cofactor,0,1,1,curation_record_1,curated_positive,
proteinB,cofactor,0,0,1,assay_record_2,measured_negative,
proteinC,mechanism,2,1,0.5,reaction_annotation_3,training_association,reaction3
```

These are format examples, not biological annotations. `label_index` is zero-based
and must use the intended F3 head vocabulary (8 mechanism and 10 cofactor outputs
in the current templates). Binary zero must mean explicit negative evidence for
that **biological label**, not an enzyme/reaction pair's `Label=0`. Positive and
negative evidence must refer to the same property/assay interpretation.

Independent `curated_positive`/`measured_negative` records require an empty
support reaction. Reaction-derived positive evidence requires its exact retained
training edge. Hidden/validation/test edges cannot supply this evidence. Missing,
unknown and conflicting cells are masked. Unobserved labels are not negatives.
Each label needs both positive and negative training evidence to be eligible;
one-class labels and families with no eligible labels are disabled and reported.
Provenance declarations are auditable, not automatically scientifically verified.

Eligible heads use confidence-weighted masked BCE, normalized over valid cells
globally across DDP ranks and over active families. The total auxiliary weight is
0 at epoch 0, 0.00667 at epoch 1, 0.01333 at epoch 2, then 0.02. It is **0.02 total**,
not 0.02 per head. Resume restores the ramp via the checkpoint epoch.

Logs include per-family losses, coverage counts, effective weight, and every 100
steps the auxiliary/retrieval gradient ratio and alignment at their shared enzyme
features. These are feature-gradient diagnostics, not full parameter-gradient norms.

Without `--aux-evidence`, heads are explicitly unsupervised (`biofp_aux_weight=0`).
The existing derived annotation files are not silently treated as verified
negative evidence. Enabling heads needs an audited export in the above contract.

## Preparation and safeguards

Run from the `horizyn` project using the existing training environment:

```bash
../env/bin/python scripts/run_circe_v3.py prepare \
  --dataset reactzyme --splits time enzyme_smi reaction_smi \
  --run-root runs/circe_v3_my_run --devices 4

../env/bin/python scripts/run_circe_v3.py prepare \
  --dataset enzymecage --run-root runs/circe_v3_my_run --devices 4
```

The default templates are the existing F3 runs in this workspace. Override
`--source-config-root` for ReactZyme or `--source-config` for EnzymeCAGE to use
another installation's matching F3 templates/caches. No downloads or extraction
are performed. A new root is required; preparation never overwrites a run.

Each split has its own train graph, optional targets, configs, audit and hashed
manifest. ReactZyme retains its existing train-derived validation and released
test sets. EnzymeCAGE uses the existing original validation; no separate test set
is invented. Paired train/validation/test overlaps are rejected.

By default, a deterministic development edge holdout requests 5% of training
associations while retaining both endpoints in training. Sparse graphs may admit
far fewer edges; the manifest records actual counts. Held-out edges are removed
from positive lookups and reaction-derived labels. Set `--hidden-fraction 0` for
an exact full-training-pair comparison, but then there is no hidden-edge diagnostic.
The cached chemistry vocabulary remains the existing train-fitted vocabulary;
this edge holdout measures association recovery, not unseen-chemistry generalization.

```bash
../env/bin/python scripts/run_circe_v3.py check \
  --run-dir runs/circe_v3_my_run/reactzyme/reaction_smi/seed42

CUDA_VISIBLE_DEVICES=0,1,2,3 ../env/bin/python scripts/run_circe_v3.py train \
  --run-dir runs/circe_v3_my_run/reactzyme/reaction_smi/seed42 \
  --detach circe_v3_reaction_smi
```

Use `time` or `enzyme_smi` in the run directory for the other splits; use
`enzymecage/original` for EnzymeCAGE. Do not launch these concurrently on the same
GPUs. The launcher checks UUID-resolved GPU occupancy, refuses overlapping runs,
and never stops an existing job. The detached log is `<run-dir>/pipeline.log`.
The preflight checks manifests and cache IDs; permitted missing UniMol2/ChIRo
modalities retain the F3 mask and are explicitly reported. It is not a full
numerical scan of the multi-terabyte residue cache or a storage-health guarantee.

To resume, add `--resume /absolute/path/to/this/run/checkpoints/last.ckpt` (or a
recovery checkpoint beneath that directory). A previous run cannot be restarted
from scratch accidentally over its output. No full GPU training is launched by
preparation or checking.

## Evaluation

```bash
../env/bin/python scripts/run_circe_v3.py evaluate \
  --run-dir runs/circe_v3_my_run/reactzyme/reaction_smi/seed42 \
  --subset test --checkpoint /absolute/path/to/selected/best.ckpt
```

`validation`, `hidden`, and (ReactZyme only) `test` evaluations report both
directions. Hidden evaluation excludes retained known positives from competitor
pools and uses all retained training proteins/reactions as candidates. Released
test evaluation keeps the existing paper-test protocol unchanged. Test metrics
are not used for model selection. Evaluation output never overwrites an earlier
result and cannot run concurrently with this launcher's training in the same root.

For within-family **development R→E** evaluation, add `--family-map families.csv`
with `protein_id,family_id`. Multiple rows allow overlapping families; every
candidate must have a family assignment. For each query, use the union of the
families of its development positives, retaining all candidates in those families.
This is a conditional specificity diagnostic, not ordinary retrieval and not a
test-time annotation requirement. The runner records the mapping hash, refuses
partial coverage, and forbids this override on the released test set. Use the
same mapping/pools for every baseline. CYP remains an external final benchmark,
not a source of hyperparameter tuning labels.

## Verification

```bash
../.capability-run-py/bin/python -m pytest -o addopts='' \
  tests/unit/test_circe_v3.py tests/unit/test_losses.py \
  tests/unit/test_biofp_targets.py tests/unit/test_protein_pooling_positive_pairs.py \
  tests/unit/test_training_options.py tests/unit/test_config_validation_sections.py \
  tests/unit/test_evaluate_protein_pooling_protocols.py -q
```
