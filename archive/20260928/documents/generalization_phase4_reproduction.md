# Reproducing the frozen phase 4 evaluation

Phase 4 is a retained failed extension, not the recommended successor to phase 2.
It improves balanced validation but reduces official six-cell mean MRR and fails
to improve the new measured aminotransferase panel over F3. Read
[findings.md](../../../findings.md) before interpreting its artifacts.

All paths below are relative to the `horizyn` repository. The campaign's absolute
paths are embedded in authenticated receipts; moving artifacts requires an
explicit provenance-preserving relocation procedure, not editing frozen files.
The shared environment is unchanged. The command environment used was:

```bash
export PYTHONPATH=/tmp/enzyme-generalization-runtime:$PWD
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
CAMPAIGN=runs/generalization_20260919_2251
PYTHON=../env/bin/python
```

The isolated runtime overlay supplies the versions recorded in the
[phase 2 reproduction guide](generalization_phase2_reproduction.md). The expanded
optional token-retrieval tests additionally used torch-geometric 2.6.1, numba
0.62.1 and llvmlite 0.45.1; their installation and test receipts are under the
campaign's `final_verification/`. If `/tmp`
has been cleared, recreate that overlay from the pinned dependencies first.
`PYTHON` and `CAMPAIGN` here are task-local shell variables.

## Model and feature identity

The immutable recipe is `$CAMPAIGN/phase4/frozen_recipe.json`, SHA256
`7353203fce81f1070649dbf535eac836602f6c4af576e7bbf8968b1e426fde2f`, frozen at
**2026-09-20 01:26:41 UTC**. It pins the parent phase 2 freeze, 30 implementation
sources, nine split/seed model tuples, native training reaction dictionaries,
seven input feature receipts and previously frozen controls.

The selected geometry is `eta_enzyme=0, eta_reaction=0.5`, with final anchor weight
0.25. Model bundles are under
`$CAMPAIGN/phase4/models/{reaction_smi,enzyme_smi,time}/{seed42,seed17,seed73,hybrid_anchor_only}/bundle.json`.
Native reaction dictionary states are under `$CAMPAIGN/phase4/native_states/`.
Their original creation commands are in `phase4/bundle_commands.json`; run those
only into a separate reconstruction directory if rebuilding. The initial screen
and independent checks are in `phase4/hybrid_anchors/` and
`phase4/production_checks/`.

Three official panels and Case 1, P450, nitrilase and aminotransferase form the
seven input panels. Every panel has a fixed catalog and authenticated feature
bundle. Case 1 uses `query_matched`; the others use `reactions`. Exact replay
uses the pinned arrays. Fresh upstream molecular extraction can shift near-tied
ranks, even with a fixed seed.

## Authenticate or generate declared predictions

```bash
$PYTHON scripts/generalization_phase4_run_predictions.py \
  --campaign "$CAMPAIGN" --gpus 0 1 2 3
```

This checks frozen sources and all feature receipts, then authenticates existing
outputs or generates missing jobs. It does not fit a model. The 28 jobs are seven
panels times four fixed variants. Existing valid scores are reused; input/output
hash mismatches fail. Each output directory contains its exact command, log,
score array, catalog and receipt. The dispatcher writes a new operational
`prediction_index.json`, so preserve its original copy if comparing run timing.
The frozen recipes and prediction arrays are not replaced on valid reuse.

Every score archive stores `selected`, `baseline` and `baseline_fp64`:

- `selected`: the declared composed model, FP64 accumulation rounded to FP32.
- `baseline`: native F3 vectors normalized once in FP32.
- `baseline_fp64`: native F3 vectors normalized once in FP64, stored in FP32.

Do not combine scores from different model keys or input feature generations.
The diagnostic hybrid-anchor-only variant is not an alternative selected after
viewing test results.

## Recompute official metrics

Use a new, empty output directory. The evaluator refuses to overwrite a
nonempty directory and authenticates every method/split score before opening
any official associations.

```bash
$PYTHON scripts/generalization_phase4_official_evaluate.py \
  --manifest "$CAMPAIGN/phase4/official_evaluation_manifest.json" \
  --feature-root "$CAMPAIGN" \
  --output /tmp/enzyme-phase4-official-replay \
  --device cpu --bootstrap-replicates 10000 --seed 20260919
```

This reports all eight declared methods, both directions, all-positive and
first-positive MRR, Hit@1/5/10, paired query intervals and reaction-group
sensitivity. The default TIGER reference comes from the campaign audit and is
explicitly a published-number comparison, not a controlled competitor rerun.

## Recompute external metrics

The outcomes are already open. Replays are computational reproductions, not new
independent confirmation. The wrapper requires an explicit release note so the
label-access event is recorded. Existing historical outputs remain immutable.

```bash
$PYTHON scripts/generalization_phase4_external_evaluate.py \
  --panel case1_p450 \
  --manifest "$CAMPAIGN/phase4/external_evaluation_manifest.json" \
  --audit-root "$CAMPAIGN" \
  --output /tmp/enzyme-phase4-case1-p450-replay \
  --evaluate --release-note 'Reproduction of already-open frozen outcomes'

$PYTHON scripts/generalization_phase4_external_evaluate.py \
  --panel nitrilase \
  --manifest "$CAMPAIGN/phase4/nitrilase_evaluation_manifest.json" \
  --audit-root "$CAMPAIGN" \
  --output /tmp/enzyme-phase4-nitrilase-replay \
  --evaluate --release-note 'Reproduction of already-open frozen outcomes'

$PYTHON scripts/generalization_phase4_external_evaluate.py \
  --panel aminotransferase \
  --manifest "$CAMPAIGN/phase4/aminotransferase_evaluation_manifest.json" \
  --audit-root "$CAMPAIGN" \
  --output /tmp/enzyme-phase4-aminotransferase-replay \
  --evaluate --release-note 'Reproduction of already-open frozen outcomes'
```

Case 1 uses heterogeneous literature evidence; P450 zeros are unlisted
associations. Nitrilase and aminotransferase use full measured panels with
assay-conditional non-detects. Their label policies and candidate rectangles
must not be substituted for one another. All 11 external methods remain in the
readouts. The aminotransferase curator's limited prior table exposure and the
panel's training overlap are disclosed in its audit.

## CLI integration and independent checks

The opt-in configuration `wet_lab/configs/phase4_generalization.yaml` uses the
same native F3 checkpoint, composed bundle and phase 2 feature-generation
contract. See [the adapter guide](../../../wet_lab/GENERALIZATION.md) for the existing
query CLI command and expected outputs. Actual prepared-input parity and fresh
CLI checks are in `$CAMPAIGN/phase4/wet_lab_adapter_check/`.

A separate implementation recomputes the main Reaction-Sim and aminotransferase
claims directly from saved scores, without importing campaign metric functions:

```bash
$PYTHON scripts/generalization_claim_audit.py --campaign "$CAMPAIGN"
```

The scientific figure is reproducible with the isolated plotting overlay:

```bash
PYTHONPATH=/tmp/enzyme-generalization-plotting:/tmp/enzyme-generalization-runtime:$PWD \
  $PYTHON scripts/generalization_campaign_figures.py --campaign "$CAMPAIGN"
```

Figure inputs, script identity and chart contract are recorded under
`$CAMPAIGN/figures/`. The source-change archive compares the final campaign with
the initial working-tree snapshot rather than incorrectly attributing all
pre-existing Git changes to this investigation.

## Reproduce the rejected post-evaluation calibration diagnostic

The later fixed training-mean calibration is a separate, development-exposed
control. It was not promoted: official six-cell MRR fell from phase2's 0.674219
to 0.672205. It changes none of the phase2/phase4 or large-pool methods above.
Its immutable transfer recipe is
`$CAMPAIGN/post_evaluation_calibration/transfer/frozen_recipe.json`
(SHA256 `52221e420f8c0789df035d0effc744294b9bec5419073992db127856da321c7b`).

The initial external evaluator failed before external labels were opened because
the exact approved nitrilase and aminotransferase feature receipts omit an
optional `freeze_sha256` field. A separate evaluation-only amendment permits
only those two receipts and requires the same checkpoint and authenticated input
and source hashes. No model, prediction, original frozen source, or metric
function was changed. The amendment is `evaluation_guard_amendment.json` beside
the transfer freeze (SHA256
`2100d8d3beba7359e615af10036d8ca0a29743b344d802bb338b8b465ae396c8`).
Use the **v2** external wrapper below; the original wrapper and failed log remain
preserved for audit. The amendment's explanatory timing claim is corrected in
[the provenance erratum](../../../runs/generalization_20260919_2251/post_evaluation_calibration/transfer/evaluation_guard_provenance_erratum.md):
nitrilase features precede the phase2 freeze, while aminotransferase features
follow it. Guard acceptance uses the exact approved identities, not those dates.

```bash
$PYTHON scripts/generalization_calibration_official_evaluate.py \
  --manifest "$CAMPAIGN/post_evaluation_calibration/transfer/official_evaluation_manifest.json" \
  --feature-root "$CAMPAIGN" \
  --output /tmp/enzyme-calibration-official-replay \
  --device cpu --bootstrap-replicates 10000 --seed 20260919

$PYTHON scripts/generalization_calibration_external_evaluate_v2.py \
  --freeze "$CAMPAIGN/post_evaluation_calibration/transfer/frozen_recipe.json" \
  --amendment "$CAMPAIGN/post_evaluation_calibration/transfer/evaluation_guard_amendment.json" \
  --output /tmp/enzyme-calibration-external-replay \
  --evaluate --release-note 'Reproduction of already-open exploratory calibration outcomes'
```

These commands retain all eight methods, all official pools, the measured
nitrilase panel, all five fixed aminotransferase rectangles, and Case1's
reaction-to-enzyme ordering check. They reuse saved scores; no feature extraction
or model selection is performed. The final centered, support-gated affinity
study under `$CAMPAIGN/post_evaluation_centered_affinity/` also rejected every
nonzero correction using validation alone and performed no external predictions.
