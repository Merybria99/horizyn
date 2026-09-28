# Supported pipeline guide

## Scope and entry points

V4 is the main, dictionary-free pipeline. F3 and CIRCEv2 retain their historical
architectures and losses. They share feature readers and training mechanics;
select the family explicitly when running either historical pipeline.

Run commands from the `horizyn` project directory, using
`../env/bin/python -m horizyn.pipelines`. After editable installation, `circe`
invokes the same CLI. Imports for new applications are:

```python
from horizyn.pipelines import RetrievalPipeline
pipeline = RetrievalPipeline.from_manifest("configs/pipelines/v4/frozen_reaction_smi.json", "cuda:0")
```

Historical `horizyn.model` imports remain valid. Neural implementations now
live in `horizyn/models/common.py`, `enzyme.py`, `reaction.py`, and `dual.py`.
`models/compat.py` contains inactive legacy classes required by shared
constructor branches. Supported entry points reject unrelated encoder modes,
directional adapters, and enzyme-prototype scoring.

## V4 architecture and source map

The enzyme encoder combines a ProtT5 global mean, a SLEEC-guided residue view,
and four learned residue views. Their projected features are fused and added
to the independently projected global representation, followed by unit
normalization. `horizyn/enzyme_multiview.py` implements these views and their
attention safeguards. `models/dual.py` connects the frozen scorer and feature
inputs to that encoder.

The reaction encoder combines ReactionT5v2, UniMol2, ChIRo, and train-fitted
reaction-set chemistry. `models/reaction.py` implements molecular pooling and
modality fusion. Input semantics are dataset-specific: the frozen ReactZyme
models use participant-set pseudo self-reactions; the frozen EnzymeMap model
uses physical reaction sides. Never interchange these feature representations.
`pipelines/preprocessing.py` supplies the shared participant normalization.

Phase 1 uses `DecoupledAllPositiveInfoNCELoss`, the existing anchor-balanced
multi-positive objective in `horizyn/losses.py`, with beta 5.0 in the V4 profiles.
The positive graph, unique anchors, opposite-modality pools, and residue
attention penalties remain implemented by the shared Lightning engine.

Phase 2 freezes both encoders and fits independent enzyme/reaction residual
heads to the full training graph. Each head is LayerNorm, Linear(512,1024),
GELU, Linear(1024,512); the final layer is zero-initialized. The normalized
representation is base + 0.2 × head(base). Training uses bidirectional
uniform-positive cross entropy at temperature 0.2 plus 10 × mean cosine
displacement from the base vectors. There is no biological-label auxiliary
supervision or dictionary branch. The trainer is `pipelines/refinement.py`;
the checkpoint-compatible head and alignment function remain in
`generalization_residual.py`.

At inference, `pipelines/fusion.py` applies the protein fusion multiplier once.
Uncalibrated frozen V4 checkpoints use multiplier 3; checkpoints with the
calibrated scalar already stored use multiplier 1. `pipelines/scoring.py`
applies half the trained residual scale (0.1) and returns the common
reaction–enzyme dot product. Final vectors are normalized. Float64 dot-product
accumulation followed by float32 output preserves the recorded evaluation
arithmetic and tie behavior. There is no ensemble, hubness correction,
dictionary score, or direction-specific reranker.

## Configuration and training

| Family | Maintained profiles | Original loss |
|---|---|---|
| V4 | `configs/pipelines/v4/reactzyme_{reaction_smi,enzyme_smi,time}.yaml`, `enzymemap.yaml` | DecoupledAllPositiveInfoNCELoss |
| F3 | `configs/pipelines/f3/published_retrieval.yaml` | FullBatchMLNCELoss |
| CIRCEv2 | `configs/pipelines/circev2/reactzyme_reaction_smi.yaml` | SampledMultiPositiveInfoNCELoss with annotation-negative sampling |

The F3 profile uses its original published-retrieval training data (`data/sota`),
not a matched-data ReactZyme retraining. CIRCEv2 uses the original ReactZyme
ReactionSim annotation-negative experiment. V4 preserves the original
dataset-specific recipe with ten encoder epochs. Data scope is explicit in
the YAML. Do not use a profile from another dataset as a fair baseline.

```bash
../env/bin/python -m horizyn.pipelines train \
  --config configs/pipelines/v4/reactzyme_reaction_smi.yaml --model-preflight-only
../env/bin/python -m horizyn.pipelines train \
  --config configs/pipelines/v4/reactzyme_reaction_smi.yaml
../env/bin/python -m horizyn.pipelines train --pipeline f3 \
  --config configs/pipelines/f3/published_retrieval.yaml
../env/bin/python -m horizyn.pipelines train --pipeline circev2 \
  --config configs/pipelines/circev2/reactzyme_reaction_smi.yaml
```

`pipelines/training.py` is the shared phase-1 engine. Resume, DDP, precision,
validation cadence, callbacks, and command-line configuration overrides use
the original implementation. The old `train.py` and `train_protein_pooling.py`
entry points delegate to it; they no longer launch the pooled Horizyn baseline.

Export the fitted V4 base encoder's training nodes before refinement:

```bash
../env/bin/python -m horizyn.pipelines export-refinement \
  --config configs/pipelines/v4/reactzyme_reaction_smi.yaml \
  --checkpoint PATH_TO_PHASE1.ckpt --output runs/v4/train_features --device cuda:0
../env/bin/python -m horizyn.pipelines refine \
  --features runs/v4/train_features --output runs/v4/refinement \
  --config configs/pipelines/v4/phase2.yaml --device cuda:0
```

Export reads only training pairs and training reaction features, uses fusion
multiplier 1 and no residual head, and saves `catalog.json`, `pairs.npz`,
`f3_features.npz`, and `manifest.json`. The legacy `f3_features` filename is
retained to read existing V4 caches. `proteins` is in catalog order;
`train_reactions` is in compact training-reaction order. Edges are unique
integer reaction/enzyme indices. Refinement indexes training nodes only.

The phase-2 config is consumed directly; explicit CLI flags override it.
Its keys are steps, hidden, scale, temperature, identity_weight, learning_rate,
weight_decay, and seed. Output uses the existing head checkpoint schema.
Selection is the configured fixed final step; no test selection or hidden
validation sweep occurs. New output paths must be empty to prevent overwriting
completed experiments. Existing frozen ReactZyme and EnzymeMap feature caches
are accepted with their original explicit training-only provenance flags.

## Features and inference manifests

Feature extraction remains separate from model inference. The maintained
extractors are `extract_prott5_residue_embeddings.py`,
`extract_reaction_t5v2_embeddings.py`, `extract_unimol2_reaction_embeddings.py`,
and `extract_chiro_reaction_embeddings.py`. Use their `--help` for feature
precision, batch size, pretrained weights, and device controls. Chemistry
features must use the schema fitted on the training split; they must not be
refitted on external queries.

An inference feature directory contains `reactiont5v2.h5`, `unimol2.h5`,
`chiro.h5`, and `reaction_set_features.npz`, all aligned to the reaction CSV.
The CSV has `reaction_id,reaction_smiles`; protein IDs are one per line and
must exist in the residue HDF5 (`ids`, `vectors`, `offsets`). Missing IDs fail
explicitly. Features must follow the checkpoint's reaction semantics.

The frozen V4 manifests under `configs/pipelines/v4/frozen_*.json` point to the
four original target-trained models, with semantic score weight zero. Historical
benchmark-result fields referring to dictionary scores were omitted. These
manifests do not imply a new benchmark evaluation.

A minimal F3/CIRCEv2 manifest contains pipeline, config, and checkpoint:

```json
{"pipeline": "circev2", "config": "config.yaml", "checkpoint": "model.ckpt"}
```

A V4 manifest additionally specifies phase2_checkpoint,
feature_manifest_sha256, fusion_multiplier, checkpoint_already_calibrated,
and residual_cap. Declare config_sha256, checkpoint_sha256, and
phase2_checkpoint_sha256 to authenticate artifacts before loading. Optional
feature_manifest/source_phase2 metadata authenticates the head's feature
lineage; calibration_receipt authenticates the permitted scalar-only change.
Dictionary-bearing protocols are rejected. Relative artifact paths resolve
against the JSON file; training YAML paths resolve from the project directory.

```bash
../env/bin/python -m horizyn.pipelines encode \
  --manifest configs/pipelines/v4/frozen_reaction_smi.json \
  --reactions reactions.csv --proteins protein_ids.txt --residues proteins.h5 \
  --features reaction_features --output final_embeddings.npz --device cuda:0
```

The output contains reactions, enzymes, reaction_ids, and enzyme_ids.
These are final retrieval vectors. For a newly trained V4 model, create a
manifest pointing to its encoder, residual head, and export manifest hash.
If a V4 manifest has no head, inference exposes phase-1 vectors for diagnostics;
that output is not the complete two-stage main model.

## Evaluation and case studies

```bash
../env/bin/python -m horizyn.pipelines evaluate \
  --embeddings final_embeddings.npz --pairs test_pairs.csv \
  --protocol reactzyme --output metrics.json --device cuda:0
```

Use `--protocol screening` for EnzymeMap: R→E BEDROC85, BEDROC20, EF5 and EF10,
without MRR. ReactZyme reports both retrieval directions and its original
all-positive MRR definition. Candidate order breaks score ties consistently.
Evaluation preserves the supplied candidate bank and errors on positive IDs
missing from it; it does not silently shrink the screening database. The bank
and queries must be those of the claimed official split/protocol. For the
paper screening comparison, supply the full official screening bank.

`scripts/run_unified_retrieval_benchmark.py` remains the generic phase-1
baseline evaluator with explicit --pipeline selection and official pool
manifests. It does not apply V4 phase 2; use the final-embedding commands above
for complete V4 scores.

The existing Rv1430, AtPAP15/HbADH2, and testosterone runners under
`../case_studies/` now import checkpoint, fusion, and scoring helpers from
the shared API. Their result paths and experiment-specific reports remain.
`case_studies/pipeline.py` and `predict.py` retain the historical CIRCEv2
sequence-extraction interface; they do not represent the default V4 model.

## Verification and historical code

Run `../env/bin/python -m pytest tests -q -o addopts=''` and
`../env/bin/python tools/organize_source.py --check` from this directory.
The dated archive holds old source and configs, not an additional maintained
pipeline. Checkpoints, datasets, results, manuscripts, and run-source snapshots
remain in place. See [the cleanup evidence](CLEANUP_20260928.md).
