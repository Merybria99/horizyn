# Reproducing the frozen generalization campaign

The campaign directory is `runs/generalization_20260919_2251`. Its immutable
`frozen_recipe.json` records the selected recipe, all nine residual checkpoints,
source hashes, candidate pools, primary method, and diagnostic controls. This
document describes execution; measured outcomes belong in `findings.md`.

## Environment

Run commands from the `horizyn` repository. The existing interpreter is
`../env/bin/python` (Python 3.12.3, PyTorch 2.4.0+cu121). Exact package and
hardware versions are in the campaign's `environment.json`. A small dependency
overlay was used without replacing the shared environment:

```bash
uv --cache-dir /tmp/enzyme-generalization-uv pip install \
  --python ../env/bin/python --target /tmp/enzyme-generalization-runtime --no-deps \
  pandas==2.2.3 pyarrow==17.0.0 biopython==1.85 python-dateutil==2.9.0.post0 \
  pytz==2026.3.post1 tzdata==2026.4 six==1.17.0
export PYTHONPATH=/tmp/enzyme-generalization-runtime:$PWD
```

Existing frozen base caches identify the checkpoint by file identity. They
contain BF16 inference stored in FP16 for the large training pool, and native
FP32 extraction for validation and test. These roles must not be interchanged.
All raw protein means use FP32 accumulation of the stored residue vectors.
All fitting statistics and anchor associations use training rows only.

## Feature export and training

`scripts/generalization_export.py` exports the original train/validation graph,
the raw modalities, and aligned frozen F3 features. The default configuration
is the official Reaction-Sim F3 checkpoint. For another split, explicitly pass
that split's configuration and compact/exact-validation caches. Raw protein
means may be reused across splits only from the identical residue source: no
learned normalizer or association is reused across splits.

Example for a fresh output directory:

```bash
../env/bin/python scripts/generalization_export.py \
  --output runs/NEW_CAMPAIGN/features

../env/bin/python scripts/generalization_full_graph.py \
  --features runs/NEW_CAMPAIGN/features \
  --output runs/NEW_CAMPAIGN/graph_uniform_s42 \
  --device cuda:0 --enzyme-weighting uniform \
  --steps 1000 --validate-every 25 --seed 42
```

The residual training has two independent 512→1024→512 MLPs. The final layer
starts at zero, the residual scale is 0.2, and output vectors are normalized.
The frozen F3 tower parameters never change. Each step includes the entire
training association graph in both contrastive denominators. Temperature is
0.07; cosine identity regularization has weight 2.0; AdamW uses learning rate
0.0001 and weight decay 0.001. Checkpoints maximize validation mean all-positive
MRR subject to four safeguards: neither aggregate direction nor either
unseen-reaction direction may drop more than 0.005 from initialization.

Seeds 42, 17, and 73 vary the residual initialization; they share a seed-42 F3
base checkpoint. They are not three independent end-to-end F3 retrainings.

`generalization_anchors.py`, `generalization_linear.py`, and
`generalization_hubness.py` are separate validation-only screens. The final
composition is in `composition_v4`; previous composition directories are
superseded numerical diagnostics. The selected anchor has 32 protein neighbors,
16 reaction neighbors, temperatures 0.03 on each side, and all four chemical
modalities. It contributes 25% of the deployed score. The remaining 75% comes
from the graph residual. Hyperparameters were fixed on Reaction-Sim validation
and transferred unchanged to the other splits.

## Frozen evaluation

The official exporter refuses to run without a frozen recipe receipt. It
checks full official candidate counts, source/cache identity, feature coverage,
and ID order. Existing campaign test features are under
`features_test_{reaction_smi,enzyme_smi,time}`. Do not refit anything from them.

Once all feature artifacts and model bundles are present, the declared model
variants can be encoded across all four GPUs:

```bash
../env/bin/python scripts/generalization_run_predictions.py \
  --campaign runs/generalization_20260919_2251 --gpus 0 1 2 3
```

This command uses features and IDs, without evaluation labels. It writes exact
commands, source/catalog/bundle hashes, score matrices, and a completion index.
Completed outputs are reused only after provenance checks. The primary method
is `seed42`; the other seeds, equal-weight ensemble, and ablations remain
prespecified diagnostics. They must not replace the primary method because of
test or Case 1 performance.

Evaluation is separate. The benchmark and external score manifests declare the
baseline, primary method, diagnostics, exact candidate order, and freeze hash.
The external evaluation keeps primary literature catalysts, patent catalysts,
and broad condition-dependent activity evidence distinct. P450 unlisted pairs
are never presented as experimentally inactive enzymes.

## Independent encoding and sparse storage

The mathematical retrieval representation is

```text
enzyme  = concat(sqrt(0.75) * graph_enzyme, 0.5 * enzyme_training_anchors)
reaction = concat(sqrt(0.75) * graph_reaction, 0.5 * reaction_training_anchors)
score = dot(reaction, enzyme)
```

Each side can be encoded before the other side is known. The anchor dictionary
contains only training reactions and proteins. Centers, neighbors, temperatures
and adjacency are fixed; no inference-candidate population statistics are fit.

The library supports a dense learned block plus a sparse anchor index, avoiding
a dense 7,489-dimensional vector for every stored enzyme:

```python
from horizyn.generalization_retrieval import GeneralizationDualEncoder

encoder, metadata = GeneralizationDualEncoder.from_bundle(
    "runs/generalization_20260919_2251/models/reaction_smi/seed42/bundle.json",
    device="cuda:0",
)
index = encoder.encode_enzyme_index(native_f3_protein_vectors, mean_prott5_vectors)
queries = encoder.encode_reactions(native_f3_query_vectors, raw_blocks, masks)
scores = encoder.score_index(queries, index)
```

Native F3 vectors must belong to the bundle's exact base checkpoint. Reaction
features must follow its training input policy and chemistry schema. In
particular, official ReactZyme F3 uses the complete participant collection as
`S>>S`; a physical equation must be converted before **all** feature extractors.
The generic wet-lab query path now records this policy explicitly and checks the
actual training catalog. It preserves directed inputs for models trained on
directed reactions.

Dot products use FP64 accumulation, rounded once to FP32, to stabilize near ties.
Raw anchor normalization and neighbor cosine calculations use FP64 reductions
with FP32 stored features. Candidate order resolves exact score ties. The final
production class and saved validation endpoint vectors were checked for exact
agreement before freezing the recipe.
