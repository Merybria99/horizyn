# Reproducing the exploratory second phase

Phase two was developed after observing the first frozen recipe's external failure. Its validation selection remains training/validation-only, but Case1 and P450 are development-exposed panels. Their subsequent results are retrospective exploratory checks. The separately registered nitrilase panel was evaluated after the phase2 freeze and did not confirm generalization; it had also been used elsewhere in the pre-existing workspace. This document preserves the phase2 execution protocol. Later experiments and complete outcomes are in `findings.md`.

All commands run from `horizyn` using the environment in [the original reproduction guide](generalization_reproduction.md). Original phase-one code/artifacts remain available; the phase-two experiment directory is `runs/generalization_20260919_2251/phase2`.

## Fixed model and selection rule

The selected method mixes **75% density-gated F3 residual** and **25% smooth training anchors**. The density gate independently measures each native F3 endpoint's maximum cosine similarity to training endpoints. It interpolates between training leave-one-out support quantiles25% and95%, clipped to[0,1]. Reaction quantiles use all training reactions; protein quantiles use the fixed4096-protein sample recorded by the fitter. This gate scales that endpoint's learned residual before normalization. The same selected rule transfers to each split, where thresholds and references are fitted from that split's training rows only.

The semantic component uses the raw mean ProtT5 protein features and the four training-centered reaction modalities. Protein endpoints transfer training associations from32 nearest training proteins at temperature0.03. Reaction endpoints use exponential responses to every training reaction at temperature0.03, with no hard reaction-neighbor cutoff. All references and centers are fixed before inference.

The actual independent endpoints are concatenated blocks:

```text
z_enzyme   = concat(sqrt(0.75) * density_enzyme, 0.5 * semantic_enzyme)
z_reaction = concat(sqrt(0.75) * density_reaction, 0.5 * semantic_reaction)
score      = FP32(FP64(z_reaction) dot FP64(z_enzyme))
```

The second-phase primary objective equally weights seen- and unseen-reaction all-positive MRR in both directions. All four aggregate/unseen safeguards against the original F3 baseline remain0.005. The selected recipe remains seed42; seeds17 and73 replicate the residual head while sharing the same base checkpoint and training dictionary. Density-only and smooth-only outputs are fixed controls, not substitutes selected on external performance.

FP32 residual forward passes were sensitive to endpoint batch shape. Before freezing this phase, the same31 density recipes and six composition weights were rerun with FP64 residual forward passes and normalization, returning FP32 endpoints. `density_gate_fp64` and `composition_fp64` are authoritative. Earlier directories remain numerical diagnostics. Native F3 feature caches were not changed.

## Two baseline score contracts

Every prediction artifact includes:

- `selected`: the declared phase-two method or fixed control.
- `baseline`: unchanged parent F3, one FP32 normalization followed by canonical FP64 dot accumulation.
- `baseline_fp64`: normalization-only sensitivity, one FP64 normalization of the same native F3 vectors, cast to FP32 before canonical scoring.

Compare against both baselines. Normalization effects must not be attributed to learned improvement. No normalization-only control was promoted after external evaluation.

## Authenticated bundles and inference

`phase2/freeze_ready_artifacts.json` inventories the exact selected states, nine split/seed tuples, original freeze, source hashes and verification records. The immutable phase-two recipe is `phase2/frozen_recipe.json`, frozen at2026-09-20 00:36:23UTC with SHA256 `624084b2535afdeb030e8f34adc1ce6cc3a340e6d5136858f408f41d94e09d76`. It approves those tuples before building final combined bundles. Each tuple binds split, seed, density bundle, smooth dictionary, parent bundle, feature-manifest identity and parent checkpoint identity. Labeling a seed73 artifact as the primary seed42 is rejected even though both checkpoints are individually approved.

For a fresh bundle directory, with the actual phase-two freeze path substituted:

```bash
../env/bin/python scripts/generalization_phase2_build_bundle.py \
  --density-bundle runs/generalization_20260919_2251/phase2/density_models/reaction_smi/seed42/density_bundle.json \
  --smooth-dictionary runs/generalization_20260919_2251/models/reaction_smi/seed42/anchors.pt \
  --base-bundle runs/generalization_20260919_2251/models/reaction_smi/seed42/bundle.json \
  --freeze runs/generalization_20260919_2251/phase2/frozen_recipe.json \
  --split reaction_smi --variant primary --output /fresh/bundle/directory
```

Valid variants are `primary`, `seed17`, `seed73`, `density_only` and `smooth_only`. The controls use seed42 artifacts. The builder does not fit or select any parameters.

`scripts/generalization_phase2_predict.py` accepts the same base/catalog/protein-mean/reaction-feature inputs as the first-phase predictor. `--input-receipt` defaults to `feature_bundle_receipt.json` beside the catalog. The original feature freeze and new model freeze are authenticated separately: existing feature receipts and arrays are not rewritten. `--reaction-key` selects the correctly declared native reaction array, such as `query_matched` for Case1. The command reads features and IDs, not evaluation labels.

Outputs include `scores.npz`, copied `catalog.json`, `support_diagnostics.npz`, and `complete.json`. The receipt records both freezes, exact input and output hashes, bundle identity, baseline contracts and source identity. Support diagnostics contain the independent density similarity and gate value for each endpoint.

## Reusable sparse indexing

```python
from horizyn.generalization_phase2 import ComposedPhase2Encoder

encoder, metadata = ComposedPhase2Encoder.from_bundle(bundle_path, device="cuda:0")
index = encoder.encode_enzyme_index(native_f3_enzymes, raw_protein_means, batch_size=512)
queries = encoder.encode_reactions(native_f3_reactions, raw_blocks, masks)
scores = encoder.score_index(queries, index)
```

The reusable index stores a dense512-dimensional learned block plus a sparse training-reaction block. The reaction semantic block remains dense, avoiding the first phase's disjoint hard-support zero scores. Neither endpoint depends on the current opposite-side query/candidate collection.

`composition_fp64/production_check.json` verifies the generic encoder against saved validation endpoints and positive ranks. `production_transfer_checks/{reaction_smi,enzyme_smi,time}.json` verifies all nine models on fixed feature subsets: full/subset/singleton endpoints and sparse/dense scores match bitwise. These guarantees concern the postprocessor on identical native feature arrays, not arbitrary upstream protein/chemistry extraction kernels. GPU reciprocal-rank aggregation may vary at about1e-7 from addition order; exact rank arrays are the stronger reproduction check.
