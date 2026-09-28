# Cofactor vocabulary v2 and explicit unknown status

This is a versioned annotation/export change, not a new trained model. Legacy
10-class labels, configurations, checkpoints, and audited reconstruction files
are preserved. The new artifact contract is `circe_cofactor_v2`.

## Classes and missingness

There are 31 chemical groups plus `unknown` (32 columns, zero-based index 31).
The first ten groups retain their previous order:

```
NAD_NADP FAD_FMN PLP TPP CoA SAM FeS heme quinone thiol_lipoate
metal_Mg metal_Mn metal_Zn metal_Fe metal_Cu metal_Co metal_Ni
metal_Ca metal_K metal_Na metal_generic metal_divalent
cobalamin biotin molybdopterin_tungsten F420 F430 folate
pyruvoyl dipyrromethane phosphopantetheine unknown
```

The mapper fixes aliases including reduced flavins, cobalamin versus CoB,
ThDP, S-adenosylmethionine, ferriheme, and additional Fe–S cluster forms.
Unspecified metal annotations do not fabricate a particular element. Plain
`Pyruvate` remains unmapped: a participant name alone does not establish a
covalent pyruvoyl group. Original names and available evidence remain in the
native source table; extraction did not retain ChEBI cross-references or notes.

The three cofactor profiles have distinct roles:

| Profile | Known positives | When `unknown` is set |
|---|---|---|
| `native_cofactor` | Representative's own native annotation | No supported native class |
| `reaction_cofactor` | Descriptors of the representative's own permitted associations | No supported reaction-derived class |
| `cofactor` | Diagnostic union of known positives from both sources | Neither source has a supported class |

Known positives and `unknown` are mutually exclusive **within each profile**.
For example, native `unknown` can coexist with reaction-derived `NAD_NADP`,
but their combined profile is `NAD_NADP`, not `NAD_NADP + unknown`.
`unknown` covers both missing annotation and unsupported names. Boolean arrays
`native_cofactor_has_annotation` and `native_cofactor_has_unmapped` distinguish
these situations; the manifest also counts unmapped names.

`unknown` means annotation status, **not absence of a biological cofactor**.
Its target and mask are 1, but its confidence is 0 in every source. Row-wise
cofactor denominators are 0 for unknown-only rows, preventing them from entering
standalone biological pretraining as active examples. Known classes retain
positive-only masks: unobserved classes are not supervised negatives. Known
native confidence is 1 and reaction-derived confidence is 0.4; these are
provenance weights, not calibrated probabilities or experimental validation.
The eight mechanism groups and direct EC representation are unchanged.

Coverage statistics count **known classes only**, never `unknown`.

## Build and verify the dataset

Run from the project root:

```bash
bash scripts/run_horizyn1_cofactor_v2_annotations.sh
```

The CPU-only chain reuses the verified native table and exact-chemistry reaction
cache. It does not download/decompress UniProt again, extract sequence embeddings,
or start training. `RUN_ROOT`, `ANNOTATION_PYTHON`, `CACHED_FEATURES`, and
`CACHED_REACTIONS` can override paths. It uses a single-writer lock, conservative
resume/provenance checks, and an independent full source replay audit.

For the default reconstruction root `data/reconstructed/horizyn1_2023_05`:

- `annotations/cofactor_v2/`: versioned sparse reaction descriptors and manifest.
- `annotations/circe_v2_cofactor_v2/`: NPZ profiles, vocabulary/manifest, direct EC,
  candidate eligibility, and separate cluster-member EC evidence.
- `logs/circe_v2_cofactor_v2_label_integrity_audit.json`: full validation result.

The audit rechecks all native rows, representative sequence hashes, own source
associations, exact cached chemistry, profile values/confidences/denominators,
unknown exclusivity, known-only coverage, eligibility, and file provenance.
Passing establishes consistency, not experimental truth or retrieval improvement.

## Training integration and compatibility

The default output is explicitly `pair_scope=unsplit_inventory`. It is **not a
held-out-safe training target**. Before training/evaluation, regenerate targets
with `build_horizyn1_circe_v2_labels_v2.py`, training-only source associations,
and `--pair-scope train`; a declaration alone does not prove leakage freedom.
Do not point the current ReactZyme configurations at the full inventory.

For a newly configured head, `model.biofp.family_dims.cofactor` is **32**, not
10. Native and reaction source-specific heads, if used, also have 32 columns.
This is a label dimension, not the cofactor embedding block width. Do not load
a 10-output cofactor head into a 32-output head; initialize/retrain the changed
head explicitly. No existing training configuration is changed automatically.
The existing dataset and auxiliary losses consume the explicit confidence and
denominator arrays; missingness is therefore not learned as a biological class.

The negative-pool builder recognizes the embedded v2 contract and preserves
native and reaction evidence separately. Its evidence-overlap score is:

```
0.75 * Jaccard(mechanism positives)
+ 0.15 * Jaccard(native known cofactor positives)
+ 0.10 * Jaccard(reaction-derived known cofactor positives)
```

An empty comparison contributes zero evidence; `unknown` never contributes.
This replaces the degenerate intersection-of-positive-masks calculation for
the new schema only. Legacy soft-target files retain their old scores and
deterministic grouping. The weights are a heuristic, not an experimentally
validated improvement. Disjoint positive evidence does not prove inactivity;
promiscuity can still make an unobserved pair a true biological positive.

`build_annotation_negative_pools.py` rejects v2 inventory profiles for training.
Use its `--candidate-eligibility` CSV when constructing pools to retain the
cluster-member conflict guard. The general BioFP dataset loader does not itself
enforce split scope, so train-only provenance remains required for other entry
points. Cluster-member EC labels remain evidence for exclusions, not transferred
gold labels for the representative.
