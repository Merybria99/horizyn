# Fresh F3 biological supervision (relative): unchanged V4 architecture

The fixed V4 recipe is retrained from scratch for each target, with biological supervision on the existing 512-dimensional embeddings. SLEEC remains frozen and retained. The first method applies the biological loss during F3 training; the second also applies it during the existing phase-2 refinement. Neither adds model parameters, inference inputs, heads, slots, or ensembles.

Both use seed 42, 10 epochs, batch 512, the original decoupled all-positive contrastive loss, and F3 biological weight 1. The optional phase-2 biological weight is 0.1. Epoch 10 and both phase-2 variants were declared before these tests. The 3× inference fusion, 100 phase-2 steps, residual multiplier 0.5 and semantic weight 0.4 are unchanged.

The relative loss compares within-category distance with distance to other annotated endpoints, using a confidence-weighted margin of 0.1. It compares relative separation instead of absolute within-category compactness. Category-complement examples are representation references, not asserted activity negatives.

| Method | Available primary metrics | Targets exceeded |
| --- | ---: | ---: |
| f3_biology | 14/14 | 14/14 |
| f3_phase2_biology | 14/14 | 14/14 |

| Benchmark / setting / metric | Comparator | Biology in F3 | Biology in F3 + phase 2 |
| --- | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.541618 | 0.541619 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.401638 | 0.401637 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.957043 | 0.957807 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.664423 | 0.664423 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.792256 | 0.803627 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.539148 | 0.539152 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.591894 | 0.591840 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.762516 | 0.762491 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.678959 | 16.685533 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 8.962760 | 8.962760 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.550344 | 0.550281 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.733475 | 0.733446 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.186411 | 16.201370 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.807355 | 8.807355 |

## Reaction-score numerical sensitivity

These secondary E→R bounds treat competitors within 10⁻⁶ of each positive score as potentially tied. They do not replace the official metric or its qualification decision. Wide intervals mean small changes in the official score should not be interpreted as equivalent changes in discrimination.

| Split / method | Pessimistic MRR | Optimistic MRR |
| --- | ---: | ---: |
| reaction_smi / f3_biology | 0.541615 | 0.541621 |
| reaction_smi / f3_phase2_biology | 0.541615 | 0.541622 |
| enzyme_smi / f3_biology | 0.948685 | 0.979389 |
| enzyme_smi / f3_phase2_biology | 0.948685 | 0.979389 |
| time / f3_biology | 0.773059 | 0.820580 |
| time / f3_phase2_biology | 0.773043 | 0.820564 |

[All tolerances and score hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/tie_sensitivity.json)

Canonicalizing the released participant sets explains much of this sensitivity. Canonical equality preserves stereochemistry and multiplicity, while removing atom-map identifiers and component ordering. Physical reaction sides are absent from these released inputs.

| Split | Positive associations with a near competitor | Those with identical canonical participants |
| --- | ---: | ---: |
| reaction_smi | 10 | 2 |
| enzyme_smi | 542 | 542 |
| time | 1288 | 1281 |

Different reaction identifiers can therefore demand distinct ranks for chemically equivalent released inputs. This diagnostic does not merge candidates, alter labels or change the official comparison. It limits the biological interpretation of small score movements.

[Reaction pairs, original strings and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/tie_chemistry_audit.json)

## Comparison with original V4

Paired query and rule-cluster resampling describe uncertainty for these fixed trained models. They do not measure variability across seeds or correct for repeated experimentation. Rules sharing a test query are joined before cluster resampling.

| Table / metric | Change vs V4, F3 + phase 2 | Rule-cluster bootstrap 95% interval |
| --- | ---: | --- |
| table1 / bedroc85 | +0.019504 | [+0.001210, +0.030644] |
| table1 / bedroc20 | +0.007057 | [-0.003659, +0.014287] |
| table1 / ef0.05 | -0.173509 | [-0.330776, +0.023755] |
| table1 / ef0.1 | +0.004067 | [-0.056194, +0.095445] |
| table2 / bedroc85 | +0.021546 | [-0.000204, +0.033442] |
| table2 / bedroc20 | +0.007043 | [-0.005928, +0.014538] |
| table2 / ef0.05 | -0.217263 | [-0.484505, +0.070157] |
| table2 / ef0.1 | +0.014252 | [-0.055145, +0.125170] |

## Representation diagnostic on unseen validation rules

This post-fit analysis uses validation annotations only for interpretation, never for training or selection. Relative category separation is (shuffled mean distance − observed category distance) / shuffled mean distance. Higher values indicate closer within-category neighborhoods relative to random assignments.

| Family | Annotated validation reactions /2652 | V4 separation | Biological F3 separation |
| --- | ---: | ---: | ---: |
| ec | 2652 | 0.741956 | 0.742609 |
| cofactor | 5 | 0.420895 | 0.425748 |
| mechanism | 1382 | 0.566445 | 0.569171 |

Changes in relative separation are ec: +0.000652, mechanism: +0.002725. Only five validation reactions receive recognized cofactor categories, so that comparison is too sparse to support a broad conclusion. Better screening scores must not be translated into a claim that the slots acquired explicit biochemical meanings.


## Contribution experiments

The matched EnzymeMap experiments remove EC, cofactor or mechanism supervision throughout both training stages, or shuffle all annotation profiles across training endpoints. The surviving loss terms keep their original coefficients and divisor of three. Their original association rows, model architecture, seed, batch and training duration are unchanged. Results are reported without selecting a control by test score.

| Annotation control, F3 + phase 2 | Table 1 BEDROC85 | Table 2 BEDROC85 |
| --- | ---: | ---: |
| without_ec | 0.584708 | 0.542402 |
| without_cofactor | 0.591790 | 0.550734 |
| without_mechanism | 0.575519 | 0.531534 |
| shuffled | 0.585440 | 0.543932 |

Both relative-loss weights have separate matched seed43/44 replications using the same completed unannotated controls. [All relative-loss seeds and metrics](v4_relative_biology_seed_sensitivity_20260921.md).

[Existing-query behavior on 128 unseen validation proteins](v4_biological_query_audit_20260921.md) measures broad attention, gate usage and permutation invariance. It does not assign biological names or catalytic-site identities to individual queries.

## Case1

Only configurations with all 14 primary comparisons above target are frozen for this follow-up. The 144 entries are scored and the separate 123-sequence analysis is preserved. The literature panel was previously examined; it remains retrospective, with no new activity measurements.

| Method / target model | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional activity AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| f3_biology / reaction_smi | 6 | 5 | 0.509112 |
| f3_biology / enzyme_smi | 5 | 8 | 0.595826 |
| f3_biology / time | 5 | 8 | 0.616990 |
| f3_biology / enzymemap | 6 | 6 | 0.485009 |
| f3_phase2_biology / reaction_smi | 6 | 5 | 0.509112 |
| f3_phase2_biology / enzyme_smi | 5 | 8 | 0.596120 |
| f3_phase2_biology / time | 5 | 8 | 0.616990 |
| f3_phase2_biology / enzymemap | 6 | 6 | 0.485009 |

## Interpretation boundaries

EC labels describe functional ancestry. Cofactor labels describe participant presence, not proven dependence. Mechanism labels are coarse atom-mapped bond-change descriptors, not established complete catalytic mechanisms. Annotations are restricted to training endpoints and do not enter inference. They add supervision relative to unannotated models; matching downstream associations does not match all pretraining or annotation resources.

ReactZyme Enzyme-Sim and Time E→R scores can be sensitive to nearly tied reaction scores; the earlier phase-2 study documents this explicitly. Fresh-F3 tie checks must be inspected before attributing large changes to biology. Beating published point estimates does not establish broad wet-lab generalization.

[Architecture and vector interpretation](v4_biological_signal_architecture.md) · [Earlier phase-2-only contribution study](v4_biological_geometry_20260921.md)

[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/protocol.json) · [Results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/comparison.json) · [Architecture and data audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/architecture_and_data_audit.json)
