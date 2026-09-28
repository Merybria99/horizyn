# Fresh F3 biological supervision (relative): unchanged V4 architecture

The fixed V4 recipe is retrained from scratch for each target, with biological supervision on the existing 512-dimensional embeddings. SLEEC remains frozen and retained. The first method applies the biological loss during F3 training; the second also applies it during the existing phase-2 refinement. Neither adds model parameters, inference inputs, heads, slots, or ensembles.

Both use seed 42, 10 epochs, batch 512, the original decoupled all-positive contrastive loss, and F3 biological weight 0.1. The optional phase-2 biological weight is 0.1. Epoch 10 and both phase-2 variants were declared before these tests. The 3× inference fusion, 100 phase-2 steps, residual multiplier 0.5 and semantic weight 0.4 are unchanged.

The relative loss compares within-category distance with distance to other annotated endpoints, using a confidence-weighted margin of 0.1. It compares relative separation instead of absolute within-category compactness. Category-complement examples are representation references, not asserted activity negatives.

| Method | Available primary metrics | Targets exceeded |
| --- | ---: | ---: |
| f3_biology | 14/14 | 14/14 |
| f3_phase2_biology | 14/14 | 14/14 |

| Benchmark / setting / metric | Comparator | Biology in F3 | Biology in F3 + phase 2 |
| --- | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.546721 | 0.546759 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.398129 | 0.398132 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.971280 | 0.971723 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.665099 | 0.665099 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.796441 | 0.796079 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.539334 | 0.539316 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.587057 | 0.587095 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.765590 | 0.765618 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.983633 | 16.996782 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 9.057582 | 9.057582 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.544873 | 0.544910 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.737335 | 0.737364 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.558643 | 16.556981 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.912204 | 8.912204 |

## Reaction-score numerical sensitivity

These secondary E→R bounds treat competitors within 10⁻⁶ of each positive score as potentially tied. They do not replace the official metric or its qualification decision. Wide intervals mean small changes in the official score should not be interpreted as equivalent changes in discrimination.

| Split / method | Pessimistic MRR | Optimistic MRR |
| --- | ---: | ---: |
| reaction_smi / f3_biology | 0.546718 | 0.546724 |
| reaction_smi / f3_phase2_biology | 0.546755 | 0.546762 |
| enzyme_smi / f3_biology | 0.948710 | 0.979410 |
| enzyme_smi / f3_phase2_biology | 0.948710 | 0.979410 |
| time / f3_biology | 0.774047 | 0.822818 |
| time / f3_phase2_biology | 0.774006 | 0.822778 |

[All tolerances and score hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/tie_sensitivity.json)

Canonicalizing the released participant sets explains much of this sensitivity. Canonical equality preserves stereochemistry and multiplicity, while removing atom-map identifiers and component ordering. Physical reaction sides are absent from these released inputs.

| Split | Positive associations with a near competitor | Those with identical canonical participants |
| --- | ---: | ---: |
| reaction_smi | 8 | 2 |
| enzyme_smi | 542 | 542 |
| time | 1284 | 1281 |

Different reaction identifiers can therefore demand distinct ranks for chemically equivalent released inputs. This diagnostic does not merge candidates, alter labels or change the official comparison. It limits the biological interpretation of small score movements.

[Reaction pairs, original strings and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/tie_chemistry_audit.json)

## Representation diagnostic on unseen validation rules

This post-fit analysis uses validation annotations only for interpretation, never for training or selection. Relative category separation is (shuffled mean distance − observed category distance) / shuffled mean distance. Higher values indicate closer within-category neighborhoods relative to random assignments.

| Family | Annotated validation reactions /2652 | V4 separation | Biological F3 separation |
| --- | ---: | ---: | ---: |
| ec | 2652 | 0.741956 | 0.740402 |
| cofactor | 5 | 0.420895 | 0.429202 |
| mechanism | 1382 | 0.566445 | 0.557459 |

Changes in relative separation are ec: -0.001554, mechanism: -0.008987. Only five validation reactions receive recognized cofactor categories, so that comparison is too sparse to support a broad conclusion. Better screening scores must not be translated into a claim that the slots acquired explicit biochemical meanings.


## Contribution experiments

The matched EnzymeMap experiments remove EC, cofactor or mechanism supervision throughout both training stages, or shuffle all annotation profiles across training endpoints. The surviving loss terms keep their original coefficients and divisor of three. Their original association rows, model architecture, seed, batch and training duration are unchanged. Results are reported without selecting a control by test score.

| Annotation control, F3 + phase 2 | Table 1 BEDROC85 | Table 2 BEDROC85 |
| --- | ---: | ---: |
| without_ec | 0.586834 | 0.545177 |
| without_cofactor | 0.579242 | 0.536224 |
| without_mechanism | 0.585958 | 0.544718 |
| shuffled | 0.589697 | 0.548546 |

Both relative-loss weights have separate matched seed43/44 replications using the same completed unannotated controls. [All relative-loss seeds and metrics](v4_relative_biology_seed_sensitivity_20260921.md).

[Existing-query behavior on 128 unseen validation proteins](v4_biological_query_audit_20260921.md) measures broad attention, gate usage and permutation invariance. It does not assign biological names or catalytic-site identities to individual queries.

## Case1

Only configurations with all 14 primary comparisons above target are frozen for this follow-up. The 144 entries are scored and the separate 123-sequence analysis is preserved. The literature panel was previously examined; it remains retrospective, with no new activity measurements.

| Method / target model | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional activity AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| f3_biology / reaction_smi | 7 | 7 | 0.551146 |
| f3_biology / enzyme_smi | 5 | 7 | 0.610229 |
| f3_biology / time | 5 | 8 | 0.632863 |
| f3_biology / enzymemap | 6 | 6 | 0.478248 |
| f3_phase2_biology / reaction_smi | 7 | 7 | 0.551146 |
| f3_phase2_biology / enzyme_smi | 5 | 7 | 0.610229 |
| f3_phase2_biology / time | 5 | 8 | 0.633157 |
| f3_phase2_biology / enzymemap | 6 | 6 | 0.478542 |

## Interpretation boundaries

EC labels describe functional ancestry. Cofactor labels describe participant presence, not proven dependence. Mechanism labels are coarse atom-mapped bond-change descriptors, not established complete catalytic mechanisms. Annotations are restricted to training endpoints and do not enter inference. They add supervision relative to unannotated models; matching downstream associations does not match all pretraining or annotation resources.

ReactZyme Enzyme-Sim and Time E→R scores can be sensitive to nearly tied reaction scores; the earlier phase-2 study documents this explicitly. Fresh-F3 tie checks must be inspected before attributing large changes to biology. Beating published point estimates does not establish broad wet-lab generalization.

[Architecture and vector interpretation](v4_biological_signal_architecture.md) · [Earlier phase-2-only contribution study](v4_biological_geometry_20260921.md)

[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/protocol.json) · [Results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/comparison.json) · [Architecture and data audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/architecture_and_data_audit.json)
