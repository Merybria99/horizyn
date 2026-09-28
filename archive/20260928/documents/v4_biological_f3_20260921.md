# Fresh F3 biological supervision (attraction): unchanged V4 architecture

The fixed V4 recipe is retrained from scratch for each target, with biological supervision on the existing 512-dimensional embeddings. SLEEC remains frozen and retained. The first method applies the biological loss during F3 training; the second also applies it during the existing phase-2 refinement. Neither adds model parameters, inference inputs, heads, slots, or ensembles.

Both use seed 42, 10 epochs, batch 512, the original decoupled all-positive contrastive loss, and F3 biological weight 0.1. The optional phase-2 biological weight is 0.1. Epoch 10 and both phase-2 variants were declared before these tests. The 3× inference fusion, 100 phase-2 steps, residual multiplier 0.5 and semantic weight 0.4 are unchanged.

The attraction loss minimizes confidence-weighted within-category cosine distance.

| Method | Available primary metrics | Targets exceeded |
| --- | ---: | ---: |
| f3_biology | 14/14 | 13/14 |
| f3_phase2_biology | 14/14 | 13/14 |

| Benchmark / setting / metric | Comparator | Biology in F3 | Biology in F3 + phase 2 |
| --- | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.532584 | 0.532637 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.402602 | 0.402709 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.954584 | 0.954336 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.664804 | 0.664792 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.794002 | 0.806383 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.539845 | 0.539817 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.586887 | 0.586748 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.761057 | 0.760990 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.838782 | 16.824537 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 8.982937 | 8.982937 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.545024 | 0.544864 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.732766 | 0.732691 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.422155 | 16.421220 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.836450 | 8.836806 |

The following comparisons do not exceed their targets:

- f3_biology: enzyme_smi / enzyme_to_reaction = 0.954584, target 0.956000.

- f3_phase2_biology: enzyme_smi / enzyme_to_reaction = 0.954336, target 0.956000.

## Reaction-score numerical sensitivity

These secondary E→R bounds treat competitors within 10⁻⁶ of each positive score as potentially tied. They do not replace the official metric or its qualification decision. Wide intervals mean small changes in the official score should not be interpreted as equivalent changes in discrimination.

| Split / method | Pessimistic MRR | Optimistic MRR |
| --- | ---: | ---: |
| reaction_smi / f3_biology | 0.532580 | 0.532587 |
| reaction_smi / f3_phase2_biology | 0.532634 | 0.532641 |
| enzyme_smi / f3_biology | 0.948825 | 0.979525 |
| enzyme_smi / f3_phase2_biology | 0.948825 | 0.979525 |
| time / f3_biology | 0.774086 | 0.821715 |
| time / f3_phase2_biology | 0.774175 | 0.821845 |

[All tolerances and score hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/tie_sensitivity.json)

Canonicalizing the released participant sets explains much of this sensitivity. Canonical equality preserves stereochemistry and multiplicity, while removing atom-map identifiers and component ordering. Physical reaction sides are absent from these released inputs.

| Split | Positive associations with a near competitor | Those with identical canonical participants |
| --- | ---: | ---: |
| reaction_smi | 14 | 2 |
| enzyme_smi | 542 | 542 |
| time | 1286 | 1281 |

Different reaction identifiers can therefore demand distinct ranks for chemically equivalent released inputs. This diagnostic does not merge candidates, alter labels or change the official comparison. It limits the biological interpretation of small score movements.

[Reaction pairs, original strings and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/tie_chemistry_audit.json)

## Comparison with original V4

Paired query and rule-cluster resampling describe uncertainty for these fixed trained models. They do not measure variability across seeds or correct for repeated experimentation. Rules sharing a test query are joined before cluster resampling.

| Table / metric | Change vs V4, F3 + phase 2 | Rule-cluster bootstrap 95% interval |
| --- | ---: | --- |
| table1 / bedroc85 | +0.014411 | [+0.000371, +0.024082] |
| table1 / bedroc20 | +0.005557 | [-0.003742, +0.012666] |
| table1 / ef0.05 | -0.034505 | [-0.108351, +0.043589] |
| table1 / ef0.1 | +0.024244 | [-0.040600, +0.086247] |
| table2 / bedroc85 | +0.016130 | [+0.000491, +0.027101] |
| table2 / bedroc20 | +0.006288 | [-0.004090, +0.014685] |
| table2 / ef0.05 | +0.002587 | [-0.089784, +0.104402] |
| table2 / ef0.1 | +0.043703 | [-0.033494, +0.114533] |

## Representation diagnostic on unseen validation rules

This post-fit analysis uses validation annotations only for interpretation, never for training or selection. Relative category separation is (shuffled mean distance − observed category distance) / shuffled mean distance. Higher values indicate closer within-category neighborhoods relative to random assignments.

| Family | Annotated validation reactions /2652 | V4 separation | Biological F3 separation |
| --- | ---: | ---: | ---: |
| ec | 2652 | 0.741956 | 0.739230 |
| cofactor | 5 | 0.420895 | 0.424126 |
| mechanism | 1382 | 0.566445 | 0.558618 |

Changes in relative separation are ec: -0.002726, mechanism: -0.007828. Only five validation reactions receive recognized cofactor categories, so that comparison is too sparse to support a broad conclusion. Better screening scores must not be translated into a claim that the slots acquired explicit biochemical meanings.


## Contribution experiments

The matched EnzymeMap experiments remove EC, cofactor or mechanism supervision throughout both training stages, or shuffle all annotation profiles across training endpoints. The surviving loss terms keep their original coefficients and divisor of three. Their original association rows, model architecture, seed, batch and training duration are unchanged. Results are reported without selecting a control by test score.

| Annotation control, F3 + phase 2 | Table 1 BEDROC85 | Table 2 BEDROC85 |
| --- | ---: | ---: |
| without_ec | 0.589563 | 0.547953 |
| without_cofactor | 0.595922 | 0.555082 |
| without_mechanism | 0.585363 | 0.543179 |
| shuffled | 0.591985 | 0.551120 |

## Matched seed controls

Two additional seeds repeat both unannotated and biologically supervised F3 training with identical settings within each pair. This comparison uses the original unannotated phase-2 objective for both, isolating the F3 supervision change. No seed is selected by its test result.

| Seed / F3 supervision | Table 1 BEDROC85 | Table 2 BEDROC85 |
| --- | ---: | ---: |
| 43 / control | 0.498949 | 0.446490 |
| 43 / biology | 0.500147 | 0.448116 |
| 44 / control | 0.471203 | 0.411597 |
| 44 / biology | 0.470104 | 0.410138 |

The complete three-seed results show substantial variation across seeds, larger than most paired biological-loss gains. The three-seed mean is not an ensemble. [All metrics, variability and paired changes](v4_biology_seed_sensitivity_20260921.md).

[Existing-query behavior on 128 unseen validation proteins](v4_biological_query_audit_20260921.md) measures broad attention, gate usage and permutation invariance. It does not assign biological names or catalytic-site identities to individual queries.

## Case1

Only configurations with all 14 primary comparisons above target are frozen for this follow-up. The 144 entries are scored and the separate 123-sequence analysis is preserved. The literature panel was previously examined; it remains retrospective, with no new activity measurements.

| Method / target model | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional activity AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| f3_biology | not run: primary targets not all exceeded | — | — |
| f3_phase2_biology | not run: primary targets not all exceeded | — | — |

## Interpretation boundaries

EC labels describe functional ancestry. Cofactor labels describe participant presence, not proven dependence. Mechanism labels are coarse atom-mapped bond-change descriptors, not established complete catalytic mechanisms. Annotations are restricted to training endpoints and do not enter inference. They add supervision relative to unannotated models; matching downstream associations does not match all pretraining or annotation resources.

ReactZyme Enzyme-Sim and Time E→R scores can be sensitive to nearly tied reaction scores; the earlier phase-2 study documents this explicitly. Fresh-F3 tie checks must be inspected before attributing large changes to biology. Beating published point estimates does not establish broad wet-lab generalization.

[Architecture and vector interpretation](v4_biological_signal_architecture.md) · [Earlier phase-2-only contribution study](v4_biological_geometry_20260921.md)

[Protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/protocol.json) · [Results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/comparison.json) · [Architecture and data audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/architecture_and_data_audit.json)
