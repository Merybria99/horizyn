# V4 with biological supervision and unchanged inference architecture

The existing SLEEC F3 encoders, phase-2 heads, semantic dictionary and scoring formula are unchanged. Only the phase-2 training objective changes. No additional learned parameters or inference inputs are introduced.

The relative biological regularizer compares within-category distances with distances to other annotated endpoints in the same family, with margin0.1. These references are not asserted activity negatives. Categories are balanced and uncertain annotation pairs receive lower weight. Missing labels are neutral. The three family terms retain a fixed divisor of three in all ablations.

EC ancestry receives increasing weights 0.125/0.25/0.5/1 at levels 1/2/3/4. Cofactor presence receives confidence 0.4; it is not proof of cofactor dependence. Mechanism descriptors are derived from atom-mapped bond changes, with mappings below confidence 0.5 omitted. They are not complete catalytic mechanisms.

The 512-dimensional learned vectors are retrieval representations regularized by biological relationships. No coordinate is assigned a biological label, and there is no positional encoding. In this phase-2 study the F3 residue attention queries remain frozen. A separate fresh-F3 study tests supervision reaching those queries.

| Variant | Primary target cells exceeded |
| --- | ---: |
| control | 14/14 |
| all_1 | 14/14 |
| without_ec | 14/14 |
| without_cofactor | 14/14 |
| without_mechanism | 14/14 |
| shuffled_1 | 14/14 |

| Benchmark / setting / metric | Target | control | all_1 | without_ec | without_cofactor | without_mechanism | shuffled_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.534366 | 0.534777 | 0.535076 | 0.534770 | 0.534101 | 0.535602 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.400381 | 0.400412 | 0.399127 | 0.400411 | 0.400387 | 0.400528 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.971355 | 0.967373 | 0.969806 | 0.971804 | 0.967430 | 0.970890 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.666249 | 0.666234 | 0.666240 | 0.666234 | 0.666294 | 0.665985 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.780669 | 0.800614 | 0.789794 | 0.815861 | 0.804038 | 0.790265 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.538285 | 0.538317 | 0.538324 | 0.538318 | 0.538280 | 0.538875 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.572337 | 0.572096 | 0.572124 | 0.572083 | 0.572242 | 0.572093 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.755433 | 0.755322 | 0.755351 | 0.755307 | 0.755364 | 0.755183 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.859042 | 16.860921 | 16.860921 | 16.860921 | 16.859042 | 16.874070 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 8.958692 | 8.956501 | 8.956501 | 8.956501 | 8.956501 | 8.958233 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.528734 | 0.528464 | 0.528496 | 0.528451 | 0.528631 | 0.528460 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.726403 | 0.726275 | 0.726308 | 0.726261 | 0.726324 | 0.726085 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.418633 | 16.435718 | 16.420760 | 16.435718 | 16.419255 | 16.430386 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.793103 | 8.803076 | 8.795596 | 8.795596 | 8.794350 | 8.785184 |

**Contribution interpretation.** Compare `all_1` with each `without_*` row to remove one signal while holding the other coefficients fixed. The shuffled control permutes annotation rows within each endpoint, retaining category counts, annotation profiles and confidences. Compare against both the unchanged control and shuffled labels; a winning benchmark score alone does not demonstrate that the biological labels caused a meaningful improvement.

| Target | Reaction EC / cofactor / mechanism coverage | Enzyme EC / cofactor / mechanism coverage |
| --- | --- | --- |
| reaction_smi | 832/6977 / 2936/6977 / 262/6977 | 146718/147299 / 41683/147299 / 6114/147299 |
| enzyme_smi | 895/7386 / 3104/7386 / 273/7386 | 152088/152643 / 43919/152643 / 6049/152643 |
| time | 881/7281 / 3054/7281 / 265/7281 | 148920/149478 / 42458/149478 / 5943/149478 |
| enzymemap | 12603/12603 / 4915/12603 / 8440/12603 | 9666/9666 / 3127/9666 / 7531/9666 |

## Conditional contribution to enzyme screening

Each cell is `all_1` minus the indicated control. A positive value favors retaining the complete loss; a negative value favors that control. Removing a family also reduces the total regularization strength. Shuffling retains the weights but breaks biological assignment.

| Comparison / setting | Δ BEDROC85 | Δ BEDROC20 | Δ EF5 | Δ EF10 |
| --- | ---: | ---: | ---: | ---: |
| control / table1 | -0.000240 | -0.000112 | +0.001878 | -0.002192 |
| control / table2 | -0.000270 | -0.000129 | +0.017085 | +0.009973 |
| without_ec / table1 | -0.000027 | -0.000029 | +0.000000 | +0.000000 |
| without_ec / table2 | -0.000032 | -0.000034 | +0.014959 | +0.007479 |
| without_cofactor / table1 | +0.000013 | +0.000015 | +0.000000 | +0.000000 |
| without_cofactor / table2 | +0.000013 | +0.000014 | +0.000000 | +0.007479 |
| without_mechanism / table1 | -0.000146 | -0.000042 | +0.001878 | +0.000000 |
| without_mechanism / table2 | -0.000167 | -0.000050 | +0.016463 | +0.008726 |
| shuffled_1 / table1 | +0.000003 | +0.000138 | -0.013149 | -0.001732 |
| shuffled_1 / table2 | +0.000004 | +0.000190 | +0.005332 | +0.017892 |

These phase2-only results must be interpreted separately from fresh F3 training. They hold the original learned queries and all other F3 parameters fixed. Preserving benchmark wins does not demonstrate that the annotation penalty improves those original representations.


**Boundaries.** Same target-specific training associations and frozen V4 inputs; added annotation supervision is explicitly different from the no-biology control. Exact V4 phase-2 controls reproduce the saved parameters bit for bit. These are single-seed exploratory experiments with repeated tests. ReactZyme tie sensitivity and Case1 follow-up must be considered before interpreting small improvements as generalization.

[Machine-readable results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_weight1_controls_20260921_v1/comparison.json) · [Frozen study protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_weight1_controls_20260921_v1/protocol.json) · [Exact control reproduction](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_weight1_controls_20260921_v1/control_reproduction.json)

## Numerical sensitivity

Large changes in Enzyme-Sim/Time E→R MRR must not be interpreted directly as biological gains. The diagnostic below allows a 1e-6 band around each positive score. Its pessimistic/optimistic bounds are secondary sensitivity checks, not replacements for the fixed official-style ranking metric.

| Target / variant | E→R MRR lower bound | Upper bound | Positive edges with near competitor |
| --- | ---: | ---: | ---: |
| reaction_smi / all_1 | 0.534773 | 0.534781 | 12/14689 |
| enzyme_smi / all_1 | 0.948367 | 0.979065 | 542/8739 |
| time / all_1 | 0.771610 | 0.819884 | 1287/12287 |
## Frozen Case1 follow-up

Methods shown below are qualified and frozen before their predictions. The panel was already examined during earlier development, so it remains retrospective.

| Variant / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| shuffled_1 / reaction_smi | 10 | 10 | 0.611111 |
| shuffled_1 / enzyme_smi | 5 | 8 | 0.583774 |
| shuffled_1 / time | 5 | 6 | 0.636978 |
| shuffled_1 / enzymemap | 6 | 6 | 0.470312 |
| without_cofactor / reaction_smi | 11 | 10 | 0.614345 |
| without_cofactor / enzyme_smi | 5 | 8 | 0.585538 |
| without_cofactor / time | 5 | 7 | 0.636390 |
| without_cofactor / enzymemap | 6 | 6 | 0.470312 |
| without_ec / reaction_smi | 10 | 10 | 0.613757 |
| without_ec / enzyme_smi | 5 | 8 | 0.584362 |
| without_ec / time | 5 | 8 | 0.637272 |
| without_ec / enzymemap | 6 | 6 | 0.470018 |
| without_mechanism / reaction_smi | 10 | 10 | 0.613169 |
| without_mechanism / enzyme_smi | 5 | 8 | 0.584950 |
| without_mechanism / time | 5 | 6 | 0.634627 |
| without_mechanism / enzymemap | 6 | 6 | 0.471193 |

All 144 entries and the separate 123-sequence analysis are preserved in each evaluated method's `case1` directory. Literature catalyst recovery and conditional activity discrimination are different outcomes. This one retrospective panel does not establish broad or prospective wet-lab generalization.
