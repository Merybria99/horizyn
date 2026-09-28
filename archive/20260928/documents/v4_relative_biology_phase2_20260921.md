# V4 with biological supervision and unchanged inference architecture

The existing SLEEC F3 encoders, phase-2 heads, semantic dictionary and scoring formula are unchanged. Only the phase-2 training objective changes. No additional learned parameters or inference inputs are introduced.

The relative biological regularizer compares within-category distances with distances to other annotated endpoints in the same family, with margin0.1. These references are not asserted activity negatives. Categories are balanced and uncertain annotation pairs receive lower weight. Missing labels are neutral. The three family terms retain a fixed divisor of three in all ablations.

EC ancestry receives increasing weights 0.125/0.25/0.5/1 at levels 1/2/3/4. Cofactor presence receives confidence 0.4; it is not proof of cofactor dependence. Mechanism descriptors are derived from atom-mapped bond changes, with mappings below confidence 0.5 omitted. They are not complete catalytic mechanisms.

The 512-dimensional learned vectors are retrieval representations regularized by biological relationships. No coordinate is assigned a biological label, and there is no positional encoding. In this phase-2 study the F3 residue attention queries remain frozen. A separate fresh-F3 study tests supervision reaching those queries.

| Variant | Primary target cells exceeded |
| --- | ---: |
| control | 14/14 |
| all_1 | 14/14 |
| all_3 | 14/14 |
| all_10 | 14/14 |
| without_ec | 14/14 |
| without_cofactor | 14/14 |
| without_mechanism | 14/14 |
| shuffled_10 | 14/14 |

| Benchmark / setting / metric | Target | control | all_1 | all_3 | all_10 | without_ec | without_cofactor | without_mechanism | shuffled_10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.534366 | 0.534777 | 0.534908 | 0.535844 | 0.535780 | 0.535936 | 0.533856 | 0.537467 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.400381 | 0.400412 | 0.400456 | 0.400608 | 0.399666 | 0.399821 | 0.400514 | 0.401378 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.971355 | 0.967373 | 0.968651 | 0.967817 | 0.971123 | 0.962799 | 0.958436 | 0.965748 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.666249 | 0.666234 | 0.666172 | 0.666156 | 0.666175 | 0.666154 | 0.666294 | 0.666503 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.780669 | 0.800614 | 0.789883 | 0.790308 | 0.780016 | 0.789927 | 0.790513 | 0.789401 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.538285 | 0.538317 | 0.538237 | 0.538722 | 0.538318 | 0.538697 | 0.538347 | 0.538556 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.572337 | 0.572096 | 0.571237 | 0.567451 | 0.567581 | 0.567549 | 0.571980 | 0.569992 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.755433 | 0.755322 | 0.755160 | 0.752611 | 0.752740 | 0.752690 | 0.755190 | 0.754108 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.859042 | 16.860921 | 16.872308 | 16.847596 | 16.847380 | 16.847596 | 16.861694 | 16.938829 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 8.958692 | 8.956501 | 8.980329 | 8.992382 | 8.992017 | 8.992017 | 8.967093 | 8.963422 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.528734 | 0.528464 | 0.527528 | 0.523364 | 0.523513 | 0.523478 | 0.528382 | 0.526214 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.726403 | 0.726275 | 0.726107 | 0.723300 | 0.723444 | 0.723392 | 0.726143 | 0.724848 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.418633 | 16.435718 | 16.452814 | 16.442394 | 16.450132 | 16.442652 | 16.442940 | 16.499071 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.793103 | 8.803076 | 8.824366 | 8.826859 | 8.826859 | 8.826859 | 8.800582 | 8.804694 |

**Contribution interpretation.** Compare `all_10` with each `without_*` row to remove one signal while holding the other coefficients fixed. The shuffled control permutes annotation rows within each endpoint, retaining category counts, annotation profiles and confidences. Compare against both the unchanged control and shuffled labels; a winning benchmark score alone does not demonstrate that the biological labels caused a meaningful improvement.

| Target | Reaction EC / cofactor / mechanism coverage | Enzyme EC / cofactor / mechanism coverage |
| --- | --- | --- |
| reaction_smi | 832/6977 / 2936/6977 / 262/6977 | 146718/147299 / 41683/147299 / 6114/147299 |
| enzyme_smi | 895/7386 / 3104/7386 / 273/7386 | 152088/152643 / 43919/152643 / 6049/152643 |
| time | 881/7281 / 3054/7281 / 265/7281 | 148920/149478 / 42458/149478 / 5943/149478 |
| enzymemap | 12603/12603 / 4915/12603 / 8440/12603 | 9666/9666 / 3127/9666 / 7531/9666 |

## Conditional contribution to enzyme screening

Each cell is `all_10` minus the indicated control. A positive value favors retaining the complete loss; a negative value favors that control. Removing a family also reduces the total regularization strength. Shuffling retains the weights but breaks biological assignment.

| Comparison / setting | Δ BEDROC85 | Δ BEDROC20 | Δ EF5 | Δ EF10 |
| --- | ---: | ---: | ---: | ---: |
| control / table1 | -0.004885 | -0.002822 | -0.011446 | +0.033690 |
| control / table2 | -0.005371 | -0.003104 | +0.023761 | +0.033756 |
| without_ec / table1 | -0.000130 | -0.000129 | +0.000216 | +0.000365 |
| without_ec / table2 | -0.000149 | -0.000145 | -0.007737 | +0.000000 |
| without_cofactor / table1 | -0.000098 | -0.000079 | +0.000000 | +0.000365 |
| without_cofactor / table2 | -0.000114 | -0.000092 | -0.000258 | +0.000000 |
| without_mechanism / table1 | -0.004529 | -0.002579 | -0.014098 | +0.025289 |
| without_mechanism / table2 | -0.005019 | -0.002843 | -0.000545 | +0.026276 |
| shuffled_10 / table1 | -0.002541 | -0.001497 | -0.091233 | +0.028961 |
| shuffled_10 / table2 | -0.002851 | -0.001548 | -0.056676 | +0.022164 |

These phase2-only results must be interpreted separately from fresh F3 training. They hold the original learned queries and all other F3 parameters fixed. Preserving benchmark wins does not demonstrate that the annotation penalty improves those original representations.


**Boundaries.** Same target-specific training associations and frozen V4 inputs; added annotation supervision is explicitly different from the no-biology control. Exact V4 phase-2 controls reproduce the saved parameters bit for bit. These are single-seed exploratory experiments with repeated tests. ReactZyme tie sensitivity and Case1 follow-up must be considered before interpreting small improvements as generalization.

[Machine-readable results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/comparison.json) · [Frozen study protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/protocol.json) · [Exact control reproduction](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/control_reproduction.json)

## Numerical sensitivity

Large changes in Enzyme-Sim/Time E→R MRR must not be interpreted directly as biological gains. The diagnostic below allows a 1e-6 band around each positive score. Its pessimistic/optimistic bounds are secondary sensitivity checks, not replacements for the fixed official-style ranking metric.

| Target / variant | E→R MRR lower bound | Upper bound | Positive edges with near competitor |
| --- | ---: | ---: | ---: |
| reaction_smi / control | 0.534357 | 0.534372 | 13/14689 |
| reaction_smi / all_1 | 0.534773 | 0.534781 | 12/14689 |
| reaction_smi / all_3 | 0.534904 | 0.534912 | 10/14689 |
| reaction_smi / all_10 | 0.535840 | 0.535849 | 16/14689 |
| reaction_smi / shuffled_10 | 0.537463 | 0.537471 | 12/14689 |
| enzyme_smi / control | 0.948309 | 0.979007 | 542/8739 |
| enzyme_smi / all_1 | 0.948367 | 0.979065 | 542/8739 |
| enzyme_smi / all_3 | 0.948367 | 0.979065 | 542/8739 |
| enzyme_smi / all_10 | 0.948325 | 0.979023 | 542/8739 |
| enzyme_smi / shuffled_10 | 0.947875 | 0.978572 | 543/8739 |
| time / control | 0.771579 | 0.819816 | 1285/12287 |
| time / all_1 | 0.771610 | 0.819884 | 1287/12287 |
| time / all_3 | 0.771501 | 0.819768 | 1286/12287 |
| time / all_10 | 0.771138 | 0.819442 | 1283/12287 |
| time / shuffled_10 | 0.770414 | 0.818719 | 1284/12287 |
## Frozen Case1 follow-up

Complete-signal variants are qualified and frozen before their predictions. The panel was already examined during earlier development, so it remains retrospective.

| Variant / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| all_1 / reaction_smi | 11 | 10 | 0.613757 |
| all_1 / enzyme_smi | 5 | 8 | 0.585538 |
| all_1 / time | 5 | 7 | 0.636390 |
| all_1 / enzymemap | 6 | 6 | 0.470018 |
| all_3 / reaction_smi | 11 | 10 | 0.615520 |
| all_3 / enzyme_smi | 5 | 8 | 0.584950 |
| all_3 / time | 5 | 8 | 0.636978 |
| all_3 / enzymemap | 6 | 6 | 0.470606 |
| all_10 / reaction_smi | 10 | 10 | 0.611993 |
| all_10 / enzyme_smi | 5 | 8 | 0.584068 |
| all_10 / time | 5 | 6 | 0.642563 |
| all_10 / enzymemap | 6 | 6 | 0.472663 |

All 144 entries and the separate 123-sequence analysis are preserved in each evaluated method's `case1` directory. Literature catalyst recovery and conditional activity discrimination are different outcomes. This one retrospective panel does not establish broad or prospective wet-lab generalization.
