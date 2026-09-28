# V4 with biological supervision and unchanged inference architecture

The existing SLEEC F3 encoders, phase-2 heads, semantic dictionary and scoring formula are unchanged. Only the phase-2 training objective changes. No additional learned parameters or inference inputs are introduced.

The annotation regularizer pulls together training endpoints sharing observed EC prefixes, cofactor descriptors or coarse mechanism descriptors. Categories are balanced and uncertain annotation pairs receive lower weight. Missing labels are neutral. The three family terms retain a fixed divisor of three in all ablations.

EC ancestry receives increasing weights 0.125/0.25/0.5/1 at levels 1/2/3/4. Cofactor presence receives confidence 0.4; it is not proof of cofactor dependence. Mechanism descriptors are derived from atom-mapped bond changes, with mappings below confidence 0.5 omitted. They are not complete catalytic mechanisms.

The 512-dimensional learned vectors are retrieval representations regularized by biological relationships. No coordinate is assigned a biological label, and there is no positional encoding. In this phase-2 study the F3 residue attention queries remain frozen. A separate fresh-F3 study tests supervision reaching those queries.

| Variant | Primary target cells exceeded |
| --- | ---: |
| control | 14/14 |
| all_0p03 | 14/14 |
| all_0p1 | 14/14 |
| all_0p3 | 14/14 |
| without_ec | 14/14 |
| without_cofactor | 14/14 |
| without_mechanism | 14/14 |
| shuffled_0p1 | 14/14 |

| Benchmark / setting / metric | Target | control | all_0p03 | all_0p1 | all_0p3 | without_ec | without_cofactor | without_mechanism | shuffled_0p1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ReactZyme / reaction_smi / enzyme_to_reaction | 0.518000 | 0.534366 | 0.534361 | 0.534374 | 0.534856 | 0.534403 | 0.534355 | 0.534396 | 0.534797 |
| ReactZyme / reaction_smi / reaction_to_enzyme | 0.337000 | 0.400381 | 0.400389 | 0.400434 | 0.400446 | 0.400375 | 0.400434 | 0.400435 | 0.400379 |
| ReactZyme / enzyme_smi / enzyme_to_reaction | 0.956000 | 0.971355 | 0.962167 | 0.969682 | 0.970436 | 0.970192 | 0.971861 | 0.957091 | 0.970668 |
| ReactZyme / enzyme_smi / reaction_to_enzyme | 0.592000 | 0.666249 | 0.666249 | 0.666250 | 0.666096 | 0.666249 | 0.666250 | 0.666250 | 0.666346 |
| ReactZyme / time / enzyme_to_reaction | 0.690000 | 0.780669 | 0.790190 | 0.778417 | 0.815391 | 0.790247 | 0.779964 | 0.780596 | 0.777073 |
| ReactZyme / time / reaction_to_enzyme | 0.372000 | 0.538285 | 0.538307 | 0.538317 | 0.538467 | 0.538324 | 0.538317 | 0.538276 | 0.538192 |
| EnzymeMap / table1 / bedroc85 | 0.486600 | 0.572337 | 0.572334 | 0.572345 | 0.572211 | 0.572281 | 0.572320 | 0.572359 | 0.572161 |
| EnzymeMap / table1 / bedroc20 | 0.666900 | 0.755433 | 0.755440 | 0.755465 | 0.755415 | 0.755432 | 0.755447 | 0.755448 | 0.755257 |
| EnzymeMap / table1 / ef0.05 | 14.910000 | 16.859042 | 16.859042 | 16.859042 | 16.854659 | 16.859042 | 16.855755 | 16.859042 | 16.859042 |
| EnzymeMap / table1 / ef0.1 | 8.180000 | 8.958692 | 8.959788 | 8.959788 | 8.956501 | 8.956501 | 8.959788 | 8.959788 | 8.956501 |
| EnzymeMap / table2 / bedroc85 | 0.451400 | 0.528734 | 0.528740 | 0.528750 | 0.528603 | 0.528688 | 0.528721 | 0.528765 | 0.528533 |
| EnzymeMap / table2 / bedroc20 | 0.614300 | 0.726403 | 0.726413 | 0.726442 | 0.726390 | 0.726409 | 0.726420 | 0.726421 | 0.726198 |
| EnzymeMap / table2 / ef0.05 | 13.570000 | 16.418633 | 16.419622 | 16.420502 | 16.440803 | 16.420760 | 16.415882 | 16.426992 | 16.425855 |
| EnzymeMap / table2 / ef0.1 | 7.810000 | 8.793103 | 8.785624 | 8.794350 | 8.803076 | 8.794350 | 8.793103 | 8.793103 | 8.785624 |

**Contribution interpretation.** Compare `all_0p1` with each `without_*` row to remove one signal while holding the other coefficients fixed. The shuffled control permutes annotation rows within each endpoint, retaining category counts, annotation profiles and confidences. Compare against both the unchanged control and shuffled labels; a winning benchmark score alone does not demonstrate that the biological labels caused a meaningful improvement.

| Target | Reaction EC / cofactor / mechanism coverage | Enzyme EC / cofactor / mechanism coverage |
| --- | --- | --- |
| reaction_smi | 832/6977 / 2936/6977 / 262/6977 | 146718/147299 / 41683/147299 / 6114/147299 |
| enzyme_smi | 895/7386 / 3104/7386 / 273/7386 | 152088/152643 / 43919/152643 / 6049/152643 |
| time | 881/7281 / 3054/7281 / 265/7281 | 148920/149478 / 42458/149478 / 5943/149478 |
| enzymemap | 12603/12603 / 4915/12603 / 8440/12603 | 9666/9666 / 3127/9666 / 7531/9666 |

**Boundaries.** Same target-specific training associations and frozen V4 inputs; added annotation supervision is explicitly different from the no-biology control. Exact V4 phase-2 controls reproduce the saved parameters bit for bit. These are single-seed exploratory experiments with repeated tests. ReactZyme tie sensitivity and Case1 follow-up must be considered before interpreting small improvements as generalization.

[Machine-readable results](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/comparison.json) · [Frozen study protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/protocol.json) · [Exact control reproduction](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/control_reproduction.json)

## Numerical sensitivity

Large changes in Enzyme-Sim/Time E→R MRR must not be interpreted directly as biological gains. The diagnostic below allows a 1e-6 band around each positive score. Its pessimistic/optimistic bounds are secondary sensitivity checks, not replacements for the fixed official-style ranking metric.

| Target / variant | E→R MRR lower bound | Upper bound | Positive edges with near competitor |
| --- | ---: | ---: | ---: |
| reaction_smi / control | 0.534357 | 0.534372 | 13/14689 |
| reaction_smi / all_0p3 | 0.534852 | 0.534860 | 14/14689 |
| enzyme_smi / control | 0.948309 | 0.979007 | 542/8739 |
| enzyme_smi / all_0p3 | 0.948367 | 0.979065 | 542/8739 |
| time / control | 0.771579 | 0.819816 | 1285/12287 |
| time / all_0p3 | 0.771696 | 0.819933 | 1282/12287 |

For Time, the reported control→0.3 E→R MRR rises from about 0.7807 to 0.8154, but the 1e-6 lower bound moves only from about 0.77158 to 0.77170. Most of the headline change is therefore numerically fragile. Reaction-Sim is much less affected. The shuffled-label control also exceeds all primary targets: preserving the V4 wins alone does not establish the biological loss's value.

## Frozen Case1 follow-up

Complete-signal variants are qualified and frozen before their predictions. The panel was already examined during earlier development, so it remains retrospective.

| Variant / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC (123 sequences) |
| --- | ---: | ---: | ---: |
| all_0p03 / reaction_smi | 10 | 10 | 0.612875 |
| all_0p03 / enzyme_smi | 5 | 8 | 0.584656 |
| all_0p03 / time | 5 | 7 | 0.634039 |
| all_0p03 / enzymemap | 6 | 6 | 0.470606 |
| all_0p1 / reaction_smi | 10 | 10 | 0.612875 |
| all_0p1 / enzyme_smi | 5 | 8 | 0.584950 |
| all_0p1 / time | 5 | 7 | 0.632863 |
| all_0p1 / enzymemap | 6 | 6 | 0.470606 |
| all_0p3 / reaction_smi | 10 | 10 | 0.611699 |
| all_0p3 / enzyme_smi | 5 | 8 | 0.584362 |
| all_0p3 / time | 5 | 8 | 0.632863 |
| all_0p3 / enzymemap | 6 | 6 | 0.470899 |

All 144 entries and the separate 123-sequence analysis are preserved in each evaluated method's `case1` directory. Literature catalyst recovery and conditional activity discrimination are different outcomes. This one retrospective panel does not establish broad or prospective wet-lab generalization.
