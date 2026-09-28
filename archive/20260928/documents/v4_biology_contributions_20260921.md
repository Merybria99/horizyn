# Biological contribution audit: fresh EnzymeMap F3, seed 42

All signals use the unchanged V4 architecture and original training associations. Each biological variant is trained from scratch for 10 epochs, then receives the existing 100-step phase-2 fit. The three leave-one-family-out models remove that family during both stages. The shuffled control permutes whole annotation profiles across training endpoints. All predeclared outcomes are reported.

| Supervision | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 0.572337 | 0.755433 | 16.859042 | 8.958692 | 0.528734 | 0.726403 | 16.418633 | 8.793103 |
| all biological signals | 0.586748 | 0.760990 | 16.824537 | 8.982937 | 0.544864 | 0.732691 | 16.421220 | 8.836806 |
| without_ec | 0.589563 | 0.762362 | 16.833559 | 8.990102 | 0.547953 | 0.733961 | 16.399055 | 8.833597 |
| without_cofactor | 0.595922 | 0.767141 | 16.929709 | 8.993549 | 0.555082 | 0.739306 | 16.495442 | 8.840990 |
| without_mechanism | 0.585363 | 0.764691 | 16.996192 | 9.037841 | 0.543179 | 0.736734 | 16.634846 | 8.893998 |
| shuffled | 0.591985 | 0.764114 | 16.893723 | 8.968677 | 0.551120 | 0.735921 | 16.462261 | 8.813709 |

## Marginal contribution of each signal

Each entry below is all-signals minus the corresponding removed-family model. Positive means retaining the signal helped that metric; negative means it hurt. These are conditional differences, not an additive decomposition.

| Retained signal | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ec | -0.002815 | -0.001372 | -0.009021 | -0.007166 | -0.003089 | -0.001270 | +0.022164 | +0.003209 |
| cofactor | -0.009174 | -0.006151 | -0.105172 | -0.010613 | -0.010218 | -0.006615 | -0.074223 | -0.004184 |
| mechanism | +0.001385 | -0.003700 | -0.171655 | -0.054905 | +0.001686 | -0.004043 | -0.213626 | -0.057192 |

The full biological loss improves BEDROC85 relative to V4, but the shuffled control is stronger on that metric. Removing cofactors also improves both BEDROC scores. Mechanism supervision slightly helps BEDROC85 while harming BEDROC20 and enrichment in this run. Therefore these experiments do not establish that each annotation family helps, or that the full gain specifically depends on correct biological assignments.

The fresh relative-loss follow-up was declared after seeing this outcome. It requires within-category closeness relative to other annotated examples, with no added model components. That is a new exploratory hypothesis, not a reinterpretation of these results.

The family-removal results are single-seed, and held-out tests have been inspected repeatedly. Matched seed-43/44 attraction-loss controls are complete: [the three-seed audit](v4_biology_seed_sensitivity_20260921.md) shows substantial initialization variation and a small average biological-loss benefit. It does not replicate the family removals across seeds. The earlier phase-2-only audit covers all three ReactZyme splits; this full-retraining family audit covers EnzymeMap.

[Architecture](v4_biological_signal_architecture.md) · [Fresh-model benchmark report](v4_biological_f3_20260921.md)

- [V4 source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json)
- [all biological signals source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_20260921_v1/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_ec source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_contributions_20260921_v1/without_ec/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_cofactor source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_contributions_20260921_v1/without_cofactor/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_mechanism source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_contributions_20260921_v1/without_mechanism/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [shuffled source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_f3_contributions_20260921_v1/shuffled/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
