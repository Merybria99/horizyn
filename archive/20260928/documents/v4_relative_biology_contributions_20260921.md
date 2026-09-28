# Biological contribution audit: relative loss

The unchanged V4 architecture is trained from scratch for 10 epochs, seed 42, with F3 biological weight 0.1. The existing 100-step phase2 uses weight 0.1 per retained family. Family removals apply throughout both stages, without renormalizing surviving family coefficients. The shuffled control permutes annotation profiles across training endpoints. Training associations and model components are unchanged. All declared controls are reported.

| Supervision | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 0.572337 | 0.755433 | 16.859042 | 8.958692 | 0.528734 | 0.726403 | 16.418633 | 8.793103 |
| all signals | 0.587095 | 0.765618 | 16.996782 | 9.057582 | 0.544910 | 0.737364 | 16.556981 | 8.912204 |
| without_ec | 0.586834 | 0.765643 | 17.020457 | 9.075676 | 0.545177 | 0.737406 | 16.607418 | 8.928939 |
| without_cofactor | 0.579242 | 0.760326 | 16.927182 | 9.024028 | 0.536224 | 0.731395 | 16.462525 | 8.860784 |
| without_mechanism | 0.585958 | 0.763724 | 16.899164 | 9.011310 | 0.544718 | 0.735465 | 16.447765 | 8.868365 |
| shuffled | 0.589697 | 0.767436 | 17.021714 | 9.015372 | 0.548546 | 0.740057 | 16.611649 | 8.859695 |

## Conditional contribution of each signal

Each value is all-signals minus the corresponding removed-family result. Positive values mean retaining the family helped that metric in this run. These are conditional differences, not an additive decomposition or evidence across seeds.

| Retained family | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ec | +0.000261 | -0.000024 | -0.023674 | -0.018095 | -0.000268 | -0.000042 | -0.050437 | -0.016736 |
| cofactor | +0.007853 | +0.005293 | +0.069600 | +0.033554 | +0.008686 | +0.005969 | +0.094456 | +0.051420 |
| mechanism | +0.001137 | +0.001894 | +0.097618 | +0.046272 | +0.000192 | +0.001899 | +0.109215 | +0.043839 |

Correct-minus-shuffled BEDROC85 is -0.002601 in Table1 and -0.003636 in Table2. Shuffling performs better in both, so the BEDROC85 gain cannot be attributed specifically to correct biological assignments.

EC ancestry, structural cofactor presence and coarse mapped bond changes are imperfect biological proxies. Missing labels are neutral. Test results have been examined repeatedly, so these comparisons are exploratory. Three-seed attraction-loss controls are complete and expose substantial variation across seeds; they are not seed replication of the relative loss.

[Architecture](v4_biological_signal_architecture.md) · [Seed sensitivity](v4_biology_seed_sensitivity_20260921.md)

- [V4 source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json)
- [all signals source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_20260921_v1/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_ec source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_contributions_20260921_v1/without_ec/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_cofactor source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_contributions_20260921_v1/without_cofactor/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_mechanism source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_contributions_20260921_v1/without_mechanism/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [shuffled source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_contributions_20260921_v1/shuffled/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
