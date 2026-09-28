# Biological contribution audit: relative loss

The unchanged V4 architecture is trained from scratch for 10 epochs, seed 42, with F3 biological weight 1. The existing 100-step phase2 uses weight 0.1 per retained family. Family removals apply throughout both stages, without renormalizing surviving family coefficients. The shuffled control permutes annotation profiles across training endpoints. Training associations and model components are unchanged. All declared controls are reported.

| Supervision | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 0.572337 | 0.755433 | 16.859042 | 8.958692 | 0.528734 | 0.726403 | 16.418633 | 8.793103 |
| all signals | 0.591840 | 0.762491 | 16.685533 | 8.962760 | 0.550281 | 0.733446 | 16.201370 | 8.807355 |
| without_ec | 0.584708 | 0.761846 | 16.854476 | 9.053650 | 0.542402 | 0.733120 | 16.435304 | 8.913614 |
| without_cofactor | 0.591790 | 0.762487 | 16.779528 | 8.977833 | 0.550734 | 0.733647 | 16.275433 | 8.816219 |
| without_mechanism | 0.575519 | 0.752571 | 16.670264 | 8.987624 | 0.531534 | 0.722150 | 16.150183 | 8.833380 |
| shuffled | 0.585440 | 0.761443 | 16.875350 | 8.930806 | 0.543932 | 0.732624 | 16.423047 | 8.759908 |

## Conditional contribution of each signal

Each value is all-signals minus the corresponding removed-family result. Positive values mean retaining the family helped that metric in this run. These are conditional differences, not an additive decomposition or evidence across seeds.

| Retained family | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ec | +0.007132 | +0.000644 | -0.168943 | -0.090890 | +0.007879 | +0.000326 | -0.233934 | -0.106259 |
| cofactor | +0.000050 | +0.000004 | -0.093995 | -0.015073 | -0.000454 | -0.000201 | -0.074063 | -0.008864 |
| mechanism | +0.016321 | +0.009920 | +0.015270 | -0.024864 | +0.018747 | +0.011296 | +0.051187 | -0.026025 |

Correct-minus-shuffled BEDROC85 is +0.006400 in Table1 and +0.006349 in Table2. These fixed-seed contrasts need replication before attributing an improvement to correct annotations.

## Uncertainty against shuffled labels

Paired reaction-rule cluster bootstrap intervals describe these fixed trained models. They do not include variation across seeds or adjustment for repeated comparisons.

| Table / metric | Correct minus shuffled | Rule-cluster bootstrap 95% interval |
| --- | ---: | --- |
| table1 / bedroc85 | +0.006400 | [-0.009031, +0.021678] |
| table1 / bedroc20 | +0.001048 | [-0.005959, +0.008150] |
| table1 / ef0.05 | -0.189817 | [-0.412424, -0.007479] |
| table1 / ef0.1 | +0.031954 | [-0.031349, +0.110347] |
| table2 / bedroc85 | +0.006349 | [-0.012712, +0.022735] |
| table2 / bedroc20 | +0.000822 | [-0.007590, +0.008770] |
| table2 / ef0.05 | -0.221678 | [-0.566695, -0.006648] |
| table2 / ef0.1 | +0.047446 | [-0.024232, +0.147449] |

[Bootstrap inputs and hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/enzymemap/paired_vs_shuffled.json)

EC ancestry, structural cofactor presence and coarse mapped bond changes are imperfect biological proxies. Missing labels are neutral. Test results have been examined repeatedly, so these comparisons are exploratory. Three-seed attraction-loss controls are complete and expose substantial variation across seeds; they are not seed replication of the relative loss.

[Architecture](v4_biological_signal_architecture.md) · [Seed sensitivity](v4_biology_seed_sensitivity_20260921.md)

- [V4 source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biological_geometry_20260921_v1/enzymemap/control/test_summary.json)
- [all signals source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_20260921_v1/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_ec source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_contributions_20260921_v1/without_ec/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_cofactor source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_contributions_20260921_v1/without_cofactor/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [without_mechanism source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_contributions_20260921_v1/without_mechanism/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
- [shuffled source](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_f3_weight1_contributions_20260921_v1/shuffled/followup/enzymemap/enzymemap/f3_phase2_biology/test_summary.json)
