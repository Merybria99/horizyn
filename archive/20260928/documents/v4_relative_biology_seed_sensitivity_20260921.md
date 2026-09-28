# Relative biological loss: matched EnzymeMap seeds

Both previously declared relative-loss weights are replicated at seeds43 and44, alongside seed42. The unannotated controls use the same target data, architecture, optimization settings and duration. This comparison holds the existing phase2 objective unannotated in all arms, isolating the change to F3 supervision. All seeds are reported; their average is not an ensemble. Seeds change initialization, batch order and dropout.

Available model/seed results: 9/9.

| Seed / method | T1 BEDROC85 | T1 BEDROC20 | T1 EF5 | T1 EF10 | T2 BEDROC85 | T2 BEDROC20 | T2 EF5 | T2 EF10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 / control | 0.572337 | 0.755433 | 16.859042 | 8.958692 | 0.528734 | 0.726403 | 16.418633 | 8.793103 |
| 42 / relative_0p1 | 0.587057 | 0.765590 | 16.983633 | 9.057582 | 0.544873 | 0.737335 | 16.558643 | 8.912204 |
| 42 / relative_1 | 0.591894 | 0.762516 | 16.678959 | 8.962760 | 0.550344 | 0.733475 | 16.186411 | 8.807355 |
| 43 / control | 0.498949 | 0.699679 | 15.877905 | 8.610980 | 0.446490 | 0.661560 | 15.325992 | 8.361068 |
| 43 / relative_0p1 | 0.494716 | 0.696783 | 15.793488 | 8.606956 | 0.442285 | 0.658592 | 15.164698 | 8.363879 |
| 43 / relative_1 | 0.511714 | 0.706242 | 15.906405 | 8.702465 | 0.460548 | 0.668847 | 15.249023 | 8.493495 |
| 44 / control | 0.471203 | 0.654135 | 14.610618 | 8.235427 | 0.411597 | 0.609709 | 13.824816 | 7.972773 |
| 44 / relative_0p1 | 0.468156 | 0.654193 | 14.576142 | 8.408587 | 0.407617 | 0.609608 | 13.764181 | 8.156358 |
| 44 / relative_1 | 0.469648 | 0.648678 | 14.307871 | 8.191949 | 0.408957 | 0.602946 | 13.466970 | 7.929286 |

| Table / metric | Control mean ± SD | Relative0.1 mean ± SD | Relative1 mean ± SD |
| --- | ---: | ---: | ---: |
| table1 / bedroc85 | 0.514163 ± 0.052255 | 0.516643 ± 0.062410 | 0.524419 ± 0.062105 |
| table1 / bedroc20 | 0.703082 ± 0.050735 | 0.705522 ± 0.056210 | 0.705812 ± 0.056920 |
| table1 / ef0.05 | 15.782522 ± 1.127243 | 15.784421 ± 1.203771 | 15.631078 ± 1.209284 |
| table1 / ef0.1 | 8.601700 ± 0.361722 | 8.691042 ± 0.332568 | 8.619058 ± 0.392116 |
| table2 / bedroc85 | 0.462274 ± 0.060143 | 0.464925 ± 0.071374 | 0.473283 ± 0.071549 |
| table2 / bedroc20 | 0.665891 ± 0.058468 | 0.668512 ± 0.064439 | 0.668423 ± 0.065265 |
| table2 / ef0.05 | 15.189814 ± 1.302260 | 15.162507 ± 1.397232 | 14.967468 ± 1.381411 |
| table2 / ef0.1 | 8.375648 ± 0.410359 | 8.477480 ± 0.390518 | 8.410045 ± 0.444943 |

| Table / metric / loss | Mean paired change | Seeds improved /3 |
| --- | ---: | ---: |
| table1 / bedroc85 / relative_0p1 | +0.002480 | 1 |
| table1 / bedroc85 / relative_1 | +0.010256 | 2 |
| table1 / bedroc20 / relative_0p1 | +0.002440 | 2 |
| table1 / bedroc20 / relative_1 | +0.002729 | 2 |
| table1 / ef0.05 / relative_0p1 | +0.001899 | 1 |
| table1 / ef0.05 / relative_1 | -0.151444 | 1 |
| table1 / ef0.1 / relative_0p1 | +0.089342 | 2 |
| table1 / ef0.1 / relative_1 | +0.017358 | 2 |
| table2 / bedroc85 / relative_0p1 | +0.002651 | 1 |
| table2 / bedroc85 / relative_1 | +0.011009 | 2 |
| table2 / bedroc20 / relative_0p1 | +0.002621 | 1 |
| table2 / bedroc20 / relative_1 | +0.002532 | 2 |
| table2 / ef0.05 / relative_0p1 | -0.027306 | 1 |
| table2 / ef0.05 / relative_1 | -0.222346 | 0 |
| table2 / ef0.1 / relative_0p1 | +0.101833 | 3 |
| table2 / ef0.1 / relative_1 | +0.034398 | 2 |

Family-removal and shuffled-label studies remain seed42 experiments. Replicating the full loss does not replicate every family contribution. Benchmark tests have been inspected repeatedly; these comparisons are exploratory. The single-seed bootstrap intervals in other reports do not include this training variability.

[All records, source hashes and paired changes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_seed_replication_20260921_v1/seed_comparison.json) · [Relative1 contributions](v4_relative_biology_weight1_contributions_20260921.md) · [Attraction seed comparison](v4_biology_seed_sensitivity_20260921.md)
