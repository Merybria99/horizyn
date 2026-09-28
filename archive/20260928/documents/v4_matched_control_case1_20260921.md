# Case1 with matched fresh unannotated F3 controls

These controls use the fast validation cadence of the fresh biological runs. Every F3 biological coefficient is zero. The secondary variant applies relative weight0.1 only in the existing phase2. Each must exceed all14 benchmark comparator cells before this retrospective Case1 follow-up. The same original training associations, fixed epochs and V4 inference settings are retained.

| Control / target | Unique paper catalysts @25 /12 (123 sequences) | Paper entries @25 /15 (144 entries) | Conditional AUROC |
| --- | ---: | ---: | ---: |
| f3_biology / reaction_smi | 8 | 9 | 0.568783 |
| f3_biology / enzyme_smi | 5 | 8 | 0.582598 |
| f3_biology / time | 4 | 6 | 0.638154 |
| f3_biology / enzymemap | 6 | 6 | 0.471193 |
| f3_phase2_biology / reaction_smi | 8 | 9 | 0.568783 |
| f3_phase2_biology / enzyme_smi | 5 | 8 | 0.582598 |
| f3_phase2_biology / time | 4 | 6 | 0.637860 |
| f3_phase2_biology / enzymemap | 6 | 6 | 0.470606 |

These results distinguish the fresh-training trajectory from biological-loss changes. Case1 contains related constructs and heterogeneous literature assays; this is not prospective validation.

[Matched benchmark differences](v4_matched_reactzyme_controls_20260921.md) · [All qualification rows and source hashes](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_matched_reactzyme_control_20260921_v1/case1_control_comparison.json)
