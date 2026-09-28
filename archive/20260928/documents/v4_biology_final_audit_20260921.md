# Biological V4: final experiment audit

The completed minimal methods keep the original target-specific V4 F3, SLEEC, learned residue queries and inference recipe. Relative EC/cofactor/mechanism supervision acts only on the existing phase2 heads. Weights1 and3 preserve all14 primary benchmark comparisons and retain the strongest reported unique-sequence Case1 recovery in this study.

| Requirement | Evidence |
| --- | --- |
| No new architecture components | [Saved-state audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_relative_biology_phase2_20260921_v1/phase2_architecture_audit.json): all12 saved heads match original parameter names, shapes and feature manifests; zero additional parameter values |
| Both benchmark comparisons | [All methods and metrics](v4_relative_biology_phase2_20260921.md): all14 primary cells exceeded for weights1,3,10 |
| Same target-specific downstream associations | Original train/dev/test34427/7287/4642 rows and261907 screening IDs rechecked against preparation hashes; separate target-trained F3 models retained |
| Correct screening validation | BEDROC85, BEDROC20, EF5 and EF10 present for all three full-signal methods; no EnzymeMap MRR |
| Biological family contributions | [Weight1](v4_relative_biology_phase2_weight1_controls_20260921.md), [weight3](v4_relative_biology_phase2_weight3_controls_20260921.md), [Case1 controls](v4_relative_biology_phase2_case1_controls_20260921.md), and [matched seed/shuffle controls](v4_biology_additional_controls_20260921.md) |
| Learned vectors explained | [Architecture](v4_biological_signal_architecture.md), [trained-query measurements](v4_biological_query_audit_20260921.md), and [actual local gradients](v4_biology_gradient_audit_20260921.md) |
| Case1 on all144 entries | Entry counts checked; separate123-sequence catalyst recovery and AUROC independently recomputed from exported rankings for all12 main variant/target combinations |
| Continue runs past three hours | Final matched-control Case1 follow-up completed2026-09-21 15:46 UTC; useful runs were not stopped at the15:37 return boundary |
| Separate branch and findings | `research/v4-biological-loss-20260921`; [findings.md](../../../findings.md) contains detailed methods, metrics and controls |

The evidence supports retained published primary point-estimate wins and a small retrospective Case1 change. It does not establish prospective wet-lab generalization, identical total pretraining resources, a universal benefit from all annotation families, or statistically confirmed superiority after repeated experimentation. The extra unique catalyst moves from26 to25; original144-entry paper recovery remains10/15. Learned queries are content selectors, not positional encodings or named biochemical detectors.

All planned runs and follow-ups are complete. Generic phase2 training receipts defer validation externally; those evaluations are finished, and the declared fixed step100 requires no further selection.

[Requirement-by-requirement machine-readable evidence, hashes and independent metric checks](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/v4_biology_final_audit_20260921.json)
