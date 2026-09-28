# Capability Test Comparison

Generated on 2026-07-06 from stored benchmark artifacts in `horizyn/results/benchmarks`.

This table compares the two newest capability-branch test runs against the strongest stored configurations. Values are MRR. `R2E` means reaction-to-enzyme retrieval; `E2R` means enzyme-to-reaction retrieval.

Protocol note: the `protein-disjoint NR90` rows are useful reference points, but they are not strictly identical to the `original test` protocol used by the newest capability runs. Strict comparisons should be made mainly within the same protocol.

Capability-vector caveat: for the two newest capability runs, `205,057 / 394,459` benchmark candidate proteins did not have exported capability vectors and therefore used the configured `zero_with_mask` behavior, so the capability branch was masked off for those candidates.

Date note: dates are reported at day precision. Training dates come from the checkpoint/config artifact date when available; test dates come from the benchmark output folder name or summary artifact timestamp.

| Run | Train date | Test date | Enzyme stack | Reaction stack | Loss / objective | Protocol | Mean MRR | Harmonic MRR | R2E mean | Horizyn R2E | Time R2E | EnzSMI R2E | RxnSMI R2E | Time E2R | EnzSMI E2R | RxnSMI E2R |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R07 R5/U2 epoch10 | 2026-06-14 | 2026-06-14 | ProT5 mean pooled | ReactionT5v2 + UniMol2 attention | FullBatchMLNCELoss, observed pairs | protein-disjoint NR90 | 0.6377 | 0.5147 | 0.4657 | 0.6082 | 0.4144 | 0.2262 | 0.6140 | 0.7939 | 0.8216 | 0.9856 |
| R10 R5/U2 last | 2026-06-14 | 2026-06-14 | ProT5 mean pooled | ReactionT5v2 + UniMol2 attention | FullBatchMLNCELoss, observed pairs | protein-disjoint NR90 | 0.6339 | 0.5042 | 0.4587 | 0.6061 | 0.4071 | 0.2157 | 0.6059 | 0.7994 | 0.8175 | 0.9853 |
| R13 R5/U2/ChIRo highpair-top1000 | 2026-06-16 | 2026-06-16 | ProT5 mean pooled | ReactionT5v2 + UniMol2 + ChIRo attention, highpair-top1000 merged features | FullBatchMLNCELoss, observed pairs | original test | 0.5724 | 0.4196 | 0.3565 | 0.4040 | 0.3333 | 0.1744 | 0.5141 | 0.7504 | 0.8522 | 0.9785 |
| R23 hybrid all-known anchor | 2026-06-19 | 2026-06-19 | ProT5 mean + SLEEC-guided pooling + frozen Lorentz/tangent | ReactionT5v2 + UniMol2 + ChIRo attention, highpair-top1000 merged features | MultiAlignmentRetrievalLoss, all-known positives, anchor-balanced | original test | 0.5684 | 0.4167 | 0.3600 | 0.4179 | 0.3265 | 0.1712 | 0.5245 | 0.7510 | 0.8012 | 0.9863 |
| Cross-attn teacher epoch99 | 2026-06-30 | 2026-06-30 | ProT5 mean + learned cross-attention teacher branch | ReactionT5v2 + UniMol2 + ChIRo highpair | FullBatchMLNCELoss plus cross-attention teacher auxiliary loss; teacher weight 0.2, supervised/distill weights 1.0 | original test | 0.5453 | 0.3972 | 0.3369 | 0.3846 | 0.3134 | 0.1636 | 0.4858 | 0.7464 | 0.8460 | 0.8774 |
| Capability warmstart epoch27 | 2026-06-28 | 2026-06-29 | ProT5 mean + SLEEC + Lorentz + capability warmstart | ReactionT5v2 + UniMol2 + ChIRo highpair | MultiAlignmentRetrievalLoss, all-known positives, R2E/E2R 0.85/0.15, detached reactions, anchor 0.05 | original test | 0.5427 | 0.4033 | 0.3513 | 0.4052 | 0.3115 | 0.1672 | 0.5212 | 0.7300 | 0.7955 | 0.8681 |
| R12 R5/U2/ChIRo improved | 2026-06-16 | 2026-06-16 | ProT5 mean pooled | ReactionT5v2 + UniMol2 + ChIRo attention | FullBatchMLNCELoss, observed pairs | original test | 0.5424 | 0.3943 | 0.3418 | 0.3911 | 0.3143 | 0.1594 | 0.5023 | 0.7200 | 0.7769 | 0.9325 |
| NEW adapter/gate capability epoch23 | 2026-07-03 | 2026-07-05 | ProT5 mean + SLEEC + Lorentz + frozen capability vector + trainable capability adapter/gate | ReactionT5v2 + UniMol2 + ChIRo attention | MultiAlignmentRetrievalLoss, all-known positives, R2E-only, detached reactions, anchor 0.04, capability consistency 0.02 | original test | 0.5396 | 0.3968 | 0.3474 | 0.3994 | 0.3145 | 0.1605 | 0.5153 | 0.7294 | 0.7907 | 0.8675 |
| R24 hybrid EE depth3 | 2026-06-19 | 2026-06-19 | ProT5 mean + SLEEC-guided pooling + frozen Lorentz/tangent | ReactionT5v2 + UniMol2 + ChIRo attention, highpair-top1000 merged features | MultiAlignmentRetrievalLoss, anchor-balanced plus enzyme-enzyme EC depth3 term | original test | 0.5391 | 0.4005 | 0.3428 | 0.3892 | 0.3217 | 0.1666 | 0.4937 | 0.7518 | 0.7564 | 0.8944 |
| NEW stable capability epoch34 | 2026-07-03 | 2026-07-05 | ProT5 mean + SLEEC + Lorentz + frozen capability vector, no capability adapter | ReactionT5v2 + UniMol2 + ChIRo attention | MultiAlignmentRetrievalLoss, all-known positives, R2E-only, detached reactions, anchor 0.02 | original test | 0.5374 | 0.3989 | 0.3438 | 0.3998 | 0.3135 | 0.1652 | 0.4968 | 0.7282 | 0.7910 | 0.8676 |
| R25 SLEEC/Lorentz R2E hardneg best85 | 2026-06-22 | 2026-06-22 | ProT5 mean + SLEEC-guided pooling + frozen Lorentz/tangent | ReactionT5v2 + UniMol2 + ChIRo attention, highpair-top1000 merged features | MultiAlignmentRetrievalLoss, all-known positives, R2E/E2R 0.85/0.15, hard negatives | original test | 0.5177 | 0.3810 | 0.3322 | 0.3751 | 0.3089 | 0.1539 | 0.4911 | 0.7300 | 0.7316 | 0.8333 |
| R26 SLEEC/Lorentz R2E hardneg last | 2026-06-22 | 2026-06-22 | ProT5 mean + SLEEC-guided pooling + frozen Lorentz/tangent | ReactionT5v2 + UniMol2 + ChIRo attention, highpair-top1000 merged features | MultiAlignmentRetrievalLoss, all-known positives, R2E/E2R 0.85/0.15, hard negatives | original test | 0.5150 | 0.3880 | 0.3346 | 0.3768 | 0.3069 | 0.1637 | 0.4910 | 0.7308 | 0.7241 | 0.8115 |
| R08 R5/U2 last original | 2026-06-14 | 2026-06-14 | ProT5 mean pooled | ReactionT5v2 + UniMol2 attention | FullBatchMLNCELoss, observed pairs | original test | 0.5110 | 0.3592 | 0.3035 | 0.3395 | 0.2806 | 0.1448 | 0.4491 | 0.7108 | 0.7600 | 0.8920 |
| R05 R5/U2 epoch10 original | 2026-06-14 | 2026-06-14 | ProT5 mean pooled | ReactionT5v2 + UniMol2 attention | FullBatchMLNCELoss, observed pairs | original test | 0.5001 | 0.3483 | 0.2953 | 0.3324 | 0.2780 | 0.1379 | 0.4328 | 0.6809 | 0.7473 | 0.8912 |

## Source Artifacts

| Run | Source summary |
| --- | --- |
| NEW stable capability epoch34 | `horizyn/results/benchmarks/enzyme_only_capability_tuning_v3_reaction_demand_bio_composite/best_epoch34_original_test_20260705_134741/original_test/summary_wide.csv` |
| NEW adapter/gate capability epoch23 | `horizyn/results/benchmarks/enzyme_only_capability_adapter_gate_v3_reaction_demand_bio_composite/best_epoch23_original_test_20260705_134741/original_test/summary_wide.csv` |
| R13 R5/U2/ChIRo highpair-top1000 | `horizyn/results/benchmarks/horizyn-source-collapse-nr90-prott5-reactiont5v2-unimol2-chiro-attention-pseudo-improved-highpair-top1000-4gpu-20260615_193528/original_test/summary_wide.csv` |
| R23 hybrid all-known anchor | `horizyn/results/benchmarks/retrieval_hybrid_multi_alignment_allknown_anchor_4gpu/original_test_best_epoch98_sharded/*/summary_wide.csv` |
| R24 hybrid EE depth3 | `horizyn/results/benchmarks/retrieval_hybrid_multi_alignment_anchor_ee_depth3_4gpu/original_test_best_epoch98_sharded/*/summary_wide.csv` |
| R25/R26 SLEEC/Lorentz R2E hardneg | `horizyn/results/benchmarks/retrieval_prott5_sleec_lorentz_multimodal_r2e_allknown_hardneg_b512_4gpu/*/summary_wide.csv` |
| R07/R10 protein-disjoint NR90 references | `horizyn/results/benchmarks/horizyn-source-collapse-nr90-prott5-reactiont5v2-unimol2-attention-pseudo-*/protein_disjoint_nr90/summary_wide.csv` |
