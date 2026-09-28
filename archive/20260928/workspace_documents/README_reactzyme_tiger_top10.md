# ReactZyme and TIGER vs Horizyn Top 10

Last updated: 2026-08-25

## Comparison Protocol

- Results are held-out ReactZyme test metrics.
- `E->R` means enzyme-to-reaction retrieval; `R->E` means reaction-to-enzyme retrieval.
- Horizyn uses the official paper test candidate pools and canonical forward reactions.
- Horizyn MRR is the all-positive ReactZyme definition; H@k records whether any known positive is in the top k.
- Horizyn models are ranked by the arithmetic mean of the six split/direction MRR cells.
- Horizyn results are single-seed results from seed 42.
- Horizyn cells were freshly recomputed from the checkpoint/config paths recorded by the historical official tests. This report uses those current-code results rather than copying the older aggregate JSON.
- S0-S3 are exploratory paper-test evaluations of checkpoints selected only by validation. S0-S2 have all six cells; S3 currently has Time-Sim only and is excluded from the ranked top ten and six-cell macro comparison.
- M0, M1, and M3 are exploratory paper-test evaluations from the F3 geometry campaign, selected only by validation. M2 is still training and is excluded.
- The summary table uses TIGER Table 1. The six H@k tables use TIGER Appendix Tables 4, 6, 8, 10, 12, and 14 without reconciling paper-internal discrepancies.
- The appendix reports H@5/H@10 only for `TIGER (Ours)`, corresponding to the paper's primary ESM2Text configuration; ProtT3 is therefore absent from the H@k tables.
- The L-series loss campaign is excluded because only validation results were retained.

## Test Leaderboard

`Rank` orders only Horizyn models. Published methods are marked `Ref.`. Bold marks the best result in each column.

| Rank | Method | Time E->R | Time R->E | Enzyme-Sim E->R | Enzyme-Sim R->E | Reaction-Sim E->R | Reaction-Sim R->E | Macro |
|:--:|:--|--:|--:|--:|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.4100 | 0.1400 | 0.8110 | 0.2930 | 0.2010 | 0.1340 | 0.3315 |
| Ref. | TIGER ESM2Text | 0.6900 | 0.3660 | 0.9560 | 0.5920 | **0.5180** | 0.3190 | 0.5735 |
| Ref. | TIGER ProtT3 | 0.6830 | 0.3720 | 0.9400 | 0.5790 | 0.4720 | 0.3370 | 0.5638 |
| 1 | Q6 Bidirectional adapters | 0.8222 | **0.5935** | 0.9691 | **0.6940** | 0.4682 | **0.4089** | **0.6593** |
| 2 | Q5 Combined | 0.8303 | 0.5714 | 0.9708 | 0.6807 | 0.4681 | 0.4010 | 0.6537 |
| 3 | Q3 Dense chemistry | 0.8199 | 0.5714 | 0.9720 | 0.6807 | 0.4686 | 0.4010 | 0.6523 |
| 4 | Q2 Hard negatives | **0.8306** | 0.5714 | **0.9727** | 0.6807 | 0.4534 | 0.4010 | 0.6516 |
| 5 | Q4 Factorized modalities | 0.8202 | 0.5714 | 0.9655 | 0.6807 | 0.4687 | 0.4010 | 0.6512 |
| 6 | Q1 E->R adapter | 0.8087 | 0.5714 | 0.9658 | 0.6807 | 0.4677 | 0.4010 | 0.6492 |
| 7 | F3 Set chemistry | 0.7997 | 0.5644 | 0.9662 | 0.6800 | 0.4822 | 0.4009 | 0.6489 |
| 8 | Q0 Fresh F3 baseline | 0.8050 | 0.5714 | 0.9671 | 0.6807 | 0.4666 | 0.4010 | 0.6486 |
| 9 | S1 Normalized F3 | 0.8015 | 0.5635 | 0.9677 | 0.6840 | 0.4788 | 0.3824 | 0.6463 |
| 10 | M0 Audited F3 control | 0.8171 | 0.5695 | 0.9722 | 0.6853 | 0.4613 | 0.3694 | 0.6458 |
| Outside top 10 | S2 Bounded F3 | 0.8099 | 0.5682 | 0.9683 | 0.6819 | 0.4752 | 0.3697 | 0.6455 |
| Outside top 10 | M1 Modality-normalized F3 | 0.8100 | 0.5664 | 0.9707 | 0.6818 | 0.4524 | 0.3750 | 0.6427 |
| Outside top 10 | M3 Gated residual + consistency | 0.8056 | 0.5445 | 0.9712 | 0.6765 | 0.4882 | 0.3667 | 0.6421 |
| Outside top 10 | S0 F3 control | 0.8149 | 0.5688 | 0.9713 | 0.6826 | 0.4368 | 0.3609 | 0.6392 |
| Partial | S3 Bounded + chemistry dropout | 0.7981 | 0.5511 | - | - | - | - | - |

## TIGER-Style Hit@k Tables

These tables reproduce the split-by-direction layout of the TIGER appendix while adding the Horizyn top ten.

### Time Split, E->R

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.2905 | 0.5365 | 0.6586 | 0.4104 |
| Ref. | TIGER (ESM2Text) | 0.5810 | 0.8190 | 0.8740 | 0.6902 |
| 1 | Q6 Bidirectional adapters | 0.7334 | 0.9306 | 0.9510 | 0.8222 |
| 2 | Q5 Combined | 0.7482 | 0.9319 | **0.9530** | 0.8303 |
| 3 | Q3 Dense chemistry | 0.7313 | 0.9293 | 0.9525 | 0.8199 |
| 4 | Q2 Hard negatives | **0.7490** | **0.9328** | 0.9526 | **0.8306** |
| 5 | Q4 Factorized modalities | 0.7327 | 0.9283 | 0.9525 | 0.8202 |
| 6 | Q1 E->R adapter | 0.7100 | 0.9282 | 0.9517 | 0.8087 |
| 7 | F3 Set chemistry | 0.6974 | 0.9208 | 0.9478 | 0.7997 |
| 8 | Q0 Fresh F3 baseline | 0.7024 | 0.9284 | 0.9526 | 0.8050 |
| 9 | S1 Normalized F3 | 0.7007 | 0.9208 | 0.9498 | 0.8015 |
| 10 | M0 Audited F3 control | 0.7290 | 0.9238 | 0.9513 | 0.8171 |
| Outside top 10 | S2 Bounded F3 | 0.7143 | 0.9261 | 0.9504 | 0.8099 |
| Outside top 10 | S0 F3 control | 0.7232 | 0.9278 | 0.9506 | 0.8149 |
| Partial | S3 Bounded + chemistry dropout | 0.6964 | 0.9203 | 0.9482 | 0.7981 |

### Time Split, R->E

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.1678 | 0.3155 | 0.3960 | 0.1400 |
| Ref. | TIGER (ESM2Text) | 0.4536 | 0.6708 | 0.7676 | 0.3658 |
| 1 | Q6 Bidirectional adapters | **0.7733** | **0.8945** | **0.9260** | **0.5935** |
| 2 | Q5 Combined | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 3 | Q3 Dense chemistry | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 4 | Q2 Hard negatives | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 5 | Q4 Factorized modalities | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 6 | Q1 E->R adapter | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 7 | F3 Set chemistry | 0.7304 | 0.8667 | 0.9077 | 0.5644 |
| 8 | Q0 Fresh F3 baseline | 0.7415 | 0.8774 | 0.9127 | 0.5714 |
| 9 | S1 Normalized F3 | 0.7285 | 0.8694 | 0.9089 | 0.5635 |
| 10 | M0 Audited F3 control | 0.7350 | 0.8743 | 0.9123 | 0.5695 |
| Outside top 10 | S2 Bounded F3 | 0.7346 | 0.8766 | 0.9127 | 0.5682 |
| Outside top 10 | S0 F3 control | 0.7354 | 0.8751 | 0.9119 | 0.5688 |
| Partial | S3 Bounded + chemistry dropout | 0.7096 | 0.8648 | 0.9043 | 0.5511 |

### Enzyme Similarity Split, E->R

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.7267 | 0.9062 | 0.9487 | 0.8112 |
| Ref. | TIGER (ESM2Text) | 0.9308 | 0.9850 | 0.9916 | 0.9561 |
| 1 | Q6 Bidirectional adapters | 0.9447 | 0.9959 | 0.9970 | 0.9691 |
| 2 | Q5 Combined | 0.9478 | 0.9959 | 0.9970 | 0.9708 |
| 3 | Q3 Dense chemistry | 0.9503 | 0.9960 | 0.9971 | 0.9720 |
| 4 | Q2 Hard negatives | **0.9518** | **0.9961** | 0.9969 | **0.9727** |
| 5 | Q4 Factorized modalities | 0.9374 | 0.9959 | 0.9971 | 0.9655 |
| 6 | Q1 E->R adapter | 0.9378 | 0.9960 | 0.9971 | 0.9658 |
| 7 | F3 Set chemistry | 0.9385 | 0.9958 | 0.9971 | 0.9662 |
| 8 | Q0 Fresh F3 baseline | 0.9406 | 0.9959 | **0.9973** | 0.9671 |
| 9 | S1 Normalized F3 | 0.9414 | **0.9961** | 0.9971 | 0.9677 |
| 10 | M0 Audited F3 control | 0.9505 | 0.9963 | 0.9973 | 0.9722 |
| Outside top 10 | S2 Bounded F3 | 0.9428 | 0.9960 | **0.9973** | 0.9683 |
| Outside top 10 | S0 F3 control | 0.9486 | **0.9961** | **0.9973** | 0.9713 |

### Enzyme Similarity Split, R->E

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.4088 | 0.6892 | 0.7953 | 0.2930 |
| Ref. | TIGER (ESM2Text) | 0.7921 | 0.9408 | 0.9688 | 0.5921 |
| 1 | Q6 Bidirectional adapters | **0.9377** | **0.9860** | **0.9917** | **0.6940** |
| 2 | Q5 Combined | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 3 | Q3 Dense chemistry | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 4 | Q2 Hard negatives | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 5 | Q4 Factorized modalities | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 6 | Q1 E->R adapter | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 7 | F3 Set chemistry | 0.9180 | 0.9809 | 0.9879 | 0.6800 |
| 8 | Q0 Fresh F3 baseline | 0.9205 | 0.9816 | 0.9892 | 0.6807 |
| 9 | S1 Normalized F3 | 0.9250 | 0.9803 | 0.9866 | 0.6840 |
| 10 | M0 Audited F3 control | 0.9269 | 0.9822 | 0.9898 | 0.6853 |
| Outside top 10 | S2 Bounded F3 | 0.9205 | 0.9797 | 0.9873 | 0.6819 |
| Outside top 10 | S0 F3 control | 0.9231 | 0.9790 | 0.9873 | 0.6826 |

### Reaction Similarity Split, E->R

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.0912 | 0.2580 | 0.4213 | 0.1856 |
| Ref. | TIGER (ESM2Text) | **0.4155** | **0.6416** | **0.6827** | **0.5180** |
| 1 | Q6 Bidirectional adapters | 0.3839 | 0.5410 | 0.6496 | 0.4682 |
| 2 | Q5 Combined | 0.3855 | 0.5371 | 0.6191 | 0.4681 |
| 3 | Q3 Dense chemistry | 0.3834 | 0.5399 | 0.6253 | 0.4686 |
| 4 | Q2 Hard negatives | 0.3787 | 0.5166 | 0.5802 | 0.4534 |
| 5 | Q4 Factorized modalities | 0.3836 | 0.5366 | 0.6228 | 0.4687 |
| 6 | Q1 E->R adapter | 0.3815 | 0.5346 | 0.6296 | 0.4677 |
| 7 | F3 Set chemistry | 0.3902 | 0.5651 | 0.6571 | 0.4822 |
| 8 | Q0 Fresh F3 baseline | 0.3826 | 0.5320 | 0.6225 | 0.4666 |
| 9 | S1 Normalized F3 | 0.3873 | 0.5791 | 0.6673 | 0.4788 |
| 10 | M0 Audited F3 control | 0.3553 | 0.5762 | 0.6746 | 0.4613 |
| Outside top 10 | S2 Bounded F3 | 0.3775 | 0.5775 | 0.6360 | 0.4752 |
| Outside top 10 | S0 F3 control | 0.3388 | 0.5217 | 0.6195 | 0.4368 |

### Reaction Similarity Split, R->E

| Rank | Method | H@1 | H@5 | H@10 | MRR |
|:--:|:--|--:|--:|--:|--:|
| Ref. | ReactZyme (UniMol-3D + ESM) | 0.0924 | 0.1332 | 0.1790 | 0.0943 |
| Ref. | TIGER (ESM2Text) | 0.4305 | 0.6113 | 0.6994 | 0.3185 |
| 1 | Q6 Bidirectional adapters | **0.5337** | **0.7150** | **0.7772** | **0.4089** |
| 2 | Q5 Combined | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 3 | Q3 Dense chemistry | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 4 | Q2 Hard negatives | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 5 | Q4 Factorized modalities | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 6 | Q1 E->R adapter | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 7 | F3 Set chemistry | **0.5337** | 0.6969 | 0.7668 | 0.4009 |
| 8 | Q0 Fresh F3 baseline | 0.5285 | 0.6917 | 0.7383 | 0.4010 |
| 9 | S1 Normalized F3 | 0.4922 | 0.6528 | 0.7280 | 0.3824 |
| 10 | M0 Audited F3 control | 0.4663 | 0.6373 | 0.7228 | 0.3694 |
| Outside top 10 | S2 Bounded F3 | 0.4767 | 0.6347 | 0.7332 | 0.3697 |
| Outside top 10 | S0 F3 control | 0.4637 | 0.6321 | 0.7228 | 0.3609 |

## Main Results

1. Q6 Bidirectional adapters has the strongest six-cell macro MRR among the evaluated Horizyn models at `0.6593`, compared with TIGER ESM2Text at `0.5735`.
2. Relative to Q5, Q6 improves R->E MRR on Time (`+0.0221`), Enzyme-Sim (`+0.0132`), and Reaction-Sim (`+0.0079`).
3. Q6 gives up E->R MRR on Time (`-0.0081`) and Enzyme-Sim (`-0.0017`), while Reaction-Sim is effectively unchanged (`+0.0001`).
4. Q6 exceeds TIGER ESM2Text in five of six MRR cells. TIGER remains strongest on Reaction-Sim E->R (`0.5180` versus `0.4682`).
5. M0 enters the top ten at macro MRR `0.6458`; M3 has the strongest Reaction-Sim E->R result among M0/M1/M3 at `0.4882` but loses enough on the other cells to remain outside the top ten.
6. M1 has the strongest M-series Reaction-Sim R->E result at `0.3750`.
7. S3 currently underperforms S0-S2 on both Time-Sim directions; its other splits are not yet represented.
8. Hit@5 and Hit@10 are measured directly from checkpoint rankings; they are not interpolated from Hit@1 or MRR.
9. The appendix reference rows are transcribed independently from the Table 1 summary because the paper reports different ReactZyme values in a few cells.

## Model Definitions

| ID | Definition |
|:--|:--|
| F0 | Directional-delta reaction composition control. |
| F1 | UniMol2 and ChIRo participants treated as one unordered molecule set. |
| F3 | F1 plus train-fitted molecule-set descriptors, fingerprints, and cofactors. |
| F6 | Factorized reaction modalities plus Rhea direction and reaction-center features. |
| M0 | Audited F3 control rerun using the F3 geometry campaign protocol. |
| M1 | M0 plus scale-preserving modality L2 normalization. |
| M3 | Factorized reaction representation with a gated residual and detached chemistry-free consistency. |
| Q0 | Freshly trained F3 baseline. |
| Q1 | Q0 plus a frozen-base E->R residual reaction adapter. |
| Q2 | Q1 plus mined E->R hard-negative training. |
| Q3 | Q1 plus dense directional and reaction-center chemistry. |
| Q4 | Q1 plus factorized raw ReactionT5v2, UniMol2, and ChIRo inputs. |
| Q5 | Combined Q1, Q2, Q3, and Q4 changes. |
| Q6 | Frozen Q0/F3 parent with Q5 dense/factorized chemistry and hard negatives, jointly training direction-specific E->R reaction and R->E enzyme adapters. |
| S0 | Exact F3 single-head attention control on the audited protocol. |
| S1 | S0 with scale-preserving modality L2 normalization. |
| S2 | S1 with prior-bounded adaptive single-head attention. |
| S3 | S2 with chemistry-only dropout at probability `0.25`. |

## Provenance

- Recomputed Hit@k artifacts: `horizyn/runs/reactzyme_tiger_top10_hitk_v1/eval/`
- Recomputed evaluation logs: `horizyn/runs/reactzyme_tiger_top10_hitk_v1/logs/`
- Complete ReactZyme ablation ledger: `documents/README_reactzyme_all_ablations.md`
- F-series source tests: `horizyn/runs/reactzyme_reaction_features_v1/eval/`
- Q-series source tests: `horizyn/runs/reactzyme_e2r_pareto_v1/eval/seed42/`
- Q6 bidirectional-adapter source tests: `horizyn/runs/reactzyme_bidirectional_adapter_v1/eval/seed42/`
- S-series exploratory tests: `horizyn/runs/reactzyme_f3_shortcut_ablation_v1/results/exploratory_test_finished/`
- S-series validation-selected checkpoints: `horizyn/runs/reactzyme_f3_shortcut_ablation_v1/results/validation/`
- M-series exploratory tests: `horizyn/runs/reactzyme_f3_geometry_v1/results/exploratory_test_user_authorized/`
- M-series validation-selected checkpoints: `horizyn/runs/reactzyme_f3_geometry_v1/results/validation/`
- TIGER paper: <https://arxiv.org/pdf/2605.24489>
