# 6 shared configurations winning both primary benchmark comparisons

Updated 2026-09-21T11:54:25.654022+00:00. Each version exceeds all **14 primary competitor point estimates**: six ReactZyme MRR cells and eight EnzymeMap screening cells. Case 1 has been evaluated for every version on all 144 entries, with a separate analysis of its 123 unique sequences.

**User-preferred configuration: V4.** Recorded after reviewing the completed results. [Detailed method and frozen checkpoints](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_recipe_alpha04_cap05_v1.md). This preference leaves the original evaluation protocols and artifacts unchanged.

All versions use the SLEEC residue-view dual encoder, beta 5, seed 42 and 100 full-graph positive-CE phase-2 updates. V1–V4 share fresh target-specific F3 fits with batch 512 and a fixed epoch-10 checkpoint; they apply a 3× internal fusion multiplier at inference. The stronger-fusion family uses a separate set of fresh fits, batch 1024 and an initial residue contribution of 0.3, with no inference fusion adjustment. Versions within a family are correlated configurations, not independent statistical replications. There is no ensemble or hubness correction.

| Version | Base training | Inference fusion multiplier | Semantic weight | Residual cap | Detailed method and Case1 report |
| --- | --- | ---: | ---: | ---: | --- |
| V1 | batch 512; scale 0.1; epoch 10 | 3.0 | 0.25 | 1.0 | [shared_recipe_beta5_b512_e10_fusion3_v2](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_recipe_beta5_b512_e10_fusion3_v2.md) |
| V2 | batch 512; scale 0.1; epoch 10 | 3.0 | 0.4 | 1.0 | [shared_recipe_alpha04_cap1_v1](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_recipe_alpha04_cap1_v1.md) |
| V3 | batch 512; scale 0.1; epoch 10 | 3.0 | 0.5 | 1.0 | [shared_recipe_alpha05_cap1_v1](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_recipe_alpha05_cap1_v1.md) |
| V4 | batch 512; scale 0.1; epoch 10 | 3.0 | 0.4 | 0.5 | [shared_recipe_alpha04_cap05_v1](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_recipe_alpha04_cap05_v1.md) |
| V5 | batch 1024; scale 0.3; epoch 10 | 1.0 | 0.25 | 1.0 | [shared_fusion03_beta5_b1024_epoch10_fixed_test_v1](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1.md) |
| V6 | batch 1024; scale 0.3; validation-selected epoch | 1.0 | 0.25 | 1.0 | [shared_fusion03_beta5_b1024_v1](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/archive/20260928/documents/shared_fusion03_beta5_b1024_v1.md) |

## ReactZyme test: all-positive MRR

| Split / direction | Strongest primary competitor | V1 | V2 | V3 | V4 | V5 | V6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reaction_smi / enzyme_to_reaction | 0.518000 | 0.523804 | 0.537749 | 0.537550 | 0.534366 | 0.524770 | 0.549547 |
| reaction_smi / reaction_to_enzyme | 0.337000 | 0.399995 | 0.401450 | 0.394653 | 0.400381 | 0.393927 | 0.412479 |
| enzyme_smi / enzyme_to_reaction | 0.956000 | 0.962493 | 0.963482 | 0.961178 | 0.971355 | 0.967982 | 0.978006 |
| enzyme_smi / reaction_to_enzyme | 0.592000 | 0.667677 | 0.666656 | 0.662794 | 0.666249 | 0.662023 | 0.672143 |
| time / enzyme_to_reaction | 0.690000 | 0.781538 | 0.782770 | 0.774713 | 0.780669 | 0.775257 | 0.799581 |
| time / reaction_to_enzyme | 0.372000 | 0.540884 | 0.538339 | 0.531367 | 0.538285 | 0.527831 | 0.547569 |

## EnzymeMap test: official screening library

Table 1: 261,907 candidate IDs and 1,521 eligible unique test queries. Table 2: 252,113 candidate IDs after excluding training IDs, and 1,337 eligible test queries. These query counts follow deduplication and eligibility rules; the original test split contains 4,642 associations.

| Setting / metric | Strongest primary competitor | V1 | V2 | V3 | V4 | V5 | V6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| table1 / bedroc85 | 0.486600 | 0.577817 | 0.577945 | 0.574570 | 0.572337 | 0.547147 | 0.545234 |
| table1 / bedroc20 | 0.666900 | 0.755572 | 0.757111 | 0.757460 | 0.755433 | 0.708884 | 0.717676 |
| table1 / ef0.05 | 14.910000 | 16.811991 | 16.864295 | 16.935198 | 16.859042 | 15.457319 | 15.817309 |
| table1 / ef0.1 | 8.180000 | 8.909144 | 8.953017 | 9.001075 | 8.958692 | 8.433043 | 8.685418 |
| table2 / bedroc85 | 0.451400 | 0.536819 | 0.534129 | 0.529326 | 0.528734 | 0.507029 | 0.502070 |
| table2 / bedroc20 | 0.614300 | 0.727165 | 0.727833 | 0.727885 | 0.726403 | 0.675522 | 0.685032 |
| table2 / ef0.05 | 13.570000 | 16.355816 | 16.408538 | 16.488139 | 16.418633 | 14.840457 | 15.238332 |
| table2 / ef0.1 | 7.810000 | 8.736791 | 8.779122 | 8.826313 | 8.793103 | 8.175664 | 8.469625 |

## Case 1: predeclared Reaction-Sim deployment checkpoint

Recovery counts below use the top 25. This target model was designated before these Case1 predictions; all other target-trained models and native pre-phase-2 scores are included in each detailed report.

| Version | Unique paper catalysts / 12 | Unique paper + patent catalysts / 24 | Paper entries / 15 (144-entry ranking) | Paper + patent entries / 36 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 9 | 11 | 11 | 11 | 0.622869 |
| V2 | 9 | 11 | 12 | 12 | 0.623751 |
| V3 | 9 | 11 | 12 | 12 | 0.624633 |
| V4 | 10 | 10 | 10 | 10 | 0.612875 |
| V5 | 7 | 9 | 7 | 11 | 0.598471 |
| V6 | 4 | 6 | 4 | 8 | 0.564668 |

The same frozen Reaction-Sim checkpoint at smaller screening budgets:

| Version | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 0 | 4 | 9 | 6/8 |
| V2 | 0 | 4 | 9 | 6/8 |
| V3 | 0 | 4 | 9 | 6/8 |
| V4 | 0 | 3 | 10 | 7/8 |
| V5 | 2 | 2 | 7 | 5/8 |
| V6 | 0 | 1 | 4 | 2/8 |

Primary-paper coverage is descriptive, not independent replication. V5 recovers two catalysts in its top five, but both are related constructs from paper S16. Highest paper-catalyst recovery at 25: V4 (10/12). These retrospective results do not change any frozen deployment choice.

**Longer-training transfer check.** V6 uses validation-selected checkpoints (epoch 20 for all ReactZyme splits; epoch 15 for EnzymeMap), from the same fresh trajectories as V5. Its Reaction-Sim Case1 paper recovery falls from V5’s 7/12 to 4/12 at 25; the native epoch-20 encoder recovers 6/12 before phase 2. All four paper catalysts recovered by V6 have retrieved training homologs at ≥90% identity. The stronger ReactZyme test scores therefore do not translate into stronger transfer on this literature panel.

Across the EnzymeMap-trained checkpoints, Case1 paper recovery is **6–7/12** at 25; conditional AUROC remains about **0.47**. This negative result is retained. Benchmark gains have not removed the dependence on the training dataset.

For V1–V4, four recovered paper catalysts have retrieved Reaction-Sim training homologs at ≥90% identity. Their additional paper catalyst H006 moves from native rank 33 into the top 25, but its lack of a qualifying MMseqs hit does not prove absence of a homolog. Related chemistry also occurs in training. This is retrospective recovery of literature-supported catalysts for one reaction, not new wet-lab validation or proof of broad generalization.

## Comparison boundaries

- Primary ReactZyme targets are the strongest main-model results in [TIGER Table 1](https://arxiv.org/html/2605.24489v1#S3.T1) and the applicable published comparisons. TIGER main Reaction-Sim E→R MRR is **0.518**. Its **0.543** MLP ablation and **0.632** SwissProt+DGN condition remain separate context. V6 also exceeds 0.543 on this MRR cell; the SwissProt+DGN condition remains higher. This does not compare every ablation metric.
- EnzymeMap targets combine the published [FGW-CLIP results](https://arxiv.org/pdf/2512.08508) and the locally reproduced released CLIPZyme checkpoint, taking the stronger value for each cell. FGW-CLIP has not been reproduced from a released checkpoint.
- Table-2 EF10 uses the published CLIPZyme **7.81** in this presentation; the original frozen qualification used its exact local reproduction **7.80813533427353**. All listed versions exceed both.
- Downstream training associations are matched. Frozen pretrained protein, molecular and SLEEC inputs differ from competitors, so this is not a comparison controlling every pretraining resource.
- Versions within each training family share a trajectory and seed. Repeated benchmark inspection makes the findings exploratory; margins are point-estimate improvements, not demonstrated statistical superiority.
- Pending studies are excluded until all benchmark tests, qualification and Case1 reporting are complete.

[Exact comparator registry](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/goal_primary_comparators_20260921.json) · [Training and result evidence audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_winners_evidence_audit_20260921.json) · [Full chronological findings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/findings.md).

Additional completed-family audits: [V5 training-family audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/evidence_audit.json) · [V6 training-family audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/evidence_audit.json).
