### Winning configuration: shared_recipe_beta5_b512_e10_fusion3_v2

Recorded 2026-09-21T07:38:56.694021+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5, equal R→E/E→R weights, seed 42, batch 512, 10-epoch checkpoint, AdamW learning rate 0.0001 and weight decay 0.01. Reaction-Sim recovered its optimizer state after a memory collision; it is not described as uninterrupted training.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.25**, and a **3×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. This fixed candidate has no EnzymeMap validation fallback or seed selection. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.523804 | 0.518000 | +0.005804 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.399995 | 0.337000 | +0.062995 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.962493 | 0.956000 | +0.006493 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.667677 | 0.592000 | +0.075677 |
| ReactZyme | time | enzyme_to_reaction | 0.781538 | 0.690000 | +0.091538 |
| ReactZyme | time | reaction_to_enzyme | 0.540884 | 0.372000 | +0.168884 |
| EnzymeMap | table1 | bedroc85 | 0.577817 | 0.486600 | +0.091217 |
| EnzymeMap | table1 | bedroc20 | 0.755572 | 0.666900 | +0.088672 |
| EnzymeMap | table1 | ef0.05 | 16.811991 | 14.910000 | +1.901991 |
| EnzymeMap | table1 | ef0.1 | 8.909144 | 8.180000 | +0.729144 |
| EnzymeMap | table2 | bedroc85 | 0.536819 | 0.451400 | +0.085419 |
| EnzymeMap | table2 | bedroc20 | 0.727165 | 0.614300 | +0.112865 |
| EnzymeMap | table2 | ef0.05 | 16.355816 | 13.570000 | +2.785816 |
| EnzymeMap | table2 | ef0.1 | 8.736791 | 7.810000 | +0.926791 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.523804 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 9 | 11 | 23 | 0.622869 |
| reaction_smi/native_before_phase2 | 9 | 9 | 21 | 0.600823 |
| enzyme_smi/selected | 5 | 8 | 20 | 0.576426 |
| enzyme_smi/native_before_phase2 | 5 | 8 | 20 | 0.594944 |
| time/selected | 5 | 10 | 20 | 0.606408 |
| time/native_before_phase2 | 5 | 8 | 19 | 0.635215 |
| enzymemap/selected | 6 | 6 | 19 | 0.469724 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 | 0.471193 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **9/12 before phase 2 to 9/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 0 | 4 | 9 | 6/8 |
| reaction_smi/native_before_phase2 | 1 | 3 | 9 | 6/8 |
| enzyme_smi/selected | 3 | 4 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 3 | 4 | 5 | 4/8 |
| time/selected | 1 | 3 | 5 | 4/8 |
| time/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| enzymemap/selected | 1 | 5 | 6 | 3/8 |
| enzymemap/native_before_phase2 | 1 | 4 | 6 | 3/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 11 | 11 | 23 |
| reaction_smi/native_before_phase2 | 10 | 10 | 23 |
| enzyme_smi/selected | 8 | 11 | 21 |
| enzyme_smi/native_before_phase2 | 8 | 11 | 21 |
| time/selected | 7 | 12 | 21 |
| time/native_before_phase2 | 8 | 11 | 21 |
| enzymemap/selected | 6 | 6 | 19 |
| enzymemap/native_before_phase2 | 6 | 6 | 19 |

**Reaction-Sim training homology.** Of 9 paper catalysts recovered at 25, 4 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel therefore cannot establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H007 | 6 | 10 | unknown (no qualifying hit) |
| H012 | 7 | 5 | unknown (no qualifying hit) |
| H011 | 8 | 6 | unknown (no qualifying hit) |
| H005 | 10 | 16 | unknown (no qualifying hit) |
| H003 | 13 | 13 | 97.5% |
| H004 | 14 | 14 | 96.8% |
| H001 | 15 | 15 | 99.1% |
| H006 | 16 | 33 | unknown (no qualifying hit) |
| H002 | 20 | 18 | 98.1% |
| H008 | 26 | 28 | 50.2% |
| H009 | 28 | 24 | 50.0% |
| H013 | 52 | 66 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_multiview_beta5_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_enzyme_smi_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_beta5_all_splits_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_time_beta5_phase2_epoch10_soft_v1/residue_views/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/calibrated/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/enzymemap_seed42/phase2/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_recipe_beta5_b512_e10_fusion3_v2/case1/unique_sequence_rankings.csv).
