### Winning configuration: shared_fusion03_beta5_b1024_epoch10_fixed_test_v1

Recorded 2026-09-21T07:38:57.118696+00:00. This configuration exceeds all **14 primary benchmark point estimates** (six ReactZyme cells and eight EnzymeMap screening cells). The comparison is exploratory after repeated benchmark inspection; it does not establish statistical superiority or control for differing pretrained resources.

**Architecture and training.** One dual encoder retains the frozen SLEEC scorer, a global protein view and four learned residue views. The compact reaction tower uses the same model configuration across targets. Each benchmark/split has its own freshly trained F3 weights and its own training-only semantic dictionary. Frozen protein/reaction encoder and SLEEC pretraining differ from competitors; matched downstream associations do not eliminate that pretraining difference.

**Stage 1.** Anchor-balanced all-positive decoupled InfoNCE, beta 5.0, equal R→E/E→R weights, seed 42, batch 1024, fixed epoch 10 snapshot and initial fused residue scale 0.3. AdamW learning rate 0.0001 and weight decay 0.01. Snapshots come from independent fresh target fits with constant learning rate; later training cannot alter these saved weights.

**Stage 2.** A residual dual encoder fits the full target training graph for 100 positive-CE updates, temperature 0.2, identity weight 10 and learning rate 0.0001. The semantic dictionary uses training associations only. Inference uses residual cap **1.0**, semantic weight **0.25**, and a **1×** multiplier on the internal fused residue contribution. Endpoint embeddings remain independently encodable; there is no model ensemble, candidate-pool hubness correction, or RefSeq search.

**Protocol.** ReactZyme uses the official three target splits and all-positive MRR in both directions. EnzymeMap uses 34,427/7,287/4,642 original train/dev/test associations, with the released 261,907-ID screening pool; Table 2 excludes training enzyme IDs, leaving 252,113 candidates. Validation reports BEDROC85, BEDROC20, EF5 and EF10; MRR is disabled for EnzymeMap. All four targets use the same fixed epoch, with no fallback within this evaluation. This epoch-10 experiment was declared after inspecting epoch-10 validation and the separate EnzymeMap epoch-15 test. It is an exploratory follow-up, not an independent prospective confirmation. It does not alter the separate predeclared 5/10/15/20-epoch validation selector. ReactZyme reaction inputs follow its participant-set representation; EnzymeMap retains physical reactant/product sides.

| Benchmark | Setting | Metric | Method ↑ | Comparison target | Difference |
| --- | --- | --- | ---: | ---: | ---: |
| ReactZyme | reaction_smi | enzyme_to_reaction | 0.524770 | 0.518000 | +0.006770 |
| ReactZyme | reaction_smi | reaction_to_enzyme | 0.393927 | 0.337000 | +0.056927 |
| ReactZyme | enzyme_smi | enzyme_to_reaction | 0.967982 | 0.956000 | +0.011982 |
| ReactZyme | enzyme_smi | reaction_to_enzyme | 0.662023 | 0.592000 | +0.070023 |
| ReactZyme | time | enzyme_to_reaction | 0.775257 | 0.690000 | +0.085257 |
| ReactZyme | time | reaction_to_enzyme | 0.527831 | 0.372000 | +0.155831 |
| EnzymeMap | table1 | bedroc85 | 0.547147 | 0.486600 | +0.060547 |
| EnzymeMap | table1 | bedroc20 | 0.708884 | 0.666900 | +0.041984 |
| EnzymeMap | table1 | ef0.05 | 15.457319 | 14.910000 | +0.547319 |
| EnzymeMap | table1 | ef0.1 | 8.433043 | 8.180000 | +0.253043 |
| EnzymeMap | table2 | bedroc85 | 0.507029 | 0.451400 | +0.055629 |
| EnzymeMap | table2 | bedroc20 | 0.675522 | 0.614300 | +0.061222 |
| EnzymeMap | table2 | ef0.05 | 14.840457 | 13.570000 | +1.270457 |
| EnzymeMap | table2 | ef0.1 | 8.175664 | 7.810000 | +0.365664 |

For Table-2 EF10, the comparison above uses the published CLIPZyme value **7.81** ([FGW-CLIP Table 2](https://arxiv.org/html/2512.08508v1#S5.T2)); the immutable original qualification used the locally reproduced **7.80813533427353**. This configuration exceeds both.

**Separate TIGER ablation context.** The primary main-model Reaction-Sim E→R target is 0.518. This configuration's 0.524770 MRR is below the two-layer-MLP ablation’s 0.543 and below the SwissProt+DGN condition’s 0.632. These are separate published conditions; this statement concerns that MRR cell only. [TIGER Tables 1–3](https://arxiv.org/html/2605.24489v1).

**Case 1.** Choices were frozen before scoring this version. All 144 literature-panel entries are retained, representing 123 unique sequences. The table below counts unique-sequence recovery; the companion CSV also ranks every original entry. Each target-trained checkpoint is reported independently, alongside its native encoder before phase 2.

| Target model / score | Paper catalysts @25 / 12 | Papers + patents @25 / 24 | Workbook actives @25 / 81 | Conditional AUROC |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 7 | 9 | 21 | 0.598471 |
| reaction_smi/native_before_phase2 | 6 | 6 | 17 | 0.534685 |
| enzyme_smi/selected | 5 | 10 | 21 | 0.577895 |
| enzyme_smi/native_before_phase2 | 5 | 9 | 20 | 0.616990 |
| time/selected | 5 | 8 | 18 | 0.526161 |
| time/native_before_phase2 | 5 | 5 | 18 | 0.572898 |
| enzymemap/selected | 7 | 7 | 17 | 0.471193 |
| enzymemap/native_before_phase2 | 7 | 7 | 17 | 0.465902 |

**Predeclared deployment result.** Reaction-Sim paper-catalyst recovery changes from **6/12 before phase 2 to 7/12 after phase 2** at 25. The other target-trained checkpoints are reported without substituting one based on Case1 performance.

**Early literature-catalyst recovery.** All three recorded cutoffs are shown to make screening-budget tradeoffs visible. The final column counts primary papers represented by at least one recovered catalyst; it is descriptive source coverage, not a count of independent validation experiments. These results do not alter the frozen model choices.

| Target model / score | Paper catalysts @5 / 12 | @10 / 12 | @25 / 12 | Primary papers represented @25 |
| --- | ---: | ---: | ---: | ---: |
| reaction_smi/selected | 2 | 2 | 7 | 5/8 |
| reaction_smi/native_before_phase2 | 0 | 1 | 6 | 3/8 |
| enzyme_smi/selected | 1 | 3 | 5 | 4/8 |
| enzyme_smi/native_before_phase2 | 1 | 4 | 5 | 4/8 |
| time/selected | 1 | 4 | 5 | 4/8 |
| time/native_before_phase2 | 0 | 4 | 5 | 4/8 |
| enzymemap/selected | 0 | 0 | 7 | 4/8 |
| enzymemap/native_before_phase2 | 0 | 0 | 7 | 4/8 |

**Requested 144-entry panel.** The same sequence can represent several literature entries; these counts are not independent confirmations. Rankings retain all entries and use the recorded deterministic tie order.

| Target model / score | Paper entries @25 / 15 | Papers + patents @25 / 36 | Workbook active entries @25 / 102 |
| --- | ---: | ---: | ---: |
| reaction_smi/selected | 7 | 11 | 21 |
| reaction_smi/native_before_phase2 | 4 | 4 | 18 |
| enzyme_smi/selected | 8 | 14 | 24 |
| enzyme_smi/native_before_phase2 | 6 | 10 | 22 |
| time/selected | 8 | 12 | 22 |
| time/native_before_phase2 | 8 | 8 | 21 |
| enzymemap/selected | 7 | 7 | 18 |
| enzymemap/native_before_phase2 | 6 | 6 | 18 |

**Reaction-Sim training homology.** Of 7 paper catalysts recovered at 25, 3 have a retrieved training hit with at least 90% sequence identity. The audit covers supervised Reaction-Sim training only, not frozen protein/SLEEC pretraining. Both query molecules also occur together within larger training participant sets. No qualifying MMseqs hit means unknown similarity, not established absence of a homolog. The panel does not establish distant catalytic generalization.

| Paper catalyst | Selected rank | Native rank | Best retrieved training identity |
| --- | ---: | ---: | ---: |
| H012 | 3 | 22 | unknown (no qualifying hit) |
| H011 | 4 | 23 | unknown (no qualifying hit) |
| H007 | 12 | 72 | unknown (no qualifying hit) |
| H005 | 16 | 58 | unknown (no qualifying hit) |
| H004 | 17 | 8 | 96.8% |
| H003 | 22 | 14 | 97.5% |
| H001 | 24 | 13 | 99.1% |
| H002 | 28 | 24 | 98.1% |
| H006 | 53 | 90 | unknown (no qualifying hit) |
| H008 | 57 | 38 | 50.2% |
| H009 | 60 | 32 | 50.0% |
| H013 | 68 | 57 | 44.4% |

Case 1 is one previously examined reaction, with dependent constructs and heterogeneous literature assays. The 42 non-detect-only sequences are conditional assay observations, not universally inactive enzymes. These results are retrospective evidence, not new wet-lab validation or proof of broad catalytic generalization. No model is selected by Case1 results.

**Frozen artifacts.**

- reaction_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/reaction_smi/phase2_followup/epoch10/training/step0100.pt).
- enzyme_smi: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzyme_smi/phase2_followup/epoch10/training/step0100.pt).
- time: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/time/phase2_followup/epoch10/training/step0100.pt).
- enzymemap: F3 [screen-epoch=09.ckpt](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/checkpoints/screen_selection/screen-epoch=09.ckpt); [training config](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/configs/train.yaml); [phase-2 head](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_v1/enzymemap/phase2_followup/epoch10/training/step0100.pt).

[Exact protocol](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/protocol.json) · [Benchmark source records](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/qualification.json) · [Case1 metrics](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/summary.json) · [All 144 rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/all_144_entry_rankings.csv) · [123 unique-sequence rankings](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/case1/unique_sequence_rankings.csv).

[Architecture, training-edge and artifact audit](/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/generalization_20260919_2251/cross_paper_retraining/shared_fusion03_beta5_b1024_epoch10_fixed_test_v1/evidence_audit.json).
