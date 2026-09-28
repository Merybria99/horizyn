# Benchmark results at the four-hour checkpoint

Updated 2026-09-21T05:07:28.083901+00:00. Branch: `research/f3-circev2-generalization-20260919`.

The three-seed EnzymeMap mean exceeds released CLIPZyme and published FGW-CLIP on all four metrics in both official screening settings. The new shared architecture has not yet improved all six ReactZyme test cells. The full joint goal remains unachieved.

## EnzymeMap: three fresh models, fixed recipe

SLEEC + global protein view + four learned residue views; compact reaction tower; anchor-balanced all-positive decoupled contrastive loss with inverse temperature 5. All trainable F3 weights start from scratch on the official target training associations. Fixed 18 epochs, batch 2,048, then 100 phase-2 positive-CE updates, temperature 0.2, identity weight 10, residual cap 1 and semantic weight 0.25.

| Setting / model | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |
| --- | ---: | ---: | ---: | ---: |
| Full released pool / Three-seed mean | 0.521656 | 0.700731 | 15.651865 | 8.516224 |
| Full released pool / Released CLIPZyme | 0.446942 | 0.629817 | 14.085125 | 8.060006 |
| Full released pool / Published FGW-CLIP | 0.486600 | 0.666900 | 14.910000 | 8.180000 |
| Training IDs excluded / Three-seed mean | 0.477018 | 0.666664 | 15.023372 | 8.297240 |
| Training IDs excluded / Released CLIPZyme | 0.391328 | 0.588619 | 13.396710 | 7.808135 |
| Training IDs excluded / Published FGW-CLIP | 0.451400 | 0.614300 | 13.570000 | 7.610000 |

Seeds 17 and 42 individually beat both comparators in all eight cells. Seed 73 does not. These are independent model results, not ensemble predictions. The [complete seed table and variability](../../../runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta5_fresh_replication_v1/comparison.md) reports every seed and sample standard deviation. Three seeds and repeated benchmark inspection do not establish statistical superiority. FGW-CLIP values are published-only; its model was not reproduced.

The [matched beta-10 controls](../../../runs/generalization_20260919_2251/cross_paper_retraining/sleec_beta10_matched18_replication_v1/comparison.md) use the same seeds and 18-epoch budget. All three training/evaluation pairs are now complete (follow-up at 2026-09-21T05:08:58.058340+00:00). Beta 5 improves 22 of 24 paired test cells and every three-seed mean metric; seed 73’s two EF10 values decline. These correlated measurements are not independent statistical samples.

## ReactZyme: actual new test results

The fixed EnzymeMap phase-2 recipe is transferred without a validation gate or parent fallback, so unsuccessful variants are tested and retained in the table. Every split uses its own official training associations. ReactZyme uses batch 512 and fixed epoch-10/20 snapshots.

| Split / variant | R→E all-positive MRR | E→R all-positive MRR |
| --- | ---: | ---: |
| reaction_smi / retained phase 2 | 0.415174 | 0.523981 |
| reaction_smi / beta 10, epoch 10 | 0.402786 | 0.470564 |
| reaction_smi / beta 10, epoch 20 | 0.405844 | 0.484152 |
| reaction_smi / beta 5, epoch 10 | 0.389372 | 0.515102 |
| reaction_smi / beta 5, epoch 20 | pending | pending |
| enzyme_smi / retained phase 2 | 0.697468 | 0.973676 |
| enzyme_smi / beta 10, epoch 10 | 0.680091 | 0.971884 |
| enzyme_smi / beta 10, epoch 20 | 0.688714 | 0.976363 |
| enzyme_smi / beta 5, epoch 10 | pending | pending |
| enzyme_smi / beta 5, epoch 20 | pending | pending |
| time / retained phase 2 | 0.599849 | 0.835167 |
| time / beta 10, epoch 10 | 0.556108 | 0.803804 |
| time / beta 10, epoch 20 | 0.574441 | 0.832144 |
| time / beta 5, epoch 10 | pending | pending |
| time / beta 5, epoch 20 | pending | pending |

At epoch 10, beta 5 raises Reaction-Sim E→R from 0.470564 to 0.515102 but lowers R→E from 0.402786 to 0.389372. This is a trade-off, not a bidirectional improvement. The beta-10 Enzyme-Sim epoch-20 model improves E→R over the retained reference, while its R→E is lower.

The retained reference exceeds TIGER’s main comparison table, but its Reaction-Sim E→R remains below the reported 0.543 two-layer-MLP ablation. No claim to beat every reported competitor is made. [Live complete transfer table](../../../runs/generalization_20260919_2251/cross_paper_retraining/reactzyme_fixed_enzymemap_recipe_transfer_v1/comparison.md).

## Generalization evidence and limitations

- A stricter secondary screen removes exact training-sequence aliases from every method, retaining 249,828 candidates and 1,333 queries. The fresh-seed mean remains above released CLIPZyme on all four metrics. This is not a remote-homology or measured-activity test. [Audit](../../../runs/generalization_20260919_2251/cross_paper_retraining/sleec_sequence_disjoint_screening_v1/comparison.md).
- The released EnzymeMap validation set contains only five reaction rules; one supplies 98.53% of evaluated validation queries. This creates a model-selection risk. The official split is preserved. [Verified release diagnostic](../../../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_validation_rule_coverage.md).
- Downstream training associations match the target benchmarks, but frozen backbone and SLEEC pretraining differ from competitors. These experiments do not isolate architecture from every pretraining difference.
- Case 1 remains retrospective literature evidence; no new wet-lab activity measurements or RefSeq experiment were run in this follow-up.

## Still running

Fresh beta-5 ReactZyme trainings continue toward epoch 20, with automatic fixed-recipe and validation-selected phase-2 evaluations. The 30-epoch Reaction-Sim base continuations completed; their phase-2 evaluations are in progress. The large-batch Reaction-Sim control also continues. These runs remain active because the full objective is not yet achieved.

CLIPZyme validation uses BEDROC85, BEDROC20, EF5 and EF10; primary checkpoint selection uses BEDROC85, and fixed-budget replications do not select seeds. MRR validation is disabled for EnzymeMap. ReactZyme keeps all-positive MRR. No ensembles or hubness corrections are used.

[Detailed findings](../../../findings.md) · [Live run status](../../../runs/generalization_20260919_2251/benchmark_status.md).

## Matched temperature control: final means

| Setting / recipe | BEDROC85 ↑ | BEDROC20 ↑ | EF5 ↑ | EF10 ↑ |
| --- | ---: | ---: | ---: | ---: |
| table1 / beta 5 | 0.521656 | 0.700731 | 15.651865 | 8.516224 |
| table1 / beta 10 | 0.494040 | 0.664343 | 14.685863 | 8.168627 |
| table2 / beta 5 | 0.477018 | 0.666664 | 15.023372 | 8.297240 |
| table2 / beta 10 | 0.439892 | 0.623451 | 13.953732 | 7.907565 |
