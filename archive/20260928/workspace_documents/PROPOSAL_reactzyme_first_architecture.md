# Proposal: B1 and B2 on reaction_smi against existing CIRCE-V2

Updated: 2026-09-09 UTC. The scratch B1/B2 runs completed 30 epochs and final test evaluation. At the user's request, the default chemical encoder is now pretrained PCQM4Mv2 Graphormer. See [the replacement record](../horizyn/README_B12_PRETRAINED.md) for details. The original campaign is preserved at `horizyn/runs/b12_reaction_smi_train_20260908_230718/`.

The first milestone is two new training runs: B1 and B2 on ReactZyme reaction_smi, both with seed 42, compared with the existing validation-selected CIRCE-V2 checkpoint. There is no B0 training run. This is an economical architecture screen; enzyme_smi, time, and additional seeds are deferred until the result justifies them. Broader-data training remains the eventual goal. A reaction_smi result alone cannot establish transfer across other distributions or experimental catalytic specificity.

The proposed candidate uses ESM-C 600M residue representations, a pretrained molecular Graphormer, and a score that retains interactions between multiple protein and chemical tokens. It requires sequence and the released chemical input. Direction, structures, mechanisms, and assays are later extensions supported by the input schema.

## 1. First architecture

| Component | Initial choice |
|---|---|
| Protein initializer | Public ESM-C 600M foundation checkpoint, frozen for the initial campaign |
| Protein representation | Project contextual residue features to width 256; 32 learned queries attend over valid residues; two small latent self-attention blocks; project to 128-dimensional local tokens |
| Chemical encoder | PCQM4Mv2-pretrained Graphormer, 12 layers, width 768, 32 heads, feed-forward width 768; identical pinned initialization for both runs |
| Chemical input | Encode each molecular component with atom/bond features, then collect its atom states into an unordered participant set |
| Chemical representation | 16 learned queries attend over the atom states; two small latent self-attention blocks; project to 128-dimensional local tokens |
| Global representation | Pool each token bank and project to a normalized 256-dimensional vector |
| Local score | Reaction-token coverage by protein tokens using cosine similarity and normalized log-sum-exp |
| Final score | Fixed equal mixture of global cosine and local score for the first interaction model |

Token counts and widths are initial engineering settings, not claims about the number of active sites or catalytic functions. Learned tokens come from residue/atom states before global pooling. They are not copies or perturbations of an already pooled protein vector, and they are not automatically interpretable sites.

Keep CIRCE-V2's protein length handling and eligibility: at most 1,022 protein tokens with ends-center truncation, recording residue coverage under the new tokenizer. Do not remove examples to make the new model easier to evaluate. Chemical components should be processed separately to avoid quadratic attention over a huge disconnected molecular union; the small set-pooling stage sees all component atom states. No silent truncation of molecular participants.

All ReactZyme participants have role `unknown`. Preserve the released chemical information, including available stereochemistry. Do not invent substrate/product assignments or create a mapped self-reaction. Molecular enumeration must not influence predictions.

For chemical tokens R_i and enzyme tokens E_j, both normalized, define A_ij = cosine(R_i, E_j). The initial local score is:

    s_local = mean_i [T * (logsumexp_j(A_ij / T) - log(K_e))]
    T = 0.1
    s_final = 0.5 * s_global + 0.5 * s_local

Use valid-token masks and the actual valid count if the banks are not full. This asks whether each chemical token has compatible protein evidence; it does not require every protein token to describe the same reaction. The same scalar pair score is used in both retrieval directions. Fixed normalized score components avoid an unconstrained gate hiding the interaction branch during the first comparison, but do not guarantee that the learned information is free of shortcuts.

The initial screen used a small scratch Graphormer, which confounded architecture quality with limited chemistry pretraining. The replacement loads the full PCQM4Mv2 molecular encoder and its original OGB preprocessing, while keeping the token pools and scoring rules. Pooling warms up for two epochs; then Graphormer trains at 1e-5 and pooling at 1e-4. The larger encoder and changed schedule mean improvement would not isolate pretraining alone. Molecular corpus overlap has not been measured; no broader-data enzyme-reaction checkpoint is used to initialize these runs.

## 2. Reuse code without inheriting an incompatible protocol

Useful existing components:

- `horizyn/horizyn/pretrained_graphormer.py`: pinned chemistry weights and original OGB graph preprocessing.
- `horizyn/horizyn/token_retrieval_data.py`: frozen native ESM-C residue features and molecular-set assembly.
- The VenusRXN encoder path remains available only to reproduce preserved scratch checkpoints.
- `horizyn/horizyn/biological_residual.py`: bounded-memory interaction scoring patterns and masks.
- `horizyn/horizyn/losses.py`: CIRCE-V2's `SampledMultiPositiveInfoNCELoss`, with its existing positive and typed-negative masks.
- Existing Horizyn ReactZyme manifests and metric implementations.

Implement a dedicated molecule-set input adapter. The VenusRXN reaction/CGR preprocessing cannot be applied unchanged to ReactZyme's non-directional molecular sets. Its full multimodal protein cross-attention path is also not needed for the initial cached-encoder campaign.

Keep a single explicit `score_pairs` interface for training, both evaluation directions, and subsequent screening. Current cosine-only evaluation or wet-lab entry points must not silently bypass the local score.

## 3. Match the existing CIRCE-V2 training recipe

Since CIRCE-V2 will be reused, adapt B1 and B2 to its established recipe. Changing the loss, negative construction, batch size, and checkpoint selector at the same time would weaken the comparison and create a reason to retrain the reference. Keep those choices fixed for this screen.

Use `horizyn/configs/reactzyme_reaction_smi_prott5_annotation_negatives.yaml` and the canonical run's saved configuration at `horizyn/runs/circe_v2_prott5_annotation_negatives/logs/train/protein_pooling_training/version_2/hparams.yaml`. The saved configuration associated with the epoch-23 checkpoint takes precedence over later config edits or earlier run versions.

- Use the existing reaction_smi train/validation pair and reaction manifests under `horizyn/data/revised_protocols/reactzyme_paper/reaction_smi/`, including its validation candidate IDs.
- Use seed 42, up to 30 epochs, AdamW with weight decay 0.01. Pooling uses 1e-4; the pretrained chemical encoder warms up frozen for two epochs and then uses 1e-5. Validate every epoch; early stopping uses patience 5 and min_delta 0.0001.
- Preserve 2,000 global sampled pair rows before entity deduplication. The authorized launch uses 1,000 rows per GPU on two GPUs for each candidate; CIRCE-V2 used 500 rows per GPU on four GPUs. Match distributed candidate gathering and loss reduction, not just the nominal batch size. Different world sizes need not reproduce the historical run's exact random draws.
- Preserve the sampler and pool at `horizyn/runs/circe_v2_annotation_negatives/data/train_negative_pools.json`: target 50% positive rows, with the negative half balanced between biological and random negatives where pools permit. Keep the existing fallback behavior.
- Use `SampledMultiPositiveInfoNCELoss`, fixed beta 10, and `positive_pair_source: all_known_in_batch`. This is a reaction-anchored objective, although retrieval is evaluated in both directions. Positive masks use training associations. Match the current CIRCE implementation by including all typed-pool negatives whose proteins occur in the distributed batch, including entries beyond the explicitly selected negative rows. Other unlabeled matrix cells are ignored. Annotation-derived negatives remain training assumptions, not measured inactivity.
- Pass `distance = 1 - score` to the loss for both B1 and B2, preserving CIRCE-V2's cosine-distance convention and logit scale. Do not introduce a new loss or trainable temperature in this screen.
- Use the reference's `32-true` precision and float32 matmul setting `high`. The pretrained chemical encoder retains its published dropout configuration and uses the explicit warmup/lower learning rate above. New pooling modules use dropout 0.1 in both B1 and B2; this is part of the replacement architecture, not a claim that every CIRCE-V2 module uses that dropout.
- Select checkpoints with `val/mean_bidirectional_mrr`, the arithmetic mean of the two directional **first-positive MRR** values. This matches CIRCE-V2's selector. Report ReactZyme all-positive MRR separately; it is not the same metric.

Profile memory before launching. Use blocked interaction scoring or recomputation to preserve the logical candidate pool. Ordinary gradient accumulation does not enlarge that pool. If the reference batch or precision is infeasible, record a common B1/B2 deviation and describe the CIRCE-V2 comparison as a comparison of complete methods with that limitation; do not quietly turn it into a different training recipe or add a B0 run.

Both candidates start from the same frozen ESM-C checkpoint, pinned pretrained chemical encoder and identically initialized token pools. Train them independently with the same sampler seed; do not warm-start B2 from B1. Their intended difference is the scoring rule and the gradients it induces. Both use the same chemical-encoder unfreezing schedule; B2 adds no learned gate or auxiliary objective.

The frozen protein features can be extracted once and shared between B1 and B2 without using pair labels. All learned pooling, graph encoders, fitted preprocessing, and train-derived annotations use reaction_smi training data only. Optional protein adapters are deferred.

## 4. Fixed CIRCE-V2 reference and evaluation protocol

Reuse the existing validation-selected epoch-23 checkpoint:

    horizyn/runs/circe_v2_prott5_annotation_negatives/checkpoints/protein-pooling-epoch=23.ckpt

Its saved bidirectional test result is:

    horizyn/runs/circe_v2_prott5_annotation_negatives/evaluation/best_epoch23/test_both.json

| Existing CIRCE-V2 metric | Reaction to enzyme | Enzyme to reaction | Arithmetic mean of directions |
|---|---:|---:|---:|
| ReactZyme all-positive MRR | 0.263017 | 0.349279 | 0.306148 |
| First-positive MRR | 0.387878 | 0.349296 | 0.368587 |

Match the saved `paper_test_candidates` protocol exactly: 386 reaction candidates, 14,688 enzyme candidates, `test_pairs_only` ground truth, and `canonical_forward_only` reaction query expansion. Matching candidate counts is insufficient: verify candidate IDs, pair manifests, eligibility, and metric definitions. Reuse the existing result if these agree. Re-evaluating the fixed checkpoint is only needed if evaluation compatibility cannot be established from the artifacts; it is not a training run.

Use CIRCE-V2's production metric functions directly, adapting only model inputs and pair scores. Validation calls `ProteinPooledLitModule._batched_retrieval_metric_values`; final testing calls `scripts/evaluate_protein_pooling.py:append_retrieval_metrics`. Preserve their existing, distinct tie policies. Paper-test proteins follow first occurrence in the test pair file; validation proteins follow the configured candidate file. The initial B1/B2 custom stable-sort evaluator was retired after six epochs; its results must not be used as the CIRCE comparison.

The main result table will contain CIRCE-V2, B1, and B2 on this one split in both directions. Include ReactZyme all-positive MRR, first-positive MRR, the existing top-K hit rates and appropriate recall, candidate coverage, and training/evaluation cost. Retain per-query ranks for paired diagnostics. Do not compare an all-positive score to the reference's first-positive score.

## 5. Two-run decision sequence

| Model | Training in this stage | Definition | Question |
|---|---|---|---|
| CIRCE-V2 | None; reuse epoch 23 | Existing ProtT5 and multimodal chemistry model | What does the current reference achieve? |
| B1 | reaction_smi, seed 42 | ESM-C plus molecular Graphormer and the proposed token banks, scored only through their pooled global vectors | Does the replacement representation work? |
| B2 | reaction_smi, seed 42 | Same encoders and token banks as B1, with the final global/local score above | Do token interactions improve it? |

1. Verify reference artifacts and candidate identities, then profile feature extraction, a training step, and complete validation scoring.
2. Train B1 and B2 independently. Choose each checkpoint using the fixed CIRCE-V2 validation selector. Freeze architecture settings and nominate a candidate using validation before inspecting either candidate's test results.
3. Evaluate both selected checkpoints once on the same reaction_smi test protocol and compare with the saved CIRCE-V2 reference. Report all three rows, including failures.
4. Decide whether the nominated candidate merits a later confirmation stage. No enzyme_smi, time, extra-seed, or broader-data training is included in this initial two-run budget.

As an initial practical screening criterion, require a candidate to improve the arithmetic mean of the two test all-positive MRR values by at least 0.01 over CIRCE-V2, with neither direction losing more than 0.01, complete candidate coverage, and affordable full-pool evaluation. Fix this criterion before new runs. It is an engineering threshold, not a statistical significance claim. The existing reference mean is 0.306148, so the corresponding mean target is at least 0.316148. Test results assess the frozen candidate; they do not trigger retuning on the same test set.

Use B2 versus B1 to assess the contribution of interactions under the new representation, and each versus CIRCE-V2 to assess the full replacement. When nominating on validation, prefer B1's lower scoring cost if B2 is effectively tied; nominate B1 if it improves while B2 does not. If both regress, their relative results help decide whether to investigate chemical representation quality or interaction scoring; they cannot uniquely diagnose the cause. Do not launch broader training solely because the architecture is new.

Report this as a single-seed screen. Paired query-level resampling can describe evaluation uncertainty, with reaction-level grouping where queries share reaction labels; it does not estimate training-seed variance. Inspect validation performance by training-set homology, chemistry similarity, protein length, and molecular-set size. Check molecule-order invariance and, within B2, whether disabling the local score changes rankings and quality. These require no extra training runs and help expose failures hidden by an aggregate, but do not establish that shortcuts are absent.

If the screen is promising, the next proposal should confirm the selected architecture with additional seeds and independent enzyme_smi/time training before treating it as a general replacement. Start those runs from the allowed foundation initialization, not from reaction_smi-trained weights, and use one declared architecture recipe. Do not combine the three training sets before their official tests: their partitions overlap. The reaction_smi decision is a feasibility gate for that later work, not evidence that all sources follow its distribution.

## 6. Compute and evaluation

Profile a representative training step and a representative full validation evaluation before launching the campaign. Measure peak memory, examples per second, cache size, and the cost of scoring the complete candidate pool. Estimate campaign GPU-hours from those measurements rather than assigning an unsupported calendar duration.

The frozen ESM-C feature extraction is shared. Learned compact token caches are rebuilt for each trained checkpoint. For later compressed storage, a 32 x 128 FP16 protein bank occupies 8,192 bytes per protein, about 1.36 GiB for 178,327 proteins or 30.5 GiB for four million proteins, excluding global vectors, IDs, indexing, and extraction caches. FP32 banks take twice that space; preserve reference evaluation precision in the initial comparison. The full frozen residue-feature cache is much larger and must be budgeted separately.

For the benchmark, compute exact scores over the configured candidate pool in blocks, with score computation/recomputation designed to avoid materializing a full query x candidate x chemical-token x protein-token tensor. Streaming memory use does not remove the extra arithmetic; the profiling step is mandatory.

For the later million-protein deployment, use the global vectors for candidate generation and local tokens for refinement, or evaluate a multi-vector index. Select shortlist size from measured positive recall. Report any approximation separately from exact benchmark results. The global component's recall must be measured; its presence in the final score does not guarantee it is a sufficient candidate generator.

## 7. Broader-data stage

After a promising screen and the later generalization checks, fix the selected architecture and train one general model on the expanded corpus. Start from the same public foundation initialization and a fresh task model; do not average independently trained benchmark models.

Before assembling the corpus, define the generalization holdouts and deduplicate across sources with provenance. Original ReactZyme tests remain held out only if broader training respects their exclusion policy; additional source data can otherwise contain the same observations. Report expanded-data results under their own training-exposure description.

Preserve exact sequence-to-reaction observations. Use homology clusters for sampling, weighting, and splitting. The current NR90/NR50 builder transfers member associations to a representative sequence; do not treat that transfer as experimentally established specificity for the representative.

The input record should already allow:

    sequence
    molecules
    participant_roles: unknown | substrate | product | cofactor
    atom_mapping: optional
    structure: optional
    conditions: optional
    evidence_type
    source_provenance

Only the sequence/molecule fields and unknown roles are required for the first ReactZyme campaign. When broader directional data arrives, add role embeddings and a transformation objective using valid labels. Add activity and within-assay comparison losses when measurements exist, with units and conditions preserved. Structural inputs can be introduced through an optional geometry module. Mask missing objectives and balance source/assay contributions; each additional supervision type should have an incremental ablation.

The larger-data stage is the appropriate point to test ESM-C adapters or selective unfreezing. Refresh feature caches after any encoder update. The objective is to improve the shared model's transfer while preserving sequence-and-chemistry inference for sparsely annotated candidates.

## References and local evidence

- [ESM implementation and checkpoints](https://github.com/Biohub/esm).
- [ColBERT](https://arxiv.org/abs/2004.12832): precedent for independently cached representations with token-level interaction in text retrieval; not evidence of enzyme-specific improvement.
- `horizyn/README_CIRCE_V2.md`, its canonical `version_2/hparams.yaml`, and the epoch-23 evaluation JSON above: existing reference, training recipe, and metric definitions.
- `documents/README_data.md` and `horizyn/data/standardized/retrieval_training_source_collapse/manifest.json`: broader corpus inventory.
- `documents/README_post_ablation_improvement_options.md`: earlier fusion, loss, and validation-selector experiments.
- `horizyn/scripts/build_retrieval_training_source_collapse.py`: cluster representative label construction.
