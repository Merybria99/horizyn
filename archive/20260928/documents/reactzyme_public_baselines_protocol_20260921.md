# ReactZyme public-competitor retraining: protocol and progress

Started 21 September 2026 on branch `research/reactzyme-public-baselines-20260921`.

**The campaign includes all four released ReactZyme families and the additional public competitors. It is still running. Full structural comparisons are not yet ready.** Test results, active processes, dependencies, and failures are maintained in [the live comparison](../../../runs/reactzyme_public_baselines_20260921/comparison.md). The earlier [release audit](reactzyme_released_competitors_20260921.md) distinguishes public implementations from paper-only references.

## Scope

| Comparator | Planned configurations | Implementation / current limitation |
|---|---:|---|
| Released ReactZyme MLP, contrastive MLP, Transformer, Bi-RNN | 96 | Four families × ESM2/SaProt × MAT-2D/MAT-3D/UniMol-2D/UniMol-3D × three splits. Native model classes loaded directly from released source. |
| Corrected contrastive-label sensitivity | 24 | Additional runs; not relabeled as the unmodified published method. |
| Original Horizyn architecture and MLNCE | 3 | Complete. Participant-set reaction adapter; original public towers, loss, and fingerprint generators. |
| CREEP, released with CARE | 3 | Supported two-modality objective, full ProtT5/rxnfp fine-tuning. Exact training-edge sampler and participant-set reaction adapter. |
| EnzGFM-650M backbone control | 12 | Frozen public EnzGFM features + released ReactZyme Transformer, four molecular feature choices × three splits. Feature preparation precedes training. |
| CLIPZyme full structural model | 3 target splits | Pending full coordinate coverage and reaction-input adapter audit; no full-model training launched. |
| EnzymeCAGE full structural model | 3 target splits | Pending missing pockets, protein-node features, and reaction-input adapter audit; no full-model training launched. |
| VenusRXN | 3 target splits | Public source competitor tracked; mapped directed reactions and release fidelity remain to be established. Local source directory lacks Git provenance. No training launched. |
| TIGER / FGW-CLIP | Reported reference rows | No verified official reproducible implementation/checkpoint in the release audit; no fabricated local reproductions. |

The first queue contains **120 runs**: 96 original-family configurations and 24 corrected-loss sensitivities. The extended executable queue contains **18 runs**: three Horizyn, three CREEP, and twelve EnzGFM controls. Structural/directional methods are explicitly pending, not counted as executable or completed runs.

## Data and evaluation contract

Every trainable retrieval model starts from fresh downstream weights for each split. Frozen pretrained molecular/protein representations are disclosed below. No pretrained enzyme–reaction retrieval checkpoint is used as a substitute for target-data training.

The source is `data/revised_protocols/reactzyme_paper`, which is the materialized split used by the existing V4 comparison. This is **not** the later `v2` or `v2_all` materialization. Exact input-file hashes and counts are in [data_audit.json](../../../runs/reactzyme_public_baselines_20260921/features/data_audit.json).

| Split | Training pairs | Validation pairs | Test pairs | Test reaction queries | Test protein candidates |
|---|---:|---:|---:|---:|---:|
| Reaction similarity | 147,393 | 16,377 | 14,689 | 386 | 14,688 |
| Enzyme similarity | 152,748 | 16,972 | 8,739 | 1,573 | 8,734 |
| Time | 149,554 | 16,618 | 12,287 | 2,634 | 12,277 |

The union contains **178,327 proteins and 7,726 reaction identifiers**. Distinct released reaction identifiers remain distinct, including cases with chemically equivalent representations. No proteins or reactions are removed to accommodate a competitor.

[candidate_parity_audit.json](../../../runs/reactzyme_public_baselines_20260921/features/candidate_parity_audit.json) verifies exact test-query and candidate-list equality, including order, against the three V4 test catalogs. Ranking uses the same `generalization_metrics.evaluate_scores` implementation and deterministic candidate-index tie handling. Both R→E and E→R are evaluated.

The main MRR is the released benchmark's **all-positive query average**: average reciprocal ranks over the positive candidates of each query, then average queries. First-positive MRR and Top-1/5/10 are also saved. They are not interchangeable metrics. This is ReactZyme retrieval, not the separate EnzymeMap screening experiment using BEDROC/EF.

Test edges are used only in the final scorer after checkpoint selection. Unlabelled test sequences/molecules may be embedded with frozen feature extractors. There is no test-based seed selection or model selection. This initial campaign uses seed 42; multi-seed uncertainty and equal-budget tuning remain necessary before a strong architecture-superiority claim.

## Released ReactZyme models

The original class definitions are imported by parsing the pinned training files, without executing their command-line parsing or training side effects:

| Family | Source | Loss |
|---|---|---|
| MLP | `train.py` | Weighted binary cross-entropy |
| Contrastive MLP | `train_contra.py` | Weighted BCE + unmodified released contrastive loss |
| Transformer | `train_tfmr.py` | Weighted BCE |
| Bi-RNN | `train_rnn.py` | Weighted BCE |

The contrastive source associates label 1 with margin repulsion and label 0 with distance attraction. Because its BCE branch uses label 1 for positive associations, the campaign retains the original convention in the main native-family row and runs a separately named `contrastive_corrected` sensitivity with inverted contrastive labels. No favorable version is silently substituted after seeing test results.

The supplied networks are pair classifiers. Their single-token cross-attention reduces algebraically to the value projection. Caching the two independent input projections therefore avoids recomputing identical features for every query–candidate pair. The cached scorer is checked against the untouched original forward. A trained Bi-RNN exceeded the tight numerical tolerance because of floating-point regrouping; it was rescored through the original pair forward from the already selected checkpoint. Future runs automatically use that fallback when needed.

Training settings: hidden dimension 128, output dimension 64, dropout 0, batch size 1,000, AdamW learning rate `1e-4`, weight decay `5e-10`, maximum 200 epochs, and early stopping after 30 epochs without validation improvement. Downstream fitting and scoring use FP32.

Two unavoidable protocol adaptations are explicit:

1. The inherited V4 train/validation split is fixed for every competitor. Upstream's validation shuffle is not seeded, and its nominal validation-best checkpoint actually follows training loss. This campaign selects minimum **validation BCE**.
2. The original negative pair files are not available in the released split artifacts. The campaign saves four deterministic corruptions per positive, alternating reaction and protein corruption within each partition. Known training positives are excluded, and validation negatives also exclude validation positives. Positive weight is 1; negative weight is 0.25. This is a controlled retraining protocol, **not an exact reproduction of the published classifier scores**. Upstream also contains a chemical-string edit-distance mining script; its transductive train-plus-test mining is not silently reproduced here.

`negative_samples.npz`, `protocol.json`, epoch metrics, `best.pt`, resumable `last.pt`, selection receipt, full test-score matrix, per-query metrics, and completion receipt are saved for every classifier configuration.

## Frozen features

| Representation | Implementation and disclosure |
|---|---|
| ESM2 | Public `esm2_t33_650M_UR50D`, 1,280 dimensions. ReactZyme extraction convention: no BOS/EOS context, mean non-padding tokens, first 5,000 residues maximum. BF16 backbone inference, FP32 mean. |
| SaProt | Public `SaProt_650M_PDB` and released ReactZyme structural strings. Exact amino-acid sequence matching before reuse; special tokens excluded from mean. Structural strings cover 148,841 proteins; the other 29,486 use ESM2, following the released fallback policy. |
| MAT-2D / MAT-3D | Released MAT network and pretrained weights, 1,024-dimensional atom mean including the dummy node, then mean over molecular occurrences. RDKit 2D coordinates or deterministic ETKDG/UFF 3D coordinates. Failed 3D conformers fall back to 2D and are recorded. |
| UniMol-2D / UniMol-3D | Public **UniMol v1 all-H**, not UniMol2. Mean atomic representation per molecule, then sum across components, following the released notebook/dataset behavior. Maximum 256 atoms and deterministic cropping; cropping and conformer failures recorded. |
| ProtT5 | Mean of raw frozen public ProtT5 residue vectors, 1,024 dimensions. Reused from complete existing feature exports; audited export code takes the residue mean before any fitted F3 transform. These are not F3-trained embeddings. |
| EnzGFM-650M | Mean of cached raw frozen official EnzGFM residue vectors, 2,048 dimensions. Existing cache excludes special tokens and uses a 1,022-residue ends/center policy. Published extraction includes special tokens and defaults to 1,000 residues. Therefore this is labeled a **backbone control**, not exact EnzGFM paper preprocessing. |

Every completed feature has a catalog hash and feature-file hash. Conformer fallbacks, cropping, structural fallback, and source-file provenance are recorded alongside the arrays. Molecular identity/participant counts are preserved; feature reuse does not add supervised reaction–enzyme associations.

## Original Horizyn comparison

The public model source and global multi-label NCE implementation are used unchanged. The native `sota` model has two 4,096-wide hidden layers:

- Reaction: 2,048 → 4,096 → 4,096 → 512.
- Protein: 1,024 → 4,096 → 4,096 → 512.
- ReLU towers, normalized outputs, cosine retrieval, global MLNCE with fixed β = 10.
- AdamW `1e-4`, weight decay 0.01, batch 16,384, 100 epochs.
- Select maximum mean bidirectional validation all-positive MRR every five epochs.
- BF16 tower training, FP32 MLNCE and scoring.

The benchmark's retrieval CSV gives a flattened participant string rather than a mapped directed reaction. The adapter writes `participants>>` and uses the original 1,024-dimensional RDKit+ and 1,024-dimensional DRFP generators. This preserves the supplied participant information but is **not the native directed-reaction representation**. Its row is named “Horizyn participant-set adapter.”

The underlying raw ReactZyme archive does contain Rhea substrate/product tables. Those tables require a separate, chemistry-only alignment audit to establish whether a flattened query corresponds to one directed reaction or a combined participant profile; no protein-associated Rhea labels are silently imported to repair a query. The participant-only adapters remain explicit in the initial comparison.

| Split | Validation-selected epoch | Test R→E MRR | Test E→R MRR |
|---|---:|---:|---:|
| Reaction similarity | 100 | 0.431815 | 0.534799 |
| Enzyme similarity | 95 | 0.699701 | 0.976990 |
| Time | 100 | 0.598203 | 0.840621 |

These are measured, newly trained results under the common evaluator. They show that the original parent model is a substantial comparator. They do not by themselves establish a win or loss against a selected V4 configuration.

## CREEP / CARE comparison

The public trainer supports protein–reaction two-modality training. The adapter preserves full fine-tuning of public ProtT5 and rxnfp BERT, 256-dimensional linear projection heads, the native protein mean over all 512 padded token positions, reaction CLS pooling, and symmetric EBM-NCE with one cyclic negative, temperature 0.1, and no training-time embedding normalization. Retrieval uses cosine-normalized projected vectors. Generative weight is zero, matching the public default.

No EC description text or EC-derived positive associations are supplied. Native EC batching can create enzyme–reaction Cartesian products within an EC class; the adapter instead shuffles the **exact** ReactZyme training edges. Reaction strings use the disclosed participant-set adapter. These are material adaptations and remain in the method label.

Training uses Adam `1e-5`, zero weight decay, 40 epochs, batch 64 (native default is 16), BF16 autocast, and FP32 parameters/optimizer. Checkpoint selection uses mean bidirectional validation all-positive MRR each epoch. The larger batch is a throughput adaptation, not claimed to be the native default.

Full fine-tuning is much more expensive than fitting fixed-feature heads. The initial reaction-similarity run uses eager T5 attention and block checkpointing. The two subsequently launched splits use fused unscaled T5 SDPA with the same learned relative bias, plus selective FFN activation recomputation. FP32 output/gradient fidelity is tested on CPU and CUDA. On a short full-size runtime probe, selective FFN recomputation used about 118 GiB and took 3.55–3.86 seconds per 64-example protein-only step; block recomputation took 4.10–4.65 seconds after warm-up. Concurrent jobs affect these measurements. Disabling checkpointing entirely exceeded the H200 memory budget and was not used for a training run.

## Structural inputs and public-source recovery

The user confirmed that no additional prepared full-coverage dataset is available. The local released EnzymeCAGE pocket collection at `../EnzymeCAGE/dataset/RHEA/2025-02-05/pockets/pocket` contains 236,439 accession-named files. Matching full sequences from its `all_enzymes.csv` to the fixed ReactZyme catalog covers **160,241/178,327 proteins**, leaving **18,086** unmatched. This checks sequence-to-file coverage; pocket residue contents and node-feature alignment still require validation.

All 18,086 unmatched sequences map to public accessions in ReactZyme's own raw UniProt table. Public coordinate retrieval is running using the [AlphaFold database's programmatic access](https://www.ebi.ac.uk/training/online/courses/navigating-alphafold-database/what-is-the-afdb/accessing-searching-afdb/programmatic-access-api/). Each downloaded PDB must contain one chain whose complete Cα residue sequence exactly matches the benchmark sequence. A newer version, accession match alone, or partial structure is insufficient. Files that fail are recorded unresolved. The first 32-protein pilot recovered 28 exact structures; four remained unresolved. Missing-pocket proteins are prioritized; full CLIPZyme coordinate coverage is a separate requirement.

The recovery worker applies the pinned native EnzymeCAGE P2Rank extraction helper to incoming structures in batches of 128, using P2Rank 2.5.1's AlphaFold configuration and the top-ranked pocket. Sixteen CPU threads perform pocket prediction while the GPUs train retrieval models. Per-protein failures are saved instead of being replaced with invented pockets. Acquisition and extraction status are under `assets/alphafold/status.json` and `assets/recovered_pockets/status.json`.

Downloading a coordinate file does **not** complete pocket extraction, ESM residue-feature alignment, reaction atom mapping, native graph construction, or full-model training. Those stages remain pending. No zero-filled geometry, silent candidate filtering, sequence-only replacement, or transferred supervised checkpoint is labeled as full CLIPZyme/EnzymeCAGE.

## Provenance, verification, and operations

[source_revisions.json](../../../runs/reactzyme_public_baselines_20260921/source_revisions.json) pins the checked upstream repositories. Per-run protocols also hash the imported model/loss files. The original shared working tree had extensive unrelated changes; this campaign uses its separate branch without resetting, staging, or committing that inherited work.

The campaign's [12 fidelity tests](../tests/unit/test_reactzyme_public_baselines.py) pass. They check original-versus-cached pair scoring for every family, deterministic negative sampling and exclusion of known positives, rejection of impossible negative pools, the all-positive metric definition, and fused T5 attention/gradient equivalence with and without selective recomputation on CPU/CUDA. The receipt and log are in the campaign directory.

Launch / recover the persistent queues from the `horizyn` repository:

```bash
../.capability-run-py/bin/python -u scripts/reactzyme_public_campaign.py
../.capability-run-py/bin/python -u scripts/reactzyme_public_extended.py
```

Both supervisors have exclusive locks and recover their own existing training processes. The official queue runs two lightweight jobs per GPU. The extended queue handles CREEP and feature-dependent EnzGFM jobs; Horizyn's completed runs are included in its report. Failures remain visible and require inspection; they are not counted as completed results.

The large EnzGFM residue archive is first copied sequentially to `/tmp/reactzyme_public_enzgfm` and pooled locally because four parallel NFS readers stalled imports and reduced throughput. The local copy is a disposable cache; results and provenance remain under the campaign directory.

Matching downstream associations is necessary for a fair comparison, but different pretrained backbones, structural inputs, objectives, batch sizes, and input adapters mean this is initially a **matched-data method comparison**. A strict architecture-only claim additionally needs common-input/common-loss ablations and comparable validation tuning budgets. Published numbers, native-method reproductions, and adapted controls remain distinguishable throughout the report.
