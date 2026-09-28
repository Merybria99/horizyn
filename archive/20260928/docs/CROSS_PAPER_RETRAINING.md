# Fixed-architecture retraining for external paper comparisons

The ReactZyme result is the primary benchmark claim. A ReactZyme-trained model
scored on another paper's data is a **zero-shot transfer test**, not a matched
comparison to that paper. A downstream-data-controlled comparison requires
the same permitted labeled training rows and supervision policy for both
methods, the same validation partition and selection rule, and identical
held-out queries, candidate pools and metrics over independent seeds. Merely
retraining F3 on a competitor's positive associations is insufficient when
that competitor also trains on explicit negatives. The F3 plus residual/anchor
towers remain fixed, but any change to loss or negative use must be declared
before testing. Even after controlling downstream data, different pretrained
backbones and input features remain part of the complete-method comparison;
they prevent an architecture-only causal claim. This protocol was written after the
campaign's P450 and other external outcomes were known. A future run on those
same panels will therefore be a transparent retrospective comparison, not a
new independent confirmation.

## Architecture lock

[`architecture_lock_v3.json`](../runs/generalization_20260919_2251/cross_paper_retraining/architecture_lock_v3.json)
is the executable inventory. Recreate it with
`scripts/generalization_cross_paper_lock.py` in a fresh path. It verifies that
the local EnzymeCAGE and ReactZyme F3 configurations have byte-for-byte equal
`model` dictionaries and matching core training settings; the canonical model
fingerprint is `f1fcac210744212878dfaf3f220c9d819048faf66ce30c132f9e5edfa7c7a551`.
The lock also hashes the relevant model, data, loss, training and extension
source files so a later code change is visible before reuse.
This is the dual encoder with independent 512-dimensional enzyme and reaction
towers, frozen cached ProtT5/ReactionT5v2/UniMol2/ChIRo inputs, chemistry
features, and the frozen SLEEC protein scorer. It does not add a pairwise
cross-encoder or change the towers in response to a paper's test scores.

The base F3 training recipe is also locked: observed-pair full-batch MLNCE
with beta 10, AdamW learning rate 1e-4 and weight decay 0.01, FP32, up to 30
epochs, patience five on validation bidirectional MRR, four-GPU DDP, seed 42.
Each paper gets a **fresh base checkpoint trained only on its permitted training
pairs**. The existing EnzymeCAGE epoch-06 checkpoint is useful as a historical
pilot, but it is not a fresh run prepared under this lock. The ProtT5/SLEEC and
chemical foundation features may remain pretrained without target test labels;
their pretraining provenance must be reported separately from pair supervision.
This locked positive-pair recipe is a valid target-data F3 experiment, but it
does not by itself produce a training-controlled comparison against a model
using additional explicit negative labels. Any negative-aware F3 loss requires
a new, versioned recipe lock while retaining the same tower architecture.

The extension keeps the same independent-tower scoring: a hidden-1024,
scale-0.2 frozen-base residual trained on the full target training graph for
1,000 steps (AdamW 1e-4, weight decay 0.001, temperature 0.07, identity
weight 2, validation every 25); a train-only smooth-anchor dictionary (32
enzyme neighbors, all training reactions, temperature 0.03), with 75% learned
branch and 25% anchor branch. The F3 support-density gate uses the training
25th/95th-percentile rule and a fixed 4,096-protein calibration sample. The
optional fixed Morgan reaction anchor
(radius 3, 4,096 chiral count features). All density quantiles, chemistry
normalization, fingerprints, graphs, and anchor dictionaries are fit from that
target's **training partition only**. The residual step may be selected by the
predeclared validation bidirectional MRR with at most 0.005 loss in either
direction versus the target-trained base. No target test scores select a method,
weight, feature, or checkpoint. Report F3, phase 2, Morgan, and CIRCEv2 as
distinct architectures/recipes; do not relabel an existing ReactZyme checkpoint
as a target retrain. Where an architecture cannot process the target inputs,
record the coverage failure before scoring; do not silently drop queries.

## EnzymeCAGE target-data experiment; supervision still unmatched

The [paper](https://www.nature.com/articles/s41929-026-01478-y) and
[official repository](https://github.com/GENTEL-lab/EnzymeCAGE) define a Rhea
training lineage and separate Enzyme-405, Orphan-335, and external family
tests. Local EnzymeCAGE data contain **207,173 unique training positives** and
an original validation partition. The lock checks pair disjointness and pins
source hashes. The [split audit](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_split_audit.json)
finds zero shared pairs or reaction IDs and 1,499 protein IDs shared between
train and validation. The F3 converter discards every `Label=0` row. The source
training CSV has 1,445,915 labeled rows: 243,890 positive rows (207,173 unique
positive pairs) and 1,202,025 negative rows. EnzymeCAGE's released seed-42
training config reads that CSV with `label_column_name: Label`, and its
training code uses binary cross-entropy over those labels. The completed F3
run therefore uses the same positive source but **not the same supervision**.
A controlled next experiment must either retrain EnzymeCAGE under a declared
positive-only protocol or train the fixed F3 towers with the same labeled
positive and negative rows under a predeclared loss. Preserve original
validation/test partitions and candidate IDs, run independent seeds, and
authenticate the released checkpoint's training provenance.

The first target-base config is
[`configs/train.yaml`](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42/configs/train.yaml);
`preparation.json` binds its source and original v1 architecture lock; v2
added the already selected density-gate constants and v3 adds implementation
source hashes, without altering F3.
Its only changes
from the already matched local F3 configuration are output locations, a run
label, and disabled W&B logging. The first launch was
[interrupted before a checkpoint](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42/interruption.json):
four workers waited on network-backed residue shards. The local-stage script
copied, SHA-verified and rebuilt the 122 GB ProT5 virtual dataset without
changing any input values. Its [archived receipt](../runs/generalization_20260919_2251/cross_paper_retraining/local_residue_staging_receipt.json)
records four shard hashes, identical IDs and offsets, and sampled row equality.
A new run uses that receipt and a fresh output path. The first config remains an audit artifact; do
not resume an uncheckpointed interrupted run. From `horizyn`, run:

The [local-cache config](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42_local/configs/train.yaml)
is now prepared and bound to v3. It differs from the first config only in the
verified residue VDS path and output locations. The
[launch receipt](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42_local/launch.json)
records the four-GPU run started at 06:05 UTC. It completed normally after
epoch 11 with early stopping. The validation-selected checkpoint is epoch 6,
mean bidirectional MRR **0.405478**. Inspect its `pipeline.log`, metrics CSV
and checkpoints. The [fresh-state replay](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_epoch06_state_replay.json)
finds all 92 selected model-state tensors exactly equal to the old epoch-6
pilot, confirming deterministic reproduction rather than a new seed.
The [replay audit](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_initial_epoch_replay.json)
finds exact equality of all seven validation MRR values at epochs 0–6 versus
the historical interrupted pilot. This supports the byte-verified cache copy
and deterministic fixed recipe; it is not a new seed or independent test.

```bash
../env/bin/python scripts/generalization_prepare_matched_enzymecage.py \
  --architecture-lock runs/generalization_20260919_2251/cross_paper_retraining/architecture_lock_v3.json \
  --staging-receipt runs/generalization_20260919_2251/cross_paper_retraining/local_residue_staging_receipt.json \
  --output runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42_local
CUDA_VISIBLE_DEVICES=0,1,2,3 ../env/bin/python scripts/train_protein_pooling.py \
  --config runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42_local/configs/train.yaml
```

The phase-2 extension has now been retrained on **this checkpoint's** target
train/validation representations and 207,173 training edges. Its frozen recipe,
validation result and retrospective P450 result are in the
[target phase-2 report](../runs/generalization_20260919_2251/cross_paper_retraining/phase2_target/README.md).
The existing campaign phase-2/Morgan checkpoints are ReactZyme-trained and
were not reused for the target extension. The source EnzymeCAGE F3 config uses physical directional reactions;
retain that policy for target training and all external queries. The ReactZyme
participant-self-reaction policy must not be mixed into its input caches.

For the P450 test, use the [pinned EnzymeCAGE evaluator](https://github.com/GENTEL-lab/EnzymeCAGE/blob/e6887c4109e7c4a861b9cb1a3e33beede90103dc/scripts/evaluate_external-test.py):
191 queries, exactly 490 candidate **IDs**, known-positive retrieval at
Top-4/14/24 (1/3/5% floor). Preserve duplicate-sequence IDs. Report raw
ranking and the official homology/reaction prior separately; the prior uses
external information and must never be applied only to our model. The
P450-finetuned EnzymeCAGE checkpoint has additional P450 supervised training:
compare it in a separate fine-tuned arm, not against our pretraining-only arm.
Unlisted P450 pairs are not measured inactive outcomes. Enzyme-405 and
Orphan-335 require their own official query/candidate construction and their
published evaluator, rather than reusing the P450 pool or ReactZyme MRR. The
[local P450 audit](../runs/generalization_20260919_2251/p450_audit/README.md)
records these distinctions.
The local official-code checkout has Enzyme-405 and Orphan-335 inference
configs **and their test CSVs**, but the referenced feature assets are
incomplete and the Enzyme-405 config names private absolute feature paths.
Their query/candidate construction and F3 inputs still need a separate audit;
neither test can be inferred from P450 scores.
The [fresh F3 score exporter](../scripts/generalization_matched_enzymecage_p450_export.py)
uses the validation-best checkpoint and the already generated **physical**
P450 input features, saving all 191 × 490 raw cosine scores before opening
labels. The [separate evaluator](../scripts/generalization_matched_enzymecage_p450_evaluate.py)
checks the official raw Top-4/14/24 metric against an independent rank count.
It reports the official external homology/reaction-prior arm separately, using
the same fixed score matrix.
The [paired comparison script](../scripts/generalization_matched_enzymecage_p450_compare.py)
then contrasts raw F3 ranks with the pinned EnzymeCAGE pretrained arm on the
same 191 queries and gives a query-bootstrap interval. It lists the
P450-finetuned arm separately because that model used additional P450 labels.
These scripts were prepared before the new base run finished, then executed
after the completion marker and validation-only checkpoint selection. The
[official evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_evaluation/summary.json)
gives fresh F3 raw Top-4/14/24 **0.02618/0.09948/0.15707** versus EnzymeCAGE
pretrained **0.02618/0.09948/0.19895**. The
[paired comparison](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_comparison/summary.json)
has Top-24 difference −0.04188, query-bootstrap interval
[−0.10995, 0.02618]. This row concerns the fixed F3 base only. The later
positive-only target phase-2 P450 Top-4/14/24 result is
**0.01571/0.12042/0.20942**, with lower first-positive MRR than the
EnzymeCAGE pretrained arm. Its different supervision prevents a controlled
method claim.
The exporter enforces the trainer's completion marker before any P450 score is
written; an active-run guard test failed closed without creating output.
The fresh config loads a saved epoch checkpoint, all 191 P450 reaction input
rows have the required modalities, and the evaluator reproduces the previous
pilot's official raw and prior-gated Top-4/14/24 values exactly in a
[score-matrix smoke test](../runs/generalization_20260919_2251/cross_paper_retraining/p450_evaluator_smoke.json).

```bash
../env/bin/python scripts/generalization_matched_enzymecage_p450_export.py \
  --run runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_seed42_local \
  --output runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_scores
.deps/cyp-external-env/bin/python scripts/generalization_matched_enzymecage_p450_evaluate.py \
  --export runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_scores \
  --output runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_evaluation
.deps/cyp-external-env/bin/python scripts/generalization_matched_enzymecage_p450_compare.py \
  --export runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_scores \
  --evaluation runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_evaluation \
  --output runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_p450_comparison
```

The [background finalizer](../scripts/generalization_wait_finalize_enzymecage.py)
ran exactly those three commands after the training success marker. Its
[completion receipt](../runs/generalization_20260919_2251/cross_paper_retraining/enzymecage_f3_finalize_status.json)
binds the model score, official evaluation and paired-comparison hashes. The
active-run guard test failed closed without writing a score.

## All-label loss-alignment diagnostic

The [all-label data preparation](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/README.md)
retains every original EnzymeCAGE training and validation row, including
explicit negatives, duplicates and 180 sequence/reaction pairs with conflicting
labels. Thirty-four missing protein feature vectors were extracted. A frozen
positive-trained F3 base plus three independent residual seeds was then fitted
with unweighted BCE over **all 1,445,915 labeled training rows**, selecting
each checkpoint by original-validation AUROC as in the EnzymeCAGE recipe.
The unfitted frozen F3 cosine gave validation AUROC **0.5305**. Best BCE
residuals gave **0.4545, 0.4641 and 0.4411** for seeds 42, 17 and 73,
all at epoch zero. This adaptation failed validation and was not promoted to
the P450 test. It aligns the labeled rows, loss family and selection metric,
but freezes a base pretrained on positives; it is **not** a full end-to-end
F3 base retrain with BCE. That later retrain was completed: fresh F3 towers
reached original-validation AUROC **0.61718**, but official P450 Top-24 only
**0.06806** against pretrained EnzymeCAGE **0.19895**. All 191 P450 queries
choose one top enzyme, showing reaction-conditioning collapse. See the
[completion receipt](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_seed42_gpu_cache/complete.json)
and [external evaluation](../runs/generalization_20260919_2251/cross_paper_retraining/labeled_parity_v1/full_f3_bce_scratch_p450_evaluation/summary.json).
Different pretrained input backbones remain a complete-method difference.

## FGW-CLIP/EnzymeMap matched experiment

The [FGW-CLIP v2 paper](https://arxiv.org/pdf/2512.08508) uses the
[CLIPZyme EnzymeMap release](https://github.com/pgmikhael/clipzyme): 34,427
training, 7,287 validation, and 4,642 test associations split by
**reaction-rule IDs**. The originally inspected local checkout lacked the
release. The [official Zenodo v4 archive](https://zenodo.org/records/15161343)
and rule split have now been checked against their published MD5s; every
extracted file has a SHA-256 and archive CRC in the
[acquisition receipt](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_data_acquisition.json).
The [release audit](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_release_audit.json)
confirms all **46,356 cached associations**, their exact three split counts,
321/5/68 disjoint rule IDs and **261,907 unique screening IDs** with a
matching 261,907 × 1,280 CLIPZyme embedding matrix. The 349,431 raw JSON
reaction rows are not the eligible association set. The paper describes the
screening sequences as at most 650 aa, but the released ID-to-sequence map has
**six longer sequences and 72 empty sequences**. All 72 empty sequences were
later recovered from the UniProt archive (50 exact 2022_01, 22 unchanged
across bracketing archived versions); preserve all released candidate IDs and
disclose this reconstruction. The CLIPZyme protein embeddings are a release
audit, **not inputs to our F3 architecture**. Our own protein/reaction input
features still need to be built for the target training graph and screening
library, followed by a fresh target retrain.
The test association metadata were inspected for input coverage and overlap
before model training, so a future held-out EnzymeMap result is a
paper-matched retrospective evaluation, not a newly blinded discovery test.
No test model scores were used to choose the architecture or extension.

The [source-order manifest](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_manifests_v2/manifest.json)
materializes separate train, validation and test association CSVs and the
unchanged screening order, each with a SHA-256. The training run may read only
the train CSV; checkpoint selection may read the dev CSV; test associations
remain evaluation-only. Rule disjointness does **not** imply total input
novelty: train/test share 183 exact protein IDs and 30 exact directional
reaction strings, although no exact UniProt-ID/reaction pair; train/dev share
218 proteins and nine reaction strings, also with no exact ID pair. The
[sequence-level audit](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_sequence_overlap.json)
finds **two train/test pairs with identical protein sequence and reaction**
under distinct UniProt IDs. Report these
exposure strata alongside the paper-matched aggregate. Do not alter the
paper's split to erase the exposure when making a direct FGW-CLIP comparison.
At the association-row level, the
[input-only stratification](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_input_exposure/summary.json)
finds **552/4,642 test rows with a training protein ID, 569 with an exact
training sequence, and 98 with an exact reaction string**. These flags are
pinned before any target model score; they support a familiar-input versus
new-input sensitivity without changing the released aggregate.
The [existing F3 ProtT5 coverage audit](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_existing_f3_feature_coverage_v2.json)
finds cached sequence features for 4,764 of 9,794 unique training protein IDs
and 173,745 of 261,907 screening IDs. Another **78,221 distinct nonempty
sequences** needed ProtT5 extraction, representing 88,090 candidate IDs.
The original 72 empty IDs occur in **162 train, two validation and four test
association rows** (38/1/3 distinct IDs). The
[strict F3 catalog preparer](../scripts/generalization_prepare_matched_clipzyme.py)
now verifies the [archive rescue receipt](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_unisave_2022_01_rescue_v3/receipt.json)
and materializes the complete [target F3 catalog](../runs/generalization_20260919_2251/cross_paper_retraining/clipzyme_f3_catalog_v1/preparation.json)
without dropping any association or screening ID. The complete library has
**222,985** unique screening sequences; **78,287** require new ProtT5
extraction after rescue. Train-only chemistry features are complete, while
ProtT5, ReactionT5v2 and UniMol2 extraction is complete. ChIRo covers
16,753 of 16,776 reactions from a read-only molecule cache; F3's declared
missing-ChIRo handling retains the other modalities. The fresh four-GPU
F3 training run is in progress.

Train the same locked architecture and extension with only the EnzymeMap
training entries; select from the 7,287 validation entries. On the held-out
4,642-entry query set, rank the **same 261,907 proteins** and compute the
official BEDROC at alpha 85 and 20 and enrichment factors at 5% and 10%.
Then repeat the paper's separate Table-2 protocol after removing all training
enzymes from the screening pool. Do not compare ReactZyme MRR or Case-1
curated-candidate recovery to those numbers. The v2 paper reports Table-1
FGW-CLIP values 48.66%/66.69% BEDROC and 14.91/8.18 enrichment; these are
paper references, **not reproduced results for our model**.

## Report gate

A direct cross-paper performance row is publishable only with: locked model
and training source hashes; target-specific train/validation/test IDs and
overlap audit; training completion and validation-only checkpoint choice;
feature coverage; identical candidate IDs/order and metric implementation;
raw scores or a hash-verifiable prediction file; and a clearly labelled
comparator training regime. The newly target-trained EnzymeCAGE P450 phase-2
row satisfies the target-data and score-provenance parts of this gate, but its
positive-only supervision still differs from EnzymeCAGE's labeled recipe.
The earlier ReactZyme-trained P450, Case 1, nitrilase, aminotransferase and
esterase scores remain transfer or diagnostic evidence. Neither set supports
a controlled method claim against EnzymeCAGE or FGW-CLIP yet.
