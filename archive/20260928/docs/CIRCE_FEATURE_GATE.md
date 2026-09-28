# Gate-first reaction fusion

This is a fresh fingerprint-free CIRCE-v3 reaction-smi run. UniMol2,
ReactionT5 participant strings and ChIRo supply the three reaction branches.
Frozen features are reused from the completed three-branch experiment; no
backbone extraction or reaction-side inference is performed.

Each branch is adapted by the reference F3-sized MLPs into a 512-dimensional
token. A joint gate sees the three tokens and their availability masks and
assigns softmax weights across available modalities independently for each
feature dimension. Their weighted sum goes into the unchanged
512→4096→4096→512 output MLP. There is no cross-modal attention before or after
this gate. The original within-branch molecule attention pooling remains.

`feature` uses channel-wise gates. The optional `scalar` control has the same
inputs, pooling, branch and output MLPs, enzyme tower, loss and global batch;
its joint gate instead outputs one scalar per modality, broadcast across
channels. Both start with uniform weights on available modalities. This control
is **not exactly original F3**: it uses joint conditioning and excludes chemistry
features. The larger capacity/pooling changes relative to the earlier compact
concat experiment are not attributable to gating alone.

## New multiview enzyme tower

`--enzyme-tower multiview` replaces the biological-factorized enzyme tower;
it is not an FFN-only modification. The default `baseline` remains unchanged.
The new sequence-only tower combines global mean pooling, a frozen-SLEEC
site view and four learned latent residue-pooling queries. These six views
are projected, feature-wise gated and transformed by an output FFN, with an
explicit global-mean residual path. SLEEC contributes a separate view instead
of imposing the same prior on every learned query. The output remains a
normalized 512-dimensional enzyme embedding, reusable for both retrieval
directions without query labels.

The four slots are latent views, **not** supervised mechanism/cofactor classes
or proven catalytic sites. The old EC projector and family-classification
branches are not part of this version. Legacy BCE classification stays disabled.
Positive-only biological anchor supervision is available with `--positive-biology`.
Frozen residue features and the pretrained frozen SLEEC scorer are reused.

Safeguards target distinct failure modes:

- Normalized keys/queries and a bounded attention scale limit score growth.
- A 5% uniform mixture over valid residues prevents exact one-residue attention;
  the raw attention is also monitored so this floor cannot conceal sharpening.
- A weak entropy-floor penalty discourages extremely narrow raw attention. Its
  target is `min(log(4), 0.5*log(valid_length))`, not uniform attention.
- A weak near-duplicate penalty acts on centered attention maps above cosine
  similarity 0.9; it does not require disjoint catalytic sites.
- A 5% total uniform gate mixture and global residual path protect against
  complete view starvation.

The objective is the existing CIRCE-v3 retrieval loss plus `0.01 * entropy_loss`
and `0.001 * diversity_loss`. These are starting hyperparameters, not a claim
of optimality. Diagnostics report raw/effective concentration, effective
support, query redundancy, gate usage and each weighted penalty.
Uniform maps are not forced apart: their raw similarity and zero eligible
centered-pair fraction are logged explicitly rather than reported as diverse.
No safeguard guarantees informative attention or better generalization.
The pooling cost is linear in sequence length, with no residue-by-residue
self-attention matrix.

The distinction between entropy collapse and head redundancy is motivated by
[Zhai et al. (2023)](https://proceedings.mlr.press/v202/zhai23a.html) and
[Li et al. (2018)](https://aclanthology.org/D18-1317/).
This implementation is not a reproduction of either method; the safeguards
above are explicit, testable design choices for this residue-pooling tower.

## Launch

### Positive-only biological supervision on four GPUs

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../env/bin/python scripts/run_circe_feature_gate.py launch \
  --gpus 0,1,2,3 --epochs 30 --variant feature \
  --enzyme-tower multiview --positive-biology --biofp-aux-weight 0.02
```

This starts a **fresh detached tmux run**, not a resume. It uses the ReactZyme
reaction-smi training split, frozen extracted features/SLEEC and freshly trained
retrieval towers. Batch size is 384 per GPU (1536 total). It does not deliberately
fill all VRAM; no maximum-throughput claim has been measured for this new variant.

Three small, label-independent readouts pool the four enzyme slots for EC,
cofactor and coarse transformation supervision. They align normalized outputs
to fixed normalized category anchors. EC anchors share specified ancestors;
other category anchors encode identities, not measured chemical distances.
They are auxiliary components of this model, not an annotation-imputation model,
and no annotations are required for retrieval inference. The four slots remain
latent views rather than forced biological classes.

Only observed positive indices are stored. Loss is `1 - cosine` to each positive
anchor, confidence-averaged within enzyme, then averaged over active enzyme rows
across DDP workers and over active families. An unannotated enzyme/family has
exactly zero annotation loss/gradient; no negative targets or unknown class are
manufactured. Local duplicate protein rows are removed before auxiliary loss;
an enzyme appearing on multiple workers still contributes once per worker.
Annotation confidence determines relative weighting among a protein's positives,
not its absolute weight versus other annotated proteins.

`biofp_aux_weight=0.02` controls the combined, family-normalized biological loss.
The existing three-epoch warmup gives effective weights 0, 0.00667, 0.01333,
then 0.02 from the fourth displayed epoch. Retrieval and the entropy/diversity
penalties are unchanged. Missing annotations are neutral; unknown reaction–enzyme
pairs remain downweighted contrastive competitors in the retrieval objective.

Targets are rebuilt once per run from raw sources. Reaction-derived evidence
must have a retained training edge. Protein annotations are restricted to training
proteins. The source hashes, coverage, filters and caveats are saved under
`annotations/positive_vocab.json`. EC/protein restriction does not establish
temporal independence of annotation sources; cofactor participants do not prove
required cofactors, and the mechanism vocabulary is coarse inferred chemistry.
Zero/invalid mapping confidence is excluded rather than promoted to weak evidence.

The existing `.capability-run-py` interpreter handles pandas/Parquet preparation
(`--annotation-python` overrides it); `../env/bin/python` remains the training
interpreter. Sparse label storage avoids a dense 147k-by-4578 EC matrix.
Logs include each raw/normalized family loss, effective lambda and weighted total,
observed-label counts, readout variance/usage and periodic shared-slot gradient
diagnostics. Fixed anchors and attention safeguards do not guarantee biological
interpretability, prevent every collapse mode, or guarantee better MRR.

### Existing unsupervised variants

On the machine with four free GPUs:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV PYTHONUNBUFFERED=1 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  ../env/bin/python scripts/run_circe_feature_gate.py launch \
  --gpus 0,1,2,3 --epochs 30 --variant feature
```

For the redesigned enzyme tower, add `--enzyme-tower multiview`. This creates
a fresh run; it does not alter or restart an already running job. Keep the
same reaction-gate variant and compare against `--enzyme-tower baseline` to
measure the enzyme redesign. With `--variant both`, both reaction-gate
variants use the selected enzyme tower.

Use `--variant both` to train feature then scalar sequentially for 30 epochs
each. Use `--gpus 0,1` for two free GPUs. Global batch remains 1536 (384 per GPU
on four devices). The original precision, optimizer, CIRCE-v3 retrieval loss, disabled
auxiliary classifiers and pretrained frozen SLEEC scorer are retained.
The multiview option additionally enables the two weak penalties above.
Trainable towers start fresh: no resume or retrieval checkpoint initialization.

The launcher creates a unique detached tmux session/run directory and prints
its pipeline log. It refuses occupied GPUs, never stops other jobs, validates
the participant-cache receipt and chemistry, and preserves older runs. Each
variant logs training to `<run>/<variant>/training.log` and CSV metrics under
`logs/protein_pooling_training`. All epoch checkpoints are retained, including
epoch=08 (nine completed epochs). Plan disk space for thirty full optimizer
checkpoints per variant.

Validation runs every epoch, plus standalone bidirectional validation of the
final checkpoint. This launcher does not evaluate the held-out test or select
hyperparameters using it. `complete.json` records successful completion.
The `prepare` stage is audit-only: its directory is not a resumable run; use
`launch` to create a fresh executable run after preflight.
Gate diagnostics report real competitive weights/entropy, not modality
availability interpreted as attention. A learned gate is not calibrated
biological reliability or proof that a modality causes a retrieval gain.
