# Native CYP external baselines

**Completion update (2026-09-20):** CLIPZyme, official Horizyn-1 development,
and EnzymeCAGE original pretrained checkpoints have all produced complete
10,105-pair native scores and been imported into
[`runs/cyp_baselines_v1/report/summary.md`](../runs/cyp_baselines_v1/report/summary.md).
The CLIPZyme wrapper needed a PyG 2.6 `Inspector.distribute` compatibility
alias; its native model weights and source were unchanged. CLIPZyme and
EnzymeCAGE used documented input recovery for missing released structures and
pockets. The sections below preserve the original blockers and recovery steps.

These wrappers leave the completed `cyp_baselines_v1` protocol and scores unchanged.
They require all 10,105 frozen query-candidate pairs and fail rather than filter
missing candidates, use zero features, or import released predictions as new inference.

## Verified runnable: official Horizyn-1 development checkpoint

From a machine with this shared checkout and a free GPU, run:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
bash scripts/launch_cyp_external.sh --gpus 0 --models horizyn1_dev
```

Replace `0` with one free physical GPU ID. This is single-GPU inference, not
training. No existing processes are stopped. The controller checks GPU occupancy
after CPU preflight; availability is not a scheduler reservation. Closing SSH is safe.

Logs: `runs/cyp_external_v1/pipeline.log` and `horizyn1_dev.log` in the same directory.
Results are strictly imported into `runs/cyp_baselines_v1/report/summary.md`.
An existing completed score file is re-imported, not recomputed or overwritten.

The runner loads the pinned official development checkpoint (not CIRCE/F3), uses
the official forward-reaction RDKitPlus+DRFP implementation and native projection
layers, and scores cosine similarities. For full-length proteins of at most 1,022
residues it averages the existing float16 residue cache in float32. Longer proteins
are freshly encoded without truncation. This cache precision difference is recorded;
the run is not claimed to be a bitwise reproduction of an unpublished feature cache.

## Initial released-input coverage blockers

- **CLIPZyme:** 126 of the 4,135 benchmark protein IDs lack the referenced CIF in
  the released evaluation structures. The wrapper uses native ESM2/EGNN and mapped
  reaction encoders, but will refuse the full-pool run until structures are supplied.
  It also requires matching structure sequences and does not apply the training
  loader's 650-residue candidate cutoff.
- **EnzymeCAGE:** original seed-42 checkpoint loads, but released pocket/GVP
  features do not cover the full pool. For example, query
  `r_A0A1D6GQ67_1804_1478` lacks features for 20 of its 475 candidates.
  The wrapper checks all required feature keys, sequences and candidate identities.
  Released reactions omit co-substrates; this is explicitly recorded as model-specific
  input scope. This is one fixed seed, not a five-seed paper ensemble.

These were the initial blockers. Both methods subsequently completed full-pool
native inference after input recovery. Their recovered-input provenance is
part of each score receipt; no filtered row entered the full-pool table.

CPU preflight (now passes after recovery):

```bash
.deps/cyp-external-env/bin/python scripts/run_cyp_external.py preflight \
  --models horizyn1_dev clipzyme_pretrained enzymecage_pretrained
```

## Environment and provenance

Isolated `.deps/cyp-external-env`: Python 3.12, Torch 2.6.0 CUDA 12.4,
PyG 2.6.1 and matching prebuilt scatter/cluster/sparse wheels, Transformers 4.57.6,
NumPy 1.26.4. DRFP 0.3.7 was installed without dependency resolution because the
official API needs `atom_index_mapping`; all 45 reaction fingerprints were tested.
No training environment was changed. A complete environment snapshot is stored in
`configs/benchmarks/cyp_external_environment.txt`.

Source revisions and model checksums accompany generated scores. Checkpoints use
Torch's restricted weights-only loading; feature pickle globals are allowlisted.
The asset extractor checks archive SHA256 and rejects archive path traversal/links.
Upstream training overlap remains **UNKNOWN**, not certified zero-shot.

Tests: `tests/unit/test_cyp_external.py`, plus existing CYP baseline/specificity tests.
Full GPU evaluation has not been launched by this implementation pass.

## Full-pool recovery (2026-09-16)

The user chose input recovery/generation, not a restricted benchmark. Detached
sessions on Tyrosine are `cyp_clip_recovery` and `cyp_cage_recovery`.
Recovery is **not completed inference**. Each chain runs native inference only if
recovery and full-pool preflight succeed; otherwise its log records the remaining gap.

- CLIPZyme checks all structure sequences. It tries exact full-length AlphaFold
  API matches, then exact-sequence aliases, then sequence-only ESMFold monomers
  from the already cached pinned snapshot. No reaction-conditioned Boltz complex
  is substituted into this dual encoder. Generated structures are recorded by
  sequence, source, model revision, and content hash in `recovery/clipzyme/structures.json`.
  This mixed recovered-structure setting is not an exact paper-feature reproduction.
  Unknown amino acids (`X`) remain unknown. For sequences containing X, recovery
  exports the predicted backbone-frame CA coordinates as a CA-only PDB, using
  `UNK` for X, since native atom masks otherwise omit these positions. CLIPZyme
  uses residue-level CA graphs; its ESM input retains the original X tokens.
  No amino acid is guessed, no sequence position is removed, and this coordinate
  policy is recorded per generated structure.
- EnzymeCAGE has 153 missing feature pairs across 16 queries. Rebuilding from its
  released Boltz complexes recovered 29 pairs; 124 complexes were absent from that
  archive. Checking the separate checksum-pinned BoltzCYP archive confirmed that
  all 124 expected complex filenames are absent there too. The recovery chain now
  includes explicit Boltz complex generation before pocket rebuilding and inference.
  Missing data still causes a hard stop, never zero features.
  Recovered GVP/ESM pocket additions are separate sidecars, leaving released caches
  intact. The authors' ESM-C special-token indexing is preserved and recorded.

Logs are `runs/cyp_external_v1/clipzyme_recovery.log` and
`runs/cyp_external_v1/enzymecage_recovery_pipeline.log`. No candidate pool or old
comparison result is overwritten. Scores are imported only with complete coverage.

### Generate the missing EnzymeCAGE complexes

`recover_cyp_enzymecage_pipeline.sh GPU_ID` now extracts the released FASTA/MSA
inputs, inventories missing complexes, generates them, rebuilds native pockets,
and runs full-pool EnzymeCAGE inference/import. It does not stop existing jobs.
Use a free GPU; the pipeline refuses an occupied device. A lock prevents duplicate
recovery pipelines. This check is not a scheduler reservation.

Generation uses a separate `.deps/cyp-boltz-env` environment, pinned by
`configs/benchmarks/cyp_boltz_environment.txt`: Boltz **0.4.1**, as stated in the
BoltzCYP README. Protein A, heme B, and substrate C are checked against the
released inputs; local released MSAs must match the full protein sequence.
The released MSA archive lacks 26 of the 39 required proteins. The pipeline
explicitly enables `--generate-missing-msas`: Boltz's ColabFold service receives
these public protein sequences, once per protein, and the resulting A3M is cached
with its sequence, checksum, server, and generation date. Search timeouts/errors
stop the run; no silent single-sequence fallback is permitted. Server databases
may differ from the authors' original searches, so this is a recovered-input
baseline, not an exact reconstruction of their original structure features.
Settings are 3 recycling steps, 200 sampling steps, one diffusion sample,
step scale 1.638, and fixed seed 42. The seed is a new reproducibility choice,
not a claim to recover the authors' original random draw.

Boltz model/CCD downloads are commit-pinned and SHA256-verified before the native
loader reads them. This isolated historical Boltz environment uses its native
trusted-asset deserialization, unlike the restricted baseline-checkpoint loaders.
Each completed complex has an input/protocol/weights receipt and CIF checksum in
`runs/cyp_external_v1/recovery/enzymecage_generated`. Restarts reuse verified
completed complexes; an interrupted prediction is rerun. Generated complexes are
not represented as released paper structures, and no score-based selection occurs.
The frozen 10,105-pair benchmark and completed baseline results are unchanged.

The EnzymeCAGE preflight compares each released reduced reaction with the mapped
benchmark reaction. For two queries (`r_P9WPP3_1012_860` and
`r_P9WPP3_1013_861`), equivalent product graphs with identical stereochemistry
have different canonical SMILES strings after atom maps are removed. Validation
now checks bidirectional stereochemical graph identity for components that differ
as strings; genuinely different products still fail. These query IDs are included
in the score provenance as `reaction_graph_equivalent_queries`.

Detached launch (replace `0` with a free physical GPU):

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
tmux new-session -d -s cyp_cage_generate -c "$PWD" \
  'exec bash scripts/recover_cyp_enzymecage_pipeline.sh 0'
tail -f runs/cyp_external_v1/enzymecage_recovery_pipeline.log
```

For a bounded generation smoke test after input extraction and inventory:

```bash
.deps/cyp-boltz-env/bin/python scripts/generate_cyp_complexes.py --gpu 0 --limit 1
```

Unit coverage: `tests/unit/test_cyp_complex_generation.py` checks matching inputs,
MSA enforcement, occupied-GPU refusal, and resume integrity. A smoke test does not
certify full recovery: every complex must still pass native pocket extraction.
