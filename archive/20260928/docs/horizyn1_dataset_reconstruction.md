# Horizyn-1 inference-dataset reconstruction

This pipeline builds an auditable approximation of the large training graph
described by Rocks et al. from the same named public releases:

- the exact development data distributed with this repository;
- Rhea release 131;
- reviewed and unreviewed UniProtKB release 2023_05;
- the pinned local EnzymeMap v2 checkout.

It does **not** claim bit-for-bit identity with the authors' unreleased dataset.
The paper does not specify every filtering, reaction-standardization, duplicate
resolution, or MMseqs2 option. The manifests compare every major count with the
paper, making those differences measurable rather than silent.

## Outputs

The default root is `data/reconstructed/horizyn1_2023_05`:

| File | Meaning |
|---|---|
| `raw/raw_pairs.tsv` | Deduplicated pre-clustering reaction-protein edges with provenance |
| `raw/raw_reactions.tsv` | Rhea development and EnzymeMap reactions |
| `raw/raw_proteins.fasta` | Pre-clustering sequences |
| `raw/raw_manifest.json` | Counts and deltas from the published raw graph |
| `clustered/pairs.tsv` | Edges collapsed onto 80%-identity protein representatives |
| `clustered/proteins.fasta` | Representative protein sequences |
| `clustered/clusters.tsv` | MMseqs2 representative-to-member mapping |
| `clustered/clustered_manifest.json` | Counts, commands, and deltas from published counts |

Reported paper checkpoints are 31,101 raw reactions, 27,099,893 raw proteins,
34,385,290 raw edges, 7,063,237 clustered proteins, and 8,897,870 clustered
edges. Reaction-direction augmentation is intentionally not applied to the
stored graph; it belongs in the training data loader.

The completed local reconstruction audited on 2026-09-08 contains:

| Stage | Protein entries | Reaction–protein pairs | Reactions |
|---|---:|---:|---:|
| Raw | 24,601,544 | 30,040,397 | 30,961 |
| Clustered representatives | 6,149,161 | 7,571,161 | 30,961 |

All raw entries have exactly one cluster membership. These are observed
public-source reconstruction counts, not a claim to reproduce the paper's
unreleased dataset exactly. Annotation coverage is measured separately by the
annotation manifests after that additional stage finishes.

## Run by stages

The preflight is read-only apart from its small JSON report:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
bash scripts/run_horizyn1_dataset_reconstruction.sh preflight
```

Download the historical archives (about 174 GiB compressed):

```bash
bash scripts/run_horizyn1_dataset_reconstruction.sh download
```

Then parse, merge, and cluster:

```bash
THREADS=128 SORT_THREADS=32 bash scripts/run_horizyn1_dataset_reconstruction.sh parse
THREADS=128 SORT_THREADS=32 bash scripts/run_horizyn1_dataset_reconstruction.sh merge
THREADS=128 SORT_THREADS=32 bash scripts/run_horizyn1_dataset_reconstruction.sh cluster
```

Or execute all stages in one foreground job:

```bash
THREADS=128 SORT_THREADS=32 bash scripts/run_horizyn1_dataset_reconstruction.sh all
```

Each expensive stage writes its manifest last and is skipped on the next run.
Downloads use wget resume into an isolated `.clean.part` file. The UniProt
archive must match its published size and MD5 before promotion to the final
filename. Set `FORCE=1` only to intentionally rebuild an already completed
stage. Set `RUN_ROOT` to place the reconstruction elsewhere.

## Integrity audit

After a stage finishes, independently check its completed artifacts with:

```bash
python3 scripts/audit_horizyn1_reconstruction.py \
  --run-root data/reconstructed/horizyn1_2023_05 \
  --stage clustered \
  --output data/reconstructed/horizyn1_2023_05/logs/clustered_integrity_audit.json
```

Use `--stage raw` to check the raw graph before clustering is complete. The
auditor measures counts from the files, checks identifiers and references,
verifies cluster membership and representative sequences, and compares the
final edge/provenance union with the raw edges mapped onto representatives.
Missing, partial, or changing artifacts cannot receive a passing report.
Reports retain the device, inode, size, and nanosecond modification/change
timestamps of every checked artifact in `artifact_signatures`. The label
auditor requires those versions still to match, so a stale graph report cannot
silently validate changed graph files. If using an older report without this
field, rerun the graph audit before the final label audit.
The auditor uses one CPU process and holds identifiers and collapsed-edge
keys in memory; it does not retain protein sequences or all raw edges.

For a running reconstruction in the `horizyn1_dataset` tmux session, the
following watcher waits for the completion manifests and runs both audits:

```bash
bash scripts/watch_horizyn1_audits.sh
```

Set `RUN_ROOT`, `PIPELINE_SESSION`, or `PYTHON` to override the defaults. Reports
are saved to `logs/raw_integrity_audit.json` and
`logs/clustered_integrity_audit.json`. An audit failure stops the audit watcher
and leaves the reconstruction process untouched for diagnosis.

Differences from the paper's counts are reported, not treated as integrity
failures. Passing this audit does not establish annotation correctness,
chemical equivalence, absence of leakage, or the MMseqs similarity thresholds;
it establishes consistency of the reconstructed graph artifacts.

## Construction details

The parser streams the release tarball and never expands the complete UniProt
flat files. A TrEMBL record is retained when one of its Rhea annotations maps to
a master Rhea reaction present in the released development graph. EnzymeMap
UniProt accessions are resolved during the same pass, including secondary to
primary accession mapping. The merge uses external GNU sort, and the 80%
sequence-identity collapse uses the repository's MMseqs2 binary followed by an
external sort/join of graph edges onto cluster representatives.

The paper only states `--min-seq-id 0.8`; the pipeline therefore records and
otherwise retains the installed MMseqs2 defaults (including its default 0.8
coverage threshold). Any extra clustering token can be supplied directly to
the Python command with repeated `--extra-cluster-arg` options once the authors
confirm their exact command.

## CIRCE-v2 sequence annotations

The additional annotation pipeline uses the **current CIRCE-v2** contract:
direct EC labels plus the ordered 8-mechanism/10-cofactor BioFP vocabulary.
These annotations support typed-negative construction; the canonical CIRCE-v2
configuration does not require new GO/domain/site labels or SLEEC retraining.

Native annotations require a second metadata-focused scan of the already
verified UniProt archive: the original reconstruction retained sequences and
Rhea links, but not EC or cofactor comments. This is CPU-only and does not
repeat clustering. Allow additional source-scanning time beyond reconstruction.

Create a small CPU environment, putting its cache on the data drive rather
than the quota-limited home filesystem:

```bash
uv --cache-dir .cache/horizyn1-annotations-uv venv \
  --python /usr/bin/python3 .venv-horizyn1-annotations
uv --cache-dir .cache/horizyn1-annotations-uv pip install \
  --python .venv-horizyn1-annotations/bin/python \
  numpy==1.26.4 pandas==2.2.3 pyarrow==18.1.0 PyYAML==6.0.2
```

After the raw and clustered graph audits pass, run the complete annotation
chain (native extraction, cached reaction descriptors, representative export,
and independent full label audit):

```bash
bash scripts/run_horizyn1_circe_v2_annotations.sh all
```

The same launcher supports `extract` and `export` separately. It uses a process
lock to prevent concurrent annotation writers. Set `RUN_ROOT`,
`ANNOTATION_PYTHON`, `NATIVE_PYTHON`, `CACHED_FEATURES`, or `CACHED_REACTIONS`
to override paths. Completed native extraction is checksum-verified on resume;
incomplete existing native outputs are preserved for diagnosis, not overwritten.
Do not modify the native extractor or its shared reconstruction reader during
an active archive scan: implementation changes deliberately invalidate it.

| Artifact under the reconstruction root | Contents |
|---|---|
| `annotations/native_uniprot.tsv.gz` | One row per raw sequence, including unresolved/mismatched/ambiguous entries; native EC/cofactor labels, ECO evidence, and sequence hashes |
| `annotations/native_uniprot_manifest.json` | Frozen-release provenance, matching/missingness counts, input signatures, and output checksum |
| `annotations/reaction_annotations.json` | Sparse reaction-associated descriptors, accepted only after exact chemistry matching to cached features |
| `annotations/circe_v2/enzyme_ec_labels.csv` | Exact representative IDs with direct EC labels and known prefix depth; multiple and incomplete ECs preserved |
| `annotations/circe_v2/enzyme_biofp_targets.npz` | Representative IDs and compatible mechanism/cofactor target/mask arrays; native and reaction-derived cofactor arrays also kept separately |
| `annotations/circe_v2/candidate_eligibility.csv` | Candidate-only gates excluding missing/ambiguous direct ECs and observed cluster-member EC conflicts |
| `annotations/circe_v2/cluster_annotation_summary.tsv.gz` | Member counts and member EC evidence, explicitly separate from representative labels |
| `annotations/circe_v2/enzyme_biofp_vocab.json` | Ordered vocabularies, provenance, scope, coverage, and limitations |
| `annotations/circe_v2/label_manifest.json` | Completed export metadata and input/output stability signatures |
| `logs/circe_v2_label_integrity_audit.json` | Independent full-artifact label checks; must pass before treating the label export as verified |

### Biological and split safeguards

- An accession match alone is insufficient: native annotations require an exact
  case-normalized sequence SHA-256 match. Missing, mismatched, and ambiguous
  sequences retain empty native label lists.
- No EC, cofactor, or mechanism label is assigned to a representative merely
  because another sequence belongs to its 80%-identity cluster. Member EC
  evidence is retained separately and can exclude unsafe negative candidates.
- Mechanism features are weak reaction-associated descriptors, not measured
  catalytic mechanisms. Only the representative's **own raw source
  associations** contribute to its weak profile, not the collapsed union of
  other cluster members' reactions.
- Masks are positive-only: an unobserved feature remains unknown, not a negative
  biological statement. Provenance weights (native 1.0, reaction-associated
  0.4) are not calibrated confidence or proof of experimental validation. Native
  UniProt annotations may themselves be inferred; their original ECO evidence
  remains available in the native table.
- The default chain produces `pair_scope=unsplit_inventory`. Before held-out
  evaluation, regenerate weak profiles using permitted **training-only raw
  source associations** and `--pair-scope train`. Merely subsetting a full-graph
  NPZ after computing it is not leakage-safe.
- Pass `--candidate-eligibility` to `build_annotation_negative_pools.py` when
  using these exports. This filters candidates without removing EC information
  needed to exclude known positives. It does not establish experimental
  inactivity. Query-specific exclusions and pool coverage checks still apply.
- Schema-compatible labels alone do not make the million-sequence dataset
  training-ready: train/validation/test splitting, feature extraction, and
  scalable pool construction are separate steps, and are not launched here.
