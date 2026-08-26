# Wet-Lab Enzyme Retrieval

This directory queries a trained Horizyn model with one real-world reaction and
returns the most compatible candidate enzymes.

## Run

Edit `wet_lab/configs/f4_horizyn.yaml`, especially:

- `reaction.id`
- `reaction.smiles`
- `model.config`
- `model.checkpoint`

Paths in the YAML are resolved relative to the repository root. Then run:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
CUDA_VISIBLE_DEVICES=1 ../env/bin/python -m wet_lab.query \
  --config wet_lab/configs/f4_horizyn.yaml
```

The example uses the fully specified D-tagatose to D-fructose C4
epimerization. A query may use either `reactants>>products` or
`reactants>agents>products` reaction SMILES.

## Outputs

Each reaction and feature configuration gets a deterministic directory under
`wet_lab/runs/`. It contains:

- `results.json`: query, model, candidate-pool, cache, and ranking provenance.
- `rankings.csv`: the complete ranking up to the largest requested cutoff.
- `top_1.csv`, `top_5.csv`, ..., `top_100.csv`: requested retrieval cutoffs.
- `top_100.fasta`: sequences for the ranked enzymes when `candidate_pool.fasta`
  is configured.
- `feature_manifest.json`: generated modalities and chemistry warnings.
- `features/`: ReactionT5v2, UniMol2, ChIRo, and train-schema chemistry inputs.
- `logs/`: one log per feature extractor.

`cosine_similarity` is a retrieval score used for ordering candidates. It is
not a calibrated probability that an enzyme catalyzes the reaction. Candidate
selection still requires sequence review, expression checks, and experimental
validation.

The encoded enzyme pool is checkpoint-specific. `target_cache_dir` reuses the
existing pool when its checkpoint, model configuration, candidate IDs, and
residue store all match. A missing cache is generated once and reused by later
queries.

The example uses the repository model environment for neural inference and
`feature_generation.chemistry_python` for the train-schema chemistry transform.
This mirrors the environments used to train F4.

## All F4 Versions

Run the same reaction through the ReactZyme `time`, `enzyme_smi`,
`reaction_smi`, and Horizyn in-domain F4 checkpoints:

```bash
CUDA_VISIBLE_DEVICES=1 ../env/bin/python -m wet_lab.query_campaign \
  --config wet_lab/configs/all_f4.yaml
```

The campaign retains every model-specific top-k output and writes
`per_model_rankings.csv`, `consensus.csv`, `results.json`, and `summary.md`.
Consensus is reciprocal-rank fusion over each model's top 100. Raw cosine
scores are not averaged because their scales are checkpoint-specific.

## Fold Top Candidates

The folding workflow reads either a campaign consensus or a single-model
`results.json`, resolves exact sequences from the candidate FASTA, and writes
the selected FASTA plus one AlphaFold 3 monomer JSON job per protein.

Run the local ESMFold backend:

```bash
CUDA_VISIBLE_DEVICES=1 ../env/bin/python -m wet_lab.fold \
  --config wet_lab/configs/fold_all_f4_top10_esmfold.yaml
```

Structures, pLDDT/pTM metrics, `status.csv`, and a provenance-complete
`manifest.json` are written under `wet_lab/structures/`. Existing structures
are reused unless `--force` is passed. ESMFold sequences over the configured
length limit are reported and skipped; they are never silently truncated.
Selenocysteine positions are retained as `CCD_SEC` modifications in AF3 jobs.
Because ESMFold has no selenocysteine token, its backend maps `U` to the
unknown token `X` and records every affected position.

Every run also creates exact AF3 inputs under `inputs/alphafold3/`. To execute
them with the official AlphaFold 3 code, edit
`wet_lab/configs/fold_all_f4_top10_af3.yaml`: set `prepare_only: false` and
configure the AF3 Python, `run_alphafold.py`, licensed model parameters, and
database paths. Then run:

```bash
../env/bin/python -m wet_lab.fold \
  --config wet_lab/configs/fold_all_f4_top10_af3.yaml
```

## Search Complete Swiss-Prot

The Swiss-Prot workflow downloads all canonical reviewed UniProtKB entries from
the official current-release FTP endpoint, records the exact release and file
hashes, normalizes FASTA headers to UniProt accessions, extracts residue-level
ProT5 embeddings, and queries all four F4 checkpoints:

```bash
GPUS=1,2,3 scripts/run_wet_lab_swissprot_all_f4.sh
```

The database and large embedding store remain under
`wet_lab/databases/swissprot/current/`; model caches and runtime state remain
under `wet_lab/`, not the home directory. The F4 checkpoints require ProT5
residue embeddings, so the official UniProt ESM embeddings are not
interchangeable. Sequences longer than 1,024 residues use the same deterministic
`ends_center` truncation policy as F4 training and evaluation.

The full-database query locally buckets proteins by effective sequence length
within bounded storage windows. This preserves the candidate set and exact
retrieval scores while avoiding nearly all padding work during enzyme encoding.
It collates the stored float16 ProT5 values without widening each protein on the
CPU, then widens the complete batch to the model's parameter dtype during the
GPU transfer; this is numerically equivalent to widening before transfer.
The merged HDF5 uses virtual datasets backed by the immutable files in
`proteins_prott5_residue_shards/`; do not move or delete those shards.

On a node with a local mirror, set
`WET_LAB_RESIDUE_EMBEDDINGS_OVERRIDE=/path/to/proteins_prott5_residue.h5`
to use it without changing the portable query configuration. Target caches
record the resolved store identity, so local and shared-store caches cannot be
mixed accidentally.

## Search NCBI RefSeq Prokaryotic Proteins

The NCBI workflow uses release-pinned, non-redundant `WP_` proteins from the
RefSeq bacterial and archaeal divisions. It discovers the current release from
the official NCBI catalog, verifies every downloaded shard against the catalog
MD5, retains versioned accessions, writes normalized query inputs, extracts
ProtT5 residue embeddings, and runs the reaction-similarity F4 checkpoint:

```bash
REFSEQ_GPUS=1,2,3 scripts/run_wet_lab_refseq_reaction_smi.sh
```

The launcher deliberately defaults to a plumbing pilot: one ordered RefSeq FTP
shard from each division and a minimum protein length of 50 residues. NCBI
release shards are ordered for distribution, not selected as a biologically
representative sample, so pilot rankings must not be interpreted as a complete
RefSeq search. Increase the bounded collection in a separate database directory
after validating disk and GPU requirements:

```bash
REFSEQ_DATABASE_DIR=wet_lab/databases/refseq/prokaryotes/five_shards \
REFSEQ_MAX_FILES_PER_DIVISION=5 \
  REFSEQ_GPUS=1,2,3 \
  scripts/run_wet_lab_refseq_reaction_smi.sh
```

`REFSEQ_ALL_FILES=true` opts into every bacterial and archaeal WP shard. This
scope is extremely large and should only be used after capacity planning. The
database is stored under `wet_lab/databases/refseq/prokaryotes/current/`; its
`manifest.json` records the release, selected divisions and shards, filters,
checksums, normalized file hashes, and sequence statistics. A changed release
or selection never silently replaces an existing database: use a new database
directory, or set `REFSEQ_FORCE_NORMALIZE=true` intentionally.
The launcher passes a candidate-pool root override to the query engine, so a
custom `REFSEQ_DATABASE_DIR` does not require a second query configuration.

Edit `wet_lab/configs/refseq/f4_reactzyme_reaction_smi.yaml` to change the
reaction. RefSeq metadata (`description`, `organism`, division, and source
shard) and exact sequences are included in the resulting top-k CSV and FASTA.
