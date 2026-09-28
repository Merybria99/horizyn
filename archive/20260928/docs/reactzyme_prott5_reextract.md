# Independent ReactZyme ProtT5 cache rebuild

This is a replacement feature cache, not a new split or a training job. It leaves
the stalled shared cache, all historical configurations, and the tyrosine
Horizyn training run unchanged. No deletion of an existing cache is implemented:
the old shared catalog also served non-ReactZyme evaluation consumers.

Run on **slurm-node-014**, within the appropriate GPU allocation:

```bash
bash /datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/launch_reactzyme_prott5_reextract.sh
tail -f /datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/reactzyme_prott5_reextract_20260910.log
```

The controller logs each stage. Detailed logs and status are inside
`runs/reactzyme_prott5_reextract_20260910/logs/` and `status.json`.
The detached session is `reactzyme_prott5_reextract`.

### If reading the launcher itself hangs

`scripts/launch_reactzyme_prott5_local_bootstrap.sh` contains a **pasteable shell
block**. Paste its contents directly into node014's terminal; executing the
file via its NFS path defeats the bypass. It creates tmux from `/`, starts
`/bin/bash` directly without login profiles or `BASH_ENV`, and writes its
bootstrap/controller console log to a unique `/tmp/reactzyme_bootstrap.*/`
directory. The old launcher file and NFS log redirection are not used.

The first dependency check is a timed read of the extraction controller. If
that read fails, it exits without starting extraction and leaves an actionable
local log. If it succeeds, the unchanged controller runs its existing storage,
runtime, GPU, and pilot gates. Model files, code, input data, and feature output
still depend on `/datastor2`; this does not repair NFS or move the dataset to
local storage. The printed `/tmp` log is on node014 and cannot be monitored
through the shared filesystem before the controller creates its stage logs.
An existing session is refused, never stopped or replaced.

## Gates and representation

1. A timed 8 MiB write/fsync/read-back test in the **new** directory. This does
   not prove sustained NFS performance; a system-wide storage issue can still
   affect fresh files. No scratch disk is assumed.
2. Exact union of the nine existing train/validation/test candidate lists:
   **178,327 protein IDs**. Original sequences come from the three official
   protocol FASTAs, avoiding the broken old shared-FASTA symlink. Conflicting
   sequences for the same ID and missing sequences are fatal. Source file
   signatures and prepared FASTA SHA256 hashes are checked on reuse.
3. Space for 110% of the calculated FP16 payload plus 15 GiB. This is shared
   filesystem free space, not a guarantee about an account quota.
4. A timed CPU-only ML-runtime import check, then idle H200 GPUs with at least
   125 GiB free each. Other jobs are never stopped.
5. A four-GPU pilot containing long sequences and shorter examples. Every pilot
   vector is read back and checked for finiteness; CIRCE's actual residue reader
   is exercised. A failure prevents the full extraction.
6. Full extraction and independent rank checkpoints, followed by a virtual HDF5
   merge (no second full-size copy). Keep the source shards permanently with it.
7. All IDs, lengths, shard completion markers, model settings, VDS source paths,
   and source-to-VDS sample equality are checked. The training reader accesses
   a deterministic sample including the shortest/longest entry of every rank.

Frozen weights: the local snapshot of
`Rostlab/prot_t5_xl_half_uniref50-enc`, revision
`94a6abc029ae13029317b140b7424e012bf8dfbf`. No network model downloads.
Stored representation: FP16, 1024 dimensions per residue, 1022-residue limit,
`ends_center` truncation, the existing extractor's amino-acid normalization.
Batch cap 256; padded token budget 65,536; length sorting enabled. The pilot
checks these caps on the actual available GPUs instead of promising they fit.

The extractor checks **every generated output** for finiteness before writing.
Full-cache validation samples stored vectors rather than reading the entire
large cache again; `--full-scan` additionally reads every stored vector. The
validation JSON distinguishes these checks explicitly.

Successful completion creates `READY.json` and `full_validation.json`. The new
path for future ReactZyme configs is:

```
/datastor2/deep-proteins/EnzymeDiscovery/horizyn/runs/reactzyme_prott5_reextract_20260910/features/proteins_prott5_residue.h5
```

No existing configuration is silently repointed. A later training campaign must
use this path explicitly and preserve its split-specific reaction features,
annotations, sampling policy, and evaluation candidates.

## Restart behavior

Rerun the launcher only after the previous session has exited. It reuses verified
preparation and committed extraction checkpoints. Changing the model, world
size, batch/token caps, or extractor code requires a **new run directory**.
Completed shards are checked by the final validator even though the existing
extractor skips them on resume. An interrupted merge rebuilds only this new
run's partial merge metadata; it never removes rank shards.

If a failed child remains blocked after targeted TERM/KILL, the controller
retains `.controller.lock` and prints its PIDs. Do not remove that lock or
relaunch until those processes are confirmed gone. HDF5 files interrupted by
SIGKILL may require recovery rather than an automatic restart; no corruption
repair or forced overwrite is attempted.

For CPU-only preparation (no GPU activity), use `/usr/bin/python3` with
`scripts/rebuild_reactzyme_prott5_cache.py prepare`. The GPU `run` action refuses
tyrosine by default.
