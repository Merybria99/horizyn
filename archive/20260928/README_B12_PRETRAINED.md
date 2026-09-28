# B1/B2 with pretrained molecular Graphormer

The September 9, 2026 replacement uses the PCQM4Mv2-pretrained Microsoft
Graphormer through the Hugging Face conversion. The scratch campaign is
preserved at `runs/b12_reaction_smi_train_20260908_230718/`; both original runs
completed 30 epochs and their final test evaluations.

## Current campaign

`runs/b12_pretrained_reaction_smi_train_20260909_012722/`, launched at
2026-09-09 01:27:22 UTC. B1 uses GPUs 0-1 (supervisor 3410781); B2 uses GPUs
2-3 (supervisor 3410782). See `launches.json` for the authoritative process
record and `monitor/status.md` for live progress. The monitor runs every 30
seconds and records failures or stalls without restarting jobs.

The user requested a longer run on September 9. The continuation is prepared
under `runs/b12_pretrained_reaction_smi_train_20260909_012722/extension_to_60/`.
Its controller waits for each original supervisor to finish, then resumes
that variant from its final `last.pt` on the same two GPUs. The new maximum
is **60 total epochs**, with early-stopping patience **10** instead of 5.
Model weights, optimizer state, per-rank RNG states, and validation selection
history are restored. All other training and evaluation settings are unchanged.
`extension_to_60/status.json` records the handoff state; `launches.json` is
updated when each continuation starts. The original segment's checkpoints
and final test results are archived under `extension_to_60/b*/before_resume/`.

Verification completed: 35 targeted tests, both two-GPU warmup/unfreezing
checks, all 7,726 released reaction inputs / 9,514 molecular components with
zero drops (maximum 413 atoms), and two real full-batch optimizer steps for
each variant with Graphormer unfrozen. Those smoke steps took approximately
2.4 seconds after the first step and used about 7.24 GiB peak allocated GPU
memory on rank 0. They do not measure all future training batches. The input,
weight and smoke audits are copied into this campaign's `preflight/` directory.

Both full runs start independently from initialization SHA-256
`0dbd9c559e6941e0f8157a39fe395543210f842e7553f78f85ecd94472b49738`;
smoke checkpoints are not loaded. The old and new campaigns' CIRCE evaluator
function fingerprints match exactly.

## Backbone and initialization

- Repository: `clefourrier/graphormer-base-pcqm4mv2`.
- Revision: `662d78215c8c8b2313837765b4b7f0d98b594a7e`.
- Local directory: `.deps/graphormer-base-pcqm4mv2/662d78215c8c8b2313837765b4b7f0d98b594a7e/`.
- Weights SHA-256: `fc92119f232e1918df3d342a51f4523d6354c9b7f1d4819e2915cc66e0ad73cc`.
- Architecture: 12 layers, width 768, 32 heads, feed-forward width 768.
- All 202 graph-encoder tensors (47,084,608 parameters) load bitwise from the
  checkpoint. The complete property model loads strictly before removing its
  six property-head tensors. Missing or altered files cause an error.
- Atom states exclude the graph token. The reaction pool projects width 768
  into its existing width 256. The pooling stacks contain 4,327,424 new
  parameters; the whole retrieval model contains 51,412,032 parameters.
- Both B1/B2 start as new runs from the same seed and initializer. Scratch
  weights and optimizer state cannot initialize the replacement.

This initializer learned molecular HOMO-LUMO property prediction without
enzyme-reaction pair supervision. Exact overlap between the molecular
pretraining corpus and ReactZyme compounds has not been measured; these
experiments cannot establish that evaluation compounds were unseen during
foundation pretraining.

## Molecular preprocessing

`horizyn/pretrained_graphormer.py` uses OGB 1.3.6 features and Microsoft's
original preprocessing and collation at commit
`a04573c40705fb174db261bb746a8258d00992f5`. The reference source and hashes live
under `.deps/graphormer_upstream/`.

Atom IDs receive the original field offsets and a single padding shift. The
Transformers 4.48.1 convenience preprocessor adds extra node shifts, so it is
not used. Shortest paths follow the original Floyd-Warshall tie handling. Edge
paths are capped to five hops during generation, matching the first five hops
of the reference tensor while avoiding oversized intermediate arrays. One
zero-filled edge hop supports isolated atoms when batched alone. Components
over 512 atoms fail explicitly; no participants or atoms are silently removed.
Stereo, multiplicity and unknown participant roles are preserved.

`/tmp/enzymediscovery_b12_graphormer_graphs/` caches label-free graph features,
keyed by preprocessing version, RDKit version, hop limit and canonical molecular
identity. It contains no learned representations. Frozen ESM-C features reuse
the existing residue cache.

## Training and evaluation

The first two epochs train pooling while Graphormer weights and dropout stay
frozen. At epoch index 2, all Graphormer parameters are unfrozen and DDP is
rebuilt to register their gradients. AdamW uses `1e-5` for Graphormer, `1e-4`
for pooling and weight decay `0.01`. ESM-C stays frozen. Molecular chunk size
is 16; pooling/token dimensions are otherwise preserved.

Both jobs use two GPUs, 2,000 global pair rows, seed 42, FP32, the same typed
negative sampler and fixed-beta CIRCE loss. The original segment has maximum
30 epochs and patience 5; the requested continuation has maximum 60 total
epochs and patience 10.
B1 uses global cosine; B2 retains the fixed 50/50 global/local score. Validation
and test still use `circe_v2_shared_v1`, including the original candidate order,
ground truth, tie handling and validation checkpoint selector. Each selected
checkpoint is tested after each completed training segment. The user also
requested an interim test at 24 completed epochs; fixed checkpoint snapshots
and its results are preserved in `interim_test_20260909_040213/` within the
current campaign.

The replacement increases capacity and adds a warmup/lower backbone learning
rate, so comparison with the scratch runs does not isolate pretraining alone.
B1 versus B2 retains matched settings.

## Runtime and verification

Use `../.b12-pretrained-py/bin/python`. It layers the packages in
`requirements-b12-pretrained.txt` over the existing B1/B2 runtime without
modifying it: Torch 2.4.0, Transformers 4.48.1 and ESM 3.2.1 remain in use.
Cython 3 supports Python 3.12; the path routine uses NumPy int64 casts.

`runs/b12_pretrained_preflight/` records strict loading, original Microsoft
feature/padding parity, isolated atoms, stereo, cache roundtrip, initialization
equality, frozen/unfrozen optimizer behavior, legacy checkpoint rejection, and
the existing scoring/loss/CIRCE evaluation tests. The input audit checks every
released molecular component without computing retrieval test predictions.
The separate two-GPU smoke checks both training stages, finite gradients and
exact parameter synchronization for B1 and B2.

## Launch and resume

From the Horizyn directory:

```bash
../.b12-pretrained-py/bin/python scripts/launch_token_retrieval_pair.py \
  --preflight-dir runs/b12_pretrained_preflight
```

This launches B1 on GPUs 0-1 and B2 on GPUs 2-3, recording source snapshots,
hashes, configs, installed versions and exact commands. A full-batch smoke uses
`--smoke-steps 2 --graph-warmup-epochs 0` to exercise Graphormer gradients
immediately. Smoke weights must not initialize full training.

For a completed-epoch resume, use the run's saved `launch_config.yaml` and
`checkpoints/last.pt` with `scripts/train_token_retrieval.py --resume`. The
architecture and warmup schedule must match. Preserved scratch configs remain
loadable through the explicit `venus_scratch` backend.

Sources: [checkpoint](https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2),
[Microsoft Graphormer](https://github.com/microsoft/Graphormer),
[PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/).
