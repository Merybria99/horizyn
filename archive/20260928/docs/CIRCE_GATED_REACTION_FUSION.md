# Three-branch gated reaction fusion

This experiment uses cached UniMol2 molecule vectors, cached ChIRo molecule
vectors, and freshly extracted frozen ReactionT5 participant-string vectors.
No CGRs, fingerprints, reconstructed equations, or inferred reactant/product
assignments enter the tower.

Both variants use position-free molecule interaction pooling in the UniMol2
and ChIRo branches, separate adapters to 256 dimensions, per-token layer norm,
0.1 modality dropout, concatenation, and a 512-dimensional normalized retrieval
embedding. The `gated` variant adds one four-head modality-attention block and
per-feature sigmoid residual gates initialized to 0.1. Missing modality keys,
queries, gate outputs, and residuals are masked. The `concat` variant omits this
block. Common layers and the subsequent enzyme tower have matched same-seed
initialization; creating the extra module preserves the RNG state.

The enzyme architecture, loss, optimizer, learning rate, precision and biological
supervision settings are inherited from the reaction-smi CIRCE-v3 base config.
Trainable towers start fresh in both arms; pretrained molecular/protein features
and the frozen SLEEC scorer are reused. The default recipe has auxiliary
classification weight zero and unknown-negative weight 0.5. No existing jobs are
stopped, and selected GPUs must be free.

## Launch

On the machine with four free GPUs:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV bash --noprofile --norc \
  scripts/launch_circe_gated_fusion.sh \
  --gpus 0,1,2,3 --variant both --steps 864
```

This creates a fresh run root, starts detached in tmux, extracts ReactionT5
features once on the first selected GPU, then trains `concat` followed by `gated`
using all selected GPUs. Each arm runs 864 optimizer steps at global batch 1536;
the per-GPU batch is 384 on four GPUs. This is a short matched-budget experiment,
not a promise of 30 complete epochs. Both directions are evaluated on the same
validation candidate pool at the final matched-budget checkpoint. The test set
is not touched. Use `--variant gated` to run just that arm. `--gpus 0,1,3` uses
three GPUs while retaining the same global batch.

The launcher prints its unique run directory, tmux session and pipeline log.
Training progress is in `<run>/<variant>/training.log`; validation metrics are
in `<run>/<variant>/evaluation/metrics.json`. `complete.json` appears only after
all selected training and evaluation stages succeed. Existing/prepared run roots
are refused; this launcher does not silently resume or overwrite checkpoints.

## Feature semantics and monitoring

Participant strings are canonicalized and molecule-sorted for deterministic
ReactionT5 extraction. They are not duplicated into `X >> X`. The legacy `_f`
key suffix and dataset self-reaction normalization are retained solely for cache
ID compatibility; molecule-set pooling ignores the artificial product copy.
ReactionT5 sees actual unordered participant strings without reaction arrows.
This is an out-of-domain transfer representation, not a known transformation.
Its input cap is 512 tokens, so long collections can be truncated. UniMol2/ChIRo
branches retain their cached molecule-set inputs.

The experiment manifest pins configs, code and source-file identities. Partial
ReactionT5 outputs are not used until IDs, dimensions and finite values pass
checks. Both arms share that exact cache. Missing molecular branches are masked
without deleting queries or positive pairs.

Training logs include `reaction_fusion/gate_<modality>`, near-closed/open gate
fractions and update norms. They are diagnostics, not evidence of causal feature
importance. For concat fusion, legacy modality-weight statistics are explicitly
renamed availability statistics rather than presented as learned weights.

CPU tests cover permutation invariance, poisoned/missing inputs, all-missing
fusion, gradients, matched initialization, configuration validation, short
training, checkpoint reload and bidirectional evaluation. Full GPU/DDP training
and full feature extraction require the user's launch; no performance gain or
VRAM utilization is guaranteed by these tests.
