# ReactZyme reaction-smi: classification 0.2, 50/50 pairs

Launch on Arginine (`slurm-node-014`), with four free H200 GPUs:

```bash
cd /
/usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C \
  /usr/bin/timeout --kill-after=2s 20s \
  /bin/bash --noprofile --norc \
  /datastor2/deep-proteins/EnzymeDiscovery/horizyn/scripts/launch_reactzyme_arginine_classification.sh
```

The launcher prints a local log path and starts tmux session
`reactzyme_arginine_cls02_5050`. It never stops another job or overwrites a run.
An initial exit code 124 indicates a timeout, not successful training; check
storage access before retrying. The existing protein cache is screened with
sampled reads, not re-extracted. Screening cannot guarantee future NFS health.

The run reuses the Glutamine reaction-smi recipe's official train/validation
split, train-only indexed graph and cached frozen backbone features. Retrieval
towers start fresh; the fixed SLEEC scorer still loads its pretrained weights.
No test-set metrics select the model or its batch size.

The loss is the existing bidirectional sampled multi-positive InfoNCE plus
`0.2 * (0.5 * mechanism_BCE + 0.5 * cofactor_BCE)`. BCE uses the existing
per-label masks and confidence weights, including legacy weak negative labels;
these are not experimentally verified negatives. Any protein whose source
reaction aggregate contains an edge absent from current training is excluded
from classification. Missing labels are masked, not all-negative examples.
There are 8 mechanism and 10 cofactor classes; labels are never query inputs.

Each rank's batch has exactly 50% genuine positive rows and 50% explicit
annotation-filtered negative rows. Negative endpoints share the positive rows'
features and retain positive support in both retrieval directions. Biological
and random negatives each request half the negative quota; an empty category
falls back to the other. Shared indexed negative masks remain enabled, so the
ratio describes sampled rows, **not all cells in the contrastive matrix**.
The original 85/15 sampler remains the default for existing runs.

Four 15-step pilots compare 400/800/1600/3200 pair rows per GPU. Five warm-up
steps are discarded. The launcher chooses the highest observed throughput
using the slowest rank, with a 15% margin on peak allocated CUDA memory.
Pilots reject non-finite training loss; selection also requires logged nonzero
classification loss and active annotations for both families on every rank.
Pilots do not validate or save checkpoints. CUDA OOM permits selecting a
smaller successful pilot only after the GPUs are clear; other failures stop.
This is a bounded calibration, not proof of a global optimum or full-epoch
memory safety. It can take longer than 60 total step times because each pilot
also initializes its model and data loaders. Larger batches change optimizer
update counts and can affect retrieval quality; the learning rate is unchanged.

Training then starts from fresh tower initialization for 30 epochs, with no
early stopping or step cap. It uses four-rank DDP, BF16 mixed precision, FP32
sensitive calculations, fused AdamW, the existing fast-I/O adapters, two workers
per rank, and prefetch 1 to limit shared-storage pressure. Validation runs every
epoch at batch 512; recovery checkpoints are saved every 50 optimizer steps.

The isolated run contains `preparation.json` (label coverage/provenance),
`batch_selection.json` (pilot comparisons), `train.yaml`, `pipeline.log`, and
`training/` (metrics and checkpoints). Benchmark weights are not resumed.
