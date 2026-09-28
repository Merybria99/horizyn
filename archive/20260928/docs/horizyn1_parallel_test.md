# Four-GPU Tyrosine-checkpoint test

Run on the machine with the GPUs, with the project and shared feature files mounted:

```bash
cd /datastor2/deep-proteins/EnzymeDiscovery/horizyn
env -u BASH_ENV -u ENV bash --noprofile --norc scripts/launch_horizyn1_test_4gpu.sh
```

The default is GPUs 0,1,2,3; `CIRCE_TEST_GPUS` can select other numeric indices.
The detached session is `horizyn_test_best16k_4gpu_v2`. Output and logs are under
`runs/horizyn1_tyrosine_test_best16k_4gpu_v2/`. The earlier failed launch directory
is preserved; its manifest is not rewritten to disguise the code change.

The controller verifies input provenance and GPU occupancy, then stops only a
same-user Python process actually executing `evaluate_horizyn1_circe_v2.py`, with the exact old output
`runs/horizyn1_tyrosine_test_best16k/test.json` and selected checkpoint. It refuses
to launch over other GPU jobs or a surviving parallel controller. Python's script
operand and `/proc/PID/exe` must match; mentioning the command inside tmux, flock,
bash, `python -c`, or another script is not sufficient. The controller and its
ancestors are protected; process trees are not expanded. Identity is rechecked
before every signal. Existing logs, features, and checkpoints are not
deleted. Incomplete *in-memory* encoding from the single-GPU evaluator is lost;
its cache is only committed after the entire protein catalog has been encoded.

## Work and restart behavior

1. Four independent encoders use the existing `encode_targets` and
   `encode_reactions` functions. Protein chunks contain 8,192 consecutive catalog
   rows, aligned to the unchanged 64-protein FP32 batches. Each successful chunk
   is atomically committed; an interrupted chunk alone needs recomputation.
2. After encoders exit, the controller streams a CPU merge into a memory-mapped
   catalog. No GPU collective or distributed-worker timeout surrounds the merge.
3. Four independent scorers partition **queries**, never the candidate universe.
   Each searches the full appropriate catalog using the existing exact ranker,
   stable candidate-row tie rule, 32-query batches, and 32,768-candidate chunks.
   Protein embeddings are held on each scoring GPU to avoid repeated host-to-GPU
   transfers. This requires at least 20,000 MiB of free GPU memory; H200s fit.
4. Results are combined using query-count weighting, including uneven tails.
   Both first-positive and all-positive MRR retain the serial definitions.

R2E uses all 6,149,161 protein candidates; E2R uses the 3,002 held-out reaction
directions. The 16,000-step validation-selected checkpoint, full test gold,
protein overlap policy, and source-collapse interpretation are unchanged.

The controller holds a shared-storage lock, rejects changed inputs/code, stops
its own workers on a worker failure, and preserves committed chunks. Once the
old session has exited, rerun the same launcher to resume. All workers must use
the same shared code version. No four-GPU speedup is guaranteed: NFS throughput
and contention can limit scaling. Use the per-GPU logs in `parallel_cache/` to
measure actual progress; the controller logs committed chunk counts every 30 s.

Verification covers serial/parallel metric equivalence on CPU (ties and uneven
tails), cache validation/restart, missing chunks, GPU occupancy guards, precise
legacy process targeting, and child failure handling. An actual-checkpoint CPU
smoke check produced bit-identical serial/chunked protein embeddings for eight
short cached samples. Four-GPU throughput and equivalence still require checking
on the launch machine; GPU batches and FP32 precision were deliberately retained.
