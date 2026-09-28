# Maintained configurations

`pipelines/v4/` is the main dictionary-free pipeline. Training profiles cover ReactZyme ReactionSim, EnzymeSim, Time, and EnzymeMap. `phase2.yaml` controls full-graph refinement; `frozen_*.json` identify the existing fitted models with authenticated residual heads.

`pipelines/f3/published_retrieval.yaml` retains the original F3 published-pool recipe. `pipelines/circev2/reactzyme_reaction_smi.yaml` retains the original CIRCEv2 annotation-negative recipe. Their training data and losses differ; these profiles are not interchangeable matched-data comparisons.

Training YAML paths resolve from the `horizyn` project directory. Relative artifact paths in model JSON resolve against that JSON file. Original frozen configs under `runs/` remain unchanged.

`benchmarks/enzyme_retrieval_unified.yaml` retains official metric/pool definitions for the phase-1 benchmark runner. Complete V4 evaluation uses final embeddings, including refinement, via `horizyn.pipelines evaluate`.

Obsolete sweeps and variants are preserved in `archive/20260928/configs/`. See [the pipeline guide](../docs/PIPELINES.md).
