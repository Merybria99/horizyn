"""Compatibility imports; maintained code lives in horizyn.pipelines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horizyn.pipelines.training import (main, build_arg_parser, _config_first,
    _configure_cpu_threads, _performance_ddp_strategy, _load_biofp_pretrain_warm_start,
    _load_partial_model_warm_start, _source_replay_kwargs, _print_training_config)

if __name__ == "__main__":
    main()
