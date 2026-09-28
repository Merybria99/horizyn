"""Runtime setup shared by CIRCE training entry points."""

import lightning.pytorch as pl
import torch


def _performance_ddp_strategy(strategy, *, gradient_as_bucket_view=False):
    if not gradient_as_bucket_view or strategy == "auto":
        return strategy
    options = {
        "ddp": False,
        "ddp_find_unused_parameters_true": True,
        "ddp_find_unused_parameters_false": False,
    }
    if strategy not in options:
        raise ValueError("ddp_gradient_as_bucket_view requires a standard ddp strategy")
    return pl.strategies.DDPStrategy(
        find_unused_parameters=options[strategy], gradient_as_bucket_view=True,
    )


def _configure_cpu_threads(num_threads):
    if num_threads is None:
        return
    torch.set_num_threads(num_threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=num_threads)
