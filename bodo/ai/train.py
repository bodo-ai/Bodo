from __future__ import annotations

import importlib.util
from typing import Any, Callable, Literal

import bodo.spawn.spawner
from bodo.pandas import BodoDataFrame, BodoSeries


def torch_import_guard():
    try:
        importlib.util.find_spec("torch")
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it to use TorchTrainer."
        )


def _init_process_group():
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)

    dist.init_process_group(backend=backend)


def torch_train(
    train_loop_per_worker: Callable[[], None] | Callable[[dict], None],
    dataset: BodoDataFrame | BodoSeries,
    train_loop_config: dict | None = None,
):
    def worker_func():
        pass

    bodo.spawn.spawner.submit_func_to_workers(worker_func, [])


def prepare_model(
    model,
    parallel_strategy: Literal["ddp", "fsdp"] | None = None,
    parallel_strategy_kwargs: dict[str, Any] | None = None,
):
    torch_import_guard()
    import torch

    assert isinstance(model, torch.nn.Module), (
        "Model should be an instance of torch.nn.Module"
    )
    _init_process_group()

    if parallel_strategy is not None:
        assert parallel_strategy in ["ddp", "fsdp"], (
            "parallel_strategy should be either 'ddp' or 'fsdp'"
        )
        if parallel_strategy_kwargs is None:
            parallel_strategy_kwargs = {}
        if parallel_strategy == "ddp":
            from torch.nn.parallel import DistributedDataParallel as DDP

            model = DDP(model, **parallel_strategy_kwargs)
        elif parallel_strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            model = FSDP(model, **parallel_strategy_kwargs)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model
