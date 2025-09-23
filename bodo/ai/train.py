from __future__ import annotations

import importlib.util
from typing import Any, Callable, Literal

from bodo.pandas import BodoDataFrame, BodoSeries


def torch_import_guard():
    try:
        importlib.util.find_spec("torch")
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it to use TorchTrainer."
        )


class TorchTrainer:
    def __init__(
        self,
        train_loop_per_worker: Callable[[], None] | Callable[[dict], None],
        dataset: BodoDataFrame | BodoSeries,
        train_loop_config: dict | None = None,
    ):
        pass


def prepare_model(
    model,
    move_to_device=True,
    parallel_strategy: Literal["ddp", "fsdp"] | None = None,
    parallel_strategy_kwargs: dict[str, Any] | None = None,
):
    torch_import_guard()
    import torch

    assert isinstance(model, torch.nn.Module), (
        "Model should be an instance of torch.nn.Module"
    )
    assert move_to_device in [True, False] or isinstance(
        move_to_device, torch.device
    ), "move_to_device should be a boolean"

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
    if move_to_device is True:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device)
    elif isinstance(move_to_device, torch.device):
        model.to(move_to_device)
    return model
