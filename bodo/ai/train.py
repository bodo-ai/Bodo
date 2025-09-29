from __future__ import annotations

import importlib.util
import socket
import typing
from typing import Any, Callable, Literal

from bodo.mpi4py import MPI

if typing.TYPE_CHECKING:
    from bodo.pandas import BodoDataFrame, BodoSeries


def torch_import_guard():
    try:
        importlib.util.find_spec("torch")
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Please install it to use TorchTrainer."
        )


def _get_open_port():
    """
    Finds and returns an available open TCP port.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("", 0))  # Bind to an ephemeral port (port 0)
        port = s.getsockname()[1]  # Get the assigned port number
        return port
    finally:
        s.close()


def _init_process_group():
    import torch
    import torch.distributed as dist

    from bodo import get_gpu_ranks

    if dist.is_initialized():
        dist.destroy_process_group()

    device = torch.accelerator.current_accelerator(check_available=True)
    pytorch_rank = MPI.COMM_WORLD.Get_rank()
    npes = len(get_gpu_ranks()) if device != "cpu" else MPI.COMM_WORLD.Get_size()
    if device is None:
        device = "cpu"
    else:
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        gpu_ranks = get_gpu_ranks()
        if mpi_rank in gpu_ranks:
            pytorch_rank = gpu_ranks.index(mpi_rank)
        else:
            pytorch_rank = None

    print(
        f"MPI Rank: {MPI.COMM_WORLD.Get_rank()}, PyTorch Rank: {pytorch_rank}, Device: {device}"
    )
    backend = torch.distributed.get_default_backend_for_device(device)
    tcp_conn_str = None
    if pytorch_rank == 0:
        port = _get_open_port()
        tcp_conn_str = f"tcp://{socket.gethostname()}:{port}"
    tcp_conn_str = MPI.COMM_WORLD.bcast(tcp_conn_str, root=0)

    if pytorch_rank is not None:
        dist.init_process_group(
            backend=backend,
            init_method=tcp_conn_str,
            rank=pytorch_rank,
            world_size=npes,
        )
    return pytorch_rank, npes, device


def torch_train(
    train_loop_per_worker: Callable[[], None] | Callable[[dict], None],
    dataset: BodoDataFrame | BodoSeries,
    train_loop_config: dict | None = None,
):
    from bodo.spawn.spawner import submit_func_to_workers

    def worker_func(data):
        train_loop_per_worker(
            data, train_loop_config
        ) if train_loop_config else train_loop_per_worker(data)

    submit_func_to_workers(worker_func, [], dataset)
    # worker_func(dataset)


def prepare_model(
    model,
    parallel_strategy: Literal["ddp", "fsdp"] | None = "ddp",
    parallel_strategy_kwargs: dict[str, Any] | None = None,
):
    torch_import_guard()
    import torch

    assert isinstance(model, torch.nn.Module), (
        "Model should be an instance of torch.nn.Module"
    )
    pytorch_rank, pytorch_world_size, device = _init_process_group()
    if pytorch_rank is None:
        return None

    model = model.to(device)

    # No need to wrap the model if only one process is used
    if pytorch_world_size == 1:
        return model

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
        model.to(pytorch_rank)
    return model
