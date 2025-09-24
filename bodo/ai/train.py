from __future__ import annotations

import importlib.util
import socket
import typing
from typing import Any, Callable, Literal

import bodo
import bodo.spawn.spawner
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

    if dist.is_initialized():
        dist.destroy_process_group()

    if device := torch.accelerator.current_accelerator(check_available=True) is None:
        device = "cpu"
    backend = torch.distributed.get_default_backend_for_device(device)
    tcp_conn_str = None
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        port = _get_open_port()
        tcp_conn_str = f"tcp://{socket.gethostname()}:{port}"
    tcp_conn_str = MPI.COMM_WORLD.bcast(tcp_conn_str, root=0)
    npes = MPI.COMM_WORLD.Get_size()

    dist.init_process_group(
        backend=backend, init_method=tcp_conn_str, rank=rank, world_size=npes
    )


def torch_train(
    train_loop_per_worker: Callable[[], None] | Callable[[dict], None],
    dataset: BodoDataFrame | BodoSeries,
    train_loop_config: dict | None = None,
):
    def worker_func():
        train_loop_per_worker(
            train_loop_config
        ) if train_loop_config else train_loop_per_worker()

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
    if (
        device := torch.accelerator.current_accelerator(check_available=True)
        is not None
    ):
        model.to(device)
    return model
