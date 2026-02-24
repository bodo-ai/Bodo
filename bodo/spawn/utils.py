"""Utilities for Spawn Mode.
This file should import JIT lazily to avoid slowing down non-JIT code paths.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import traceback
import typing as pt
import uuid
from collections.abc import Callable
from enum import Enum
from time import sleep

import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from pandas.core.arrays.arrow import ArrowExtensionArray

import bodo.user_logging


class CommandType(str, Enum):
    """
    Enum of the different types of commands that the spawner
    can send to the workers.
    """

    EXEC_FUNCTION = "exec"
    EXIT = "exit"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    GATHER = "gather"
    DELETE_RESULT = "delete_result"
    REGISTER_TYPE = "register_type"
    SET_CONFIG = "set_config"
    SPAWN_PROCESS = "spawn_process"
    STOP_PROCESS = "stop_process"
    SCATTER_JIT = "scatter_jit"
    GATHER_JIT = "gather_jit"


def poll_for_barrier(comm: MPI.Comm, poll_freq: float | None = 0.1):
    """
    Barrier that doesn't busy-wait, but instead polls on a defined interval.
    The poll_freq kwarg controls the rate of polling. When set to None it will
    busy-wait.
    """
    # Start a non-blocking barrier operation
    req = comm.Ibarrier()
    if not poll_freq:
        # If polling is disabled, just wait for the barrier synchronously
        req.Wait()
    else:
        # Check if the barrier has completed and sleep if not.
        # TODO Add exponential backoff (e.g. start with 0.01 and go up
        # to 0.1). This could provide a faster response in many cases.
        while not req.Test():
            sleep(poll_freq)


def debug_msg(logger: logging.Logger, msg: str):
    """Send debug message to logger if Bodo verbose level 2 is enabled"""
    if bodo.user_logging.get_verbose_level() >= 2:
        logger.debug(msg)


class ArgMetadata(str, Enum):
    """Argument metadata to inform workers about other arguments to receive separately.
    E.g. broadcast or scatter a dataframe from spawner to workers.
    Used for DataFrame/Series/Index/array arguments.
    """

    BROADCAST = "broadcast"
    SCATTER = "scatter"
    LAZY = "lazy"


def set_global_config(config_name: str, config_value: pt.Any):
    """Set global configuration value by name (for internal testing use only)
    (e.g. "bodo.hiframes.boxing._use_dict_str_type")
    """
    # Get module and attribute sections of config_name
    # (e.g. "bodo.hiframes.boxing._use_dict_str_type" -> "bodo.hiframes.boxing"
    # and "_use_dict_str_type")
    c_split = config_name.split(".")
    attr = c_split[-1]
    mod_name = ".".join(c_split[:-1])
    locs = {}
    exec(f"import {mod_name}; mod = {mod_name}", globals(), locs)
    mod = locs["mod"]
    setattr(mod, attr, config_value)


class WorkerProcess:
    _uuid: uuid.UUID
    _rank_to_pid: dict[int, int] = {}

    def __init__(self, rank_to_pid: dict[int, int] = {}):
        """Initialize WorkerProcess with a mapping of ranks to PIDs."""
        self._uuid = uuid.uuid4()
        self._rank_to_pid = rank_to_pid


def is_jupyter_on_windows() -> bool:
    """Returns True if running in Jupyter on Windows"""

    # Flag for testing purposes
    if os.environ.get("BODO_OUTPUT_REDIRECT_TEST", "0") == "1":
        return True

    return sys.platform == "win32" and (
        "JPY_SESSION_NAME" in os.environ
        or "PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING" in os.environ
    )


def is_jupyter_on_bodo_platform() -> bool:
    """Returns True if running in Jupyter on Bodo Platform"""

    platform_cloud_provider = os.environ.get("BODO_PLATFORM_CLOUD_PROVIDER", None)
    return (platform_cloud_provider is not None) and (
        "JPY_SESSION_NAME" in os.environ
        or "PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING" in os.environ
    )


def sync_and_reraise_error(
    err,
    _is_parallel=False,
    bcast_lowest_err: bool = True,
    default_generic_err_msg: str | None = None,
):  # pragma: no cover
    """
    If `err` is an Exception on any rank, raise an error on all ranks.
    If 'bcast_lowest_err' is True, we will broadcast the error from the
    "lowest" rank that has an error and raise it on all the ranks without
    their own error. If 'bcast_lowest_err' is False, we will raise a
    generic error on ranks without their own error. This is useful in
    cases where the error could be something that's not safe to broadcast
    (e.g. not pickle-able).
    This is a no-op if all ranks are exception-free.

    Args:
        err (Exception or None): Could be None or an exception
        _is_parallel (bool): Whether this is being called from many ranks
        bcast_lowest_err (bool): Whether to broadcast the error from the
            lowest rank. Only applicable in the _is_parallel case.
        default_generic_err_msg (str, optional): If bcast_lowest_err = False,
            this message will be used for the exception raised on
            ranks without their own error. Only applicable in the
            _is_parallel case.
    """
    comm = MPI.COMM_WORLD

    if _is_parallel:
        # If any rank raises an exception, re-raise that error on all non-failing
        # ranks to prevent deadlock on future MPI collective ops.
        # We use allreduce with MPI.MAXLOC to communicate the rank of the lowest
        # failing process, then broadcast the error backtrace across all ranks.
        err_on_this_rank = int(err is not None)
        err_on_any_rank, failing_rank = comm.allreduce(
            (err_on_this_rank, comm.Get_rank()), op=MPI.MAXLOC
        )
        if err_on_any_rank:
            if comm.Get_rank() == failing_rank:
                lowest_err = err
            else:
                lowest_err = None
            if bcast_lowest_err:
                lowest_err = comm.bcast(lowest_err, root=failing_rank)
            else:
                err_msg = (
                    default_generic_err_msg
                    if (default_generic_err_msg is not None)
                    else "Exception on some ranks. See other ranks for error."
                )
                lowest_err = Exception(err_msg)

            # Each rank that already has an error will re-raise their own error, and
            # any rank that doesn't have an error will re-raise the lowest rank's error.
            if err_on_this_rank:
                raise err
            else:
                raise lowest_err
    else:
        if err is not None:
            raise err


def import_compiler_on_workers():
    """Import the JIT compiler on all workers. Done as necessary since import time
    can be significant.
    """
    spawner = bodo.spawn.spawner.get_spawner()
    spawner.import_compiler_on_workers()


def import_bodosql_compiler_on_workers():
    """Import the BodoSQL JIT compiler extensions on all workers.
    Done as necessary since import time can be significant.
    """
    spawner = bodo.spawn.spawner.get_spawner()
    spawner.import_bodosql_compiler_on_workers()


def gatherv_nojit(data, root, comm):
    """A no-JIT version of gatherv for use in spawn mode. This avoids importing the JIT
    compiler which can be slow.
    """
    import bodo
    from bodo.ext import hdist
    from bodo.pandas.utils import (
        BODO_NONE_DUMMY,
        cpp_table_to_df,
        df_to_cpp_table,
    )

    original_data = data

    if isinstance(data, pd.arrays.DatetimeArray):
        arr = pa.array(data)
        data = pd.array(arr, dtype=pd.ArrowDtype(arr.type))

    # Get data type on receiver since it doesn't have any local data
    rank = bodo.get_rank()

    if comm is not None:
        # Receiver has to set root to MPI.ROOT in case of intercomm
        is_receiver = root == MPI.ROOT
        if is_receiver:
            data = comm.recv(source=0, tag=11)
        elif rank == 0:
            if data is not None:
                sample = (
                    data.head(0)
                    if isinstance(data, (pd.DataFrame, pd.Series))
                    else data[:0]
                )
            else:
                sample = None
            comm.send(
                sample,
                dest=0,
                tag=11,
            )

    if data is None:
        return None

    # Fallback to JIT version if unsupported type
    if (
        data is not None
        and not isinstance(data, ArrowExtensionArray)
        and not _is_supported_gather_scatter_type(data)
    ):
        # Import compiler lazily
        import bodo.decorators
        from bodo.libs.distributed_api import gatherv

        return gatherv(original_data, False, True, root, comm)

    is_series = isinstance(data, pd.Series)
    is_array = isinstance(data, ArrowExtensionArray)

    if is_series:
        # None name doesn't round-trip to dataframe correctly so we use a dummy name
        # that is replaced with None in wrap_plan
        name = BODO_NONE_DUMMY if data.name is None else data.name
        data = data.to_frame(name=name)

    if is_array:
        data = pd.DataFrame({"__arrow_data__": data})

    comm_ptr = 0 if comm is None else MPI._addressof(comm)
    cpp_table_ptr, in_schema = df_to_cpp_table(data)
    out_ptr = hdist.gatherv_py_wrapper(cpp_table_ptr, root, comm_ptr)
    out = cpp_table_to_df(out_ptr, in_schema)

    if is_series:
        out = out.iloc[:, 0]
        # Reset name to None if it was originally None
        if out.name == BODO_NONE_DUMMY:
            out.name = None

    if is_array:
        out = out.iloc[:, 0].array

    return out


def _is_supported_gather_scatter_type(data) -> bool:
    """Make sure data is a DataFrame or Series with types supported in gather/scatter (without categorical dtypes currently)."""

    if isinstance(data, pd.DataFrame):
        for dtype in data.dtypes:
            if isinstance(dtype, pd.CategoricalDtype):
                return False

    if isinstance(data, pd.Series):
        if isinstance(data.dtype, pd.CategoricalDtype):
            return False

    return isinstance(data, (pd.DataFrame, pd.Series)) and not isinstance(
        data.index, (pd.CategoricalIndex, pd.PeriodIndex, pd.IntervalIndex)
    )


def scatterv_nojit(data, root, comm):
    """A no-JIT version of scatterv for use in spawn mode. This avoids importing the JIT
    compiler which can be slow.
    """
    from bodo.ext import hdist
    from bodo.pandas.utils import (
        BODO_NONE_DUMMY,
        cpp_table_to_df,
        df_to_cpp_table,
    )

    is_sender = root == MPI.ROOT

    sample_data = comm.bcast(_get_data_sample(data) if is_sender else None, root)
    original_data = data
    data = sample_data if not is_sender else data

    # Fallback to JIT version if unsupported type
    if not _is_supported_gather_scatter_type(data):
        # Import compiler lazily
        from bodo.libs.distributed_api import scatterv

        return scatterv(original_data, None, True, root, comm)

    is_series = isinstance(data, pd.Series)

    if is_series:
        # None name doesn't round-trip to dataframe correctly so we use a dummy name
        # that is replaced with None in wrap_plan
        name = BODO_NONE_DUMMY if data.name is None else data.name
        data = data.to_frame(name=name)

    comm_ptr = MPI._addressof(comm)
    cpp_table_ptr, in_schema = df_to_cpp_table(data)
    out_ptr = hdist.scatterv_py_wrapper(cpp_table_ptr, root, comm_ptr)
    out = cpp_table_to_df(out_ptr, in_schema)

    if is_series:
        out = out.iloc[:, 0]
        # Reset name to None if it was originally None
        if out.name == BODO_NONE_DUMMY:
            out.name = None

    return out


def _get_data_sample(data):
    """Get an empty sample of the data for sending to workers to determine
    data type and structure.
    Avoids head(0) for BodoDataFrame/BodoSeries since the serialized lazy block manager
    causes issues on the worker side.
    """
    from bodo.pandas.base import _empty_like
    from bodo.pandas.frame import BodoDataFrame
    from bodo.pandas.series import BodoSeries

    if isinstance(data, ArrowExtensionArray):
        return data[:0]

    if data is None:
        return None

    if isinstance(data, (BodoDataFrame, BodoSeries, pd.DataFrame, pd.Series)):
        # NOTE: handles object columns correctly using Arrow schema inference for Pandas
        return _empty_like(data)

    if isinstance(data, dict):
        return {k: _get_data_sample(v) for k, v in data.items()}

    try:
        return data[:0]
    except Exception:
        return data


def run_rank0(func: Callable, bcast_result: bool = True, result_default=None):
    """
    Utility function decorator to run a function on just rank 0
    but re-raise any Exceptions safely on all ranks.
    NOTE: 'func' must be a simple python function that doesn't require
    any synchronization.
    e.g. Using a bodo.jit function might be unsafe in this situation.
    Similarly, a function that uses any MPI collective
    operation would be unsafe and could result in a hang.

    Args:
        func: Function to run.
        bcast_result (bool, optional): Whether the function should be
            broadcasted to all ranks. Defaults to True.
        result_default (optional): Default for result. This is only
            useful in the bcase_result=False case. Defaults to None.
    """

    @functools.wraps(func)
    def inner(*args, **kwargs):
        comm = MPI.COMM_WORLD
        result = result_default
        err = None
        # Run on rank 0 and catch any exceptions.
        if comm.Get_rank() == 0:
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                print("".join(traceback.format_exception(None, e, e.__traceback__)))
                err = e
        # Synchronize and re-raise any exception on all ranks.
        err = comm.bcast(err)
        if isinstance(err, Exception):
            raise err
        # Broadcast the result to all ranks.
        if bcast_result:
            result = comm.bcast(result)
        return result  # type: ignore

    return inner
