# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Worker process to handle compiling and running python functions with
Bodo - note that this module should only be run with MPI.Spawn and not invoked
directly"""

import logging
import os
import sys
import typing as pt
import uuid

import cloudpickle
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from numba import typed
from pandas.core.arrays.arrow import ArrowExtensionArray

import bodo
from bodo.mpi4py import MPI
from bodo.submit.spawner import ArgMetadata, BodoSQLContextMetadata
from bodo.submit.utils import (
    CommandType,
    DistributedReturnMetadata,
    debug_msg,
    poll_for_barrier,
)
from bodo.submit.worker_state import set_is_worker

DISTRIBUTED_RETURN_HEAD_SIZE: int = 5


def _recv_arg(arg: pt.Any | ArgMetadata, spawner_intercomm: MPI.Intercomm):
    """Receive argument if it is a DataFrame/Series/Index/array value.

    Args:
        arg: argument value or metadata
        spawner_intercomm: spawner intercomm handle

    Returns:
        Any: received function argument
    """
    if isinstance(arg, ArgMetadata):
        if arg.is_broadcast:
            return bodo.libs.distributed_api.bcast(None, root=0, comm=spawner_intercomm)
        else:
            return bodo.libs.distributed_api.scatterv(
                None, root=0, comm=spawner_intercomm
            )

    if isinstance(arg, BodoSQLContextMetadata):
        from bodosql import BodoSQLContext

        tables = {
            tname: _recv_arg(tmeta, spawner_intercomm)
            for tname, tmeta in arg.tables.items()
        }
        return BodoSQLContext(tables, arg.catalog, arg.default_tz)

    # Handle distributed data nested inside tuples
    if isinstance(arg, tuple):
        return tuple(_recv_arg(v, spawner_intercomm) for v in arg)

    return arg


RESULT_REGISTRY: dict[str, pt.Any] = {}

# Once >3.12 is our minimum version we can use the below instead
# type is_distributed_t = bool + list[is_distributed_t] | tuple[is_distributed_t]
is_distributed_t: pt.TypeAlias = (
    bool | list["is_distributed_t"] | tuple["is_distributed_t"]
)


distributed_return_metadata_t: pt.TypeAlias = (
    DistributedReturnMetadata
    | list["distributed_return_metadata_t"]
    | dict[pt.Any, "distributed_return_metadata_t"]
)


def _build_distributed_return_metadata(
    res: pt.Any, logger: logging.Logger
) -> distributed_return_metadata_t:
    global RESULT_REGISTRY

    if isinstance(res, list):
        return [_build_distributed_return_metadata(val, logger) for val in res]
    if isinstance(res, (dict, typed.typeddict.Dict)):
        return {
            key: _build_distributed_return_metadata(val, logger)
            for key, val in res.items()
        }

    debug_worker_msg(logger, "Generating result id")
    res_id = str(
        comm_world.bcast(uuid.uuid4() if bodo.get_rank() == 0 else None, root=0)
    )
    debug_worker_msg(logger, f"Result id: {res_id}")
    RESULT_REGISTRY[res_id] = res
    debug_worker_msg(logger, f"Calculating total result length for {type(res)}")
    total_res_len = comm_world.reduce(len(res), op=MPI.SUM, root=0)
    index_data = None
    if isinstance(res, (pd.DataFrame, pd.Series)) and type(res.index) is pd.Index:
        # Convert index data to ArrowExtensionArray because we have a lazy ArrowExtensionArray
        index_data = _build_distributed_return_metadata(
            ArrowExtensionArray(pa.array(res.index._data)), logger
        )
        assert isinstance(index_data, DistributedReturnMetadata)
    return DistributedReturnMetadata(
        result_id=res_id,
        head=res.head(DISTRIBUTED_RETURN_HEAD_SIZE)
        if isinstance(res, pd.DataFrame)
        else res[:DISTRIBUTED_RETURN_HEAD_SIZE],
        nrows=total_res_len,
        index_data=index_data,
    )


def _send_output(
    res,
    is_distributed: is_distributed_t,
    spawner_intercomm: MPI.Intercomm,
    logger: logging.Logger,
):
    """Send function output to spawner. Uses gatherv for distributed data and also
    handles tuples.

    Args:
        res: output to send to spawner
        is_distributed: distribution info for output
        spawner_intercomm: MPI intercomm for spawner
    """
    # Tuple elements can have different distributions (tuples without distrubuted data
    # are treated like scalars)
    if isinstance(res, tuple) and isinstance(is_distributed, (tuple, list)):
        for val, dist in zip(res, is_distributed):
            _send_output(val, dist, spawner_intercomm, logger)
        return

    if is_distributed:
        distributed_return_metadata = _build_distributed_return_metadata(res, logger)
        debug_worker_msg(logger, f"{distributed_return_metadata=}")
        if bodo.get_rank() == 0:
            debug_worker_msg(logger, "Sending distributed result metadata to spawner")
            # Send the result id and a small chunk to the spawner
            spawner_intercomm.send(
                distributed_return_metadata,
                dest=0,
            )
    else:
        if bodo.get_rank() == 0:
            # Send non-distributed results
            spawner_intercomm.send(res, dest=0)


def _gather_res(
    is_distributed: is_distributed_t, res: pt.Any
) -> tuple[is_distributed_t, pt.Any]:
    """
    If any output is marked as distributed and empty on rank 0, gather the results and return an updated is_distributed flag and result
    """
    if isinstance(res, tuple) and isinstance(is_distributed, (tuple, list)):
        all_updated_is_distributed = []
        all_updated_res = []
        for val, dist in zip(res, is_distributed):
            updated_is_distributed, updated_res = _gather_res(dist, val)
            all_updated_is_distributed.append(updated_is_distributed)
            all_updated_res.append(updated_res)
        return tuple(all_updated_is_distributed), tuple(all_updated_res)

    # BSE-4101: Support lazy numpy arrays
    if is_distributed and (
        (
            comm_world.bcast(
                res is None or len(res) == 0 if bodo.get_rank() == 0 else None, root=0
            )
        )
        or isinstance(res, np.ndarray)
    ):
        # If the result is empty on rank 0, we can't send a head to the spawner
        # so just gather the results and send it all to to the spawner
        # We could probably optimize this by sending from all worker ranks to the spawner
        # but this shouldn't happen often

        return False, bodo.gatherv(res, root=0)

    return is_distributed, res


def exec_func_handler(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Callback to compile and execute the function being sent over
    driver_intercomm by the spawner"""
    global RESULT_REGISTRY

    # Receive function arguments
    pickled_args = spawner_intercomm.bcast(None, 0)
    (args, kwargs) = cloudpickle.loads(pickled_args)
    args = tuple(_recv_arg(arg, spawner_intercomm) for arg in args)
    kwargs = {name: _recv_arg(arg, spawner_intercomm) for name, arg in kwargs.items()}

    # Receive function dispatcher
    pickled_func = spawner_intercomm.bcast(None, 0)
    debug_worker_msg(logger, "Received pickled pyfunc from spawner.")

    caught_exception = None
    res = None
    func = None
    try:
        func = cloudpickle.loads(pickled_func)
        # ensure that we have a CPUDispatcher to compile and execute code
        assert isinstance(
            func, numba.core.registry.CPUDispatcher
        ), "Unexpected function type"
    except Exception as e:
        logger.error(f"Exception while trying to receive code: {e}")
        # TODO: check that all ranks raise an exception
        # forward_exception(e, comm_world, spawner_intercomm)
        func = None
        caught_exception = e

    if caught_exception is None:
        try:
            # Try to compile and execute it. Catch and share any errors with the spawner.
            debug_worker_msg(logger, "Compiling and executing func")
            res = func(*args, **kwargs)
        except Exception as e:
            debug_worker_msg(logger, f"Exception while trying to execute code: {e}")
            caught_exception = e

    poll_for_barrier(spawner_intercomm)
    has_exception = caught_exception is not None
    any_has_exception = comm_world.allreduce(has_exception, op=MPI.LOR)
    debug_worker_msg(logger, f"Propagating exception {has_exception=}")
    # Propagate any exceptions
    spawner_intercomm.gather(caught_exception, root=0)
    if any_has_exception:
        # Functions that raise exceptions don't have a return value
        return

    is_distributed = False
    if func is not None and len(func.signatures) > 0:
        # There should only be one signature compiled for the input function
        sig = func.signatures[0]
        assert sig in func.overloads

        # Extract return value distribution from metadata
        is_distributed = func.overloads[sig].metadata["is_return_distributed"]
    debug_worker_msg(logger, f"Function result {is_distributed=}")
    debug_worker_msg(logger, f"Result {res=}")

    is_distributed, res = _gather_res(is_distributed, res)
    debug_worker_msg(logger, f"Is_distributed after gathering empty {is_distributed=}")

    if bodo.get_rank() == 0:
        spawner_intercomm.send(is_distributed, dest=0)

    _send_output(res, is_distributed, spawner_intercomm, logger)


def worker_loop(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Main loop for the worker to listen and receive commands from driver_intercomm"""
    global RESULT_REGISTRY
    # Stored last data value received from scatterv/bcast for testing gatherv purposes

    while True:
        debug_worker_msg(logger, "Waiting for command")
        # TODO Change this to a wait that doesn't spin cycles
        # unnecessarily (e.g. see end_py in bodo/dl/utils.py)
        command = spawner_intercomm.bcast(None, 0)
        debug_worker_msg(logger, f"Received command: {command}")

        if command == CommandType.EXEC_FUNCTION.value:
            exec_func_handler(comm_world, spawner_intercomm, logger)
        elif command == CommandType.EXIT.value:
            debug_worker_msg(logger, "Exiting...")
            return
        elif command == CommandType.BROADCAST.value:
            bodo.libs.distributed_api.bcast(None, root=0, comm=spawner_intercomm)
            debug_worker_msg(logger, "Broadcast done")
        elif command == CommandType.SCATTER.value:
            data = bodo.libs.distributed_api.scatterv(
                None, root=0, comm=spawner_intercomm
            )
            res_id = str(
                comm_world.bcast(uuid.uuid4() if bodo.get_rank() == 0 else None, root=0)
            )
            RESULT_REGISTRY[res_id] = data
            spawner_intercomm.send(res_id, dest=0)
            debug_worker_msg(logger, "Scatter done")
        elif command == CommandType.GATHER.value:
            res_id = spawner_intercomm.bcast(None, 0)
            bodo.libs.distributed_api.gatherv(
                RESULT_REGISTRY.pop(res_id, None), root=0, comm=spawner_intercomm
            )
            debug_worker_msg(logger, f"Gather done for result {res_id}")

        elif command == CommandType.DELETE_RESULT.value:
            res_id = spawner_intercomm.bcast(None, 0)
            del RESULT_REGISTRY[res_id]
            debug_worker_msg(logger, f"Deleted result {res_id}")
        else:
            raise ValueError(f"Unsupported command '{command}!")


def debug_worker_msg(logger, msg):
    """Add worker number to message and send it to logger"""
    debug_msg(logger, f"Bodo Worker {bodo.get_rank()} {msg}")


if __name__ == "__main__":
    set_is_worker()
    # See comment in spawner about STDIN and MPI_Spawn
    # To allow some way to access stdin for debugging with pdb, the environment
    # variable BODO_WORKER0_INPUT can be set to a pipe, e.g.:
    # Run the following in a shell
    #   mkfifo /tmp/input # create a FIFO pipe
    #   export BODO_WORKER0_INPUT=/tmp/input
    #   export BODO_NUM_WORKERS=1
    #   python -u some_script_that_has_breakpoint_in_code_executed_by_worker.py
    # In a separate shell, do:
    #   cat > /tmp/input
    # Now you can write to the stdin of rank 0 by submitting input in the second
    # shell. Note that the worker will hang until there is at least one writer on
    # the pipe.
    if bodo.get_rank() == 0 and (infile := os.environ.get("BODO_WORKER0_INPUT")):
        fd = os.open(infile, os.O_RDONLY)
        os.dup2(fd, 0)
    else:
        sys.stdin.close()

    log_lvl = int(os.environ.get("BODO_WORKER_VERBOSE_LEVEL", "0"))
    bodo.set_verbose_level(log_lvl)

    comm_world: MPI.Intracomm = MPI.COMM_WORLD
    spawner_intercomm: MPI.Intercomm | None = comm_world.Get_parent()

    worker_loop(
        comm_world,
        spawner_intercomm,
        bodo.user_logging.get_current_bodo_verbose_logger(),
    )
