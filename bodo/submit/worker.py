# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Worker process to handle compiling and running python functions with
Bodo - note that this module should only be run with MPI.Spawn and not invoked
directly"""

import logging
import os
import sys

import cloudpickle
import numba

import bodo
from bodo.mpi4py import MPI
from bodo.submit.spawner import ArgMetadata, BodoSQLContextMetadata
from bodo.submit.utils import CommandType, poll_for_barrier
from bodo.submit.worker_state import set_is_worker


def _recv_arg(arg, spawner_intercomm):
    """Receive argument if it is a DataFrame/Series/Index/array value.

    Args:
        arg (Any or ArgMetadata): argument value or metadata
        spawner_intercomm (MPI.Intercomm): spawner intercomm handle

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

    return arg


def exec_func_handler(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Callback to compile and execute the function being sent over
    driver_intercomm by the spawner"""

    # Receive function arguments
    (args, kwargs) = spawner_intercomm.bcast(None, 0)
    args = tuple(_recv_arg(arg, spawner_intercomm) for arg in args)
    kwargs = {name: _recv_arg(arg, spawner_intercomm) for name, arg in kwargs.items()}

    # Receive function dispatcher
    pickled_func = spawner_intercomm.bcast(None, 0)
    logger.debug("Received pickled pyfunc from spawner.")

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
            logger.debug("Compiling and executing func")
            res = func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception while trying to execute code: {e}")
            caught_exception = e

    poll_for_barrier(spawner_intercomm)
    has_exception = caught_exception is not None
    logger.debug(f"Propagating exception {has_exception=}")
    # Propagate any exceptions
    spawner_intercomm.gather(caught_exception, root=0)

    is_distributed = False
    if func is not None and len(func.signatures) > 0:
        # There should only be one signature compiled for the input function
        sig = func.signatures[0]
        assert sig in func.overloads
        # Extract return value distribution from metadata
        is_distributed = func.overloads[sig].metadata["is_return_distributed"]
    logger.debug(f"Gathering result {is_distributed=}")

    spawner_intercomm.gather(is_distributed, root=0)
    if is_distributed:
        # Combine distributed results with gatherv
        bodo.gatherv(res, root=0, comm=spawner_intercomm)
    else:
        if bodo.get_rank() == 0:
            # broadcast non-distributed results
            spawner_intercomm.send(res, dest=0)


def worker_loop(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Main loop for the worker to listen and receive commands from driver_intercomm"""
    # Stored last data value received from scatterv/bcast for testing gatherv purposes
    last_received_data = None

    while True:
        logger.debug("Waiting for command")
        # TODO Change this to a wait that doesn't spin cycles
        # unnecessarily (e.g. see end_py in bodo/dl/utils.py)
        command = spawner_intercomm.bcast(None, 0)
        logger.debug(f"Received command: {command}")

        if command == CommandType.EXEC_FUNCTION.value:
            exec_func_handler(comm_world, spawner_intercomm, logger)
        elif command == CommandType.EXIT.value:
            logger.debug("Exiting...")
            return
        elif command == CommandType.BROADCAST.value:
            last_received_data = bodo.libs.distributed_api.bcast(
                None, root=0, comm=spawner_intercomm
            )
            logger.debug("Broadcast done")
        elif command == CommandType.SCATTER.value:
            last_received_data = bodo.libs.distributed_api.scatterv(
                None, root=0, comm=spawner_intercomm
            )
            logger.debug("Scatter done")
        elif command == CommandType.GATHER.value:
            bodo.libs.distributed_api.gatherv(
                last_received_data, root=0, comm=spawner_intercomm
            )
            logger.debug("Gather done")
        else:
            raise ValueError(f"Unsupported command '{command}!")


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

    logger = logging.getLogger(f"Bodo Worker {bodo.get_rank()}")

    log_lvl = int(os.environ.get("BODO_WORKER_LOG_LEVEL", "10"))
    logger.setLevel(log_lvl)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # create formatter
    formater = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler.setFormatter(formater)
    logger.addHandler(handler)

    comm_world: MPI.Intracomm = MPI.COMM_WORLD
    spawner_intercomm: MPI.Intercomm | None = comm_world.Get_parent()

    worker_loop(comm_world, spawner_intercomm, logger)
