# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Worker process to handle compiling and running python functions with
Bodo - note that this module should only be run with MPI.Spawn and not invoked
directly"""

import logging
import os
import sys
import types as pytypes

import cloudpickle

import bodo
from bodo.mpi4py import MPI
from bodo.submit.spawner import SubmitDispatcher
from bodo.submit.utils import CommandType, poll_for_barrier


def exec_func_handler(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Callback to compile and execute the function being sent over
    driver_intercomm by the spawner"""

    ## Receive pyfunc ##
    # Get the python function and arguments
    pickled_func = spawner_intercomm.bcast(None, 0)
    logger.debug("Received pickled pyfunc from spawner.")

    # For testing SQL
    recv_bsql_context = spawner_intercomm.bcast(None, 0)
    if recv_bsql_context:
        arg0_bc = spawner_intercomm.bcast(None, 0)
        logger.debug("Received pickled BodoSQLContext from spawner.")

    pyfunc = None
    decorator = None

    try:
        pyfunc = cloudpickle.loads(pickled_func)
    except Exception as e:
        logger.error(f"Exception while trying to unpickle code: {e}")
        # TODO: check that all ranks raise an exception
        # forward_exception(e, comm_world, spawner_intercomm)
        pyfunc = None

    caught_exception = None
    try:
        # Convert all recursively used SubmitDispatchers to regular Numba Dispatchers.
        for objname, obj in list(pyfunc.__globals__.items()):
            if isinstance(obj, SubmitDispatcher):
                pyfunc.__globals__[objname] = obj.dispatcher
        # fix references to the module that the submitted func originates from
        # TODO(aneesh) can this be done via import/importlib instead?
        if pyfunc.__module__ not in sys.modules:
            sys.modules[pyfunc.__module__] = pytypes.ModuleType(pyfunc.__module__)

        # Assert that this function is from @submit_jit and retrieve any
        # additional arguments to be sent to bodo.jit.
        assert pyfunc.__dict__.get("is_submit_jit")
        decorator_args = pyfunc.__dict__.get("submit_jit_args")

        # Create an alias for all new imports - this is to workaround the
        # compile not being able to pick up the module aliases even though
        # they are present in pyfunc.__globals__.
        # TODO Find a more robust solution.
        for k, v in pyfunc.__globals__.items():
            if isinstance(v, pytypes.ModuleType):
                assert v in sys.modules.values()
                if k not in globals():
                    globals()[k] = v

        decorator = bodo.jit
        if len(decorator_args):
            decorator = bodo.jit(**decorator_args)

        # Apply decorator to get the dispatcher
        func = decorator(pyfunc)

        # Try to compile and execute it. Catch and share any errors with the spawner.
        logger.debug("Compiling and executing func")
        if recv_bsql_context:
            func(arg0_bc)
        else:
            func()
    except Exception as e:
        logger.error(f"Exception while trying to execute code: {e}")
        caught_exception = e

    poll_for_barrier(spawner_intercomm)
    # Propagate any exceptions
    spawner_intercomm.gather(caught_exception, root=0)


def worker_loop(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Main loop for the worker to listen and receive commands from driver_intercomm"""
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
            bodo.libs.distributed_api.bcast(None, root=0, comm=spawner_intercomm)
            logger.debug("Broadcast done")
        else:
            raise ValueError(f"Unsupported command '{command}!")


if __name__ == "__main__":
    # See comment in spawner about STDIN and MPI_Spawn
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

    comm_world.barrier()
    worker_loop(comm_world, spawner_intercomm, logger)