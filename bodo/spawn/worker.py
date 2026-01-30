from __future__ import annotations

"""Worker process to handle compiling and running python functions with
Bodo - note that this module should only be run with MPI.Spawn and not invoked
directly
This file should import JIT lazily to
avoid slowing down non-JIT code paths.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import typing as pt
import uuid
import warnings
from copy import deepcopy

import cloudpickle
import numpy as np
import pandas as pd
import pyarrow as pa
from mpi4py import MPI
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.base import ExtensionArray

import bodo
from bodo.spawn.spawner import BodoSQLContextMetadata, env_var_prefix
from bodo.spawn.utils import (
    ArgMetadata,
    CommandType,
    WorkerProcess,
    debug_msg,
    poll_for_barrier,
    set_global_config,
)
from bodo.spawn.worker_state import set_is_worker

if pt.TYPE_CHECKING:
    from bodo.pandas import LazyMetadata

    distributed_return_metadata_t = (
        LazyMetadata
        | list["distributed_return_metadata_t"]
        | dict[pt.Any, "distributed_return_metadata_t"]
        | ExtensionArray
    )


DISTRIBUTED_RETURN_HEAD_SIZE: int = 5

# PID of spawning process - used to signal the spawner of function completion
spawnerpid = None


_recv_arg_return_t = (
    tuple[pt.Any, ArgMetadata | None] | tuple["_recv_arg_return_t", ...]
)


def _recv_arg(
    arg: pt.Any | ArgMetadata, spawner_intercomm: MPI.Intercomm
) -> _recv_arg_return_t:
    """Receive argument if it is a DataFrame/Series/Index/array value.

    Args:
        arg: argument value or metadata
        spawner_intercomm: spawner intercomm handle

    Returns:
        Any: received function argument
    """
    import bodo

    if isinstance(arg, ArgMetadata):
        if arg == ArgMetadata.BROADCAST:
            # Import compiler lazily
            import bodo.decorators  # isort:skip # noqa

            return (
                bodo.libs.distributed_api.bcast(None, root=0, comm=spawner_intercomm),
                arg,
            )
        elif arg == ArgMetadata.SCATTER:
            return (
                bodo.scatterv(None, root=0, comm=spawner_intercomm),
                arg,
            )
        elif arg == ArgMetadata.LAZY:
            res_id = spawner_intercomm.bcast(None, root=0)
            return (RESULT_REGISTRY[res_id], arg)

    if isinstance(arg, BodoSQLContextMetadata):
        from bodosql import BodoSQLContext, TablePath

        tables = {
            tname: tmeta
            if isinstance(tmeta, TablePath)
            else _recv_arg(tmeta, spawner_intercomm)[0]
            for tname, tmeta in arg.tables.items()
        }
        # BSE-4154: Support lazy data structures in bodosql context arguments
        return BodoSQLContext(tables, arg.catalog, arg.default_tz), None

    # Handle distributed data nested inside tuples
    if isinstance(arg, tuple):
        if len(arg) == 0:
            return arg, ()
        args, args_meta = zip(*[_recv_arg(v, spawner_intercomm) for v in arg])
        return args, args_meta

    return arg, None


RESULT_REGISTRY: dict[str, pt.Any] = {}
PROCESS_REGISTRY: dict[uuid.UUID, subprocess.Popen | None] = {}

# Once >3.12 is our minimum version we can use the below instead
# type is_distributed_t = bool + list[is_distributed_t] | tuple[is_distributed_t]
is_distributed_t = bool | list["is_distributed_t"] | tuple["is_distributed_t", ...]


def _build_index_data(
    res: pt.Any, logger: logging.Logger
) -> distributed_return_metadata_t | None:
    """
    Construct distributed return metadata for the index of res if it has an index
    """
    from bodo.pandas.utils import BODO_NONE_DUMMY

    if isinstance(res, (pd.DataFrame, pd.Series)):
        if isinstance(res.index, pd.MultiIndex):
            res.index.names = [
                name if name is not None else BODO_NONE_DUMMY
                for name in res.index.names
            ]
            return _build_distributed_return_metadata(
                res.index.to_frame(index=False, allow_duplicates=True), logger
            )
        elif isinstance(res.index, pd.IntervalIndex):
            return (
                _build_distributed_return_metadata(
                    ArrowExtensionArray(pa.array(res.index.left)), logger
                ),
                _build_distributed_return_metadata(
                    ArrowExtensionArray(pa.array(res.index.right)), logger
                ),
            )
        elif isinstance(
            res.index, (pd.CategoricalIndex, pd.DatetimeIndex, pd.TimedeltaIndex)
        ):
            return bodo.gatherv(res.index._data)
        elif isinstance(res.index, pd.PeriodIndex):
            # This is a hack since we can't unbox a numpy array created from res.index._data for PeriodIndex
            # since we're missing a proper PeriodArray but it's fine since we'll replace this
            # with lazy numpy soon
            return bodo.gatherv(res.index)
        elif isinstance(res.index, pd.Index):
            # Convert index data to ArrowExtensionArray because we have a lazy ArrowExtensionArray
            return _build_distributed_return_metadata(
                ArrowExtensionArray(pa.array(res.index._data)), logger
            )

    return None


def _build_distributed_return_metadata(
    res: pt.Any, logger: logging.Logger
) -> distributed_return_metadata_t:
    from numba import typed

    import bodo
    from bodo.pandas import LazyMetadata

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
    index_data = _build_index_data(res, logger)
    return LazyMetadata(
        result_id=res_id,
        head=res.head(DISTRIBUTED_RETURN_HEAD_SIZE)
        if isinstance(res, (pd.DataFrame, pd.Series))
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


def _is_table_type(t):
    """Helper for checking Table type that imports the JIT compiler lazily."""
    # Import compiler lazily
    import bodo
    import bodo.decorators  # isort:skip # noqa

    import bodo.hiframes
    import bodo.hiframes.table

    return isinstance(t, bodo.hiframes.table.Table)


def _gather_res(
    is_distributed: is_distributed_t, res: pt.Any
) -> tuple[is_distributed_t, pt.Any]:
    """
    If any output is marked as distributed and empty on rank 0, gather the results and return an updated is_distributed flag and result
    """

    from bodo import BodoWarning

    if isinstance(res, tuple) and isinstance(is_distributed, (tuple, list)):
        all_updated_is_distributed = []
        all_updated_res = []
        for val, dist in zip(res, is_distributed):
            updated_is_distributed, updated_res = _gather_res(dist, val)
            all_updated_is_distributed.append(updated_is_distributed)
            all_updated_res.append(updated_res)
        return tuple(all_updated_is_distributed), tuple(all_updated_res)

    # Handle corner case of returning BodoSQLContext
    if type(res).__name__ == "BodoSQLContext":
        # Import bodosql lazily to avoid import overhead when not necessary
        from bodosql import BodoSQLContext

        assert isinstance(res, BodoSQLContext), "invalid BodoSQLContext"
        warnings.warn(
            BodoWarning(
                "Gathering tables of BodoSQLContext since returning distributed BodoSQLContext from spawned JIT functions not supported yet."
            )
        )
        res.tables = {
            name: (bodo.gatherv(table, root=0) if is_distributed[i] else table)
            for i, (name, table) in enumerate(res.tables.items())
        }
        return False, res

    # BSE-4101: Support lazy numpy arrays
    if is_distributed and (
        (
            comm_world.bcast(
                res is None or len(res) == 0 if bodo.get_rank() == 0 else None, root=0
            )
        )
        or isinstance(res, np.ndarray)
        # TODO[BSE-4198]: support lazy Index wrappers
        or isinstance(res, pd.Index)
        or isinstance(res, pd.Categorical)
        or isinstance(res, pd.arrays.IntervalArray)
        # TODO[BSE-4205]: move DatetimeArray to use Arrow
        or isinstance(res, pd.arrays.DatetimeArray)
        or isinstance(res, pa.lib.NullArray)
        or (type(res).__name__ == "Table" and _is_table_type(res))
    ):
        # If the result is empty on rank 0, we can't send a head to the spawner
        # so just gather the results and send it all to to the spawner
        #
        # We could probably optimize this by sending from all worker ranks to the spawner
        # but this shouldn't happen often

        return False, bodo.gatherv(res, root=0)

    return is_distributed, res


def _send_updated_arg(
    arg: pt.Any | tuple[pt.Any, ...],
    arg_meta: ArgMetadata | None | tuple[ArgMetadata, ...],
    spawner_intercomm: MPI.Comm,
    logger: logging.Logger,
):
    """Send updated arguments to spawner if needed"""
    if isinstance(arg, tuple):
        assert isinstance(arg_meta, tuple)
        for a, m in zip(arg, arg_meta):
            _send_updated_arg(a, m, spawner_intercomm, logger)
        return

    if not isinstance(arg_meta, ArgMetadata):
        return
    if arg_meta is ArgMetadata.LAZY:
        debug_worker_msg(logger, "Sending updated lazy arg to spawner")
        distributed_return_meta = _build_distributed_return_metadata(arg, logger)

        if bodo.get_rank() == 0:
            spawner_intercomm.send(distributed_return_meta, dest=0)


def _send_updated_args(
    args: tuple[pt.Any, ...],
    args_meta: tuple[ArgMetadata | None, ...],
    kwargs: dict[str, pt.Any],
    kwargs_meta: dict[str, ArgMetadata | None],
    spawner_intercomm: MPI.Comm,
    logger: logging.Logger,
):
    """Send updated args and kwargs to spawner if needed, this can happen
    when a lazy object is passed as an argument and modified in the function.
    We don't check if it's been modified, we just always update the head for lazy objects"""
    for arg, arg_meta in zip(args, args_meta):
        _send_updated_arg(arg, arg_meta, spawner_intercomm, logger)
    for arg, arg_meta in zip(kwargs.values(), kwargs_meta.values()):
        _send_updated_arg(arg, arg_meta, spawner_intercomm, logger)


def _update_env_var(new_env_var, propagate_env):
    """Updates environment variables received from spawner.

    Args:
        new_env_var (dict[str, str]): env vars to set
        propagate_env: additional env vars to track"""

    # BODO_HDFS_CORE_SITE_LOC_DIR is set by workers during import bodo and
    # might not exist if taken from the spawner. We want to avoid inheritting
    # unless explicitly specified i.e. with propagate_env.
    # TODO [BSE-4219] To prevent this issue, we can make the directory
    # creation via LazyTemporaryDirectory spawn mode aware.
    worker_override_env = {"BODO_HDFS_CORE_SITE_LOC_DIR"}

    for env_var in new_env_var:
        if env_var not in worker_override_env or env_var in propagate_env:
            os.environ[env_var] = new_env_var[env_var]
    for env_var in os.environ:
        if env_var not in new_env_var:
            if env_var.startswith(env_var_prefix) or env_var in propagate_env + [
                "DYLD_INSERT_LIBRARIES"
            ]:
                del os.environ[env_var]


def _is_distributable_result(res):
    """
    Check if the worker result is a distributable type which requires gather to spawner.
    Avoids importing the compiler as much as possible to reduce overheads.
    """
    from pandas.core.arrays.arrow import ArrowExtensionArray

    import bodo

    if isinstance(
        res, (pd.DataFrame, pd.Series, pd.Index, np.ndarray, ArrowExtensionArray)
    ):
        return True

    if pd.api.types.is_scalar(res):
        return False

    # df.to_iceberg returns a list of lists of tuples with information
    # about each file written. We check for this case separately to avoid
    # importing the compiler.
    if isinstance(res, list) and (
        len(res) == 0
        or (
            isinstance(res[0], list)
            and (len(res[0]) == 0 or isinstance(res[0][0], tuple))
        )
    ):
        return False

    # Import compiler lazily
    import bodo.decorators  # isort:skip # noqa

    return bodo.utils.utils.is_distributable_typ(bodo.typeof(res))


def exec_func_handler(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Callback to compile and execute the function being sent over
    driver_intercomm by the spawner"""

    import bodo

    global RESULT_REGISTRY
    debug_worker_msg(logger, "Begin listening for function.")

    # Update environment variables
    new_env_var = spawner_intercomm.bcast(None, 0)
    propagate_env = spawner_intercomm.bcast(None, 0)
    original_env_var = deepcopy(os.environ)
    _update_env_var(new_env_var, propagate_env)

    # Receive function arguments
    pickled_args = spawner_intercomm.bcast(None, 0)
    args_and_meta, kwargs_and_meta = cloudpickle.loads(pickled_args)
    debug_worker_msg(
        logger, "Received replicated and meta args and kwargs from spawner"
    )
    args, args_meta = [], []
    for arg in args_and_meta:
        arg, meta = _recv_arg(arg, spawner_intercomm)
        args.append(arg)
        args_meta.append(meta)
    args, args_meta = tuple(args), tuple(args_meta)
    kwargs, kwargs_meta = {}, {}
    for name, arg in kwargs_and_meta.items():
        kwargs[name], kwargs_meta[name] = _recv_arg(arg, spawner_intercomm)
    debug_worker_msg(logger, "Received args and kwargs from spawner.")

    # Receive function dispatcher
    pickled_func = spawner_intercomm.bcast(None, 0)
    debug_worker_msg(logger, "Received pickled pyfunc from spawner.")

    caught_exception = None
    res = None
    func = None
    is_dispatcher = False
    try:
        func = cloudpickle.loads(pickled_func)
        is_dispatcher = type(func).__name__ == "CPUDispatcher"
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

    debug_worker_msg(logger, "signaling spawner of completion")

    has_exception = caught_exception is not None
    any_has_exception = comm_world.allreduce(has_exception, op=MPI.LOR)

    if sys.platform == "win32":
        # If windows, use poll_for_barrier since kill isn't very portable.
        # We set the frequency to None here because it's okay to busy wait on
        # the workers - on the spawner, we will sleep more often to avoid taking
        # up the CPU.
        # TODO(aneesh): We could instead use blocking IO on a socket to put the
        # spawner to sleep (and send on the socket from worker 0 to wake). This
        # would also allow us to not enforce that worker 0 and the spawner
        # reside on the same physical machine.
        poll_for_barrier(spawner_intercomm, poll_freq=None)
    else:
        # Wake up spawner so recieve results/errors. We use a signal instead of
        # a barrier to avoid busy waiting on the spawner
        if bodo.get_rank() == 0:
            os.kill(spawnerpid, signal.SIGUSR1)

    debug_worker_msg(logger, f"Propagating exception {has_exception=}")
    # Propagate any exceptions
    spawner_intercomm.gather(caught_exception, root=0)
    if any_has_exception:
        # Functions that raise exceptions don't have a return value
        return

    is_distributed = False
    if is_dispatcher and func is not None and len(func.signatures) > 0:
        # There should only be one signature compiled for the input function
        sig = func.signatures[0]
        assert sig in func.overloads

        # Extract return value distribution from metadata
        is_distributed = func.overloads[sig].metadata["is_return_distributed"]

    # TODO(ehsan): handle other types like scalars. The challenge is that scalars may
    # not be replicated in the non-JIT cases like map_partitions, so we have to define
    # the semantics (e.g. gather all values across ranks in a list?).
    if not is_dispatcher:
        is_distributed = _is_distributable_result(res)

    debug_worker_msg(logger, f"Function result {is_distributed=}")

    is_distributed, res = _gather_res(is_distributed, res)
    debug_worker_msg(logger, f"Is_distributed after gathering empty {is_distributed=}")

    if bodo.get_rank() == 0:
        spawner_intercomm.send(is_distributed, dest=0)

    debug_worker_msg(logger, "Sending output to spawner")
    _send_output(res, is_distributed, spawner_intercomm, logger)

    debug_worker_msg(
        logger, "Sending updated args and kwargs to spawner after function execution"
    )
    _send_updated_args(args, args_meta, kwargs, kwargs_meta, spawner_intercomm, logger)

    # restore env var
    os.environ = original_env_var


def handle_spawn_process(
    command: str | list[str],
    env: dict[str, str],
    cwd: str | None,
    comm_world: MPI.Intracomm,
    logger: logging.Logger,
):
    """Handle spawning a new process and return the process handle"""
    pid = None
    popen = None
    if bodo.get_rank() in bodo.get_nodes_first_ranks(comm_world):
        debug_worker_msg(logger, f"Spawning process with command {command}")
        popen = subprocess.Popen(
            command,
            env=env,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid = popen.pid
        debug_worker_msg(logger, f"Spawned process with pid {pid}")
    ranks_to_pids = comm_world.gather((bodo.get_rank(), pid), root=0)

    worker_process = None
    if bodo.get_rank() == 0:
        ranks_to_pids = {r: p for r, p in ranks_to_pids if p is not None}

        worker_process = WorkerProcess(ranks_to_pids)

    worker_process = comm_world.bcast(worker_process, root=0)

    PROCESS_REGISTRY[worker_process._uuid] = popen
    return worker_process


def handle_stop_process(
    worker_process: WorkerProcess,
    logger: logging.Logger,
):
    """Handle stopping a process and return the process handle"""
    debug_worker_msg(logger, f"Stopping process with uuid {worker_process._uuid}")
    if worker_process._uuid not in PROCESS_REGISTRY:
        raise ValueError(f"Process with uuid {worker_process._uuid} not found")

    popen = PROCESS_REGISTRY.pop(worker_process._uuid)
    # The process is managed by a different rank
    if popen is None:
        return

    if popen.poll() is None:
        debug_worker_msg(logger, f"Killing process with pid {popen.pid}")
        popen.terminate()
        # Wait for the process to terminate
        try:
            popen.wait(timeout=5)
        except subprocess.TimeoutExpired:
            debug_worker_msg(
                logger, f"Process with pid {popen.pid} did not stop in time"
            )
            # If the process does not stop, we can try to forcefully terminate it
            popen.kill()
            debug_worker_msg(
                logger, f"Process with pid {popen.pid} forcefully terminated"
            )
        else:
            debug_worker_msg(
                logger, f"Process with pid {popen.pid} stopped successfully"
            )
    else:
        debug_worker_msg(logger, f"Process with pid {popen.pid} already stopped")


class StdoutQueue:
    """Replacement for stdout/stderr that sends output to the spawner via a ZeroMQ socket."""

    def __init__(self, out_socket, is_stderr):
        self.out_socket = out_socket
        self.is_stderr = is_stderr
        self.closed = False

    def write(self, msg):
        # TODO(Ehsan): buffer output similar to ipykernel
        self.out_socket.send_string(f"{int(self.is_stderr)}{msg}")

    def flush(self):
        if self.is_stderr:
            sys.__stderr__.flush()
        else:
            sys.__stdout__.flush()

    def close(self):
        self.closed = True


def worker_loop(
    comm_world: MPI.Intracomm, spawner_intercomm: MPI.Intercomm, logger: logging.Logger
):
    """Main loop for the worker to listen and receive commands from driver_intercomm"""
    import bodo

    global RESULT_REGISTRY
    global spawnerpid

    spawnerpid = spawner_intercomm.bcast(None, 0)
    if bodo.get_rank() == 0:
        spawner_hostname = spawner_intercomm.recv(source=0)
        assert spawner_hostname == socket.gethostname(), (
            "Spawner and worker 0 must be on the same machine"
        )

    # Send output to spawner manually if we are in Jupyter on Windows since child processes
    # don't inherit file descriptors from the parent process.
    out_socket = None
    if (
        bodo.spawn.utils.is_jupyter_on_windows()
        or bodo.spawn.utils.is_jupyter_on_bodo_platform()
    ):
        import zmq

        connection_info = spawner_intercomm.bcast(None, 0)

        context = zmq.Context()
        out_socket = context.socket(zmq.PUSH)
        out_socket.connect(connection_info)
        sys.stdout = StdoutQueue(out_socket, False)
        sys.stderr = StdoutQueue(out_socket, True)

    while True:
        debug_worker_msg(logger, "Waiting for command")
        # TODO Change this to a wait that doesn't spin cycles
        # unnecessarily
        command = spawner_intercomm.bcast(None, 0)
        debug_worker_msg(logger, f"Received command: {command}")

        if command == CommandType.EXEC_FUNCTION.value:
            exec_func_handler(comm_world, spawner_intercomm, logger)
        elif command == CommandType.EXIT.value:
            debug_worker_msg(logger, "Exiting...")
            if out_socket:
                out_socket.close()

            for worker_process_uuid in list(PROCESS_REGISTRY.keys()):
                worker_process = WorkerProcess()
                worker_process._uuid = worker_process_uuid
                handle_stop_process(worker_process, logger)

            return
        elif command == CommandType.BROADCAST.value:
            # Import compiler lazily
            import bodo.decorators  # isort:skip # noqa

            bodo.libs.distributed_api.bcast(None, root=0, comm=spawner_intercomm)
            debug_worker_msg(logger, "Broadcast done")
        elif command == CommandType.SCATTER.value:
            data = bodo.scatterv(None, root=0, comm=spawner_intercomm)
            res_id = str(
                comm_world.bcast(uuid.uuid4() if bodo.get_rank() == 0 else None, root=0)
            )
            RESULT_REGISTRY[res_id] = data
            if bodo.get_rank() == 0:
                spawner_intercomm.send(res_id, dest=0)
            debug_worker_msg(logger, "Scatter done")
        elif command == CommandType.GATHER.value:
            res_id = spawner_intercomm.bcast(None, 0)
            bodo.gatherv(
                RESULT_REGISTRY.pop(res_id, None), root=0, comm=spawner_intercomm
            )
            debug_worker_msg(logger, f"Gather done for result {res_id}")

        elif command == CommandType.DELETE_RESULT.value:
            res_id = spawner_intercomm.bcast(None, 0)
            del RESULT_REGISTRY[res_id]
            debug_worker_msg(logger, f"Deleted result {res_id}")
        elif command == CommandType.REGISTER_TYPE.value:
            from numba.core import types

            (type_name, type_value) = spawner_intercomm.bcast(None, 0)
            setattr(types, type_name, type_value)
            debug_worker_msg(logger, f"Added type {type_name}")
        elif command == CommandType.SET_CONFIG.value:
            (config_name, config_value) = spawner_intercomm.bcast(None, 0)
            set_global_config(config_name, config_value)
            debug_worker_msg(logger, f"Set config {config_name}={config_value}")
        elif command == CommandType.SPAWN_PROCESS.value:
            command, env, cwd = spawner_intercomm.bcast(None, 0)
            worker_process = handle_spawn_process(command, env, cwd, comm_world, logger)
            if bodo.get_rank() == 0:
                spawner_intercomm.send(worker_process, dest=0)
        elif command == CommandType.STOP_PROCESS.value:
            worker_process = spawner_intercomm.bcast(None, 0)
            handle_stop_process(worker_process, logger)
            if bodo.get_rank() == 0:
                spawner_intercomm.send(None, dest=0)
        elif command == CommandType.SCATTER_JIT.value:
            data = bodo.libs.distributed_api.scatterv(
                None, root=0, comm=spawner_intercomm
            )
            res_id = str(
                comm_world.bcast(uuid.uuid4() if bodo.get_rank() == 0 else None, root=0)
            )
            RESULT_REGISTRY[res_id] = data
            if bodo.get_rank() == 0:
                spawner_intercomm.send(res_id, dest=0)
            debug_worker_msg(logger, "Scatter jit done")
        elif command == CommandType.GATHER_JIT.value:
            res_id = spawner_intercomm.bcast(None, 0)
            bodo.libs.distributed_api.gatherv(
                RESULT_REGISTRY.pop(res_id, None), root=0, comm=spawner_intercomm
            )
            debug_worker_msg(logger, f"Gather jit done for result {res_id}")
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
