# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Spawner-worker compilation implementation"""

import atexit
import contextlib
import itertools
import logging
import os
import sys
import time
import typing as pt

import cloudpickle
import numba
import pandas as pd
import psutil
from pandas.core.arrays.arrow.array import ArrowExtensionArray

import bodo
import bodo.user_logging
from bodo.mpi4py import MPI
from bodo.pandas import (
    BodoDataFrame,
    BodoSeries,
    LazyArrowExtensionArray,
    get_lazy_manager_class,
    get_lazy_single_manager_class,
)
from bodo.submit.utils import (
    CommandType,
    DistributedReturnMetadata,
    debug_msg,
    poll_for_barrier,
)
from bodo.utils.utils import is_distributable_typ

# Reference to BodoSQLContext class to be lazily initialized if BodoSQLContext
# is detected
BodoSQLContextCls = None


@contextlib.contextmanager
def no_stdin():
    """Temporarily close stdin and execute a block of code"""
    # Save a refence to the original stdin
    stdin_dup = os.dup(0)
    # Close stdin
    os.close(0)
    # open /dev/null as fd 0
    nullfd = os.open("/dev/null", os.O_RDONLY)
    os.dup2(nullfd, 0)
    try:
        yield
    finally:
        # Restore the saved fd
        os.dup2(stdin_dup, 0)


def get_num_workers():
    """Returns the number of workers to spawn.
    If BODO_NUM_WORKERS is set, spawn that many workers. Else, we will spawn as
    many workers as there are physical cores on this machine."""
    n_pes = 2
    if n_pes_env := os.environ.get("BODO_NUM_WORKERS"):
        n_pes = int(n_pes_env)
    elif cpu_count := psutil.cpu_count(logical=False):
        n_pes = cpu_count
    return n_pes


class ArgMetadata:
    """Argument metadata to inform workers about other arguments to receive separately.
    E.g. broadcast or scatter a dataframe from spawner to workers.
    Used for DataFrame/Series/Index/array arguments.
    """

    def __init__(self, is_broadcast):
        self.is_broadcast = is_broadcast


class BodoSQLContextMetadata:
    """Argument metadata for BodoSQLContext values which allows reconstructing
    BodoSQLContext on workers properly by receiving table DataFrames separately.
    """

    def __init__(self, tables, catalog, default_tz):
        self.tables = tables
        self.catalog = catalog
        self.default_tz = default_tz


class Spawner:
    """
    State for the Spawner/User program that will spawn
    the worker processes and communicate with them to execute
    JIT functions.
    """

    logger: logging.Logger
    comm_world: MPI.Intracomm
    worker_intercomm: MPI.Intercomm
    exec_intercomm_addr: int

    def __init__(self):
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()

        self.comm_world = MPI.COMM_WORLD

        n_pes = get_num_workers()
        debug_msg(self.logger, f"Trying to spawn {n_pes} workers...")
        errcodes = [0] * n_pes
        t0 = time.monotonic()

        # MPI_Spawn (using MPICH) will spawn a Hydra process for each rank which
        # then spawns the command provided below. Hydra handles STDIN by calling
        # poll on fd 0, and then forwarding input to the first local process.
        # However, if the spawner was NOT run with mpiexec, then Hydra will fail to
        # forward STDIN for the worker and kill the spawner. The worker does not
        # need STDIN, so we instead close STDIN before spawning the Hydra process,
        # and then restore STDIN afterwards. This is necessary for environments where
        # interactivity is needed, e.g. ipython/python REPL.
        with no_stdin():
            # Copy over all BODO* environment variables - note that this isn't
            # needed if the parent process is spawned with MPI and the env vars
            # are set via `mpiexec -env`.
            environ_args = []
            for k in os.environ:
                # DYLD_INSERT_LIBRARIES can be difficult to propogate to child
                # process. e.g.:
                # https://stackoverflow.com/questions/43941322/dyld-insert-libraries-ignored-when-calling-application-through-bash
                # So for now, we use BODO_DYLD_INSERT_LIBRARIES as a way to
                # inform the spawner to set the variable for the child processes
                if k == "BODO_DYLD_INSERT_LIBRARIES":
                    environ_args.append(f"DYLD_INSERT_LIBRARIES={os.environ[k]}")
                elif k.startswith("BODO"):
                    environ_args.append(f"{k}={os.environ[k]}")
            # linux uses LD_PRELOAD instead of DYLD_INSERT_LIBRARIES, which
            # needs no special handling beyond ensuring that the child picks up
            # the variable.
            if "LD_PRELOAD" in os.environ:
                preload = os.environ["LD_PRELOAD"]
                environ_args.append(f"LD_PRELOAD={preload}")

            # Send spawner log level to workers
            environ_args.append(
                f"BODO_WORKER_VERBOSE_LEVEL={bodo.user_logging.get_verbose_level()}"
            )

            # run python with -u to prevent STDOUT from buffering
            self.worker_intercomm = self.comm_world.Spawn(
                # get the same python executable that is currently running
                "env",
                environ_args + [sys.executable, "-u", "-m", "bodo.submit.worker"],
                n_pes,
                MPI.INFO_NULL,
                0,
                errcodes,
            )
        debug_msg(
            self.logger, f"Spawned {n_pes} workers in {(time.monotonic()-t0):0.4f}s"
        )
        self.exec_intercomm_addr = MPI._addressof(self.worker_intercomm)

    def _recv_output(self, output_is_distributed: pt.Union[bool, list[bool]]):
        """Receive output of function execution from workers

        Args:
            output_is_distributed: distribution info of output

        Returns:
            Any: output value
        """
        # Tuple elements can have different distribution info
        if isinstance(output_is_distributed, (tuple, list)):
            return tuple(self._recv_output(d) for d in output_is_distributed)
        if output_is_distributed:
            debug_msg(
                self.logger,
                "Getting distributed return metadata for distributed output",
            )
            distributed_return_metadata = self.worker_intercomm.recv(source=0)
            res = self.wrap_distributed_result(distributed_return_metadata)
        else:
            debug_msg(self.logger, "Getting replicated result")
            res = self.worker_intercomm.recv(source=0)

        return res

    def submit_func_to_workers(self, dispatcher: "SubmitDispatcher", *args, **kwargs):
        """Send func to be compiled and executed on spawned process"""

        bcast_root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL
        self.worker_intercomm.bcast(CommandType.EXEC_FUNCTION.value, bcast_root)

        # Send arguments and update dispatcher distributed flags for arguments
        self._send_args_update_dist_flags(dispatcher, args, kwargs, bcast_root)

        # Send dispatcher
        pickled_func = cloudpickle.dumps(dispatcher)
        self.worker_intercomm.bcast(pickled_func, root=bcast_root)

        # Wait for execution to finish
        gather_root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL
        poll_for_barrier(self.worker_intercomm)
        caught_exceptions = self.worker_intercomm.gather(None, root=gather_root)

        assert caught_exceptions is not None
        if any(caught_exceptions):
            types = {type(excep) for excep in caught_exceptions}
            msgs = {
                str(excep) if excep is not None else None for excep in caught_exceptions
            }
            all_ranks_failed = all(caught_exceptions)
            if all_ranks_failed and len(types) == 1 and len(msgs) == 1:
                excep = caught_exceptions[0]
                raise Exception("All ranks failed with the same exception") from excep
            else:
                # Annotate exceptions with their rank
                exceptions = []
                for i, excep in enumerate(caught_exceptions):
                    if excep is None:
                        continue
                    excep.add_note(f"^ From rank {i}")
                    exceptions.append(excep)

                # Combine all exceptions into a single chain
                accumulated_exception = None
                for excep in exceptions:
                    try:
                        raise excep from accumulated_exception
                    except Exception as e:
                        accumulated_exception = e
                # Raise the combined exception
                raise Exception("Some ranks failed") from accumulated_exception

        # Get output from workers
        output_is_distributed = self.worker_intercomm.recv(source=0)
        res = self._recv_output(output_is_distributed)

        return res

    def wrap_distributed_result(
        self, distributed_return_metadata: DistributedReturnMetadata | list | dict
    ) -> BodoDataFrame | BodoSeries | LazyArrowExtensionArray | list | dict:
        """Wrap the distributed return of a function into a BodoDataFrame, BodoSeries, or LazyArrowExtensionArray."""
        root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL

        def collect_func(res_id: str):
            self.worker_intercomm.bcast(CommandType.GATHER.value, root=root)
            self.worker_intercomm.bcast(res_id, root=root)
            return bodo.libs.distributed_api.gatherv(
                None, root=root, comm=self.worker_intercomm
            )

        def del_func(res_id: str):
            self.worker_intercomm.bcast(CommandType.DELETE_RESULT.value, root=root)
            self.worker_intercomm.bcast(res_id, root=root)

        if isinstance(distributed_return_metadata, list):
            return [
                self.wrap_distributed_result(d) for d in distributed_return_metadata
            ]
        if isinstance(distributed_return_metadata, dict):
            return {
                key: self.wrap_distributed_result(val)
                for key, val in distributed_return_metadata.items()
            }
        res_id = distributed_return_metadata.result_id
        nrows = distributed_return_metadata.nrows
        head = distributed_return_metadata.head
        index_data = None
        if distributed_return_metadata.index_data is not None:
            index_data = self.wrap_distributed_result(
                distributed_return_metadata.index_data
            )

        if isinstance(head, pd.DataFrame):
            lazy_manager = get_lazy_manager_class()(
                [],
                [],
                result_id=res_id,
                nrows=nrows,
                head=head._mgr,
                collect_func=collect_func,
                del_func=del_func,
                index_data=index_data,
            )
            return BodoDataFrame.from_lazy_mgr(lazy_manager, head)
        elif isinstance(head, pd.Series):
            lazy_manager = get_lazy_single_manager_class()(
                None,
                None,
                result_id=res_id,
                nrows=nrows,
                head=head._mgr,
                collect_func=collect_func,
                del_func=del_func,
                index_data=index_data,
            )
            return BodoSeries.from_lazy_mgr(lazy_manager, head)
        elif isinstance(head, ArrowExtensionArray):
            return LazyArrowExtensionArray(
                None,
                nrows=nrows,
                result_id=res_id,
                head=head,
                collect_func=collect_func,
                del_func=del_func,
            )
        else:
            raise Exception(f"Got unexpected distributed result type: {type(head)}")

    def _get_arg_metadata(self, arg, arg_name, is_replicated, dist_flags):
        """Replace argument with metadata for later bcast/scatter if it is a DataFrame,
        Series, Index or array type.
        Also adds scatter argument to distributed flags list to upate dispatcher later.

        Args:
            arg (Any): argument value
            arg_name (str): argument name
            is_replicated (bool): true if the argument is set to be replicated by user
            dist_flags (list[str]): list of distributed arguments to update

        Returns:
            ArgMetadata or Any: argument potentially replaced with ArgMetadata
        """
        # Arguments could be functions which fail in typeof.
        # See bodo/tests/test_series_part2.py::test_series_map_func_cases1
        # Similar to dispatcher argument handling:
        # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/dispatcher.py#L689
        try:
            data_type = bodo.typeof(arg)
        except ValueError:
            return arg

        if data_type is None:
            return arg

        if is_distributable_typ(data_type):
            dist_flags.append(arg_name)
            return ArgMetadata(is_replicated)

        # Send metadata to receive tables and reconstruct BodoSQLContext on workers
        # properly.
        if type(arg).__name__ == "BodoSQLContext":
            # Import bodosql lazily to avoid import overhead when not necessary
            from bodosql import BodoSQLContext

            assert isinstance(arg, BodoSQLContext), "invalid BodoSQLContext"
            table_metas = {
                tname: ArgMetadata(is_replicated) for tname in arg.tables.keys()
            }
            dist_flags.append(arg_name)
            return BodoSQLContextMetadata(table_metas, arg.catalog, arg.default_tz)

        # Handle distributed data inside tuples
        if isinstance(arg, tuple):
            return tuple(
                self._get_arg_metadata(val, arg_name, is_replicated, dist_flags)
                for val in arg
            )

        return arg

    def _send_arg_meta(self, arg: pt.Any, out_arg: pt.Any, bcast_root: int):
        """Send arguments that are replaced with metadata (bcast or scatter)

        Args:
            arg: input argument
            out_arg: input argument potentially replaced with metadata
            bcast_root: MPI bcast root rank
        """
        if isinstance(out_arg, ArgMetadata):
            if out_arg.is_broadcast:
                bodo.libs.distributed_api.bcast(
                    arg, root=bcast_root, comm=spawner.worker_intercomm
                )
            else:
                bodo.libs.distributed_api.scatterv(
                    arg, root=bcast_root, comm=spawner.worker_intercomm
                )

        # Send table DataFrames for BodoSQLContext
        if isinstance(out_arg, BodoSQLContextMetadata):
            for tname, tmeta in out_arg.tables.items():
                if tmeta.is_broadcast:
                    bodo.libs.distributed_api.bcast(
                        arg.tables[tname],
                        root=bcast_root,
                        comm=spawner.worker_intercomm,
                    )
                else:
                    bodo.libs.distributed_api.scatterv(
                        arg.tables[tname],
                        root=bcast_root,
                        comm=spawner.worker_intercomm,
                    )

        # Send distributed data nested inside tuples
        if isinstance(out_arg, tuple):
            for val, out_val in zip(arg, out_arg):
                self._send_arg_meta(val, out_val, bcast_root)

    def _send_args_update_dist_flags(
        self, dispatcher: "SubmitDispatcher", args, kwargs, bcast_root
    ):
        """Send function arguments from spawner to workers. DataFrame/Series/Index/array
        arguments are sent separately using broadcast or scatter (depending on flags).

        Also adds scattered arguments to the dispatchers distributed flags for proper
        compilation on the worker.

        Args:
            dispatcher (SubmitDispatcher): dispatcher to run on workers
            args (tuple[Any]): positional arguments
            kwargs (dict[str, Any]): keyword arguments
            bcast_root (int): root value for broadcast (MPI.ROOT on spawner)
        """
        param_names = list(numba.core.utils.pysignature(dispatcher.py_func).parameters)
        replicated = set(dispatcher.decorator_args.get("replicated", ()))
        dist_flags = []
        out_args = tuple(
            self._get_arg_metadata(
                arg, param_names[i], param_names[i] in replicated, dist_flags
            )
            for i, arg in enumerate(args)
        )
        out_kwargs = {
            name: self._get_arg_metadata(arg, name, name in replicated, dist_flags)
            for name, arg in kwargs.items()
        }
        # Using cloudpickle for arguments since there could be functions.
        # See bodo/tests/test_series_part2.py::test_series_map_func_cases1
        pickled_args = cloudpickle.dumps((out_args, out_kwargs))
        self.worker_intercomm.bcast(pickled_args, root=bcast_root)
        dispatcher.decorator_args["distributed_block"] = (
            dispatcher.decorator_args.get("distributed_block", []) + dist_flags
        )

        # Send DataFrame/Series/Index/array arguments (others are already sent)
        for arg, out_arg in itertools.chain(
            zip(args, out_args), zip(kwargs.values(), out_kwargs.values())
        ):
            self._send_arg_meta(arg, out_arg, bcast_root)

    def reset(self):
        """Destroy spawned processes"""
        try:
            debug_msg(self.logger, "Destroying spawned processes")
        except Exception:
            # We might not be able to log during process teardown
            pass
        bcast_root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL
        self.worker_intercomm.bcast(CommandType.EXIT.value, root=bcast_root)


spawner: Spawner | None = None


def get_spawner():
    """Get the global instance of Spawner, creating it if it isn't initialized"""
    global spawner
    if spawner is None:
        spawner = Spawner()
    return spawner


def destroy_spawner():
    """Destroy the global spawner instance.
    It is safe to call get_spawner to obtain a new Spawner instance after
    calling destroy_spawner."""
    global spawner
    if spawner is not None:
        spawner.reset()
        spawner = None


atexit.register(destroy_spawner)


def submit_func_to_workers(dispatcher: "SubmitDispatcher", *args, **kwargs):
    """Get the global spawner and submit `func` for execution"""
    spawner = get_spawner()
    return spawner.submit_func_to_workers(dispatcher, *args, **kwargs)


class SubmitDispatcher:
    """Pickleable wrapper that lazily sends a function and the arguments needed
    to compile to the workers"""

    def __init__(self, py_func, decorator_args):
        self.py_func = py_func
        self.decorator_args = decorator_args

    def __call__(self, *args, **kwargs):
        return submit_func_to_workers(self, *args, **kwargs)

    @classmethod
    def get_dispatcher(cls, py_func, decorator_args):
        # Instead of unpickling into a new SubmitDispatcher, we call bodo.jit to
        # return the real dispatcher
        decorator = bodo.jit(**decorator_args)
        return decorator(py_func)

    def __reduce__(self):
        # Pickle this object by pickling the underlying function (which is
        # guaranteed to have the extra properties necessary to build the actual
        # dispatcher via bodo.jit on the worker side)
        return SubmitDispatcher.get_dispatcher, (self.py_func, self.decorator_args)
