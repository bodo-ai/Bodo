# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""Spawner-worker compilation implementation"""

import atexit
import contextlib
import dis
import itertools
import logging
import os
import sys
import time

import cloudpickle
import numba
import psutil

import bodo
import bodo.user_logging
from bodo.mpi4py import MPI
from bodo.submit.utils import CommandType, poll_for_barrier
from bodo.utils.utils import is_array_typ

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

        # If BODO_NUM_WORKERS is set, spawn that many workers. Else,
        # we will spawn as many workers as there are physical cores
        # on this machine.
        n_pes = 2
        if n_pes_env := os.environ.get("BODO_NUM_WORKERS"):
            n_pes = int(n_pes_env)
        elif cpu_count := psutil.cpu_count(logical=False):
            n_pes = cpu_count
        self.logger.debug(f"Trying to spawn {n_pes} workers...")
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
        self.logger.debug(f"Spawned {n_pes} workers in {(time.monotonic()-t0):0.4f}s")
        self.exec_intercomm_addr = MPI._addressof(self.worker_intercomm)

    def collect_results(self):
        # TODO: result collection needs modification to gatherv API, then we can
        # call bodo.gatherv on the intercomm
        return None

    def submit_func_to_workers(self, func, *args, **kwargs):
        """Send func to be compiled and executed on spawned process"""

        # Check that the only return is a "return None"
        pyfunc_bytecode = dis.Bytecode(func.py_func)
        for inst in pyfunc_bytecode:
            if inst.opname == "RETURN_VALUE" or (
                inst.opname == "RETURN_CONST" and inst.argrepr != "None"
            ):
                raise NotImplementedError(
                    "Functions that return values are not supported by submit_jit yet"
                )

        # Send pyfunc:
        pickled_func = cloudpickle.dumps(func.py_func)
        bcast_root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL
        self.worker_intercomm.bcast(CommandType.EXEC_FUNCTION.value, bcast_root)
        self.worker_intercomm.bcast(pickled_func, root=bcast_root)

        # Send arguments
        self._send_func_args(func, args, kwargs, bcast_root)

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

        return self.collect_results()

    def _get_arg_metadata(self, arg_name, arg, replicated):
        """Replace argument with metadata for later bcast/scatter if it is a DataFrame,
        Series, Index or array type.

        Args:
            arg_name (str): argument name
            arg (Any): argument value
            replicated (set[str]): set of replicated arguments (based on flags)

        Returns:
            ArgMetadata or Any: argument potentially replaced with ArgMetadata
        """
        data_type = bodo.typeof(arg)

        if is_array_typ(data_type, True) or isinstance(
            data_type, (bodo.DataFrameType, bodo.TableType)
        ):
            return ArgMetadata(arg_name in replicated)

        # Send metadata to receive tables and reconstruct BodoSQLContext on workers
        # properly.
        if type(arg).__name__ == "BodoSQLContext":
            # Import bodosql lazily to avoid import overhead when not necessary
            from bodosql import BodoSQLContext

            assert isinstance(arg, BodoSQLContext), "invalid BodoSQLContext"
            table_metas = {
                tname: ArgMetadata(arg_name in replicated)
                for tname in arg.tables.keys()
            }
            return BodoSQLContextMetadata(table_metas, arg.catalog, arg.default_tz)

        return arg

    def _send_func_args(self, func, args, kwargs, bcast_root):
        """Send function arguments from spawner to workers. DataFrame/Series/Index/array
        arguments are sent separately using broadcast or scatter (depending on flags).

        Args:
            func (CPUDispatcher): dispatcher to run on workers
            args (tuple[Any]): positional arguments
            kwargs (dict[str, Any]): keyword arguments
            bcast_root (int): root value for broadcast (MPI.ROOT on spawner)
        """
        param_names = list(numba.core.utils.pysignature(func.py_func).parameters)
        replicated = set(func.submit_jit_args.get("replicated", ()))
        out_args = tuple(
            self._get_arg_metadata(param_names[i], arg, replicated)
            for i, arg in enumerate(args)
        )
        out_kwargs = {
            name: self._get_arg_metadata(name, arg, replicated)
            for name, arg in kwargs.items()
        }
        self.worker_intercomm.bcast((out_args, out_kwargs), root=bcast_root)

        # Send DataFrame/Series/Index/array arguments
        for arg, out_arg in itertools.chain(
            zip(args, out_args), zip(kwargs.values(), out_kwargs.values())
        ):
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

    def reset(self):
        """Destroy spawned processes"""
        try:
            self.logger.debug("Destroying spawned processes")
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


def submit_func_to_workers(func, *args, **kwargs):
    """Get the global spawner and submit `func` for execution"""
    spawner = get_spawner()
    spawner.submit_func_to_workers(func, *args, **kwargs)


class SubmitDispatcher(object):
    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def __call__(self, *args, **kwargs):
        func = self.dispatcher
        submit_func_to_workers(func, *args, **kwargs)

    def __getstate__(self):
        return self.dispatcher.py_func

    def __setstate__(self, state):
        pyfunc = state
        # Extract JIT args from the function object to reconstruct the correct
        # dispatcher
        assert pyfunc.__dict__.get("is_submit_jit")
        decorator_args = pyfunc.__dict__.get("submit_jit_args")
        decorator = submit_jit(**decorator_args)
        rebuild_submit_dispatcher = decorator(pyfunc)
        self.dispatcher = rebuild_submit_dispatcher.dispatcher


def submit_dispatcher_wrapper(numba_jit_wrapper, **kwargs):  # pragma: no cover
    """Wrapper that stores the JIT wrapper to be passed a real function later.
    This is the case when we run code that invokes `submit_jit` without a
    function argument (usually because additional arguments were suppplied),
    e.g.:
      @submit_jit(cache=True)
      def fn():
        ...
    """

    def _wrapper(pyfunc):
        # Add JIT args onto the function so that the worker can construct the
        # appropriate dispatcher using these args
        pyfunc.is_submit_jit = True
        pyfunc.submit_jit_args = kwargs
        dispatcher = numba_jit_wrapper(pyfunc)
        return SubmitDispatcher(dispatcher)

    return _wrapper


def submit_jit(signature_or_function=None, pipeline_class=None, **options):
    """Spawn mode execution entry point decorator. Functions decorated with submit_jit
    will be compiled and executed on a spawned group."""

    # Spawner distributes arguments by default (with scatterv)
    options["all_args_distributed_block"] = True

    if signature_or_function:
        # Add JIT args onto the function so that the worker can construct the
        # appropriate dispatcher using these args
        signature_or_function.is_submit_jit = True
        signature_or_function.submit_jit_args = {**options}
        signature_or_function.submit_jit_args["pipeline_class"] = pipeline_class

    numba_jit = bodo.jit(
        signature_or_function, pipeline_class=pipeline_class, **options
    )
    # When options are passed, this function is called with
    # signature_or_function==None, so numba.jit doesn't return a Dispatcher
    # object. it returns a decorator ("_jit.<locals>.wrapper") to apply
    # to the Python function, and we need to wrap that around our own
    # decorator
    if isinstance(numba_jit, numba.core.dispatcher._DispatcherBase):
        f = SubmitDispatcher(numba_jit)
    else:
        kwargs = options
        if pipeline_class is not None:
            kwargs["pipeline_class"] = pipeline_class
        f = submit_dispatcher_wrapper(numba_jit, **kwargs)
    return f
