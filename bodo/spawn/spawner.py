"""Spawner-worker compilation implementation"""

import atexit
import contextlib
import inspect
import itertools
import linecache
import logging
import os
import signal
import socket
import sys
import time
import typing as pt
from collections import deque

import cloudpickle
import numba
import pandas as pd
import psutil
from numba.core import types
from pandas.core.arrays.arrow.array import ArrowExtensionArray

import bodo
import bodo.user_logging
from bodo.mpi4py import MPI
from bodo.pandas import (
    BodoDataFrame,
    BodoSeries,
    LazyArrowExtensionArray,
    LazyMetadata,
)
from bodo.pandas.lazy_wrapper import BodoLazyWrapper
from bodo.spawn.utils import (
    ArgMetadata,
    CommandType,
    debug_msg,
    poll_for_barrier,
)
from bodo.utils.utils import is_distributable_typ

# Reference to BodoSQLContext class to be lazily initialized if BodoSQLContext
# is detected
BodoSQLContextCls = None

env_var_prefix = (
    "BODO_",
    "AWS_",
    "AZURE_",
    "LD_",
    "PYTHONPATH",
    "__BODOSQL",
    "MINIO_",
    "CLASSPATH",
    "OMP",
    "MKL",
    "OPENBLAS",
    "NUMBA",
)


@contextlib.contextmanager
def no_stdin():
    """Temporarily close stdin and execute a block of code"""
    # Save a refence to the original stdin
    stdin_dup = os.dup(0)
    # Close stdin
    os.close(0)
    # open /dev/null as fd 0
    nullfd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nullfd, 0)
    try:
        yield
    finally:
        # Restore the saved fd
        os.dup2(stdin_dup, 0)


def get_num_workers():
    """Returns the number of workers to spawn.

    If BODO_NUM_WORKERS is set, spawn that many workers.
    If MPI_UNIVERSE_SIZE is set, spawn that many workers.
    Else, fallback to spawning as
    many workers as there are physical cores on this machine."""
    n_pes = 2
    with no_stdin():
        if n_pes_env := os.environ.get("BODO_NUM_WORKERS"):
            n_pes = int(n_pes_env)
        elif universe_size := MPI.COMM_WORLD.Get_attr(MPI.UNIVERSE_SIZE):
            n_pes = universe_size
        elif cpu_count := psutil.cpu_count(logical=False):
            n_pes = cpu_count
    return n_pes


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
    destroyed: bool

    def __init__(self):
        self.logger = bodo.user_logging.get_current_bodo_verbose_logger()
        self.destroyed = False

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
            command, args = self._get_spawn_command_args()

            # run python with -u to prevent STDOUT from buffering
            self.worker_intercomm = self.comm_world.Spawn(
                # get the same python executable that is currently running
                command,
                args,
                n_pes,
                MPI.INFO_NULL,
                0,
                errcodes,
            )

            # Send PID of spawner to worker
            self.worker_intercomm.bcast(os.getpid(), self.bcast_root)
            self.worker_intercomm.send(socket.gethostname(), dest=0)
        debug_msg(
            self.logger, f"Spawned {n_pes} workers in {(time.monotonic() - t0):0.4f}s"
        )
        self.exec_intercomm_addr = MPI._addressof(self.worker_intercomm)

        # Make sure worker output is displayed in Jupyter notebooks on Windows
        self._init_win_jupyter()

        # A queue of del tasks to send delete command to workers.
        # Necessary since garbage collection can be triggered at any time during
        # execution of commands, leading to invalid data being broadcast to workers.
        self._del_queue = deque()
        self._is_running = False

    def _get_spawn_command_args(self) -> tuple[str, list[str]]:
        """
        Get the platform-dependent command to launch spawned processes.

        Returns:
            tuple[str, list[str]]: The command to run and a list of arguments.
        """
        py_args = ["-u", "-m", "bodo.spawn.worker"]

        if sys.platform == "win32":
            # TODO [BSE-4553] set logging level on Windows / general environment management
            return sys.executable, py_args

        else:
            # Send spawner log level to workers
            environ_args = [
                f"BODO_WORKER_VERBOSE_LEVEL={bodo.user_logging.get_verbose_level()}"
            ]
            if "BODO_DYLD_INSERT_LIBRARIES" in os.environ:
                environ_args.append(
                    f"DYLD_INSERT_LIBRARIES={os.environ['BODO_DYLD_INSERT_LIBRARIES']}"
                )

            return "env", environ_args + [sys.executable] + py_args

    @property
    def bcast_root(self):
        """MPI bcast root rank"""
        return MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL

    def _recv_output(self, output_is_distributed: bool | list[bool]):
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

    def _recv_updated_args(
        self,
        args: tuple[pt.Any],
        args_meta: tuple[ArgMetadata | None, ...],
        kwargs: dict[str, pt.Any],
        kwargs_meta: dict[str, ArgMetadata | None],
    ):
        """Receive updated arguments from workers and update the original arguments to match.
        Only does anything for lazy arguments."""

        def _recv_updated_arg(arg, arg_meta):
            if isinstance(arg, tuple):
                assert isinstance(arg_meta, tuple)
                for i in range(len(arg)):
                    _recv_updated_arg(arg[i], arg_meta[i])

            if isinstance(arg_meta, ArgMetadata) and arg_meta is ArgMetadata.LAZY:
                return_meta = self.worker_intercomm.recv(source=0)
                arg.update_from_lazy_metadata(return_meta)

        for i in range(len(args)):
            _recv_updated_arg(args[i], args_meta[i])
        for name in kwargs.keys():
            _recv_updated_arg(kwargs[name], kwargs_meta[name])

    def _send_env_var(self, bcast_root, propagate_env):
        """Send environment variables from spawner to workers.

        Args:
            bcast_root (int): root value for broadcast (MPI.ROOT on spawner)
            propagate_env (list[str]): additional env vars to propagate"""
        new_env_var = {}
        for var in os.environ:
            # DYLD_INSERT_LIBRARIES can be difficult to propogate to child
            # process. e.g.:
            # https://stackoverflow.com/questions/43941322/dyld-insert-libraries-ignored-when-calling-application-through-bash
            # So for now, we use BODO_DYLD_INSERT_LIBRARIES as a way to
            # inform the spawner to set the variable for the child processes
            if var == "BODO_DYLD_INSERT_LIBRARIES":
                new_env_var["DYLD_INSERT_LIBRARIES"] = os.environ[var]
            elif var.startswith(env_var_prefix) or var in propagate_env:
                new_env_var[var] = os.environ[var]
        self.worker_intercomm.bcast(new_env_var, bcast_root)
        self.worker_intercomm.bcast(propagate_env, bcast_root)

    def submit_func_to_workers(
        self, dispatcher: "SpawnDispatcher", propagate_env, *args, **kwargs
    ):
        """Send func to be compiled and executed on spawned process"""
        assert not self._is_running, "submit_func_to_workers: already running"
        self._is_running = True

        if sys.platform != "win32":
            # Install a signal handler for SIGUSR1 as a notification mechanism
            # to determine when the worker has finished execution. We use
            # signals instead of MPI barriers to avoid consuming CPU resources
            # on the spawner.
            signaled = False

            def handler(*args, **kwargs):
                nonlocal signaled
                signaled = True

            signal.signal(signal.SIGUSR1, handler)

        debug_msg(self.logger, "submit_func_to_workers")
        self.worker_intercomm.bcast(CommandType.EXEC_FUNCTION.value, self.bcast_root)

        # Send environment variables
        self._send_env_var(self.bcast_root, propagate_env)

        # Send arguments and update dispatcher distributed flags for arguments
        args_meta, kwargs_meta = self._send_args_update_dist_flags(
            dispatcher, args, kwargs
        )

        # Send dispatcher
        pickled_func = cloudpickle.dumps(dispatcher)
        self.worker_intercomm.bcast(pickled_func, root=self.bcast_root)
        debug_msg(self.logger, "submit_func_to_workers - wait for results")

        if sys.platform == "win32":
            # Signals work differently on Windows, so use an async MPI barrier
            # instead
            # NOTE: polling req.Test() manually seems to hang on Windows with Intel MPI
            poll_for_barrier(self.worker_intercomm, None)
        else:
            # Wait for execution to finish
            while not signaled:
                # wait for any signal. SIGUSR1's handler will set signaled to
                # True, any other signals can be ignored here (the
                # appropriate/default handler for any signal will still be
                # invoked)
                signal.pause()
            # TODO(aneesh) create a context manager for restoring signal
            # disposition Restore SIGUSR1's default handler
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)

        gather_root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL
        caught_exceptions = self.worker_intercomm.gather(None, root=gather_root)

        assert caught_exceptions is not None
        if any(caught_exceptions):
            self._is_running = False
            self._run_del_queue()
            types = {type(excep) for excep in caught_exceptions}
            msgs = {
                str(excep) if excep is not None else None for excep in caught_exceptions
            }
            all_ranks_failed = all(caught_exceptions)
            if all_ranks_failed and len(types) == 1 and len(msgs) == 1:
                excep = caught_exceptions[0]
                raise excep
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

        self._recv_updated_args(args, args_meta, kwargs, kwargs_meta)

        self._is_running = False
        self._run_del_queue()
        return res

    def wrap_distributed_result(
        self,
        lazy_metadata: LazyMetadata | list | dict | tuple,
    ) -> BodoDataFrame | BodoSeries | LazyArrowExtensionArray | list | dict | tuple:
        """Wrap the distributed return of a function into a BodoDataFrame, BodoSeries, or LazyArrowExtensionArray."""
        root = MPI.ROOT if self.comm_world.Get_rank() == 0 else MPI.PROC_NULL

        def collect_func(res_id: str):
            # collect is sometimes triggered during receive (e.g. for unsupported types
            # like IntervalIndex) so we may be in the middle of function execution
            # already.
            initial_running = self._is_running
            if not initial_running:
                self._is_running = True
            self.worker_intercomm.bcast(CommandType.GATHER.value, root=root)
            self.worker_intercomm.bcast(res_id, root=root)
            res = bodo.libs.distributed_api.gatherv(
                None, root=root, comm=self.worker_intercomm
            )
            if not initial_running:
                self._is_running = False
                self._run_del_queue()
            return res

        def del_func(res_id: str):
            self._del_queue.append(res_id)
            self._run_del_queue()

        if isinstance(lazy_metadata, list):
            return [self.wrap_distributed_result(d) for d in lazy_metadata]
        if isinstance(lazy_metadata, dict):
            return {
                key: self.wrap_distributed_result(val)
                for key, val in lazy_metadata.items()
            }
        if isinstance(lazy_metadata, tuple):
            return tuple([self.wrap_distributed_result(d) for d in lazy_metadata])
        head = lazy_metadata.head
        if lazy_metadata.index_data is not None and isinstance(
            lazy_metadata.index_data, (LazyMetadata, list, dict, tuple)
        ):
            lazy_metadata.index_data = self.wrap_distributed_result(
                lazy_metadata.index_data
            )

        if isinstance(head, pd.DataFrame):
            return BodoDataFrame.from_lazy_metadata(
                lazy_metadata, collect_func, del_func
            )
        elif isinstance(head, pd.Series):
            return BodoSeries.from_lazy_metadata(lazy_metadata, collect_func, del_func)
        elif isinstance(head, ArrowExtensionArray):
            return LazyArrowExtensionArray.from_lazy_metadata(
                lazy_metadata, collect_func, del_func
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
            ArgMetadata or None: ArgMetadata if argument is distributable, None otherwise
        """
        dist_comm_meta = ArgMetadata.BROADCAST if is_replicated else ArgMetadata.SCATTER
        if isinstance(arg, BodoLazyWrapper):
            if arg._lazy:
                return ArgMetadata.LAZY
            dist_flags.append(arg_name)
            return dist_comm_meta

        # Handle distributed data inside tuples
        if isinstance(arg, tuple):
            return tuple(
                self._get_arg_metadata(val, arg_name, is_replicated, dist_flags)
                for val in arg
            )

        # Arguments could be functions which fail in typeof.
        # See bodo/tests/test_series_part2.py::test_series_map_func_cases1
        # Similar to dispatcher argument handling:
        # https://github.com/numba/numba/blob/53e976f1b0c6683933fa0a93738362914bffc1cd/numba/core/dispatcher.py#L689
        try:
            data_type = bodo.typeof(arg)
        except ValueError:
            return None

        if data_type is None:
            return None

        if is_distributable_typ(data_type):
            dist_flags.append(arg_name)
            return dist_comm_meta

        # Send metadata to receive tables and reconstruct BodoSQLContext on workers
        # properly.
        if type(arg).__name__ == "BodoSQLContext":
            # Import bodosql lazily to avoid import overhead when not necessary
            from bodosql import BodoSQLContext, TablePath

            assert isinstance(arg, BodoSQLContext), "invalid BodoSQLContext"
            table_metas = {
                tname: table if isinstance(table, TablePath) else dist_comm_meta
                for tname, table in arg.tables.items()
            }

            # BodoSQLContext without table data is treated as replicated in distributed
            # analysis
            if len(table_metas) == 0:
                return None

            dist_flags.append(arg_name)
            return BodoSQLContextMetadata(table_metas, arg.catalog, arg.default_tz)

        return None

    def _send_arg_meta(self, arg: pt.Any, arg_meta: ArgMetadata | None):
        """Send arguments that are replaced with metadata (bcast or scatter)

        Args:
            arg: input argument
            out_arg: input argument metadata
        """
        if isinstance(arg_meta, ArgMetadata):
            match arg_meta:
                case ArgMetadata.BROADCAST:
                    bodo.libs.distributed_api.bcast(
                        arg, root=self.bcast_root, comm=spawner.worker_intercomm
                    )
                case ArgMetadata.SCATTER:
                    bodo.libs.distributed_api.scatterv(
                        arg, root=self.bcast_root, comm=spawner.worker_intercomm
                    )
                case ArgMetadata.LAZY:
                    spawner.worker_intercomm.bcast(
                        arg._get_result_id(), root=self.bcast_root
                    )

        # Send table DataFrames for BodoSQLContext
        if isinstance(arg_meta, BodoSQLContextMetadata):
            for tname, tmeta in arg_meta.tables.items():
                if tmeta is ArgMetadata.BROADCAST:
                    bodo.libs.distributed_api.bcast(
                        arg.tables[tname],
                        root=self.bcast_root,
                        comm=spawner.worker_intercomm,
                    )
                elif tmeta is ArgMetadata.SCATTER:
                    bodo.libs.distributed_api.scatterv(
                        arg.tables[tname],
                        root=self.bcast_root,
                        comm=spawner.worker_intercomm,
                    )

        # Send distributed data nested inside tuples
        if isinstance(arg_meta, tuple):
            for val, out_val in zip(arg, arg_meta):
                self._send_arg_meta(val, out_val)

    def _send_args_update_dist_flags(
        self, dispatcher: "SpawnDispatcher", args, kwargs
    ) -> tuple[tuple[ArgMetadata | None, ...], dict[str, ArgMetadata | None]]:
        """Send function arguments from spawner to workers. DataFrame/Series/Index/array
        arguments are sent separately using broadcast or scatter (depending on flags).

        Also adds scattered arguments to the dispatchers distributed flags for proper
        compilation on the worker.

        Args:
            dispatcher (SpawnDispatcher): dispatcher to run on workers
            args (tuple[Any]): positional arguments
            kwargs (dict[str, Any]): keyword arguments
        """
        param_names = list(numba.core.utils.pysignature(dispatcher.py_func).parameters)
        replicated = set(dispatcher.decorator_args.get("replicated", ()))
        dist_flags = []
        args_meta = tuple(
            self._get_arg_metadata(
                arg, param_names[i], param_names[i] in replicated, dist_flags
            )
            for i, arg in enumerate(args)
        )
        kwargs_meta = {
            name: self._get_arg_metadata(arg, name, name in replicated, dist_flags)
            for name, arg in kwargs.items()
        }

        def compute_args_to_send(arg, arg_meta):
            if isinstance(arg, tuple):
                return tuple(
                    compute_args_to_send(unwrapped_arg, unwrapped_arg_meta)
                    for unwrapped_arg, unwrapped_arg_meta in zip(arg, arg_meta)
                )
            if arg_meta is None:
                return arg
            return arg_meta

        args_to_send = [
            compute_args_to_send(arg, arg_meta)
            for arg, arg_meta in zip(args, args_meta)
        ]
        kwargs_to_send = {
            name: compute_args_to_send(arg, arg_meta)
            for name, (arg, arg_meta) in zip(
                kwargs.keys(), zip(kwargs.values(), kwargs_meta.values())
            )
        }

        # Using cloudpickle for arguments since there could be functions.
        # See bodo/tests/test_series_part2.py::test_series_map_func_cases1
        pickled_args = cloudpickle.dumps((args_to_send, kwargs_to_send))
        self.worker_intercomm.bcast(pickled_args, root=self.bcast_root)
        dispatcher.decorator_args["distributed_block"] = (
            dispatcher.decorator_args.get("distributed_block", []) + dist_flags
        )
        # Send DataFrame/Series/Index/array arguments (others are already sent)
        for arg, arg_meta in itertools.chain(
            zip(args, args_meta), zip(kwargs.values(), kwargs_meta.values())
        ):
            self._send_arg_meta(arg, arg_meta)
        return args_meta, kwargs_meta

    def _init_win_jupyter(self):
        """Make sure worker output is displayed in Jupyter notebooks on Windows.
        Windows child processes don't inherit file descriptors from the parent,
        so Jupyter's normal output routing doesn't work.
        This creates a socket using ZMQ to receive output from workers and print it.
        A separate thread is used to receive messages and print them.
        Currently only works on single-node Windows setups.

        Some relevant links:
        https://discourse.jupyter.org/t/jupyterlab-no-longer-allows-ouput-to-be-redirected-to-stdout/11535/2
        https://github.com/jupyterlab/jupyterlab/issues/9668
        https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
        https://stackoverflow.com/questions/7714868/multiprocessing-how-can-i-%ca%80%e1%b4%87%ca%9f%c9%aa%e1%b4%80%ca%99%ca%9f%ca%8f-redirect-stdout-from-a-child-process
        https://stackoverflow.com/questions/23947281/python-multiprocessing-redirect-stdout-of-a-child-process-to-a-tkinter-text
        """

        # Skip if not in Jupyter or not on Windows
        if not bodo.utils.utils.is_jupyter_on_windows():
            self.worker_output_thread = None
            return

        import errno
        import threading

        import zmq

        context = zmq.Context()
        out_socket = context.socket(zmq.PULL)
        max_attempts = 100

        # Similar to:
        # https://github.com/ipython/ipykernel/blob/dab3b39e3f1e0258d99b189867d8f2e2d36c976e/ipykernel/kernelapp.py#L252
        try:
            win_in_use = errno.WSAEADDRINUSE  # type:ignore[attr-defined]
        except AttributeError:
            win_in_use = None

        for attempt in range(max_attempts):
            try:
                port = out_socket.bind_to_random_port("tcp://127.0.0.1")
            except zmq.ZMQError as ze:
                # Raise if we have any error not related to socket binding
                if ze.errno != errno.EADDRINUSE and ze.errno != win_in_use:
                    raise
                if attempt == max_attempts - 1:
                    raise

        self.worker_intercomm.bcast(port, self.bcast_root)

        def worker_output_thread_func():
            """Thread that receives all worker outputs and prints them."""

            # ipykernel redirects the output of threads to the first cell using
            # _thread_to_parent_header. We need to remove this thread to make sure
            # prints apprear in the correct cell.
            # https://github.com/ipython/ipykernel/blob/dab3b39e3f1e0258d99b189867d8f2e2d36c976e/ipykernel/ipkernel.py#L766
            # https://github.com/ipython/ipykernel/blob/dab3b39e3f1e0258d99b189867d8f2e2d36c976e/ipykernel/iostream.py#L525
            sys.stdout._thread_to_parent_header.pop(
                threading.current_thread().ident, None
            )
            sys.stderr._thread_to_parent_header.pop(
                threading.current_thread().ident, None
            )

            while True:
                message = out_socket.recv_string()
                is_stderr = message.startswith("1")
                message = message[1:]
                print(message, end="", file=sys.stderr if is_stderr else sys.stdout)

        self.worker_output_thread = threading.Thread(
            target=worker_output_thread_func, daemon=True
        )
        self.worker_output_thread.start()

    def _run_del_queue(self):
        """Run delete tasks in the queue if no other tasks are running."""
        if not self._is_running and self._del_queue and not self.destroyed:
            self._is_running = True
            res_id = self._del_queue.popleft()
            self.worker_intercomm.bcast(
                CommandType.DELETE_RESULT.value, root=self.bcast_root
            )
            self.worker_intercomm.bcast(res_id, root=self.bcast_root)
            self._is_running = False
            self._run_del_queue()

    def set_config(self, name: str, value: pt.Any):
        """Set configuration value on workers"""
        assert not self._is_running, "set_config: already running"
        self._is_running = True
        spawner.worker_intercomm.bcast(CommandType.SET_CONFIG.value, self.bcast_root)
        spawner.worker_intercomm.bcast((name, value), self.bcast_root)
        self._is_running = False
        self._run_del_queue()

    def register_type(self, type_name: str, type_value: types.Type):
        """Register a new type on workers"""
        assert not self._is_running, "register_type: already running"
        self._is_running = True
        spawner.worker_intercomm.bcast(CommandType.REGISTER_TYPE.value, self.bcast_root)
        spawner.worker_intercomm.bcast((type_name, type_value), self.bcast_root)
        self._is_running = False
        self._run_del_queue()

    def reset(self):
        """Destroy spawned processes"""
        assert not self._is_running, "reset: already running"
        self._is_running = True
        try:
            debug_msg(self.logger, "Destroying spawned processes")
        except Exception:
            # We might not be able to log during process teardown
            pass
        self.worker_intercomm.bcast(CommandType.EXIT.value, root=self.bcast_root)
        self._is_running = False
        self.destroyed = True


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


def submit_func_to_workers(
    dispatcher: "SpawnDispatcher", propagate_env, *args, **kwargs
):
    """Get the global spawner and submit `func` for execution"""
    spawner = get_spawner()
    return spawner.submit_func_to_workers(dispatcher, propagate_env, *args, **kwargs)


class SpawnDispatcher:
    """Pickleable wrapper that lazily sends a function and the arguments needed
    to compile to the workers"""

    def __init__(self, py_func, decorator_args):
        self.py_func = py_func
        self.decorator_args = decorator_args
        # Extra globals to pickle (used for BodoSQL globals that are not visible to
        # cloudpickle, e.g. inside CASE implementation string)
        self.extra_globals = {}

    def __call__(self, *args, **kwargs):
        return submit_func_to_workers(
            self, self.decorator_args.get("propagate_env", []), *args, **kwargs
        )

    @classmethod
    def get_dispatcher(cls, py_func, decorator_args, extra_globals, linecache_entry):
        # Instead of unpickling into a new SpawnDispatcher, we call bodo.jit to
        # return the real dispatcher
        py_func.__globals__.update(extra_globals)
        decorator = bodo.jit(**decorator_args)
        if linecache_entry:
            linecache.cache[linecache_entry[0]] = linecache_entry[1]
        return decorator(py_func)

    def _get_ipython_cache_entry(self):
        """Get IPython cell entry in linecache for the function to send to workers,
        which is necessary for inspect.getsource to work (used in caching).
        """
        linecache_entry = None
        source_path = inspect.getfile(self.py_func)
        if source_path.startswith("<ipython-") or os.path.basename(
            os.path.dirname(source_path)
        ).startswith("ipykernel_"):
            linecache_entry = (source_path, linecache.cache[source_path])

        return linecache_entry

    def __reduce__(self):
        # Pickle this object by pickling the underlying function (which is
        # guaranteed to have the extra properties necessary to build the actual
        # dispatcher via bodo.jit on the worker side)
        return SpawnDispatcher.get_dispatcher, (
            self.py_func,
            self.decorator_args,
            self.extra_globals,
            self._get_ipython_cache_entry(),
        )

    def add_extra_globals(self, glbls):
        """Add extra globals to be pickled (used for BodoSQL globals that are not visible to
        cloudpickle, e.g. inside CASE implementation strings)
        """
        self.extra_globals.update(glbls)


# Raise error for VS Code notebooks if jupyter.disableZMQSupport is not set to avoid
# VS Code crashes (during restart, etc).
# See https://github.com/microsoft/vscode-jupyter/issues/16283

vs_code_nb_msg = """
VS Code has a problem running MPI (and therefore Bodo) inside Jupyter notebooks.
To fix it, please turn off VS Code Jupyter extension's ZMQ to use Bodo in VS Code
notebooks. Add `"jupyter.disableZMQSupport": true,` to VS Code settings and restart
VS Code (e.g., using
"Preferences: Open User Settings (JSON)" in Command Pallette (Ctrl/CMD+Shift+P),
see https://code.visualstudio.com/docs/getstarted/settings#_user-settings).
"""

# Detect VS Code Jupyter extension using this environment variable:
# https://github.com/microsoft/vscode-jupyter/blob/f80bf701a710328b20c5931d621e8d83813055ea/src/kernels/raw/launcher/kernelEnvVarsService.node.ts#L134
# Detect Jupyter session (no ZMQ) using JPY_SESSION_NAME
if (
    "PYDEVD_IPYTHON_COMPATIBLE_DEBUGGING" in os.environ
    and "JPY_SESSION_NAME" not in os.environ
):
    raise bodo.utils.typing.BodoError(vs_code_nb_msg)
