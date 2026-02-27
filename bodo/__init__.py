"""
Top-level init file for bodo package
"""


def _global_except_hook(exctype, value, traceback):
    """Custom excepthook function that replaces sys.excepthook (see sys.excepthook
    documentation for more details on its API)
    Our function calls MPI_Abort() to force all processes to abort *if not all
    processes raise the same unhandled exception*
    """

    import sys
    import time
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Calling MPI_Abort() aborts the program with a non-zero exit code and
    # MPI will print a message such as
    # "application called MPI_Abort(MPI_COMM_WORLD, 1) - process 0"
    # Therefore, we only want to call MPI_Abort if there is going to be a hang
    # (for example when some processes but not all exit with an unhandled
    # exception). To detect a hang, we wait on a non-blocking barrier for a
    # specified amount of time.
    HANG_TIMEOUT = 3.0
    is_hang = True
    req = comm.Ibarrier()
    start = time.time()
    while time.time() - start < HANG_TIMEOUT:
        time.sleep(0.1)
        if req.Test():
            # everyone reached the barrier before the timeout, so there is no hang
            is_hang = False
            break

    try:
        global _orig_except_hook
        # first we print the exception with the original excepthook
        if _orig_except_hook:
            _orig_except_hook(exctype, value, traceback)
        else:
            sys.__excepthook__(exctype, value, traceback)
        if is_hang:
            # if we are aborting, print a message
            sys.stderr.write(
                "\n*****************************************************\n"
            )
            sys.stderr.write(f"   Uncaught exception detected on rank {rank}. \n")
            sys.stderr.write("   Calling MPI_Abort() to shut down MPI...\n")
            sys.stderr.write("*****************************************************\n")
            sys.stderr.write("\n")
        sys.stderr.flush()
    finally:
        if is_hang:
            try:
                from bodo.spawn.worker_state import is_worker

                if is_worker():
                    MPI.COMM_WORLD.Get_parent().Abort(1)
                else:
                    MPI.COMM_WORLD.Abort(1)
            except:
                sys.stderr.write(
                    "*****************************************************\n"
                )
                sys.stderr.write(
                    "We failed to stop MPI, this process will likely hang.\n"
                )
                sys.stderr.write(
                    "*****************************************************\n"
                )
                sys.stderr.flush()
                raise


import sys

# Add a global hook function that captures unhandled exceptions.
# The function calls MPI_Abort() to force all processes to abort *if not all
# processes raise the same unhandled exception*
_orig_except_hook = sys.excepthook
sys.excepthook = _global_except_hook


class BodoWarning(Warning):
    """
    Warning class for Bodo-related potential issues such as prevention of
    parallelization by unsupported functions.
    """


# ------------------------------ Version Import ------------------------------
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bodo")
except PackageNotFoundError:
    # package is not installed
    pass


# ----------------------------- Streaming Config -----------------------------
import os
import platform

# Flag to track if we should use the streaming plan in BodoSQL.
bodosql_use_streaming_plan = os.environ.get("BODO_STREAMING_ENABLED", "1") != "0"
# Number of rows to process at once for BodoSQL. This is used to test
# the streaming plan in BodoSQL on the existing unit tests that may only
# have one batch worth of data.
# NOTE: should be the same as the default value for STREAMING_BATCH_SIZE in _shuffle.h
bodosql_streaming_batch_size = int(os.environ.get("BODO_STREAMING_BATCH_SIZE", 32768))
# How many iterations to run a streaming loop for before synchronizing
# -1 means it's adaptive and is updated based on shuffle buffer sizes
stream_loop_sync_iters = int(os.environ.get("BODO_STREAM_LOOP_SYNC_ITERS", -1))
# Default value for above to use if not specified by user
# NOTE: should be the same as DEFAULT_SYNC_ITERS in _shuffle.h
default_stream_loop_sync_iters = 1000
# If BodoSQL encounters a Snowflake Table that is also an Iceberg table should
# it attempt to read it as an Iceberg table?
enable_snowflake_iceberg = os.environ.get("BODO_ENABLE_SNOWFLAKE_ICEBERG", "1") != "0"
# Flag used to enable reading TIMESTAMP_TZ as its own type instead of as an alias
# for TIMESTAMP_LTZ. (will be removed once TIMESTAMP_TZ support is complete)
enable_timestamp_tz = os.environ.get("BODO_ENABLE_TIMESTAMP_TZ", "1") != "0"
# When applying multiple filters in a single call to runtime_join_filter, materialization
# occurs after each filter unless the table has at least this many variable-length type
# columns at which point materialization occurs just once after all filters have been applied.
runtime_join_filters_copy_threshold = os.environ.get(
    "BODO_RUNTIME_JOIN_FILTERS_COPY_THRESHOLD", 1
)
# TODO(aneesh) remove this flag once streaming sort is fully implemented
# Flag used to enable streaming sort
enable_streaming_sort = os.environ.get("BODO_ENABLE_STREAMING_SORT", "1") != "0"
# Flag used to enable streaming sort
enable_streaming_sort_limit_offset = (
    os.environ.get("BODO_ENABLE_STREAMING_SORT_LIMIT_OFFSET", "1") != "0"
)
# Flag used to enable creating theta sketches for columns when writing with Iceberg
enable_theta_sketches = os.environ.get("BODO_ENABLE_THETA_SKETCHES", "1") != "0"
# Should Bodo use decimal types when specified by BodoSQL.
bodo_use_decimal = os.environ.get("BODO_USE_DECIMAL", "0") != "0"
# Which SQL defaults should BODOSQL use (Snowflake vs Spark)
bodo_sql_style = os.environ.get("BODO_SQL_STYLE", "SNOWFLAKE").upper()
# Should we enable full covering set caching.
bodosql_full_caching = os.environ.get("BODO_USE_PARTIAL_CACHING", "0") != "0"
# If enabled, always uses the hash-based implementation instead of the
# sorting-based implementation for streaming window function execution.
bodo_disable_streaming_window_sort = (
    os.environ.get("BODO_DISABLE_STREAMING_WINDOW_SORT", "0") != "0"
)
# If enabled, generate a prefetch function call to load metadata paths for
# Snowflake-managed Iceberg tables in the BodoSQL plan.
prefetch_sf_iceberg = os.environ.get("BODO_PREFETCH_SF_ICEBERG", "1") != "0"

spawn_mode = os.environ.get("BODO_SPAWN_MODE", "1") != "0"


def get_sql_config_str() -> str:
    """
    Get a string that encapsulates all configurations relevant to compilation
    of SQL queries.

    Returns:
        str: Configuration string
    """
    conf_str = (
        f"{bodosql_use_streaming_plan=};{bodosql_streaming_batch_size=};{stream_loop_sync_iters=};{enable_snowflake_iceberg=};"
        f"{enable_timestamp_tz=};{runtime_join_filters_copy_threshold=};{enable_streaming_sort=};"
        f"{enable_streaming_sort_limit_offset=};{enable_theta_sketches=};{bodo_use_decimal=};"
        f"{bodo_sql_style=};{bodosql_full_caching=};{bodo_disable_streaming_window_sort=};{prefetch_sf_iceberg=};{spawn_mode=};"
    )
    return conf_str

check_parquet_schema = os.environ.get("BODO_CHECK_PARQUET_SCHEMA", "0") != "0"

# --------------------------- End Streaming Config ---------------------------

# ---------------------------- SQL Caching Config ----------------------------

# Directory where sql plans generated during compilation should be stored.
# This is expected to be a distributed filesystem which all nodes have access to.
sql_plan_cache_loc = os.environ.get("BODO_SQL_PLAN_CACHE_DIR")

# -------------------------- End SQL Caching Config --------------------------

# ---------------------------- GPU Config ----------------------------

# Flag to enable Bodo to use GPUs when available.
gpu_enabled = os.environ.get("BODO_GPU", "0") != "0"

# Flag to use async streams when GPU enabled.
gpu_async_enabled = os.environ.get("BODO_GPU_ASYNC", "0") != "0"

# ---------------------------- GPU Config ----------------------------

# ---------------------------- DataFrame Library Config ----------------------------

# Flag to enable Bodo DataFrames (bodo.pandas). When disabled, these classes
# will fallback to Pandas.
dataframe_library_enabled = os.environ.get("BODO_ENABLE_DATAFRAME_LIBRARY", "1") != "0"

# Run tests utilizing check_func in dataframe library mode (replaces)
# 'import pandas as pd' with 'import bodo.pandas as pd' when running the func.
test_dataframe_library_enabled = os.environ.get("BODO_ENABLE_TEST_DATAFRAME_LIBRARY", "0") != "0"

# Runs the DataFrame library in parallel mode if enabled (disable for debugging on a
# single core).
dataframe_library_run_parallel = os.environ.get("BODO_DATAFRAME_LIBRARY_RUN_PARALLEL", "1") != "0"

# If enabled (non-zero), dumps the dataframe library plans pre and post
# optimized plans to the screen.
dataframe_library_dump_plans = os.environ.get("BODO_DATAFRAME_LIBRARY_DUMP_PLANS", "0") != "0"

# If enabled (non-zero), profiles the dataframe library.
dataframe_library_profile = os.environ.get("BODO_DATAFRAME_LIBRARY_PROFILE", "0") != "0"

# If enabled (non-zero), captures the dataframe library API usage.
dataframe_library_capture = os.environ.get("BODO_DATAFRAME_LIBRARY_CAPTURE", "0") != "0"

# If enabled (non-zero), generate no fallback warnings.
dataframe_library_warn = os.environ.get("BODO_DATAFRAME_LIBRARY_WARN", "1") != "0"

# -------------------------- End DataFrame Library Config --------------------------

bodo_use_native_type_inference = (
    os.environ.get("BODO_NATIVE_TYPE_INFERENCE_ENABLED", "0") != "0"
)

tracing_level = int(os.environ.get("BODO_TRACING_LEVEL", "1"))

# For pip version of Bodo:
# Bodo needs to use the same libraries as Arrow (the same library files that pyarrow
# loads at runtime). We don't know what the path to these could be, so we have to
# preload them into memory to make sure the dynamic linker finds them
import pyarrow
import pyarrow.parquet

if platform.system() == "Windows":
    # For Windows pip we need to ensure impi DLLs are added to the search path
    # This is required for Python 3.14+ due to stricter DLL loading behavior
    # Search common locations where impi-rt installs DLLs
    base_dirs = []
    try:
        import sys
        base_dirs.append(sys.prefix)
        import impi_rt
        base_dirs.append(os.path.dirname(impi_rt.__file__))
    except ImportError:
        pass

    # Search for impi DLLs in common installation locations
    for base_dir in base_dirs:
        for search_dir in [
            os.path.join(base_dir, "Library", "bin"),
            os.path.join(base_dir, "Scripts"),
            os.path.join(base_dir, "Lib", "site-packages"),
            os.path.join(base_dir, "lib", "site-packages"),
        ]:
            if os.path.isdir(search_dir):
                os.add_dll_directory(search_dir)

    # importing our modified mpi4py (see buildscripts/mpi4py-pip/patch-3.1.2.diff)
    # guarantees that impi.dll is loaded, and therefore found when MPI calls are made
    import mpi4py

    # For Windows pip we need to ensure pyarrow DLLs are added to the search path
    for lib_dir in pyarrow.get_library_dirs():
        os.add_dll_directory(lib_dir)

# set number of threads to 1 for Numpy to avoid interference with Bodo's parallelism
# NOTE: has to be done before importing Numpy, and for all threading backends
orig_OPENBLAS_NUM_THREADS = os.environ.get("OPENBLAS_NUM_THREADS")
orig_OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS")
orig_MKL_NUM_THREADS = os.environ.get("MKL_NUM_THREADS")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# NOTE: 'pandas_compat' has to be imported first in bodo package to make sure all Numba
# patches are applied before Bodo's use.
import bodo.pandas_compat


def jit(*args, **kwargs):
    # Import compiler lazily
    from bodo.decorators import jit as _jit
    return _jit(*args, **kwargs)


class prange:
    """Dummy prange that is replaced in bodo.compiler when JIT is imported.
    """
    def __new__(cls, *args):
        return range(*args)


def typeof(*args, **kwargs):
    import numba
    # Import compiler lazily
    import bodo.decorators

    return numba.typeof(*args, **kwargs)


# The JIT version is replaced in decorators.py
def is_jit_execution():  # pragma: no cover
    return False


def wrap_python(*args, **kwargs):
    # Import compiler lazily
    from bodo.decorators import wrap_python as _wrap_python
    return _wrap_python(*args, **kwargs)


def jitclass(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.numba_compat import jitclass as _jitclass
    return _jitclass(*args, **kwargs)


def get_rank():
    # Avoid compiler imports
    from bodo.ext import hdist
    return hdist.get_rank_py_wrapper()


def get_size():
    # Avoid compiler imports
    from bodo.ext import hdist
    return hdist.get_size_py_wrapper()


def barrier():
    # Avoid compiler imports
    from bodo.ext import hdist
    return hdist.barrier_py_wrapper()


def parallel_print(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import parallel_print
    parallel_print(*args, **kwargs)


def allgatherv(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import allgatherv
    return allgatherv(*args, **kwargs)


def gatherv(data, allgather=False, warn_if_rep=True, root=0, comm=None):
    # Fall back to JIT version if not a spawn gatherv (workers may use comm=None)
    if allgather is True or warn_if_rep is False:
        # Import compiler lazily
        import bodo.decorators
        from bodo.libs.distributed_api import gatherv
        return gatherv(data, allgather, warn_if_rep, root, comm)

    # Avoid compiler imports for DataFrame library
    from bodo.spawn.utils import gatherv_nojit
    return gatherv_nojit(data, root, comm)


def scatterv(data, send_counts=None, warn_if_dist=True, root=0, comm=None):
    # Fall back to JIT version if not a spawn scatterv
    if send_counts is not None or warn_if_dist is False or comm is None:
        # Import compiler lazily
        import bodo.decorators
        from bodo.libs.distributed_api import scatterv
        return scatterv(data, send_counts, warn_if_dist, root, comm)

    # Avoid compiler imports for DataFrame library
    from bodo.spawn.utils import scatterv_nojit
    return scatterv_nojit(data, root, comm)


def get_start(total_size, pes, rank):  # pragma: no cover
    """Same as bodo.libs.distributed_api.get_start() but avoiding JIT compiler import
    here.
    """
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return rank * blk_size + min(rank, res)


def get_end(total_size, pes, rank):  # pragma: no cover
    """Same as bodo.libs.distributed_api.get_end() but avoiding JIT compiler import
    here.
    """
    res = total_size % pes
    blk_size = (total_size - res) // pes
    return (rank + 1) * blk_size + min(rank + 1, res)


def get_nodes_first_ranks(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import get_nodes_first_ranks
    return get_nodes_first_ranks(*args, **kwargs)


def rebalance(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import rebalance
    return rebalance(*args, **kwargs)


def random_shuffle(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import random_shuffle
    return random_shuffle(*args, **kwargs)

def get_num_nodes(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import get_num_nodes
    return get_num_nodes(*args, **kwargs)

def get_gpu_ranks(*args, **kwargs):
    # Import compiler lazily
    import bodo.decorators
    from bodo.libs.distributed_api import get_gpu_ranks
    return get_gpu_ranks(*args, **kwargs)


from bodo.spawn.spawner import spawn_process_on_nodes, stop_process_on_nodes


parquet_validate_schema = True

from bodo.user_logging import set_bodo_verbose_logger, set_verbose_level

# Restore thread limit. We don't want to limit other libraries like Arrow.
if orig_OPENBLAS_NUM_THREADS is None:
    os.environ.pop("OPENBLAS_NUM_THREADS", None)
else:
    os.environ["OPENBLAS_NUM_THREADS"] = orig_OPENBLAS_NUM_THREADS
if orig_OMP_NUM_THREADS is None:
    os.environ.pop("OMP_NUM_THREADS", None)
else:
    os.environ["OMP_NUM_THREADS"] = orig_OMP_NUM_THREADS
if orig_MKL_NUM_THREADS is None:
    os.environ.pop("MKL_NUM_THREADS", None)
else:
    os.environ["MKL_NUM_THREADS"] = orig_MKL_NUM_THREADS

# threshold for not inlining complex case statements to reduce compilation time (unit: number of lines in generated body code)
COMPLEX_CASE_THRESHOLD = 100


# Set our Buffer Pool as the default memory pool for PyArrow.
# Note that this will initialize the Buffer Pool.
import bodo.memory



########### finalize MPI, disconnect hdfs when exiting ############



def call_finalize():  # pragma: no cover
    from bodo.spawn.spawner import destroy_spawner
    from bodo.io import hdfs_reader
    from bodo.ext import hdist

    # Destroy the spawner before finalize since it uses MPI
    destroy_spawner()
    # Cleanup default buffer pool before finalize since it uses MPI inside
    bodo.memory_cpp.default_buffer_pool_cleanup()
    hdist.finalize_py_wrapper()
    hdfs_reader.disconnect_hdfs_py_wrapper()


def flush_stdout():
    # using a function since pytest throws an error sometimes
    # if flush function is passed directly to atexit
    if not sys.stdout.closed:
        sys.stdout.flush()


import atexit
atexit.register(call_finalize)
# Flush output before finalize
atexit.register(flush_stdout)
