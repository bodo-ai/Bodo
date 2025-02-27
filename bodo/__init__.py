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
    from bodo.mpi4py import MPI

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
bodosql_streaming_batch_size = int(os.environ.get("BODO_STREAMING_BATCH_SIZE", 4096))
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


# Should Bodo use the new Arrow Azure FileSystem implementation instead of
# the old HDFS implementation.
enable_azure_fs = os.environ.get("BODO_USE_AZURE_FS", "1") != "0"
check_parquet_schema = os.environ.get("BODO_CHECK_PARQUET_SCHEMA", "0") != "0"

# --------------------------- End Streaming Config ---------------------------

# ---------------------------- SQL Caching Config ----------------------------

# Directory where sql plans generated during compilation should be stored.
# This is expected to be a distributed filesystem which all nodes have access to.
sql_plan_cache_loc = os.environ.get("BODO_SQL_PLAN_CACHE_DIR")

# -------------------------- End SQL Caching Config --------------------------

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
    # importing our modified mpi4py (see buildscripts/mpi4py-pip/patch-3.1.2.diff)
    # guarantees that impi.dll is loaded, and therefore found when MPI calls are made
    import bodo.mpi4py

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

# NOTE: 'numba_compat' has to be imported first in bodo package to make sure all Numba
# patches are applied before Bodo's Numba use (e.g. 'overload' is replaced properly)
import bodo.numba_compat  # isort:skip
import numba
from numba import (  # re-export from Numba
    gdb,
    gdb_breakpoint,
    gdb_init,
    pndindex,
    prange,
    stencil,
    threading_layer,
    typed,
    typeof,
)
from numba.core.types import *

from bodo.ir.object_mode import (
    warning_objmode as objmode,
    no_warning_objmode,
)
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)


def set_numba_environ_vars():
    """
    Set environment variables so that the Numba configuration can persist after reloading by re-setting config
    variables directly from environment variables.
    These should be tested in `test_numba_warn_config.py`.
    """
    # This env variable is set by the platform and points to the central cache directory
    # on the shared filesystem.
    if (cache_loc := os.environ.get("BODO_PLATFORM_CACHE_LOCATION")) is not None:
        if ("NUMBA_CACHE_DIR" in os.environ) and (
            os.environ["NUMBA_CACHE_DIR"] != cache_loc
        ):
            import warnings

            warnings.warn(
                "Since BODO_PLATFORM_CACHE_LOC is set, the value set for NUMBA_CACHE_DIR will be ignored"
            )
        numba.config.CACHE_DIR = cache_loc
        # In certain cases, numba reloads its config variables from the
        # environment. In those cases, the above line would be overridden.
        # Therefore, we also set it to the env var that numba reloads from.
        os.environ["NUMBA_CACHE_DIR"] = cache_loc

    # avoid Numba parallel performance warning when there is no Parfor in the IR
    numba.config.DISABLE_PERFORMANCE_WARNINGS = 1
    bodo_env_vars = {
        "NUMBA_DISABLE_PERFORMANCE_WARNINGS": "1",
    }
    os.environ.update(bodo_env_vars)


set_numba_environ_vars()

from bodo.numba_compat import jitclass

datetime64ns = numba.core.types.NPDatetime("ns")
timedelta64ns = numba.core.types.NPTimedelta("ns")

from numba.core.types import List

import bodo.ext
import bodo.libs
import bodo.libs.distributed_api
import bodo.libs.memory_budget
import bodo.libs.vendored.timsort
import bodo.libs.query_profile_collector
import bodo.libs.streaming.join
import bodo.libs.streaming.groupby
import bodo.libs.streaming.dict_encoding
import bodo.libs.streaming.sort
import bodo.libs.streaming.union
import bodo.libs.streaming.window
import bodo.libs.table_builder

import bodo.io

# Rexport HDFS Locations
# Needed for bodo.io.snowflake_write
from bodo.io.snowflake_hdfs import HDFS_CORE_SITE_LOC_DIR, HDFS_CORE_SITE_LOC
import bodo.io.np_io
import bodo.io.csv_iterator_ext
import bodo.io.iceberg
import bodo.io.snowflake_write
import bodo.io.iceberg.stream_iceberg_write
import bodo.io.stream_parquet_write

from bodo.libs.distributed_api import (
    allgatherv,
    barrier,
    dist_time,
    gatherv,
    get_rank,
    get_size,
    get_nodes_first_ranks,
    parallel_print,
    rebalance,
    random_shuffle,
    scatterv,
)
import bodo.hiframes.boxing
import bodo.hiframes.pd_timestamp_ext
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.null_arr_ext import null_array_type, null_dtype
from bodo.libs.str_ext import string_type
import bodo.libs.binops_ext
import bodo.libs.array_ops
from bodo.utils.utils import cprint
from bodo.hiframes.datetime_date_ext import datetime_date_type, datetime_date_array_type
from bodo.hiframes.time_ext import (
    TimeType,
    TimeArrayType,
    Time,
    parse_time_string,
)
from bodo.hiframes.timestamptz_ext import (
    TimestampTZ,
    TimestampTZType,
    timestamptz_type,
    timestamptz_array_type,
)
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    datetime_timedelta_array_type,
    pd_timedelta_type,
)
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.pd_timestamp_ext import (
    PandasTimestampType,
    pd_timestamp_tz_naive_type,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.primitive_arr_ext import PrimitiveArrayType
from bodo.libs.map_arr_ext import MapArrayType, MapScalarType
from bodo.libs.nullable_tuple_ext import NullableTupleType
from bodo.libs.struct_arr_ext import StructArrayType, StructType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.libs.csr_matrix_ext import CSRMatrixType
from bodo.libs.matrix_ext import MatrixType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType, pd_datetime_tz_naive_type
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import (
    DatetimeIndexType,
    NumericIndexType,
    PeriodIndexType,
    IntervalIndexType,
    CategoricalIndexType,
    RangeIndexType,
    StringIndexType,
    BinaryIndexType,
    TimedeltaIndexType,
)
from bodo.hiframes.pd_offsets_ext import (
    month_begin_type,
    month_end_type,
    week_type,
    date_offset_type,
)
from bodo.hiframes.pd_categorical_ext import (
    PDCategoricalDtype,
    CategoricalArrayType,
)
from bodo.utils.typing import register_type
from bodo.libs.logging_ext import LoggingLoggerType
from bodo.hiframes.table import TableType


import bodo.compiler  # isort:skip

use_pandas_join = False
use_cpp_drop_duplicates = True
from bodo.decorators import is_jit_execution, jit, wrap_python

multithread_mode = False
parquet_validate_schema = True

import bodo.utils.tracing
import bodo.utils.tracing_py
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

# Check for addition of new methods and attributes in pandas documentation for Series. Needs to be checked for every new Pandas release.
# New methods and attributes need to be added to the unsupported_xxx list in the appropriate _ext.py file.
# NOTE: This check needs to happen last.
import bodo.utils.pandas_coverage_tracking
