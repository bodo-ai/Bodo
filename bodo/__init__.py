# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Top-level init file for bodo package

isort:skip_file
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


import os
import platform

# For pip version of Bodo:
# Bodo needs to use the same libraries as Arrow (the same library files that pyarrow
# loads at runtime). We don't know what the path to these could be, so we have to
# preload them into memory to make sure the dynamic linker finds them
import pyarrow
import pyarrow.parquet

if platform.system() == "Windows":
    # importing our modified mpi4py (see buildscripts/mpi4py-pip/patch-3.1.2.diff)
    # guarantees that msmpi.dll is loaded, and therefore found when MPI calls are made
    import mpi4py

# set number of threads to 1 for Numpy to avoid interference with Bodo's parallelism
# NOTE: has to be done before importing Numpy, and for all threading backends
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

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
    objmode,
    pndindex,
    prange,
    stencil,
    threading_layer,
    typed,
    typeof,
)
from numba.core.types import *


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

import bodo.libs
import bodo.libs.distributed_api
import bodo.libs.timsort
import bodo.io
import bodo.io.np_io
import bodo.io.csv_iterator_ext
import bodo.io.iceberg
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
from bodo.libs.str_ext import string_type
import bodo.libs.binops_ext
import bodo.libs.array_ops
from bodo.utils.utils import cprint
from bodo.hiframes.datetime_date_ext import datetime_date_type, datetime_date_array_type
from bodo.hiframes.time_ext import TimeType, TimeArrayType, Time, time_from_str
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    datetime_timedelta_array_type,
    pd_timedelta_type,
)
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.pd_timestamp_ext import pd_timestamp_type
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.dict_arr_ext import dict_str_arr_type
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.nullable_tuple_ext import NullableTupleType
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.libs.csr_matrix_ext import CSRMatrixType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
import bodo.libs.bodosql_array_kernel_utils
import bodo.libs.bodosql_datetime_array_kernels
import bodo.libs.bodosql_string_array_kernels
import bodo.libs.bodosql_regexp_array_kernels
import bodo.libs.bodosql_numeric_array_kernels
import bodo.libs.bodosql_variadic_array_kernels
import bodo.libs.bodosql_other_array_kernels
import bodo.libs.bodosql_trig_array_kernels
import bodo.libs.bodosql_window_agg_array_kernels
import bodo.libs.bodosql_array_kernels
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
import bodo.dl

use_pandas_join = False
use_cpp_drop_duplicates = True
from bodo.decorators import is_jit_execution, jit
from bodo.master_mode import init_master_mode

multithread_mode = False
parquet_validate_schema = True

import bodo.utils.tracing
import bodo.utils.tracing_py
from bodo.user_logging import set_bodo_verbose_logger, set_verbose_level

# clear thread limit. We don't want to limit other libraries like Arrow
os.environ.pop("OPENBLAS_NUM_THREADS", None)
os.environ.pop("OMP_NUM_THREADS", None)
os.environ.pop("MKL_NUM_THREADS", None)

# Create a temp dir name that will be used for writing temporary
# core-site files for ADLS read/writes. At this time, this is
# used by Snowflake write when writing to an ADLS backed Snowflake
# stage. This can also be used by users for regular ADLS read/writes.
# This is *required* when both Snowflake write and regular ADLS I/O
# need to happen as part of the same process since Hadoop picks the
# first core-site.xml it finds in the CLASSPATH (but the contents
# of the core-site.xml file can be modified dynamically once it's
# picked). We are using a LazyTemporaryDirectory, i.e. the name is
# decided and synchronized across all rank (since is_parallel=True) now,
# but the directory needs to be initialized when there's an actual
# use for it. This is to avoid any unnecessary filesystem operations.
# Note that the initialization step must be called from all ranks
# since is_parallel=True, else it will lead to a hang. Initialization
# is idempotent, and can be safely repeated.
from bodo.io.lazy_tempdir import LazyTemporaryDirectory

HDFS_CORE_SITE_LOC_DIR = LazyTemporaryDirectory(is_parallel=True)
HDFS_CORE_SITE_LOC = os.path.join(HDFS_CORE_SITE_LOC_DIR.name, "core-site.xml")

# Add this location to the front of the CLASSPATH so that when/if we create
# a core-site, it gets picked up. If we never create one ourselves, Hadoop
# will continue looking in the CLASSPATH, so there shouldn't be any negative
# side-effects.
os.environ["CLASSPATH"] = f"{HDFS_CORE_SITE_LOC_DIR.name}:" + os.environ.get(
    "CLASSPATH", ""
)
# Also expose the location as an evironment variable. This will be used
# by bodo_azurefs_sas_token_provider to locate the SAS token file location.
# See BodoSASTokenProvider.java.
os.environ["BODO_HDFS_CORE_SITE_LOC_DIR"] = HDFS_CORE_SITE_LOC_DIR.name

# Try to import our SAS Token Provider implementation which will
# add the jar to the CLASSPATH from the onset, avoiding issues
# with interleaving with other JVM related operations. e.g.
# if we don't do this import, and the user doesn't either in their code,
# then if we read from / write to ADLS before a Snowflake write operation,
# the JVM would already be initialized from the first operation without
# the jar for the SAS token provider class in the CLASSPATH, and then
# the Snowflake write would fail (even though this import will be done
# as part of Snowflake write). If this import creates issues or
# degrades performance, we could require it to be done by the user
# instead.
try:
    import bodo_azurefs_sas_token_provider
except:
    pass
