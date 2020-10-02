# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Top-level init file for bodo package
"""
# set number of threads to 1 for Numpy to avoid interference with Bodo's parallelism
# NOTE: has to be done before importing Numpy, and for all threading backends
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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

from bodo.numba_compat import jitclass

datetime64ns = numba.core.types.NPDatetime("ns")

from numba.core.types import List

import bodo.config
import bodo.hiframes.boxing
import bodo.hiframes.pd_timestamp_ext
import bodo.io
import bodo.io.np_io
import bodo.libs
import bodo.libs.dict_ext
import bodo.libs.distributed_api
import bodo.libs.timsort
from bodo.libs.distributed_api import (
    allgatherv,
    barrier,
    dist_time,
    gatherv,
    get_rank,
    get_size,
    parallel_print,
    rebalance,
    scatterv,
)
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.utils.utils import cprint

import bodo.compiler  # isort:skip

use_pandas_join = False
use_cpp_drop_duplicates = True
# sql_access_method = "multiple_access_by_block"
sql_access_method = "multiple_access_nb_row_first"
from bodo.decorators import is_jit_execution, jit
from bodo.master_mode import init_master_mode

multithread_mode = False


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
