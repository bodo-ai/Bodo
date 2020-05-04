# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Top-level init file for bodo package
"""
# NOTE: 'numba_compat' has to be imported first in bodo package to make sure all Numba
# patches are applied before Bodo's Numba use (e.g. 'overload' is replaced properly)
import bodo.numba_compat
import numba

# re-export from Numba
from numba import (
    typeof,
    prange,
    pndindex,
    gdb,
    gdb_breakpoint,
    gdb_init,
    stencil,
    threading_layer,
    jitclass,
    objmode,
    typed,
)
from numba.core.types import *

datetime64ns = numba.core.types.NPDatetime("ns")

import bodo.libs
import bodo.libs.dict_ext
import bodo.libs.distributed_api
from bodo.libs.distributed_api import (
    dist_time,
    parallel_print,
    get_rank,
    get_size,
    barrier,
    gatherv,
    allgatherv,
    scatterv,
)

from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type
from numba.core.types import List
from bodo.utils.utils import cprint, distribution_report
import bodo.compiler
import bodo.io
import bodo.io.np_io
import bodo.hiframes.pd_timestamp_ext
import bodo.hiframes.boxing
import bodo.config
import bodo.libs.timsort

use_pandas_join = False
use_cpp_drop_duplicates = True
# sql_access_method = "multiple_access_by_block"
sql_access_method = "multiple_access_nb_row_first"
from bodo.decorators import jit
from bodo.master_mode import init_master_mode

multithread_mode = False


from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
