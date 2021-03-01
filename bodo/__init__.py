# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Top-level init file for bodo package

isort:skip_file
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
timedelta64ns = numba.core.types.NPTimedelta("ns")

from numba.core.types import List

import bodo.libs
import bodo.libs.dict_ext
import bodo.libs.distributed_api
import bodo.libs.timsort
import bodo.io
import bodo.io.np_io
from bodo.libs.distributed_api import (
    allgatherv,
    barrier,
    dist_time,
    gatherv,
    get_rank,
    get_size,
    parallel_print,
    rebalance,
    random_shuffle,
    scatterv,
)
import bodo.config
import bodo.hiframes.boxing
import bodo.hiframes.pd_timestamp_ext
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
import bodo.libs.binops_ext
from bodo.utils.utils import cprint
from bodo.hiframes.datetime_date_ext import datetime_date_type, datetime_date_array_type
from bodo.hiframes.datetime_timedelta_ext import (
    datetime_timedelta_type,
    datetime_timedelta_array_type,
    pd_timedelta_type,
)
from bodo.hiframes.datetime_datetime_ext import datetime_datetime_type
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import Decimal128Type, DecimalArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.map_arr_ext import MapArrayType
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import (
    DatetimeIndexType,
    NumericIndexType,
    PeriodIndexType,
    RangeIndexType,
    StringIndexType,
    TimedeltaIndexType,
)
from bodo.hiframes.pd_offsets_ext import (
    month_end_type,
    week_type,
    date_offset_type,
)
from bodo.hiframes.pd_categorical_ext import (
    PDCategoricalDtype,
    CategoricalArray,
)


import bodo.compiler  # isort:skip
import bodo.dl

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
