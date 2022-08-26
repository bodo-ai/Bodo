# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Equivalent of __init__.py for all BodoSQL array kernel files
"""

from bodo.libs.bodosql_array_kernel_utils import *  # noqa
from bodo.libs.bodosql_datetime_array_kernels import *  # noqa
from bodo.libs.bodosql_numeric_array_kernels import *  # noqa
from bodo.libs.bodosql_other_array_kernels import *  # noqa
from bodo.libs.bodosql_regexp_array_kernels import *  # noqa
from bodo.libs.bodosql_string_array_kernels import *  # noqa
from bodo.libs.bodosql_trig_array_kernels import *  # noqa
from bodo.libs.bodosql_variadic_array_kernels import *  # noqa
from bodo.libs.bodosql_window_agg_array_kernels import *  # noqa

broadcasted_fixed_arg_functions = {
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "change_event",
    "cos",
    "cosh",
    "radians",
    "degrees",
    "bitand",
    "bitleftshift",
    "bitnot",
    "bitor",
    "bitrightshift",
    "bitxor",
    "booland",
    "boolnot",
    "boolor",
    "boolxor",
    "char",
    "cond",
    "conv",
    "day_timestamp",
    "dayname",
    "div0",
    "editdistance_no_max",
    "editdistance_with_max",
    "equal_null",
    "format",
    "getbit",
    "haversine",
    "initcap",
    "instr",
    "int_to_days",
    "last_day",
    "left",
    "log",
    "lpad",
    "makedate",
    "month_diff",
    "monthname",
    "negate",
    "nullif",
    "ord_ascii",
    "regexp_count",
    "regexp_instr",
    "regexp_like",
    "regexp_replace",
    "regexp_substr",
    "regr_valx",
    "regr_valy",
    "repeat",
    "replace",
    "reverse",
    "right",
    "rpad",
    "second_timestamp",
    "sin",
    "sinh",
    "space",
    "split_part",
    "strcmp",
    "strtok",
    "substring",
    "substring_index",
    "tan",
    "tanh",
    "translate",
    "weekday",
    "width_bucket",
    "year_timestamp",
    "yearofweekiso",
}


broadcasted_variadic_functions = {"coalesce", "decode"}
