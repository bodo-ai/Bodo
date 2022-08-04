# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Equivalent of __init__.py for all BodoSQL array kernel files
"""

from bodo.libs.bodosql_array_kernel_utils import *  # noqa
from bodo.libs.bodosql_datetime_array_kernels import *  # noqa
from bodo.libs.bodosql_numeric_array_kernels import *  # noqa
from bodo.libs.bodosql_other_array_kernels import *  # noqa
from bodo.libs.bodosql_string_array_kernels import *  # noqa
from bodo.libs.bodosql_variadic_array_kernels import *  # noqa

broadcasted_fixed_arg_functions = {
    "char",
    "cond",
    "conv",
    "day_timestamp",
    "dayname",
    "div0",
    "format",
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
    "regr_valx",
    "regr_valy",
    "repeat",
    "replace",
    "reverse",
    "right",
    "rpad",
    "second_timestamp",
    "space",
    "strcmp",
    "substring",
    "substring_index",
    "weekday",
    "year_timestamp",
    "yearofweekiso",
}


broadcasted_variadic_functions = {"coalesce"}
