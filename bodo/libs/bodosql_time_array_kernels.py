# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements time array kernels that are specific to BodoSQL
"""

import numba

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


def make_time_to_time_util(name):
    """Generate the util for `TIME` and `TO_TIME`"""

    @numba.generated_jit(nopython=True)
    def util(arr):  # pragma: no cover
        """Kernel for `TO_TIME` and `TIME`"""

        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]

        if is_valid_int_arg(arr):
            scalar_text = "res[i] = bodo.Time(0, 0, arg0)"
        elif is_valid_string_arg(arr):
            scalar_text = "res[i] = bodo.time_from_str(arg0)"
        else:
            raise_bodo_error(
                f"{name} argument must be an integer, string, integer or string column, or null"
            )

        out_dtype = bodo.TimeArrayType(9)

        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return util


time_util = make_time_to_time_util("TIME")
to_time_util = make_time_to_time_util("TO_TIME")


@numba.generated_jit(nopython=True)
def time_from_parts_util(hour, minute, second, nanosecond):  # pragma: no cover
    """Kernel for `TIME_FROM_PARTS`"""

    verify_int_arg(hour, "TIME_FROM_PARTS", "hour")
    verify_int_arg(minute, "TIME_FROM_PARTS", "minute")
    verify_int_arg(second, "TIME_FROM_PARTS", "second")
    verify_int_arg(nanosecond, "TIME_FROM_PARTS", "nanosecond")

    arg_names = ["hour", "minute", "second", "nanosecond"]
    arg_types = [hour, minute, second, nanosecond]
    propagate_null = [True] * 4
    scalar_text = "res[i] = bodo.Time(arg0, arg1, arg2, arg3)"

    out_dtype = bodo.TimeType(9)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
