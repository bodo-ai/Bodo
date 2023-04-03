# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements time array kernels that are specific to BodoSQL
"""

import numba

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import get_overload_const_bool, raise_bodo_error


@numba.generated_jit(nopython=True)
def to_time(arr, _try):
    """Handles TIME/TO_TIME/TRY_TO_TIME and forwards
    to the appropriate version of the real implementation"""

    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_time_util",
            [
                "arr",
                "_try",
            ],
            0,
        )

    def impl(arr, _try):  # pragma: no cover
        return to_time_util(arr, _try)

    return impl


@numba.generated_jit(nopython=True)
def to_time_util(arr, _try):  # pragma: no cover
    """Kernel for `TO_TIME`, `TIME`, and `TRY_TO_TIME`"""

    arg_names = ["arr", "_try"]
    arg_types = [arr, _try]
    propagate_null = [True]

    _try = get_overload_const_bool(_try)

    if is_valid_string_arg(arr) or is_overload_none(arr):
        scalar_text = "hr, mi, sc, ns, succeeded = bodo.parse_time_string(arg0)\n"
        scalar_text += "if succeeded:\n"
        scalar_text += "   res[i] = bodo.Time(hr, mi, sc, nanosecond=ns, precision=9)\n"
        scalar_text += "else:\n"
        if _try:
            scalar_text += "  bodo.libs.array_kernels.setna(res, i)"
        else:
            scalar_text += "  raise ValueError('Invalid time string')"
    elif is_valid_tz_naive_datetime_arg(arr) or is_valid_tz_aware_datetime_arg(arr):
        scalar_text = "ts = bodo.utils.conversion.box_if_dt64(arg0)\n"
        scalar_text += "res[i] = bodo.Time(ts.hour, ts.minute, ts.second, microsecond=ts.microsecond, nanosecond=ts.nanosecond, precision=9)\n"
    else:
        raise_bodo_error(
            "TIME/TO_TIME/TRY_TO_TIME argument must be a string, timestamp, or null"
        )
    out_dtype = bodo.TimeArrayType(9)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def time_from_parts(hour, minute, second, nanosecond):

    args = [hour, minute, second, nanosecond]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.time_from_parts",
                ["hour", "minute", "second", "nanosecond"],
                i,
            )

    def impl(hour, minute, second, nanosecond):  # pragma: no cover
        return time_from_parts_util(hour, minute, second, nanosecond)

    return impl


@numba.generated_jit(nopython=True)
def time_from_parts_util(hour, minute, second, nanosecond):  # pragma: no cover
    """Kernel for `TIMEFROMPARTS` and `TIME_FROM_PARTS`"""

    verify_int_arg(hour, "TIME_FROM_PARTS", "hour")
    verify_int_arg(minute, "TIME_FROM_PARTS", "minute")
    verify_int_arg(second, "TIME_FROM_PARTS", "second")
    verify_int_arg(nanosecond, "TIME_FROM_PARTS", "nanosecond")

    arg_names = ["hour", "minute", "second", "nanosecond"]
    arg_types = [hour, minute, second, nanosecond]
    propagate_null = [True] * 4
    scalar_text = "res[i] = bodo.Time(arg0, arg1, arg2, nanosecond=arg3, precision=9)"

    out_dtype = bodo.TimeArrayType(9)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
