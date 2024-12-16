"""
Implements time array kernels that are specific to BodoSQL
"""

import numba
from numba.core import types

import bodo
import bodosql
from bodo.utils.typing import (
    get_overload_const_bool,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.kernels.array_kernel_utils import (
    convert_numeric_to_int,
    gen_vectorized,
    is_valid_string_arg,
    is_valid_timestamptz_arg,
    is_valid_tz_aware_datetime_arg,
    is_valid_tz_naive_datetime_arg,
    unopt_argument,
    verify_int_arg,
)


@numba.generated_jit(nopython=True, no_unliteral=True)
def to_time(arr, format_str, _try, dict_encoding_state=None, func_id=-1):
    """Handles TIME/TO_TIME/TRY_TO_TIME and forwards
    to the appropriate version of the real implementation"""

    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.time_array_kernels.to_time_util",
            [
                "arr",
                "format_str",
                "_try",
                "dict_encoding_state",
                "func_id",
            ],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(
        arr, format_str, _try, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return to_time_util(arr, format_str, _try, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def to_time_util(
    arr, format_str, _try, dict_encoding_state, func_id
):  # pragma: no cover
    """Kernel for `TO_TIME`, `TIME`, and `TRY_TO_TIME`"""

    arg_names = ["arr", "format_str", "_try", "dict_encoding_state", "func_id"]
    arg_types = [arr, format_str, _try, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False, False]

    _try = get_overload_const_bool(_try, "to_time", "_try")

    if is_valid_string_arg(arr) or is_overload_none(arr):
        if is_overload_none(format_str):
            scalar_text = "hr, mi, sc, ns, succeeded = bodo.parse_time_string(arg0)\n"
        else:
            scalar_text = "py_format_str = bodosql.kernels.snowflake_conversion_array_kernels.convert_snowflake_date_format_str_to_py_format(arg1)\n"
            scalar_text += "succeeded, val = bodosql.kernels.snowflake_conversion_array_kernels.pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
            scalar_text += (
                "hr, mi, sc, ns = val.hour, val.minute, val.second, val.nanosecond\n"
            )
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
    elif is_valid_timestamptz_arg(arr):
        scalar_text = "ts = arg0.local_timestamp()\n"
        scalar_text += "res[i] = bodo.Time(ts.hour, ts.minute, ts.second, microsecond=ts.microsecond, nanosecond=ts.nanosecond, precision=9)\n"
    else:
        raise_bodo_error(
            "TIME/TO_TIME/TRY_TO_TIME argument must be a string, timestamp, or null"
        )
    out_dtype = bodo.TimeArrayType(9)
    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def time_from_parts(hour, minute, second, nanosecond):  # pragma: no cover
    args = [hour, minute, second, nanosecond]
    arg_names = ["hour", "minute", "second", "nanosecond"]

    return convert_numeric_to_int(
        "bodosql.kernels.time_array_kernels.time_from_parts_unopt_util",
        arg_names,
        args,
        arg_names,
    )


@numba.generated_jit(nopython=True)
def time_from_parts_unopt_util(hour, minute, second, nanosecond):  # pragma: no cover
    args = [hour, minute, second, nanosecond]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodosql.kernels.time_array_kernels.time_from_parts_unopt_util",
                ["hour", "minute", "second", "nanosecond"],
                i,
            )

    def impl(hour, minute, second, nanosecond):
        return bodosql.kernels.time_array_kernels.time_from_parts_util(
            hour, minute, second, nanosecond
        )

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
