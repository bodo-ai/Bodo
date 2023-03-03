# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements a number of array kernels that handling casting functions for BodoSQL
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    get_literal_value,
    get_overload_const_bool,
    is_literal_type,
    is_overload_constant_str,
    is_overload_none,
    is_str_arr_type,
    raise_bodo_error,
)


def make_to_boolean(_try):
    """Generate utility functions to unopt TO_BOOLEAN arguments"""

    @numba.generated_jit(nopython=True)
    def func(arr):
        """Handles cases where TO_BOOLEAN receives optional arguments and forwards
        to the appropriate version of the real implementation"""
        if isinstance(arr, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.to_boolean", ["arr"], 0
            )

        def impl(arr):  # pragma: no cover
            return to_boolean_util(arr, numba.literally(_try))

        return impl

    return func


try_to_boolean = make_to_boolean(True)
to_boolean = make_to_boolean(False)


@numba.generated_jit(nopython=True)
def to_boolean_util(arr, _try=False):
    """A dedicated kernel for the SQL function TO_BOOLEAN which takes in a
    number (or column) and returns True if it is not zero and not null,
    False if it is zero, and NULL otherwise.


    Args:
        arr (numerical array/series/scalar): the number(s) being operated on
        _try (bool): whether to return NULL (iff true) on error or raise an exception

    Returns:
        boolean series/scalar: the boolean value of the number(s) with the
        specified null handling rules
    """
    verify_string_numeric_arg(arr, "TO_BOOLEAN", "arr")
    is_string = is_valid_string_arg(arr)
    is_float = is_valid_float_arg(arr)
    _try = get_overload_const_bool(_try)

    if _try:
        on_fail = "bodo.libs.array_kernels.setna(res, i)\n"
    else:
        if is_string:
            err_msg = "string must be one of {'true', 't', 'yes', 'y', 'on', '1'} or {'false', 'f', 'no', 'n', 'off', '0'}"
        else:
            err_msg = "value must be a valid numeric expression"
        on_fail = (
            f"""raise ValueError("invalid value for boolean conversion: {err_msg}")"""
        )

    arg_names = ["arr", "_try"]
    arg_types = [arr, _try]
    propagate_null = [True, False]

    prefix_code = None
    if is_string:
        prefix_code = "true_vals = {'true', 't', 'yes', 'y', 'on', '1'}\n"
        prefix_code += "false_vals = {'false', 'f', 'no', 'n', 'off', '0'}"
    if is_string:
        scalar_text = "s = arg0.lower()\n"
        scalar_text += f"is_true_val = s in true_vals\n"
        scalar_text += f"res[i] = is_true_val\n"
        scalar_text += f"if not (is_true_val or s in false_vals):\n"
        scalar_text += f"  {on_fail}\n"
    elif is_float:
        # TODO: fix this for float case (see above)
        # np.isnan should error here, but it will not reach because
        # NaNs will be caught since propogate_null[0] is True
        scalar_text = "if np.isinf(arg0) or np.isnan(arg0):\n"
        scalar_text += f"  {on_fail}\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = bool(arg0)\n"
    else:
        scalar_text = f"res[i] = bool(arg0)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


def to_date(conversionVal, format_str):  # pragma: no cover
    return


def try_to_date(conversionVal, format_str):  # pragma: no cover
    return


def to_timestamp(conversionVal, format_str, time_zone, scale):  # pragma: no cover
    return


def to_date_util(conversionVal, format_str):  # pragma: no cover
    return


def try_to_date_util(conversionVal, format_str):  # pragma: no cover
    return


def create_date_cast_util(func, error_on_fail):
    """Creates an overload for a dedicated kernel for TO_DATE/TRY_TO_DATE
    Takes in 2 arguments: the name of the kernel being created and whether it should
    have an error when it has a failure (as opposed to outputting null),

    Returns an overload that accepts 2 arguments: the value to be converted
    (a series or scalar of mulitiple possible types), and an optional format string
    for cases where the input is a string.

    The full specification is noted here:
    https://docs.snowflake.com/en/sql-reference/functions/to_date.html
    """
    if error_on_fail:
        error_str = "raise ValueError('Invalid input while converting to date value')"
    else:
        error_str = "bodo.libs.array_kernels.setna(res, i)"

    def overload_impl(conversionVal, format_str):
        verify_string_arg(format_str, func, "format_str")

        # When returning a scalar we return a pd.Timestamp type.
        is_out_arr = bodo.utils.utils.is_array_typ(
            conversionVal, True
        ) or bodo.utils.utils.is_array_typ(format_str, True)
        unbox_str = "unbox_if_tz_naive_timestamp" if is_out_arr else ""

        # If the format string is specified, then arg0 must be a string
        if not is_overload_none(format_str):
            verify_string_arg(conversionVal, func, "conversionVal")
            scalar_text = (
                "py_format_str = convert_sql_date_format_str_to_py_format(arg1)\n"
            )
            scalar_text += "was_successful, tmp_val = pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
            scalar_text += "if not was_successful:\n"
            scalar_text += f"  {error_str}\n"
            scalar_text += "else:\n"
            scalar_text += f"  res[i] = {unbox_str}(tmp_val.normalize())\n"

        # NOTE: gen_vectorized will automatically map this function over the values dictionary
        # of a dict encoded string array instead of decoding it whenever possible
        elif is_valid_string_arg(conversionVal):
            """
            If no format string is specified, attempt to parse the string according to these date formats:
            https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
            handled by pd.to_datetime() in Bodo jit code.

            It will also check if the string is convertable to int, IE '12345' or '-4321'"""

            # Conversion needs to be done incase arg0 is unichr array
            scalar_text = "arg0 = str(arg0)\n"
            scalar_text += "if (arg0.isnumeric() or (len(arg0) > 1 and arg0[0] == '-' and arg0[1:].isnumeric())):\n"
            scalar_text += f"   res[i] = {unbox_str}(number_to_datetime(np.int64(arg0)).normalize())\n"

            scalar_text += "else:\n"
            scalar_text += (
                "   was_successful, tmp_val = pd_to_datetime_error_checked(arg0)\n"
            )
            scalar_text += "   if not was_successful:\n"
            scalar_text += f"      {error_str}\n"
            scalar_text += "   else:\n"
            scalar_text += f"      res[i] = {unbox_str}(tmp_val.normalize())\n"

        # If a tz-aware timestamp, construct a tz-naive timestamp with the same date
        elif is_valid_tz_aware_datetime_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(year=arg0.year, month=arg0.month, day=arg0.day))\n"

        # If a non-tz timestamp/datetime, round it down to the nearest day
        elif is_valid_datetime_or_date_arg(
            conversionVal
        ) or is_valid_tz_naive_datetime_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0).normalize())\n"

        else:  # pragma: no cover
            raise raise_bodo_error(
                f"Internal error: unsupported type passed to to_date_util for argument conversionVal: {conversionVal}"
            )

        arg_names = ["conversionVal", "format_str"]
        arg_types = [conversionVal, format_str]
        propagate_null = [True, False]

        out_dtype = types.Array(bodo.datetime64ns, 1, "C")

        extra_globals = {
            "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
            "number_to_datetime": number_to_datetime,
            "convert_sql_date_format_str_to_py_format": convert_sql_date_format_str_to_py_format,
            "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
        }
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            extra_globals=extra_globals,
        )

    return overload_impl


def create_date_cast_func(func_name):
    """Takes in a function name (either TO_DATE or TRY_TO_DATE) and generates
    the wrapper function for the corresponding kernel.
    """

    def overload_func(conversionVal, format_str):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [conversionVal, format_str]
        for i in range(len(args)):
            if isinstance(args[i], types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name.lower()}_util",
                    ["conversionVal", "format_str"],
                    i,
                )

        func_text = "def impl(conversionVal, format_str):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{func_name.lower()}_util(conversionVal, format_str)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def _install_date_cast_overloads():
    date_cast_fns = [
        ("TO_DATE", to_date, to_date_util, True),
        ("TRY_TO_DATE", try_to_date, try_to_date_util, False),
    ]
    for func_name, func, util_func, error_on_fail in date_cast_fns:
        overload(func)(create_date_cast_func(func_name))
        overload(util_func)(create_date_cast_util(func_name, error_on_fail))


_install_date_cast_overloads()


def try_to_timestamp(conversionVal, format_str, time_zone, scale):  # pragma: no cover
    return


def to_timestamp_util(conversionVal, format_str, time_zone, scale):  # pragma: no cover
    return


def try_to_timestamp_util(
    conversionVal, format_str, time_zone, scale
):  # pragma: no cover
    return


def create_timestamp_cast_util(func, error_on_fail):
    """Creates an overload for a dedicated kernel for one of the timestamp
    casting functions:

    - TO_TIMESTAMP
    - TRY_TO_TIMESTAMP
    - TO_TIMESTAMP_TZ
    - TRY_TO_TIMESTAMP_TZ
    - TO_TIMESTAMP_LTZ
    - TRY_TO_TIMESTAMP_LTZ
    - TO_TIMESTAMP_NTZ
    - TRY_TO_TIMESTAMP_NTZ

    Takes in 4 arguments: the name of the kernel being created, whether it should
    have an error when it has a failure (as opposed to outputting null),
    whether it should keep the time or truncate it (e.g. TO_DATE returns a datetime
    type that is truncated to midnight), and the scale if the argument is numeric
    (i.e. 0 = seconds, 3 = milliseconds, 6 = microseconds, 9 = nanoseconds)

    Returns an overload that accepts 3 arguments: the value to be converted
    (a series or scalar of mulitiple possible types), and two literals.
    The first is a format string for cases where the input is a string. The second
    is a time zone for the output data.

    The full specification is noted here:
    https://docs.snowflake.com/en/sql-reference/functions/to_timestamp.html
    """
    if error_on_fail:
        error_str = "raise ValueError('Invalid input while converting to date value')"
    else:
        error_str = "bodo.libs.array_kernels.setna(res, i)"

    def overload_impl(conversionVal, format_str, time_zone, scale):
        verify_string_arg(format_str, func, "format_str")

        # The scale must be a constant scalar, per Snowflake
        if not isinstance(scale, types.Integer):
            raise_bodo_error(
                f"{func}: scale argument must be a scalar integer between 0 and 9"
            )

        prefix_code = "if not (0 <= scale <= 9):\n"
        prefix_code += f"   raise ValueError('{func}: scale must be between 0 and 9')\n"

        # Infer the correct way to adjust the timezones of the Timestamps calculated
        # based on the timezone of the current data (if there is any), and the target timezone.
        current_tz = get_tz_if_exists(conversionVal)
        if current_tz is None:
            time_zone = get_literal_value(time_zone)
            if is_overload_constant_str(time_zone):
                # NTZ -> TZ
                localize_str = f".tz_localize('{time_zone}')"
            elif is_overload_none(time_zone):
                # NTZ -> NTZ
                localize_str = ""
                time_zone = None
            else:
                raise_bodo_error("time_zone argument must be a scalar string or None")
        else:
            if is_overload_constant_str(time_zone):
                time_zone = get_literal_value(time_zone)
                if time_zone == current_tz:
                    # TZ -> Same TZ
                    localize_str = ""
                else:
                    # TZ -> Different TZ
                    localize_str = f".tz_localize(None).tz_localize('{time_zone}')"
            elif is_overload_none(time_zone):
                # TZ -> NTZ
                localize_str = f".tz_localize(None)"
                time_zone = None
            else:
                raise_bodo_error("time_zone argument must be a scalar string or None")

        is_out_arr = bodo.utils.utils.is_array_typ(
            conversionVal, True
        ) or bodo.utils.utils.is_array_typ(format_str, True)

        # When returning a scalar we return a pd.Timestamp type.
        unbox_str = "unbox_if_tz_naive_timestamp" if is_out_arr else ""

        # If the format string is specified, then arg0 must be string
        if not is_overload_none(format_str):
            verify_string_arg(conversionVal, func, "conversionVal")
            scalar_text = (
                "py_format_str = convert_sql_date_format_str_to_py_format(arg1)\n"
            )
            scalar_text += "was_successful, tmp_val = pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
            scalar_text += "if not was_successful:\n"
            scalar_text += f"  {error_str}\n"
            scalar_text += "else:\n"
            scalar_text += f"  res[i] = {unbox_str}(tmp_val{localize_str})\n"

        # NOTE: gen_vectorized will automatically map this function over the values dictionary
        # of a dict encoded string array instead of decoding it whenever possible
        elif is_valid_string_arg(conversionVal):
            """
            If no format string is specified, attempt to parse the string according to these date formats:
            https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
            handled by pd.to_datetime() in Bodo jit code.

            It will also check if the string is convertable to int, IE '12345' or '-4321'"""

            # Conversion needs to be done incase arg0 is unichr array
            scalar_text = "arg0 = str(arg0)\n"
            scalar_text += "if (arg0.isnumeric() or (len(arg0) > 1 and arg0[0] == '-' and arg0[1:].isnumeric())):\n"
            scalar_text += f"   res[i] = {unbox_str}(number_to_datetime(np.int64(arg0)){localize_str})\n"

            scalar_text += "else:\n"
            scalar_text += (
                "   was_successful, tmp_val = pd_to_datetime_error_checked(arg0)\n"
            )
            scalar_text += "   if not was_successful:\n"
            scalar_text += f"      {error_str}\n"
            scalar_text += "   else:\n"
            scalar_text += f"      res[i] = {unbox_str}(tmp_val{localize_str})\n"

        elif is_valid_int_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0 * (10 ** (9 - arg3))){localize_str})\n"

        elif is_valid_float_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0 * (10 ** (9 - arg3))){localize_str})\n"

        elif is_valid_datetime_or_date_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0){localize_str})\n"

        elif is_valid_tz_aware_datetime_arg(conversionVal):
            scalar_text = f"res[i] = {unbox_str}(arg0{localize_str})\n"

        else:  # pragma: no cover
            raise raise_bodo_error(
                f"Internal error: unsupported type passed to to_timestamp_util for argument conversionVal: {conversionVal}"
            )

        arg_names = ["conversionVal", "format_str", "time_zone", "scale"]
        arg_types = [conversionVal, format_str, time_zone, scale]
        propagate_null = [True, False, False, False]

        # Determine the output dtype. If a timezone is provided then we have a
        # tz-aware output. Otherwise we output datetime64ns.
        if time_zone is not None:
            out_dtype = bodo.DatetimeArrayType(time_zone)
        else:
            out_dtype = types.Array(bodo.datetime64ns, 1, "C")

        extra_globals = {
            "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
            "number_to_datetime": number_to_datetime,
            "convert_sql_date_format_str_to_py_format": convert_sql_date_format_str_to_py_format,
            "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
        }
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            prefix_code=prefix_code,
            extra_globals=extra_globals,
        )

    return overload_impl


def create_timestamp_cast_func(func_name):
    def overload_func(conversionVal, format_str, time_zone, scale):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [conversionVal, format_str, time_zone, scale]
        for i in range(len(args)):
            if isinstance(args[i], types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name.lower()}_util",
                    ["conversionVal", "format_str", "time_zone", "scale"],
                    i,
                )

        func_text = "def impl(conversionVal, format_str, time_zone, scale):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{func_name.lower()}_util(conversionVal, format_str, time_zone, scale)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def _install_timestamp_cast_overloads():
    timestamp_cast_fns = [
        ("TO_TIMESTAMP", to_timestamp, to_timestamp_util, True),
        ("TRY_TO_TIMESTAMP", try_to_timestamp, try_to_timestamp_util, False),
    ]
    for func_name, func, util_func, error_on_fail in timestamp_cast_fns:
        overload(func)(create_timestamp_cast_func(func_name))
        overload(util_func)(create_timestamp_cast_util(func_name, error_on_fail))


_install_timestamp_cast_overloads()


@numba.generated_jit(nopython=True)
def to_binary(arr):
    """Handles cases where TO_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_binary_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return to_binary_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def try_to_binary(arr):
    """Handles cases where TRY_TO_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.try_to_binary_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return try_to_binary_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def to_char(arr):
    """Handles cases where TO_CHAR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.to_char", ["arr"], 0)

    def impl(arr):  # pragma: no cover
        return to_char_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def to_char_util(arr):
    """A dedicated kernel for the SQL function TO_CHAR which takes in a
    number (or column) and returns a string representation of it.

    Args:
        arr (numerical array/series/scalar): the number(s) being operated on
        opt_fmt_str (string array/series/scalar): the format string(s) to use

    Returns:
        string series/scalar: the string representation of the number(s) with the
        specified null handling rules
    """
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]

    if is_str_arr_type(arr):
        # Strings are unchanged.
        return lambda arr: arr  # pragma: no cover
    # TODO [BE-3744]: support binary data for to_char
    elif is_valid_binary_arg(arr):
        # currently only support hex encoding
        scalar_text = "with bodo.objmode(r=bodo.string_type):\n"
        scalar_text += "  r = arg0.hex()\n"
        scalar_text += "res[i] = r"
    elif is_valid_time_arg(arr):
        # Currently, time -> string conversions always use the default format
        # of HH:MM:SS (1 digits are always extended to 2, and sub-second units
        # are ignored)
        scalar_text = "res[i] = format(arg0.hour, '0>2') + ':' + format(arg0.minute, '0>2') + ':' + format(arg0.second, '0>2')"
    elif isinstance(arr, bodo.TimeType) or (
        bodo.utils.utils.is_array_typ(arr) and isinstance(arr.dtype, bodo.TimeType)
    ):
        scalar_text = (
            "h_str = str(arg0.hour) if arg0.hour > 10 else '0' + str(arg0.hour)\n"
        )
        scalar_text += (
            "m_str = str(arg0.minute) if arg0.minute > 10 else '0' + str(arg0.minute)\n"
        )
        scalar_text += (
            "s_str = str(arg0.second) if arg0.second > 10 else '0' + str(arg0.second)\n"
        )
        scalar_text += "ms_str = str(arg0.millisecond) if arg0.millisecond > 100 else ('0' + str(arg0.millisecond) if arg0.millisecond > 10 else '00' + str(arg0.millisecond))\n"
        scalar_text += "us_str = str(arg0.microsecond) if arg0.microsecond > 100 else ('0' + str(arg0.microsecond) if arg0.microsecond > 10 else '00' + str(arg0.microsecond))\n"
        scalar_text += "ns_str = str(arg0.nanosecond) if arg0.nanosecond > 100 else ('0' + str(arg0.nanosecond) if arg0.nanosecond > 10 else '00' + str(arg0.nanosecond))\n"
        scalar_text += "part_str = h_str + ':' + m_str + ':' + s_str\n"
        scalar_text += "if arg0.nanosecond > 0:\n"
        scalar_text += "  part_str = part_str + '.' + ms_str + us_str + ns_str\n"
        scalar_text += "elif arg0.microsecond > 0:\n"
        scalar_text += "  part_str = part_str + '.' + ms_str + us_str\n"
        scalar_text += "elif arg0.millisecond > 0:\n"
        scalar_text += "  part_str = part_str + '.' + ms_str\n"
        scalar_text += "res[i] = part_str"
    elif is_valid_timedelta_arg(arr):
        scalar_text = "v = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg0)\n"
        scalar_text += "with bodo.objmode(r=bodo.string_type):\n"
        scalar_text += "    r = str(v)\n"
        scalar_text += "res[i] = r"
    elif is_valid_datetime_or_date_arg(arr):
        if is_valid_tz_aware_datetime_arg(arr):
            # strftime returns (-/+) HHMM for UTC offset, when the default Bodo
            # timezone format is (-/+) HH:MM. So we must manually insert a ":" character
            scalar_text = "tz_raw = arg0.strftime('%z')\n"
            scalar_text += 'tz = tz_raw[:3] + ":" + tz_raw[3:]\n'
            scalar_text += "res[i] = arg0.isoformat(' ') + tz\n"
        else:
            scalar_text = "res[i] = pd.Timestamp(arg0).isoformat(' ')\n"
    elif is_valid_float_arg(arr):
        scalar_text = "if np.isinf(arg0):\n"
        scalar_text += "  res[i] = 'inf' if arg0 > 0 else '-inf'\n"
        # currently won't use elif branch since np.nan is caught by
        # propagate_null[0] being True, presently
        # TODO [BE-3491]: treat NaNs and nulls differently
        scalar_text += "elif np.isnan(arg0):\n"
        scalar_text += "  res[i] = 'NaN'\n"
        scalar_text += "else:\n"
        scalar_text += "  res[i] = str(arg0)"
    elif is_valid_boolean_arg(arr):
        scalar_text = "res[i] = 'true' if arg0 else 'false'"
    else:
        int_types = {
            8: np.int8,
            16: np.int16,
            32: np.int32,
            64: np.int64,
        }
        if is_valid_int_arg(arr):
            if hasattr(arr, "dtype"):
                bw = arr.dtype.bitwidth
            else:
                bw = arr.bitwidth
            scalar_text = f"if arg0 == {np.iinfo(int_types[bw]).min}:\n"
            scalar_text += f"  res[i] = '{np.iinfo(int_types[bw]).min}'\n"
            scalar_text += "else:\n"
            scalar_text += "  res[i] = str(arg0)"
        else:
            scalar_text = "res[i] = str(arg0)"

    out_dtype = bodo.string_array_type

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def make_to_double(_try):
    """Generate utility functions to unopt TO_DOUBLE arguments"""

    @numba.generated_jit(nopython=True)
    def func(val, optional_format_string):
        """Handles cases where TO_DOUBLE receives optional arguments and forwards
        to the appropriate version of the real implementation"""
        args = [val, optional_format_string]
        for i in range(2):
            if isinstance(val, types.optional):  # pragma: no cover
                return unopt_argument(
                    "bodo.libs.bodosql_array_kernels.to_double",
                    ["val", "optional_format_string"],
                    i,
                )

        def impl(val, optional_format_string):  # pragma: no cover
            return to_double_util(val, optional_format_string, numba.literally(_try))

        return impl

    return func


try_to_double = make_to_double(True)
to_double = make_to_double(False)


@register_jitable
def is_string_numeric(expr):  # pragma: no cover
    """Determines whether a string represents a valid Snowflake numeric,
    following the spec [+-][digits][.digits][e[+-]digits]
    Reference: https://docs.snowflake.com/en/sql-reference/data-types-numeric.html#numeric-constants

    Args
        expr (str): String to validate

    Returns True iff expr is a valid numeric constant
    """
    if len(expr) == 0:
        return False
    i = 0

    # [+-]
    if i < len(expr) and (expr[i] == "+" or expr[i] == "-"):
        i += 1

    # Early exit for special cases
    if expr[i:].lower() in ("nan", "inf", "infinity"):
        return True

    # [digits]
    has_digits = False
    while i < len(expr) and expr[i].isdigit():
        has_digits = True
        i += 1

    # [.digits]
    if i < len(expr) and expr[i] == ".":
        i += 1
    while i < len(expr) and expr[i].isdigit():
        has_digits = True
        i += 1

    if not has_digits:
        return False

    # [e[+-]digits]
    if i < len(expr) and (expr[i] == "e" or expr[i] == "E"):
        i += 1

        if i < len(expr) and (expr[i] == "+" or expr[i] == "-"):
            i += 1

        has_digits = False
        while i < len(expr) and expr[i].isdigit():
            has_digits = True
            i += 1

        if not has_digits:
            return False

    return i == len(expr)


@numba.generated_jit(nopython=True)
def to_double_util(val, optional_format_string, _try=False):
    """A dedicated kernel for the SQL function TO_DOUBLE which takes in a
    number (or column) and converts it to float64.


    Args:
        val (numerical array/series/scalar): the number(s) being operated on
        optional_format_string (string array/series/scalar): format string. Only valid if arr is a string
        _try (bool): whether to return NULL (iff true) on error or raise an exception

    Returns:
        double series/scalar: the double value of the number(s) with the
        specified null handling rules
    """
    verify_string_numeric_arg(val, "TO_DOUBLE and TRY_TO_DOUBLE", "val")
    verify_string_arg(
        optional_format_string, "TO_DOUBLE and TRY_TO_DOUBLE", "optional_format_string"
    )
    is_string = is_valid_string_arg(val)
    is_float = is_valid_float_arg(val)
    is_int = is_valid_int_arg(val)
    _try = get_overload_const_bool(_try)

    if _try:  # pragma: no cover
        on_fail = "bodo.libs.array_kernels.setna(res, i)\n"
    else:
        if is_string:
            err_msg = "string must be a valid numeric expression"
        else:  # pragma: no cover
            err_msg = "value must be a valid numeric expression"
        on_fail = (
            f"""raise ValueError("invalid value for double conversion: {err_msg}")"""
        )

    # Format string not supported
    if not is_overload_none(optional_format_string):  # pragma: no cover
        raise raise_bodo_error(
            f"Internal error: Format string not supported for TO_DOUBLE / TRY_TO_DOUBLE"
        )
    elif is_string:
        scalar_text = "if is_string_numeric(arg0):\n"
        scalar_text += f"  res[i] = np.float64(arg0)\n"
        scalar_text += f"else:\n"
        scalar_text += f"  {on_fail}\n"
    elif is_float:  # pragma: no cover
        scalar_text = f"res[i] = arg0\n"
    elif is_int:  # pragma: no cover
        scalar_text = f"res[i] = np.float64(arg0)\n"
    else:  # pragma: no cover
        raise raise_bodo_error(
            f"Internal error: unsupported type passed to to_double_util for argument val: {val}"
        )

    arg_names = ["val", "optional_format_string", "_try"]
    arg_types = [val, optional_format_string, _try]
    propagate_null = [True, False, False]

    if bodo.libs.float_arr_ext._use_nullable_float:
        out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)
    else:  # pragma: no cover
        out_dtype = types.Array(bodo.float64, 1, "C")

    extra_globals = {
        "is_string_numeric": is_string_numeric,
    }
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


@register_jitable
def convert_sql_date_format_str_to_py_format(val):  # pragma: no cover
    """Helper fn for the TO_DATE/TO_TIMESTAMP fns. This fn takes a format string
    in SQL syntax, and converts it to the python syntax.
    SQL syntax reference: https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    Python syntax reference: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    """

    # TODO: https://bodo.atlassian.net/browse/BE-3614
    raise RuntimeError(
        "Converting to date/timestamp values with format strings not currently supported"
    )


@numba.generated_jit(nopython=True)
def number_to_datetime(val):
    """Helper fn for the snowflake TO_DATE fns. For this fns, argument is integer or float.

    If the format of the input parameter is a string that contains an number:
    After the string is converted to an number (if needed), the number is treated as a number of seconds, milliseconds, microseconds, or nanoseconds after the start of the Unix epoch (1970-01-01 00:00:00.000000000 UTC).
    If the number is less than 31536000000 (the number of milliseconds in a year), then the value is treated as a number of seconds.
    If the value is greater than or equal to 31536000000 and less than 31536000000000, then the value is treated as milliseconds.
    If the value is greater than or equal to 31536000000000 and less than 31536000000000000, then the value is treated as microseconds.
    If the value is greater than or equal to 31536000000000000, then the value is treated as nanoseconds.

    See https://docs.snowflake.com/en/sql-reference/functions/to_date.html#usage-notes

    This function does NOT floor the resulting datetime (relies on calling fn to do so if needed).

    Note, for negatives, the absolute value is taken when choosing the unit.
    """

    def impl(val):  # pragma: no cover
        if abs(val) < 31536000000:
            retval = pd.to_datetime(val, unit="s")
        elif abs(val) < 31536000000000:
            retval = pd.to_datetime(val, unit="ms")
        elif abs(val) < 31536000000000000:
            retval = pd.to_datetime(val, unit="us")
        else:
            retval = pd.to_datetime(val, unit="ns")
        return retval

    return impl


@register_jitable
def pd_to_datetime_error_checked(
    val,
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):  # pragma: no cover
    """Helper fn that determines if we have a parsable datetime string, by calling
    pd.to_datetime in objmode, which returns a tuple (success flag, value). If
    the success flag evaluates to True, then the paired value is the correctly parsed timestamp, otherwise  the paired value is a dummy timestamp.
    """
    # Manual check for invalid date strings to match behavior of Snowflake
    # since Pandas accepts dates in more formats. The legal patterns for
    # the date component in Snowflake:
    # YYYY/MM/DD
    # YYYY-MM-DD
    # YYYY-M-DD
    # YYYY-M-D
    # YYYY-MM-D
    if val is not None:
        # account for case where val has a time component by finding everything
        # before the first character that is not a digit/dash/slash
        split_index = 0
        for i in range(len(val)):
            if not (val[i].isdigit() or val[i] in "/-"):
                break
            split_index += 1
        date_comp = val[:split_index]

        # The legal range is 8, 9 or 10 characters
        if len(date_comp) == 8 or len(date_comp) == 9:
            # 8/9 characters: must be YYYY-M-D, YYYY-M-DD or YYYY-M-DD
            if date_comp.count("-") != 2:
                return (False, None)
        elif len(date_comp) == 10:
            # 10 characters: must be YYYY-MM-DD or YYYY/MM/DD
            if ~((date_comp.count("/") == 2) ^ (date_comp.count("-") == 2)):
                return (False, None)
        else:
            return (False, None)

    with numba.objmode(ret_val="pd_timestamp_tz_naive_type", success_flag="bool_"):
        success_flag = True
        ret_val = pd.Timestamp(0)

        tmp = pd.to_datetime(
            val,
            errors="coerce",
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
        if pd.isna(tmp):
            success_flag = False
        else:
            ret_val = tmp

    return (success_flag, ret_val)


def cast_tz_naive_to_tz_aware(arr, tz):  # pragma: no cover
    pass


@overload(cast_tz_naive_to_tz_aware, no_unliteral=True)
def overload_cast_tz_naive_to_tz_aware(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_tz_naive_to_tz_aware(): 'tz' must be a literal value")
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_tz_naive_to_tz_aware",
            ["arr", "tz"],
            0,
        )

    def impl(arr, tz):  # pragma: no cover
        return cast_tz_naive_to_tz_aware_util(arr, tz)

    return impl


def cast_tz_naive_to_tz_aware_util(arr, tz):  # pragma: no cover
    pass


@overload(cast_tz_naive_to_tz_aware_util, no_unliteral=True)
def overload_cast_tz_naive_to_tz_aware_util(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_tz_naive_to_tz_aware(): 'tz' must be a literal value")
    verify_datetime_arg(arr, "cast_tz_naive_to_tz_aware", "arr")
    arg_names = ["arr", "tz"]
    arg_types = [arr, tz]
    # tz can never be null
    propagate_null = [True, False]
    # If we have an array input we must cast to a timestamp
    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr)
        else ""
    )
    scalar_text = f"res[i] = {box_str}(arg0).tz_localize(arg1)"
    tz = get_literal_value(tz)
    out_dtype = bodo.DatetimeArrayType(tz)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def cast_tz_aware_to_tz_naive(arr, normalize):  # pragma: no cover
    pass


@overload(cast_tz_aware_to_tz_naive, no_unliteral=True)
def overload_cast_tz_aware_to_tz_naive(arr, normalize):
    if not is_overload_constant_bool(normalize):
        raise_bodo_error(
            "cast_tz_aware_to_tz_naive(): 'normalize' must be a literal value"
        )
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_tz_aware_to_tz_naive",
            ["arr", "normalize"],
            0,
        )

    def impl(arr, normalize):  # pragma: no cover
        return cast_tz_aware_to_tz_naive_util(arr, normalize)

    return impl


def cast_tz_aware_to_tz_naive_util(arr, normalize):  # pragma: no cover
    pass


@overload(cast_tz_aware_to_tz_naive_util, no_unliteral=True)
def overload_cast_tz_aware_to_tz_naive_util(arr, normalize):
    if not is_overload_constant_bool(normalize):
        raise_bodo_error(
            "cast_tz_aware_to_tz_naive(): 'normalize' must be a literal value"
        )
    normalize = get_overload_const_bool(normalize)
    verify_datetime_arg_require_tz(arr, "cast_tz_aware_to_tz_naive", "arr")
    arg_names = ["arr", "normalize"]
    arg_types = [arr, normalize]
    # normalize can never be null
    propagate_null = [True, False]
    # If we have an array we must cast the output to a datetime64
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(arr)
        else ""
    )
    scalar_text = ""
    if normalize:
        scalar_text += (
            "ts = pd.Timestamp(year=arg0.year, month=arg0.month, day=arg0.day)\n"
        )
    else:
        scalar_text += "ts = arg0.tz_localize(None)\n"
    scalar_text += f"res[i] = {unbox_str}(ts)"
    out_dtype = types.Array(bodo.datetime64ns, 1, "C")
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def cast_str_to_tz_aware(arr, tz):  # pragma: no cover
    pass


@overload(cast_str_to_tz_aware, no_unliteral=True)
def overload_cast_str_to_tz_aware(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_str_to_tz_aware(): 'tz' must be a literal value")
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_str_to_tz_aware",
            ["arr", "tz"],
            0,
        )

    def impl(arr, tz):  # pragma: no cover
        return cast_str_to_tz_aware_util(arr, tz)

    return impl


def cast_str_to_tz_aware_util(arr, tz):  # pragma: no cover
    pass


@overload(cast_str_to_tz_aware_util, no_unliteral=True)
def overload_cast_str_to_tz_aware_util(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_str_to_tz_aware(): 'tz' must be a literal value")
    verify_string_arg(arr, "cast_str_to_tz_aware", "arr")
    arg_names = ["arr", "tz"]
    arg_types = [arr, tz]
    # tz can never be null
    propagate_null = [True, False]
    # Note: pd.to_datetime doesn't support tz as an argument.
    scalar_text = f"res[i] = pd.to_datetime(arg0).tz_localize(arg1)"
    tz = get_literal_value(tz)
    out_dtype = bodo.DatetimeArrayType(tz)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def to_binary_util(arr):  # pragma: no cover
    pass


def try_to_binary_util(arr):  # pragma: no cover
    pass


# TODO ([BE-4344]): implement and test to_binary with other formats
def create_to_binary_util_overload(fn_name, error_on_fail):
    def impl(arr):  # pragma: no cover
        verify_string_binary_arg(fn_name, arr, "arr")
        if error_on_fail:
            fail_str = 'raise ValueError("invalid value for binary (HEX) conversion")'
        else:
            fail_str = "bodo.libs.array_kernels.setna(res, i)"
        if is_valid_string_arg(arr):
            # If the input is string data, make sure there are an even number of characters
            # and all of them are hex characters
            scalar_text = "failed = len(arg0) % 2 != 0\n"
            scalar_text += "if not failed:\n"
            scalar_text += "  for char in arg0:\n"
            scalar_text += "    if char not in '0123456789ABCDEFabcdef':\n"
            scalar_text += "      failed = True\n"
            scalar_text += "      break\n"
            scalar_text += "if failed:\n"
            scalar_text += f"  {fail_str}\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = bodo.libs.binary_arr_ext.bytes_fromhex(arg0)"
        else:
            # If the input is binary data, just copy it directly
            scalar_text = "res[i] = arg0"
        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]
        out_dtype = bodo.binary_array_type
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
        )

    return impl


def _install_to_binary_funcs():
    funcs = [
        ("to_binary", to_binary_util, True),
        ("try_to_binary", try_to_binary_util, False),
    ]
    for fn_name, func, error_on_fail in funcs:
        overload(func)(create_to_binary_util_overload(fn_name, error_on_fail))


_install_to_binary_funcs()


@numba.generated_jit(nopython=True)
def to_number(expr):  # pragma: no cover
    """Handle TO_NUMBER and it's variants."""
    if isinstance(expr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_snowflake_conversion_array_kernels.to_number_util",
            ["expr"],
            None,
        )

    def impl(expr):  # pragma: no cover
        return to_number_util(expr, numba.literally(False))

    return impl


@numba.generated_jit(nopython=True)
def try_to_number(expr):  # pragma: no cover
    """Handle TRY_TO_NUMBER and it's variants."""
    if isinstance(expr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_snowflake_conversion_array_kernels.to_number_util",
            ["expr"],
            None,
        )

    def impl(expr):  # pragma: no cover
        return to_number_util(expr, numba.literally(True))

    return impl


@numba.generated_jit(nopython=True)
def _is_string_numeric(expr):  # pragma: no cover
    """Check if a string is numeric."""

    def impl(expr):
        if len(expr) == 0:
            return False

        if expr[0] == "-":
            expr = expr[1:]

        if expr.count(".") > 1:
            return False

        expr = expr.replace(".", "")

        if len(expr) == 0:
            return False

        if not expr.isdigit():
            return False

        return True

    return impl


@numba.generated_jit(nopython=True)
def to_number_util(expr, _try=False):  # pragma: no cover
    """Equivalent to the SQL [TRY] TO_NUMBER/TO_NUMERIC/TO_DECIMAL function.
    With the default args, this converts the input to a 64-bit integer.
    TODO: support non-default `scale` arg, which could result in float.

    Args:
        expr (numeric or string series/scalar): the number/string to convert to a number of type int64

    Returns:
        numeric series/scalar: the converted number
    """
    arg_names = ["expr", "_try"]
    arg_types = [expr, _try]
    propagate_null = [True]

    is_string = is_valid_string_arg(expr)
    if not is_string:
        verify_int_float_arg(expr, "TO_NUMBER", "expr")

    _try = get_overload_const_bool(_try)

    out_dtype = bodo.IntegerArrayType(types.int64)

    if is_string:
        scalar_text = (
            "if bodo.libs.bodosql_snowflake_conversion_array_kernels._is_string_numeric(arg0):\n"
            "  res[i] = np.int64(np.float64(arg0))\n"
            "else:\n"
        )
        if _try:
            scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
        else:
            scalar_text += (
                "  raise ValueError('unable to convert string literal to number')\n"
            )
    else:
        scalar_text = "res[i] = np.int64(arg0)"

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
