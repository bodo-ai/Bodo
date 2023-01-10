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
    is_overload_none,
    raise_bodo_error,
)


@numba.generated_jit(nopython=True)
def try_to_boolean(arr):
    """Handles cases where TO_BOOLEAN receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.to_boolean", ["arr"], 0)

    def impl(arr):
        return to_boolean_util(arr, numba.literally(True))

    return impl


@numba.generated_jit(nopython=True)
def to_boolean(arr):
    """Handles cases where TO_BOOLEAN receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.to_boolean", ["arr"], 0)

    def impl(arr):
        return to_boolean_util(arr, numba.literally(False))

    return impl


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


@numba.generated_jit(nopython=True)
def try_to_date(conversionVal, optionalConversionFormatString):

    args = [conversionVal, optionalConversionFormatString]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.try_to_date",
                ["conversionVal", "optionalConversionFormatString"],
                i,
            )

    def impl(conversionVal, optionalConversionFormatString):  # pragma: no cover
        return to_date_util(
            conversionVal, optionalConversionFormatString, numba.literally(False)
        )

    return impl


@numba.generated_jit(nopython=True)
def to_date(conversionVal, optionalConversionFormatString):

    args = [conversionVal, optionalConversionFormatString]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.to_date",
                ["conversionVal", "optionalConversionFormatString"],
                i,
            )

    def impl(conversionVal, optionalConversionFormatString):  # pragma: no cover
        return to_date_util(
            conversionVal, optionalConversionFormatString, numba.literally(True)
        )

    return impl


@numba.generated_jit(nopython=True)
def to_char(arr):
    """Handles cases where TO_CHAR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):
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

    # TODO [BE-2744]: support binary data for to_char
    if is_valid_binary_arg(arr):
        # currently only support hex encoding
        scalar_text = "with bodo.objmode(r=bodo.string_type):\n"
        scalar_text += "  r = arg0.hex()\n"
        scalar_text += "res[i] = r"
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


@register_jitable
def convert_sql_date_format_str_to_py_format(val):  # pragma: no cover
    """Helper fn for the TO_DATE fns. This fn takes a format string in SQL syntax, and converts it to
    the python syntax.
    SQL syntax reference: https://docs.snowflake.com/en/sql-reference/functions-conversion.html#label-date-time-format-conversion
    Python syntax reference: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    """

    # TODO: https://bodo.atlassian.net/browse/BE-3614
    raise RuntimeError(
        "Converting to date values with format strings not currently supported"
    )


@numba.generated_jit(nopython=True)
def int_to_datetime(val):
    """Helper fn for the snowflake TO_DATE fns. For this fns, argument is integer.

    If the format of the input parameter is a string that contains an integer:
    After the string is converted to an integer (if needed), the integer is treated as a number of seconds, milliseconds, microseconds, or nanoseconds after the start of the Unix epoch (1970-01-01 00:00:00.000000000 UTC).
    If the integer is less than 31536000000 (the number of milliseconds in a year), then the value is treated as a number of seconds.
    If the value is greater than or equal to 31536000000 and less than 31536000000000, then the value is treated as milliseconds.
    If the value is greater than or equal to 31536000000000 and less than 31536000000000000, then the value is treated as microseconds.
    If the value is greater than or equal to 31536000000000000, then the value is treated as nanoseconds.

    See https://docs.snowflake.com/en/sql-reference/functions/to_date.html#usage-notes

    This function does NOT floor the resulting datetime (relies on calling fn to do so if needed)
    """

    def impl(val):  # pragma: no cover
        if val < 31536000000:
            retval = pd.to_datetime(val, unit="s")
        elif val < 31536000000000:
            retval = pd.to_datetime(val, unit="ms")
        elif val < 31536000000000000:
            retval = pd.to_datetime(val, unit="us")
        else:
            retval = pd.to_datetime(val, unit="ns")
        return retval

    return impl


@numba.generated_jit(nopython=True)
def float_to_datetime(val):
    """Helper fn for the snowflake TO_DATE fns. For this fns, argument is integer.

    If the format of the input parameter is a string that contains an integer:
    After the string is converted to an integer (if needed), the integer is treated as a number of seconds, milliseconds, microseconds, or nanoseconds after the start of the Unix epoch (1970-01-01 00:00:00.000000000 UTC).
    If the integer is less than 31536000000 (the number of milliseconds in a year), then the value is treated as a number of seconds.
    If the value is greater than or equal to 31536000000 and less than 31536000000000, then the value is treated as milliseconds.
    If the value is greater than or equal to 31536000000000 and less than 31536000000000000, then the value is treated as microseconds.
    If the value is greater than or equal to 31536000000000000, then the value is treated as nanoseconds.

    See https://docs.snowflake.com/en/sql-reference/functions/to_date.html#usage-notes

    This function does NOT floor the resulting datetime (relies on calling fn to do so if needed)
    """

    def impl(val):  # pragma: no cover
        if val < 31536000000:
            retval = pd.Timestamp(val, unit="s")
        elif val < 31536000000000:
            retval = pd.Timestamp(val, unit="ms")
        elif val < 31536000000000000:
            retval = pd.Timestamp(val, unit="us")
        else:
            retval = pd.Timestamp(val, unit="ns")
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
    # since Pandas accepts dates like the following: "2020", "2020-01"
    if val is not None:
        # account for case where val is a timestamp
        val_arg0 = val.split(" ")[0]

        # Check the number of characters to prevent those invalid cases
        # YYYY/MM/DD, YYYY-MM-DD, etc. all have exactly 10 characters
        # With timestamp information, they have even greater.
        if len(val_arg0) < 10:
            return (False, None)
        else:
            # Check that the number of "/", and "-" are either 0 or 2
            slash_flag = val_arg0.count("/") in [0, 2]
            dash_flag = val_arg0.count("-") in [0, 2]

            if not (slash_flag and dash_flag):
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


@numba.generated_jit(nopython=True)
def to_date_util(
    conversionVal, optionalConversionFormatString, errorOnFail, _keep_time=False
):
    """A dedicated kernel for the SQL function DATE, TO_DATE and TRY_TO_DATE which attempts
    to convert arg0 into a date value. Arg0 can be string, integer, or a datetime type. If
    arg0 is a string, can optionally accept an arg1
    string value used for parsing. As we don't support proper date types in bodosql, these
    functions return floored dt64 values. ErrorOnFail is a boolean literal, which determines if
    the fn returns null, or errors when attempting conversion (the difference between TO_DATE
    and TRY_TO_DATE).

    Args:
        conversionVal (string, datetime/timestamp value, or integer array/series/scalar): the value to be converted to date
        optionalConversionFormatString (string array/series/scalar): format string. Only valid if arg0 is string
        errorOnFail (boolean literal): The null/error handling behavior. True for TO_DATE, which errors on failure
        False for TRY_TO_DATE, which instead returns null.

    Returns:
        datetime series/scalar: the converted date values
    """
    errorOnFail = get_overload_const_bool(errorOnFail)
    _keep_time = get_overload_const_bool(_keep_time)

    if errorOnFail:
        errorString = "raise ValueError('Invalid input while converting to date value')"
    else:
        errorString = "bodo.libs.array_kernels.setna(res, i)"

    if _keep_time:
        floor_str = ""
    else:
        floor_str = ".normalize()"

    verify_string_arg(
        optionalConversionFormatString,
        "TO_DATE and TRY_TO_DATE",
        "optionalConversionFormatString",
    )

    is_out_arr = bodo.utils.utils.is_array_typ(
        conversionVal, True
    ) or bodo.utils.utils.is_array_typ(optionalConversionFormatString, True)

    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = "unbox_if_tz_naive_timestamp" if is_out_arr else ""

    # If the format string is specified, then arg0 must be string
    if not is_overload_none(optionalConversionFormatString):
        verify_string_arg(
            conversionVal, "TO_DATE and TRY_TO_DATE", "optionalConversionFormatString"
        )
        scalar_text = "py_format_str = convert_sql_date_format_str_to_py_format(arg1)\n"
        scalar_text += "was_successful, tmp_val = pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
        scalar_text += "if not was_successful:\n"
        scalar_text += f"  {errorString}\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = {unbox_str}(tmp_val{floor_str})\n"

    # NOTE: gen_vectorized will automatically map this function over the values dictionary
    # of a dict encoded string array instead of decoding it whenever possible
    elif is_valid_string_arg(conversionVal):
        """
        If no format string is specified, snowflake will use attempt to parse the string according to these date formats:
        https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
        handled by pd.to_datetime() in Bodo jit code.

        It will also check if the string is convertable to int, IE '12345' or '-4321'"""

        # Conversion needs to be done incase arg0 is unichr array
        scalar_text = "arg0 = str(arg0)\n"
        scalar_text += "if (arg0.isnumeric() or (len(arg0) > 1 and arg0[0] == '-' and arg0[1:].isnumeric())):\n"
        scalar_text += (
            f"   res[i] = {unbox_str}(int_to_datetime(np.int64(arg0)){floor_str})\n"
        )

        scalar_text += "else:\n"
        scalar_text += (
            "   was_successful, tmp_val = pd_to_datetime_error_checked(arg0)\n"
        )
        scalar_text += "   if not was_successful:\n"
        scalar_text += f"      {errorString}\n"
        scalar_text += "   else:\n"
        scalar_text += f"      res[i] = {unbox_str}(tmp_val{floor_str})\n"

    elif is_valid_int_arg(conversionVal):
        scalar_text = f"res[i] = {unbox_str}(int_to_datetime(arg0){floor_str})\n"

    elif is_valid_float_arg(conversionVal):
        scalar_text = f"res[i] = {unbox_str}(float_to_datetime(arg0){floor_str})\n"

    elif is_valid_datetime_or_date_arg(conversionVal):
        scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0){floor_str})\n"
    elif is_valid_tz_aware_datetime_arg(conversionVal):
        scalar_text = f"res[i] = arg0{floor_str}\n"
    else:
        raise raise_bodo_error(
            f"Internal error: unsupported type passed to to_date_util for argument conversionVal: {conversionVal}"
        )

    arg_names = [
        "conversionVal",
        "optionalConversionFormatString",
        "errorOnFail",
        "_keep_time",
    ]
    arg_types = [conversionVal, optionalConversionFormatString, errorOnFail, _keep_time]
    propagate_null = [True, False, False, False]

    # Determine the output dtype. If the input is a tz-aware Timestamp
    # then we have a tz-aware output. Otherwise we output datetime64ns.
    if isinstance(conversionVal, bodo.DatetimeArrayType) or (
        isinstance(conversionVal, bodo.PandasTimestampType)
        and conversionVal.tz is not None
    ):
        out_dtype = bodo.DatetimeArrayType(conversionVal.tz)
    else:
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")

    extra_globals = {
        "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
        "int_to_datetime": int_to_datetime,
        "float_to_datetime": float_to_datetime,
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
