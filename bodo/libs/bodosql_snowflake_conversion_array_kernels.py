# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements a number of array kernels that handling casting functions for BodoSQL
"""

import numba
import pandas as pd
from numba.core import types
from numba.extending import register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    BodoError,
    dtype_to_array_type,
    get_overload_const_bool,
    is_overload_none,
    raise_bodo_error,
    to_nullable_type,
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
        raise BodoError("to_char(): binary input currently unsupported")

    if is_valid_datetime_or_date_arg(arr):
        scalar_text = "res[i] = pd.Timestamp(arg0).isoformat(' ')"
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


@register_jitable
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
    if val < 31536000000:
        retval = pd.to_datetime(val, unit="s")
    elif val < 31536000000000:
        retval = pd.to_datetime(val, unit="ms")
    elif val < 31536000000000000:
        retval = pd.to_datetime(val, unit="us")
    else:
        retval = pd.to_datetime(val, unit="ns")
    return retval


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

    with numba.objmode(ret_val="pd_timestamp_type", success_flag="bool_"):
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
def to_date_util(conversionVal, optionalConversionFormatString, errorOnFail):
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

    if errorOnFail:
        errorString = "raise ValueError('Invalid input while converting to date value')"
    else:
        errorString = "bodo.libs.array_kernels.setna(res, i)"

    verify_string_arg(
        optionalConversionFormatString,
        "TO_DATE and TRY_TO_DATE",
        "optionalConversionFormatString",
    )

    is_out_arr = bodo.utils.utils.is_array_typ(
        conversionVal, True
    ) or bodo.utils.utils.is_array_typ(optionalConversionFormatString, True)

    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = "unbox_if_timestamp" if is_out_arr else ""

    # If the format string is specified, then arg0 must be string
    if not is_overload_none(optionalConversionFormatString):
        verify_string_arg(
            conversionVal, "TO_DATE and TRY_TO_DATE", "optionalConversionFormatString"
        )
        scalar_text = "py_format_str = convert_sql_date_format_str_to_py_format(arg1)\n"
        scalar_text += "was_successful, tmp_val = pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
        scalar_text += "if not was_successful:\n"
        scalar_text += f"   {errorString}\n"
        scalar_text += "else:\n"
        scalar_text += f"   res[i] = {unbox_str}(tmp_val.floor(freq='D'))\n"

    # NOTE: gen_vectorized will automatically map this function over the values dictionary
    # of a dict encoded string array instead of decoding it whenever possible
    elif is_valid_string_arg(conversionVal):
        """
        If no format string is specified, snowflake will use attempt to parse the string according to these date formats:
        https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
        handled by pd.to_datetime() in Bodo jit code.

        It will also check if the string is covnertable to int, IE '12345' or '-4321'"""

        # Conversion needs to be done incase arg0 is unichr array
        scalar_text = "arg0 = str(arg0)\n"
        scalar_text += "if (arg0.isnumeric() or (len(arg0) > 1 and arg0[0] == '-' and arg0[1:].isnumeric())):\n"
        scalar_text += f'   res[i] = {unbox_str}(int_to_datetime(np.int64(arg0)).floor(freq="D"))\n'

        scalar_text += "else:\n"
        scalar_text += (
            "   was_successful, tmp_val = pd_to_datetime_error_checked(arg0)\n"
        )
        scalar_text += "   if not was_successful:\n"
        scalar_text += f"      {errorString}\n"
        scalar_text += "   else:\n"
        scalar_text += f"      res[i] = {unbox_str}(tmp_val.floor(freq='D'))\n"

    elif isinstance(conversionVal, types.Integer) or (
        bodo.utils.utils.is_array_typ(conversionVal, True)
        and isinstance(conversionVal.dtype, types.Integer)
    ):
        scalar_text = f'res[i] = {unbox_str}(int_to_datetime(arg0).floor(freq="D"))\n'

    elif is_valid_datetime_or_date_arg(conversionVal):
        scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0).floor(freq='D'))\n"
    else:
        raise raise_bodo_error(
            f"Internal error: unsupported type passed to to_date_util for argument conversionVal: {conversionVal}"
        )

    arg_names = ["conversionVal", "optionalConversionFormatString", "errorOnFail"]
    arg_types = [conversionVal, optionalConversionFormatString, errorOnFail]
    propagate_null = [True, False, False]

    out_dtype = to_nullable_type(dtype_to_array_type(bodo.datetime64ns))

    extra_globals = {
        "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
        "int_to_datetime": int_to_datetime,
        "convert_sql_date_format_str_to_py_format": convert_sql_date_format_str_to_py_format,
        "unbox_if_timestamp": bodo.utils.conversion.unbox_if_timestamp,
    }
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )
