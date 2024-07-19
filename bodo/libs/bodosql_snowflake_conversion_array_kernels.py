import datetime
from typing import Optional

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.hiframes.datetime_date_ext import DatetimeDateArrayType
from bodo.hiframes.timestamptz_ext import TimestampTZ
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    BodoError,
    get_literal_value,
    get_overload_const_bool,
    get_overload_const_int,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ


def make_to_boolean(_try):
    """Generate utility functions to unopt TO_BOOLEAN arguments"""

    func_name = "to_boolean"
    if _try:
        func_name = "try_to_boolean"

    @numba.generated_jit(nopython=True)
    def func_impl(arr, dict_encoding_state, func_id):
        """Handles cases where TO_BOOLEAN receives optional arguments and forwards
        to the appropriate version of the real implementation"""
        if isinstance(arr, types.optional):  # pragma: no cover
            return unopt_argument(
                f"bodo.libs.bodosql_array_kernels.{func_name}",
                ["arr", "dict_encoding_state", "func_id"],
                0,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

        def impl(arr, dict_encoding_state, func_id):  # pragma: no cover
            return to_boolean_util(
                arr, numba.literally(_try), dict_encoding_state, func_id
            )

        return impl

    @numba.njit
    def func(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return func_impl(arr, dict_encoding_state, func_id)

    return func


try_to_boolean = make_to_boolean(True)
to_boolean = make_to_boolean(False)


@numba.generated_jit(nopython=True, no_unliteral=True)
def to_boolean_util(arr, _try, dict_encoding_state, func_id):
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

    arg_names = ["arr", "_try", "dict_encoding_state", "func_id"]
    arg_types = [arr, _try, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False]

    prefix_code = None
    if is_string:
        prefix_code = "true_vals = {'true', 't', 'yes', 'y', 'on', '1'}\n"
        prefix_code += "false_vals = {'false', 'f', 'no', 'n', 'off', '0'}"
    if is_string:
        scalar_text = "s = arg0.strip().lower()\n"
        scalar_text += f"is_true_val = s in true_vals\n"
        scalar_text += f"res[i] = is_true_val\n"
        scalar_text += f"if not (is_true_val or s in false_vals):\n"
        scalar_text += f"  {on_fail}\n"
    elif is_float:
        # TODO: fix this for float case (see above)
        # np.isnan should error here, but it will not reach because
        # NaNs will be caught since propagate_null[0] is True
        scalar_text = "if np.isinf(arg0) or np.isnan(arg0):\n"
        scalar_text += f"  {on_fail}\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = bool(arg0)\n"
    else:
        scalar_text = f"res[i] = bool(arg0)"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def to_date(
    conversion_val, format_str, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    return


def try_to_date(
    conversion_val, format_str, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    return


def to_date_util(
    conversion_val, format_str, dict_encoding_state, func_id
):  # pragma: no cover
    return


def try_to_date_util(
    conversion_val, format_str, dict_encoding_state, func_id
):  # pragma: no cover
    return


def create_date_cast_util(func, error_on_fail):
    """Creates an overload for a dedicated kernel for TO_DATE/TRY_TO_DATE
    Takes in 2 arguments: the name of the kernel being created and whether it should
    have an error when it has a failure (as opposed to outputting null),

    Returns an overload that accepts 2 arguments: the value to be converted
    (a series or scalar of multiple possible types), and an optional format string
    for cases where the input is a string.

    The full specification is noted here:
    https://docs.snowflake.com/en/sql-reference/functions/to_date.html
    """
    if error_on_fail:
        error_str = "raise ValueError('Invalid input while converting to date value')"
    else:
        error_str = "bodo.libs.array_kernels.setna(res, i)"

    def overload_impl(conversion_val, format_str, dict_encoding_state, func_id):
        verify_string_arg(format_str, func, "format_str")

        # When returning a scalar we return a pd.Timestamp type.
        is_out_arr = bodo.utils.utils.is_array_typ(
            conversion_val, True
        ) or bodo.utils.utils.is_array_typ(format_str, True)
        unbox_str = "unbox_if_tz_naive_timestamp" if is_out_arr else ""

        # If the format string is specified, then arg0 must be a string
        if not is_overload_none(format_str):
            verify_string_arg(conversion_val, func, "conversion_val")
            scalar_text = (
                "was_successful, tmp_val = to_date_error_checked(arg0, arg1)\n"
            )
            scalar_text += "if not was_successful:\n"
            scalar_text += f"  {error_str}\n"
            scalar_text += "else:\n"
            scalar_text += f"  res[i] = {unbox_str}(tmp_val.date())\n"

        # NOTE: gen_vectorized will automatically map this function over the values dictionary
        # of a dict encoded string array instead of decoding it whenever possible
        elif is_valid_string_arg(conversion_val):
            """
            If no format string is specified, attempt to parse the string according to these date formats:
            https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
            handled by pd.to_datetime() in Bodo jit code.

            It will also check if the string is convertible to int, IE '12345' or '-4321'
            """

            # Conversion needs to be done incase arg0 is unichr array
            scalar_text = "arg0 = str(arg0)\n"
            scalar_text += "if (arg0.isnumeric() or (len(arg0) > 1 and arg0[0] == '-' and arg0[1:].isnumeric())):\n"
            scalar_text += "   int_val = np.int64(arg0)\n"
            scalar_text += "   if int_val < 31536000000:\n"
            scalar_text += (
                f"      res[i] = {unbox_str}(pd.Timestamp(int_val, unit='s').date())\n"
            )
            scalar_text += "   elif int_val < 31536000000000:\n"
            scalar_text += (
                f"      res[i] = {unbox_str}(pd.Timestamp(int_val, unit='ms').date())\n"
            )
            scalar_text += "   elif int_val < 31536000000000000:\n"
            scalar_text += (
                f"      res[i] = {unbox_str}(pd.Timestamp(int_val, unit='us').date())\n"
            )
            scalar_text += "   else:\n"
            scalar_text += (
                f"      res[i] = {unbox_str}(pd.Timestamp(int_val, unit='ns').date())\n"
            )

            scalar_text += "else:\n"
            # Fast paths for regular date formats
            scalar_text += (
                "   was_successful, tmp_val = to_date_auto_error_checked(arg0)\n"
            )
            # Slower paths to accept timestamp values. Its unclear to exact rules that
            # are supported by Snowflake here.
            scalar_text += "   if not was_successful:\n"
            scalar_text += "     was_successful, tmp_timestamp_value = pd_to_datetime_error_checked(arg0)\n"
            scalar_text += "     tmp_val = tmp_timestamp_value.date()\n"
            scalar_text += "   if not was_successful:\n"
            scalar_text += f"        {error_str}\n"
            scalar_text += "   else:\n"
            scalar_text += f"      res[i] = tmp_val\n"

        # For date just assign equality
        elif is_valid_date_arg(conversion_val):
            scalar_text = f"res[i] = arg0\n"

        # If a tz-aware timestamp, extract the date
        elif is_valid_tz_aware_datetime_arg(conversion_val):
            scalar_text = f"res[i] = arg0.date()\n"

        # If a non-tz timestamp/datetime, round it down to the nearest day
        elif is_valid_datetime_or_date_arg(conversion_val):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0).date())\n"

        # If a tz timestamp, extract the date from the local timestamp
        elif is_valid_timestamptz_arg(conversion_val):
            scalar_text = f"res[i] = arg0.local_timestamp().date()\n"

        else:  # pragma: no cover
            raise raise_bodo_error(
                f"Internal error: unsupported type passed to to_date_util for argument conversion_val: {conversion_val}"
            )

        arg_names = ["conversion_val", "format_str", "dict_encoding_state", "func_id"]
        arg_types = [conversion_val, format_str, dict_encoding_state, func_id]
        propagate_null = [True, False, False, False]

        out_dtype = DatetimeDateArrayType()

        extra_globals = {
            "to_date_error_checked": to_date_error_checked,
            "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
            "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
            "to_date_auto_error_checked": to_date_auto_error_checked,
        }
        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            extra_globals=extra_globals,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_impl


def create_date_cast_func(func_name):
    """Takes in a function name (either TO_DATE or TRY_TO_DATE) and generates
    the wrapper function for the corresponding kernel.
    """

    def overload_func(conversion_val, format_str, dict_encoding_state=None, func_id=-1):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [conversion_val, format_str]
        for i, arg in enumerate(args):
            if isinstance(arg, types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name.lower()}_util",
                    ["conversion_val", "format_str", "dict_encoding_state", "func_id"],
                    i,
                    default_map={"dict_encoding_state": None, "func_id": -1},
                )

        func_text = "def impl(conversion_val, format_str, dict_encoding_state=None, func_id=-1):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{func_name.lower()}_util(conversion_val, format_str, dict_encoding_state, func_id)"
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


def to_timestamp(
    conversion_val, format_str, time_zone, scale, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    return


def try_to_timestamp(
    conversion_val, format_str, time_zone, scale, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    return


def to_timestamp_util(
    conversion_val, format_str, time_zone, scale, dict_encoding_state, func_id
):  # pragma: no cover
    return


def try_to_timestamp_util(
    conversion_val, format_str, time_zone, scale, dict_encoding_state, func_id
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
    (a series or scalar of multiple possible types), and two literals.
    The first is a format string for cases where the input is a string. The second
    is a time zone for the output data.

    The full specification is noted here:
    https://docs.snowflake.com/en/sql-reference/functions/to_timestamp.html
    """
    if error_on_fail:
        error_str = "raise ValueError('Invalid input while converting to date value')"
    else:
        error_str = "bodo.libs.array_kernels.setna(res, i)"

    def overload_impl(
        conversion_val, format_str, time_zone, scale, dict_encoding_state, func_id
    ):
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
        current_tz = get_tz_if_exists(conversion_val)
        time_zone = get_literal_value(time_zone)
        if is_overload_constant_str(time_zone):
            time_zone = str(time_zone)
        elif is_overload_none(time_zone):
            time_zone = None
        else:
            raise_bodo_error("time_zone argument must be a scalar string or None")

        if current_tz is None:
            if time_zone is not None:
                # NTZ -> LTZ
                localize_str = f".tz_localize('{time_zone}')"
            else:
                # NTZ -> NTZ
                localize_str = ""
        else:
            if time_zone is not None:
                if time_zone == current_tz:
                    # LTZ -> Same TZ
                    localize_str = ""
                else:
                    # LTZ -> Different TZ
                    localize_str = f".tz_localize(None).tz_localize('{time_zone}')"
            else:
                # LTZ -> NTZ
                localize_str = f".tz_localize(None)"
                time_zone = None

        is_out_arr = bodo.utils.utils.is_array_typ(
            conversion_val, True
        ) or bodo.utils.utils.is_array_typ(format_str, True)

        # When returning a scalar we return a pd.Timestamp type.
        unbox_str = "unbox_if_tz_naive_timestamp" if is_out_arr else ""

        # If the format string is specified, then arg0 must be string
        if not is_overload_none(format_str):
            verify_string_arg(conversion_val, func, "conversion_val")
            scalar_text = (
                "py_format_str = convert_snowflake_date_format_str_to_py_format(arg1)\n"
            )
            scalar_text += "was_successful, tmp_val = pd_to_datetime_error_checked(arg0, format=py_format_str)\n"
            scalar_text += "if not was_successful:\n"
            scalar_text += f"  {error_str}\n"
            scalar_text += "else:\n"
            scalar_text += f"  res[i] = {unbox_str}(tmp_val{localize_str})\n"

        # NOTE: gen_vectorized will automatically map this function over the values dictionary
        # of a dict encoded string array instead of decoding it whenever possible
        elif is_valid_string_arg(conversion_val):
            """
            If no format string is specified, attempt to parse the string according to these date formats:
            https://docs.snowflake.com/en/user-guide/date-time-input-output.html#date-formats. All of the examples listed are
            handled by pd.to_datetime() in Bodo jit code.

            It will also check if the string is convertable to int, IE '12345' or '-4321'
            """

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

        elif is_valid_int_arg(conversion_val):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0 * (10 ** (9 - arg3))){localize_str})\n"

        elif is_valid_float_arg(conversion_val):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0 * (10 ** (9 - arg3))){localize_str})\n"

        elif is_valid_tz_aware_datetime_arg(conversion_val):
            scalar_text = f"res[i] = {unbox_str}(arg0{localize_str})\n"

        elif is_valid_datetime_or_date_arg(conversion_val):
            scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0){localize_str})\n"
        elif is_valid_timestamptz_arg(conversion_val):
            # TimestampTZ is slightly different from the other types - if we are
            # casting to a different timezone, we need to convert the timestamp to
            # UTC first, then convert it to the target timezone. Otherwise, we
            # can extract the local timestamp and return it.
            if time_zone is not None:
                scalar_text = f"res[i] = {unbox_str}(arg0.utc_timestamp.tz_localize('UTC').tz_convert('{time_zone}'))\n"
            else:
                scalar_text = f"res[i] = {unbox_str}(arg0.local_timestamp())\n"
        elif conversion_val == bodo.null_array_type:
            # Note: We could just pass a null array, but this adds typing information
            # + validates the other arguments.
            scalar_text = f"res[i] = None\n"
        else:  # pragma: no cover
            raise raise_bodo_error(
                f"Internal error: unsupported type passed to to_timestamp_util for argument conversion_val: {conversion_val}"
            )

        arg_names = [
            "conversion_val",
            "format_str",
            "time_zone",
            "scale",
            "dict_encoding_state",
            "func_id",
        ]
        arg_types = [
            conversion_val,
            format_str,
            time_zone,
            scale,
            dict_encoding_state,
            func_id,
        ]
        propagate_null = [True, False, False, False, False, False]

        # Determine the output dtype. If a timezone is provided then we have a
        # tz-aware output. Otherwise we output datetime64ns.
        if time_zone is not None:
            out_dtype = bodo.DatetimeArrayType(time_zone)
        else:
            out_dtype = types.Array(bodo.datetime64ns, 1, "C")

        extra_globals = {
            "pd_to_datetime_error_checked": pd_to_datetime_error_checked,
            "number_to_datetime": number_to_datetime,
            "convert_snowflake_date_format_str_to_py_format": convert_snowflake_date_format_str_to_py_format,
            "unbox_if_tz_naive_timestamp": bodo.utils.conversion.unbox_if_tz_naive_timestamp,
        }
        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            prefix_code=prefix_code,
            extra_globals=extra_globals,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_impl


def create_timestamp_cast_func(func_name):
    def overload_func(
        conversion_val,
        format_str,
        time_zone,
        scale,
        dict_encoding_state=None,
        func_id=-1,
    ):
        """Handles cases where func_name receives an optional argument and forwards
        to the appropriate version of the real implementation"""
        args = [conversion_val, format_str, time_zone, scale]
        for i, arg in enumerate(args):
            if isinstance(arg, types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name.lower()}_util",
                    [
                        "conversion_val",
                        "format_str",
                        "time_zone",
                        "scale",
                        "dict_encoding_state",
                        "func_id",
                    ],
                    i,
                    default_map={"dict_encoding_state": None, "func_id": -1},
                )

        func_text = "def impl(conversion_val, format_str, time_zone, scale, dict_encoding_state=None, func_id=-1):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{func_name.lower()}_util(conversion_val, format_str, time_zone, scale, dict_encoding_state, func_id)"
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


def to_timestamptz(
    conversion_val, time_zone, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    return


def to_timestamptz_util(
    conversion_val, time_zone, dict_encoding_state, func_id
):  # pragma: no cover
    return


@overload(to_timestamptz, no_unliteral=True)
def overload_to_timestamptz(
    conversion_val, time_zone, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    """Handles cases where TO_TIMESTAMP_TZ receives optional arguments and
    forwards to the appropriate version of the real implementation"""
    args = [conversion_val, time_zone]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.to_timestamptz",
                ["conversion_val", "time_zone", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(conversion_val, time_zone, dict_encoding_state=None, func_id=-1):
        return to_timestamptz_util(
            conversion_val, time_zone, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def to_timestamptz_string_parser(str_val):
    """Parses timestamps with the timezone fully specified. Returns None if the
    timezone couldn't be parsed, and a TimestampTZ otherwise."""

    def impl(str_val: str) -> Optional[TimestampTZ]:
        if not len(str_val):
            return None

        sign = 1
        str_val = str_val.strip()
        if str_val.endswith("Z"):
            # Z for 0 offset
            timestamp_str = str_val[:-1].strip()
            offset = 0
        else:
            # extract the timestamp string and offset by finding the offset
            # suffix
            if "+" in str_val:
                timestamp_str, offset_str = str_val.rsplit("+", 1)
            else:
                # Filter out all empty strings
                parts = [p for p in str_val.split(" ") if p]
                # If we can't clearly split the string into [date, time,
                # offset], maybe the time and offset don't have a space between
                # them - this means the delimiter must be '-', since '+' is
                # handled above.
                if len(parts) != 3:
                    if "-" in str_val:
                        timestamp_str, offset_str = str_val.rsplit("-", 1)
                        sign = -1
                    else:
                        return None
                else:
                    timestamp_str = parts[0] + " " + parts[1]
                    offset_str = parts[2]
                    # If there was a sign, record it and remove it from offset_str
                    if offset_str[0] == "-":
                        sign = -1
                        offset_str = offset_str[1:]
                    elif offset_str[0] == "+":
                        sign = 1
                        offset_str = offset_str[1:]

            if ":" in offset_str:
                # parse offset as one of HH:MM, H:MM, HH:M
                parts = offset_str.split(":")
                if not len(parts) == 2:
                    return None
                hours, minutes = parts
                if not hours.isdigit() or not minutes.isdigit():
                    return None

                hours = int(hours)
                if hours < 0 or hours > 23:
                    return None
                minutes = int(minutes)
                if minutes < 0 or minutes > 59:
                    return None

                offset = int(hours) * 60 + int(minutes)
            else:
                # parse offset as one of HH, HMM, HHMM
                if not offset_str.isdigit():
                    return None

                if len(offset_str) == 2:
                    # HH
                    if not offset_str.isdigit():
                        return None
                    hours = int(offset_str)
                    if hours < 0 or hours > 23:
                        return None
                    offset = int(offset_str) * 60
                elif len(offset_str) == 3:
                    # HMM
                    offset_raw = int(offset_str)
                    hours = offset_raw // 100
                    minutes = offset_raw % 100
                    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
                        return None
                    offset = hours * 60 + minutes
                elif len(offset_str) == 4:
                    # HHMM
                    # This is only actually allowed in Snowflake if the string is in a
                    # specific format, but we'll allow it here for simplicity
                    offset_raw = int(offset_str)
                    hours = offset_raw // 100
                    minutes = offset_raw % 100
                    if hours < 0 or hours > 23 or minutes < 0 or minutes > 59:
                        return None
                    offset = hours * 60 + minutes
                else:
                    return None

            # apply the sign
            offset *= sign

        local_timestamp = bodo.libs.bodosql_array_kernels.to_timestamp(
            timestamp_str, None, None, 0
        )
        if local_timestamp is None:
            return None
        return bodo.hiframes.timestamptz_ext.init_timestamptz_from_local(
            local_timestamp, offset
        )

    return impl


@overload(to_timestamptz_util, no_unliteral=True)
def overload_to_timestamptz_util(
    conversion_val, time_zone, dict_encoding_state, func_id
):  # pragma: no cover
    """A dedicated kernel for the SQL function TO_TIMESTAMP_TZ which converts
    values to a timestamp with a timezone.
    """
    if is_overload_none(conversion_val) or is_valid_timestamptz_arg(conversion_val):
        return lambda conversion_val, dict_encoding_state, func_id: conversion_val

    time_zone = get_literal_value(time_zone)
    if is_overload_constant_str(time_zone):
        time_zone = str(time_zone)
    elif is_overload_none(time_zone):
        time_zone = None
    else:
        raise_bodo_error("time_zone argument must be a scalar string or None")

    use_dict_caching = not is_overload_none(dict_encoding_state)
    prefix_code = ""
    if is_valid_string_arg(conversion_val):
        scalar_text = "val = bodo.libs.bodosql_snowflake_conversion_array_kernels.to_timestamptz_string_parser(arg0)\n"
        # The timestamp might be None if the string didn't have a timezone -
        # attempt to parse it as a normal timestamp
        scalar_text += "if val is not None:\n"
        scalar_text += "  res[i] = val\n"
        scalar_text += "else:\n"
        scalar_text += "  offset = 0\n"
        scalar_text += "  local_timestamp = bodo.libs.bodosql_array_kernels.to_timestamp(arg0, None, None, 0)\n"
        if time_zone is not None:
            # If the string isn't a numeric value, then we need to use the default timezone as the offset
            scalar_text += "  if not arg0.isdigit():\n"
            scalar_text += f"    offset = int(local_timestamp.tz_localize('{time_zone}').utcoffset().total_seconds() / 60)\n"
            # If there's a time in the string, then We need to use the default timezone as the offset
        scalar_text += "  if local_timestamp is None:\n"
        scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "  else:\n"
        scalar_text += "    res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz_from_local(local_timestamp, offset)\n"
    elif is_valid_tz_aware_datetime_arg(conversion_val):
        scalar_text = "offset = int(arg0.utcoffset().total_seconds() / 60)\n"
        scalar_text += "utc_timestamp = arg0.tz_convert('UTC').tz_localize(None)\n"
        scalar_text += "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz(utc_timestamp, offset)\n"
    elif is_valid_tz_naive_datetime_arg(conversion_val):
        scalar_text = "local_timestamp = pd.Timestamp(arg0)\n"
        if time_zone is not None:
            scalar_text += f"offset = int(local_timestamp.tz_localize('{time_zone}').utcoffset().total_seconds() / 60)\n"
            scalar_text += (
                "utc_timestamp = local_timestamp - pd.Timedelta(minutes=offset)\n"
            )
        else:
            scalar_text += "offset = 0\n"
            scalar_text += "utc_timestamp = local_timestamp\n"
        scalar_text += "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz(utc_timestamp, offset)\n"
    elif is_valid_date_arg(conversion_val) or is_valid_numeric_bool(conversion_val):
        # Call to_timestamp to get the local timestamp, then convert it to a TimestampTZ
        prefix_code = "local_timestamp = bodo.libs.bodosql_array_kernels.to_timestamp(conversion_val, None, None, 0, dict_encoding_state=dict_encoding_state, func_id=func_id)\n"
        if is_array_typ(conversion_val):
            scalar_text = "arg0_local = pd.Timestamp(local_timestamp[i])\n"
        else:
            scalar_text = "arg0_local = pd.Timestamp(local_timestamp)\n"
        if time_zone is not None:
            scalar_text += f"offset = int(arg0_local.tz_localize('{time_zone}').utcoffset().total_seconds() / 60)\n"
            scalar_text += "utc_timestamp = arg0_local - pd.Timedelta(minutes=offset)\n"

        else:
            scalar_text += "offset = 0\n"
            scalar_text += "utc_timestamp = arg0_local\n"
        scalar_text += "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz(utc_timestamp, offset)\n"
    else:  # pragma: no cover
        raise_bodo_error(
            f"Internal error: unsupported type passed to to_timestamptz_util for argument conversion_val: {conversion_val}"
        )
    return gen_vectorized(
        ["conversion_val", "time_zone", "dict_encoding_state", "func_id"],
        [conversion_val, time_zone, dict_encoding_state, func_id],
        [True, False, False, False],
        scalar_text,
        bodo.timestamptz_array_type,
        prefix_code=prefix_code,
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def to_binary(arr, dict_encoding_state=None, func_id=-1):
    """Handles cases where TO_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_binary_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return to_binary_util(arr, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def try_to_binary(arr, dict_encoding_state=None, func_id=-1):
    """Handles cases where TRY_TO_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.try_to_binary_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return try_to_binary_util(arr, dict_encoding_state, func_id)

    return impl


def to_char(arr, format_str=None, is_scalar=False):  # pragma: no cover
    pass


@overload(to_char, no_unliteral=True)
def overload_to_char(arr, format_str=None, is_scalar=False):  # pragma: no cover
    """Handles cases where TO_CHAR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, format_str]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.to_char",
                ["arr", "format_str", "is_scalar"],
                i,
                default_map={"format_str": None, "is_scalar": False},
            )

    def impl(arr, format_str=None, is_scalar=False):
        return to_char_util(arr, format_str, is_scalar)

    return impl


@numba.generated_jit(nopython=True)
def to_char_helper(input, format_str):  # pragma: no cover
    """A simple helper function used to handle fields of inner type"""
    return (
        (lambda input, format_str: f'"{input}"')
        if is_valid_string_arg(input)
        else (lambda input, format_str: to_char(input, format_str, True))
    )


def to_char_util(arr, format_str, is_scalar):  # pragma: no cover
    pass


@overload(to_char_util, no_unliteral=True)
def overload_to_char_util(arr, format_str, is_scalar):  # pragma: no cover
    """A dedicated kernel for the SQL function TO_CHAR which takes in a
    number (or column) and returns a string representation of it.

    Args:
        arr (numerical array/series/scalar): the number(s) being operated on
        opt_fmt_str (string array/series/scalar): the format string(s) to use

        This argument is used to properly handle
        converting both Timestamp and Date values while we are in the process of
        transitioning to a dedicated date type.

    Returns:
        string series/scalar: the string representation of the number(s) with the
        specified null handling rules
    """
    if is_overload_none(arr):
        return lambda arr, format_str, is_scalar: None
    arg_names = ["arr", "format_str", "is_scalar"]
    arg_types = [arr, format_str, is_scalar]
    propagate_null = [True, False, False]
    are_arrays = [
        is_array_typ(arr)
        if is_overload_none(is_scalar)
        else not get_overload_const_bool(is_scalar, "to_char", "is_scalar"),
        is_array_typ(format_str),
        False,
    ]
    inner_type = arr.dtype if are_arrays[0] else arr
    out_dtype = bodo.string_array_type
    convert_func_str = "bodo.libs.bodosql_snowflake_conversion_array_kernels.convert_snowflake_date_format_str_to_py_format"
    # Check if we can use one of our array kernels to handle the conversion.
    if is_array_typ(arr) and isinstance(arr, bodo.DecimalArrayType):

        def impl(arr, format_str, is_scalar):  # pragma: no cover
            return bodo.libs.bodosql_array_kernels.decimal_array_to_str_array(arr)

        return impl

    elif is_valid_decimal_arg(inner_type):
        scalar_text = (
            "res[i] = bodo.libs.bodosql_array_kernels.decimal_scalar_to_str(arg0)"
        )
    elif is_array_typ(inner_type):
        scalar_text = "arr_str = ''\n"
        scalar_text += "for idx0 in range(len(arg0)):\n"
        scalar_text += "  arr_str += ',' + ('undefined' if bodo.libs.array_kernels.isna(arg0, idx0) else bodo.libs.bodosql_array_kernels.to_char_helper(arg0[idx0], arg1))\n"
        scalar_text += "res[i] = '[' + arr_str[1:] + ']'"
    elif is_valid_string_arg(inner_type):
        # Strings are unchanged.
        return lambda arr, format_str, is_scalar: arr
    # TODO [BE-3744]: support binary data for to_char
    elif is_valid_binary_arg(inner_type):
        # TODO(Yipeng): Support all binary encoding. Currently only hex encoding is supported.
        # Using bodo.libs.bodosql_array_kernels.hex_encode(arg0, 0) here will break test_cast_char_other[Bytes]
        scalar_text = "with bodo.objmode(r=bodo.string_type):\n"
        scalar_text += "  r = arg0.hex()\n"
        scalar_text += "res[i] = r"
    elif is_valid_time_arg(inner_type):
        scalar_text = (
            "if bodo.libs.array_kernels.isna(format_str, i):\n"
            if are_arrays[1]
            else f"if {is_overload_none(format_str)}:\n"
        )
        # Use the default format of HH:MM:SS (1 digits are always extended to 2, and sub-second units are ignored)
        scalar_text += "  res[i] = format(arg0.hour, '0>2') + ':' + format(arg0.minute, '0>2') + ':' + format(arg0.second, '0>2')\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = arg0.strftime({convert_func_str}(arg1))"
    elif is_valid_timedelta_arg(inner_type):
        scalar_text = "arg0 = bodo.utils.conversion.unbox_if_tz_naive_timestamp(arg0)\n"
        scalar_text += "with bodo.objmode(r=bodo.string_type):\n"
        scalar_text += "  r = str(arg0)\n"
        scalar_text += "res[i] = r"
    elif is_valid_datetime_or_date_arg(inner_type):
        scalar_text = "arg0 = bodo.utils.conversion.box_if_dt64(arg0)\n"
        scalar_text += (
            "if bodo.libs.array_kernels.isna(format_str, i):\n"
            if are_arrays[1]
            else f"if {is_overload_none(format_str)}:\n"
        )
        if is_valid_date_arg(inner_type):
            scalar_text += "  res[i] = str(arg0)\n"
        elif is_valid_tz_aware_datetime_arg(inner_type):
            scalar_text += "  res[i] = arg0.isoformat(' ')\n"
        else:
            scalar_text += "  res[i] = pd.Timestamp(arg0).isoformat(' ')\n"
        scalar_text += "else:\n"
        scalar_text += f"  res[i] = arg0.strftime({convert_func_str}(arg1))"
    elif is_valid_timestamptz_arg(inner_type):
        # If the current row has a format string, use it on the local timestamp.
        # This won't work for every type of format string since the offset is
        # not passed along.
        scalar_text = (
            "if not bodo.libs.array_kernels.isna(format_str, i):\n"
            if are_arrays[1]
            else f"if not {is_overload_none(format_str)}:\n"
        )
        scalar_text += (
            "  local_ts = bodo.hiframes.timestamptz_ext.get_local_timestamp(arg0)\n"
        )
        scalar_text += (
            "  return bodo.libs.bodosql_array_kernels.to_char(local_ts, arg1, True)\n"
        )
        scalar_text += "else:\n"
        scalar_text += "  offset = abs(arg0.offset_minutes)\n"
        scalar_text += "  offset_sign = '+' if arg0.offset_minutes >= 0 else '-'\n"
        scalar_text += "  hours = offset // 60\n"
        scalar_text += "  minutes = offset % 60\n"
        scalar_text += "  offset_str = f'{offset_sign}{hours:02}{minutes:02}'\n"
        scalar_text += (
            "  res[i] = arg0.local_timestamp().isoformat(' ') + ' ' + offset_str\n"
        )
    elif is_valid_float_arg(inner_type):
        scalar_text = "if np.isinf(arg0):\n"
        scalar_text += "  res[i] = 'inf' if arg0 > 0 else '-inf'\n"
        # currently won't use elif branch since np.nan is caught by
        # propagate_null[0] being True, presently
        # TODO [BE-3491]: treat NaNs and nulls differently
        scalar_text += "elif np.isnan(arg0):\n"
        scalar_text += "  res[i] = 'NaN'\n"
        scalar_text += "else:\n"
        scalar_text += "  res[i] = str(arg0)"
    elif is_valid_boolean_arg(inner_type):
        scalar_text = "res[i] = 'true' if arg0 else 'false'"
    elif is_overload_none(inner_type):
        scalar_text = "res[i] = None"
    elif is_valid_int_arg(inner_type):
        int_types = {
            8: np.int8,
            16: np.int16,
            32: np.int32,
            64: np.int64,
        }
        scalar_text = f"if arg0 == {np.iinfo(int_types[inner_type.bitwidth]).min}:\n"
        scalar_text += f"  res[i] = '{np.iinfo(int_types[inner_type.bitwidth]).min}'\n"
        scalar_text += "else:\n"
        scalar_text += "  res[i] = str(arg0)"
    elif isinstance(inner_type, bodo.StructType):
        scalar_text = "arr_str = ''\n"
        for i in range(len(inner_type.names)):
            name_w_quotes = '"' + inner_type.names[i] + '"'
            scalar_text += f"if not bodo.libs.struct_arr_ext.is_field_value_null(arg0, {name_w_quotes}):\n"
            scalar_text += f"  arr_str += ',{name_w_quotes}:' + bodo.libs.bodosql_array_kernels.to_char_helper(arg0[{name_w_quotes}], arg1)\n"
        scalar_text += "res[i] = '{' + arr_str[1:] + '}'"
    elif isinstance(inner_type, types.DictType):
        scalar_text = "arr_str = ''\n"
        scalar_text += "for idx0 in range(len(arg0._keys)):\n"
        scalar_text += "  if not bodo.libs.array_kernels.isna(arg0._keys, idx0) and not bodo.libs.array_kernels.isna(arg0._values, idx0):\n"
        scalar_text += "    arr_str += ',' + bodo.libs.bodosql_array_kernels.to_char_helper(arg0._keys[idx0], arg1) + ':' + bodo.libs.bodosql_array_kernels.to_char_helper(arg0._values[idx0], arg1)\n"
        scalar_text += "res[i] = '{' + arr_str[1:] + '}'"
    elif isinstance(inner_type, bodo.libs.map_arr_ext.MapScalarType):
        scalar_text = "arr_str = ''\n"
        scalar_text += "for idx0 in range(len(arg0._keys)):\n"
        scalar_text += "  if not bodo.libs.array_kernels.isna(arg0._keys, idx0) and not bodo.libs.array_kernels.isna(arg0._values, idx0):\n"
        scalar_text += "    arr_str += ',' + bodo.libs.bodosql_array_kernels.to_char_helper(arg0._keys[idx0], arg1) + ':' + bodo.libs.bodosql_array_kernels.to_char_helper(arg0._values[idx0], arg1)\n"
        scalar_text += "res[i] = '{' + arr_str[1:] + '}'"
    else:
        scalar_text = "res[i] = str(arg0)"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def make_to_double(_try):
    """Generate utility functions to unopt TO_DOUBLE arguments"""

    func_name = "to_double"
    if _try:
        func_name = "try_to_double"

    @numba.generated_jit(nopython=True)
    def func(val, optional_format_string, dict_encoding_state=None, func_id=-1):
        """Handles cases where TO_DOUBLE receives optional arguments and forwards
        to the appropriate version of the real implementation"""
        args = [val, optional_format_string]
        for i, arg in enumerate(args):
            if isinstance(arg, types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name}",
                    ["val", "optional_format_string", "dict_encoding_state", "func_id"],
                    i,
                    default_map={"dict_encoding_state": None, "func_id": -1},
                )

        def impl(
            val, optional_format_string, dict_encoding_state=None, func_id=-1
        ):  # pragma: no cover
            return to_double_util(
                val,
                optional_format_string,
                numba.literally(_try),
                dict_encoding_state,
                func_id,
            )

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


@numba.generated_jit(nopython=True, no_unliteral=True)
def to_double_util(val, optional_format_string, _try, dict_encoding_state, func_id):
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
    verify_string_numeric_arg(
        val, "TO_DOUBLE and TRY_TO_DOUBLE", "val", include_decimal=True
    )
    verify_string_arg(
        optional_format_string, "TO_DOUBLE and TRY_TO_DOUBLE", "optional_format_string"
    )
    is_string = is_valid_string_arg(val)
    is_float = is_valid_float_arg(val)
    is_int = is_valid_int_arg(val)
    is_bool = is_valid_boolean_arg(val)
    is_decimal = is_valid_decimal_arg(val)
    _try = get_overload_const_bool(_try)

    if _try:  # pragma: no cover
        on_fail = "bodo.libs.array_kernels.setna(res, i)\n"
    else:
        if is_string:
            err_msg = "string must be a valid numeric expression"
        else:  # pragma: no cover
            err_msg = "value must be a valid numeric expression"
        on_fail = f'raise ValueError("invalid value for double conversion: {err_msg}")'

    # Format string not supported
    if not is_overload_none(optional_format_string):  # pragma: no cover
        raise raise_bodo_error(
            f"Internal error: Format string not supported for TO_DOUBLE / TRY_TO_DOUBLE"
        )
    elif is_decimal:
        if is_array_typ(val):
            # We can use a dedicated array kernel for decimal arrays.
            def impl(
                val, optional_format_string, _try, dict_encoding_state, func_id
            ):  # pragma: no cover
                return bodo.libs.decimal_arr_ext.decimal_arr_to_float64(val)

            return impl
        else:
            scalar_text = (
                f"res[i] = bodo.libs.decimal_arr_ext.decimal_to_float64(arg0)\n"
            )
    elif is_string:
        scalar_text = "arg0 = arg0.strip()\n"
        scalar_text += "if is_string_numeric(arg0):\n"
        scalar_text += "  res[i] = np.float64(arg0)\n"
        scalar_text += "else:\n"
        scalar_text += f"  {on_fail}\n"
    elif is_float:  # pragma: no cover
        scalar_text = f"res[i] = arg0\n"
    elif is_int or is_bool:  # pragma: no cover
        scalar_text = f"res[i] = np.float64(arg0)\n"
    else:  # pragma: no cover
        raise raise_bodo_error(
            f"Internal error: unsupported type passed to to_double_util for argument val: {val}"
        )

    arg_names = [
        "val",
        "optional_format_string",
        "_try",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [val, optional_format_string, _try, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False, False]

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

    extra_globals = {
        "is_string_numeric": is_string_numeric,
    }
    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@register_jitable
def convert_snowflake_date_format_str_to_py_format(format_str):  # pragma: no cover
    """Helper fn for the TO_DATE/TO_TIMESTAMP fns. This fn takes a format string
    in SQL syntax, and converts it to the python syntax.
    Snowflake syntax reference: https://docs.snowflake.com/en/user-guide/date-time-input-output
    Python syntax reference: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    Snowflake allows for arbitrary format strings with patterns YYYY, YY, MMMM, MM, MON, DD, DY
    """
    format_str = format_str.upper()
    format_map = {
        "YYYY": "%Y",
        "YY": "%y",
        "MMMM": "%B",
        "MM": "%m",
        "MON": "%b",
        "HH12": "%I",
        "HH24": "%H",
        "AM": "%p",
        "PM": "%p",
        "DD": "%d",
        "DY": "%a",
        "MI": "%M",
        "SS": "%S",
    }

    for elem in format_map:
        format_str = format_str.replace(elem, format_map[elem])

    return format_str


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
    origin="unix",
    cache=True,
):  # pragma: no cover
    """Helper fn that determines if we have a parsable datetime string, by calling
    pd.to_datetime in objmode, which returns a tuple (success flag, value). If
    the success flag evaluates to True, then the paired value is the correctly parsed timestamp, otherwise  the paired value is a dummy timestamp.
    """

    if format is None:
        list_tokens = val.split()
        if len(list_tokens) == 0:
            return (False, pd.Timestamp(0))
        else:
            first_token = list_tokens[0]

            dash_count = first_token.count("-")
            slash_count = first_token.count("/")

            # Acceptable cases are: two dashes or two slashes, the other must be 0
            is_date_format_1 = dash_count == 2 and slash_count == 0
            is_date_format_2 = dash_count == 0 and slash_count == 2

            if not (is_date_format_1 or is_date_format_2):
                return (False, pd.Timestamp(0))

    with bodo.objmode(ret_val="pd_timestamp_tz_naive_type", success_flag="bool_"):
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
            origin=origin,
            cache=cache,
        )
        if pd.isna(tmp):
            success_flag = False
        else:
            if tmp.tz is not None:
                ret_val = tmp.tz_localize(None)
            else:
                ret_val = tmp

    return (success_flag, ret_val)


@register_jitable
def to_date_error_checked(val, format):  # pragma: no cover
    """
    Helper function to convert a date string with format string to a datetime.date object
    """

    py_format = convert_snowflake_date_format_str_to_py_format(format)
    if py_format == "":
        return (False, None)
    with bodo.objmode(ret_val="pd_timestamp_tz_naive_type", success_flag="bool_"):
        success_flag = True
        ret_val = pd.Timestamp(0)

        tmp = pd.to_datetime(
            val,
            errors="coerce",
            format=py_format,
        )
        if pd.isna(tmp):
            success_flag = False
        else:
            ret_val = tmp

    return (success_flag, ret_val)


@register_jitable
def to_date_strings_conversion(year_str, month_str, day_str):  # pragma: no cover
    """
    Convert a year string, month string, and day string to an output date type.
    Return a tuple of (success_flag, output_date), where if success_flag is False,
    then the output date is garbage and the date is invalid to avoid trigger an
    exception.

    Args:
        val (types.unicode_type): An input string to convert to a date.

    Returns:
        types.Tuple(types.bool_, bodo.datetime_date_type): A tuple of where the conversion
        was a success and the resulting date. If the success is false the date is garbage.
    """
    # Initialize date_val for an invalid value.
    date_val = datetime.date(1970, 1, 1)
    if (
        len(day_str) > 2
        or len(month_str) > 2
        or len(year_str) > 4
        or not (year_str.isdigit() and month_str.isdigit() and day_str.isdigit())
    ):
        return (False, date_val)
    # convert the values to numbers
    year = int(year_str)
    month = int(month_str)
    day = int(day_str)
    # Validate now to avoid exceptions
    # Other than negative years, we don't need to check the year
    # because its not strictly defined.
    if year < 0:
        return (False, date_val)
    # Check the month
    if month < 1 or month > 12:
        return (False, date_val)
    max_day = bodo.hiframes.pd_timestamp_ext.get_days_in_month(year, month)
    # Day must be within the range of the month
    if day < 1 or day > max_day:
        return (False, date_val)
    # We have a valid date, so return it.
    return (True, datetime.date(year, month, day))


@register_jitable
def to_date_auto_error_checked(val):  # pragma: no cover
    """
    Converts a string to a date type based on the auto parsing format
    for Snowflake.

    https://docs.snowflake.com/en/user-guide/date-time-input-output#date-formats

    Snowflake allows multiple different inputs to satisfy format values.
    https://docs.snowflake.com/en/user-guide/date-time-input-output#using-the-correct-number-of-digits-with-format-elements

    Snowflake seems to allow checking additional Timestamp formats and then truncating those to
    date. As a result we use this as a fast path and fallback to the Timestamp conversion if
    we can't parse.

    Return a tuple of (success_flag, output_date), where if success_flag is False,
    then the output date is garbage and the date is invalid to avoid trigger an
    exception.

    Args:
        val (types.unicode_type): An input string to convert to a date.

    Returns:
        types.Tuple(types.bool_, bodo.datetime_date_type): A tuple of where the conversion
        was a success and the resulting date. If the success is false the date is garbage.
    """
    is_valid = False
    # Initialize date_val for returning the final value.
    date_val = datetime.date(1970, 1, 1)
    # Map the month name to the month number. We keep this as a
    # string for helper functions.
    month_map = {
        "jan": "1",
        "feb": "2",
        "mar": "3",
        "apr": "4",
        "may": "5",
        "jun": "6",
        "jul": "7",
        "aug": "8",
        "sep": "9",
        "oct": "10",
        "nov": "11",
        "dec": "12",
    }
    if "-" in val:
        parts = val.split("-")
        if len(parts) != 3:
            return (is_valid, date_val)
        month_part = parts[1]
        if month_part.isdigit():
            # This must be year, month, day path.
            year_part = parts[0]
            day_part = parts[2]
            is_valid, date_val = to_date_strings_conversion(
                year_part, month_part, day_part
            )
        else:
            year_part = parts[2]
            day_part = parts[0]
            month_part = month_part.lower()
            if month_part in month_map:
                # This must be the day, month name, year path.
                month_digit = month_map[month_part]
                is_valid, date_val = to_date_strings_conversion(
                    year_part, month_digit, day_part
                )
            else:
                # This is invalid
                return (is_valid, date_val)

    else:
        parts = val.split("/")
        if len(parts) != 3:
            return (is_valid, date_val)
        # This is day, month, year
        month_part = parts[0]
        day_part = parts[1]
        year_part = parts[2]
        is_valid, date_val = to_date_strings_conversion(year_part, month_part, day_part)

    return (is_valid, date_val)


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


def cast_date_to_tz_aware(arr, tz):  # pragma: no cover
    pass


@overload(cast_date_to_tz_aware, no_unliteral=True)
def overload_cast_date_to_tz_aware(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_date_to_tz_aware(): 'tz' must be a literal value")
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_date_to_tz_aware",
            ["arr", "tz"],
            0,
        )

    def impl(arr, tz):  # pragma: no cover
        return cast_date_to_tz_aware_util(arr, tz)

    return impl


def cast_date_to_tz_aware_util(arr, tz):  # pragma: no cover
    pass


@overload(cast_date_to_tz_aware_util, no_unliteral=True)
def overload_cast_date_to_tz_aware_util(arr, tz):
    if not is_literal_type(tz):
        raise_bodo_error("cast_date_to_tz_aware(): 'tz' must be a literal value")
    verify_datetime_arg(arr, "cast_date_to_tz_aware", "arr")
    arg_names = ["arr", "tz"]
    arg_types = [arr, tz]
    # tz can never be null
    propagate_null = [True, False]
    scalar_text = "res[i] = pd.Timestamp(arg0).tz_localize(arg1)"
    tz = get_literal_value(tz)
    out_dtype = bodo.DatetimeArrayType(tz)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def cast_tz_aware_to_tz_naive(arr):  # pragma: no cover
    pass


@overload(cast_tz_aware_to_tz_naive, no_unliteral=True)
def overload_cast_tz_aware_to_tz_naive(arr):
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_tz_aware_to_tz_naive",
            ["arr"],
            0,
        )

    def impl(arr):  # pragma: no cover
        return cast_tz_aware_to_tz_naive_util(arr)

    return impl


def cast_tz_aware_to_tz_naive_util(arr):  # pragma: no cover
    pass


@overload(cast_tz_aware_to_tz_naive_util, no_unliteral=True)
def overload_cast_tz_aware_to_tz_naive_util(arr):
    verify_datetime_arg_require_tz(arr, "cast_tz_aware_to_tz_naive", "arr")
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    # If we have an array we must cast the output to a datetime64
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(arr)
        else ""
    )
    scalar_text = "ts = arg0.tz_localize(None)\n"
    scalar_text += f"res[i] = {unbox_str}(ts)"
    out_dtype = types.Array(bodo.datetime64ns, 1, "C")
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def cast_str_to_tz_aware(
    arr, tz, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(cast_str_to_tz_aware, no_unliteral=True)
def overload_cast_str_to_tz_aware(arr, tz, dict_encoding_state=None, func_id=-1):
    if not is_literal_type(tz):
        raise_bodo_error("cast_str_to_tz_aware(): 'tz' must be a literal value")
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.cast_str_to_tz_aware",
            ["arr", "tz", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, tz, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return cast_str_to_tz_aware_util(arr, tz, dict_encoding_state, func_id)

    return impl


def cast_str_to_tz_aware_util(
    arr, tz, dict_encoding_state, func_id
):  # pragma: no cover
    pass


@overload(cast_str_to_tz_aware_util, no_unliteral=True)
def overload_cast_str_to_tz_aware_util(arr, tz, dict_encoding_state, func_id):
    if not is_literal_type(tz):
        raise_bodo_error("cast_str_to_tz_aware(): 'tz' must be a literal value")
    verify_string_arg(arr, "cast_str_to_tz_aware", "arr")
    arg_names = ["arr", "tz", "dict_encoding_state", "func_id"]
    arg_types = [arr, tz, dict_encoding_state, func_id]
    # tz can never be null
    propagate_null = [True, False, False, False]
    # Note: pd.to_datetime doesn't support tz as an argument.
    scalar_text = f"res[i] = pd.to_datetime(arg0).tz_localize(arg1)"
    tz = get_literal_value(tz)
    out_dtype = bodo.DatetimeArrayType(tz)
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


def to_binary_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


def try_to_binary_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


# TODO ([BE-4344]): implement and test to_binary with other formats
def create_to_binary_util_overload(fn_name, error_on_fail):
    def impl(arr, dict_encoding_state, func_id):  # pragma: no cover
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
        arg_names = ["arr", "dict_encoding_state", "func_id"]
        arg_types = [arr, dict_encoding_state, func_id]
        propagate_null = [True, False, False]
        out_dtype = bodo.binary_array_type
        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
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


def make_to_number(_try):
    """Generate utility functions to unopt TO_NUMBER (and its variants) arguments"""
    func_name = "to_number"
    if _try:
        func_name = "try_to_number"

    def to_number_helper(
        expr, prec, scale, outputs_decimal, dict_encoding_state=None, func_id=-1
    ):
        return

    @overload(to_number_helper, no_unliteral=True)
    def to_number_helper_overload(
        expr, prec, scale, outputs_decimal, dict_encoding_state=None, func_id=-1
    ):
        """Handles cases where TO_NUMBER receives optional arguments and forwards
        to the appropriate version of the real implementation"""

        args = [expr, prec, scale]
        for i in range(len(args)):
            if isinstance(args[i], types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.{func_name}",
                    [
                        "expr",
                        "prec",
                        "scale",
                        "outputs_decimal",
                        "dict_encoding_state",
                        "func_id",
                    ],
                    i,
                    default_map={
                        "dict_encoding_state": None,
                        "func_id": -1,
                    },
                )

        def impl(
            expr, prec, scale, outputs_decimal, dict_encoding_state=None, func_id=-1
        ):  # pragma: no cover
            return to_number_util(
                expr,
                prec,
                scale,
                _try,
                outputs_decimal,
                dict_encoding_state,
                func_id,
            )

        return impl

    return to_number_helper


try_to_number = make_to_number(True)
to_number = make_to_number(False)


def string_to_decimal(expr, precision, scale, null_on_error):  # pragma: no cover
    pass


@overload(string_to_decimal, no_unliteral=True)
def string_to_decimal_overload(expr, precision, scale, null_on_error):
    """
    Cast a string expression array or scalar to a decimal type with the given precision and scale.
    If null_on_error=True then values that either don't fit in the specified decimal or don't parse
    properly are replaced with nulls. The exception is that scale values should be rounded half up.
    When null_on_error=False, the function will throw an error if the string can't be converted to a decimal.

    Note: This function shouldn't be called directly from BodoSQL because it doesn't
    have optional handling so it should be called by to_number.

    Args:
        expr (numeric array or scalar): the input string array or scalar
        precision (positive integer literal): the precision of the decimal type
        scale (positive integer literal): the scale of the decimal type
        null_on_error (bool literal): if True, return None for invalid values, otherwise throw an error

    Returns:
        decimal array or scalar: the converted decimal array or scalar
    """
    if not is_overload_constant_int(precision):
        raise_bodo_error("string_to_decimal: prec must be a literal value if provided")
    if not is_overload_constant_int(scale):
        raise_bodo_error("string_to_decimal: scale must be a literal value if provided")
    if not is_overload_constant_bool(null_on_error):
        raise_bodo_error("string_to_decimal: null_on_error must be a literal value")

    # Note: We don't use gen_vectorized here because there are optimized
    # array kernels in C++.
    if is_overload_none(expr):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return None

        return impl

    elif expr == bodo.string_type or is_overload_constant_str(expr):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.str_to_decimal_scalar(
                expr, precision, scale, null_on_error
            )

        return impl
    elif expr == bodo.string_array_type:

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.str_to_decimal_array(
                expr, precision, scale, null_on_error
            )

        return impl

    else:
        assert (
            expr == bodo.dict_str_arr_type
        ), "string_to_decimal_overload: dictionary-encoded string array type expected"

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            # Just cast the data array. Note: Since a value may no longer exist
            # in the dictionary we cannot error on invalid values.
            old_data = expr._data
            indices = expr._indices
            decimal_data = bodo.libs.decimal_arr_ext.str_to_decimal_array(
                old_data, precision, scale, True
            )
            out_array = bodo.libs.decimal_arr_ext.alloc_decimal_array(
                len(expr), precision, scale
            )
            for i in range(len(out_array)):
                if bodo.libs.array_kernels.isna(expr, i):
                    bodo.libs.array_kernels.setna(out_array, i)
                    continue
                index = indices[i]
                if bodo.libs.array_kernels.isna(decimal_data, index):
                    # Note: We assume we don't need to check old data for nulls because the index
                    # should always be updated.
                    # TODO: Verify this is enforced.
                    if not null_on_error:
                        raise BodoError(
                            f"String value is out of range for decimal or doesn't parse properly"
                        )
                    bodo.libs.array_kernels.setna(out_array, i)
                    continue
                out_array[i] = decimal_data[index]
            return out_array

        return impl


def numeric_to_decimal(expr, precision, scale, null_on_error):  # pragma: no cover
    pass


@overload(numeric_to_decimal, no_unliteral=True)
def numeric_to_decimal_overload(expr, precision, scale, null_on_error):
    """
    Cast a numeric expression (either integer or decimal) array or scalar
    to a decimal type with the given precision and scale. If the leading digits
    don't fit in the precision, the function will throw an error if null_on_error is False,
    otherwise it will return None for any invalid values.

    Note: This function shouldn't be called directly from BodoSQL because it doesn't
    have optional handling so it should be called by to_number.

    Args:
        expr (numeric array or scalar): the input numeric array or scalar
        precision (positive integer literal): the precision of the decimal type
        scale (positive integer literal): the scale of the decimal type
        null_on_error (bool literal): if True, return None for invalid values, otherwise throw an error

    Returns:
        decimal array or scalar: the converted decimal array or scalar
    """
    if not is_overload_constant_int(precision):
        raise_bodo_error("numeric_to_decimal: prec must be a literal value if provided")
    if not is_overload_constant_int(scale):
        raise_bodo_error(
            "numeric_to_decimal: scale must be a literal value if provided"
        )
    if not is_overload_constant_bool(null_on_error):
        raise_bodo_error("numeric_to_decimal: null_on_error must be a literal value")

    # Note: We don't use gen_vectorized here because there are optimized
    # array kernels in C++.
    if is_overload_none(expr):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return None

        return impl
    elif isinstance(expr, bodo.DecimalArrayType):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.cast_decimal_to_decimal_array(
                expr, precision, scale, null_on_error
            )

        return impl
    elif isinstance(expr, bodo.Decimal128Type):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.cast_decimal_to_decimal_scalar(
                expr, precision, scale, null_on_error
            )

        return impl
    elif (
        is_array_typ(expr, False) and isinstance(expr.dtype, types.Integer)
    ) or isinstance(expr, types.Integer):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            result = bodo.libs.decimal_arr_ext.int_to_decimal(expr)
            return bodo.libs.bodosql_array_kernels.numeric_to_decimal(
                result, precision, scale, null_on_error
            )

        return impl

    elif isinstance(expr, types.Float):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.float_to_decimal_scalar(
                expr, precision, scale, null_on_error
            )

        return impl

    elif is_array_typ(expr, False) and isinstance(expr.dtype, types.Float):

        def impl(expr, precision, scale, null_on_error):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.float_to_decimal_array(
                expr, precision, scale, null_on_error
            )

        return impl

    else:
        raise BodoError(f"numeric_to_decimal: invalid input type {expr}")


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


def to_number_util(
    expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id
):  # pragma: no cover
    pass


@overload(to_number_util, no_unliteral=True)
def to_number_util_overload(
    expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id
):  # pragma: no cover
    """Equivalent to the SQL [TRY] TO_NUMBER/TO_NUMERIC/TO_DECIMAL function.
    With the default args, this converts the input to a 64-bit integer.
    Depending on scale and precision, this could return a float, or a much smaller number

    Args:
        expr (numeric or string series/scalar): the number/string to convert to a number of type int64
        prec (positive integer literal): By SQL semantics, this is the number of allowed digits
                in the resulting number. The current behavior is to return a lower bitwidth number
                depending on the value of prec.
        scale (positive integer literal): By SQL semantics, the number of digits to the right of the
                decimal point. The current behavior is to return a float64 if scale > 0.

    Returns:
        numeric series/scalar: the converted number
    """
    arg_names = [
        "expr",
        "prec",
        "scale",
        "_try",
        "outputs_decimal",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False, False, False, False]

    verify_int_arg(prec, "TO_NUMBER", "prec")
    verify_int_arg(scale, "TO_NUMBER", "scale")

    if not is_overload_constant_int(prec):
        raise_bodo_error("TO_NUMBER: prec must be a literal value")
    if not is_overload_constant_int(scale):
        raise_bodo_error("TO_NUMBER: scale must be a literal value")
    if not is_overload_constant_bool(_try):
        raise_bodo_error("TO_NUMBER: _try must be a literal value")
    if not is_overload_constant_bool(outputs_decimal):
        raise_bodo_error("TO_NUMBER: outputs_decimal must be a literal value")

    _try = get_overload_const_bool(_try)
    prec = get_overload_const_int(prec)
    scale = get_overload_const_int(scale)
    outputs_decimal = get_overload_const_bool(outputs_decimal)

    is_null_arg = is_overload_none(expr) or expr == bodo.null_array_type
    is_string = is_valid_string_arg(expr)
    if not is_string:
        verify_numeric_arg(expr, "TO_NUMBER", "expr")

    if outputs_decimal and bodo.bodo_use_decimal:
        if is_null_arg:

            def impl(
                expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id
            ):  # pragma: no cover
                return expr

            return impl
        elif is_string:

            def impl(
                expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id
            ):
                return string_to_decimal(expr, prec, scale, _try)

            return impl
        else:

            def impl(
                expr, prec, scale, _try, outputs_decimal, dict_encoding_state, func_id
            ):
                return numeric_to_decimal(expr, prec, scale, _try)

            return impl

    # Import must occur within this function, otherwise we hit a circular import error
    from bodo.io.snowflake import precision_to_numpy_dtype

    # Calculate the output integer type based on the precision assuming scale = 0
    # This may not be used if the scale > 0, but calculating it up front makes the
    # code clearer
    output_scalar_int_type = precision_to_numpy_dtype(prec)
    # use int64 if we can't fit the number in the output type
    if output_scalar_int_type is None:
        output_scalar_int_type = types.int64

    if scale == 0:
        out_dtype = bodo.IntegerArrayType(output_scalar_int_type)
    else:
        out_dtype = bodo.FloatingArrayType(types.float64)

    # NOTE: we can't use continue/early return with gen vectorized,
    # so we use a flag to indicate if we've already seen an invalid value/should skip
    # later checks
    scalar_text = "seen_invalid = False\n"

    # First, handle casting the input if the input is string
    if is_string:
        scalar_text += "if not bodo.libs.bodosql_snowflake_conversion_array_kernels._is_string_numeric(arg0):\n"
        if _try:
            scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "  seen_invalid=True\n"
        else:
            scalar_text += (
                "  raise ValueError('unable to convert string literal to number')\n"
            )

        # Convert to float, to simplify subsequent logic
        scalar_text += "arg0 = np.float64(arg0)\n"

    # Next, check that the number of digits to the LHS of the decimal is <= prec - scale
    # if not, throw an error/return null depending on the value of _try
    allowed_digits_lhs_decimal_point = prec - scale

    if allowed_digits_lhs_decimal_point <= 0:
        raise_bodo_error(
            "TO_NUMBER: difference between prec and scale must be greater than 0"
        )

    # 2 ** 63 - 1 < 10^19, so only bother checking if we expect < 19 digits
    if allowed_digits_lhs_decimal_point < 19:
        scalar_text += "if not seen_invalid:\n"

        # Since np.int64 will always round towards zero
        # the cast to np.int64 is fine for our purposes
        scalar_text += f"  if np.abs(np.int64(arg0)) >= {10 ** allowed_digits_lhs_decimal_point}:\n"
        if _try:
            scalar_text += "    bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "    seen_invalid=True\n"
        else:
            scalar_text += "    raise ValueError('Value has too many digits to the left of the decimal')\n"

    # Finally, perform the actual conversion, and set the output
    # Snowflake behavior is that if the number of digits to the right of the decimal is greater than scale,
    # the input is truncated, rounding up
    # numpy truncates to the integer nearest to zero, so we
    # use some helper functions to round up

    bodo.libs.bodosql_array_kernels.round_half_always_up

    scalar_text += "if not seen_invalid:\n"
    if scale == 0:
        cast_str = str(output_scalar_int_type)
        if is_valid_int_arg(expr):
            # If the input is already an integer, we can just cast it
            scalar_text += f"  res[i] = np.{cast_str}(arg0)\n"
        else:
            # Otherwise, we need to round it
            scalar_text += f"  res[i] = np.{cast_str}(round_half_always_up(arg0, 0))\n"
    else:
        scalar_text += f"  res[i] = np.float64(round_half_always_up(arg0, {scale}))\n"

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
        extra_globals={
            "round_half_always_up": bodo.libs.bodosql_array_kernels.round_half_always_up
        },
    )


def convert_timezone_ntz(source_tz, target_tz, data):  # pragma: no cover
    pass


@overload(convert_timezone_ntz, no_unliteral=True)
def overload_convert_timezone_ntz(source_tz, target_tz, data):
    """Handles cases where convert_timezone (ntz version) receives optional
    arguments and forwards to the appropriate version of the real implementation"""
    args = [source_tz, target_tz, data]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.convert_timezone_ntz",
                ["source_tz", "target_tz", "data"],
                i,
            )

    def impl(source_tz, target_tz, data):  # pragma: no cover
        return convert_timezone_ntz_util(source_tz, target_tz, data)

    return impl


def convert_timezone_ntz_util(source_tz, target_tz, data):  # pragma: no cover
    pass


@overload(convert_timezone_ntz_util, no_unliteral=True)
def overload_convert_timezone_ntz_util(source_tz, target_tz, data):
    """
    Converts <data> from <source_tz> to <target_tz> timezone, as if
    translating the wallclock time from a person in one part of the world
    to another part of the world in the same epoch moment.

    Args:
        source_tz (Literal[str]): string of timezone to convert from
        target_tz (Literal[str]): string of timezone to convert to
        data (datetime/timestamp scalar/array): input data

    Returns:
        data with its timezone adjusted
    """
    if not (
        is_overload_constant_str(source_tz) and is_overload_constant_str(target_tz)
    ):
        raise_bodo_error(
            "CONVERT_TIMEZONE currently only supported with constant strings passed in for time zones."
        )
    verify_time_or_datetime_arg_allow_tz(data, "convert_timezone_ntz", "data")

    arg_names = ["source_tz", "target_tz", "data"]
    arg_types = [source_tz, target_tz, data]
    propagate_null = [True, True, True]

    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(data, True)
        else ""
    )

    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(data, True)
        else ""
    )

    out_dtype = types.Array(bodo.datetime64ns, 1, "C")
    scalar_text = f"res[i] = {unbox_str}({box_str}(arg2).tz_localize(None).tz_localize(arg0).tz_convert(arg1).tz_localize(None))\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def convert_timezone_tz(target_tz, data):  # pragma: no cover
    pass


@overload(convert_timezone_tz, no_unliteral=True)
def overload_convert_timezone_tz(target_tz, data):
    """Handles cases where convert_timezone (tz version) receives optional
    arguments and forwards to the appropriate version of the real implementation"""
    args = [target_tz, data]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.convert_timezone_tz",
                ["target_tz", "data"],
                i,
            )

    def impl(target_tz, data):  # pragma: no cover
        return convert_timezone_tz_util(target_tz, data)

    return impl


def convert_timezone_tz_util(target_tz, data):  # pragma: no cover
    pass


@overload(convert_timezone_tz_util, no_unliteral=True)
def overload_convert_timezone_tz_util(target_tz, data):
    """
    Converts <data> to <target_tz> timezone, as if translating the wallclock time
    from a person in one part of the world to another part of the world in the
    same epoch moment.

    Args:
        target_tz (Literal[str]): string of timezone to convert to
        data (datetime/timestamp scalar/array): input data

    Returns:
        data with its timezone adjusted
    """
    if not is_overload_constant_str(target_tz):
        raise_bodo_error(
            "CONVERT_TIMEZONE currently only supported with constant strings passed in for time zones."
        )
    verify_timestamp_tz_arg(data, "convert_timezone_tz", "data")

    arg_names = ["target_tz", "data"]
    arg_types = [target_tz, data]
    propagate_null = [True, True]
    out_dtype = bodo.timestamptz_array_type
    scalar_text = "current_target_offset = arg1.utc_timestamp.tz_localize(arg0).utcoffset().value // 60_000_000_000\n"
    scalar_text += f"res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz(arg1.utc_timestamp, current_target_offset)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )
