# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements datetime array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


def add_interval_years(amount, start_dt):  # pragma: no cover
    return


def add_interval_months(amount, start_dt):  # pragma: no cover
    return


def add_interval_weeks(amount, start_dt):  # pragma: no cover
    return


def add_interval_days(amount, start_dt):  # pragma: no cover
    return


def add_interval_hours(amount, start_dt):  # pragma: no cover
    return


def add_interval_minutes(amount, start_dt):  # pragma: no cover
    return


def add_interval_seconds(amount, start_dt):  # pragma: no cover
    return


def add_interval_milliseconds(amount, start_dt):  # pragma: no cover
    return


def add_interval_microseconds(amount, start_dt):  # pragma: no cover
    return


def add_interval_nanoseconds(amount, start_dt):  # pragma: no cover
    return


@numba.generated_jit(nopython=True)
def dayname(arr):
    """Handles cases where DAYNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.dayname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return dayname_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def day_timestamp(arr):
    """Handles cases where day_timestamp receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.day_timestamp_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return day_timestamp_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def int_to_days(arr):
    """Handles cases where int_to_days receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.int_to_days_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return int_to_days_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def last_day(arr):
    """Handles cases where LAST_DAY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.last_day_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return last_day_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def makedate(year, day):
    """Handles cases where MAKEDATE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [year, day]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.makedate", ["year", "day"], i
            )

    def impl(year, day):  # pragma: no cover
        return makedate_util(year, day)

    return impl


@numba.generated_jit(nopython=True)
def monthname(arr):
    """Handles cases where MONTHNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.monthname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return monthname_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def month_diff(arr0, arr1):
    """Handles cases where month_diff receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.month_diff",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return month_diff_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def next_day(arr0, arr1):
    """Handles cases where next_day receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.next_day",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return next_day_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def previous_day(arr0, arr1):
    """Handles cases where previous_day receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.previous_day",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return previous_day_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def second_timestamp(arr):
    """Handles cases where second_timestamp receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.second_timestamp_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return second_timestamp_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def weekday(arr):
    """Handles cases where WEEKDAY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.weekday_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return weekday_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def yearofweekiso(arr):
    """Handles cases where YEAROFWEEKISO receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.yearofweekiso_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return yearofweekiso_util(arr)

    return impl


def add_interval_years_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_months_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_weeks_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_days_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_hours_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_minutes_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_seconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_milliseconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_microseconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_nanoseconds_util(amount, start_dt):  # pragma: no cover
    return


def create_add_interval_func_overload(unit):  # pragma: no cover
    def overload_func(amount, start_dt):
        """Handles cases where this interval addition function recieves optional
        arguments and forwards to the appropriate version of the real implementation"""
        args = [amount, start_dt]
        for i in range(2):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.add_interval_{unit}",
                    ["amount", "start_dt"],
                    i,
                )

        func_text = "def impl(amount, start_dt):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.add_interval_{unit}_util(amount, start_dt)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_add_interval_util_overload(unit):  # pragma: no cover
    """Creates an overload function to support add_interval functions on
       an integer and a datetime

    Args:
        unit: what is the unit of the integer argument

    Returns:
        (function): a utility that takes in an integer amount and a datetime
        (either can be scalars or vectors) and adds the integer amount (in
        the unit specified) to the datetime.
    """

    date_args = {"years", "months", "weeks", "days"}
    func = "DateOffset" if unit in date_args else "Timedelta"

    def overload_add_datetime_interval_util(amount, start_dt):
        verify_int_arg(amount, "add_interval_" + unit, "amount")
        verify_datetime_arg(start_dt, "add_interval_" + unit, "start_dt")

        arg_names = ["amount", "start_dt"]
        arg_types = [amount, start_dt]
        propagate_null = [True] * 2
        # Scalars will return Timestamp values while vectors will remain
        # in datetime64 format
        unbox_str = (
            "bodo.utils.conversion.unbox_if_timestamp"
            if bodo.utils.utils.is_array_typ(amount, True)
            or bodo.utils.utils.is_array_typ(start_dt, True)
            else ""
        )
        # pd.Timedelta does not have a nanosecond argument, instead when an
        # integer is passed in without a keyword-arg it is assumed to be ns
        if unit == "nanoseconds":
            expr = "bodo.utils.conversion.box_if_dt64(arg1) + pd.Timedelta(arg0)"
        else:
            expr = f"bodo.utils.conversion.box_if_dt64(arg1) + pd.{func}({unit}=arg0)"
        scalar_text = f"res[i] = {unbox_str}({expr})"

        out_dtype = np.dtype("datetime64[ns]")

        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
        )

    return overload_add_datetime_interval_util


def _install_add_interval_overload():
    """Creates and installs the overloads for interval addition functions"""
    funcs_utils_names = [
        ("years", add_interval_years, add_interval_years_util),
        ("months", add_interval_months, add_interval_months_util),
        ("weeks", add_interval_weeks, add_interval_weeks_util),
        ("days", add_interval_days, add_interval_days_util),
        ("hours", add_interval_hours, add_interval_hours_util),
        ("minutes", add_interval_minutes, add_interval_minutes_util),
        ("seconds", add_interval_seconds, add_interval_seconds_util),
        ("milliseconds", add_interval_milliseconds, add_interval_milliseconds_util),
        ("microseconds", add_interval_microseconds, add_interval_microseconds_util),
        ("nanoseconds", add_interval_nanoseconds, add_interval_nanoseconds_util),
    ]
    for unit, func, util in funcs_utils_names:
        func_overload_impl = create_add_interval_func_overload(unit)
        overload(func)(func_overload_impl)
        util_overload_impl = create_add_interval_util_overload(unit)
        overload(util)(util_overload_impl)


_install_add_interval_overload()


@numba.generated_jit(nopython=True)
def dayname_util(arr):
    """A dedicated kernel for the SQL function DAYNAME which takes in a datetime
    and returns the day of the week as a string


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose dayname is being
        searched for

    Returns:
        string series/scalar: the day of the week from the input timestamp(s)
    """

    verify_datetime_arg_allow_tz(arr, "DAYNAME", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = bodo.utils.conversion.box_if_dt64(arg0).day_name()"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def day_timestamp_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    a timestamp in days.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to datetime(s)

    Returns:
        datetime series/scalar: the number/column in days
    """

    verify_int_arg(arr, "day_timestamp", "arr")
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0, unit='D'))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def int_to_days_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    interval days.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to timedelta(s)

    Returns:
        timedelta series/scalar: the number/column of days
    """

    verify_int_arg(arr, "int_to_days", "arr")
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {unbox_str}(pd.Timedelta(days=arg0))"

    out_dtype = np.dtype("timedelta64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def last_day_util(arr):
    """A dedicated kernel for the SQL function LAST_DAY which takes in a datetime
    and returns the last day from that month


    Args:
        arr (datetime array/series/scalar): the timestamp whose last day is being
        searched for

    Returns:
        datetime series/scalar: the last day(s) from the month(s)
    """

    verify_datetime_arg(arr, "LAST_DAY", "arr")
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0) + pd.tseries.offsets.MonthEnd(n=0, normalize=True))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def makedate_util(year, day):
    """A dedicated kernel for the SQL function MAKEDATE which takes in two integers
    (or columns) and uses them to construct a specific date


    Args:
        year (int array/series/scalar): the year(s) of the timestamp
        day (int array/series/scalar): the day(s) of the year of the timestamp

    Returns:
        datetime series/scalar: the constructed date(s)
    """
    verify_int_arg(year, "MAKEDATE", "year")
    verify_int_arg(day, "MAKEDATE", "day")
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if bodo.utils.utils.is_array_typ(year, True)
        or bodo.utils.utils.is_array_typ(day, True)
        else ""
    )

    arg_names = ["year", "day"]
    arg_types = [year, day]
    propagate_null = [True] * 2
    scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(year=arg0, month=1, day=1) + pd.Timedelta(days=arg1-1))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def monthname_util(arr):
    """A dedicated kernel for the SQL function MONTHNAME which takes in a datetime
    and returns the name of the month


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose month name is being
        searched for

    Returns:
        string series/scalar: the month name from the input timestamp(s)
    """

    verify_datetime_arg(arr, "MONTHNAME", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = pd.Timestamp(arg0).month_name()"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def next_day_util(arr0, arr1):
    """A dedicated kernel for the SQL function NEXT_DAY which takes in a datetime
    and a string and returns the previous day of the week from the input datetime


    Args:
        arr0 (datetime array/series/scalar): the timestamp(s) with the day in question
        arr1 (string array/series/scalar): the day of the week whose previous day is being
        searched for

    Returns:
        datetime series/scalar: the previous day of the week from the input timestamp(s)
    """

    verify_datetime_arg_allow_tz(arr0, "NEXT_DAY", "arr0")
    verify_string_arg(arr1, "NEXT_DAY", "arr1")
    is_input_tz_aware = is_valid_tz_aware_datetime_arg(arr0)

    # When returning a scalar we always return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if not is_input_tz_aware
        and (
            bodo.utils.utils.is_array_typ(arr0, True)
            or bodo.utils.utils.is_array_typ(arr1, True)
        )
        else ""
    )

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    # TODO: lower the dictionary as a global rather that defined in the function text
    prefix_code = (
        "dow_map = {'mo': 0, 'tu': 1, 'we': 2, 'th': 3, 'fr': 4, 'sa': 5, 'su': 6}"
    )
    # Note: Snowflake removes leading whitespace and ignore any characters aside from the first two
    # values, case insensitive. https://docs.snowflake.com/en/sql-reference/functions/next_day.html#arguments
    scalar_text = f"arg1_trimmed = arg1.lstrip()[:2].lower()\n"
    if is_input_tz_aware:
        arg0_timestamp = "arg0"
    else:
        arg0_timestamp = "bodo.utils.conversion.box_if_dt64(arg0)"
    scalar_text += f"res[i] = {unbox_str}({arg0_timestamp}.normalize() + pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed]))\n"

    if is_input_tz_aware:
        out_dtype = bodo.DatetimeArrayType(arr0.tz)
    else:
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def previous_day_util(arr0, arr1):
    """A dedicated kernel for the SQL function PREVIOUS_DAY which takes in a datetime
    and a string and returns the previous day of the week from the input datetime


    Args:
        arr0 (datetime array/series/scalar): the timestamp(s) with the day in question
        arr1 (string array/series/scalar): the day of the week whose previous day is being
        searched for

    Returns:
        datetime series/scalar: the previous day of the week from the input timestamp(s)
    """

    verify_datetime_arg_allow_tz(arr0, "PREVIOUS_DAY", "arr0")
    verify_string_arg(arr1, "PREVIOUS_DAY", "arr1")
    is_input_tz_aware = is_valid_tz_aware_datetime_arg(arr0)

    # When returning a scalar we always return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if not is_input_tz_aware
        and (
            bodo.utils.utils.is_array_typ(arr0, True)
            or bodo.utils.utils.is_array_typ(arr1, True)
        )
        else ""
    )

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    # TODO: lower the dictionary as a global rather that defined in the function text
    prefix_code = (
        "dow_map = {'mo': 0, 'tu': 1, 'we': 2, 'th': 3, 'fr': 4, 'sa': 5, 'su': 6}"
    )
    # Note: Snowflake removes leading whitespace and ignore any characters aside from the first two
    # values, case insensitive. https://docs.snowflake.com/en/sql-reference/functions/previous_day.html#arguments
    scalar_text = f"arg1_trimmed = arg1.lstrip()[:2].lower()\n"
    if is_input_tz_aware:
        arg0_timestamp = "arg0"
    else:
        arg0_timestamp = "bodo.utils.conversion.box_if_dt64(arg0)"
    scalar_text += f"res[i] = {unbox_str}({arg0_timestamp}.normalize() - pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed]))\n"

    if is_input_tz_aware:
        out_dtype = bodo.DatetimeArrayType(arr0.tz)
    else:
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def month_diff_util(arr0, arr1):
    """A dedicated kernel for obtaining the floor of the difference in months
    between two Datetimes (or columns)


    Args:
        arr0 (datetime array/series/scalar): the date(s) being subtracted from
        arr1 (datetime array/series/scalar): the date(s) being subtracted

    Returns:
        int series/scalar: the difference in months between the two dates
    """

    verify_datetime_arg(arr0, "month_diff", "arr0")
    verify_datetime_arg(arr1, "month_diff", "arr1")

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    scalar_text = "A0 = bodo.utils.conversion.box_if_dt64(arg0)\n"
    scalar_text += "A1 = bodo.utils.conversion.box_if_dt64(arg1)\n"
    scalar_text += "delta = 12 * (A0.year - A1.year) + (A0.month - A1.month)\n"
    scalar_text += "remainder = ((A0 - pd.DateOffset(months=delta)) - A1).value\n"
    scalar_text += "if delta > 0 and remainder < 0:\n"
    scalar_text += "   res[i] = -(delta - 1)\n"
    scalar_text += "elif delta < 0 and remainder > 0:\n"
    scalar_text += "   res[i] = -(delta + 1)\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = -delta"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def second_timestamp_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    a timestamp in seconds.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to datetime(s)

    Returns:
        datetime series/scalar: the number/column in seconds
    """

    verify_int_arg(arr, "second_timestamp", "arr")
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0, unit='s'))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def weekday_util(arr):
    """A dedicated kernel for the SQL function WEEKDAY which takes in a datetime
    and returns the day of the week (enumerated 0-6)


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose day of the
        week is being searched for

    Returns:
        int series/scalar: the day of the week from the input timestamp(s)
    """

    verify_datetime_arg(arr, "WEEKDAY", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "dt = pd.Timestamp(arg0)\n"
    scalar_text += "res[i] = bodo.hiframes.pd_timestamp_ext.get_day_of_week(dt.year, dt.month, dt.day)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def yearofweekiso_util(arr):
    """A dedicated kernel for the SQL function YEAROFWEEKISO which takes in a datetime
    (or column) and returns the year of the input date(s)


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose year is being
        searched for

    Returns:
        int series/scalar: the year from the input timestamp(s)
    """

    verify_datetime_arg(arr, "YEAROFWEEKISO", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "dt = pd.Timestamp(arg0)\n"
    scalar_text += "res[i] = dt.isocalendar()[0]"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
