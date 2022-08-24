# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements datetime array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
from numba.core import types

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


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

    verify_datetime_arg(arr, "DAYNAME", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = pd.Timestamp(arg0).day_name()"

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

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0, unit='D'))"

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

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = (
        "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timedelta(days=arg0))"
    )

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

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0) + pd.tseries.offsets.MonthEnd(n=0, normalize=True))"

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

    arg_names = ["year", "day"]
    arg_types = [year, day]
    propagate_null = [True] * 2
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(year=arg0, month=1, day=1) + pd.Timedelta(days=arg1-1))"

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
def month_diff_util(arr0, arr1):
    """A dedicated kernel for obtaining the floor of the difference in months
    between two Datetimes (or columns)


    Args:
        arr0 (datetime array/series/scalar): the date(s) being subtraced from
        arr1 (datetime array/series/scalar): the date(s) being subtraced

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

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0, unit='s'))"

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
