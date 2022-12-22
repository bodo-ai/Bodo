# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements datetime array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
import pandas as pd
import pytz
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


@numba.generated_jit(nopython=True)
def add_interval(start_dt, interval):
    """Handles cases where adding intervals receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [start_dt, interval]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.add_interval_util", ["arr"], i
            )

    def impl(start_dt, interval):  # pragma: no cover
        return add_interval_util(start_dt, interval)

    return impl


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


def dayname(arr):  # pragma: no cover
    return


def dayofmonth(arr):  # pragma: no cover
    return


def dayofweek(arr):  # pragma: no cover
    return


def dayofweekiso(arr):  # pragma: no cover
    return


def dayofyear(arr):  # pragma: no cover
    return


def get_year(arr):  # pragma: no cover
    return


def get_quarter(arr):  # pragma: no cover
    return


def get_month(arr):  # pragma: no cover
    return


def get_week(arr):  # pragma: no cover
    return


def get_hour(arr):  # pragma: no cover
    return


def get_minute(arr):  # pragma: no cover
    return


def get_second(arr):  # pragma: no cover
    return


def get_millisecond(arr):  # pragma: no cover
    return


def get_microsecond(arr):  # pragma: no cover
    return


def get_nanosecond(arr):  # pragma: no cover
    return


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


@numba.generated_jit(nopython=True)
def add_interval_util(start_dt, interval):
    """A dedicated kernel adding a timedelta to a datetime

    Args:
        start_dt (datetime array/series/scalar): the datetimes that are being
        added to
        interval (timedelta array/series/scalar): the offset being added to start_dt

    Returns:
        datetime series/scalar: start_dt + interval
    """

    verify_datetime_arg_allow_tz(start_dt, "add_interval", "start_dt")
    time_zone = get_tz_if_exists(start_dt)

    arg_names = ["start_dt", "interval"]
    arg_types = [start_dt, interval]
    propagate_null = [True] * 2
    scalar_text = ""
    is_vector = bodo.utils.utils.is_array_typ(
        interval, True
    ) or bodo.utils.utils.is_array_typ(start_dt, True)
    extra_globals = None
    # Modified logic from add_interval_xxx functions
    if time_zone is not None:
        if bodo.hiframes.pd_offsets_ext.tz_has_transition_times(time_zone):
            tz_obj = pytz.timezone(time_zone)
            trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view("i8")
            deltas = np.array(tz_obj._transition_info)[:, 0]
            deltas = (
                (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                .astype(np.int64)
                .values
            )
            extra_globals = {"trans": trans, "deltas": deltas}
            scalar_text += f"start_value = arg0.value\n"
            scalar_text += "end_value = start_value + arg0.value\n"
            scalar_text += (
                "start_trans = np.searchsorted(trans, start_value, side='right') - 1\n"
            )
            scalar_text += (
                "end_trans = np.searchsorted(trans, end_value, side='right') - 1\n"
            )
            scalar_text += "offset = deltas[start_trans] - deltas[end_trans]\n"
            scalar_text += "arg1 = pd.Timedelta(arg1.value + offset)\n"
        scalar_text += f"res[i] = arg0 + arg1\n"
        out_dtype = bodo.DatetimeArrayType(time_zone)
    else:
        unbox_str = (
            "bodo.utils.conversion.unbox_if_tz_naive_timestamp" if is_vector else ""
        )
        box_str = "bodo.utils.conversion.box_if_dt64" if is_vector else ""
        scalar_text = f"res[i] = {unbox_str}({box_str}(arg0) + arg1)\n"

        out_dtype = types.Array(bodo.datetime64ns, 1, "C")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


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

    def overload_add_datetime_interval_util(amount, start_dt):
        verify_int_arg(amount, "add_interval_" + unit, "amount")
        verify_datetime_arg_allow_tz(start_dt, "add_interval_" + unit, "start_dt")
        time_zone = get_tz_if_exists(start_dt)

        arg_names = ["amount", "start_dt"]
        arg_types = [amount, start_dt]
        propagate_null = [True] * 2
        is_vector = bodo.utils.utils.is_array_typ(
            amount, True
        ) or bodo.utils.utils.is_array_typ(start_dt, True)
        extra_globals = None

        # Code path generated for timezone-aware data
        if time_zone is not None:

            # Find the transition times / deltas for the timezone in question.
            # These arrays will be lowered via global variables in the exec env
            if bodo.hiframes.pd_offsets_ext.tz_has_transition_times(time_zone):
                tz_obj = pytz.timezone(time_zone)
                trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view(
                    "i8"
                )
                deltas = np.array(tz_obj._transition_info)[:, 0]
                deltas = (
                    (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                    .astype(np.int64)
                    .values
                )
                extra_globals = {"trans": trans, "deltas": deltas}

            # Handle months/years via the following steps:
            # 1. Find the starting ns
            # 2. Find the ending ns by converting to tz-native then adding
            #    a date offset with the corresponding number of months/years
            # 3. Find the deltas in the starting & ending datetime by finding
            #    their positions within the transition times array
            # 4. Create a timedelta combines adds the ns jump from step 2 with
            #    the difference in deltas from step 3
            # (If the timezone does not have transitions, treat the offset
            #  as if it were zero)
            if unit in ("months", "years"):
                scalar_text = f"td = pd.DateOffset({unit}=arg0)\n"
                scalar_text += f"start_value = arg1.value\n"
                scalar_text += "end_value = (pd.Timestamp(arg1.value) + td).value\n"
                if bodo.hiframes.pd_offsets_ext.tz_has_transition_times(time_zone):
                    scalar_text += "start_trans = np.searchsorted(trans, start_value, side='right') - 1\n"
                    scalar_text += "end_trans = np.searchsorted(trans, end_value, side='right') - 1\n"
                    scalar_text += "offset = deltas[start_trans] - deltas[end_trans]\n"
                    scalar_text += (
                        "td = pd.Timedelta(end_value - start_value + offset)\n"
                    )
                else:
                    scalar_text += "td = pd.Timedelta(end_value - start_value)\n"

            # Handle months/years via the following steps:
            # 1. Find the starting ns
            # 2. Find the ending ns by extracting the ns and adding the ns
            #    value of the timedelta
            # 3. Find the deltas in the starting & ending datetime by finding
            #    their positions within the transition times array
            # 4. Create a timedelta combines adds the ns value of the timedelta
            #    with the difference in deltas from step 3
            # (If the timezone does not have transitions, skip these steps
            #  and just use the Timedelta used for step 2)
            else:
                if unit == "nanoseconds":
                    scalar_text = "td = pd.Timedelta(arg0)\n"
                else:
                    scalar_text = f"td = pd.Timedelta({unit}=arg0)\n"
                if bodo.hiframes.pd_offsets_ext.tz_has_transition_times(time_zone):
                    scalar_text += f"start_value = arg1.value\n"
                    scalar_text += "end_value = start_value + td.value\n"
                    scalar_text += "start_trans = np.searchsorted(trans, start_value, side='right') - 1\n"
                    scalar_text += "end_trans = np.searchsorted(trans, end_value, side='right') - 1\n"
                    scalar_text += "offset = deltas[start_trans] - deltas[end_trans]\n"
                    scalar_text += "td = pd.Timedelta(td.value + offset)\n"

            # Add the calculated timedelta to the original timestamp
            scalar_text += f"res[i] = arg1 + td\n"

            out_dtype = bodo.DatetimeArrayType(time_zone)

        # Code path generated for timezone-native data by directly adding to
        # a DateOffset or TimeDelta with the corresponding units
        else:

            # Scalars will return Timestamp values while vectors will remain
            # in datetime64 format
            unbox_str = (
                "bodo.utils.conversion.unbox_if_tz_naive_timestamp" if is_vector else ""
            )
            box_str = "bodo.utils.conversion.box_if_dt64" if is_vector else ""

            if unit in ("months", "years"):
                scalar_text = f"res[i] = {unbox_str}({box_str}(arg1) + pd.DateOffset({unit}=arg0))\n"
            elif unit == "nanoseconds":
                scalar_text = (
                    f"res[i] = {unbox_str}({box_str}(arg1) + pd.Timedelta(arg0))\n"
                )
            else:
                scalar_text = f"res[i] = {unbox_str}({box_str}(arg1) + pd.Timedelta({unit}=arg0))\n"

            out_dtype = types.Array(bodo.datetime64ns, 1, "C")

        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            extra_globals=extra_globals,
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


def dayname_util(arr):  # pragma: no cover
    return


def dayofmonth_util(arr):  # pragma: no cover
    return


def dayofweek_util(arr):  # pragma: no cover
    return


def dayofweekiso_util(arr):  # pragma: no cover
    return


def dayofyear_util(arr):  # pragma: no cover
    return


def get_year_util(arr):  # pragma: no cover
    return


def get_quarter_util(arr):  # pragma: no cover
    return


def get_month_util(arr):  # pragma: no cover
    return


def get_week_util(arr):  # pragma: no cover
    return


def get_hour_util(arr):  # pragma: no cover
    return


def get_minute_util(arr):  # pragma: no cover
    return


def get_second_util(arr):  # pragma: no cover
    return


def get_millisecond_util(arr):  # pragma: no cover
    return


def get_microsecond_util(arr):  # pragma: no cover
    return


def get_nanosecond_util(arr):  # pragma: no cover
    return


def create_dt_extract_fn_overload(fn_name):  # pragma: no cover
    def overload_func(arr):
        """Handles cases where this dt extraction function recieves optional
        arguments and forwards to the appropriate version of the real implementation"""
        if isinstance(arr, types.optional):
            return unopt_argument(
                f"bodo.libs.bodosql_array_kernels.{fn_name}",
                ["arr"],
                0,
            )

        func_text = "def impl(arr):\n"
        func_text += f"  return bodo.libs.bodosql_array_kernels.{fn_name}_util(arr)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_dt_extract_fn_util_overload(fn_name):  # pragma: no cover
    """Creates an overload function to support datetime extraction functions
       on a datetime.

    Args:
        fn_name: the function being implemented

    Returns:
        (function): a utility that takes in a datetime (either can be scalars
        or vectors) and returns the corresponding component based on the desired
        function.
    """

    def overload_dt_extract_fn(arr):
        if fn_name in (
            "get_hour",
            "get_minute",
            "get_second",
            "get_microsecond",
            "get_millisecond",
            "get_nanosecond",
        ):
            verify_time_or_datetime_arg_allow_tz(arr, fn_name, "arr")
        else:
            verify_datetime_arg_allow_tz(arr, fn_name, "arr")
        tz = get_tz_if_exists(arr)
        box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""
        # For Timestamp, ms and us are stored in the same value.
        # For Time, they are stored seperately.
        ms_str = "microsecond // 1000" if not is_valid_time_arg(arr) else "millisecond"
        format_strings = {
            "get_year": f"{box_str}(arg0).year",
            "get_quarter": f"{box_str}(arg0).quarter",
            "get_month": f"{box_str}(arg0).month",
            "get_week": f"{box_str}(arg0).week",
            "get_hour": f"{box_str}(arg0).hour",
            "get_minute": f"{box_str}(arg0).minute",
            "get_second": f"{box_str}(arg0).second",
            "get_millisecond": f"{box_str}(arg0).{ms_str}",
            "get_microsecond": f"{box_str}(arg0).microsecond",
            "get_nanosecond": f"{box_str}(arg0).nanosecond",
            # [BE-4098] TODO: switch this to be dictionary-encoded output
            "dayname": f"{box_str}(arg0).day_name()",
            "dayofmonth": f"{box_str}(arg0).day",
            "dayofweek": f"({box_str}(arg0).dayofweek + 1) % 7",
            "dayofweekiso": f"{box_str}(arg0).dayofweek + 1",
            "dayofyear": f"{box_str}(arg0).dayofyear",
        }
        dtypes = {
            "get_year": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_quarter": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_month": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_week": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_hour": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_minute": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_second": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_millisecond": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_microsecond": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "get_nanosecond": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "dayname": bodo.string_array_type,
            "dayofmonth": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "dayofweek": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "dayofweekiso": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
            "dayofyear": bodo.libs.int_arr_ext.IntegerArrayType(types.int64),
        }

        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]
        scalar_text = f"res[i] = {format_strings[fn_name]}"

        out_dtype = dtypes[fn_name]

        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_dt_extract_fn


def _install_dt_extract_fn_overload():
    """Creates and installs the overloads for datetime extraction functions"""
    funcs_utils_names = [
        ("get_year", get_year, get_year_util),
        ("get_quarter", get_quarter, get_quarter_util),
        ("get_month", get_month, get_month_util),
        ("get_week", get_week, get_week_util),
        ("get_hour", get_hour, get_hour_util),
        ("get_minute", get_minute, get_minute_util),
        ("get_second", get_second, get_second_util),
        ("get_millisecond", get_millisecond, get_millisecond_util),
        ("get_microsecond", get_microsecond, get_microsecond_util),
        ("get_nanosecond", get_nanosecond, get_nanosecond_util),
        ("dayname", dayname, dayname_util),
        ("dayofmonth", dayofmonth, dayofmonth_util),
        ("dayofweek", dayofweek, dayofweek_util),
        ("dayofweekiso", dayofweekiso, dayofweekiso_util),
        ("dayofyear", dayofyear, dayofyear_util),
    ]
    for fn_name, func, util in funcs_utils_names:
        func_overload_impl = create_dt_extract_fn_overload(fn_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_dt_extract_fn_util_overload(fn_name)
        overload(util)(util_overload_impl)


_install_dt_extract_fn_overload()


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
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
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

    verify_datetime_arg_allow_tz(arr, "LAST_DAY", "arr")
    time_zone = get_tz_if_exists(arr)

    # When returning a scalar we return a pd.Timestamp type.
    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]

    if time_zone is None:
        scalar_text = f"res[i] = {unbox_str}({box_str}(arg0) + pd.tseries.offsets.MonthEnd(n=0, normalize=True))"
        out_dtype = np.dtype("datetime64[ns]")
    else:
        scalar_text = "y = arg0.year\n"
        scalar_text += "m = arg0.month\n"
        scalar_text += "d = bodo.hiframes.pd_offsets_ext.get_days_in_month(y, m)\n"
        scalar_text += (
            f"res[i] = pd.Timestamp(year=y, month=m, day=d, tz={repr(time_zone)})\n"
        )
        out_dtype = bodo.DatetimeArrayType(time_zone)

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
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
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

    verify_datetime_arg_allow_tz(arr, "MONTHNAME", "arr")
    tz = get_tz_if_exists(arr)
    box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {box_str}(arg0).month_name()"

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
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
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
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
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
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
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

    verify_datetime_arg_allow_tz(arr, "YEAROFWEEKISO", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "dt = pd.Timestamp(arg0)\n"
    scalar_text += "res[i] = dt.isocalendar()[0]"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def to_days(arr):  # pragma: no cover
    pass


@overload(to_days)
def overload_to_days(arr):
    """
    Equivalent to MYSQL's TO_DAYS function. Returns the number of days passed since
    YEAR 0 of the gregorian calendar.

    Note: Since the SQL input must always be a Date we can assume the input always
    Timestamp never has any values smaller than days.

    Args:
        arr (types.Type): A tz-naive datetime array or Timestamp scalar.

    Returns:
        types.Type: Integer or Integer Array
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_days_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return to_days_util(arr)

    return impl


def to_days_util(arr):  # pragma: no cover
    pass


@overload(to_days_util)
def overload_to_days_util(arr):
    """
    Equivalent to MYSQL's TO_DAYS function. Returns the number of days passed since
    YEAR 0 of the gregorian calendar.

    Note: Since the SQL input must always be a Date we can assume the input always
    Timestamp never has any values smaller than days.

    Args:
        arr (types.Type): A tz-naive datetime array or Timestamp scalar.

    Returns:
        types.Type: Integer or Integer Array
    """
    verify_datetime_arg(arr, "TO_DAYS", "arr")
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    # Value to add to days since unix time
    prefix_code = "unix_days_to_year_zero = 719528\n"
    # divisor to convert value -> days
    prefix_code += "nanoseconds_divisor = 86400000000000\n"
    out_dtype = bodo.IntegerArrayType(types.int64)
    # Note if the input is an array then we just operate directly on datetime64
    # to avoid Timestamp boxing.
    is_input_arr = bodo.utils.utils.is_array_typ(arr, False)
    if is_input_arr:
        scalar_text = (
            "  in_value = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arg0)\n"
        )
    else:
        scalar_text = "  in_value = arg0.value\n"
    scalar_text += (
        "  res[i] = (in_value // nanoseconds_divisor) + unix_days_to_year_zero\n"
    )

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


def from_days(arr):  # pragma: no cover
    pass


@overload(from_days)
def overload_from_days(arr):
    """
    Equivalent to MYSQL's FROM_DAYS function. Returns the Date created from the
    number of days passed since YEAR 0 of the gregorian calendar.

    Note: Since the SQL output should be a date but we will output tz-naive
    Timestamp at this time.

    Args:
        arr (types.Type): A integer array or scalar.

    Returns:
        types.Type: dt64 array or Timestamp without a timezone
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.from_days_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return from_days_util(arr)

    return impl


def from_days_util(arr):  # pragma: no cover
    pass


@overload(from_days_util)
def overload_from_days_util(arr):
    """
    Equivalent to MYSQL's FROM_DAYS function. Returns the Date created from the
    number of days passed since YEAR 0 of the gregorian calendar.

    Note: Since the SQL output should be a date but we will output tz-naive
    Timestamp at this time.

    Args:
        arr (types.Type): A integer array or scalar.

    Returns:
        types.Type: dt64 array or Timestamp without a timezone
    """
    verify_int_arg(arr, "TO_DAYS", "arr")
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    is_input_arr = bodo.utils.utils.is_array_typ(arr, False)
    if is_input_arr:
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")
    else:
        out_dtype = bodo.pd_timestamp_tz_naive_type

    # Value to subtract to days to get to unix time
    prefix_code = "unix_days_to_year_zero = 719528\n"
    # multiplier to convert days -> nanoseconds
    prefix_code += "nanoseconds_divisor = 86400000000000\n"
    scalar_text = (
        "  nanoseconds = (arg0 - unix_days_to_year_zero) * nanoseconds_divisor\n"
    )
    if is_input_arr:
        # Avoid unboxing into a Timestamp for the array case.
        scalar_text += (
            "  res[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(nanoseconds)\n"
        )
    else:
        scalar_text += "  res[i] = pd.Timestamp(nanoseconds)\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


def to_seconds(arr):  # pragma: no cover
    pass


@overload(to_seconds)
def overload_to_seconds(arr):
    """
    Equivalent to MYSQL's TO_SECONDS function. Returns the number of seconds passed since
    YEAR 0 of the gregorian calendar.

    Note: Since the SQL input truncates any values less than seconds.

    Args:
        arr (types.Type): A tz-naive or tz-aware array or Timestamp scalar.

    Returns:
        types.Type: Integer or Integer Array
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_seconds_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return to_seconds_util(arr)

    return impl


def to_seconds_util(arr):  # pragma: no cover
    pass


@overload(to_seconds_util)
def overload_to_seconds_util(arr):
    """
    Equivalent to MYSQL's TO_SECONDS function. Returns the number of seconds passed since
    YEAR 0 of the gregorian calendar.

    Note: Since the SQL input truncates any values less than seconds.

    Args:
        arr (types.Type): A tz-naive or tz-aware array or Timestamp scalar.

    Returns:
        types.Type: Integer or Integer Array
    """
    verify_datetime_arg_allow_tz(arr, "TO_SECONDS", "arr")
    timezone = get_tz_if_exists(arr)
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    # Value to add to seconds since unix time
    prefix_code = "unix_seconds_to_year_zero = 62167219200\n"
    # divisor to convert value -> seconds.
    # Note: This function does a floordiv for < seconds
    prefix_code += "nanoseconds_divisor = 1000000000\n"
    out_dtype = bodo.IntegerArrayType(types.int64)
    is_input_arr = bodo.utils.utils.is_array_typ(arr, False)
    if is_input_arr and not timezone:
        # Note if the input is an array then we just operate directly on datetime64
        # to avoid Timestamp boxing.
        scalar_text = (
            f"  in_value = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arg0)\n"
        )
    else:
        scalar_text = f"  in_value = arg0.value\n"
    # Note: This function just calculates the seconds since via UTC time, so this is
    # accurate for all timezones.
    scalar_text += (
        "  res[i] = (in_value // nanoseconds_divisor) + unix_seconds_to_year_zero\n"
    )

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )
