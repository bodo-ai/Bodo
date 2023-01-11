# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements datetime array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
import pandas as pd
import pytz
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import raise_bodo_error


def standardize_snowflake_date_time_part(part_str):  # pragma: no cover
    pass


@overload(standardize_snowflake_date_time_part)
def overload_standardize_snowflake_date_time_part(part_str):
    """
    Standardizes all of the valid snowflake aliases for
    Date and Time parts into the standard categories.
    See: https://docs.snowflake.com/en/sql-reference/functions-date-time.html#label-supported-date-time-parts

    Args:
        part_str (types.unicode_type): String representing the name or time part or alias.

    Raises:
        ValueError: An invalid string is passed in.

    Returns:
        types.unicode_type: The date or time part converting all aliases to standard part.
    """
    # Note we lower arrays to reduce compilation time as there would be many large
    # tuples or lists

    # Date values with aliases
    year_aliases = pd.array(["year", "y", "yy", "yyy", "yyyy", "yr", "years", "yrs"])
    month_aliases = pd.array(["month", "mm", "mon", "mons", "months"])
    day_aliases = pd.array(["day", "d", "dd", "days", "dayofmonth"])
    dayofweek_aliases = pd.array(["dayofweek", "weekday", "dow", "dw"])
    week_aliases = pd.array(["week", "w", "wk", "weekofyear", "woy", "wy"])
    weekiso_aliases = pd.array(
        ["weekiso", "week_iso", "weekofyeariso", "weekofyear_iso"]
    )
    quarter_aliases = pd.array(["quarter", "q", "qtr", "qtrs", "quarters"])

    # Time values with aliases
    hour_aliases = pd.array(["hour", "h", "hh", "hr", "hours", "hrs"])
    minute_aliases = pd.array(["minute", "m", "mi", "min", "minutes", "mins"])
    second_aliases = pd.array(["second", "s", "sec", "seconds", "secs"])
    millisecond_aliases = pd.array(["millisecond", "ms", "msec", "milliseconds"])
    microsecond_aliases = pd.array(["microsecond", "us", "usec", "microseconds"])
    nanosecond_aliases = pd.array(
        [
            "nanosecond",
            "ns",
            "nsec",
            "nanosec",
            "nsecond",
            "nanoseconds",
            "nanosecs",
            "nseconds",
        ]
    )
    epoch_second_aliases = pd.array(["epoch_second", "epoch", "epoch_seconds"])
    epoch_millisecond_aliases = pd.array(["epoch_millisecond", "epoch_milliseconds"])
    epoch_microsecond_aliases = pd.array(["epoch_microsecond", "epoch_microseconds"])
    epoch_nanosecond_aliases = pd.array(["epoch_nanosecond", "epoch_nanoseconds"])
    timezone_hour_aliases = pd.array(["timezone_hour", "tzh"])
    timezone_minute_aliases = pd.array(["timezone_minute", "tzm"])

    # These values map to themselves and have no aliases
    no_aliases = pd.array(["yearofweek", "yearofweekiso"])

    def impl(part_str):  # pragma: no cover
        # Snowflake date/time parts are case insensitive
        part_str = part_str.lower()
        if part_str in year_aliases:
            return "year"
        elif part_str in month_aliases:
            return "month"
        elif part_str in day_aliases:
            return "day"
        elif part_str in dayofweek_aliases:
            return "dayofweek"
        elif part_str in week_aliases:
            return "week"
        elif part_str in weekiso_aliases:
            return "weekiso"
        elif part_str in quarter_aliases:
            return "quarter"
        elif part_str in hour_aliases:
            return "hour"
        elif part_str in minute_aliases:
            return "minute"
        elif part_str in second_aliases:
            return "second"
        elif part_str in millisecond_aliases:
            return "millisecond"
        elif part_str in microsecond_aliases:
            return "microsecond"
        elif part_str in nanosecond_aliases:
            return "nanosecond"
        elif part_str in epoch_second_aliases:
            return "epoch_second"
        elif part_str in epoch_millisecond_aliases:
            return "epoch_millisecond"
        elif part_str in epoch_microsecond_aliases:
            return "epoch_microsecond"
        elif part_str in epoch_nanosecond_aliases:
            return "epoch_nanosecond"
        elif part_str in timezone_hour_aliases:
            return "timezone_hour"
        elif part_str in timezone_minute_aliases:
            return "timezone_minute"
        elif part_str in no_aliases:
            return part_str
        else:
            # TODO: Add part_str in the error when we can have non constant exceptions
            raise ValueError(
                "Invalid date or time part passed into Snowflake array kernel"
            )

    return impl


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


def add_interval_quarters(amount, start_dt):  # pragma: no cover
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
def dayname(arr):  # pragma: no cover
    """Handles cases where DAYNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.dayname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return dayname_util(arr)

    return impl


def dayofmonth(arr):  # pragma: no cover
    return


def dayofweek(arr):  # pragma: no cover
    return


def dayofweekiso(arr):  # pragma: no cover
    return


def dayofyear(arr):  # pragma: no cover
    return


def diff_day(arr0, arr1):  # pragma: no cover
    return


def diff_hour(arr0, arr1):  # pragma: no cover
    return


def diff_microsecond(arr0, arr1):  # pragma: no cover
    return


def diff_minute(arr0, arr1):  # pragma: no cover
    return


def diff_month(arr0, arr1):  # pragma: no cover
    return


def diff_nanosecond(arr0, arr1):  # pragma: no cover
    return


def diff_quarter(arr0, arr1):  # pragma: no cover
    return


def diff_second(arr0, arr1):  # pragma: no cover
    return


def diff_week(arr0, arr1):  # pragma: no cover
    return


def diff_year(arr0, arr1):  # pragma: no cover
    return


def get_year(arr):  # pragma: no cover
    return


def get_quarter(arr):  # pragma: no cover
    return


def get_month(arr):  # pragma: no cover
    return


def get_week(arr):  # pragma: no cover
    return


def get_weekofyear(arr):  # pragma: no cover
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
            scalar_text += "end_value = start_value + arg1.value\n"
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
        scalar_text = f"res[i] = arg0 + arg1\n"

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


def add_interval_quarters_util(amount, start_dt):  # pragma: no cover
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
        """Handles cases where this interval addition function receives optional
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
        if unit in (
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ):
            verify_time_or_datetime_arg_allow_tz(
                start_dt, "add_interval_" + unit, "start_dt"
            )
        else:
            verify_datetime_arg_allow_tz(start_dt, "add_interval_" + unit, "start_dt")
        time_zone = get_tz_if_exists(start_dt)

        arg_names = ["amount", "start_dt"]
        arg_types = [amount, start_dt]
        propagate_null = [True] * 2
        is_vector = bodo.utils.utils.is_array_typ(
            amount, True
        ) or bodo.utils.utils.is_array_typ(start_dt, True)
        extra_globals = None

        # Code path generated for time data
        if is_valid_time_arg(start_dt):
            precision = start_dt.precision
            if unit == "hours":
                unit_val = 3600000000000
            elif unit == "minutes":
                unit_val = 60000000000
            elif unit == "seconds":
                unit_val = 1000000000
            elif unit == "milliseconds":
                precision = max(precision, 3)
                unit_val = 1000000
            elif unit == "microseconds":
                precision = max(precision, 6)
                unit_val = 1000
            elif unit == "nanoseconds":
                precision = max(precision, 9)
                unit_val = 1
            scalar_text = f"amt = bodo.hiframes.time_ext.cast_time_to_int(arg1) + {unit_val} * arg0\n"
            scalar_text += f"res[i] = bodo.hiframes.time_ext.cast_int_to_time(amt % 86400000000000, precision={precision})"
            out_dtype = types.Array(bodo.hiframes.time_ext.TimeType(precision), 1, "C")

        # Code path generated for timezone-aware data
        elif time_zone is not None:

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
            if unit in ("months", "quarters", "years"):
                if unit == "quarters":
                    scalar_text = f"td = pd.DateOffset(months=3*arg0)\n"
                else:
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

            # Handle other units via the following steps:
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
            elif unit == "quarters":
                scalar_text = f"res[i] = {unbox_str}({box_str}(arg1) + pd.DateOffset(months=3*arg0))\n"
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
        ("quarters", add_interval_quarters, add_interval_quarters_util),
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


def get_weekofyear_util(arr):  # pragma: no cover
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
        # For Time, they are stored separately.
        ms_str = "microsecond // 1000" if not is_valid_time_arg(arr) else "millisecond"
        format_strings = {
            "get_year": f"{box_str}(arg0).year",
            "get_quarter": f"{box_str}(arg0).quarter",
            "get_month": f"{box_str}(arg0).month",
            "get_week": f"{box_str}(arg0).week",
            "get_weekofyear": f"{box_str}(arg0).weekofyear",
            "get_hour": f"{box_str}(arg0).hour",
            "get_minute": f"{box_str}(arg0).minute",
            "get_second": f"{box_str}(arg0).second",
            "get_millisecond": f"{box_str}(arg0).{ms_str}",
            "get_microsecond": f"{box_str}(arg0).microsecond",
            "get_nanosecond": f"{box_str}(arg0).nanosecond",
            "dayofmonth": f"{box_str}(arg0).day",
            "dayofweek": f"({box_str}(arg0).dayofweek + 1) % 7",
            "dayofweekiso": f"{box_str}(arg0).dayofweek + 1",
            "dayofyear": f"{box_str}(arg0).dayofyear",
        }

        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]
        scalar_text = f"res[i] = {format_strings[fn_name]}"

        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

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
        ("get_weekofyear", get_weekofyear, get_weekofyear_util),
        ("get_hour", get_hour, get_hour_util),
        ("get_minute", get_minute, get_minute_util),
        ("get_second", get_second, get_second_util),
        ("get_millisecond", get_millisecond, get_millisecond_util),
        ("get_microsecond", get_microsecond, get_microsecond_util),
        ("get_nanosecond", get_nanosecond, get_nanosecond_util),
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


def diff_day_util(arr0, arr1):  # pragma: no cover
    return


def diff_hour_util(arr0, arr1):  # pragma: no cover
    return


def diff_microsecond_util(arr0, arr1):  # pragma: no cover
    return


def diff_minute_util(arr0, arr1):  # pragma: no cover
    return


def diff_month_util(arr0, arr1):  # pragma: no cover
    return


def diff_nanosecond_util(arr0, arr1):  # pragma: no cover
    return


def diff_quarter_util(arr0, arr1):  # pragma: no cover
    return


def diff_second_util(arr0, arr1):  # pragma: no cover
    return


def diff_week_util(arr0, arr1):  # pragma: no cover
    return


def diff_year_util(arr0, arr1):  # pragma: no cover
    return


@register_jitable
def get_iso_weeks_between_years(year0, year1):  # pragma: no cover
    """Takes in two years and returns the number of ISO weeks between the first
       week of the first year and the first week of the second year.

       Logic for the calculations based on: https://en.wikipedia.org/wiki/ISO_week_date

    Args:
        year0 (integer): the first year
        year1 (integer): the second year

    Returns: the number of ISO weeks betwen year0 and year1
    """
    sign = 1
    if year1 < year0:
        year0, year1 = year1, year0
        sign = -1
    weeks = 0
    for y in range(year0, year1):
        weeks += 52
        # Calculate the starting day-of-week of the first week of the current
        # and previous year. If the current is a Thursday, or the previous is a
        # Wednesday, then the year has 53 weeks instead of 52
        dow_curr = (y + (y // 4) - (y // 100) + (y // 400)) % 7
        dow_prev = ((y - 1) + ((y - 1) // 4) - ((y - 1) // 100) + ((y - 1) // 400)) % 7
        if dow_curr == 4 or dow_prev == 3:
            weeks += 1
    return sign * weeks


def create_dt_diff_fn_overload(unit):  # pragma: no cover
    def overload_func(arr0, arr1):
        """Handles cases where this dt difference function recieves optional
        arguments and forwards to the appropriate version of the real implementation"""
        args = [arr0, arr1]
        for i in range(len(args)):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodo.libs.bodosql_array_kernels.diff_{unit}",
                    ["arr0", "arr1"],
                    i,
                )

        func_text = "def impl(arr0, arr1):\n"
        func_text += (
            f"  return bodo.libs.bodosql_array_kernels.diff_{unit}_util(arr0, arr1)"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_dt_diff_fn_util_overload(unit):  # pragma: no cover
    """Creates an overload function to support datetime difference functions.

    Args:
        unt: the unit that the difference shoud be returned in terms of.

    Returns:
        (function): a utility that takes in a two datetimes (either can be scalars
        or vectors) and returns the difference in the specified unit

    Note: the output dtype is int64 for NANOSECONDS and int32 for all other units,
    in agreement with Calcite's typing rules.
    """

    def overload_dt_diff_fn(arr0, arr1):
        verify_datetime_arg_allow_tz(arr0, "diff_" + unit, "arr0")
        verify_datetime_arg_allow_tz(arr1, "diff_" + unit, "arr1")
        tz = get_tz_if_exists(arr0)
        if get_tz_if_exists(arr1) != tz:
            raise_bodo_error(f"diff_{unit}: both arguments must have the same timezone")

        arg_names = ["arr0", "arr1"]
        arg_types = [arr0, arr1]
        propagate_null = [True] * 2
        extra_globals = None

        # A dictionary of variable definitions shared between the kernels
        diff_defns = {
            "yr_diff": "arg1.year - arg0.year",
            "qu_diff": "arg1.quarter - arg0.quarter",
            "mo_diff": "arg1.month - arg0.month",
            "y0, w0, _": "arg0.isocalendar()",
            "y1, w1, _": "arg1.isocalendar()",
            "iso_yr_diff": "bodo.libs.bodosql_array_kernels.get_iso_weeks_between_years(y0, y1)",
            "wk_diff": "w1 - w0",
            "da_diff": "(pd.Timestamp(arg1.year, arg1.month, arg1.day) - pd.Timestamp(arg0.year, arg0.month, arg0.day)).days",
            "ns_diff": "arg1.value - arg0.value",
        }
        # A dictionary mapping each kernel to the list of definitions needed'
        req_defns = {
            "year": ["yr_diff"],
            "quarter": ["yr_diff", "qu_diff"],
            "month": ["yr_diff", "mo_diff"],
            "week": ["y0, w0, _", "y1, w1, _", "iso_yr_diff", "wk_diff"],
            "day": ["da_diff"],
            "nanosecond": ["ns_diff"],
        }

        scalar_text = ""
        if tz == None:
            scalar_text += "arg0 = bodo.utils.conversion.box_if_dt64(arg0)\n"
            scalar_text += "arg1 = bodo.utils.conversion.box_if_dt64(arg1)\n"

        # Load in all of the required definitions
        for req_defn in req_defns.get(unit, []):
            scalar_text += f"{req_defn} = {diff_defns[req_defn]}\n"

        if unit == "nanosecond":
            out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)
        else:
            out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

        if unit == "year":
            scalar_text += "res[i] = yr_diff"
        elif unit == "quarter":
            scalar_text += "res[i] = 4 * yr_diff + qu_diff"
        elif unit == "month":
            scalar_text += "res[i] = 12 * yr_diff + mo_diff"
        elif unit == "week":
            scalar_text += "res[i] = iso_yr_diff + wk_diff"
        elif unit == "day":
            scalar_text += "res[i] = da_diff"
        elif unit == "nanosecond":
            scalar_text += "res[i] = ns_diff"
        else:
            if unit == "hour":
                divisor = 3600000000000
            if unit == "minute":
                divisor = 60000000000
            if unit == "second":
                divisor = 1000000000
            if unit == "microsecond":
                divisor = 1000
            scalar_text += f"res[i] = np.floor_divide((arg1.value), ({divisor})) - np.floor_divide((arg0.value), ({divisor}))\n"

        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            extra_globals=extra_globals,
        )

    return overload_dt_diff_fn


def _install_dt_diff_fn_overload():
    """Creates and installs the overloads for datetime difference functions"""
    funcs_utils_names = [
        ("day", diff_day, diff_day_util),
        ("hour", diff_hour, diff_hour_util),
        ("microsecond", diff_microsecond, diff_microsecond_util),
        ("minute", diff_minute, diff_minute_util),
        ("month", diff_month, diff_month_util),
        ("nanosecond", diff_nanosecond, diff_nanosecond_util),
        ("quarter", diff_quarter, diff_quarter),
        ("second", diff_second, diff_second_util),
        ("week", diff_week, diff_week_util),
        ("year", diff_year, diff_year_util),
    ]
    for unit, func, util in funcs_utils_names:
        func_overload_impl = create_dt_diff_fn_overload(unit)
        overload(func)(func_overload_impl)
        util_overload_impl = create_dt_diff_fn_util_overload(unit)
        overload(util)(util_overload_impl)


_install_dt_diff_fn_overload()


def date_trunc(date_or_time_part, ts_arg):  # pragma: no cover
    pass


@overload(date_trunc)
def overload_date_trunc(date_or_time_part, ts_arg):
    """
    Truncates a given Timestamp argument to the provided
    date_or_time_part. This corresponds to DATE_TRUNC inside snowflake

    Args:
        date_or_time_part (types.Type): A string scalar or array stating how to truncate
            the timestamp
        tz_arg (types.Type): A tz-aware or tz-naive Timestamp or Timestamp array to be truncated.

    Returns:
        types.Type: A tz-aware or tz-naive Timestamp or Timestamp array
    """
    if isinstance(date_or_time_part, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.date_trunc",
            ["date_or_time_part", "ts_arg"],
            0,
        )
    if isinstance(ts_arg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.date_trunc",
            ["date_or_time_part", "ts_arg"],
            1,
        )

    def impl(date_or_time_part, ts_arg):  # pragma: no cover
        return date_trunc_util(date_or_time_part, ts_arg)

    return impl


def date_trunc_util(date_or_time_part, ts_arg):  # pragma: no cover
    pass


@overload(date_trunc_util)
def overload_date_trunc_util(date_or_time_part, ts_arg):
    """
    Truncates a given Timestamp argument to the provided
    date_or_time_part. This corresponds to DATE_TRUNC inside snowflake

    Args:
        date_or_time_part (types.Type): A string scalar or array stating how to truncate
            the timestamp
        tz_arg (types.Type): A tz-aware or tz-naive Timestamp or Timestamp array to be truncated.

    Returns:
        types.Type: A tz-aware or tz-naive Timestamp or Timestamp array
    """
    verify_string_arg(date_or_time_part, "DATE_TRUNC", "date_or_time_part")
    verify_datetime_arg_allow_tz(ts_arg, "DATE_TRUNC", "ts_arg")
    tz_literal = get_tz_if_exists(ts_arg)

    arg_names = ["date_or_time_part", "ts_arg"]
    arg_types = [date_or_time_part, ts_arg]
    propagate_null = [True, True]
    # We perform computation on Timestamp types.
    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(ts_arg, True)
        else ""
    )
    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(ts_arg, True)
        else ""
    )
    # Standardize the input to limit the condition in the loop
    scalar_text = "part_str = bodo.libs.bodosql_array_kernels.standardize_snowflake_date_time_part(arg0)\n"
    if tz_literal is None:
        scalar_text += f"arg1 = {box_str}(arg1)\n"
    scalar_text += "if part_str == 'quarter':\n"
    scalar_text += "    out_val = pd.Timestamp(year=arg1.year, month= (3*(arg1.quarter - 1)) + 1, day=1, tz=tz_literal)\n"
    scalar_text += "elif part_str == 'year':\n"
    scalar_text += (
        "    out_val = pd.Timestamp(year=arg1.year, month=1, day=1, tz=tz_literal)\n"
    )
    scalar_text += "elif part_str == 'month':\n"
    scalar_text += "    out_val = pd.Timestamp(year=arg1.year, month=arg1.month, day=1, tz=tz_literal)\n"
    scalar_text += "elif part_str == 'day':\n"
    scalar_text += "    out_val = arg1.normalize()\n"
    scalar_text += "elif part_str == 'week':\n"
    # If we are already at the start of the week just normalize.
    scalar_text += "    if arg1.dayofweek == 0:\n"
    scalar_text += "        out_val = arg1.normalize()\n"
    scalar_text += "    else:\n"
    scalar_text += (
        "        out_val = arg1.normalize() - pd.tseries.offsets.Week(n=1, weekday=0)\n"
    )
    scalar_text += "elif part_str == 'hour':\n"
    scalar_text += "    out_val = arg1.floor('H')\n"
    scalar_text += "elif part_str == 'minute':\n"
    scalar_text += "    out_val = arg1.floor('min')\n"
    scalar_text += "elif part_str == 'second':\n"
    scalar_text += "    out_val = arg1.floor('S')\n"
    scalar_text += "elif part_str == 'millisecond':\n"
    scalar_text += "    out_val = arg1.floor('ms')\n"
    scalar_text += "elif part_str == 'microsecond':\n"
    scalar_text += "    out_val = arg1.floor('us')\n"
    scalar_text += "elif part_str == 'nanosecond':\n"
    scalar_text += "    out_val = arg1\n"
    scalar_text += "else:\n"
    # TODO: Include part_str when non-constant exception strings are supported.
    scalar_text += "    raise ValueError('Invalid date or time part for DATE_TRUNC')\n"
    if tz_literal is None:
        # In the tz-naive array case we have to convert the Timestamp to dt64
        scalar_text += f"res[i] = {unbox_str}(out_val)\n"
    else:
        scalar_text += f"res[i] = out_val\n"

    if tz_literal is None:
        out_dtype = types.Array(bodo.datetime64ns, 1, "C")
    else:
        out_dtype = bodo.DatetimeArrayType(tz_literal)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals={"tz_literal": tz_literal},
    )


@numba.generated_jit(nopython=True)
def dayname_util(arr):
    """A dedicated kernel for returning the name of the day of the week of
       a datetime (or column of datetimes).


    Args:
        arr (datetime array/series/scalar): the datetime(s) whose day name's
        are being sought.

    Returns:
        string array/scalar: the name of the day of the week of the datetime(s).
        Returns a dictionary encoded array if the input is an array.
    """
    verify_datetime_arg_allow_tz(arr, "dayname", "arr")
    tz = get_tz_if_exists(arr)
    box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {box_str}(arg0).day_name()"
    out_dtype = bodo.string_array_type

    # If the input is an array, make the output dictionary encoded
    synthesize_dict_if_vector = ["V"]
    dows = pd.array(
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ],
    )
    extra_globals = {"day_of_week_dict_arr": dows}

    synthesize_dict_setup_text = "dict_res = day_of_week_dict_arr"
    synthesize_dict_scalar_text = f"res[i] = {box_str}(arg0).dayofweek"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        synthesize_dict_if_vector=synthesize_dict_if_vector,
        synthesize_dict_setup_text=synthesize_dict_setup_text,
        synthesize_dict_scalar_text=synthesize_dict_scalar_text,
        extra_globals=extra_globals,
        synthesize_dict_global=True,
        synthesize_dict_unique=True,
    )


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
        Returns a dictionary encoded array if the input is an array.
    """
    verify_datetime_arg_allow_tz(arr, "monthname", "arr")
    tz = get_tz_if_exists(arr)
    box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = f"res[i] = {box_str}(arg0).month_name()"
    out_dtype = bodo.string_array_type

    # If the input is an array, make the output dictionary encoded
    synthesize_dict_if_vector = ["V"]
    mons = pd.array(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
    )
    extra_globals = {"month_names_dict_arr": mons}
    synthesize_dict_setup_text = "dict_res = month_names_dict_arr"
    synthesize_dict_scalar_text = f"res[i] = {box_str}(arg0).month - 1"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        synthesize_dict_if_vector=synthesize_dict_if_vector,
        synthesize_dict_setup_text=synthesize_dict_setup_text,
        synthesize_dict_scalar_text=synthesize_dict_scalar_text,
        extra_globals=extra_globals,
        synthesize_dict_global=True,
        synthesize_dict_unique=True,
    )


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
        if (
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
    scalar_text += f"new_timestamp = {arg0_timestamp}.normalize() + pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed])\n"
    # The output is suppose to be a date. Since we output a Timestamp still we need to make it naive.
    scalar_text += f"res[i] = {unbox_str}(new_timestamp.tz_localize(None))\n"

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
        if (
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
    scalar_text += f"new_timestamp = {arg0_timestamp}.normalize() - pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed])\n"
    # The output is suppose to be a date. Since we output a Timestamp still we need to make it naive.
    scalar_text += f"res[i] = {unbox_str}(new_timestamp.tz_localize(None))\n"

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
    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )
    scalar_text = f"dt = {box_str}(arg0)\n"
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
    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )
    scalar_text = f"dt = {box_str}(arg0)\n"
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


def tz_aware_interval_add(tz_arg, interval_arg):  # pragma: no cover
    pass


@overload(tz_aware_interval_add)
def overload_tz_aware_interval_add(tz_arg, interval_arg):
    """
    Equivalent to adding a SQL interval type to a Timezone aware
    Timestamp argument. The interval type can either be a pd.DatetimeOffset
    or a pd.Timedelta. In either case the Timestamp value should move by the
    effective amount of time, updating the Timestamp by additional time if we
    have to cross a UTC offset, no matter the unit, so the local time is always
    updated.

    For example if the Timestamp is

        Timestamp('2022-11-06 00:59:59-0400', tz='US/Eastern')

    and we add 2 hours, then the end time is

        Timestamp('2022-11-06 02:59:59-0500', tz='US/Eastern')


    We make this decision because although time has advanced more than 2 hours,
    this ensures all extractable fields (e.g. Days, Hours, etc.) advance by the
    desired amount of time. There is not well defined behavior here as Snowflake
    never handles Daylight Savings, so we opt to resemble Snowflake's output. This
    could change in the future based on customer feedback.

    Args:
        tz_arg (types.Type): A tz-aware array or Timestamp scalar.
        interval_arg (types.Type): The interval, either a Timedelta scalar/array or a DateOffset.

    Returns:
        types.Type: A tz-aware array or Timestamp scalar.
    """
    if isinstance(tz_arg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.tz_aware_interval_add",
            ["tz_arg", "interval_arg"],
            0,
        )
    if isinstance(interval_arg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.tz_aware_interval_add",
            ["tz_arg", "interval_arg"],
            1,
        )

    def impl(tz_arg, interval_arg):  # pragma: no cover
        return tz_aware_interval_add_util(tz_arg, interval_arg)

    return impl


def tz_aware_interval_add_util(tz_arg, interval_arg):  # pragma: no cover
    pass


@overload(tz_aware_interval_add_util)
def overload_tz_aware_interval_add_util(tz_arg, interval_arg):
    """
    Equivalent to adding a SQL interval type to a Timezone aware
    Timestamp argument. The interval type can either be a pd.DatetimeOffset
    or a pd.Timedelta. In either case the Timestamp value should move by the
    effective amount of time, updating the Timestamp by additional time if we
    have to cross a UTC offset, no matter the unit, so the local time is always
    updated.

    For example if the Timestamp is

        Timestamp('2022-11-06 00:59:59-0400', tz='US/Eastern')

    and we add 2 hours, then the end time is

        Timestamp('2022-11-06 02:59:59-0500', tz='US/Eastern')


    We make this decision because although time has advanced more than 2 hours,
    this ensures all extractable fields (e.g. Days, Hours, etc.) advance by the
    desired amount of time. There is not well defined behavior here as Snowflake
    never handles Daylight Savings, so we opt to resemble Snowflake's output. This
    could change in the future based on customer feedback.

    Args:
        tz_arg (types.Type): A tz-aware array or Timestamp scalar.
        interval_arg (types.Type): The interval, either a Timedelta scalar/array or a DateOffset.

    Returns:
        types.Type: A tz-aware array or Timestamp scalar.
    """
    verify_datetime_arg_require_tz(tz_arg, "INTERVAL_ADD", "tz_arg")
    verify_sql_interval(interval_arg, "INTERVAL_ADD", "interval_arg")
    timezone = get_tz_if_exists(tz_arg)
    arg_names = ["tz_arg", "interval_arg"]
    arg_types = [tz_arg, interval_arg]
    propagate_null = [True, True]
    if timezone is not None:
        out_dtype = bodo.DatetimeArrayType(timezone)
    else:
        # Handle a default case if the timezone value is NA.
        # Note this doesn't matter because we will output NA.
        out_dtype = bodo.datetime64ns
    # Note: We don't have support for TZAware + pd.DateOffset yet.
    # As a result we must compute a Timedelta from the DateOffset instead.
    if interval_arg == bodo.date_offset_type:
        # Although the pd.DateOffset should just have months and n, its unclear if
        # months >= 12 can ever roll over into months and years. As a result we convert
        # the years into months to be more robust (via years * 12).
        scalar_text = "  timedelta = bodo.libs.pd_datetime_arr_ext.convert_months_offset_to_days(arg0.year, arg0.month, arg0.day, ((arg1._years * 12) + arg1._months) * arg1.n)\n"
    else:
        scalar_text = "  timedelta = arg1\n"
    # Check for changing utc offsets
    scalar_text += "  timedelta = bodo.hiframes.pd_offsets_ext.update_timedelta_with_transition(arg0, timedelta)\n"
    scalar_text += "  res[i] = arg0 + timedelta\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )
