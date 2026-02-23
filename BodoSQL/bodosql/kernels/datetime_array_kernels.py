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
import bodosql
from bodo.hiframes.datetime_date_ext import DatetimeDateArrayType
from bodo.hiframes.pd_offsets_ext import CombinedIntervalType
from bodo.libs.pd_datetime_arr_ext import (
    python_timezone_from_bodo_timezone_info,
)
from bodo.utils.typing import (
    BodoError,
    assert_bodo_error,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.kernels.array_kernel_utils import (
    convert_numeric_to_int,
    gen_vectorized,
    get_tz_if_exists,
    is_valid_date_arg,
    is_valid_time_arg,
    is_valid_timestamptz_arg,
    is_valid_tz_aware_datetime_arg,
    unopt_argument,
    verify_date_arg,
    verify_date_or_datetime_arg_forbid_tz,
    verify_datetime_arg,
    verify_datetime_arg_allow_tz,
    verify_datetime_arg_require_tz,
    verify_int_arg,
    verify_int_float_arg,
    verify_sql_interval,
    verify_string_arg,
    verify_td_arg,
    verify_time_or_datetime_arg_allow_tz,
    verify_timestamp_arg_allow_tz,
)


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


# overload_standardize_snowflake_date_time_part is only usable at runtime,
# setting the function like this should make it possible to call
# from regular python since this function never type checks
standardize_snowflake_date_time_part_compile_time = (
    overload_standardize_snowflake_date_time_part
)


@numba.generated_jit(nopython=True)
def add_interval(start_dt, interval):
    """Handles cases where adding intervals receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [start_dt, interval]
    arg_names = ["start_dt", "interval"]

    return convert_numeric_to_int(
        "bodosql.kernels.datetime_array_kernels.add_interval_unopt_util",
        arg_names,
        args,
        ["interval"],
    )


@numba.generated_jit(nopython=True)
def add_interval_unopt_util(start_dt, interval):
    """Handles cases where adding intervals receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [start_dt, interval]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.datetime_array_kernels.add_interval_unopt_util",
                ["start_dt", "interval"],
                i,
            )

    if isinstance(interval, CombinedIntervalType):

        def impl(start_dt, interval):  # pragma: no cover
            result = start_dt
            for interval_part in interval.intervals:
                result = add_interval_util(result, interval_part)
            return result

    else:

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
def construct_timestamp(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    """Handles cases where TIMESTAMP_FROM_PARTS receives numeric
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day, hour, minute, second, nanosecond, time_zone]
    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]

    return convert_numeric_to_int(
        "bodosql.kernels.datetime_array_kernels.construct_timestamp_unopt_util",
        arg_names,
        args,
        ["year", "month", "day", "hour", "minute", "second", "nanosecond"],
    )


@numba.generated_jit(nopython=True)
def construct_timestamp_unopt_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    """Handles cases where TIMESTAMP_FROM_PARTS receives optional
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day, hour, minute, second, nanosecond, time_zone]
    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.datetime_array_kernels.construct_timestamp_unopt_util",
                arg_names,
                i,
            )

    def impl(year, month, day, hour, minute, second, nanosecond, time_zone):
        return bodosql.kernels.datetime_array_kernels.construct_timestamp_util(
            year, month, day, hour, minute, second, nanosecond, time_zone
        )

    return impl


def timestamp_tz_from_parts(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    pass


@overload(timestamp_tz_from_parts, no_unliteral=True)
def overload_timestamp_tz_from_parts(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    """Handles cases where TIMESTAMP_TZ_FROM_PARTS receives numeric
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day, hour, minute, second, nanosecond, time_zone]
    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]

    return convert_numeric_to_int(
        "bodosql.kernels.datetime_array_kernels.timestamp_tz_from_parts_unopt_util",
        arg_names,
        args,
        ["year", "month", "day", "hour", "minute", "second", "nanosecond"],
    )


def timestamp_tz_from_parts_unopt_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    pass


@overload(timestamp_tz_from_parts_unopt_util, no_unliteral=True)
def overload_timestamp_tz_from_parts_unopt_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    """Handles cases where TIMESTAMP_TZ_FROM_PARTS receives optional
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day, hour, minute, second, nanosecond, time_zone]
    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.datetime_array_kernels.timestamp_tz_from_parts_unopt_util",
                arg_names,
                i,
            )

    def impl(year, month, day, hour, minute, second, nanosecond, time_zone):
        return bodosql.kernels.datetime_array_kernels.timestamp_tz_from_parts_util(
            year, month, day, hour, minute, second, nanosecond, time_zone
        )

    return impl


@numba.generated_jit(nopython=True)
def date_from_parts(year, month, day):  # pragma: no cover
    """Handles cases where DATE_FROM_PARTS receives numeric
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day]
    arg_names = ["year", "month", "day"]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.date_from_parts",
                arg_names,
                i,
            )

    return convert_numeric_to_int(
        "bodosql.kernels.datetime_array_kernels.date_from_parts_unopt_util",
        arg_names,
        args,
        arg_names,
    )


@numba.generated_jit(nopython=True)
def date_from_parts_unopt_util(year, month, day):  # pragma: no cover
    """Handles cases where DATE_FROM_PARTS receives optional
    arguments and forwards to the appropriate version of the real implementation"""
    args = [year, month, day]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.datetime_array_kernels.date_from_parts_unopt_util",
                ["year", "month", "day"],
                i,
            )

    def impl(year, month, day):
        return bodosql.kernels.datetime_array_kernels.date_from_parts_util(
            year, month, day
        )

    return impl


@numba.generated_jit(nopython=True)
def dayname(arr):  # pragma: no cover
    """Handles cases where DAYNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.datetime_array_kernels.dayname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return dayname_util(arr)

    return impl


def dayofmonth(arr):  # pragma: no cover
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


def diff_millisecond(arr0, arr1):  # pragma: no cover
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
            "bodosql.kernels.datetime_array_kernels.int_to_days_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return int_to_days_util(arr)

    return impl


def last_day_year(date_or_time_expr):  # pragma: no cover
    return


def last_day_quarter(date_or_time_expr):  # pragma: no cover
    return


def last_day_month(date_or_time_expr):  # pragma: no cover
    return


def last_day_week(date_or_time_expr):  # pragma: no cover
    return


@numba.generated_jit(nopython=True)
def makedate(year, day):
    """Handles cases where MAKEDATE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [year, day]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument("bodosql.kernels.makedate", ["year", "day"], i)

    def impl(year, day):  # pragma: no cover
        return makedate_util(year, day)

    return impl


@numba.generated_jit(nopython=True)
def monthname(arr):
    """Handles cases where MONTHNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.datetime_array_kernels.monthname_util", ["arr"], 0
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
                "bodosql.kernels.next_day",
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
                "bodosql.kernels.previous_day",
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
            "bodosql.kernels.datetime_array_kernels.second_timestamp_util", ["arr"], 0
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
            "bodosql.kernels.datetime_array_kernels.weekday_util", ["arr"], 0
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
            "bodosql.kernels.datetime_array_kernels.yearofweekiso_util", ["arr"], 0
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
    verify_time_or_datetime_arg_allow_tz(
        start_dt, "add_interval", "start_dt", allow_timestamp_tz=True
    )
    verify_sql_interval(interval, "add_interval", "interval")
    time_zone = get_tz_if_exists(start_dt)

    arg_names = ["start_dt", "interval"]
    arg_types = [start_dt, interval]
    propagate_null = [True] * 2
    scalar_text = ""
    extra_globals = None

    # Scalars will return Timestamp values while vectors will remain
    # in datetime64 format
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(start_dt, True)
        or bodo.utils.utils.is_array_typ(interval, True)
        else ""
    )
    box_str0 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(start_dt, True)
        else ""
    )
    box_str1 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(interval, True)
        else ""
    )

    if is_valid_time_arg(start_dt):
        scalar_text += "td_val = bodo.utils.conversion.box_if_dt64(arg1).value\n"
        scalar_text += "value = (arg0.value + td_val) % 86400000000000\n"
        scalar_text += "res[i] = bodo.types.Time(nanosecond=value)"
        out_dtype = bodo.types.TimeArrayType(9)
    elif is_valid_date_arg(start_dt):
        # If the time unit is smaller than or equal to hour, returns timestamp objects
        scalar_text += f"res[i] = {unbox_str}(pd.Timestamp(arg0) + {box_str1}(arg1))\n"
        out_dtype = bodo.types.DatetimeArrayType(None)
    # Modified logic from add_interval_xxx functions
    elif time_zone is not None:
        if (
            bodo.hiframes.pd_timestamp_ext.tz_has_transition_times(time_zone)
            and interval != bodo.types.date_offset_type
        ):
            tz_obj = pytz.timezone(time_zone)
            trans = np.array(tz_obj._utc_transition_times, dtype="M8[ns]").view("i8")
            deltas = np.array(tz_obj._transition_info)[:, 0]
            deltas = (
                (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                .astype(np.int64)
                .values
            )
            extra_globals = {"trans": trans, "deltas": deltas}
            scalar_text += "start_value = arg0.value\n"
            # Wrap the interval in a pd.Timedelta if dealing with an array of intervals
            if bodo.utils.utils.is_array_typ(interval, True):
                scalar_text += "arg1 = bodo.utils.conversion.box_if_dt64(arg1)\n"
            scalar_text += "end_value = start_value + arg1.value\n"
            scalar_text += (
                "start_trans = np.searchsorted(trans, start_value, side='right') - 1\n"
            )
            scalar_text += (
                "end_trans = np.searchsorted(trans, end_value, side='right') - 1\n"
            )
            scalar_text += "offset = deltas[start_trans] - deltas[end_trans]\n"
            scalar_text += "arg1 = pd.Timedelta(end_value - start_value + offset)\n"
        scalar_text += "res[i] = arg0 + arg1\n"
        out_dtype = bodo.types.DatetimeArrayType(time_zone)
    elif is_valid_timestamptz_arg(start_dt):
        # For TIMESTAMP_TZ, add the timedelta to the local timestamp, then create
        # a new TIMESTAMP_TZ from that local timestamp and the original offset.
        # We do this because the datetime arithmetic semantics are different
        # versus if we add the timedelta/offset directly to the UTC timestamp.
        scalar_text = (
            "local_ts = bodo.hiframes.timestamptz_ext.get_local_timestamp(arg0)\n"
        )
        scalar_text += "new_local_ts = local_ts + arg1\n"
        scalar_text += "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz_from_local(new_local_ts, arg0.offset_minutes)\n"
        out_dtype = bodo.types.timestamptz_array_type
    else:
        # For regular timestamps, perform the standard arithmetic on the datetime and
        # interval after unwrapping them, then re-wrap the result
        scalar_text = f"res[i] = {unbox_str}({box_str0}(arg0) + {box_str1}(arg1))\n"
        out_dtype = bodo.types.DatetimeArrayType(None)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


def add_interval_years_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_years_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_quarters_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_quarters_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_months_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_months_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_weeks_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_weeks_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_days_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_days_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_hours_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_hours_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_minutes_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_minutes_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_seconds_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_seconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_milliseconds_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_milliseconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_microseconds_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_microseconds_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_nanoseconds_unopt_util(amount, start_dt):  # pragma: no cover
    return


def add_interval_nanoseconds_util(amount, start_dt):  # pragma: no cover
    return


def create_add_interval_func_overload(unit):  # pragma: no cover
    def overload_func(amount, start_dt):
        """Handles cases where this interval addition function receives numeric
        arguments and forwards to the appropriate version of the real implementation"""
        args = [amount, start_dt]
        arg_names = ["amount", "start_dt"]
        return convert_numeric_to_int(
            f"bodosql.kernels.datetime_array_kernels.add_interval_{unit}_unopt_util",
            arg_names,
            args,
            ["amount"],
        )

    return overload_func


def create_add_interval_func_unopt_util_overload(unit):  # pragma: no cover
    def overload_func(amount, start_dt):
        """Handles cases where this interval addition function receives optional
        arguments and forwards to the appropriate version of the real implementation"""
        args = [amount, start_dt]
        for i in range(2):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodosql.kernels.add_interval_{unit}",
                    ["amount", "start_dt"],
                    i,
                )

        func_text = "def impl(amount, start_dt):\n"

        verify_int_arg(amount, "add_interval_" + unit, "amount")
        func_text += f"  return bodosql.kernels.datetime_array_kernels.add_interval_{unit}_util(amount, start_dt)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql, "np": np}, loc_vars)

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
        if unit in (
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ):
            verify_time_or_datetime_arg_allow_tz(
                start_dt, "add_interval_" + unit, "start_dt", allow_timestamp_tz=True
            )
        else:
            verify_datetime_arg_allow_tz(
                start_dt, "add_interval_" + unit, "start_dt", allow_timestamp_tz=True
            )
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

        # Code path generated for date data
        elif is_valid_date_arg(start_dt):
            unbox_str = (
                "bodo.utils.conversion.unbox_if_tz_naive_timestamp" if is_vector else ""
            )
            if unit in ("years", "quarters", "months", "weeks", "days"):
                # datetime.timedelta doesn't take years, quarters and months as argument,
                # need to calculate manually
                if unit == "years":
                    scalar_text = "year = arg1.year + arg0\n"
                    scalar_text += "res[i] = datetime.date(year, arg1.month, arg1.day)"
                elif unit == "months":
                    scalar_text = "year = arg1.year + (arg1.month + arg0 - 1) // 12\n"
                    scalar_text += "month = (arg1.month + arg0 - 1) % 12 + 1\n"
                    scalar_text += "res[i] = datetime.date(year, month, arg1.day)"
                elif unit == "quarters":
                    scalar_text = (
                        "year = arg1.year + (arg1.month + 3 * arg0 - 1) // 12\n"
                    )
                    scalar_text += "month = (arg1.month + 3 * arg0 - 1) % 12 + 1\n"
                    scalar_text += "res[i] = datetime.date(year, month, arg1.day)"
                # weeks and days
                else:
                    scalar_text = f"td = datetime.timedelta({unit}=arg0)\n"
                    scalar_text += "res[i] = arg1 + td"
                # If the time unit is larger than or equal to day, returns date objects
                out_dtype = DatetimeDateArrayType()
            else:
                if unit == "nanoseconds":
                    scalar_text = "td = pd.Timedelta(arg0)\n"
                else:
                    scalar_text = f"td = pd.Timedelta({unit}=arg0)\n"
                scalar_text += f"res[i] = {unbox_str}(pd.Timestamp(arg1) + td)"
                # If the time unit is smaller than or equal to hour, returns timestamp objects
                out_dtype = bodo.types.DatetimeArrayType(None)

        # Code path generated for timezone-aware data
        elif time_zone is not None:
            # Find the transition times / deltas for the timezone in question.
            # These arrays will be lowered via global variables in the exec env
            if bodo.hiframes.pd_timestamp_ext.tz_has_transition_times(time_zone):
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
                    scalar_text = "td = pd.DateOffset(months=3*arg0)\n"
                else:
                    scalar_text = f"td = pd.DateOffset({unit}=arg0)\n"
                scalar_text += "start_value = arg1.value\n"
                scalar_text += "end_value = (pd.Timestamp(arg1.value) + td).value\n"
                if bodo.hiframes.pd_timestamp_ext.tz_has_transition_times(time_zone):
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
                if bodo.hiframes.pd_timestamp_ext.tz_has_transition_times(time_zone):
                    scalar_text += "start_value = arg1.value\n"
                    scalar_text += "end_value = start_value + td.value\n"
                    scalar_text += "start_trans = np.searchsorted(trans, start_value, side='right') - 1\n"
                    scalar_text += "end_trans = np.searchsorted(trans, end_value, side='right') - 1\n"
                    scalar_text += "offset = deltas[start_trans] - deltas[end_trans]\n"
                    scalar_text += "td = pd.Timedelta(td.value + offset)\n"

            # Add the calculated timedelta to the original timestamp
            scalar_text += "res[i] = arg1 + td\n"

            out_dtype = bodo.types.DatetimeArrayType(time_zone)

        # Code path generated for timezone-native data by directly adding to
        # a DateOffset or TimeDelta with the corresponding units
        else:
            is_timestamp_tz = is_valid_timestamptz_arg(start_dt)
            # Setup the logic required to unwrap the argument and re-wrap it as
            # the correct type for the answer. For TIMESTAMP_TZ, the unwrapping
            # is getting the local timestamp and the re-wrapping is converting it
            # back to a TIMESTAMP_TZ using the original offset. For NTZ, scalars will
            # return Timestamp values while vectors will remain in datetime64 format.
            wrap_str = (
                "bodo.hiframes.timestamptz_ext.init_timestamptz_from_local"
                if is_timestamp_tz
                else (
                    "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
                    if is_vector
                    else ""
                )
            )
            wrap_suffix = ", arg1.offset_minutes" if is_timestamp_tz else ""
            unwrap_str = (
                "bodo.hiframes.timestamptz_ext.get_local_timestamp"
                if is_timestamp_tz
                else ("bodo.utils.conversion.box_if_dt64" if is_vector else "")
            )

            if unit in ("months", "years"):
                scalar_text = f"res[i] = {wrap_str}({unwrap_str}(arg1) + pd.DateOffset({unit}=arg0){wrap_suffix})\n"
            elif unit == "quarters":
                scalar_text = f"res[i] = {wrap_str}({unwrap_str}(arg1) + pd.DateOffset(months=3*arg0){wrap_suffix})\n"
            elif unit == "nanoseconds":
                scalar_text = f"res[i] = {wrap_str}({unwrap_str}(arg1) + pd.Timedelta(arg0){wrap_suffix})\n"
            else:
                scalar_text = f"res[i] = {wrap_str}({unwrap_str}(arg1) + pd.Timedelta({unit}=arg0){wrap_suffix})\n"

            out_dtype = (
                bodo.types.timestamptz_array_type
                if is_timestamp_tz
                else bodo.types.DatetimeArrayType(None)
            )

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
        "years",
        "quarters",
        "months",
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    ]
    for unit in funcs_utils_names:
        func_overload_impl = create_add_interval_func_overload(unit)
        overload(eval(f"add_interval_{unit}"))(func_overload_impl)
        func_overload_impl = create_add_interval_func_unopt_util_overload(unit)
        overload(eval(f"add_interval_{unit}_unopt_util"))(func_overload_impl)
        util_overload_impl = create_add_interval_util_overload(unit)
        overload(eval(f"add_interval_{unit}_util"))(util_overload_impl)


_install_add_interval_overload()


def last_day_year_util(arr):  # pragma: no cover
    return


def last_day_quarter_util(arr):  # pragma: no cover
    return


def last_day_month_util(arr):  # pragma: no cover
    return


def last_day_week_util(arr):  # pragma: no cover
    return


def create_last_day_func_overload(unit):  # pragma: no cover
    def overload_func(date_or_time_expr):
        """Handles cases where LAST_DAY receives optional arguments and forwards
        to the appropriate version of the real implementation"""

        if isinstance(date_or_time_expr, types.optional):  # pragma: no cover
            return unopt_argument(
                f"bodosql.kernels.last_day_{unit}",
                ["date_or_time_expr"],
                0,
            )

        func_text = "def impl(date_or_time_expr):\n"
        func_text += f"  return bodosql.kernels.datetime_array_kernels.last_day_{unit}_util(date_or_time_expr)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_last_day_util_overload(unit):
    def overload_last_day_util(date_or_time_expr):
        verify_datetime_arg_allow_tz(
            date_or_time_expr,
            "last_day_" + unit,
            "date_or_time_expr",
            allow_timestamp_tz=True,
        )

        if is_valid_timestamptz_arg(date_or_time_expr):
            box_str = "bodo.hiframes.timestamptz_ext.get_local_timestamp"
        elif bodo.utils.utils.is_array_typ(date_or_time_expr, True):
            box_str = "bodo.utils.conversion.box_if_dt64"
        else:
            box_str = ""

        arg_names = ["date_or_time_expr"]
        arg_types = [date_or_time_expr]
        propagate_null = [True]

        scalar_text = f"arg0 = {box_str}(arg0)\n"
        if unit == "year":
            scalar_text += "res[i] = datetime.date(arg0.year, 12, 31)\n"
        elif unit == "quarter":
            # month 1-3 is the first quarter in a year,
            # and the last month of the nth quarter is the (3*n)th month
            scalar_text += "y = arg0.year\n"
            scalar_text += "m = ((arg0.month - 1) // 3 + 1) * 3\n"
            scalar_text += "d = bodo.hiframes.pd_offsets_ext.get_days_in_month(y, m)\n"
            scalar_text += "res[i] = datetime.date(y, m, d)\n"
        elif unit == "month":
            scalar_text += "y = arg0.year\n"
            scalar_text += "m = arg0.month\n"
            scalar_text += "d = bodo.hiframes.pd_offsets_ext.get_days_in_month(y, m)\n"
            scalar_text += "res[i] = datetime.date(y, m, d)\n"
        else:  # week
            if is_valid_date_arg(date_or_time_expr):
                scalar_text += (
                    "res[i] = arg0 + datetime.timedelta(days=(6-arg0.weekday()))\n"
                )
            else:  # timestamp need to be transformed to date first
                scalar_text += "res[i] = arg0.date() + datetime.timedelta(days=(6-arg0.weekday()))\n"

        out_dtype = DatetimeDateArrayType()

        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_last_day_util


def _install_last_day_overload():
    """Creates and installs the overloads for last_day functions"""
    funcs_utils_names = [
        ("year", last_day_year, last_day_year_util),
        ("quarter", last_day_quarter, last_day_quarter_util),
        ("month", last_day_month, last_day_month_util),
        ("week", last_day_week, last_day_week_util),
    ]
    for unit, func, util in funcs_utils_names:
        func_overload_impl = create_last_day_func_overload(unit)
        overload(func)(func_overload_impl)
        util_overload_impl = create_last_day_util_overload(unit)
        overload(util)(util_overload_impl)


_install_last_day_overload()


def dayofmonth_util(arr):  # pragma: no cover
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
        """Handles cases where this dt extraction function receives optional
        arguments and forwards to the appropriate version of the real implementation"""
        if isinstance(arr, types.optional):
            return unopt_argument(
                f"bodosql.kernels.{fn_name}",
                ["arr"],
                0,
            )

        func_text = "def impl(arr):\n"
        func_text += (
            f"  return bodosql.kernels.datetime_array_kernels.{fn_name}_util(arr)"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def get_timestamp_unwrapping_str(ts_type, tz_to_utc=False):
    """
    Returns the string representation of the function required to
    unwrap a scalar of the given timestamp type.

    Args:
        ts_type (datatype): the type of the timestamp input
        tz_to_utc (bool): if True, convert timestamp_tz to its underlying
        UTC timestamp, if false convert it to its local timestamp.
    """
    if is_valid_timestamptz_arg(ts_type):
        # TIMESTAMP_TZ: extract the timestamp struct
        if tz_to_utc:
            return "bodo.hiframes.timestamptz_ext.get_utc_timestamp"
        else:
            return "bodo.hiframes.timestamptz_ext.get_local_timestamp"
    elif get_tz_if_exists(ts_type) is None:
        # TIMESTAMP_NTZ: convert datetime64 to a timestamp struct
        return "bodo.utils.conversion.box_if_dt64"
    else:
        # TIMESTAMP_LTZ: no unwrapping required
        return ""


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
            verify_time_or_datetime_arg_allow_tz(
                arr, fn_name, "arr", allow_timestamp_tz=True
            )
        else:
            verify_datetime_arg_allow_tz(arr, fn_name, "arr", allow_timestamp_tz=True)

        # Generate the code to convert arg0 to a timestamp once extracted
        # from the corresponding array
        unwrap_str = get_timestamp_unwrapping_str(arr)

        # For Timestamp, ms and us are stored in the same value.
        # For Time, they are stored separately.
        ms_str = "microsecond // 1000" if not is_valid_time_arg(arr) else "millisecond"
        us_str = "microsecond % 1000" if not is_valid_time_arg(arr) else "microsecond"

        # The specifications of how kernel should extract the relevant value
        # if the input is a date type
        date_format_strings = {
            "get_year": "arg0.year",
            "get_quarter": "(arg0.month + 2) // 3 ",
            "get_month": "arg0.month",
            "get_weekofyear": "arg0.isocalendar()[1]",
            "dayofmonth": "arg0.day",
            "dayofweekiso": "arg0.weekday() + 1",
            "dayofyear": "bodo.hiframes.datetime_date_ext._day_of_year(arg0.year, arg0.month, arg0.day)",
        }
        # The specifications of how kernel should extract the relevant value
        # if the input is a time or timestamp type
        other_format_strings = {
            "get_year": f"{unwrap_str}(arg0).year",
            "get_quarter": f"{unwrap_str}(arg0).quarter",
            "get_month": f"{unwrap_str}(arg0).month",
            "get_weekofyear": f"{unwrap_str}(arg0).weekofyear",
            "get_hour": f"{unwrap_str}(arg0).hour",
            "get_minute": f"{unwrap_str}(arg0).minute",
            "get_second": f"{unwrap_str}(arg0).second",
            "get_millisecond": f"{unwrap_str}(arg0).{ms_str}",
            "get_microsecond": f"{unwrap_str}(arg0).{us_str}",
            "get_nanosecond": f"{unwrap_str}(arg0).nanosecond",
            "dayofmonth": f"{unwrap_str}(arg0).day",
            "dayofweekiso": f"{unwrap_str}(arg0).dayofweek + 1",
            "dayofyear": f"{unwrap_str}(arg0).dayofyear",
        }

        arg_names = ["arr"]
        arg_types = [arr]
        propagate_null = [True]
        if is_valid_date_arg(arr):
            scalar_text = f"res[i] = {date_format_strings[fn_name]}"
        else:
            scalar_text = f"res[i] = {other_format_strings[fn_name]}"

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
        ("get_weekofyear", get_weekofyear, get_weekofyear_util),
        ("get_hour", get_hour, get_hour_util),
        ("get_minute", get_minute, get_minute_util),
        ("get_second", get_second, get_second_util),
        ("get_millisecond", get_millisecond, get_millisecond_util),
        ("get_microsecond", get_microsecond, get_microsecond_util),
        ("get_nanosecond", get_nanosecond, get_nanosecond_util),
        ("dayofmonth", dayofmonth, dayofmonth_util),
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


def diff_millisecond_util(arr0, arr1):  # pragma: no cover
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

    Returns: the number of ISO weeks between year0 and year1
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
        """Handles cases where this dt difference function receives optional
        arguments and forwards to the appropriate version of the real implementation"""
        args = [arr0, arr1]
        for i in range(len(args)):
            if isinstance(args[i], types.optional):
                return unopt_argument(
                    f"bodosql.kernels.diff_{unit}",
                    ["arr0", "arr1"],
                    i,
                )

        func_text = "def impl(arr0, arr1):\n"
        func_text += f"  return bodosql.kernels.datetime_array_kernels.diff_{unit}_util(arr0, arr1)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_dt_diff_fn_util_overload(unit):  # pragma: no cover
    """Creates an overload function to support datetime difference functions.

    Note: In Snowflake you can perform a datediff between a timestamp_ntz
    and a timestamp_ltz, in which case the ntz is cast to ltz. We replicate
    this behavior if we one input with a timezone and one without by casting
    the input without a timezone to the other's timezone.

    Args:
        unit: the unit that the difference should be returned in terms of.

    Returns:
        (function): a utility that takes in a two datetimes (either can be scalars
        or vectors) and returns the difference in the specified unit

    Note: the output dtype is int64 for NANOSECONDS and int32 for all other units,
    in agreement with Calcite's typing rules.
    """

    def overload_dt_diff_fn(arr0, arr1):
        # Handle the body of the loop
        scalar_text = ""
        # Globals for casting.
        extra_globals = {}

        # Check for valid time inputs. We don't support mixing
        # time with other types yet.
        if is_valid_time_arg(arr0) or is_valid_time_arg(arr1):
            supported = (is_overload_none(arr0) or is_valid_time_arg(arr0)) and (
                is_overload_none(arr1) or is_valid_time_arg(arr1)
            )
            if not supported:
                raise BodoError(
                    "DateDiff(): If a time type is provided both arguments must be time types."
                )
            first_arg = "arg0"
            second_arg = "arg1"
        else:
            verify_datetime_arg_allow_tz(
                arr0, "diff_" + unit, "arr0", allow_timestamp_tz=True
            )
            verify_datetime_arg_allow_tz(
                arr1, "diff_" + unit, "arr1", allow_timestamp_tz=True
            )
            # Determine all the information for casting
            is_date_arr0 = is_valid_date_arg(arr0)
            is_date_arr1 = is_valid_date_arg(arr1)
            arr0_tz = get_tz_if_exists(arr0)
            arr1_tz = get_tz_if_exists(arr1)
            # Set the cast information. Timezone either must
            # match or 1 must be None.
            if arr0_tz == arr1_tz:
                cast_tz = arr0_tz
            elif arr0_tz is None:
                cast_tz = arr1_tz
            elif arr1_tz is None:
                cast_tz = arr0_tz
            else:
                raise BodoError(
                    "DateDiff is not supported between two timezone aware columns with different timezones."
                )
            extra_globals["_cast_tz"] = cast_tz
            # Add casts to the prefix code.
            if is_date_arr0 or (arr0_tz != cast_tz):
                scalar_text += (
                    "arg0 = bodosql.kernels.to_timestamp(arg0, None, _cast_tz, 0)\n"
                )
            if is_date_arr1 or (arr1_tz != cast_tz):
                scalar_text += (
                    "arg1 = bodosql.kernels.to_timestamp(arg1, None, _cast_tz, 0)\n"
                )

            # Convenience function to unwrap the arguments to a common
            # representation
            def unwrap_arg(type_, arg_name):
                if is_valid_timestamptz_arg(type_):
                    # If we are working with timestamp_tz, we need to extract
                    # the local timestamp
                    return (
                        f"bodo.hiframes.timestamptz_ext.get_local_timestamp({arg_name})"
                    )
                elif cast_tz is None:
                    # If we are working with tz_naive data we need to keep the result in a timestamp.
                    return f"bodo.utils.conversion.box_if_dt64({arg_name})"
                else:
                    # Otherwise, the arguments can be used directly
                    return arg_name

            first_arg = unwrap_arg(arr0, "arg0")
            second_arg = unwrap_arg(arr1, "arg1")

        arg_names = ["arr0", "arr1"]
        arg_types = [arr0, arr1]
        propagate_null = [True] * 2

        # A dictionary of variable definitions shared between the kernels
        diff_defns = {
            "yr_diff": f"{second_arg}.year - {first_arg}.year",
            "qu_diff": f"{second_arg}.quarter - {first_arg}.quarter",
            "mo_diff": f"{second_arg}.month - {first_arg}.month",
            "y0, w0, _": f"{first_arg}.isocalendar()",
            "y1, w1, _": f"{second_arg}.isocalendar()",
            "iso_yr_diff": "bodosql.kernels.get_iso_weeks_between_years(y0, y1)",
            "wk_diff": "w1 - w0",
            "da_diff": f"(pd.Timestamp({second_arg}.year, {second_arg}.month, {second_arg}.day) - pd.Timestamp({first_arg}.year, {first_arg}.month, {first_arg}.day)).days",
            "ns_diff": f"{second_arg}.value - {first_arg}.value",
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

        # Load in all of the required definitions
        for req_defn in req_defns.get(unit, []):
            scalar_text += f"{req_defn} = {diff_defns[req_defn]}\n"

        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)
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
            if unit == "millisecond":
                divisor = 1000000
            if unit == "microsecond":
                divisor = 1000
            scalar_text += f"res[i] = np.floor_divide(({second_arg}.value), ({divisor})) - np.floor_divide(({first_arg}.value), ({divisor}))\n"

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
        ("millisecond", diff_millisecond, diff_millisecond_util),
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


def date_trunc(
    date_or_time_part, date_or_time_expr, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(date_trunc)
def overload_date_trunc(
    date_or_time_part, date_or_time_expr, dict_encoding_state=None, func_id=-1
):
    """
    Truncates a given Timestamp argument to the provided
    date_or_time_part. This corresponds to DATE_TRUNC inside snowflake

    Args:
        date_or_time_part (types.Type): A string scalar or array stating how to truncate
            the timestamp
        date_or_time_expr (types.Type): A bodo.types.Time object or bodo.types.Time array or tz-aware or tz-naive Timestamp or
            Timestamp array to be truncated.

    Returns:
        types.Type: The bodo.types.Time/timestamp after being truncated, which has same type as date_or_time_expr
    """
    args = [date_or_time_part, date_or_time_expr]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.date_trunc",
                [
                    "date_or_time_part",
                    "date_or_time_expr",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        date_or_time_part, date_or_time_expr, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return date_trunc_util(
            date_or_time_part, date_or_time_expr, dict_encoding_state, func_id
        )

    return impl


def date_trunc_util(
    date_or_time_part, date_or_time_expr, dict_encoding_state, func_id
):  # pragma: no cover
    pass


@overload(date_trunc_util)
def overload_date_trunc_util(
    date_or_time_part, date_or_time_expr, dict_encoding_state, func_id
):
    """
    Truncates a given bodo.types.Time/datetime.date/Timestamp argument to the provided
    date_or_time_part. This corresponds to DATE_TRUNC inside snowflake

    Args:
        date_or_time_part (types.Type): A string scalar or array stating how to truncate
            the bodo.types.Time/datetime.date/Timestamp.
        date_or_time_expr (types.Type): bodo.types.Time/datetime.date/Timestamp object or array to be truncated.

    Returns:
        types.Type: The bodo.types.Time/datetime.date/Timestamp after being truncated,
                    which has same type as date_or_time_expr.
    """
    verify_string_arg(date_or_time_part, "DATE_TRUNC", "date_or_time_part")
    arg_names = [
        "date_or_time_part",
        "date_or_time_expr",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [date_or_time_part, date_or_time_expr, dict_encoding_state, func_id]
    propagate_null = [True, True, False, False]
    use_dict_caching = not is_overload_none(dict_encoding_state)
    dict_encoding_state_name = "dict_encoding_state" if use_dict_caching else None
    func_id_name = "func_id" if use_dict_caching else None
    # Standardize the input to limit the condition in the loop
    scalar_text = "part_str = bodosql.kernels.datetime_array_kernels.standardize_snowflake_date_time_part(arg0)\n"
    if is_valid_time_arg(date_or_time_expr):  # truncate a bodo.types.Time object/array
        scalar_text += "if part_str in ('quarter', 'year', 'month', 'week', 'day'):\n"
        # date_or_time_part is too large, set everything to 0
        scalar_text += "    res[i] = bodo.types.Time()\n"
        scalar_text += "else:\n"
        scalar_text += "    if part_str == 'hour':\n"
        scalar_text += "        res[i] = bodo.types.Time(arg1.hour)\n"
        scalar_text += "    elif part_str == 'minute':\n"
        scalar_text += "        res[i] = bodo.types.Time(arg1.hour, arg1.minute)\n"
        scalar_text += "    elif part_str == 'second':\n"
        scalar_text += (
            "        res[i] = bodo.types.Time(arg1.hour, arg1.minute, arg1.second)\n"
        )
        scalar_text += "    elif part_str == 'millisecond':\n"
        scalar_text += (
            "        res[i] = bodo.types.Time(arg1.hour, arg1.minute, arg1.second, "
            "arg1.millisecond)\n"
        )
        scalar_text += "    elif part_str == 'microsecond':\n"
        scalar_text += (
            "        res[i] = bodo.types.Time(arg1.hour, arg1.minute, arg1.second, "
            "arg1.millisecond, arg1.microsecond)\n"
        )
        scalar_text += "    elif part_str == 'nanosecond':\n"
        scalar_text += (
            "        res[i] = bodo.types.Time(arg1.hour, arg1.minute, arg1.second, "
            "arg1.millisecond, arg1.microsecond, arg1.nanosecond)\n"
        )
        scalar_text += "    else:\n"
        scalar_text += "        raise ValueError('Invalid time part for DATE_TRUNC')\n"
        out_dtype = bodo.types.TimeArrayType(9)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            dict_encoding_state_name=dict_encoding_state_name,
            func_id_name=func_id_name,
        )
    elif is_valid_date_arg(date_or_time_expr):
        scalar_text += "if part_str == 'year':\n"
        scalar_text += "    res[i] = datetime.date(arg1.year, 1, 1)\n"
        scalar_text += "elif part_str == 'quarter':\n"
        scalar_text += "    month = arg1.month - (arg1.month - 1) % 3\n"
        scalar_text += "    res[i] = datetime.date(arg1.year, month, 1)\n"
        scalar_text += "elif part_str == 'month':\n"
        scalar_text += "    res[i] = datetime.date(arg1.year, arg1.month, 1)\n"
        scalar_text += "elif part_str == 'week':\n"
        scalar_text += "    res[i] = arg1 - datetime.timedelta(days=arg1.weekday())\n"
        scalar_text += "else:\n"  # when time unit is smaller than or equal to day, return the same date
        scalar_text += "    res[i] = arg1\n"
        out_dtype = DatetimeDateArrayType()
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            dict_encoding_state_name=dict_encoding_state_name,
            func_id_name=func_id_name,
        )
    else:  # Truncate a timestamp object/array
        verify_datetime_arg_allow_tz(
            date_or_time_expr,
            "DATE_TRUNC",
            "date_or_time_expr",
            allow_timestamp_tz=True,
        )
        is_timestamp_tz = is_valid_timestamptz_arg(date_or_time_expr)

        tz_literal = get_tz_if_exists(date_or_time_expr)
        # We perform computation on Timestamp types.
        box_str = (
            "bodo.utils.conversion.box_if_dt64"
            if not is_timestamp_tz
            and bodo.utils.utils.is_array_typ(date_or_time_expr, True)
            and tz_literal is None
            else ""
        )
        # When returning a scalar we return a pd.Timestamp type.
        unbox_str = (
            "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
            if bodo.utils.utils.is_array_typ(date_or_time_expr, True)
            else ""
        )

        if is_timestamp_tz:
            # For TIMESTAMPTZ, DATE_TRUNC operates on the local representation
            # of the timestamp and wraps the output in a TIMESTAMPTZ with the
            # same offset as the input.
            scalar_text += (
                "in_val = bodo.hiframes.timestamptz_ext.get_local_timestamp(arg1)\n"
            )
            out_dtype = bodo.types.timestamptz_array_type
        elif tz_literal is None:
            scalar_text += f"in_val = {box_str}(arg1)\n"
            out_dtype = types.Array(bodo.types.datetime64ns, 1, "C")
        else:
            scalar_text += "in_val = arg1\n"
            out_dtype = bodo.types.DatetimeArrayType(tz_literal)

        scalar_text += "if part_str == 'quarter':\n"
        scalar_text += "    out_val = pd.Timestamp(year=in_val.year, month= (3*(in_val.quarter - 1)) + 1, day=1, tz=tz_literal)\n"
        scalar_text += "elif part_str == 'year':\n"
        scalar_text += "    out_val = pd.Timestamp(year=in_val.year, month=1, day=1, tz=tz_literal)\n"
        scalar_text += "elif part_str == 'month':\n"
        scalar_text += "    out_val = pd.Timestamp(year=in_val.year, month=in_val.month, day=1, tz=tz_literal)\n"
        scalar_text += "elif part_str == 'day':\n"
        scalar_text += "    out_val = in_val.normalize()\n"
        scalar_text += "elif part_str == 'week':\n"
        # If we are already at the start of the week just normalize.
        scalar_text += "    if in_val.dayofweek == 0:\n"
        scalar_text += "        out_val = in_val.normalize()\n"
        scalar_text += "    else:\n"
        scalar_text += "        out_val = in_val.normalize() - pd.tseries.offsets.Week(n=1, weekday=0)\n"
        scalar_text += "elif part_str == 'hour':\n"
        scalar_text += "    out_val = in_val.floor('h')\n"
        scalar_text += "elif part_str == 'minute':\n"
        scalar_text += "    out_val = in_val.floor('min')\n"
        scalar_text += "elif part_str == 'second':\n"
        scalar_text += "    out_val = in_val.floor('s')\n"
        scalar_text += "elif part_str == 'millisecond':\n"
        scalar_text += "    out_val = in_val.floor('ms')\n"
        scalar_text += "elif part_str == 'microsecond':\n"
        scalar_text += "    out_val = in_val.floor('us')\n"
        scalar_text += "elif part_str == 'nanosecond':\n"
        scalar_text += "    out_val = in_val\n"
        scalar_text += "else:\n"
        # TODO: Include part_str when non-constant exception strings are supported.
        scalar_text += (
            "    raise ValueError('Invalid date or time part for DATE_TRUNC')\n"
        )
        if is_timestamp_tz:
            # Wrap the result in TIMESTAMPTZ preserving the offset
            scalar_text += "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz_from_local(out_val, arg1.offset_minutes)\n"
        elif tz_literal is None:
            # In the tz-naive array case we have to convert the Timestamp to dt64
            scalar_text += f"res[i] = {unbox_str}(out_val)\n"
        else:
            scalar_text += "res[i] = out_val\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals={"tz_literal": tz_literal},
        dict_encoding_state_name=dict_encoding_state_name,
        func_id_name=func_id_name,
    )


def generate_ts_construction_string():
    """
    Returns the scalar text used to generate a timestamp using the year, month, day,
    hour, minute, second, and nanoseconds provided. Handle the overflow from months
    into years directly, then handle all remaining unit arguments by constructing a
    timedelta (which will take care of the overflow of the remaining units) and adding
    it to the year/month value.
    """
    text = "months, month_overflow = 1 + ((arg1 - 1) % 12), (arg1 - 1) // 12\n"
    text += "ts = pd.Timestamp(year=arg0+month_overflow, month=months, day=1)\n"
    text += "ts = ts + pd.Timedelta(days=arg2-1, hours=arg3, minutes=arg4, seconds=arg5) + pd.Timedelta(arg6)\n"
    return text


@numba.generated_jit(nopython=True, no_unliteral=True)
def construct_timestamp_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):
    """A dedicated kernel for building a timestamp from the components

    Args:
        year (int array/series/scalar): the year of the new timestamp
        month (int array/series/scalar): the month of the new timestamp
        day (int array/series/scalar): the day of the new timestamp
        hour (int array/series/scalar): the hour of the new timestamp
        minute (int array/series/scalar): the minute of the new timestamp
        second (int array/series/scalar): the second of the new timestamp
        nanosecond (int array/series/scalar): the nanosecond of the new timestamp
        time_zone (optional string literal): the time zone of the new timestamp

    Returns:
        timestamp array/scalar: the timestamp with the components specified above,
        with the timezone as specified.

    Note: this function allows cases where the arguments are outside of the "normal"
    ranges, for example:

    construct_timestamp_util(2000, 0, 100, 0, 0, 0, 0, None) returns the 100th day
    of the year 2000.

    construct_timestamp_util(2015, 7, 4, 12, 150, 0, 0, None) returns 2:30 pm on
    July 4th, 2015.
    """

    verify_int_arg(year, "construct_timestamp", "year")
    verify_int_arg(month, "construct_timestamp", "month")
    verify_int_arg(day, "construct_timestamp", "day")
    verify_int_arg(hour, "construct_timestamp", "hour")
    verify_int_arg(minute, "construct_timestamp", "minute")
    verify_int_arg(second, "construct_timestamp", "second")
    verify_int_arg(nanosecond, "construct_timestamp", "nanosecond")

    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]
    arg_types = [year, month, day, hour, minute, second, nanosecond, time_zone]
    propagate_null = [True] * 7 + [False]

    args = [year, month, day, hour, minute, second, nanosecond, time_zone]
    if any(arg == bodo.types.none for arg in args):
        tz = None
        localize_str = ""
    elif is_overload_constant_str(time_zone):
        tz = get_overload_const_str(time_zone)
        # Need to use localize after calculating the regular timestamp
        # since daylight savings jumps are not meant to be considered when
        # constructing the timestamp. E.g. year=2000, hour= 24*50 hours should
        # output April 10 at midnight, instead of 1am.
        localize_str = f".tz_localize('{tz}')"
    else:
        raise_bodo_error(
            "construct_timestamp: time_zone argument must be a scalar string literal or None"
        )

    # When returning a scalar we return a pd.Timestamp type.
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if tz is None
        and (
            bodo.utils.utils.is_array_typ(year, True)
            or bodo.utils.utils.is_array_typ(month, True)
            or bodo.utils.utils.is_array_typ(day, True)
            or bodo.utils.utils.is_array_typ(hour, True)
            or bodo.utils.utils.is_array_typ(minute, True)
            or bodo.utils.utils.is_array_typ(second, True)
            or bodo.utils.utils.is_array_typ(nanosecond, True)
        )
        else ""
    )

    scalar_text = generate_ts_construction_string()
    scalar_text += f"res[i] = {unbox_str}(ts{localize_str})"

    if tz is None:
        out_dtype = types.Array(bodo.types.datetime64ns, 1, "C")
    else:
        out_dtype = bodo.types.DatetimeArrayType(tz)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def timestamp_tz_from_parts_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):  # pragma: no cover
    pass


@overload(timestamp_tz_from_parts_util, no_unliteral=True)
def overload_timestamp_tz_from_parts_util(
    year, month, day, hour, minute, second, nanosecond, time_zone
):
    """A dedicated kernel for building a timestamp_tz from the components

    Args:
        year (int array/series/scalar): the year of the new timestamp
        month (int array/series/scalar): the month of the new timestamp
        day (int array/series/scalar): the day of the new timestamp
        hour (int array/series/scalar): the hour of the new timestamp
        minute (int array/series/scalar): the minute of the new timestamp
        second (int array/series/scalar): the second of the new timestamp
        nanosecond (int array/series/scalar): the nanosecond of the new timestamp
        time_zone (string literal): the time zone used to create the offset of the new timestamp

    Returns:
        timestamp_tz array/scalar: the timestamp_tz with the components specified above,
        with the offset based on the timezone as specified.
    """

    verify_int_arg(year, "construct_timestamp", "year")
    verify_int_arg(month, "construct_timestamp", "month")
    verify_int_arg(day, "construct_timestamp", "day")
    verify_int_arg(hour, "construct_timestamp", "hour")
    verify_int_arg(minute, "construct_timestamp", "minute")
    verify_int_arg(second, "construct_timestamp", "second")
    verify_int_arg(nanosecond, "construct_timestamp", "nanosecond")

    arg_names = [
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "nanosecond",
        "time_zone",
    ]
    arg_types = [year, month, day, hour, minute, second, nanosecond, time_zone]
    propagate_null = [True] * 7 + [False]

    if not is_overload_constant_str(time_zone):
        raise_bodo_error(
            "TIMESTAMP_TZ_FROM_PARTS: time_zone argument must be a constant string literal"
        )
    tz = get_overload_const_str(time_zone)

    scalar_text = generate_ts_construction_string()
    # Get the UTC offset of the timestamp in the timezone, then convert from
    # nanoseconds to minutes.
    scalar_text += (
        f"offset = ts.tz_localize('{tz}').utcoffset().value // 60_000_000_000\n"
    )
    scalar_text += (
        "res[i] = bodo.hiframes.timestamptz_ext.init_timestamptz_from_local(ts, offset)"
    )

    out_dtype = bodo.types.timestamptz_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def date_from_parts_util(year, month, day):
    """A dedicated kernel for building a date from the components

    Args:
        year (int array/series/scalar): the year of the new date
        month (int array/series/scalar): the month of the new date
        day (int array/series/scalar): the day of the new date

    Returns:
        datetime.date array/scalar: the date with the components specified above.

    Note: this function allows cases where the arguments are outside of the "normal"
    ranges, for example:

    date_from_parts_util(2000, 0, 100) returns 2000-03-09

    date_from_parts_util(2004, -1, -1) returns 2003-10-30
    """

    verify_int_arg(year, "date_from_parts", "year")
    verify_int_arg(month, "date_from_parts", "month")
    verify_int_arg(day, "date_from_parts", "day")

    arg_names = [
        "year",
        "month",
        "day",
    ]
    arg_types = [year, month, day]
    propagate_null = [True] * 3

    # Handle the overflow from months into years directly, then handle all remaining
    # unit arguments by constructing a timedelta (which will take care of the
    # overflow of the remaining units) and adding it to the year/month value
    scalar_text = "months, month_overflow = 1 + ((arg1 - 1) % 12), (arg1 - 1) // 12\n"
    scalar_text += "date = datetime.date(arg0+month_overflow, months, 1)\n"
    scalar_text += "date = date + datetime.timedelta(days=arg2-1)\n"
    scalar_text += "res[i] = date"

    out_dtype = DatetimeDateArrayType()

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def timestamp_from_date_and_time(date_expr, time_expr):  # pragma: no cover
    args = [date_expr, time_expr]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.timestamp_from_date_and_time",
                ["date_expr", "time_expr"],
                i,
            )

    def impl(date_expr, time_expr):  # pragma: no cover
        return timestamp_from_date_and_time_util(date_expr, time_expr)

    return impl


@numba.generated_jit(nopython=True)
def timestamp_from_date_and_time_util(date_expr, time_expr):
    """A dedicated kernel for building a timestamp from a date and time

    Args:
        date_expr (types.Type): tz-naive Timestamp or Timestamp array
        time_expr (types.Type): bodo.types.Time or bodo.types.Time array

    Returns:
        timestamp array/scalar: a tz-naive timestamp with the date and time as
        specified above.
    """
    arg_names = ["date_expr", "time_expr"]
    arg_types = [date_expr, time_expr]
    propagate_null = [True, True]
    if is_valid_date_arg(time_expr):
        raise Exception("Expected time expression for second argument")

    scalar_text = ""
    # The input timezone doesn't matter for this function, we will always treat
    # all timestamps as having no timezone info
    if not is_valid_time_arg(time_expr):
        scalar_text += "arg1 = pd.Timestamp(arg1)\n"
    ts = "pd.Timestamp(year=arg0.year, month=arg0.month, day=arg0.day"
    ts += ", hour=arg1.hour, minute=arg1.minute, second=arg1.second"
    if is_valid_time_arg(time_expr):
        ts += ", microsecond=arg1.millisecond * 1000 + arg1.microsecond"
    else:
        # timestamps don't have milliseconds as a property - it's covered by
        # microsecond + nanosecond
        ts += ", microsecond=arg1.microsecond"
    ts += ", nanosecond=arg1.nanosecond)"
    scalar_text += f"res[i] = {ts}\n"

    out_dtype = bodo.types.DatetimeArrayType(None)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
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
    verify_datetime_arg_allow_tz(arr, "dayname", "arr", allow_timestamp_tz=True)
    unwrap_str = get_timestamp_unwrapping_str(arr)
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    if is_valid_date_arg(arr):
        scalar_text = "val = day_of_week_dict_arr[arg0.weekday()]\n"
    else:
        scalar_text = f"val = {unwrap_str}(arg0).day_name()\n"
    scalar_text += "res[i] = val[:3]\n"

    out_dtype = bodo.types.string_array_type

    # If the input is an array, make the output dictionary encoded
    synthesize_dict_if_vector = ["V"]
    dows = pd.array(
        [
            "Mon",
            "Tue",
            "Wed",
            "Thu",
            "Fri",
            "Sat",
            "Sun",
        ],
    )
    extra_globals = {"day_of_week_dict_arr": dows}

    synthesize_dict_setup_text = "dict_res = day_of_week_dict_arr"
    if is_valid_date_arg(arr):
        synthesize_dict_scalar_text = "res[i] = arg0.weekday()"
    else:
        synthesize_dict_scalar_text = f"res[i] = {unwrap_str}(arg0).dayofweek"

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
    scalar_text = (
        "ord = bodo.hiframes.datetime_date_ext._days_before_year(arg0) + arg1\n"
    )
    scalar_text += "y, m, d = bodo.hiframes.datetime_date_ext._ord2ymd(ord)\n"
    scalar_text += "res[i] = datetime.date(y, m, d)"

    out_dtype = DatetimeDateArrayType()

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
    verify_datetime_arg_allow_tz(arr, "monthname", "arr", allow_timestamp_tz=True)
    unwrap_str = get_timestamp_unwrapping_str(arr)
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    if is_valid_date_arg(arr):
        scalar_text = (
            "mons = ('January', 'February', 'March', 'April', 'May', 'June', "
            "'July', 'August', 'September', 'October', 'November', 'December')\n"
        )
        scalar_text += "val = mons[arg0.month - 1]\n"
    else:
        scalar_text = f"val = {unwrap_str}(arg0).month_name()\n"
    scalar_text += "res[i] = val[:3]\n"
    out_dtype = bodo.types.string_array_type

    # If the input is an array or date object, make the output dictionary encoded
    synthesize_dict_if_vector = ["V"]
    mons = pd.array(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    extra_globals = {"month_names_dict_arr": mons}
    synthesize_dict_setup_text = "dict_res = month_names_dict_arr"
    synthesize_dict_scalar_text = f"res[i] = {unwrap_str}(arg0).month - 1"

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

    verify_datetime_arg_allow_tz(arr0, "NEXT_DAY", "arr0", allow_timestamp_tz=True)
    verify_string_arg(arr1, "NEXT_DAY", "arr1")
    is_timestamp_tz = is_valid_timestamptz_arg(arr0)
    is_input_tz_aware = is_valid_tz_aware_datetime_arg(arr0)
    is_date = is_valid_date_arg(arr0)

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    # TODO: lower the dictionary as a global rather that defined in the function text
    prefix_code = (
        "dow_map = {'mo': 0, 'tu': 1, 'we': 2, 'th': 3, 'fr': 4, 'sa': 5, 'su': 6}"
    )
    # Note: Snowflake removes leading whitespace and ignore any characters aside from the first two
    # values, case insensitive. https://docs.snowflake.com/en/sql-reference/functions/next_day.html#arguments
    scalar_text = "arg1_trimmed = arg1.lstrip()[:2].lower()\n"
    if is_timestamp_tz:
        arg0_timestamp = "bodo.hiframes.timestamptz_ext.get_local_timestamp(arg0)"
    elif is_input_tz_aware:
        arg0_timestamp = "arg0"
    elif is_date:
        arg0_timestamp = "pd.Timestamp(year=arg0.year, month=arg0.month, day=arg0.day)"
    else:
        arg0_timestamp = "bodo.utils.conversion.box_if_dt64(arg0)"
    scalar_text += f"new_timestamp = {arg0_timestamp}.normalize() + pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed])\n"
    scalar_text += "res[i] = new_timestamp.date()\n"

    out_dtype = DatetimeDateArrayType()

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

    verify_datetime_arg_allow_tz(arr0, "PREVIOUS_DAY", "arr0", allow_timestamp_tz=True)
    verify_string_arg(arr1, "PREVIOUS_DAY", "arr1")
    is_timestamp_tz = is_valid_timestamptz_arg(arr0)
    is_input_tz_aware = is_valid_tz_aware_datetime_arg(arr0)
    is_date = is_valid_date_arg(arr0)

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    # TODO: lower the dictionary as a global rather that defined in the function text
    prefix_code = (
        "dow_map = {'mo': 0, 'tu': 1, 'we': 2, 'th': 3, 'fr': 4, 'sa': 5, 'su': 6}"
    )
    # Note: Snowflake removes leading whitespace and ignore any characters aside from the first two
    # values, case insensitive. https://docs.snowflake.com/en/sql-reference/functions/previous_day.html#arguments
    scalar_text = "arg1_trimmed = arg1.lstrip()[:2].lower()\n"
    if is_timestamp_tz:
        arg0_timestamp = "bodo.hiframes.timestamptz_ext.get_local_timestamp(arg0)"
    elif is_input_tz_aware:
        arg0_timestamp = "arg0"
    elif is_date:
        arg0_timestamp = "pd.Timestamp(year=arg0.year, month=arg0.month, day=arg0.day)"
    else:
        arg0_timestamp = "bodo.utils.conversion.box_if_dt64(arg0)"
    scalar_text += f"new_timestamp = {arg0_timestamp}.normalize() - pd.tseries.offsets.Week(weekday=dow_map[arg1_trimmed])\n"
    scalar_text += "res[i] = new_timestamp.date()\n"

    out_dtype = DatetimeDateArrayType()

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
            "bodosql.kernels.datetime_array_kernels.to_days_util", ["arr"], 0
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
    out_dtype = bodo.types.IntegerArrayType(types.int64)
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
            "bodosql.kernels.datetime_array_kernels.from_days_util", ["arr"], 0
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
    out_dtype = DatetimeDateArrayType()

    # Value to subtract to days to get to unix time
    prefix_code = "unix_days_to_year_zero = 719528\n"
    # multiplier to convert days -> nanoseconds
    prefix_code += "nanoseconds_divisor = 86400000000000\n"
    scalar_text = (
        "  nanoseconds = (arg0 - unix_days_to_year_zero) * nanoseconds_divisor\n"
    )

    scalar_text += "  res[i] = pd.Timestamp(nanoseconds).date()\n"
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
            "bodosql.kernels.datetime_array_kernels.to_seconds_util", ["arr"], 0
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
    out_dtype = bodo.types.IntegerArrayType(types.int64)
    is_input_arr = bodo.utils.utils.is_array_typ(arr, False)
    if is_input_arr and not timezone:
        # Note if the input is an array then we just operate directly on datetime64
        # to avoid Timestamp boxing.
        scalar_text = (
            "  in_value = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arg0)\n"
        )
    else:
        scalar_text = "  in_value = arg0.value\n"
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
            "bodosql.kernels.tz_aware_interval_add",
            ["tz_arg", "interval_arg"],
            0,
        )
    if isinstance(interval_arg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.tz_aware_interval_add",
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
        out_dtype = bodo.types.DatetimeArrayType(timezone)
    else:
        # Handle a default case if the timezone value is NA.
        # Note this doesn't matter because we will output NA.
        out_dtype = bodo.types.datetime64ns
    # Note: We don't have support for TZAware + pd.DateOffset yet.
    # As a result we must compute a Timedelta from the DateOffset instead.
    if interval_arg == bodo.types.date_offset_type:
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


def interval_multiply(interval_arg, integer_arg):  # pragma: no cover
    pass


@overload(interval_multiply)
def overload_interval_multiply(interval_arg, integer_arg):
    """BodoSQL kernel for multiplying an interval by an integer.
    Typically intervals are scalars, so in most cases this kernel
    will just handle scalars. However, we do allow certain intervals
    to be array, so we can have array outputs. The integer argument
    should only be an array if the interval can be represented as
    an array.

    Args:
        interval_arg (types.Type): Either a DateOffset or Timedelta scalar or array
        integer_arg (types.Type): An integer scalar (or in rare cases array)

    Returns:
        types.Type: Interval scalar or array returned after multiplying the results.
    """
    for i, arg in enumerate((interval_arg, integer_arg)):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.interval_multiply",
                ["interval_arg", "integer_arg"],
                i,
            )

    def impl(interval_arg, integer_arg):  # pragma: no cover
        return interval_multiply_util(interval_arg, integer_arg)

    return impl


def interval_multiply_util(interval_arg, integer_arg):  # pragma: no cover
    pass


@overload(interval_multiply_util)
def overload_interval_multiply_util(interval_arg, integer_arg):
    """BodoSQL kernel for multiplying an interval by an integer.
    Typically intervals are scalars, so in most cases this kernel
    will just handle scalars. However, we do allow certain intervals
    to be array, so we can have array outputs. The integer argument
    should only be an array if the interval can be represented as
    an array.

    Args:
        interval_arg (types.Type): Either a DateOffset or Timedelta scalar or array
        integer_arg (types.Type): An integer scalar (or in rare cases array)

    Returns:
        types.Type: Interval scalar or array returned after multiplying the results.
    """
    verify_sql_interval(interval_arg, "INTERVAL_MULTIPLY", "interval_arg")
    verify_int_arg(integer_arg, "INTERVAL_MULTIPLY", "integer_arg")
    arg_names = ["interval_arg", "integer_arg"]
    arg_types = [interval_arg, integer_arg]
    propagate_null = [True, True]

    is_interval_arr = bodo.utils.utils.is_array_typ(interval_arg, True)
    is_integer_arr = bodo.utils.utils.is_array_typ(integer_arg, True)

    if interval_arg == bodo.types.date_offset_type:
        if is_integer_arr:
            raise BodoError(
                "interval_multiply(): Integer array cannot be provided if multiplying a date offset."
            )
        out_dtype = bodo.types.date_offset_type
        # all year to month intervals are based on month
        scalar_text = "res[i] = pd.DateOffset(months=arg0._months * arg1)\n"
    else:
        out_dtype = types.Array(bodo.types.timedelta64ns, 1, "C")

        unbox_str = (
            "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
            if is_interval_arr or is_integer_arr
            else ""
        )
        box_str = "bodo.utils.conversion.box_if_dt64" if is_interval_arr else ""

        scalar_text = f"res[i] = {unbox_str}({box_str}(arg0) * arg1)\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def interval_add_interval(arr0, arr1):  # pragma: no cover
    pass


@overload(interval_add_interval)
def overload_interval_add_interval(arr0, arr1):
    """BodoSQL kernel for adding two intervals together. While this should
    typically just be between two scalars the array case is technically possible.
    Note: We do not support adding a DateOffset with a Timedelta.

    Args:
        arr0 (types.Type): Either a DateOffset or Timedelta scalar or array
        arr1 (types.Type): Either a DateOffset or Timedelta scalar or array

    Returns:
        types.Type: Interval scalar or array returned after adding the results.
    """
    for i, arg in enumerate((arr0, arr1)):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.interval_add_interval",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return interval_add_interval_util(arr0, arr1)

    return impl


def interval_add_interval_util(arr0, arr1):  # pragma: no cover
    pass


@overload(interval_add_interval_util)
def overload_interval_add_interval_util(arr0, arr1):
    """BodoSQL kernel for adding two intervals together. While this should
    typically just be between two scalars the array case is technically possible.
    Note: We do not support adding a DateOffset with a Timedelta.

    Args:
        arr0 (types.Type): Either a DateOffset or Timedelta scalar or array
        arr1 (types.Type): Either a DateOffset or Timedelta scalar or array

    Returns:
        types.Type: Interval scalar or array returned after adding the results.
    """
    if not isinstance(arr0, CombinedIntervalType):
        verify_td_arg(arr0, "INTERVAL_ADD_INTERVAL", "arr0")
    if not isinstance(arr1, CombinedIntervalType):
        verify_td_arg(arr1, "INTERVAL_ADD_INTERVAL", "arr1")
    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True, True]

    out_dtype = types.Array(bodo.types.timedelta64ns, 1, "C")
    box_str0 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr0, True)
        else ""
    )
    box_str1 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr1, True)
        else ""
    )
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(arr0, True)
        or bodo.utils.utils.is_array_typ(arr1, True)
        else ""
    )

    scalar_text = f"res[i] = {unbox_str}({box_str0}(arg0) + {box_str1}(arg1))\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def create_timestamp(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


@overload(create_timestamp)
def overload_create_timestamp(arr, dict_encoding_state=None, func_id=-1):
    """BodoSQL array kernel to create a Timestamp. This function will accept
    anything accepted by the `pd.Timestamp()` constructor
    now, so we don't type check.

    Args:
        arr (types.Types): scalar or array of input values

    Returns:
        types.Type: scalar or array of Timestamp values.
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.datetime_array_kernels.create_timestamp_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return create_timestamp_util(arr, dict_encoding_state, func_id)

    return impl


def create_timestamp_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


@overload(create_timestamp_util)
def overload_create_timestamp_util(
    arr, dict_encoding_state, func_id
):  # pragma: no cover
    """BodoSQL array kernel to create a Timestamp. This function will accept
    anything accepted by the `pd.Timestamp()` constructor
    now, so we don't type check.

    Args:
        arr (types.Types): scalar or array of input values

    Returns:
        types.Type: scalar or array of Timestamp values.
    """
    arg_names = ["arr", "dict_encoding_state", "func_id"]
    arg_types = [arr, dict_encoding_state, func_id]
    propagate_null = [True, False, False]
    out_dtype = types.Array(bodo.types.datetime64ns, 1, "C")
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp"
        if bodo.utils.utils.is_array_typ(arr, True)
        else ""
    )

    scalar_text = f"res[i] = {unbox_str}(pd.Timestamp(arg0))\n"
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
def date_format(arr0, arr1):  # pragma: no cover
    for i, arg in enumerate((arr0, arr1)):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.date_format",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return date_format_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def date_format_util(arr0, arr1):
    """
    Python kernel for DATE_FORMAT function

    Args:
        arr0: A column/scalar of date/timestamp object
        arr1: The format string

    Returns:
        A scalar or array of strings that includes the information of the date/timestamp
        in the format of format string
    """
    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True, True]
    out_dtype = bodo.types.string_array_type

    box_str = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(arr0, True)
        else ""
    )

    verify_datetime_arg_allow_tz(arr0, "DATE_FORMAT", "arr0")
    scalar_text = f"res[i] = {box_str}(arg0).strftime(arg1)"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def add_date_interval_to_date(start_dt, interval):
    """Handles cases where adding intervals receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [start_dt, interval]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.add_date_interval_to_date",
                ["start_dt", "interval"],
                i,
            )

    if isinstance(interval, CombinedIntervalType):

        def impl(start_dt, interval):  # pragma: no cover
            result = start_dt
            for interval_part in interval.intervals:
                result = add_date_interval_to_date_util(result, interval_part)
            return result

    else:

        def impl(start_dt, interval):  # pragma: no cover
            return add_date_interval_to_date_util(start_dt, interval)

    return impl


@numba.generated_jit(nopython=True)
def add_date_interval_to_date_util(start_dt, interval):
    """A dedicated kernel adding a timedelta with date unit to a datetime.date object
    Args:
        start_dt (datetime array/series/scalar): the datetime.date objects that are being
        added to
        interval (timedelta array/series/scalar): the offset being added to start_dt
    Returns:
        datetime series/scalar: start_dt + interval
    """
    verify_date_arg(start_dt, "add_date_interval_to_date", "start_dt")
    verify_sql_interval(interval, "add_date_interval_to_date", "interval")

    arg_names = ["start_dt", "interval"]
    arg_types = [start_dt, interval]
    propagate_null = [True] * 2

    is_interval_vector = bodo.utils.utils.is_array_typ(interval, True)
    box_str = "bodo.utils.conversion.box_if_dt64" if is_interval_vector else ""
    scalar_text = f"dt = pd.Timestamp(arg0) + {box_str}(arg1)\n"
    scalar_text += "res[i] = dt.date()\n"
    out_dtype = DatetimeDateArrayType()

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def dayofweek(arr, week_start):
    args = [arr, week_start]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.dayofweek",
                ["arr", "week_start"],
                i,
            )

    def impl(arr, week_start):  # pragma: no cover
        return dayofweek_util(arr, week_start)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def dayofweek_util(arr, week_start):
    """
    Python kernel for DAYOFWEEK.

    Returns the day of the week, values ranging from 1-7.

    Takes as input <week_start>, which determines what day of the week
    the week starts at. Per Snowflake behavior, 0/1 correspond to Monday,
    7 corresponds to Sunday.

    Args:
        arr (datetime/timestamp scalar/series): the data
        week_start (Literal[int]) (0-7): day of the week to start with

    Returns:
        int (scalar/series): the days of the week of the data values
    """
    verify_datetime_arg_allow_tz(arr, "dayofweek", "arr")
    assert_bodo_error(
        is_overload_constant_int(week_start),
        "Invalid week_start parameter! Must be an integer",
    )

    week_start_val = get_overload_const_int(week_start)
    if week_start_val < 0 or week_start_val > 7:
        raise_bodo_error(
            "Invalid week_start parameter! Must be between 0 and 7 (0 and 1 both map to Monday)"
        )

    arg_names = ["arr", "week_start"]
    arg_types = [arr, week_start]
    propagate_null = [True] * 2
    tz = get_tz_if_exists(arr)

    box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""

    # pd.Timestamp weekday() and datetime dayofweek are 0-indexed
    # where 0 is Monday and 6 is Sunday.
    if is_valid_date_arg(arr):
        dayofweek_str = "arg0.weekday()"
    else:
        dayofweek_str = f"{box_str}(arg0).dayofweek"

    scalar_text = "start_day = max(0, arg1 - 1)\n"
    scalar_text += f"res[i] = ({dayofweek_str} - start_day + 1) % 7\n"

    out_dtype = bodo.types.IntegerArrayType(numba.int64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def get_epoch(arr, unit):  # pragma: no cover
    pass


def get_epoch_util(arr, unit):  # pragma: no cover
    pass


@overload(get_epoch, no_unliteral=True)
def overload_get_epoch(arr, unit):
    """
    Kernel that computes all of the EPOCH_XXX
    calculations for DATE_PART. Unit determines which
    function is being performed.

    Args:
        arr (datetime_like): An array or scalar of datetime like objects
            that are either tz_aware or naive.
        unit (string Literal): Must be one of 's', 'ms', 'us', 'ns' and
            this dictates the output precision.

    Returns:
        An int64 giving the time since the start of Unix time in the specified units.
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodosql.kernels.get_epoch", ["arr", "unit"], 0)

    def impl(arr, unit):  # pragma: no cover
        return get_epoch_util(arr, unit)

    return impl


@overload(get_epoch_util, no_unliteral=True)
def overload_get_epoch_util(arr, unit):
    """
    Kernel that computes all of the EPOCH_XXX
    calculations for DATE_PART. Unit determines which
    function is being performed.

    Args:
        arr (datetime_like): An array or scalar of datetime like objects
            that are either tz_aware or naive.
        unit (string Literal): Must be one of 's', 'ms', 'us', 'ns' and
            this dictates the output precision.

    Returns:
        An int64 giving the time since the start of Unix time in the specified units.
    """
    verify_datetime_arg_allow_tz(arr, "get_epoch", "arr", allow_timestamp_tz=True)
    if not is_overload_constant_str(unit):  # pragma: no cover
        raise_bodo_error("get_epoch(): unit must be a string literal")

    # Fetch the unit and ignore casing.
    unit = get_overload_const_str(unit).lower()
    if unit not in ("s", "ms", "us", "ns"):  # pragma: no cover
        raise BodoError(
            f"get_epoch(): unit must be one of 's', 'ms', 'us', 'ns'. Found: '{unit}'"
        )

    divisor_map = {"ns": 1, "us": 1000, "ms": 1000 * 1000, "s": 1000 * 1000 * 1000}
    divisor = divisor_map[unit]

    arg_names = ["arr", "unit"]
    arg_types = [arr, unit]
    propagate_null = [True, False] * 2
    unwrap_str = get_timestamp_unwrapping_str(arr, tz_to_utc=True)
    scalar_text = f"res[i] = {unwrap_str}(arg0).value // {divisor}\n"
    out_dtype = bodo.types.IntegerArrayType(numba.int64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


def get_timezone_offset(arr, unit):  # pragma: no cover
    pass


def get_timezone_offset_util(arr, unit):  # pragma: no cover
    pass


@overload(get_timezone_offset, no_unliteral=True)
def overload_get_timezone_offset(arr, unit):
    """
    Kernel that computes all of the TIMEZONE_XXX
    calculations for DATE_PART. Unit determines which
    function is being performed.

    Args:
        arr (datetime_like): An array or scalar of timestamp like objects
            that are either tz_aware or naive.
        unit (string Literal): Must be one of 'hr' or 'min'. This dictates
            the calculation before performed.

    Returns:
        An int64 giving the time since the start of Unix time in the specified units.
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodosql.kernels.get_timezone_offset", ["arr", "unit"], 0)

    def impl(arr, unit):  # pragma: no cover
        return get_timezone_offset_util(arr, unit)

    return impl


@overload(get_timezone_offset_util, no_unliteral=True)
def overload_get_timezone_offset_util(arr, unit):
    """
    Kernel that computes all of the TIMEZONE_XXX
    calculations for DATE_PART. Unit determines which
    function is being performed.

    The TIMEZONE_XXX, although not well documented in Snowflake,
    correspond to determining to the offset from UTC for each element.
    HOUR gives the hours component and minute gives the minutes component.
    NOTE:
        MINUTE != HOUR * 60,
        OFFSET_MINUTES = HOUR * 60 + MINUTE

    This depends on the timezone in question, which basically leads to 3
    possible cases:

    TIMEZONE NAIVE (e.g. TIMESTAMP_NTZ):
        Always output a constant 0.
    TIMEZONE AWARE with a constant offset/no transition times (e.g. UTC):
        Calculate the offset from UTC as a compile time constant and output
        the result.
    TIMEZONE AWARE with transition times (e.g. America/New_York):
        The offset must be computed at runtime as it depends on the time in question.

    Args:
        arr (datetime_like): An array or scalar of timestamp like objects
            that are either tz_aware or naive.
        unit (string Literal): Must be one of 'hr' or 'min'. This dictates
            the calculation before performed.

    Returns:
        An int64 giving the time since the start of Unix time in the specified units.
    """
    verify_timestamp_arg_allow_tz(arr, "get_timezone_offset", "arr")
    if not is_overload_constant_str(unit):  # pragma: no cover
        raise_bodo_error("get_timezone_offset(): unit must be a string literal")

    # Fetch the unit and ignore casing.
    unit = get_overload_const_str(unit).lower()
    if unit not in ("hr", "min"):  # pragma: no cover
        raise BodoError(
            f"get_timezone_offset(): unit must be one of 'hr', 'min'. Found: '{unit}'"
        )

    # The offsets we derive will be in nanoseconds.
    # This logic occurs for both the compile time and runtime paths.
    to_minutes_divisor = 1000 * 1000 * 1000 * 60
    modulo_map = {
        "min": 60,
        "hr": 24,
    }
    modulo = modulo_map[unit]
    scalar_text = ""
    extra_globals = None
    tz = get_tz_if_exists(arr)
    if is_valid_timestamptz_arg(arr):
        scalar_text += "sign = 1 if arg0.offset_minutes >= 0 else -1\n"
        if unit == "hr":
            # For tzh, extract the hour component of the offset
            # e.g. -08:15 -> -510 minuntes, so -510 would return -8
            scalar_text += "res[i] = sign * (abs(arg0.offset_minutes) // 60)"
        else:
            # For tzh, extract the minute component of the offset
            # e.g. -08:15 -> -510 minuntes, so -510 would return -15
            scalar_text += "res[i] = sign * (abs(arg0.offset_minutes) % 60)"
    elif tz is None:
        # No timezone is always 0.
        scalar_text += "res[i] = 0\n"
    else:
        tz_info = python_timezone_from_bodo_timezone_info(tz)
        if not bodo.hiframes.pd_timestamp_ext.tz_has_transition_times(tz):
            # If there are no transition times the offset is constant.
            # We compute this by calling utcoffset() on any time.
            ts = pd.Timestamp.now(tz=tz_info)
            datetime_offset = ts.utcoffset()
            # Convert to Pandas and extract nanoseconds.
            nanoseconds = pd.to_timedelta(datetime_offset).value
            # Truncate to minutes
            minute_offset = nanoseconds // to_minutes_divisor
            if unit == "hr":
                # Remove the minutes
                float_offset = minute_offset / 60
                if minute_offset < 0:
                    constant_offset = np.ceil(float_offset)
                else:
                    constant_offset = np.floor(float_offset)
            else:
                # Remove the hours
                constant_offset = minute_offset % 60
                # Python modulo converts - number to +
                if nanoseconds < 0:
                    constant_offset = constant_offset - modulo
            scalar_text += f"res[i] = {constant_offset}\n"
        else:
            # We have a timezone with transition times, so we must compute
            # them at runtime.
            box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""
            trans = np.array(tz_info._utc_transition_times, dtype="M8[ns]").view("i8")
            deltas = np.array(tz_info._transition_info)[:, 0]
            deltas = (
                (pd.Series(deltas).dt.total_seconds() * 1_000_000_000)
                .astype(np.int64)
                .values
            )
            extra_globals = {"trans": trans, "deltas": deltas}
            # Implementation
            scalar_text += f"arg0 = {box_str}(arg0)\n"
            scalar_text += "start_value = arg0.value\n"
            scalar_text += (
                "idx = np.searchsorted(trans, arg0.value, side='right') - 1\n"
            )
            scalar_text += "nanoseconds_offset = deltas[idx]\n"
            scalar_text += (
                f"minute_offset = nanoseconds_offset // {to_minutes_divisor}\n"
            )
            if unit == "hr":
                scalar_text += "float_offset = minute_offset / 60 \n"
                scalar_text += "if minute_offset < 0:\n"
                scalar_text += "  final_offset = np.ceil(float_offset)\n"
                scalar_text += "else:\n"
                scalar_text += "  final_offset = np.floor(float_offset)\n"
            else:
                # Remove the hours
                scalar_text += "final_offset = minute_offset % 60\n"
                # Python modulo converts - number to +
                scalar_text += "if nanoseconds_offset < 0:\n"
                scalar_text += f"  final_offset = final_offset - {modulo}\n"
            scalar_text += "res[i] = final_offset\n"

    arg_names = ["arr", "unit"]
    arg_types = [arr, unit]
    propagate_null = [True, False] * 2
    # Note: This might be smaller but snowflake bound it at a Number(9, 0), which is
    # an int32. Since they/we may need to support out dated transition times, we will
    # be conservative and match Snowflake.
    out_dtype = bodo.types.IntegerArrayType(numba.int32)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
    )


@numba.generated_jit(nopython=True)
def months_between(dt0, dt1):  # pragma: no cover
    args = [dt0, dt1]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.months_between",
                ["dt0", "dt1"],
                i,
            )

    def impl(dt0, dt1):  # pragma: no cover
        return months_between_util(dt0, dt1)

    return impl


@numba.generated_jit(nopython=True)
def months_between_util(dt0, dt1):
    """
    Python kernel for MONTHS_BETWEEN function.
    Returns the number of months between two DATE or TIMESTAMP values.

    Args:
        dt0: A column/scalar of date/timestamp object
        dt1: A column/scalar of date/timestamp object

    Returns:
        A scalar or array of floats that denotes the months
        between dt0 and dt1.
    """
    verify_time_or_datetime_arg_allow_tz(dt0, "months_between", "arg0")
    verify_time_or_datetime_arg_allow_tz(dt1, "months_between", "arg1")

    arg_names = ["dt0", "dt1"]
    arg_types = [dt0, dt1]
    propagate_null = [True] * 2

    box_str0 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(dt0, True)
        else ""
    )
    box_str1 = (
        "bodo.utils.conversion.box_if_dt64"
        if bodo.utils.utils.is_array_typ(dt1, True)
        else ""
    )

    scalar_text = f"arg0 = {box_str0}(arg0)\n"
    scalar_text += f"arg1 = {box_str1}(arg1)\n"
    scalar_text += "arg0_next = arg0 + datetime.timedelta(days=1)\n"
    scalar_text += "arg1_next = arg1 + datetime.timedelta(days=1)\n"
    scalar_text += (
        "months_int_count = (arg0.year - arg1.year) * 12 + (arg0.month - arg1.month)\n"
    )

    # Per Snowflake docs, if the two date's have the same day number or they are
    # both the last day of the month, the months between value is an integer value.
    scalar_text += (
        "if (arg0.day == arg1.day) or (arg0_next.day == 1 and arg1_next.day == 1):\n"
    )
    scalar_text += "  months_frac_count = 0\n"
    scalar_text += "else:\n"

    # Otherwise, there is a fractional component, which is the difference in the
    # day numbers / 31 (per Snowflake docs), rounded to 6 decimal places.
    scalar_text += "  months_frac_count = round((arg0.day - arg1.day)/31.0, 6)\n"
    scalar_text += "res[i] = months_int_count + months_frac_count\n"

    out_dtype = bodo.types.FloatingArrayType(bodo.types.float64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def week(arr, week_start, week_of_year_policy):
    # NOTE (allai5): this is an alias for weekofyear
    # We need to have a kernel named week as well
    # in order to do filter pushdown in Python.
    args = [arr, week_start, week_of_year_policy]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.datetime_array_kernels.weekofyear_util",
                ["arr", "week_start", "week_of_year_policy"],
                i,
            )

    def impl(arr, week_start, week_of_year_policy):  # pragma: no cover
        return weekofyear_util(arr, week_start, week_of_year_policy)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def weekofyear(arr, week_start, week_of_year_policy):
    args = [arr, week_start, week_of_year_policy]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.weekofyear",
                ["arr", "week_start", "week_of_year_policy"],
                i,
            )

    def impl(arr, week_start, week_of_year_policy):  # pragma: no cover
        return weekofyear_util(arr, week_start, week_of_year_policy)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def weekofyear_util(arr, week_start, week_of_year_policy):
    """
    Python kernel for WEEKOFYEAR.

    Returns week of the year. Accepts week_start and
    week_of_year policy parameters, which control what
    day of the week to start with and what the first week
    of the year is defined as. See Snowflake docs for more detail.

    NOTE (allai5): edge case behavior for WEEK_OF_YEAR_POLICY = 0
    and WEEK_START != 0 or 1 differs from Snowflake for
    dates around the last days of December and the first days of
    January.

    Args:
        arr (datetime/timestamp scalar/series): the data
        week_start (Literal[int]) (0-7): day of the week to start with
        week_of_year_policy (Literal[int]) (0-1): week of year policy flag

    Returns:
        datetime/timestamp scalar/series: week of year values
    """
    verify_datetime_arg_allow_tz(arr, "weekofyear", "arr", allow_timestamp_tz=True)

    assert_bodo_error(
        is_overload_constant_int(week_start),
        "Invalid week_start parameter! Must be an integer",
    )

    assert_bodo_error(
        is_overload_constant_int(week_of_year_policy),
        "Invalid week_of_year_policy parameter! Must be an integer",
    )

    week_start_val = get_overload_const_int(week_start)
    if week_start_val < 0 or week_start_val > 7:
        raise_bodo_error(
            "Invalid week_start parameter! Must be between 0 and 7 (0 and 1 both map to Monday)"
        )

    arg_names = ["arr", "week_start", "week_of_year_policy"]
    arg_types = [arr, week_start, week_of_year_policy]
    propagate_null = [True] * 3

    unwrap_str = get_timestamp_unwrapping_str(arr)

    date_to_int_str = "bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int_ns"
    if is_valid_date_arg(arr):
        scalar_text = (
            f"arg0 = pd.Timestamp({date_to_int_str}(arg0)).tz_localize(None)\n"
        )
    else:
        scalar_text = f"arg0 = {unwrap_str}(arg0).tz_localize(None)\n"

    scalar_text += "start_day = max(0, arg1 - 1)\n"

    if get_overload_const_int(week_of_year_policy) == 1:
        scalar_text += (
            "first_day_of_year = pd.Timestamp(year=arg0.year, month=1, day=1)\n"
        )
        scalar_text += "days = (start_day - first_day_of_year.weekday()) % 7\n"
        scalar_text += "first_start_day_of_year = first_day_of_year + datetime.timedelta(days=days)\n"
        scalar_text += "days_from_start = (arg0 - first_start_day_of_year).days\n"
        scalar_text += "week_number = (days_from_start // 7) + 1\n"
        scalar_text += "if days != 0:\n"
        scalar_text += "  week_number = week_number + 1\n"
        scalar_text += "res[i] = week_number\n"
    else:
        scalar_text += "start_day_offset = -1 * start_day if start_day != 6 else 1\n"
        scalar_text += (
            "offset_date = (arg0 + datetime.timedelta(days=start_day_offset))\n"
        )
        scalar_text += "res[i] = offset_date.isocalendar()[1]\n"

    out_dtype = bodo.types.IntegerArrayType(numba.int64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def yearofweek(arr, week_start, week_of_year_policy):
    args = [arr, week_start, week_of_year_policy]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.yearofweek",
                ["arr", "week_start", "week_of_year_policy"],
                i,
            )

    def impl(arr, week_start, week_of_year_policy):  # pragma: no cover
        return yearofweek_util(arr, week_start, week_of_year_policy)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def yearofweek_util(arr, week_start, week_of_year_policy):
    """
    Python kernel for YEAROFWEEK.

    Returns year that corresponds with WEEKOFYEAR. Accepts week_start and
    week_of_year policy parameters, which control what
    day of the week to start with and what the first week
    of the year is defined as. See Snowflake docs for more detail.

    NOTE (allai5): edge case behavior for WEEK_OF_YEAR_POLICY = 0
    and WEEK_START != 0 or 1 differs from Snowflake for
    dates around the last days of December and the first days of
    January.

    Args:
        arr (datetime/timestamp scalar/series): the data
        week_start (Literal[int]) (0-7): day of the week to start with
        week_of_year_policy (Literal[int]) (0-1): week of year policy flag

    Returns:
        datetime/timestamp scalar/series: year of week values
    """

    verify_datetime_arg_allow_tz(arr, "yearofweek", "arr")

    assert_bodo_error(
        is_overload_constant_int(week_start),
        "Invalid week_start parameter! Must be an integer",
    )

    assert_bodo_error(
        is_overload_constant_int(week_of_year_policy),
        "Invalid week_of_year_policy parameter! Must be an integer",
    )

    week_start_val = get_overload_const_int(week_start)
    if week_start_val < 0 or week_start_val > 7:
        raise_bodo_error(
            "Invalid week_start parameter! Must be between 0 and 7 (0 and 1 both map to Monday)"
        )

    arg_names = ["arr", "week_start", "week_of_year_policy"]
    arg_types = [arr, week_start, week_of_year_policy]
    propagate_null = [True] * 3
    tz = get_tz_if_exists(arr)

    box_str = "bodo.utils.conversion.box_if_dt64" if tz is None else ""

    date_to_int_str = "bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int_ns"
    if is_valid_date_arg(arr):
        scalar_text = (
            f"arg0 = pd.Timestamp({date_to_int_str}(arg0)).tz_localize(None)\n"
        )
    else:
        scalar_text = f"arg0 = {box_str}(arg0).tz_localize(None)\n"

    scalar_text += "start_day = max(0, arg1 - 1)\n"

    if get_overload_const_int(week_of_year_policy) == 1:
        scalar_text += (
            "first_day_of_year = pd.Timestamp(year=arg0.year, month=1, day=1)\n"
        )
        scalar_text += "days = (start_day - first_day_of_year.weekday()) % 7\n"
        scalar_text += "first_start_day_of_year = first_day_of_year + datetime.timedelta(days=days)\n"
        scalar_text += "days_from_start = (arg0 - first_start_day_of_year).days\n"
        scalar_text += (
            "year_of_week = arg0.year if days_from_start < 365 else arg0.year - 1\n"
        )
    else:
        scalar_text += "start_day_offset = -1 * start_day if start_day != 6 else 1\n"
        scalar_text += (
            "offset_date = (arg0 + datetime.timedelta(days=start_day_offset))\n"
        )
        scalar_text += "year_of_week = offset_date.isocalendar()[0]\n"
    scalar_text += "res[i] = year_of_week\n"

    out_dtype = bodo.types.IntegerArrayType(numba.int64)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def add_months(dt0, num_months):  # pragma: no cover
    args = [dt0, num_months]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.add_months",
                ["dt0", "num_months"],
                i,
            )

    def impl(dt0, num_months):  # pragma: no cover
        return add_months_util(dt0, num_months)

    return impl


@numba.generated_jit(nopython=True)
def add_months_util(dt0, num_months):
    """
    Python kernel for ADD_MONTHS function.
    Adds <num_months> months to dt0.

    Args:
        dt0: A column/scalar of date/timestamp object
        num_months: A column/scalar of numeric values.

    Returns:
        A column/scalar of date/timestamp object.
    """

    verify_time_or_datetime_arg_allow_tz(dt0, "add_months", "arg0")
    verify_int_float_arg(num_months, "add_months", "num_months")

    time_zone = get_tz_if_exists(dt0)

    arg_names = ["dt0", "num_months"]
    arg_types = [dt0, num_months]
    propagate_null = [True] * 2

    dt0_is_array = bodo.utils.utils.is_array_typ(dt0, True)
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp" if dt0_is_array else ""
    )

    box_str = "bodo.utils.conversion.box_if_dt64" if dt0_is_array else ""

    # Box dt64/date to Timestamp
    date_to_int_str = "bodo.hiframes.datetime_date_ext.cast_datetime_date_to_int_ns"
    if is_valid_date_arg(dt0):
        scalar_text = f"arg0 = pd.Timestamp({date_to_int_str}(arg0))\n"
    else:
        scalar_text = f"arg0 = {box_str}(arg0)\n"

    # Per Snowflake docs, the num_months argument can be any numeric value,
    # so we have to round the value in the beginning to get an integer.
    scalar_text += "arg1 = round(arg1)\n"

    # # If the input date is the last day of the month,
    # # the output date also must be the last day of the month
    scalar_text += "if (arg0.is_month_end and not (arg0 + pd.DateOffset(months=arg1)).is_month_end):\n"
    scalar_text += "  new_arg = arg0 + pd.DateOffset(months=arg1) + pd.tseries.offsets.MonthEnd()\n"
    scalar_text += "else:\n"
    scalar_text += "  new_arg = arg0 + pd.DateOffset(months=arg1)\n"

    if time_zone is not None:
        out_dtype = bodo.types.DatetimeArrayType(time_zone)
        scalar_text += "res[i] = new_arg\n"
    else:
        if is_valid_date_arg(dt0):
            out_dtype = bodo.types.datetime_date_array_type
            scalar_text += f"res[i] = {unbox_str}(new_arg.date())\n"
        else:
            out_dtype = types.Array(bodo.types.datetime64ns, 1, "C")
            scalar_text += f"res[i] = {unbox_str}(new_arg)\n"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def time_slice(
    arr, slice_length, date_time_part, start_or_end, start_day
):  # pragma: no cover
    args = [arr, slice_length, date_time_part, start_or_end, start_day]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.time_slice",
                ["arr", "slice_length", "date_time_part", "start_or_end", "start_day"],
                i,
            )

    def impl(
        arr, slice_length, date_time_part, start_or_end, start_day
    ):  # pragma: no cover
        return time_slice_util(
            arr, slice_length, date_time_part, start_or_end, start_day
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def time_slice_util(arr, slice_length, date_time_part, start_or_end, start_day):
    """
    Python kernel for TIME_SLICE function.

    Args:
        arr: A column/scalar of date/timestamp object
        slice_length: A column/scalar of integer values.
        date_time_part: A column/scalar of strings.
        start_or_end: A column/scalar of strings.
        start_day (Literal[int]): start day of the week

    Returns:
        A column/scalar of date/timestamp objects.
    """
    verify_date_or_datetime_arg_forbid_tz(arr, "time_slice", "arr")
    verify_int_arg(slice_length, "time_slice", "slice_length")
    verify_string_arg(start_or_end, "time_slice", "start_or_end")

    assert_bodo_error(
        is_overload_constant_str(date_time_part),
        "date_time_part must be a string literal!",
    )
    datetime_part = get_overload_const_str(date_time_part)

    assert_bodo_error(
        is_overload_constant_int(start_day), "start_day must be an integer!"
    )
    # Map 0 - 7 to 0 - 6
    day_index = max(0, get_overload_const_int(start_day) - 1)

    # The start day's to align with based on the start_day parameter.
    day_map = {
        6: "1969-12-28",  # Sunday
        0: "1969-12-29",  # Monday
        1: "1969-12-30",  # Tuesday
        2: "1969-12-31",  # Wednesday
        3: "1970-01-01",  # Thursday
        4: "1969-12-26",  # Friday
        5: "1969-12-27",  # Saturday
    }

    beginning_str = day_map[day_index]

    arg_names = ["arr", "slice_length", "date_time_part", "start_or_end", "start_day"]
    arg_types = [arr, slice_length, date_time_part, start_or_end, start_day]
    propagate_null = [True] * 5

    data_is_array = bodo.utils.utils.is_array_typ(arr, True)
    unbox_str = (
        "bodo.utils.conversion.unbox_if_tz_naive_timestamp" if data_is_array else ""
    )
    box_str = "bodo.utils.conversion.box_if_dt64" if data_is_array else ""
    scalar_text = f"arg0 = {box_str}(arg0)\n"

    if datetime_part == "YEAR":
        scalar_text += f"res[i] = {unbox_str}(year_slice(arg0, arg1, arg3))\n"
    elif datetime_part == "QUARTER":
        scalar_text += f"res[i] = {unbox_str}(month_slice(arg0, arg1 * 3, arg3))\n"
    elif datetime_part == "MONTH":
        scalar_text += f"res[i] = {unbox_str}(month_slice(arg0, arg1, arg3))\n"
    elif datetime_part == "WEEK":
        scalar_text += f"res[i] = {unbox_str}(day_slice(arg0, arg1 * 7, pd.Timestamp('{beginning_str}'), arg3))\n"
    elif datetime_part == "DAY":
        scalar_text += f"res[i] = {unbox_str}(day_slice(arg0, arg1, pd.Timestamp('1970-01-01'), arg3))\n"
    elif datetime_part in ("HOUR", "MINUTE", "SECOND"):
        scalar_text += (
            f"res[i] = {unbox_str}(time_slice_helper(arg0, arg1, arg2, arg3))\n"
        )

    out_dtype = types.Array(bodo.types.datetime64ns, 1, "C")
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals={
            "year_slice": year_slice,
            "month_slice": month_slice,
            "day_slice": day_slice,
            "time_slice_helper": time_slice_helper,
        },
    )


@numba.generated_jit(nopython=True)
def year_slice(x, slice_length, start_or_end):
    def impl(x, slice_length, start_or_end):  # pragma: no cover
        beginning = pd.Timestamp("1970-01-01")
        delta_years = slice_length * int((x.year - beginning.year) / slice_length)

        ## Year is before 1970
        if delta_years < 0:
            end = beginning + pd.DateOffset(years=delta_years)
            start = end - pd.DateOffset(years=slice_length)
        else:
            start = beginning + pd.DateOffset(years=delta_years)
            end = start + pd.DateOffset(years=slice_length)

        return start if start_or_end == "START" else end

    return impl


@numba.generated_jit(nopython=True)
def month_slice(x, slice_length, start_or_end):
    def impl(x, slice_length, start_or_end):  # pragma: no cover
        beginning = pd.Timestamp("1970-01-01")

        # Get me the nearest multiple of X months
        delta_in_months = slice_length * int(
            ((x.year - beginning.year) * 12 + (x.month - beginning.month))
            / slice_length
        )

        delta_years, delta_months = divmod(delta_in_months, 12)

        ## Year is before 1970
        if delta_in_months < 0:
            end = beginning + pd.DateOffset(years=delta_years, months=delta_months)
            start = end - pd.DateOffset(months=slice_length)
        else:
            start = beginning + pd.DateOffset(years=delta_years, months=delta_months)
            end = start + pd.DateOffset(months=slice_length)

        return start if start_or_end == "START" else end

    return impl


@numba.generated_jit(nopython=True)
def day_slice(x, slice_length, beginning, start_or_end):
    def impl(x, slice_length, beginning, start_or_end):  # pragma: no cover
        delta_in_days = slice_length * int((x - beginning).days / slice_length)

        ## Year is before beginning.year
        if delta_in_days < 0:
            end = beginning + pd.DateOffset(days=delta_in_days)
            start = end - pd.DateOffset(days=slice_length)
        else:
            start = beginning + pd.DateOffset(days=delta_in_days)
            end = start + pd.DateOffset(days=slice_length)

        return start if start_or_end == "START" else end

    return impl


@numba.generated_jit(nopython=True)
def time_slice_helper(x, slice_length, date_or_time_part, start_or_end):
    def impl(x, slice_length, date_or_time_part, start_or_end):  # pragma: no cover
        multiplier_map = {
            "HOUR": (10**9) * 60 * 60,
            "MINUTE": (10**9) * 60,
            "SECOND": 10**9,
        }

        beginning = pd.Timestamp("1970-01-01")

        unit_multiplier = multiplier_map[date_or_time_part]

        # Get me the nearest multiple of X <insert time part>
        delta = slice_length * int(
            ((x - beginning).value / unit_multiplier) / slice_length
        )
        if delta < 0:
            end = beginning + pd.Timedelta(delta * unit_multiplier)
            start = end - pd.Timedelta(slice_length * unit_multiplier)
        else:
            start = beginning + pd.Timedelta(delta * unit_multiplier)
            end = start + pd.Timedelta(slice_length * unit_multiplier)

        return start if start_or_end == "START" else end

    return impl
