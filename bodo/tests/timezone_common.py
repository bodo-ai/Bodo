# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Common fixtures used for timezone testing."""
import datetime

import pandas as pd
import pytest

import bodo
from bodo.tests.utils import nanoseconds_to_other_time_units


@pytest.fixture(params=["Poland", None])
def sample_tz(request):
    return request.param


# create a fixture that's representative of all timezones
@pytest.fixture(
    params=[
        "UTC",
        "US/Pacific",  # timezone behind UTC
        "Europe/Berlin",  # timezone ahead of UTC
        "Africa/Casablanca",  # timezone that's ahead of UTC only during DST
        "Asia/Kolkata",  # timezone that's offset by 30 minutes
        "Asia/Kathmandu",  # timezone that's offset by 45 minutes
        "Australia/Lord_Howe",  # timezone that's offset by 30 minutes only during DST
        "Pacific/Honolulu",  # timezone that has no DST,
        "Etc/GMT+8",  # timezone that has fixed offset from UTC as opposed to zone
    ]
)
def representative_tz(request):
    return request.param


def generate_date_trunc_func(part: str):
    """
    Generate a function that can be used in Series.map
    to compute the expected output for date_trunc with timestamp input.

    Args:
        part (str): Part to truncate the input to.

    Return:
        Function: Function to use in Series.map to match
            DATE_TRUNC behavior.
    """

    @bodo.jit
    def standardize_part(part):
        return bodo.libs.bodosql_array_kernels.standardize_snowflake_date_time_part(
            part
        )

    # Standardize the part using our snowflake mapping kernel.
    if part is not None:
        standardized_part = standardize_part(part)
    else:
        standardized_part = part

    def date_trunc_scalar_fn(ts_input):
        if pd.isna(part) or pd.isna(ts_input):
            return None
        else:
            if not isinstance(ts_input, pd.Timestamp):
                # Convert tz-naive to a timestamp
                ts_input = pd.Timestamp(ts_input)
            tz = ts_input.tz
            if standardized_part == "quarter":
                quarter = ts_input.quarter
                month_val = 3 * (quarter - 1) + 1
                return pd.Timestamp(year=ts_input.year, month=month_val, day=1, tz=tz)
            elif standardized_part == "year":
                return pd.Timestamp(year=ts_input.year, month=1, day=1, tz=tz)
            elif standardized_part == "week":
                if ts_input.dayofweek == 0:
                    return ts_input.normalize()
                else:
                    return ts_input - pd.tseries.offsets.Week(
                        1, normalize=True, weekday=0
                    )
            elif standardized_part == "month":
                return pd.Timestamp(
                    year=ts_input.year, month=ts_input.month, day=1, tz=tz
                )
            elif standardized_part == "day":
                return ts_input.normalize()
            elif standardized_part == "hour":
                return ts_input.floor("H")
            elif standardized_part == "minute":
                return ts_input.floor("T")
            elif standardized_part == "second":
                return ts_input.floor("S")
            elif standardized_part == "millisecond":
                return ts_input.floor("ms")
            elif standardized_part == "microsecond":
                return ts_input.floor("U")
            else:
                assert standardized_part == "nanosecond"
                return ts_input

    return date_trunc_scalar_fn


def generate_date_trunc_time_func(part_str: str):
    """
    Generate a function that can be used in Series.map
    to compute the expected output for date_trunc with Time type input.

    Args:
        part (str): Part to truncate the input to.

    Return:
        Function: Function to use in Series.map to match
            DATE_TRUNC behavior.
    """

    @bodo.jit
    def standardize_part(part_str):
        return bodo.libs.bodosql_array_kernels.standardize_snowflake_date_time_part(
            part_str
        )

    # Standardize the part using our snowflake mapping kernel.
    if part_str is not None:
        standardized_part = standardize_part(part_str)
    else:
        standardized_part = part_str

    def date_trunc_time_scalar_fn(time_input):
        if pd.isna(part_str) or pd.isna(time_input):
            return None
        else:
            if standardized_part in ('quarter', 'year', 'month', 'week', 'day'):
                # date_or_time_part is too large, set everything to 0
                return bodo.Time()
            else:
                return trunc_time(time_input, standardized_part)

    return date_trunc_time_scalar_fn


def generate_date_trunc_date_func(part_str: str):
    """
    Generate a function that can be used in Series.map
    to compute the expected output for date_trunc with date type input.

    Args:
        part (str): Part to truncate the input to.

    Return:
        Function: Function to use in Series.map to match
            DATE_TRUNC behavior.
    """

    @bodo.jit
    def standardize_part(part_str):
        return bodo.libs.bodosql_array_kernels.standardize_snowflake_date_time_part(
            part_str
        )

    # Standardize the part using our snowflake mapping kernel.
    if part_str is not None:
        standardized_part = standardize_part(part_str)
    else:
        standardized_part = part_str

    def date_trunc_date_scalar_fn(date_input):
        if pd.isna(part_str) or pd.isna(date_input):
            return None
        else:
            return trunc_date(date_input, standardized_part)

    return date_trunc_date_scalar_fn


def trunc_time(time, time_part):
    time_args = []
    time_units = ('hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond')
    for unit in time_units:
        time_args.append(getattr(time, unit))
        if time_part == unit:
            break
    return bodo.Time(*time_args)


def trunc_date(date, date_part):
    if date_part == 'year':
        return datetime.date(date.year, 1, 1)
    elif date_part == 'quarter':
        month = date.month - (date.month - 1) % 3
        return datetime.date(date.year, month, 1)
    elif date_part == 'month':
        return datetime.date(date.year, date.month, 1)
    elif date_part == 'week':
        return date - datetime.timedelta(date.weekday())
    elif date_part == 'day':
        return datetime.date(date.year, date.month, date.day)

def date_sub_unit_time_fn(part_str, time1, time2):
    if pd.isna(part_str) or pd.isna(time1) or pd.isna(time2):
        return None
    @bodo.jit
    def standardize_part(part_str):
        return bodo.libs.bodosql_array_kernels.standardize_snowflake_date_time_part(
            part_str
        )

    # Standardize the part using our snowflake mapping kernel.
    if part_str is not None:
        standardized_part = standardize_part(part_str)
    else:
        standardized_part = part_str

    trunced_time1 = trunc_time(time1, standardized_part)
    trunced_time2 = trunc_time(time2, standardized_part)
    return nanoseconds_to_other_time_units(
        trunced_time2.value - trunced_time1.value,
        standardized_part
    )
