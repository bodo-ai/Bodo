# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Common fixtures used for timezone testing."""
import pandas as pd
import pytest

import bodo


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
    to compute the expected output for date_trunc.

    Note

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
