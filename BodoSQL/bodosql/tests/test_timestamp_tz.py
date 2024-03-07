# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Test correctness of SQL datetime functions with BodoSQL on TIMESTAMP_TZ data
"""

import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.timezone_common import (  # noqa
    representative_tz,
)
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture
def timestamp_tz_data():
    """
    Creates a DataFrame with an index column
    and a column of TimestampTZ data.
    """
    return pd.DataFrame(
        {
            "I": np.arange(8),
            "T": [
                bodo.TimestampTZ.fromLocal("2024-04-01 12:00:00", 0),
                None,
                bodo.TimestampTZ.fromLocal("2024-07-04 20:30:14.250", -30),
                bodo.TimestampTZ.fromLocal("1999-12-31 23:59:59.999526500", 60),
                bodo.TimestampTZ.fromLocal("2024-01-01", -240),
                bodo.TimestampTZ.fromLocal("2024-02-29 6:45:00", 330),
                None,
                bodo.TimestampTZ.fromLocal("2024-04-01 12:00:00", -480),
            ],
        }
    )


@pytest.mark.parametrize(
    "extract_term, answer",
    [
        pytest.param(
            "EXTRACT(YEAR FROM T)",
            pd.array(
                [
                    2024,
                    None,
                    2024,
                    1999,
                    2024,
                    2024,
                    None,
                    2024,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="year",
        ),
        pytest.param(
            "QUARTER(T)",
            pd.array(
                [
                    2,
                    None,
                    3,
                    4,
                    1,
                    1,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="quarter",
        ),
        pytest.param(
            "DATE_PART(mon, T)",
            pd.array(
                [
                    4,
                    None,
                    7,
                    12,
                    1,
                    2,
                    None,
                    4,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="month",
        ),
        pytest.param(
            "HOUR(T)",
            pd.array(
                [
                    12,
                    None,
                    20,
                    23,
                    0,
                    6,
                    None,
                    12,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="hour",
        ),
        pytest.param(
            "DATE_PART(tzh, T)",
            pd.array(
                [
                    0,
                    None,
                    0,
                    1,
                    -4,
                    5,
                    None,
                    -8,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="tz_hour",
        ),
        pytest.param(
            "DATE_PART(tzm, T)",
            pd.array(
                [
                    0,
                    None,
                    -30,
                    0,
                    0,
                    30,
                    None,
                    0,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="tz_minute",
        ),
    ],
)
def test_timestamp_tz_extraction(
    timestamp_tz_data, extract_term, answer, memory_leak_check
):
    """
    Tests that extraction functions work correctly on timestamp_tz data
    """
    query = f"SELECT I, {extract_term} FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = pd.DataFrame(
        {
            "I": np.arange(8),
            "A": answer,
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        enable_timestamp_tz=True,
    )


def test_timestamp_tz_ordering(timestamp_tz_data, memory_leak_check):
    """
    Tests that sorting works correctly on timestamp_tz data
    """
    query = f"SELECT I, T FROM TABLE1 ORDER BY T ASC NULLS LAST, I"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = timestamp_tz_data.iloc[[3, 4, 5, 0, 7, 2, 1, 6], :]
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        enable_timestamp_tz=True,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "cast_term, answer",
    [
        pytest.param(
            "CAST(T AS VARCHAR)",
            pd.array(
                [
                    "2024-04-01 12:00:00 +0000",
                    None,
                    "2024-07-04 20:30:14.250 -00:30",
                    "1999-12-31 23:59:59.999526500 +01:00",
                    "2024-01-01 00:00:00 -04:00",
                    "2024-02-29 06:45:00 +05:30",
                    None,
                    "2024-04-01 12:00:00 -08:00",
                ]
            ),
            id="tz_to_varchar",
            marks=pytest.mark.skip("[BSE-2754]"),
        ),
        pytest.param(
            "TO_DATE(T)",
            pd.array(
                [
                    datetime.date(2024, 4, 1),
                    None,
                    datetime.date(2024, 7, 4),
                    datetime.date(1999, 12, 31),
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 2, 29),
                    None,
                    datetime.date(2024, 4, 1),
                ]
            ),
            id="tz_to_date",
        ),
        pytest.param(
            "TO_TIME(T)",
            pd.array(
                [
                    bodo.Time(12, 0, 0, 0, 0, 0),
                    None,
                    bodo.Time(20, 30, 14, 250, 0, 0),
                    bodo.Time(23, 59, 59, 999, 526, 500),
                    bodo.Time(0, 0, 0, 0, 0, 0),
                    bodo.Time(6, 45, 0, 0, 0, 0),
                    None,
                    bodo.Time(12, 0, 0, 0, 0, 0),
                ]
            ),
            id="tz_to_time",
        ),
        pytest.param(
            "TO_TIMESTAMP_NTZ(T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-04 20:30:14.250"),
                    pd.Timestamp("1999-12-31 23:59:59.999526500"),
                    pd.Timestamp("2024-01-01 00:00:00"),
                    pd.Timestamp("2024-02-29 06:45:00"),
                    None,
                    pd.Timestamp("2024-04-01 12:00:00"),
                ]
            ),
            id="tz_to_ntz",
            marks=pytest.mark.skip("[BSE-2754]"),
        ),
        pytest.param(
            "TO_TIMESTAMP_LTZ(T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 05:00:00", tz="America/Los_Angeles"),
                    None,
                    pd.Timestamp("2024-07-04 14:00:14.250", tz="America/Los_Angeles"),
                    pd.Timestamp(
                        "1999-12-31 14:59:59.999526500", tz="America/Los_Angeles"
                    ),
                    pd.Timestamp("2023-12-31 20:00:00", tz="America/Los_Angeles"),
                    pd.Timestamp("2024-02-28 17:15:00", tz="America/Los_Angeles"),
                    None,
                    pd.Timestamp("2024-04-01 11:00:00.000", tz="America/Los_Angeles"),
                ]
            ),
            id="tz_to_ltz",
            marks=pytest.mark.skip("[BSE-2754]"),
        ),
    ],
)
def test_casting_tz_to_type(timestamp_tz_data, cast_term, answer, memory_leak_check):
    """
    Tests that casting works correctly to transform timestamp_tz data into other types.
    """
    query = f"SELECT I, {cast_term} FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = pd.DataFrame(
        {
            "I": np.arange(8),
            "A": answer,
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        enable_timestamp_tz=True,
        session_tz="America/Los_Angeles",
    )


@pytest.mark.parametrize(
    "data_col, session_tz, answer",
    [
        pytest.param(
            pd.array(
                [
                    "2024-04-01 12:00:00 +00:00",
                    None,
                    "2024-07-04 20:30:14.250",
                    "1999-12-31 23:59:59.999526500 +01:00",
                    "2024-01-01 00:00:00",
                    "2024-02-29 06:45:00 +05:30",
                    None,
                    "2024-04-01 12:00:00 -08:00",
                ]
            ),
            "America/Los_Angeles",
            pd.array(
                [
                    bodo.TimestampTZ.fromLocal("2024-04-01 12:00:00", 0),
                    None,
                    bodo.TimestampTZ.fromLocal("2024-07-04 20:30:14.250", -420),
                    bodo.TimestampTZ.fromLocal("1999-12-31 23:59:59.999526500", 60),
                    bodo.TimestampTZ.fromLocal("2024-01-01 00:00:00", -480),
                    bodo.TimestampTZ.fromLocal("2024-02-29 06:45:00", 330),
                    None,
                    bodo.TimestampTZ.fromLocal("2024-04-01 12:00:00", -480),
                ]
            ),
            id="varchar_to_tz",
            marks=pytest.mark.skip("[BSE-2756]"),
        ),
        pytest.param(
            pd.array(
                [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 1, 22),
                    datetime.date(2024, 2, 12),
                    None,
                    datetime.date(2024, 3, 13),
                    datetime.date(2024, 4, 28),
                    datetime.date(2024, 5, 16),
                    datetime.date(2024, 5, 27),
                    datetime.date(2024, 6, 29),
                    None,
                    datetime.date(2024, 7, 4),
                    datetime.date(2024, 8, 20),
                    datetime.date(2024, 9, 9),
                    datetime.date(2024, 9, 30),
                    None,
                ]
            ),
            "Europe/Berlin",
            pd.array(
                [
                    bodo.TimestampTZ.fromLocal("2024-01-01", 120),
                    bodo.TimestampTZ.fromLocal("2024-01-22", 120),
                    bodo.TimestampTZ.fromLocal("2024-02-12", 120),
                    None,
                    bodo.TimestampTZ.fromLocal("2024-03-13", 120),
                    bodo.TimestampTZ.fromLocal("2024-04-28", 180),
                    bodo.TimestampTZ.fromLocal("2024-05-16", 180),
                    bodo.TimestampTZ.fromLocal("2024-05-27", 180),
                    bodo.TimestampTZ.fromLocal("2024-06-29", 180),
                    None,
                    bodo.TimestampTZ.fromLocal("2024-07-04", 180),
                    bodo.TimestampTZ.fromLocal("2024-08-20", 180),
                    bodo.TimestampTZ.fromLocal("2024-09-09", 180),
                    bodo.TimestampTZ.fromLocal("2024-09-30", 180),
                    None,
                ]
            ),
            id="date_to_tz",
            marks=pytest.mark.skip("[BSE-2756]"),
        ),
        pytest.param(
            pd.array(
                [
                    pd.Timestamp("2024-01-01 12:00:00", tz="Africa/Casablanca"),
                    pd.Timestamp("2024-02-04", tz="Africa/Casablanca"),
                    pd.Timestamp("2024-03-09 20:30:00", tz="Africa/Casablanca"),
                    None,
                    pd.Timestamp("2024-04-16 16:45:13.249500", tz="Africa/Casablanca"),
                    pd.Timestamp("2024-05-25 01:00:00", tz="Africa/Casablanca"),
                ]
            ),
            "Africa/Casablanca",
            pd.array(
                [
                    bodo.TimestampTZ.fromLocal("2024-01-01 12:00:00", 60),
                    bodo.TimestampTZ.fromLocal("2024-02-04", 60),
                    bodo.TimestampTZ.fromLocal("2024-03-09 20:30:00", 60),
                    None,
                    bodo.TimestampTZ.fromLocal("2024-04-16 16:45:13.249500", 0),
                    bodo.TimestampTZ.fromLocal("2024-05-25 01:00:00", 0),
                ]
            ),
            id="ltz_to_tz",
            marks=pytest.mark.skip("[BSE-2756]"),
        ),
    ],
)
def test_casting_type_to_tz(data_col, session_tz, answer, memory_leak_check):
    """
    Tests that casting works correctly to transform other types into timestamp_tz data.
    """
    query = f"SELECT I, TO_TIMESTAMP_TZ(T) FROM TABLE1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "I": np.arange(len(data_col)),
                "T": data_col,
            }
        )
    }
    expected_output = pd.DataFrame(
        {
            "I": np.arange(len(data_col)),
            "A": answer,
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        session_tz=session_tz,
        enable_timestamp_tz=True,
    )
