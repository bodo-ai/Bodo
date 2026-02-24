"""
Test correctness of SQL datetime functions with BodoSQL on TIMESTAMP_TZ data
"""

import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
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
                bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", 0),
                None,
                bodo.types.TimestampTZ.fromLocal("2024-07-04 20:30:14.250", 30),
                bodo.types.TimestampTZ.fromLocal("1999-12-31 23:59:59.999526500", -60),
                bodo.types.TimestampTZ.fromLocal("2024-01-01", -240),
                bodo.types.TimestampTZ.fromLocal("2024-02-29 6:45:00", 330),
                None,
                bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", -480),
            ],
        }
    )


@pytest.fixture
def expanded_timestamp_tz_data():
    """
    Same as timestamp_tz_data but with some rows that have the same UTC
    timestamp even though they have different offsets.

    Specifically, the following rows are equivalent:

    - Row 0 = Row 1 = Row 2
    - Row 3 = Row 5 = Row 7
    """
    return pd.DataFrame(
        {
            "I": np.arange(10),
            "T": [
                bodo.types.TimestampTZ.fromUTC("2024-03-09 22:55:34", 0),
                bodo.types.TimestampTZ.fromUTC("2024-03-09 22:55:34", 60),
                bodo.types.TimestampTZ.fromUTC("2024-03-09 22:55:34", -480),
                bodo.types.TimestampTZ.fromUTC("1999-12-31 23:59:59.999999999", 0),
                bodo.types.TimestampTZ.fromUTC("2024-07-04", 30),
                bodo.types.TimestampTZ.fromUTC("1999-12-31 23:59:59.999999999", -30),
                bodo.types.TimestampTZ.fromUTC("2024-04-01 06:45:00", 0),
                bodo.types.TimestampTZ.fromUTC("1999-12-31 23:59:59.999999999", 450),
                bodo.types.TimestampTZ.fromUTC("2024-04-01 08:00:00", 75),
                None,
            ],
        }
    )


@pytest.mark.parametrize(
    "dateadd_calc, local_answer",
    [
        pytest.param(
            "DATEADD(year, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2026-07-04 20:30:14.250"),
                    pd.Timestamp("1996-12-31 23:59:59.999526500"),
                    pd.Timestamp("2028-01-01"),
                    pd.Timestamp("2019-02-28 6:45:00"),
                    None,
                    pd.Timestamp("2017-04-01 12:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            id="year",
        ),
        pytest.param(
            "TIMEADD(quarter, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2025-01-04 20:30:14.250"),
                    pd.Timestamp("1999-3-31 23:59:59.999526500"),
                    pd.Timestamp("2025-01-01"),
                    pd.Timestamp("2022-11-29 6:45:00"),
                    None,
                    pd.Timestamp("2022-07-01 12:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            id="quarter",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TIMEADD(month, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-09-04 20:30:14.250"),
                    pd.Timestamp("1999-09-30 23:59:59.999526500"),
                    pd.Timestamp("2024-05-01"),
                    pd.Timestamp("2023-09-29 6:45:00"),
                    None,
                    pd.Timestamp("2023-09-01 12:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            id="month",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "DATEADD(day, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-06 20:30:14.250"),
                    pd.Timestamp("1999-12-28 23:59:59.999526500"),
                    pd.Timestamp("2024-01-05"),
                    pd.Timestamp("2024-02-24 6:45:00"),
                    None,
                    pd.Timestamp("2024-03-25 12:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            id="day",
        ),
        pytest.param(
            "TIMEADD(hour, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-04 22:30:14.250"),
                    pd.Timestamp("1999-12-31 20:59:59.999526500"),
                    pd.Timestamp("2024-01-01 04:00:00"),
                    pd.Timestamp("2024-02-29 01:45:00"),
                    None,
                    pd.Timestamp("2024-04-01 05:00:00"),
                ],
                dtype="datetime64[ns]",
            ),
            id="hour",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TIMESTAMPADD(second, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-04 20:30:16.250"),
                    pd.Timestamp("1999-12-31 23:59:56.999526500"),
                    pd.Timestamp("2024-01-01 00:00:04"),
                    pd.Timestamp("2024-02-29 6:44:55"),
                    None,
                    pd.Timestamp("2024-04-01 11:59:53"),
                ],
                dtype="datetime64[ns]",
            ),
            id="second",
        ),
        pytest.param(
            "DATEADD(microsecond, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-04 20:30:14.250002"),
                    pd.Timestamp("1999-12-31 23:59:59.999523500"),
                    pd.Timestamp("2024-01-01 00:00:00.000004"),
                    pd.Timestamp("2024-02-29 06:44:59.999995"),
                    None,
                    pd.Timestamp("2024-04-01 11:59:59.999993"),
                ],
                dtype="datetime64[ns]",
            ),
            id="microsecond",
        ),
        pytest.param(
            "TIMEADD(nanosecond, I * POW(-1, I), T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 12:00:00"),
                    None,
                    pd.Timestamp("2024-07-04 20:30:14.250000002"),
                    pd.Timestamp("1999-12-31 23:59:59.999526497"),
                    pd.Timestamp("2024-01-01 00:00:00.000000004"),
                    pd.Timestamp("2024-02-29 06:44:59.999999995"),
                    None,
                    pd.Timestamp("2024-04-01 11:59:59.999999993"),
                ],
                dtype="datetime64[ns]",
            ),
            id="nanosecond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "T - INTERVAL 1 MONTH",
            [
                pd.Timestamp("2024-03-01 12:00:00"),
                None,
                pd.Timestamp("2024-06-04 20:30:14.250"),
                pd.Timestamp("1999-11-30 23:59:59.999526500"),
                pd.Timestamp("2023-12-01"),
                pd.Timestamp("2024-01-29 6:45:00"),
                None,
                pd.Timestamp("2024-03-01 12:00:00"),
            ],
            id="interval_month",
        ),
        pytest.param(
            "T + INTERVAL 10 days",
            [
                pd.Timestamp("2024-04-11 12:00:00"),
                None,
                pd.Timestamp("2024-07-14 20:30:14.250"),
                pd.Timestamp("2000-01-10 23:59:59.999526500"),
                pd.Timestamp("2024-01-11"),
                pd.Timestamp("2024-03-10 6:45:00"),
                None,
                pd.Timestamp("2024-04-11 12:00:00"),
            ],
            id="interval_day",
        ),
        pytest.param(
            "T + INTERVAL 8 hours",
            [
                pd.Timestamp("2024-04-01 20:00:00"),
                None,
                pd.Timestamp("2024-07-05 04:30:14.250"),
                pd.Timestamp("2000-01-01 07:59:59.999526500"),
                pd.Timestamp("2024-01-01 08:00:00"),
                pd.Timestamp("2024-02-29 14:45:00"),
                None,
                pd.Timestamp("2024-04-01 20:00:00"),
            ],
            id="interval_hour",
        ),
    ],
)
def test_timestamp_tz_dateadd(
    timestamp_tz_data, dateadd_calc, local_answer, memory_leak_check
):
    """
    Tests the datetime arithmetic operations on TIMESTMAP_TZ data.

    timestamp_tz_data: the fixture containing the input data.
    dateadd_calc: the expression used to calculate a dateadd call with the columns from timestamp_tz_data.
    local_answer: the expected local timestamp (in NTZ) produced from dateadd_calc.
    """
    query = f"SELECT I, {dateadd_calc} as N, DATE_PART(tzh, {dateadd_calc}) as H, DATE_PART(tzm, {dateadd_calc}) as M FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    offsets = [0, None, 30, -60, -240, 330, None, -480]
    expected_output = pd.DataFrame(
        {
            "I": np.arange(8),
            "A": [
                None if off is None else bodo.types.TimestampTZ.fromLocal(ans, off)
                for ans, off in zip(local_answer, offsets)
            ],
            "H": pd.array([0, None, 0, -1, -4, 5, None, -8], dtype=pd.Int16Dtype()),
            "M": pd.array([0, None, 30, 0, 0, 30, None, 0], dtype=pd.Int16Dtype()),
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


@pytest.mark.parametrize(
    "datediff_calc, diff_answer",
    [
        pytest.param(
            "DATEDIFF(year, TO_TIMESTAMP_TZ('1970-07-01 00:00:00 +0000'), T)",
            pd.array(
                [
                    54,
                    None,
                    54,
                    29,
                    54,
                    54,
                    None,
                    54,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="year",
        ),
        pytest.param(
            "TIMEDIFF(month, TO_TIMESTAMP_TZ('2000-03-01 00:00:00 +0000'), T)",
            pd.array(
                [
                    289,
                    None,
                    292,
                    -3,
                    286,
                    287,
                    None,
                    289,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="month",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            # Note that in the constant timestamp in this query, the UTC date
            # is a different day from the local date - this tests that datediff
            # is comparing using the UTC date
            "TIMESTAMPDIFF(day, T, TO_TIMESTAMP_TZ('2024-01-01 00:00:00 +1234'))",
            pd.array(
                [
                    (datetime.date(2024, 1, 1) - datetime.date(2024, 4, 1)).days,
                    None,
                    (datetime.date(2024, 1, 1) - datetime.date(2024, 7, 4)).days,
                    (datetime.date(2024, 1, 1) - datetime.date(1999, 12, 31)).days,
                    (datetime.date(2024, 1, 1) - datetime.date(2024, 1, 1)).days,
                    (datetime.date(2024, 1, 1) - datetime.date(2024, 2, 29)).days,
                    None,
                    (datetime.date(2024, 1, 1) - datetime.date(2024, 4, 1)).days,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="day",
        ),
        pytest.param(
            "DATEDIFF(hour, TO_TIMESTAMP_TZ(DATE_TRUNC(day, TO_TIMESTAMP_NTZ(T))), T)",
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
            "DATEDIFF(minute, TO_TIMESTAMP_TZ(DATE_TRUNC(day, TO_TIMESTAMP_NTZ(T))), T)",
            pd.array(
                [
                    720,
                    None,
                    1230,
                    1439,
                    0,
                    405,
                    None,
                    720,
                ],
                dtype=pd.Int32Dtype(),
            ),
            id="minute",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_timestamp_tz_datediff(
    timestamp_tz_data, datediff_calc, diff_answer, memory_leak_check
):
    """
    Tests the datetime subtraction arithmetic operations on TIMESTMAP_TZ data.

    timestamp_tz_data: the fixture containing the input data.
    datediff_calc: the expression used to calculate a dateadd call with the columns from timestamp_tz_data.
    diff_answer: the expected difference between the two dates in the desired interval.
    """
    query = f"SELECT I, {datediff_calc} as D FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = pd.DataFrame(
        {
            "I": np.arange(8),
            "D": diff_answer,
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


@pytest.mark.parametrize(
    "unit, answer",
    [
        pytest.param(
            "DAY",
            pd.array(
                [
                    str(bodo.types.TimestampTZ.fromLocal("2024-04-01", 0)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-07-04", 30)),
                    str(bodo.types.TimestampTZ.fromLocal("1999-12-31", -60)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", -240)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-02-29", 330)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-04-01", -480)),
                ],
            ),
            id="day",
        ),
        pytest.param(
            "YEAR",
            pd.array(
                [
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", 0)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", 30)),
                    str(bodo.types.TimestampTZ.fromLocal("1999-01-01", -60)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", -240)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", 330)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", -480)),
                ],
            ),
            id="year",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SECOND",
            pd.array(
                [
                    str(bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", 0)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-07-04 20:30:14", 30)),
                    str(bodo.types.TimestampTZ.fromLocal("1999-12-31 23:59:59", -60)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-01-01", -240)),
                    str(bodo.types.TimestampTZ.fromLocal("2024-02-29 6:45:00", 330)),
                    None,
                    str(bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", -480)),
                ],
            ),
            id="second",
        ),
    ],
)
def test_timestamp_tz_datetrunc(timestamp_tz_data, unit, answer, memory_leak_check):
    """
    Tests the datetime subtraction arithmetic operations on TIMESTMAP_TZ data.

    timestamp_tz_data: the fixture containing the input data.
    datediff_calc: the expression used to calculate a dateadd call with the columns from timestamp_tz_data.
    diff_answer: the expected difference between the two dates in the desired interval.
    """
    query = f"SELECT I, TO_VARCHAR(DATE_TRUNC('{unit}', T)) as D FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = pd.DataFrame(
        {
            "I": np.arange(8),
            "D": answer,
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
                    -1,
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
                    30,
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
    query = "SELECT I, T FROM TABLE1 ORDER BY T ASC NULLS LAST, I"
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
    "union_all",
    [
        pytest.param(True, id="all"),
        pytest.param(False, id="distinct"),
    ],
)
def test_timestamp_tz_union(expanded_timestamp_tz_data, union_all, memory_leak_check):
    """
    Tests that UNION ALL/DISTINCT works correctly on timestamp_tz data.
    """
    df = pd.DataFrame({"T": expanded_timestamp_tz_data["T"]})
    operator = "ALL" if union_all else "DISTINCT"
    query = f"(SELECT * FROM TABLE1) UNION {operator} (SELECT * FROM TABLE2)"
    pattern_1 = [0, 2, 4, 6, 2, 4, 2, 8, 9] * 100
    pattern_2 = [0, 1, 2, 3, 5, 9, 9, 7, 0, 2, 5, 2, 7, 9] * 300
    if union_all:
        pattern_3 = pattern_1 + pattern_2
    else:
        # rows 0/1/2 and 3/5/7 are equivalent so they are not all included
        pattern_3 = [0, 3, 4, 6, 8, 9]
    ctx = {
        "TABLE1": df.iloc[pattern_1, :],
        "TABLE2": df.iloc[pattern_2, :],
    }
    expected_output = df.iloc[pattern_3, :]
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        enable_timestamp_tz=True,
    )


def test_timestamp_tz_groupby(expanded_timestamp_tz_data, memory_leak_check):
    """
    Tests grouping by TIMESTAMP_TZ keys.
    """
    query = "SELECT T, SUM(I) as S FROM TABLE1 GROUP BY T"
    ctx = {"TABLE1": pd.concat([expanded_timestamp_tz_data] * 5)}
    expected_output = pd.DataFrame(
        {
            "T": expanded_timestamp_tz_data["T"].iloc[[0, 3, 4, 6, 8, 9]],
            "S": [15, 75, 20, 30, 40, 45],
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


@pytest.mark.parametrize(
    "cast_term, answer",
    [
        pytest.param(
            "CAST(T AS VARCHAR)",
            pd.array(
                [
                    "2024-04-01 12:00:00 +0000",
                    None,
                    "2024-07-04 20:30:14.250000 +0030",
                    "1999-12-31 23:59:59.999526500 -0100",
                    "2024-01-01 00:00:00 -0400",
                    "2024-02-29 06:45:00 +0530",
                    None,
                    "2024-04-01 12:00:00 -0800",
                ]
            ),
            id="tz_to_varchar",
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
                    bodo.types.Time(12, 0, 0, 0, 0, 0),
                    None,
                    bodo.types.Time(20, 30, 14, 250, 0, 0),
                    bodo.types.Time(23, 59, 59, 999, 526, 500),
                    bodo.types.Time(0, 0, 0, 0, 0, 0),
                    bodo.types.Time(6, 45, 0, 0, 0, 0),
                    None,
                    bodo.types.Time(12, 0, 0, 0, 0, 0),
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
                ],
                dtype="datetime64[ns]",
            ),
            id="tz_to_ntz",
        ),
        pytest.param(
            "TO_TIMESTAMP_LTZ(T)",
            pd.array(
                [
                    pd.Timestamp("2024-04-01 05:00:00", tz="America/Los_Angeles"),
                    None,
                    pd.Timestamp("2024-07-04 13:00:14.250", tz="America/Los_Angeles"),
                    pd.Timestamp(
                        "1999-12-31 16:59:59.999526500", tz="America/Los_Angeles"
                    ),
                    pd.Timestamp("2023-12-31 20:00:00", tz="America/Los_Angeles"),
                    pd.Timestamp("2024-02-28 17:15:00", tz="America/Los_Angeles"),
                    None,
                    pd.Timestamp("2024-04-01 13:00:00.000", tz="America/Los_Angeles"),
                ],
                dtype="datetime64[ns, America/Los_Angeles]",
            ),
            id="tz_to_ltz",
        ),
    ],
)
def test_casting_tz_to_type(timestamp_tz_data, cast_term, answer, memory_leak_check):
    """
    Tests that casting works correctly to transform timestamp_tz data into other types.
    """
    query = f"SELECT I, {cast_term} as A FROM TABLE1"
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
                    # TODO: test with other formats
                    "2024-04-01 12:00:00 +0000",
                    None,
                    "2024-07-04 20:30:14.250",
                    "1999-12-31 23:59:59.999526500 +0100",
                    "2024-01-01 00:00:00",
                    "2024-02-29 06:45:00 +0530",
                    None,
                    "2024-04-01 12:00:00 -0800",
                ]
            ),
            "America/Los_Angeles",
            pd.array(
                [
                    bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", 0),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2024-07-04 20:30:14.250", -420),
                    bodo.types.TimestampTZ.fromLocal(
                        "1999-12-31 23:59:59.999526500", 60
                    ),
                    bodo.types.TimestampTZ.fromLocal("2024-01-01 00:00:00", -480),
                    bodo.types.TimestampTZ.fromLocal("2024-02-29 06:45:00", 330),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2024-04-01 12:00:00", -480),
                ]
            ),
            id="varchar_to_tz",
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
                    bodo.types.TimestampTZ.fromLocal("2024-01-01", 60),
                    bodo.types.TimestampTZ.fromLocal("2024-01-22", 60),
                    bodo.types.TimestampTZ.fromLocal("2024-02-12", 60),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2024-03-13", 60),
                    bodo.types.TimestampTZ.fromLocal("2024-04-28", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-05-16", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-05-27", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-06-29", 120),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2024-07-04", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-08-20", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-09-09", 120),
                    bodo.types.TimestampTZ.fromLocal("2024-09-30", 120),
                    None,
                ]
            ),
            id="date_to_tz",
        ),
        pytest.param(
            pd.array(
                [
                    pd.Timestamp("2024-01-01 12:00:00", tz="Africa/Casablanca"),
                    pd.Timestamp("2024-02-04", tz="Africa/Casablanca"),
                    pd.Timestamp("2024-03-09 20:30:00", tz="Africa/Casablanca"),
                    None,
                    pd.Timestamp("2026-02-16 16:45:13.249500", tz="Africa/Casablanca"),
                    pd.Timestamp("2026-05-25 01:00:00", tz="Africa/Casablanca"),
                ],
                dtype="datetime64[ns, Africa/Casablanca]",
            ),
            "Africa/Casablanca",
            pd.array(
                [
                    bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", 60),
                    bodo.types.TimestampTZ.fromLocal("2024-02-04", 60),
                    bodo.types.TimestampTZ.fromLocal("2024-03-09 20:30:00", 60),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2026-02-16 16:45:13.249500", 0),
                    bodo.types.TimestampTZ.fromLocal("2026-05-25 01:00:00", 60),
                ]
            ),
            id="ltz_to_tz",
        ),
    ],
)
def test_casting_type_to_tz(data_col, session_tz, answer, memory_leak_check):
    """
    Tests that casting works correctly to transform other types into timestamp_tz data.

    TODO: support TRY_TO_TIMESTAMP_TZ
    """
    query = "SELECT I, TO_TIMESTAMP_TZ(T) as A FROM TABLE1"
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


@pytest.mark.parametrize(
    "calc, answer",
    [
        pytest.param(
            "TIMESTAMP_TZ_FROM_PARTS(2020+I, I, I*I, 12, 0, 0)",
            [
                bodo.types.TimestampTZ.fromLocal("2021-01-01 12:00:00", -480),
                bodo.types.TimestampTZ.fromLocal("2022-02-04 12:00:00", -480),
                None,
                bodo.types.TimestampTZ.fromLocal("2024-04-16 12:00:00", -420),
                bodo.types.TimestampTZ.fromLocal("2025-05-25 12:00:00", -420),
            ],
            id="no_case-no_ns-no_tz",
        ),
        pytest.param(
            "CASE WHEN I <= 1 THEN NULL ELSE TIMESTAMP_TZ_FROM_PARTS(2024, 12, 31, 06, 45, 15, POW(16, I)) END",
            [
                None,
                bodo.types.TimestampTZ.fromLocal("2024-12-31 06:45:15.000000256", -480),
                None,
                bodo.types.TimestampTZ.fromLocal("2024-12-31 06:45:15.000065536", -480),
                bodo.types.TimestampTZ.fromLocal("2024-12-31 06:45:15.001048576", -480),
            ],
            id="with_case-with_ns-no_tz",
        ),
        pytest.param(
            "TIMESTAMPTZFROMPARTS(2020+I, I, I*I, 12, 0, 0, 926000000, 'Europe/Berlin')",
            [
                bodo.types.TimestampTZ.fromLocal("2021-01-01 12:00:00.926", 60),
                bodo.types.TimestampTZ.fromLocal("2022-02-04 12:00:00.926", 60),
                None,
                bodo.types.TimestampTZ.fromLocal("2024-04-16 12:00:00.926", 120),
                bodo.types.TimestampTZ.fromLocal("2025-05-25 12:00:00.926", 120),
            ],
            id="no_case-with_ns-with_tz",
        ),
    ],
)
def test_timestamp_tz_from_parts(calc, answer, memory_leak_check):
    """
    Tests calling the TIMESTAMP_TZ_FROM_PARTS function.
    """
    query = f"SELECT I, TO_TIMESTAMP_NTZ({calc}) as N, DATE_PART(tzh, {calc}) as H, DATE_PART(tzm, {calc}) as M FROM TABLE1"
    df = pd.DataFrame({"I": pd.array([1, 2, None, 4, 5])})
    ctx = {"TABLE1": df}
    ntz_answer = pd.array([None if t is None else t.local_timestamp() for t in answer])
    hour_answer = pd.array(
        [
            None
            if t is None
            else (abs(t._offset_minutes) // 60) * (1 if t._offset_minutes >= 0 else -1)
            for t in answer
        ]
    )
    minute_answer = pd.array(
        [
            None
            if t is None
            else (abs(t._offset_minutes) % 60) * (1 if t._offset_minutes >= 0 else -1)
            for t in answer
        ]
    )
    expected_output = pd.DataFrame(
        {
            "I": df["I"],
            "N": ntz_answer,
            "H": hour_answer,
            "M": minute_answer,
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        session_tz="America/Los_Angeles",
        enable_timestamp_tz=True,
    )


@pytest.mark.parametrize(
    "calculation, answer",
    [
        pytest.param(
            "T = '2024-07-04 20:30:14.250 +0030' :: TIMESTAMP_TZ OR "
            + "T = '2024-02-29 1:15:00 -0000' :: TIMESTAMP_TZ",
            pd.array(
                [False, None, True, False, False, True, None, False],
                dtype=pd.BooleanDtype(),
            ),
            id="equals_literals",
        ),
        pytest.param(
            "DATE_PART(hour, '2024-07-04 20:30:14.250' :: TIMESTAMP_TZ)",
            pd.array([20] * 8, dtype=pd.Int16Dtype()),
            id="literal_hour",
        ),
        pytest.param(
            "DATE_PART(minute, '2024-07-04 20:30:14.250 -1045' :: TIMESTAMP_TZ)",
            pd.array([30] * 8, dtype=pd.Int16Dtype()),
            id="literal_minute",
        ),
        pytest.param(
            "DATE_PART(tzh, '2024-07-04 20:30:14.250' :: TIMESTAMP_TZ)",
            pd.array([-7] * 8, dtype=pd.Int16Dtype()),
            id="literal_hour_offset",
        ),
        pytest.param(
            "DATE_PART(tzm, '2024-07-04 20:30:14.250 -1045' :: TIMESTAMP_TZ)",
            pd.array([-45] * 8, dtype=pd.Int16Dtype()),
            id="literal_minute_offset",
        ),
    ],
)
def test_timestamp_tz_literal(
    timestamp_tz_data, calculation, answer, memory_leak_check
):
    """
    Tests queries that involve TIMESTAMP_TZ literals.
    """
    query = f"SELECT I, {calculation} as A FROM TABLE1"
    ctx = {"TABLE1": timestamp_tz_data}
    expected_output = pd.DataFrame(
        {
            "I": np.arange(len(answer)),
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
        session_tz="America/Los_Angeles",
        enable_timestamp_tz=True,
    )
