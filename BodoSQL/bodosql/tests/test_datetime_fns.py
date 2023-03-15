# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL datetime functions with BodoSQL
"""
import datetime

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import bodosql_use_date_type, check_query

from bodo import Time
from bodo.tests.conftest import (  # noqa
    date_df,
    day_part_strings,
    time_df,
    time_part_strings,
)
from bodo.tests.timezone_common import (  # noqa
    generate_date_trunc_date_func,
    generate_date_trunc_func,
    generate_date_trunc_time_func,
    representative_tz,
)

EQUIVALENT_SPARK_DT_FN_MAP = {
    "WEEK": "WEEKOFYEAR",
    "CURDATE": "CURRENT_DATE",
}


def interval_day_add_func(num_days: int):
    """
    Generates a function to be passed
    to Series.map to emulate the result of adding
    an interval to Timestamp value. This moves by
    N days, in the local timezone.

    Args:
        num_days (int): Number of days to move

    Returns
        (Function): Function that can be used
        in Series.map to move each row by num_days.
    """
    offset = pd.Timedelta(days=num_days)

    def map_func(val):
        if pd.isna(val):
            return None
        new_ts = val + offset
        return pd.Timestamp(
            year=new_ts.year,
            month=new_ts.month,
            day=new_ts.day,
            hour=val.hour,
            minute=val.minute,
            second=val.second,
            microsecond=val.microsecond,
            nanosecond=val.nanosecond,
            tz=new_ts.tz,
        )

    return map_func


@pytest.fixture(
    params=[
        "timestamps",
        pytest.param("timestamps_normalized", marks=pytest.mark.slow),
        "datetime_strings",
    ]
)
def timestamp_date_string_cols(request):
    return request.param


@pytest.fixture(
    params=[
        "DATE_SUB",
        pytest.param("SUBDATE", marks=pytest.mark.slow),
    ]
)
def subdate_equiv_fns(request):
    return request.param


@pytest.fixture(
    params=[
        "quarter",
        "yyy",
        "MONTH",
        "WEEK",
        "DAY",
        "HOUR",
        "MINUTE",
        "SECOND",
        "ms",
        "microsecond",
        "nanosecs",
    ]
)
def date_trunc_literal(request):
    return request.param


@pytest.fixture(
    params=[
        "SECOND",
        "MINUTE",
        "HOUR",
        "DAY",
        "WEEK",
        pytest.param(
            "MONTH",
            marks=pytest.mark.skip(
                "Literal Month intervals not supported in our Visitor, see BS-216"
            ),
        ),
        pytest.param(
            "YEAR",
            marks=pytest.mark.skip(
                "Literal Year intervals not supported in our Visitor, see BS-216"
            ),
        ),
    ]
)
def mysql_interval_str(request):
    return request.param


@pytest.fixture(params=["EUR", "USA", "JIS", "ISO"])
def get_format_str(request):
    return request.param


@pytest.fixture
def tz_aware_df():
    # Transition to Daylight Savings
    # "1D2H37T48S" --> 1 day, 2 hours, 37 minutes, 48 seconds
    to_dst_series = pd.date_range(
        start="11/3/2021", freq="1D2H37T48S", periods=30, tz="US/Pacific"
    ).to_series()

    # Transition back from Daylight Savings
    from_dst_series = pd.date_range(
        start="03/1/2022", freq="0D12H30T1S", periods=60, tz="US/Pacific"
    ).to_series()

    # February is weird with leap years
    feb_leap_year_series = pd.date_range(
        start="02/20/2020", freq="1D0H30T0S", periods=20, tz="US/Pacific"
    ).to_series()

    second_quarter_series = pd.date_range(
        start="05/01/2015", freq="2D0H1T59S", periods=20, tz="US/Pacific"
    ).to_series()

    third_quarter_series = pd.date_range(
        start="08/17/2000", freq="10D1H1T10S", periods=20, tz="US/Pacific"
    ).to_series()

    df = pd.DataFrame(
        {
            "A": pd.concat(
                [
                    to_dst_series,
                    from_dst_series,
                    feb_leap_year_series,
                    second_quarter_series,
                    third_quarter_series,
                ]
            )
        }
    )

    return {"table1": df}


@pytest.fixture
def dt_fn_dataframe():
    dt_strings = [
        "2011-01-01",
        "1971-02-02",
        "2021-03-03",
        "2021-05-31",
        None,
        "2020-12-01T13:56:03.172",
        "2007-01-01T03:30",
        None,
        "2001-12-01T12:12:02.21",
        "2100-10-01T13:00:33.1",
    ]
    timestamps = pd.Series(
        [np.datetime64(x) if x is not None else x for x in dt_strings],
        dtype="datetime64[ns]",
    )
    normalized_ts = timestamps.dt.normalize()
    invalid_dt_strings = [
        "__" + str(x) + "__" if x is not None else x for x in dt_strings
    ]
    df = pd.DataFrame(
        {
            "timestamps": timestamps,
            "timestamps_normalized": normalized_ts,
            "intervals": [
                np.timedelta64(10, "Y"),
                np.timedelta64(9, "M"),
                np.timedelta64(8, "W"),
                np.timedelta64(6, "h"),
                np.timedelta64(5, "m"),
                None,
                np.timedelta64(4, "s"),
                np.timedelta64(3, "ms"),
                np.timedelta64(2000000, "us"),
                None,
            ],
            "datetime_strings": dt_strings,
            "invalid_dt_strings": invalid_dt_strings,
            "positive_integers": pd.Series(
                [1, 2, 31, 400, None, None, 123, 13, 7, 80], dtype=pd.Int64Dtype()
            ),
            "small_positive_integers": pd.Series(
                [1, 2, 3, None, 4, 5, 6, None, 7, 8], dtype=pd.Int64Dtype()
            ),
            "dt_format_strings": [
                "%Y",
                "%a, %b, %c",
                "%D, %d, %f, %p, %S",
                "%Y, %M, %D",
                "%y",
                "%T",
                None,
                "%r",
                "%j",
                None,
            ],
            "valid_year_integers": pd.Series(
                [
                    2000,
                    None,
                    2100,
                    1990,
                    2020,
                    None,
                    2021,
                    1998,
                    2200,
                    1970,
                ],
                dtype=pd.Int64Dtype(),
            ),
            "mixed_integers": pd.Series(
                [None, 0, 1, -2, 3, -4, 5, -6, 7, None], dtype=pd.Int64Dtype()
            ),
            "digit_strings": [
                None,
                "-13",
                "0",
                "-2",
                "23",
                "5",
                "5",
                "-66",
                "1234",
                None,
            ],
            "days_of_week": [
                "mo",
                "tu",
                "we",
                "th",
                "fr",
                "sa",
                None,
                "su",
                "mo",
                None,
            ],
        }
    )
    return {"table1": df}


@pytest.fixture(
    params=[
        pytest.param((x, ["timestamps"], ("1", "2")), id=x)
        for x in [
            "SECOND",
            "MINUTE",
            "DAYOFYEAR",
            "HOUR",
            "DAYOFMONTH",
            "MONTH",
            "QUARTER",
            "YEAR",
        ]
    ]
    + [
        pytest.param(
            (
                "WEEKDAY",
                ["timestamps"],
                (
                    "1",
                    "2",
                ),
            ),
            id="WEEKDAY",
        )
    ]
    + [
        pytest.param(
            (
                "LAST_DAY",
                ["timestamps"],
                (
                    "TIMESTAMP '1971-02-02'",
                    "TIMESTAMP '2021-03-03'",
                ),
            ),
            id="LAST_DAY",
        )
    ]
    + [
        pytest.param((x, ["timestamps"], ("1", "2")), id=x)
        for x in [
            "WEEK",
            "WEEKOFYEAR",
        ]
    ]
    + [
        pytest.param(
            ("CURDATE", [], ("TIMESTAMP 2020-04-25", "TIMESTAMP 2020-04-25")),
        ),
        pytest.param(
            ("CURRENT_DATE", [], ("TIMESTAMP 2020-04-25", "TIMESTAMP 2020-04-25")),
        ),
    ]
    + [
        pytest.param(
            (
                "DATE",
                ["datetime_strings"],
                ("TIMESTAMP 2020-04-25", "TIMESTAMP 2020-04-25"),
            ),
            marks=pytest.mark.skip(
                "Parser change required to, support this will be done in a later PR"
            ),
        ),
        pytest.param(
            (
                "TIMESTAMP",
                ["datetime_strings", ("TIMESTAMP 2020-04-25", "TIMESTAMP 2020-04-25")],
            ),
            marks=pytest.mark.skip(
                "Parser change required to, support this will be done in a later PR"
            ),
        ),
    ]
)
def dt_fn_info(request):
    """fixture that returns information used to test datatime functionss
    First argument is function name,
    the second is a list of arguments to use with the function
    The third argument is tuple of two possible return values for the function, which
    are used while checking scalar cases
    """
    return request.param


def test_dt_fns_cols(spark_info, dt_fn_info, dt_fn_dataframe, memory_leak_check):
    """tests that the specified date_time functions work on columns"""
    bodo_fn_name = dt_fn_info[0]
    arglistString = ", ".join(dt_fn_info[1])
    bodo_fn_call = f"{bodo_fn_name}({arglistString})"

    # spark extends scalar values to be equal to the length of the input table
    # We don't, so this workaround is needed
    if len(dt_fn_info[1]) == 0:
        query = f"SELECT {bodo_fn_call}"
    else:
        query = f"SELECT {bodo_fn_call} FROM table1"

    if bodo_fn_name in EQUIVALENT_SPARK_DT_FN_MAP:
        spark_fn_name = EQUIVALENT_SPARK_DT_FN_MAP[bodo_fn_name]
        spark_fn_call = f"{spark_fn_name}({arglistString})"
        # spark extends scalar values to be equal to the length of the input table
        # We don't, so this workaround is needed
        if len(dt_fn_info[1]) == 0:
            spark_query = f"SELECT {spark_fn_call}"
        else:
            spark_query = f"SELECT {spark_fn_call} FROM table1"
    else:
        spark_query = None
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_dt_fns_scalars(spark_info, dt_fn_info, dt_fn_dataframe, memory_leak_check):
    """tests that the specified string functions work on Scalars"""
    bodo_fn_name = dt_fn_info[0]

    if len(dt_fn_info[1]) == 0:
        return

    arglistString = ", ".join(dt_fn_info[1])
    bodo_fn_call = f"{bodo_fn_name}({arglistString})"
    retval_1 = dt_fn_info[2][0]
    retval_2 = dt_fn_info[2][1]
    query = f"SELECT CASE WHEN {bodo_fn_call} = {retval_1} THEN {retval_2} ELSE {bodo_fn_call} END FROM table1"

    if bodo_fn_name in EQUIVALENT_SPARK_DT_FN_MAP:
        spark_fn_name = EQUIVALENT_SPARK_DT_FN_MAP[bodo_fn_name]
        spark_fn_call = f"{spark_fn_name}({arglistString})"
        spark_query = f"SELECT CASE WHEN {spark_fn_call} = {retval_1} THEN {retval_2} ELSE {spark_fn_call} END FROM table1"
    else:
        spark_query = None

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.skip("BS-263")
def test_get_format(get_format_str, dt_fn_dataframe, spark_info, memory_leak_check):
    query = f"SELECT DATE_FORMAT(timestamps, {get_format_str}) from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT A, GETDATE() from table1",
            id="no_case-just_getdate",
        ),
        pytest.param(
            "SELECT A, GETDATE() - interval '6' months from table1",
            id="no_case-minus_interval-month",
        ),
        pytest.param(
            "SELECT A, GETDATE() + interval '5' weeks from table1",
            id="no_case-plus_interval-week",
        ),
        pytest.param(
            "SELECT A, GETDATE() - interval '8 weeks' from table1",
            id="no_case-minus_interval-week-sf-syntax",
        ),
        pytest.param(
            "SELECT A, GETDATE() - interval '8' weeks from table1",
            id="no_case-minus_interval-week",
        ),
        pytest.param(
            "SELECT A, GETDATE() + interval '5' days from table1",
            id="no_case-plus_interval-day",
        ),
        pytest.param(
            "SELECT A, CASE WHEN EXTRACT(MONTH from GETDATE()) = A then 'y' ELSE 'n' END from table1",
            id="case",
        ),
    ],
)
def test_getdate(query, spark_info, memory_leak_check):
    """Tests the snowflake GETDATE() function"""
    spark_query = query.replace("GETDATE()", "CURRENT_DATE()")
    ctx = {
        "table1": pd.DataFrame(
            {"A": pd.Series(list(range(1, 13)), dtype=pd.Int32Dtype())}
        )
    }

    if query == "SELECT A, GETDATE() - interval '8 weeks' from table1":
        spark_query = "SELECT A, CURRENT_DATE() - interval '8' weeks from table1"

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        only_jit_1DVar=True,
    )


def compute_valid_times(start_time, timeout=3):
    """Computes a list of valid days, hours, and minutes that are at most `timeout`
    minutes after a given pd.Timestamp. Used to test NOW, LOCALTIME, SYSDATE, and
    other equivalent functions.

    Given the timestamp "2022-12-31 23:58:58", this function returns as strings:
        valid_days = "31, 1"
        valid_hours = "23, 0"
        valid_minutes = "58, 59, 0"

    Args
        start_time [pd.Timestamp]: Starting timestamp, usually pd.Timestamp.now().
        timeout [int]: Max number of minutes that the test is expected to take.

    Returns: (valid_days [str], valid_hours [str], valid_minutes [str])
    """
    valid_days = set()
    valid_hours = set()
    valid_minutes = set()

    for t in range(timeout + 1):
        current_time = start_time + pd.Timedelta(minutes=t)
        valid_days.add(str(current_time.day))
        valid_hours.add(str(current_time.hour))
        valid_minutes.add(str(current_time.minute))

    valid_days = ", ".join(valid_days)
    valid_hours = ", ".join(valid_hours)
    valid_minutes = ", ".join(valid_minutes)
    return valid_days, valid_hours, valid_minutes


@pytest.fixture(
    params=[
        "CURRENT_TIMESTAMP",
        pytest.param("GETDATE", marks=pytest.mark.slow),
        pytest.param("LOCALTIMESTAMP", marks=pytest.mark.slow),
        pytest.param("SYSTIMESTAMP", marks=pytest.mark.slow),
        pytest.param("NOW", marks=pytest.mark.slow),
    ]
)
def now_equiv_fns(request):
    return request.param


def test_now_equivalents_cols(basic_df, now_equiv_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the current timestamp,
    without timezone info from the Snowflake Catalog.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  DATE_TRUNC('DAY', {now_equiv_fns}()) AS date_trunc, "
        f"  EXTRACT(DAY from {now_equiv_fns}()) IN ({valid_days}) AS is_valid_day, "
        f"  EXTRACT(HOUR from {now_equiv_fns}()) IN ({valid_hours}) AS is_valid_hour, "
        f"  EXTRACT(MINUTE from {now_equiv_fns}()) IN ({valid_minutes}) AS is_valid_minute "
        f"FROM table1"
    )
    py_output = pd.DataFrame(
        {
            "A": basic_df["table1"]["A"],
            "date_trunc": current_time.normalize(),
            "is_valid_day": True,
            "is_valid_hour": True,
            "is_valid_minute": True,
        }
    )

    check_query(query, basic_df, None, expected_output=py_output, check_dtype=False)


def test_now_equivalents_case(now_equiv_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the current timestamp in case,
    without timezone info from the Snowflake Catalog.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  CASE WHEN A THEN DATE_TRUNC('DAY', {now_equiv_fns}()) END AS date_trunc, "
        f"  CASE WHEN A THEN EXTRACT(DAY from {now_equiv_fns}()) IN ({valid_days}) END AS is_valid_day, "
        f"  CASE WHEN A THEN EXTRACT(HOUR from {now_equiv_fns}()) IN ({valid_hours}) END AS is_valid_hour, "
        f"  CASE WHEN A THEN EXTRACT(MINUTE from {now_equiv_fns}()) IN ({valid_minutes}) END AS is_valid_minute "
        f"FROM table1"
    )

    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    ctx = {"table1": df}
    D = pd.Series(current_time.normalize(), index=np.arange(len(df)))
    D[~df.A] = None
    S = pd.Series(True, index=np.arange(len(df)))
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "date_trunc": D,
            "is_valid_day": S,
            "is_valid_hour": S,
            "is_valid_minute": S,
        }
    )
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.fixture(
    params=[
        "LOCALTIME",
        pytest.param("CURRENT_TIME", marks=pytest.mark.slow),
    ]
)
def localtime_equiv_fns(request):
    return request.param


def test_localtime_equivalents_cols(basic_df, localtime_equiv_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the local time,
    without timezone info from the Snowflake Catalog.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    _, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  EXTRACT(HOUR from {localtime_equiv_fns}()) IN ({valid_hours}) AS is_valid_hour, "
        f"  EXTRACT(MINUTE from {localtime_equiv_fns}()) IN ({valid_minutes}) AS is_valid_minute "
        f"FROM table1"
    )
    py_output = pd.DataFrame(
        {
            "A": basic_df["table1"]["A"],
            "is_valid_hour": True,
            "is_valid_minute": True,
        }
    )

    check_query(query, basic_df, None, expected_output=py_output, check_dtype=False)


def test_localtime_equivalents_case(localtime_equiv_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the local time in case,
    without timezone info from the Snowflake Catalog.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    _, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  CASE WHEN A THEN EXTRACT(HOUR from {localtime_equiv_fns}()) IN ({valid_hours}) END AS is_valid_hour, "
        f"  CASE WHEN A THEN EXTRACT(MINUTE from {localtime_equiv_fns}()) IN ({valid_minutes}) END AS is_valid_minute "
        f"FROM table1"
    )

    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    ctx = {"table1": df}
    S = pd.Series(True, index=np.arange(len(df)))
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "is_valid_hour": S,
            "is_valid_minute": S,
        }
    )
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.fixture(
    params=[
        "UTC_TIMESTAMP",
        pytest.param("SYSDATE", marks=pytest.mark.slow),
    ]
)
def sysdate_equiv_fns(request):
    return request.param


def test_sysdate_equivalents_cols(
    basic_df, sysdate_equiv_fns, spark_info, memory_leak_check
):
    """
    Tests the group of equivalent functions which return the UTC timestamp.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  DATE_TRUNC('DAY', {sysdate_equiv_fns}()) AS date_trunc, "
        f"  EXTRACT(DAY from {sysdate_equiv_fns}()) IN ({valid_days}) AS is_valid_day, "
        f"  EXTRACT(HOUR from {sysdate_equiv_fns}()) IN ({valid_hours}) AS is_valid_hour, "
        f"  EXTRACT(MINUTE from {sysdate_equiv_fns}()) IN ({valid_minutes}) AS is_valid_minute "
        f"FROM table1"
    )
    py_output = pd.DataFrame(
        {
            "A": basic_df["table1"]["A"],
            "date_trunc": current_time.normalize(),
            "is_valid_day": True,
            "is_valid_hour": True,
            "is_valid_minute": True,
        }
    )

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


def test_sysdate_equivalents_case(sysdate_equiv_fns, spark_info, memory_leak_check):
    """
    Tests the group of equivalent functions which return the UTC timestamp in case.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  CASE WHEN A THEN DATE_TRUNC('DAY', {sysdate_equiv_fns}()) END AS date_trunc, "
        f"  CASE WHEN A THEN EXTRACT(DAY from {sysdate_equiv_fns}()) IN ({valid_days}) END AS is_valid_day, "
        f"  CASE WHEN A THEN EXTRACT(HOUR from {sysdate_equiv_fns}()) IN ({valid_hours}) END AS is_valid_hour, "
        f"  CASE WHEN A THEN EXTRACT(MINUTE from {sysdate_equiv_fns}()) IN ({valid_minutes}) END AS is_valid_minute "
        f"FROM table1"
    )

    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    ctx = {"table1": df}
    D = pd.Series(current_time.normalize(), index=np.arange(len(df)))
    D[~df.A] = None
    S = pd.Series(True, index=np.arange(len(df)))
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "date_trunc": D,
            "is_valid_day": S,
            "is_valid_hour": S,
            "is_valid_minute": S,
        }
    )

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.slow
def test_utc_date(basic_df, spark_info, memory_leak_check):
    """tests utc_date"""

    query = f"SELECT A, EXTRACT(day from UTC_DATE()), (EXTRACT(HOUR from UTC_DATE()) + EXTRACT(MINUTE from UTC_DATE()) + EXTRACT(SECOND from UTC_DATE()) ) = 0  from table1"
    expected_output = pd.DataFrame(
        {
            "unknown_name1": basic_df["table1"]["A"],
            "unknown_name2": pd.Timestamp.now(tz="UTC").day,
            "unknown_name5": True,
        }
    )
    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.fixture(
    params=
    # check the values for which the format strings are the same
    [
        (x, x)
        for x in [
            "%a",
            "%b",
            "%f",
            "%H",
            "%j",
            "%m",
            "%p",
            "%d",
            "%Y",
            "%y",
            "%U",
            "%S",
        ]
    ]
    +
    # check the values for which the format strings have a 1 to 1
    [
        ("%i", "%M"),
        ("%M", "%B"),
        ("%r", "%X %p"),
        ("%s", "%S"),
        ("%T", "%X"),
        ("%u", "%W"),
        ('% %a %\\, %%a, %%, %%%%, "%", %', ' %a \\, %%a, %%, %%%%, "", %'),
    ]
    # TODO: add addition format characters when/if they become supported
)
def python_mysql_dt_format_strings(request):
    """returns a tuple of python mysql string, and the equivalent python format string"""
    return request.param


def test_date_format(
    spark_info, dt_fn_dataframe, python_mysql_dt_format_strings, memory_leak_check
):
    """tests the date format function"""

    mysql_format_str = python_mysql_dt_format_strings[0]
    python_format_str = python_mysql_dt_format_strings[1]

    query = f"SELECT DATE_FORMAT(timestamps, '{mysql_format_str}') from table1"
    expected_output = pd.DataFrame(
        {
            "unkown_name": dt_fn_dataframe["table1"]["timestamps"].dt.strftime(
                python_format_str
            )
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_microseconds(spark_info, dt_fn_dataframe, memory_leak_check):
    """spark has no equivalent MICROSECOND function, so we need to test it manually"""

    query1 = "SELECT MICROSECOND(timestamps) as microsec_time from table1"
    query2 = "SELECT CASE WHEN MICROSECOND(timestamps) > 1 THEN MICROSECOND(timestamps) ELSE -1 END as microsec_time from table1"

    expected_output = pd.DataFrame(
        {"microsec_time": dt_fn_dataframe["table1"]["timestamps"].dt.microsecond}
    )

    check_query(
        query1,
        dt_fn_dataframe,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_microsecond(tz_aware_df, memory_leak_check):
    """simplest test for microsecond on timezone aware data"""
    query = "SELECT MICROSECOND(A) as microsec_time from table1"
    expected_output = pd.DataFrame(
        {"microsec_time": tz_aware_df["table1"]["A"].dt.microsecond}
    )

    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_microsecond_case(tz_aware_df, memory_leak_check):
    """test for microsecond within case statement on timezone aware data"""
    query = "SELECT CASE WHEN MICROSECOND(A) > 1 THEN MICROSECOND(A) ELSE -1 END as microsec_time from table1"

    micro_series = tz_aware_df["table1"]["A"].dt.microsecond
    micro_series[micro_series <= 1] = -1

    expected_output = pd.DataFrame({"microsec_time": micro_series})

    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_dayname_cols(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests the dayname function on column inputs. Needed since the equivalent function has different syntax"""
    query = "SELECT DAYNAME(timestamps) from table1"
    spark_query = "SELECT DATE_FORMAT(timestamps, 'EEEE') from table1"
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_dayname_scalars(basic_df, spark_info, memory_leak_check):
    """tests the dayname function on scalar inputs. Needed since the equivalent function has different syntax"""

    # since dayname is a fn we defined, don't need to worry about calcite performing optimizations
    # Use basic_df so the input is expanded and we don't have to worry about empty arrays
    query = "SELECT A, DAYNAME(TIMESTAMP '2021-03-03'), DAYNAME(TIMESTAMP '2021-03-13'), DAYNAME(TIMESTAMP '2021-03-01') from table1"
    spark_query = "SELECT A, DATE_FORMAT('2021-03-03', 'EEEE'), DATE_FORMAT('2021-03-13', 'EEEE'), DATE_FORMAT('2021-03-01', 'EEEE') from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_dayname_date_cols(date_df, memory_leak_check):
    """tests the dayname function on column inputs of date objects."""
    query = "SELECT DAYNAME(A) from table1"
    outputs = pd.DataFrame({"output": date_df["table1"]["A"].map(day_name_func)})

    check_query(
        query,
        date_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


def day_name_func(date):
    if date is None:
        return None
    dows = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return dows[date.weekday()]


def test_dayname_date_scalars(basic_df, spark_info, memory_leak_check):
    """tests the dayname function on scalar inputs of date objects."""

    # since dayname is a fn we defined, don't need to worry about calcite performing optimizations
    # Use basic_df so the input is expanded and we don't have to worry about empty arrays
    query = f"SELECT DAYNAME(TO_DATE('2021-03-03')), DAYNAME(TO_DATE('2021-05-13')), DAYNAME(TO_DATE('2021-07-03'))"
    outputs = pd.DataFrame({"A": ["Wednesday"], "B": ["Thursday"], "C": ["Saturday"]})

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


@pytest.mark.parametrize(
    "fn_name", ["MONTHNAME", pytest.param("MONTH_NAME", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("wrap_case", [True, False])
def test_monthname_cols(
    fn_name, wrap_case, spark_info, dt_fn_dataframe, memory_leak_check
):
    """tests the monthname function on column inputs. Needed since the equivalent function has different syntax"""

    if wrap_case:
        query = f"SELECT CASE WHEN timestamps IS NULL THEN {fn_name}(timestamps) else {fn_name}(timestamps) END FROM table1"
    else:
        query = f"SELECT {fn_name}(timestamps) from table1"

    spark_query = "SELECT DATE_FORMAT(timestamps, 'MMMM') from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize("fn_name", ["MONTHNAME", "MONTH_NAME"])
def test_monthname_scalars(fn_name, basic_df, spark_info, memory_leak_check):
    """tests the monthname function on scalar inputs. Needed since the equivalent function has different syntax"""

    # since monthname is a fn we defined, don't need to worry about calcite performing optimizations
    query = f"SELECT {fn_name}(TIMESTAMP '2021-03-03'), {fn_name}(TIMESTAMP '2021-03-13'), {fn_name}(TIMESTAMP '2021-03-01')"
    spark_query = "SELECT DATE_FORMAT('2021-03-03', 'MMMM'), DATE_FORMAT('2021-03-13', 'MMMM'), DATE_FORMAT('2021-03-01', 'MMMM')"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize(
    "fn_name", ["MONTHNAME", pytest.param("MONTH_NAME", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("wrap_case", [True, False])
def test_monthname_date_cols(fn_name, wrap_case, date_df, memory_leak_check):
    """tests the monthname function on column inputs of date objects."""

    if wrap_case:
        query = f"SELECT CASE WHEN A IS NULL THEN {fn_name}(A) else {fn_name}(A) END FROM table1"
    else:
        query = f"SELECT {fn_name}(A) from table1"

    outputs = pd.DataFrame({"output": date_df["table1"]["A"].map(month_name_func)})

    check_query(
        query,
        date_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


def month_name_func(date):
    if date is None:
        return None
    mons = [
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
    ]
    return mons[date.month - 1]


@pytest.mark.parametrize("fn_name", ["MONTHNAME", "MONTH_NAME"])
def test_monthname_date_scalars(fn_name, basic_df, memory_leak_check):
    """tests the monthname function on scalar inputs of date objects."""

    query = f"SELECT {fn_name}(DATE '2021-03-03'), {fn_name}(DATE '2021-05-13'), {fn_name}(DATE '2021-07-01')"
    outputs = pd.DataFrame({"A": ["March"], "B": ["May"], "C": ["July"]})

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


def test_make_date_cols(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests makedate on column values"""

    # TODO: fix null issues with make_date: https://bodo.atlassian.net/browse/BE-3640
    ctx_dict = {"table1": dt_fn_dataframe["table1"].fillna(method="bfill")}

    # Spark's make_date, takes three arguments: Y, M, D, where MYSQL's makedate is Y, D
    query = "SELECT makedate(valid_year_integers, positive_integers) from table1"
    spark_query = "SELECT DATE_ADD(MAKE_DATE(valid_year_integers, 1, 1), positive_integers-1) from table1"

    # spark requires certain the second argument of make_date to not be of type bigint,
    # but all pandas integer types are currently interpreted bigint when creating
    # a spark dataframe from a pandas dataframe. Therefore, we need to cast the spark table
    # to a valid type
    cols_to_cast = {"table1": [("positive_integers", "int")]}

    check_query(
        query,
        ctx_dict,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
    )


def test_make_date_scalar(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests makedate on scalar values"""

    query = "SELECT CASE WHEN makedate(valid_year_integers, positive_integers) > TIMESTAMP '2211-01-01' THEN TIMESTAMP '2000-01-01' ELSE makedate(valid_year_integers, positive_integers) END from table1"
    spark_query = "SELECT CASE WHEN DATE_ADD(make_date(valid_year_integers, 1, 1), positive_integers-1) > TIMESTAMP '2211-01-01' THEN TIMESTAMP '2000-01-01' ELSE DATE_ADD(make_date(valid_year_integers, 1, 1), positive_integers-1) END from table1"

    # spark requires certain the second argument of make_date to not be of type bigint,
    # but all pandas integer types are currently interpreted bigint when creating
    # a spark dataframe from a pandas dataframe. Therefore, we need to cast the spark table
    # to a valid type
    cols_to_cast = {"table1": [("positive_integers", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
    )


@pytest.mark.slow
@pytest.mark.skip("Currently, not checking for invalid year/day values")
def test_make_date_edgecases(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests makedate on edgecases"""

    query = "SELECT makedate(valid_year_integers, mixed_integers), makedate(positive_integers, positive_integers) from table1"
    spark_query = "SELECT DATE_ADD(make_date(valid_year_integers, 1, 1), mixed_integers), DATE_ADD(make_date(positive_integers, 1, 1), positive_integers) from table1"

    # spark requires certain arguments of make_date to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"table1": [("positive_integers", "int"), ("mixed_integers", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
    )


@pytest.fixture(
    params=[
        "MICROSECOND",
        "SECOND",
        "MINUTE",
        "HOUR",
        "DAY",
        "MONTH",
        "QUARTER",
        "YEAR",
    ]
)
def valid_extract_strings(request):
    return request.param


def test_extract_cols(
    spark_info, dt_fn_dataframe, valid_extract_strings, memory_leak_check
):
    query = f"SELECT EXTRACT({valid_extract_strings} from timestamps) from table1"

    # spark does not allow the microsecond argument for extract, and to compensate, the
    # second argument returns a float. Therefore, in these cases we need to manually
    # generate the expected output
    if valid_extract_strings == "SECOND":
        expected_output = pd.DataFrame(
            {"unkown_name": dt_fn_dataframe["table1"]["timestamps"].dt.second}
        )
    elif valid_extract_strings == "MICROSECOND":
        expected_output = pd.DataFrame(
            {"unkown_name": dt_fn_dataframe["table1"]["timestamps"].dt.microsecond}
        )
    else:
        expected_output = None

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_extract_scalars(
    spark_info, dt_fn_dataframe, valid_extract_strings, memory_leak_check
):
    query = f"SELECT CASE WHEN EXTRACT({valid_extract_strings} from timestamps) < 0 THEN -1 ELSE EXTRACT({valid_extract_strings} from timestamps) END from table1"

    # spark does not allow the microsecond argument for extract, and to compensate, the
    # second argument returns a float. Therefore, in these cases we need to manually
    # generate the expected output
    if valid_extract_strings == "SECOND":
        expected_output = pd.DataFrame(
            {"unkown_name": dt_fn_dataframe["table1"]["timestamps"].dt.second}
        )
    elif valid_extract_strings == "MICROSECOND":
        expected_output = pd.DataFrame(
            {"unkown_name": dt_fn_dataframe["table1"]["timestamps"].dt.microsecond}
        )
    else:
        expected_output = None

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "query_fmt, answer",
    [
        pytest.param(
            "DATE_PART({!r}, col_dt)",
            pd.DataFrame(
                {
                    "YR": [None, 2010, 2011, 2012, 2013],
                    "QU": [None, 1, 1, 2, 4],
                    "MO": [None, 1, 2, 5, 10],
                    "WE": [None, 2, 8, 19, 43],
                    "DA": [None, 17, 26, 9, 22],
                    "DW": [None, 0, 6, 3, 2],
                    "HR": [None, 0, 3, 16, 5],
                    "MI": [None, 0, 36, 43, 32],
                    "SE": [None, 0, 1, 16, 21],
                }
            ),
            id="vector-no_case",
        ),
        pytest.param(
            "CASE WHEN EXTRACT(YEAR from col_dt) = 2013 THEN NULL else DATE_PART({!r}, col_dt) END",
            pd.DataFrame(
                {
                    "YR": [None, 2010, 2011, 2012, None],
                    "QU": [None, 1, 1, 2, None],
                    "MO": [None, 1, 2, 5, None],
                    "WE": [None, 2, 8, 19, None],
                    "DA": [None, 17, 26, 9, None],
                    "DW": [None, 0, 6, 3, None],
                    "HR": [None, 0, 3, 16, None],
                    "MI": [None, 0, 36, 43, None],
                    "SE": [None, 0, 1, 16, None],
                }
            ),
            id="vector-case",
        ),
    ],
)
def test_date_part(query_fmt, answer, spark_info, memory_leak_check):
    selects = []
    for unit in ["year", "q", "mons", "wk", "dayofmonth", "dow", "hrs", "min", "s"]:
        selects.append(query_fmt.format(unit))
    query = f"SELECT {', '.join(selects)} FROM table1"

    ctx = {
        "table1": pd.DataFrame(
            {
                "col_dt": pd.Series(
                    [
                        None,
                        pd.Timestamp("2010-01-17"),
                        pd.Timestamp("2011-02-26 03:36:01"),
                        pd.Timestamp("2012-05-09 16:43:16.123456"),
                        pd.Timestamp("2013-10-22 05:32:21.987654321"),
                    ]
                )
            }
        )
    }

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "query_fmt",
    [
        pytest.param(
            "DATE_PART({!r}, A) AS my_{}",
            id="vector-no_case",
        ),
    ],
)
@pytest.mark.tz_aware
def test_tz_aware_date_part(tz_aware_df, query_fmt, spark_info, memory_leak_check):
    selects = []
    for unit in ["year", "q", "mons", "wk", "dayofmonth", "hrs", "min", "s"]:
        selects.append(query_fmt.format(unit, unit))
    query = f"SELECT {', '.join(selects)} FROM table1"
    df = tz_aware_df["table1"]
    py_output = pd.DataFrame(
        {
            "my_year": df.A.dt.year,
            "my_q": df.A.dt.quarter,
            "my_mons": df.A.dt.month,
            "my_wk": df.A.dt.weekofyear,
            "my_dayofmonth": df.A.dt.day,
            "my_hrs": df.A.dt.hour,
            "my_min": df.A.dt.minute,
            "my_s": df.A.dt.second,
        }
    )

    check_query(
        query,
        tz_aware_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


def make_spark_interval(interval_str, value):
    """simple helper function that takes a value and an timeunit str, and returns a spark interval"""
    if interval_str == "MICROSECOND":
        return f"MAKE_INTERVAL(0, 0, 0, 0, 0, 0, 0.{value})"
    elif interval_str == "SECOND":
        return f"MAKE_INTERVAL(0, 0, 0, 0, 0, 0, {value})"
    elif interval_str == "MINUTE":
        return f"MAKE_INTERVAL(0, 0, 0, 0, 0, {value}, 0)"
    elif interval_str == "HOUR":
        return f"MAKE_INTERVAL(0, 0, 0, 0, {value}, 0, 0)"
    elif interval_str == "DAY":
        return f"MAKE_INTERVAL(0, 0, 0, {value}, 0, 0, 0)"
    elif interval_str == "WEEK":
        return f"MAKE_INTERVAL(0, 0, {value}, 0, 0, 0, 0)"
    elif interval_str == "MONTH":
        return f"MAKE_INTERVAL(0, {value}, 0, 0, 0, 0, 0)"
    elif interval_str == "YEAR":
        return f"MAKE_INTERVAL({value}, 0, 0, 0, 0, 0, 0)"
    else:
        raise Exception(f"Error, need a case for timeunit: {interval_str}")


@pytest.fixture
def dateadd_df():
    """Returns the context used by test_snowflake_dateadd"""
    return {
        "table1": pd.DataFrame(
            {
                "col_int": pd.Series([10, 1, None, -10, 100], dtype=pd.Int32Dtype()),
                "col_dt": pd.Series(
                    [
                        None,
                        pd.Timestamp("2013-10-27"),
                        pd.Timestamp("2015-4-1 12:00:15"),
                        pd.Timestamp("2020-2-3 05:15:12.501"),
                        pd.Timestamp("2021-12-13 23:15:06.025999500"),
                    ]
                ),
            }
        )
    }


@pytest.fixture(
    params=[
        pytest.param(
            (
                "DATEADD({!r}, col_int, col_dt)",
                ["year", "month", "week", "day"],
                pd.DataFrame(
                    {
                        "year": [
                            None,
                            pd.Timestamp("2014-10-27 00:00:00"),
                            None,
                            pd.Timestamp("2010-02-03 05:15:12.501000"),
                            pd.Timestamp("2121-12-13 23:15:06.025999500"),
                        ],
                        "month": [
                            None,
                            pd.Timestamp("2013-11-27 00:00:00"),
                            None,
                            pd.Timestamp("2019-04-03 05:15:12.501000"),
                            pd.Timestamp("2030-04-13 23:15:06.025999500"),
                        ],
                        "week": [
                            None,
                            pd.Timestamp("2013-11-03 00:00:00"),
                            None,
                            pd.Timestamp("2019-11-25 05:15:12.501000"),
                            pd.Timestamp("2023-11-13 23:15:06.025999500"),
                        ],
                        "day": [
                            None,
                            pd.Timestamp("2013-10-28 00:00:00"),
                            None,
                            pd.Timestamp("2020-01-24 05:15:12.501000"),
                            pd.Timestamp("2022-03-23 23:15:06.025999500"),
                        ],
                    }
                ),
            ),
            id="vector-date_units",
        ),
        pytest.param(
            (
                "TIMEADD({!r}, col_int, col_dt)",
                ["hour", "minute", "second"],
                pd.DataFrame(
                    {
                        "hour": [
                            None,
                            pd.Timestamp("2013-10-27 01:00:00"),
                            None,
                            pd.Timestamp("2020-02-02 19:15:12.501000"),
                            pd.Timestamp("2021-12-18 03:15:06.025999500"),
                        ],
                        "minute": [
                            None,
                            pd.Timestamp("2013-10-27 00:01:00"),
                            None,
                            pd.Timestamp("2020-02-03 05:05:12.501000"),
                            pd.Timestamp("2021-12-14 00:55:06.025999500"),
                        ],
                        "second": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:01"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:02.501000"),
                            pd.Timestamp("2021-12-13 23:16:46.025999500"),
                        ],
                    }
                ),
            ),
            id="vector-time_units",
        ),
        pytest.param(
            (
                "TIMESTAMPADD({!r}, col_int, col_dt)",
                ["millisecond", "microsecond", "nanosecond"],
                pd.DataFrame(
                    {
                        "millisecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.001000"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:12.491000"),
                            pd.Timestamp("2021-12-13 23:15:06.125999500"),
                        ],
                        "microsecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000001"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:12.500990"),
                            pd.Timestamp("2021-12-13 23:15:06.026099500"),
                        ],
                        "nanosecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000000001"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:12.500999990"),
                            pd.Timestamp("2021-12-13 23:15:06.025999600"),
                        ],
                    }
                ),
            ),
            id="vector-subsecond_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!r}, -25, col_dt) END",
                ["year", "month", "week", "day"],
                pd.DataFrame(
                    {
                        "year": [
                            None,
                            pd.Timestamp("1988-10-27 00:00:00"),
                            pd.Timestamp("1990-04-01 12:00:15"),
                            None,
                            pd.Timestamp("1996-12-13 23:15:06.025999500"),
                        ],
                        "month": [
                            None,
                            pd.Timestamp("2011-09-27 00:00:00"),
                            pd.Timestamp("2013-03-01 12:00:15"),
                            None,
                            pd.Timestamp("2019-11-13 23:15:06.025999500"),
                        ],
                        "week": [
                            None,
                            pd.Timestamp("2013-05-05 00:00:00"),
                            pd.Timestamp("2014-10-08 12:00:15"),
                            None,
                            pd.Timestamp("2021-06-21 23:15:06.025999500"),
                        ],
                        "day": [
                            None,
                            pd.Timestamp("2013-10-02 00:00:00"),
                            pd.Timestamp("2015-03-07 12:00:15"),
                            None,
                            pd.Timestamp("2021-11-18 23:15:06.025999500"),
                        ],
                    }
                ),
            ),
            id="case-date_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else TIMESTAMPADD({!r}, -25, col_dt) END",
                ["hour", "minute", "second"],
                pd.DataFrame(
                    {
                        "hour": [
                            None,
                            pd.Timestamp("2013-10-25 23:00:00"),
                            pd.Timestamp("2015-03-31 11:00:15"),
                            None,
                            pd.Timestamp("2021-12-12 22:15:06.025999500"),
                        ],
                        "minute": [
                            None,
                            pd.Timestamp("2013-10-26 23:35:00"),
                            pd.Timestamp("2015-04-01 11:35:15"),
                            None,
                            pd.Timestamp("2021-12-13 22:50:06.025999500"),
                        ],
                        "second": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:35"),
                            pd.Timestamp("2015-04-01 11:59:50"),
                            None,
                            pd.Timestamp("2021-12-13 23:14:41.025999500"),
                        ],
                    }
                ),
            ),
            id="case-time_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!r}, -25, col_dt) END",
                ["millisecond", "microsecond", "nanosecond"],
                pd.DataFrame(
                    {
                        "millisecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.975000"),
                            pd.Timestamp("2015-04-01 12:00:14.975000"),
                            None,
                            pd.Timestamp("2021-12-13 23:15:06.000999500"),
                        ],
                        "microsecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999975"),
                            pd.Timestamp("2015-04-01 12:00:14.999975"),
                            None,
                            pd.Timestamp("2021-12-13 23:15:06.025974500"),
                        ],
                        "nanosecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999999975"),
                            pd.Timestamp("2015-04-01 12:00:14.999999975"),
                            None,
                            pd.Timestamp("2021-12-13 23:15:06.025999475"),
                        ],
                    }
                ),
            ),
            id="case-subsecond_units",
        ),
    ]
)
def dateadd_queries(request):
    """Returns specifications used for queries in test_snowflake_dateadd in
    the following format:
    - The query format that the units are injected into
    - The list of units used for this test
    - The outputs for this query when used on dateadd_df"""
    return request.param


def test_snowflake_dateadd(dateadd_df, dateadd_queries, memory_leak_check):
    """Tests the Snowflake version of DATEADD with inputs (unit, amount, dt_val).
    Currently takes in the unit as a scalar string instead of a DT unit literal.
    Does not currently support quarter, or check any of the alternative
    abbreviations of these units."""
    query_fmt, units, answers = dateadd_queries
    selects = []
    for unit in units:
        selects.append(query_fmt.format(unit))
    query = "SELECT " + ", ".join(selects) + " FROM table1"

    check_query(
        query,
        dateadd_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answers,
        only_jit_1DVar=True,
    )


@pytest.fixture
def dateadd_date_df():
    """Returns the context used by test_snowflake_dateadd_date"""
    return {
        "table1": pd.DataFrame(
            {
                "col_int": pd.Series([10, 1, None, -10, 100], dtype=pd.Int32Dtype()),
                "col_dt": pd.Series(
                    [
                        None,
                        datetime.date(2013, 10, 27),
                        datetime.date(2015, 4, 1),
                        datetime.date(2020, 2, 3),
                        datetime.date(2021, 12, 13),
                    ]
                ),
            }
        )
    }


@pytest.fixture(
    params=[
        pytest.param(
            (
                "DATEADD({!r}, col_int, col_dt)",
                ["year", "quarter", "month", "week", "day"],
                pd.DataFrame(
                    {
                        "year": [
                            None,
                            datetime.date(2014, 10, 27),
                            None,
                            datetime.date(2010, 2, 3),
                            datetime.date(2121, 12, 13),
                        ],
                        "quarter": [
                            None,
                            datetime.date(2014, 1, 27),
                            None,
                            datetime.date(2017, 8, 3),
                            datetime.date(2046, 12, 13),
                        ],
                        "month": [
                            None,
                            datetime.date(2013, 11, 27),
                            None,
                            datetime.date(2019, 4, 3),
                            datetime.date(2030, 4, 13),
                        ],
                        "week": [
                            None,
                            datetime.date(2013, 11, 3),
                            None,
                            datetime.date(2019, 11, 25),
                            datetime.date(2023, 11, 13),
                        ],
                        "day": [
                            None,
                            datetime.date(2013, 10, 28),
                            None,
                            datetime.date(2020, 1, 24),
                            datetime.date(2022, 3, 23),
                        ],
                    }
                ),
            ),
            id="vector-date_units",
        ),
        pytest.param(
            (
                "TIMEADD({!r}, col_int, col_dt)",
                ["hour", "minute", "second"],
                pd.DataFrame(
                    {
                        "hour": [
                            None,
                            pd.Timestamp("2013-10-27 01:00:00"),
                            None,
                            pd.Timestamp("2020-02-02 14:00:00"),
                            pd.Timestamp("2021-12-17 04:00:00"),
                        ],
                        "minute": [
                            None,
                            pd.Timestamp("2013-10-27 00:01:00"),
                            None,
                            pd.Timestamp("2020-02-02 23:50:00"),
                            pd.Timestamp("2021-12-13 01:40:00"),
                        ],
                        "second": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:01"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:50"),
                            pd.Timestamp("2021-12-13 00:01:40"),
                        ],
                    }
                ),
            ),
            id="vector-time_units",
        ),
        pytest.param(
            (
                "TIMESTAMPADD({!r}, col_int, col_dt)",
                ["millisecond", "microsecond", "nanosecond"],
                pd.DataFrame(
                    {
                        "millisecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.001"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:59.990"),
                            pd.Timestamp("2021-12-13 00:00:00.100"),
                        ],
                        "microsecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000001"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:59.999990"),
                            pd.Timestamp("2021-12-13 00:00:00.000100"),
                        ],
                        "nanosecond": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000000001"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:59.999999990"),
                            pd.Timestamp("2021-12-13 00:00:00.000000100"),
                        ],
                    }
                ),
            ),
            id="vector-subsecond_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!r}, -25, col_dt) END",
                ["year", "quarter", "month", "week", "day"],
                pd.DataFrame(
                    {
                        "year": [
                            None,
                            datetime.date(1988, 10, 27),
                            datetime.date(1990, 4, 1),
                            None,
                            datetime.date(1996, 12, 13),
                        ],
                        "quarter": [
                            None,
                            datetime.date(2007, 7, 27),
                            datetime.date(2009, 1, 1),
                            None,
                            datetime.date(2015, 9, 13),
                        ],
                        "month": [
                            None,
                            datetime.date(2011, 9, 27),
                            datetime.date(2013, 3, 1),
                            None,
                            datetime.date(2019, 11, 13),
                        ],
                        "week": [
                            None,
                            datetime.date(2013, 5, 5),
                            datetime.date(2014, 10, 8),
                            None,
                            datetime.date(2021, 6, 21),
                        ],
                        "day": [
                            None,
                            datetime.date(2013, 10, 2),
                            datetime.date(2015, 3, 7),
                            None,
                            datetime.date(2021, 11, 18),
                        ],
                    }
                ),
            ),
            id="case-date_units",
            marks=pytest.mark.skip(reason="TODO: support date in CASE statements"),
            # Calcite will set the return type to timestamp when parsing the case statement,
            # so case statements with time units work but those with date units don't
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else TIMESTAMPADD({!r}, -25, col_dt) END",
                ["hour", "minute", "second"],
                pd.DataFrame(
                    {
                        "hour": [
                            None,
                            pd.Timestamp("2013-10-25 23:00:00"),
                            pd.Timestamp("2015-03-30 23:00:00"),
                            None,
                            pd.Timestamp("2021-12-11 23:00:00"),
                        ],
                        "minute": [
                            None,
                            pd.Timestamp("2013-10-26 23:35:00"),
                            pd.Timestamp("2015-03-31 23:35:00"),
                            None,
                            pd.Timestamp("2021-12-12 23:35:00"),
                        ],
                        "second": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:35"),
                            pd.Timestamp("2015-03-31 23:59:35"),
                            None,
                            pd.Timestamp("2021-12-12 23:59:35"),
                        ],
                    }
                ),
            ),
            id="case-time_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!r}, -25, col_dt) END",
                ["millisecond", "microsecond", "nanosecond"],
                pd.DataFrame(
                    {
                        "millisecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.975"),
                            pd.Timestamp("2015-03-31 23:59:59.975"),
                            None,
                            pd.Timestamp("2021-12-12 23:59:59.975"),
                        ],
                        "microsecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999975"),
                            pd.Timestamp("2015-03-31T23:59:59.999975"),
                            None,
                            pd.Timestamp("2021-12-12T23:59:59.999975"),
                        ],
                        "nanosecond": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999999975"),
                            pd.Timestamp("2015-03-31 23:59:59.999999975"),
                            None,
                            pd.Timestamp("2021-12-12T23:59:59.999999975"),
                        ],
                    }
                ),
            ),
            id="case-subsecond_units",
        ),
    ]
)
def dateadd_date_queries(request):
    """Returns specifications used for queries in test_snowflake_dateadd_date in
    the following format:
    - The query format that the units are injected into
    - The list of units used for this test
    - The outputs for this query when used on dateadd_df"""
    return request.param


def test_snowflake_dateadd_date(dateadd_date_df, dateadd_date_queries, memory_leak_check):
    """
    Tests the Snowflake version of DATEADD/TIMEADD/TIMESTAMPADD with date inputs for dt_val.
    Currently takes in the unit as a scalar string instead of a DT unit literal.
    """
    query_fmt, units, answers = dateadd_date_queries
    selects = []
    for unit in units:
        selects.append(query_fmt.format(unit))
    query = "SELECT " + ", ".join(selects) + " FROM table1"

    with bodosql_use_date_type():
        check_query(
            query,
            dateadd_date_df,
            None,
            check_names=False,
            expected_output=answers,
            only_jit_1DVar=True,
        )


@pytest.mark.parametrize(
    "time_zone, has_case",
    [
        pytest.param(None, False, id="no_tz-no_case"),
        pytest.param(None, True, id="no_tz-with_case", marks=pytest.mark.slow),
        pytest.param("US/Pacific", False, id="with_tz-no_case", marks=pytest.mark.slow),
        pytest.param("Europe/Berlin", True, id="with_tz-with_case"),
    ],
)
def test_snowflake_quarter_dateadd(time_zone, has_case, memory_leak_check):
    """Followup to test_snowflake_dateadd but for the QUARTER unit"""
    if has_case:
        query = "SELECT DT, CASE WHEN YEAR(DT) < 2000 THEN NULL ELSE DATEADD('quarter', N, DT) END FROM table1"
    else:
        query = "SELECT DT, DATEADD('quarter', N, DT) FROM table1"
    input_strings = [
        "2020-01-14 00:00:00.000",
        "2020-02-29 07:00:00.000",
        "2020-04-15 14:00:00.000",
        "2020-05-31 21:00:00.000",
        "2020-07-17 04:00:00.000",
        "2020-09-01 11:00:00.000",
        "2020-10-17 18:00:00.000",
        "2020-12-03 01:00:00.000",
        "2021-01-18 08:00:00.000",
        "2021-03-05 15:00:00.000",
        "2021-04-20 22:00:00.000",
        "2021-06-06 05:00:00.000",
        "2021-07-22 12:00:00.000",
        "2021-09-06 19:00:00.000",
        "2021-10-23 02:00:00.000",
    ]
    # Answers obtained from Snowflake. The correct bebhavior is for the date to
    # advance by exactly 3 months per quarted added without the day of month or
    # the time of day changing. If that day of month does not exist for the
    # new month, then it is rounded back to the last day of the new month.
    # For example, adding 4 quarters to Feb. 29 of a leap year will get you
    # to Feb. 28 of the next year.
    answer_strings = [
        "2016-07-14 00:00:00.000",
        "2023-02-28 07:00:00.000",
        "2022-01-15 14:00:00.000",
        "2020-11-30 21:00:00.000",
        "2019-10-17 04:00:00.000",
        "2018-09-01 11:00:00.000",
        "2017-07-17 18:00:00.000",
        "2024-03-03 01:00:00.000",
        "2023-01-18 08:00:00.000",
        "2021-12-05 15:00:00.000",
        "2020-10-20 22:00:00.000",
        "2019-09-06 05:00:00.000",
        "2018-07-22 12:00:00.000",
        "2025-03-06 19:00:00.000",
        "2024-01-23 02:00:00.000",
    ]
    df = pd.DataFrame(
        {
            "DT": [pd.Timestamp(ts, tz=time_zone) for ts in input_strings],
            "N": [-14, 12, 7, 2, -3, -8, -13, 13, 8, 3, -2, -7, -12, 14, 9],
        }
    )
    answer = pd.DataFrame(
        {0: df.DT, 1: [pd.Timestamp(ts, tz=time_zone) for ts in answer_strings]}
    )

    check_query(
        query,
        {"table1": df},
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
    )


@pytest.fixture(
    params=[
        pytest.param(
            ("'mm'", 158, "2035-5-12 20:30:00", "2036-1-6 0:45:00"), id="month"
        ),
        pytest.param(("'d'", 1, "2022-3-13 20:30:00", "2022-11-7 0:45:00"), id="day"),
        pytest.param(("'h'", 8, "2022-3-13 4:30:00", "2022-11-6 8:45:00"), id="hour"),
    ]
)
def tz_dateadd_args(request):
    """Parametrization of several input values used by tz_dateadd_data to
    construct its outputs for each timezone. Outputs 4-tuples in the
    following format:

    unit: the string literal provided to DATEADD to control which unit
    is added to the timestamp

    amt: the amount of the unit to add to the timestamp

    springRes: the result of adding that amount of the unit to the timestamp
    corresponding to shortly before the Spring daylight savings switch

    fallRes: the result of adding that amount of the unit to the timestamp
    corresponding to shortly before the Fall daylight savings switch
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param("US/Pacific", id="pacific"),
        pytest.param("GMT", id="gmt"),
        pytest.param("US/Eastern", id="eastern", marks=pytest.mark.slow),
    ]
)
def tz_dateadd_data(request, tz_dateadd_args):
    """Returns the context, calculations and answers used by test_snowflake_tz_dateadd.
    The timestamps correspond to a short period before the a daylight savings
    time switch, and the units provided ensure that the values often
    jump across a switch."""
    unit, amt, springRes, fallRes = tz_dateadd_args
    ctx = {
        "table1": pd.DataFrame(
            {
                "dt_col": pd.Series(
                    [pd.Timestamp("2022-3-12 20:30:00", tz=request.param)] * 3
                    + [None]
                    + [pd.Timestamp("2022-11-6 0:45:00", tz=request.param)] * 3
                ),
                "bool_col": pd.Series([True] * 7),
            }
        )
    }
    calculation = f"DATEADD({unit}, {amt}, dt_col)"
    answer = pd.DataFrame(
        {
            0: pd.Series(
                [pd.Timestamp(springRes, tz=request.param)] * 3
                + [None]
                + [pd.Timestamp(fallRes, tz=request.param)] * 3
            )
        }
    )
    return ctx, calculation, answer


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case"),
    ],
)
def test_snowflake_tz_dateadd(tz_dateadd_data, case):
    ctx, calculation, answer = tz_dateadd_data
    if case:
        query_fmt = "SELECT CASE WHEN bool_col THEN {:} ELSE NULL END FROM table1"
    else:
        query_fmt = "SELECT {:} FROM table1"
    query = query_fmt.format(calculation)
    check_query(
        query,
        ctx,
        None,
        sort_output=False,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.fixture
def timeadd_dataframe():
    time_args_list = [
        (0, 0, 0, 0),
        (1, 1, 1, 1),
        (2, 4, 8, 16),
        None,
        (14, 52, 48, 20736),
        (16, 25, 37, 28561),
        (18, 1, 44, 38416),
    ]
    return {
        "table1": pd.DataFrame(
            {
                "T": [
                    None
                    if t is None
                    else Time(hour=t[0], minute=t[1], second=t[2], nanosecond=t[3])
                    for t in time_args_list
                ],
                "N": [-50, 7, -22, 13, -42, -17, 122],
            }
        )
    }


@pytest.fixture(
    params=[
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ]
)
def timeadd_arguments(request, timeadd_dataframe):
    time_args_lists = {
        "hour": [
            (22, 0, 0, 0),
            (8, 1, 1, 1),
            (4, 4, 8, 16),
            None,
            (20, 52, 48, 20736),
            (23, 25, 37, 28561),
            (20, 1, 44, 38416),
        ],
        "minute": [
            (23, 10, 0, 0),
            (1, 8, 1, 1),
            (1, 42, 8, 16),
            None,
            (14, 10, 48, 20736),
            (16, 8, 37, 28561),
            (20, 3, 44, 38416),
        ],
        "second": [
            (23, 59, 10, 0),
            (1, 1, 8, 1),
            (2, 3, 46, 16),
            None,
            (14, 52, 6, 20736),
            (16, 25, 20, 28561),
            (18, 3, 46, 38416),
        ],
        "millisecond": [
            (23, 59, 59, 950000000),
            (1, 1, 1, 7000001),
            (2, 4, 7, 978000016),
            None,
            (14, 52, 47, 958020736),
            (16, 25, 36, 983028561),
            (18, 1, 44, 122038416),
        ],
        "microsecond": [
            (23, 59, 59, 999950000),
            (1, 1, 1, 7001),
            (2, 4, 7, 999978016),
            None,
            (14, 52, 47, 999978736),
            (16, 25, 37, 11561),
            (18, 1, 44, 160416),
        ],
        "nanosecond": [
            (23, 59, 59, 999999950),
            (1, 1, 1, 8),
            (2, 4, 7, 999999994),
            None,
            (14, 52, 48, 20694),
            (16, 25, 37, 28544),
            (18, 1, 44, 38538),
        ],
    }
    answer = pd.DataFrame(
        {
            0: timeadd_dataframe["table1"]["T"],
            1: [
                None
                if t is None
                else Time(hour=t[0], minute=t[1], second=t[2], nanosecond=t[3])
                for t in time_args_lists[request.param]
            ],
        }
    )
    return request.param, answer


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(
            True,
            id="with_case",
            marks=pytest.mark.skip(reason="TODO: support time in CASE statements"),
        ),
    ],
)
def test_timeadd(timeadd_dataframe, timeadd_arguments, use_case, memory_leak_check):
    unit, answer = timeadd_arguments
    # Decide which function to use based on the unit
    func = {
        "hour": "DATEADD",
        "minute": "TIMEADD",
        "second": "DATEADD",
        "millisecond": "TIMEADD",
        "microsecond": "DATEADD",
        "nanosecond": "TIMEADD",
    }[unit]
    if use_case:
        query = f"SELECT T, CASE WHEN N < -100 THEN NULL ELSE {func}('{unit}', N, T) END FROM TABLE1"
    else:
        query = f"SELECT T, {func}('{unit}', N, T) FROM TABLE1"
    check_query(
        query,
        timeadd_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(
            True,
            id="with_case",
            marks=pytest.mark.skip(reason="TODO: support time in CASE statements"),
        ),
    ],
)
def test_timestampadd_time(
    timeadd_dataframe, timeadd_arguments, use_case, memory_leak_check
):
    unit, answer = timeadd_arguments
    unit_str = {
        "hour": unit,
        "minute": f"'{unit}'",
        "second": unit,
        "millisecond": f"'{unit}'",
        "microsecond": unit,
        "nanosecond": f"'{unit}'",
    }[unit]
    if use_case:
        query = f"SELECT T, CASE WHEN N < -100 THEN NULL ELSE TIMESTAMPADD({unit_str}, N, T) END FROM TABLE1"
    else:
        query = f"SELECT T, TIMESTAMPADD({unit_str}, N, T) FROM TABLE1"
    check_query(
        query,
        timeadd_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "dateadd_fn, dt_val",
    [
        pytest.param("DATEADD", "datetime_strings", id="DATEADD-vector_string_dt"),
        pytest.param("DATE_ADD", "'2020-10-13'", id="DATE_ADD-scalar_string_dt"),
        pytest.param("ADDDATE", "timestamps", id="ADDDATE-vector_timestamp_dtr"),
        pytest.param(
            "DATEADD", "TIMESTAMP '2022-3-5'", id="DATEADD-scalar_timestamp_dtr"
        ),
    ],
)
@pytest.mark.parametrize(
    "amt_val",
    [
        pytest.param("mixed_integers", id="vector_amount"),
        pytest.param("100", id="scalar_amount"),
    ],
)
def test_mysql_dateadd(
    dt_fn_dataframe, dateadd_fn, dt_val, amt_val, case, spark_info, memory_leak_check
):
    """Tests the MySQL version of DATEADD and all of its equivalent functions
    with and without cases, scalar and vector data, and on all accepted input
    types. Meanings of the parametrized arguments:

    dt_fn_dataframe: The fixture containing the datetime-equivialent data

    dateadd_fn: which function name is being used.

    dt_val: The scalar string or column name representing the datetime that
            the integer amount of days areadded to. Relevent column names:
         - datetime_strings: strings that can be casted to timestamps
         - timestamps: already in timestamp form
         - tz_timestamps: timestamps with timezone data
         - valid_year_integers: integers representing a year

    amt_val: The scalar integer or column name representing the amount of days
             to add to dt_val.

    case: Should a case statement be used? If so, case on the value of
          another column of dt_fn_dataframe (positive_integers)
    """

    if case:
        query = f"SELECT CASE WHEN positive_integers < 100 THEN {dateadd_fn}({dt_val}, {amt_val}) ELSE NULL END from table1"
        spark_query = f"SELECT CASE WHEN positive_integers < 100 THEN DATE_ADD({dt_val}, {amt_val}) ELSE NULL END from table1"
    else:
        query = f"SELECT {dateadd_fn}({dt_val}, {amt_val}) from table1"
        spark_query = f"SELECT DATE_ADD({dt_val}, {amt_val}) from table1"

    # spark requires certain arguments of adddate to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"table1": [("mixed_integers", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "dateadd_fn, case",
    [
        pytest.param("DATEADD", False, id="DATEADD-no_case"),
        pytest.param("ADDDATE", True, id="ADDATE-with_case", marks=pytest.mark.slow),
    ],
)
def test_tz_mysql_dateadd(dateadd_fn, case, memory_leak_check):
    """A minor extension of test_mysql_dateadd specifically for tz-aware data"""
    if case:
        query = (
            f"SELECT CASE WHEN N < 0 THEN NULL ELSE {dateadd_fn}(A, N) END from table1"
        )
    else:
        query = f"SELECT {dateadd_fn}(A, N) from table1"

    timestamp_strings = [
        "2025-5-3",
        "2024-4-7",
        None,
        "2023-3-10",
        "2022-2-13",
        "2025-12-2",
        None,
        "2024-11-4",
        "2023-10-6",
    ]
    day_offsets = [None, 10, 2, 1, 60, -30, None, 380, 33]
    adjusted_timestamp_strings = [
        None,
        "2024-4-17",
        None,
        "2023-3-11",
        "2022-4-14",
        None if case else "2025-11-2",
        None,
        "2025-11-19",
        "2023-11-8",
    ]
    tz_timestamps = pd.Series(
        [
            None if s is None else pd.Timestamp(s, tz="US/Pacific")
            for s in timestamp_strings
        ]
    )
    res = pd.Series(
        [
            None if s is None else pd.Timestamp(s, tz="US/Pacific")
            for s in adjusted_timestamp_strings
        ]
    )
    ctx = {
        "table1": pd.DataFrame(
            {"A": tz_timestamps, "N": pd.Series(day_offsets, dtype=pd.Int32Dtype())}
        )
    }
    expected_output = pd.DataFrame({"res": res})

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_subdate_cols_int_arg1(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    "tests that date_sub/subdate works when the second argument is an integer, on column values"
    query = f"SELECT {subdate_equiv_fns}({timestamp_date_string_cols}, positive_integers) from table1"
    spark_query = (
        f"SELECT DATE_SUB({timestamp_date_string_cols}, positive_integers) from table1"
    )

    # spark requires certain arguments of subdate to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"table1": [("positive_integers", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
    )


@pytest.mark.slow
def test_subdate_scalar_int_arg1(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    "tests that date_sub/subdate works when the second argument is an integer, on scalar values"

    # Spark's date_add seems to truncate everything after the day in the scalar case, so we use normalized timestamps for bodosql
    query = f"SELECT CASE WHEN {subdate_equiv_fns}({timestamp_date_string_cols}, positive_integers) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE TO_DATE(SUBDATE({timestamp_date_string_cols}, positive_integers)) END from table1"
    spark_query = f"SELECT CASE WHEN DATE_SUB({timestamp_date_string_cols}, positive_integers) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE DATE_SUB({timestamp_date_string_cols}, positive_integers) END from table1"

    # spark requires certain arguments of subdate to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"table1": [("positive_integers", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        spark_input_cols_to_cast=cols_to_cast,
    )


def test_subdate_cols_td_arg1(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    """tests that date_sub/subdate works on timedelta 2nd arguments, with column inputs"""
    query = f"SELECT {subdate_equiv_fns}({timestamp_date_string_cols}, intervals) from table1"

    expected_output = pd.DataFrame(
        {
            "unknown_column_name": pd.to_datetime(
                dt_fn_dataframe["table1"][timestamp_date_string_cols]
            )
            - dt_fn_dataframe["table1"]["intervals"]
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_subdate_td_scalars(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    """tests that subdate works on timedelta 2nd arguments, with scalar inputs"""
    query = f"SELECT CASE WHEN {subdate_equiv_fns}({timestamp_date_string_cols}, intervals) < TIMESTAMP '1700-01-01' THEN TIMESTAMP '1970-01-01' ELSE {subdate_equiv_fns}({timestamp_date_string_cols}, intervals) END from table1"

    expected_output = pd.DataFrame(
        {
            "unknown_column_name": pd.to_datetime(
                dt_fn_dataframe["table1"][timestamp_date_string_cols]
            )
            - dt_fn_dataframe["table1"]["intervals"]
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case"),
    ],
)
@pytest.mark.parametrize(
    "interval_amt",
    [
        pytest.param(100, id="integer"),
        pytest.param("INTERVAL '4' days + INTERVAL '6' hours", id="timedelta_add"),
        pytest.param("INTERVAL '4' days - INTERVAL '6' hours", id="timedelta_sub"),
    ],
)
def test_tz_aware_subdate(use_case, interval_amt, memory_leak_check):
    if use_case:
        query = f"SELECT DT, CASE WHEN DT IS NULL THEN NULL ELSE SUBDATE(DT, {interval_amt}) END from table1"
    else:
        query = f"SELECT DT, DATE_SUB(DT, {interval_amt}) from table1"

    input_ts = [
        "2018-3-1 12:30:59.251125999",
        "2018-3-12",
        None,
        "2018-9-29 6:00:00",
        "2018-11-6",
    ]
    if isinstance(interval_amt, int):
        output_ts = [
            "2017-11-21 12:30:59.251125999",
            "2017-12-02",
            None,
            "2018-06-21 06:00:00",
            "2018-07-29",
        ]
    elif "+" in interval_amt:
        output_ts = [
            "2018-02-25 06:30:59.251125999",
            "2018-03-07 18:00:00",
            None,
            "2018-09-25",
            "2018-11-01 18:00:00",
        ]
    else:
        output_ts = [
            "2018-02-25 18:30:59.251125999",
            "2018-03-08 6:00:00",
            None,
            "2018-09-25 12:00:00",
            "2018-11-02 6:00:00",
        ]

    tz = "US/Pacific" if use_case else "Poland"
    table1 = pd.DataFrame(
        {
            "DT": pd.Series(
                [None if ts is None else pd.Timestamp(ts, tz=tz) for ts in input_ts]
            )
        }
    )
    expected_output = pd.DataFrame(
        {
            0: table1.DT,
            1: pd.Series(
                [None if ts is None else pd.Timestamp(ts, tz=tz) for ts in output_ts]
            ),
        }
    )

    check_query(
        query,
        {"table1": table1},
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        only_jit_1DVar=True,
    )


def test_yearweek(spark_info, dt_fn_dataframe, memory_leak_check):
    """Test for YEARWEEK, which returns a 6-character string
    with the date's year and week (1-53) concatenated together"""
    query = "SELECT YEARWEEK(timestamps) from table1"

    expected_output = pd.DataFrame(
        {
            "expected": dt_fn_dataframe["table1"]["timestamps"].dt.year * 100
            + dt_fn_dataframe["table1"]["timestamps"].dt.isocalendar().week
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearweek(tz_aware_df, memory_leak_check):
    """Test for YEARWEEK on timezone aware data.
    YEARWEEK returns a 6-character string with the date's year
    and week (1-53) concatenated together"""

    query = "SELECT YEARWEEK(A) from table1"

    expected_output = pd.DataFrame(
        {
            "expected": tz_aware_df["table1"]["A"].dt.year * 100
            + tz_aware_df["table1"]["A"].dt.isocalendar().week
        }
    )
    check_query(
        query,
        tz_aware_df,
        spark=None,
        check_names=False,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


def test_yearweek_scalars(spark_info, dt_fn_dataframe, memory_leak_check):
    query = "SELECT CASE WHEN YEARWEEK(timestamps) = 0 THEN -1 ELSE YEARWEEK(timestamps) END from table1"

    expected_output = pd.DataFrame(
        {
            "expected": dt_fn_dataframe["table1"]["timestamps"].dt.year * 100
            + dt_fn_dataframe["table1"]["timestamps"].dt.isocalendar().week
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


def test_to_date_cols(
    spark_info, timestamp_date_string_cols, dt_fn_dataframe, memory_leak_check
):
    query = f"SELECT TO_DATE({timestamp_date_string_cols}) from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_to_date_scalar(
    spark_info, timestamp_date_string_cols, dt_fn_dataframe, memory_leak_check
):
    query = f"SELECT CASE WHEN TO_DATE({timestamp_date_string_cols}) = TIMESTAMP '2021-05-31' THEN TIMESTAMP '2021-05-30' ELSE TO_DATE({timestamp_date_string_cols}) END from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_date_trunc_time_part(time_df, time_part_strings, memory_leak_check):
    query = f"SELECT DATE_TRUNC('{time_part_strings}', A) as output from table1"
    scalar_func = generate_date_trunc_time_func(time_part_strings)
    output = pd.DataFrame({"output": time_df["table1"]["A"].map(scalar_func)})
    check_query(
        query,
        time_df,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=output,
    )


def test_date_trunc_day_part_handling(time_df, day_part_strings, memory_leak_check):
    query = f"SELECT DATE_TRUNC('{day_part_strings}', A) as output from table1"
    output = pd.DataFrame({"output": []})
    with pytest.raises(
        Exception,
        match=f'Unsupported unit for DATE_TRUNC with TIME input: "{day_part_strings}"',
    ):
        check_query(
            query,
            time_df,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=output,
        )


def test_date_trunc_date(date_df, day_part_strings, memory_leak_check):
    """
    test DATE_TRUNC works for datetime.date input
    """
    query = f"SELECT DATE_TRUNC('{day_part_strings}', A) as output from table1"
    scalar_func = generate_date_trunc_date_func(day_part_strings)
    output = pd.DataFrame({"output": date_df["table1"]["A"].map(scalar_func)})
    with bodosql_use_date_type():
        check_query(
            query,
            date_df,
            None,
            check_names=False,
            expected_output=output,
        )


def test_date_trunc_time_part_handling(date_df, time_part_strings, memory_leak_check):
    """
    test DATE_TRUNC can return the same date when date_or_time_expr is datetime.date
    and date_or_time_part is smaller than day.
    """
    query = f"SELECT DATE_TRUNC('{time_part_strings}', A) as output from table1"
    output = pd.DataFrame({"output": date_df["table1"]["A"]})
    with bodosql_use_date_type():
        check_query(
            query,
            date_df,
            None,
            check_names=False,
            expected_output=output,
        )


def test_date_trunc_timestamp(dt_fn_dataframe, date_trunc_literal, memory_leak_check):
    query = (
        f"SELECT DATE_TRUNC('{date_trunc_literal}', TIMESTAMPS) as output from table1"
    )
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    py_output = pd.DataFrame(
        {"output": dt_fn_dataframe["table1"]["timestamps"].map(scalar_func)}
    )
    check_query(query, dt_fn_dataframe, None, expected_output=py_output)


def test_yearofweek(dt_fn_dataframe, memory_leak_check):
    """
    Test Snowflake's yearofweek function on columns.
    """
    query = f"SELECT YEAROFWEEKISO(TIMESTAMPS) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {
            "A": dt_fn_dataframe["table1"]["timestamps"]
            .dt.isocalendar()
            .year.astype("Int64")
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark=None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweek(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweek function on columns.
    """
    query = f"SELECT YEAROFWEEKISO(A) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame({"A": tz_aware_df["table1"]["A"].dt.year})
    check_query(
        query,
        tz_aware_df,
        spark=None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_yearofweekiso(spark_info, dt_fn_dataframe, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on columns.
    """
    query = f"SELECT YEAROFWEEKISO(TIMESTAMPS) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {"A": dt_fn_dataframe["table1"]["timestamps"].dt.isocalendar().year}
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweekiso(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on timezone-aware columns.
    """
    query = f"SELECT YEAROFWEEKISO(A) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {"A": tz_aware_df["table1"]["A"].dt.isocalendar().year}
    )
    check_query(
        query,
        tz_aware_df,
        spark=None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_yearofweekiso_scalar(spark_info, dt_fn_dataframe, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on scalars.
    """
    query = f"SELECT CASE WHEN YEAROFWEEKISO(TIMESTAMPS) > 2015 THEN 1 ELSE 0 END as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {
            "A": dt_fn_dataframe["table1"]["timestamps"]
            .dt.isocalendar()
            .year.apply(lambda x: 1 if pd.notna(x) and x > 2015 else 0)
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweekiso_scalar(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on timezone-aware scalars.
    """
    query = (
        f"SELECT CASE WHEN YEAROFWEEKISO(A) > 2015 THEN 1 ELSE 0 END as A from table1"
    )
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {
            "A": tz_aware_df["table1"]["A"]
            .dt.isocalendar()
            .year.apply(lambda x: 1 if pd.notna(x) and x > 2015 else 0)
        }
    )
    check_query(
        query,
        tz_aware_df,
        spark=None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_weekiso(spark_info, dt_fn_dataframe, memory_leak_check):
    query = "SELECT WEEKISO(timestamps) from table1"
    spark_query = "SELECT WEEK(timestamps) from table1"

    expected_output = pd.DataFrame(
        {"expected": dt_fn_dataframe["table1"]["timestamps"].dt.isocalendar().week}
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        only_python=True,
        equivalent_spark_query=spark_query,
        expected_output=expected_output,
    )


def test_weekiso_scalar(spark_info, dt_fn_dataframe, memory_leak_check):
    query = "SELECT CASE WHEN WEEKISO(timestamps) = 0 THEN -1 ELSE WEEKISO(timestamps) END from table1"
    spark_query = "SELECT CASE WHEN WEEK(timestamps) = 0 THEN -1 ELSE WEEK(timestamps) END from table1"

    expected_output = pd.DataFrame(
        {"expected": dt_fn_dataframe["table1"]["timestamps"].dt.isocalendar().week}
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        only_python=True,
        equivalent_spark_query=spark_query,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_weekiso(tz_aware_df, memory_leak_check):
    """simplest weekiso test on timezone aware data"""
    query = "SELECT WEEKISO(A) from table1"

    expected_output = pd.DataFrame(
        {"expected": tz_aware_df["table1"]["A"].dt.isocalendar().week}
    )
    check_query(
        query,
        tz_aware_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_weekiso_case(tz_aware_df, memory_leak_check):
    """weekiso test in case statement on timezone aware data"""
    query = "SELECT CASE WHEN WEEKISO(A) > 2 THEN WEEKISO(A) ELSE 0 END from table1"

    weekiso_series = tz_aware_df["table1"]["A"].dt.isocalendar().week
    weekiso_series[weekiso_series <= 2] = 0

    expected_output = pd.DataFrame({"expected": weekiso_series})
    check_query(
        query,
        tz_aware_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


dm = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}


@pytest.mark.parametrize("next_or_prev", ["NEXT", "PREVIOUS"])
@pytest.mark.parametrize("dow_str", ["days_of_week", "su"])
def test_next_previous_day_cols(
    spark_info, dt_fn_dataframe, next_or_prev, dow_str, memory_leak_check
):
    if dow_str in dm.keys():
        query = f"SELECT {next_or_prev}_DAY(timestamps, '{dow_str}') from table1"
    else:
        query = f"SELECT {next_or_prev}_DAY(timestamps, {dow_str}) from table1"

    mlt = 1 if next_or_prev == "NEXT" else -1
    next_prev_day = lambda ts, dow: (
        ts
        + mlt
        * pd.to_timedelta(
            7 - (mlt * (ts.dt.dayofweek.values - pd.Series(dow).map(dm).values) % 7),
            unit="D",
        )
    ).dt.normalize()

    dow_col = (
        dt_fn_dataframe["table1"]["days_of_week"]
        if dow_str == "days_of_week"
        else np.array([dow_str])
    )
    py_output = pd.DataFrame(
        {"A": next_prev_day(dt_fn_dataframe["table1"]["timestamps"], dow_col)}
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize("next_or_prev", ["NEXT", "PREVIOUS"])
@pytest.mark.parametrize("dow_str", ["days_of_week", "su"])
def test_next_previous_day_scalars(
    spark_info, dt_fn_dataframe, next_or_prev, dow_str, memory_leak_check
):
    if dow_str in dm.keys():
        query = f"SELECT CASE WHEN MONTH({next_or_prev}_DAY(timestamps, '{dow_str}')) < 4 THEN  TIMESTAMP '2021-05-31' ELSE {next_or_prev}_DAY(timestamps, '{dow_str}') END from table1"
    else:
        query = f"SELECT CASE WHEN MONTH({next_or_prev}_DAY(timestamps, {dow_str})) < 4 THEN  TIMESTAMP '2021-05-31' ELSE {next_or_prev}_DAY(timestamps, {dow_str}) END from table1"

    mlt = 1 if next_or_prev == "NEXT" else -1
    next_prev_day = lambda ts, dow: (
        ts
        + mlt
        * pd.to_timedelta(
            7 - (mlt * (ts.dt.dayofweek.values - pd.Series(dow).map(dm).values) % 7),
            unit="D",
        )
    ).dt.normalize()

    def next_prev_day_case(ts, dow):
        ret = next_prev_day(ts, dow)
        ret[ret.dt.month < 4] = pd.Timestamp("2021-05-31")
        return ret

    dow_col = (
        dt_fn_dataframe["table1"]["days_of_week"]
        if dow_str == "days_of_week"
        else np.array([dow_str])
    )
    py_output = pd.DataFrame(
        {"A": next_prev_day_case(dt_fn_dataframe["table1"]["timestamps"], dow_col)}
    )
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_day(tz_aware_df, memory_leak_check):
    query = "SELECT DAY(A) as m from table1"
    df = tz_aware_df["table1"]
    py_output = pd.DataFrame({"m": df.A.dt.day})
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_day_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN DAY(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="145D27H37T48S", periods=30, tz="Poland"
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"table1": df}

    day_series = df.A.dt.day
    day_series[~df.B] = None
    py_output = pd.DataFrame({"m": day_series})

    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_extract_yhms(tz_aware_df, memory_leak_check):
    query = "SELECT EXTRACT(YEAR from A) as my_yr, EXTRACT(HOUR from A) as h, \
                    EXTRACT(MINUTE from A) as m, EXTRACT(SECOND from A) as s \
                    from table1"
    df = tz_aware_df["table1"]
    py_output = pd.DataFrame(
        {
            "my_yr": df.A.dt.year,
            "h": df.A.dt.hour,
            "m": df.A.dt.minute,
            "s": df.A.dt.second,
        }
    )
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_year_hr_min_sec(tz_aware_df, memory_leak_check):
    query = "SELECT YEAR(A) as my_yr, HOUR(A) as h, MINUTE(A) as m, SECOND(A) as s from table1"
    df = tz_aware_df["table1"]
    py_output = pd.DataFrame(
        {
            "my_yr": df.A.dt.year,
            "h": df.A.dt.hour,
            "m": df.A.dt.minute,
            "s": df.A.dt.second,
        }
    )
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_month(tz_aware_df, memory_leak_check):
    query = "SELECT MONTH(A) as m from table1"
    df = tz_aware_df["table1"]
    py_output = pd.DataFrame({"m": df.A.dt.month})
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_month_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN MONTH(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Poland"
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"table1": df}
    month_series = df.A.dt.month
    month_series[~df.B] = None
    py_output = pd.DataFrame({"m": month_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.fixture(
    params=[
        pytest.param(("US/Pacific", "1/1/2023", "H"), id="pacific-by_hour"),
        pytest.param(("GMT", "1/1/2021", "49MIN"), id="gmt-by_49_minutes"),
        pytest.param(
            ("Australia/Sydney", "1/1/2027", "W"),
            id="sydney-by_week",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("US/Eastern", "6/12/2021", "1234567891234ns"),
            id="eastern-by_many_ns",
            marks=pytest.mark.slow,
        ),
    ]
)
def large_tz_df(request):
    tz, end, freq = request.param
    D = pd.date_range(start="1/1/2020", tz=tz, end=end, freq=freq)
    return pd.DataFrame(
        {
            "A": D.to_series(index=pd.RangeIndex(len(D))),
            "B": [i % 2 == 0 for i in range(len(D))],
        }
    )


@pytest.mark.tz_aware
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="case"),
    ],
)
def test_tz_aware_week_quarter_dayname(large_tz_df, case, memory_leak_check):
    """Tests the BodoSQL functions WEEK, QUARTER and DAYNAME on timezone aware
    data with and without case statements. The queries are in the following
    forms:

    1. SELECT A, WEEK(A), ... from table1
    2. SELECT A, CASE WHEN B THEN WEEK(A) ELSE NULL END, ... from table1
    """
    calculations = []
    for func in ["WEEK", "QUARTER", "DAYNAME", "MONTHNAME", "MONTH_NAME"]:
        if case:
            calculations.append(f"CASE WHEN B THEN {func}(A) ELSE NULL END")
        else:
            calculations.append(f"{func}(A)")
    query = f"SELECT A, {', '.join(calculations)} FROM table1"

    ctx = {"table1": large_tz_df}

    py_output = pd.DataFrame(
        {
            "a": large_tz_df.A,
            "w": large_tz_df.A.dt.isocalendar().week,
            "q": large_tz_df.A.dt.quarter,
            "d": large_tz_df.A.dt.day_name(),
            "m": large_tz_df.A.dt.month_name(),
            "m2": large_tz_df.A.dt.month_name(),
        }
    )
    if case:
        py_output["w"][~large_tz_df["B"]] = None
        py_output["q"][~large_tz_df["B"]] = None
        py_output["d"][~large_tz_df["B"]] = None
        py_output["m"][~large_tz_df["B"]] = None
        py_output["m2"][~large_tz_df["B"]] = None

    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.tz_aware
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="case"),
    ],
)
def test_tz_aware_dayof_fns(large_tz_df, case, memory_leak_check):
    """Tests the BodoSQL functions DAYOFWEEK, DAYOFWEEKISO, DAYOFMONTH and
    DAYOFYEAR on timezone aware data with and without case statements. The queries
    are in the following forms:

    1. SELECT A, DAYOFWEEK(A), ... from table1
    2. SELECT A, CASE WHEN B THEN DAYOFWEEK(A) ELSE NULL END, ... from table1

    Note, the two DOY functions have the following correspondance to day anmes:
      DAYNAME  DAYOFWEEK DAYOFWEKISO
       Monday          1           1
      Tuesday          2           2
    Wednesday          3           3
     Thursday          4           4
       Friday          5           5
     Saturday          6           6
       Sunday          0           7
    """
    calculations = []
    for func in ["DAYOFWEEK", "DAYOFWEEKISO", "DAYOFMONTH", "DAYOFYEAR"]:
        if case:
            calculations.append(f"CASE WHEN B THEN {func}(A) ELSE NULL END")
        else:
            calculations.append(f"{func}(A)")
    query = f"SELECT A, {', '.join(calculations)} FROM table1"

    ctx = {"table1": large_tz_df}
    py_output = pd.DataFrame(
        {
            "A": large_tz_df.A,
            "dow": (large_tz_df.A.dt.dayofweek + 1) % 7,
            "dowiso": large_tz_df.A.dt.dayofweek + 1,
            "dom": large_tz_df.A.dt.day,
            "doy": large_tz_df.A.dt.dayofyear,
        }
    )
    if case:
        py_output["dow"][~large_tz_df["B"]] = None
        py_output["dowiso"][~large_tz_df["B"]] = None
        py_output["dom"][~large_tz_df["B"]] = None
        py_output["doy"][~large_tz_df["B"]] = None

    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_weekofyear(memory_leak_check):
    query = "SELECT WEEKOFYEAR(A) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Poland"
            ).to_series()
        }
    )
    ctx = {"table1": df}
    py_output = pd.DataFrame({"m": df.A.dt.isocalendar().week})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_weekofyear_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN WEEKOFYEAR(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Poland"
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"table1": df}
    week_series = df.A.dt.isocalendar().week
    week_series[~df.B] = None
    py_output = pd.DataFrame({"m": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_next_day(memory_leak_check):
    query = "SELECT next_day(A, B) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Africa/Casablanca"
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
        }
    )
    ctx = {"table1": df}
    out_series = df.apply(
        lambda row: (
            row["A"].normalize()
            + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    py_output = pd.DataFrame({"m": out_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_next_day_case(
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN next_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Europe/Berlin"
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"table1": df}
    week_series = df.apply(
        lambda row: (
            row["A"].normalize()
            + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"m": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_previous_day(memory_leak_check):
    query = "SELECT previous_day(A, B) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Poland"
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
        }
    )
    ctx = {"table1": df}
    out_series = df.apply(
        lambda row: (
            row["A"].normalize()
            - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    py_output = pd.DataFrame({"m": out_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_previous_day_case(
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN previous_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz="Pacific/Honolulu"
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"table1": df}
    week_series = df.apply(
        lambda row: (
            row["A"].normalize()
            - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"m": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_date_trunc_tz_aware(date_trunc_literal, memory_leak_check):
    query = f"SELECT DATE_TRUNC('{date_trunc_literal}', A) as output from table1"
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None] * 2,
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    py_output = pd.DataFrame({"output": df["A"].map(scalar_func)})
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_date_trunc_tz_aware_case(date_trunc_literal, memory_leak_check):
    query = f"SELECT CASE WHEN B THEN DATE_TRUNC('{date_trunc_literal}', A) END as output from table1"
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None] * 2,
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    S = df["A"].map(scalar_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_year(representative_tz, memory_leak_check):
    """
    Test +/- Interval Year on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT A + Interval 1 Year as output from table1"
    query2 = "SELECT A - Interval 2 Year as output from table1"
    py_output = pd.DataFrame({"output": df.A + pd.DateOffset(years=1)})
    check_query(query1, ctx, None, expected_output=py_output)
    py_output = pd.DataFrame({"output": df.A - pd.DateOffset(years=2)})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_year_case(representative_tz, memory_leak_check):
    """
    Test +/- Interval Year on tz-aware data with case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Year END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Year END as output from table1"
    S = df.A + pd.DateOffset(years=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    S = df.A - pd.DateOffset(years=2)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_month(representative_tz, memory_leak_check):
    """
    Test +/- Interval Month on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT A + Interval 1 Month as output from table1"
    query2 = "SELECT A - Interval 2 Month as output from table1"
    py_output = pd.DataFrame({"output": df.A + pd.DateOffset(months=1)})
    check_query(query1, ctx, None, expected_output=py_output)
    py_output = pd.DataFrame({"output": df.A - pd.DateOffset(months=2)})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_month_case(representative_tz, memory_leak_check):
    """
    Test +/- Interval Month on tz-aware data with case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Month END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Month END as output from table1"
    S = df.A + pd.DateOffset(months=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    S = df.A - pd.DateOffset(months=2)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_day(representative_tz, memory_leak_check):
    """
    Test +/- Interval Day on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT A + Interval 1 Day as output from table1"
    query2 = "SELECT A - Interval 2 Day as output from table1"
    # Function used to simulate the result of adding by a day
    scalar_add_func = interval_day_add_func(1)
    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)
    py_output = pd.DataFrame({"output": df.A.map(scalar_add_func)})
    check_query(query1, ctx, None, expected_output=py_output)
    py_output = pd.DataFrame({"output": df.A.map(scalar_sub_func)})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_day_case(representative_tz, memory_leak_check):
    """
    Test +/- Interval Day on tz-aware data with case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Day END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Day END as output from table1"
    # Function used to simulate the result of adding by a day
    scalar_add_func = interval_day_add_func(1)
    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)
    S = df.A.map(scalar_add_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    S = df.A.map(scalar_sub_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_subdate_integer(memory_leak_check):
    """
    Test subdate on tz-aware data with an integer argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT SUBDATE(A, 3) as output from table1"
    query2 = "SELECT DATE_SUB(A, 3) as output from table1"

    # Function used to simulate the result of subtracting 3 days
    scalar_sub_func = interval_day_add_func(-3)
    py_output = pd.DataFrame({"output": df.A.map(scalar_sub_func)})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_subdate_integer_case(memory_leak_check):
    """
    Test subdate on tz-aware data with an integer argument and case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT CASE WHEN B THEN SUBDATE(A, 3) END as output from table1"
    query2 = "SELECT CASE WHEN B THEN DATE_SUB(A, 3) END as output from table1"

    # Function used to simulate the result of subtracting 3 days
    scalar_sub_func = interval_day_add_func(-3)
    S = df.A.map(scalar_sub_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_subdate_interval_day(memory_leak_check):
    """
    Test subdate on tz-aware data with a Day Interval argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT SUBDATE(A, Interval 2 Days) as output from table1"
    query2 = "SELECT DATE_SUB(A, Interval 2 Days) as output from table1"

    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)

    py_output = pd.DataFrame({"output": df.A.map(scalar_sub_func)})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_subdate_interval_day_case(memory_leak_check):
    """
    Test subdate on tz-aware data with a Day Interval argument and case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = (
        "SELECT CASE WHEN B THEN SUBDATE(A, Interval 2 Days) END as output from table1"
    )
    query2 = (
        "SELECT CASE WHEN B THEN DATE_SUB(A, Interval 2 Days) END as output from table1"
    )

    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)
    S = df.A.map(scalar_sub_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
@pytest.mark.skip(reason="[BE-4239] Fix how BodoSQL processes Interval Month literals")
def test_tz_aware_subdate_interval_month(memory_leak_check):
    """
    Test subdate on tz-aware data with a Month Interval argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT SUBDATE(A, Interval 4 Months) as output from table1"
    query2 = "SELECT DATE_SUB(A, Interval 4 Months) as output from table1"

    py_output = pd.DataFrame({"output": df.A - pd.DateOffset(months=4)})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
def test_tz_aware_subdate_interval_month_case(memory_leak_check):
    """
    Test subdate on tz-aware data with a Month Interval argument and case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="16D5H", periods=30, tz="US/Pacific"
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"table1": df}
    query1 = "SELECT CASE WHEN B THEN SUBDATE(A, Interval 1 Months) END as output from table1"
    query2 = "SELECT CASE WHEN B THEN DATE_SUB(A, Interval 1 Months) END as output from table1"

    S = df.A - pd.DateOffset(months=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"output": S})
    check_query(query1, ctx, None, expected_output=py_output)
    check_query(query2, ctx, None, expected_output=py_output)


@pytest.fixture
def date_from_parts_data():
    years = pd.Series([2015, 2018, 2021, 2024, 2025, 2027, 2030], dtype=pd.Int64Dtype())
    months = pd.Series([1, None, 7, 20, -1, 0, 7], dtype=pd.Int64Dtype())
    days = pd.Series([1, 12, 4, -5, 12, 1, 0], dtype=pd.Int64Dtype())
    answer = pd.Series(
        [
            datetime.date(2015, 1, 1),
            None,
            datetime.date(2021, 7, 4),
            datetime.date(2025, 7, 26),
            datetime.date(2024, 11, 12),
            datetime.date(2026, 12, 1),
            datetime.date(2030, 6, 30),
        ]
    )
    return years, months, days, answer


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case"),
    ],
)
def test_date_from_parts(date_from_parts_data, use_case, memory_leak_check):
    if use_case:
        query = "SELECT CASE WHEN YR < 0 THEN NULL ELSE DATE_FROM_PARTS(YR, MO, DA) END FROM table1"
    else:
        query = "SELECT DATE_FROM_PARTS(YR, MO, DA) FROM table1"
    year, month, day, answer = date_from_parts_data
    df = pd.DataFrame(
        {
            "YR": year,
            "MO": month,
            "DA": day,
        }
    )
    ctx = {"table1": df}
    py_output = pd.DataFrame({0: pd.Series([s for s in answer])})
    with bodosql_use_date_type():
        check_query(
            query,
            ctx,
            None,
            expected_output=py_output,
            check_dtype=False,
            check_names=False,
        )


@pytest.fixture(
    params=[pytest.param(True, id="with_ns"), pytest.param(False, id="no_ns")]
)
def timestamp_from_parts_data(request):
    year = pd.Series([2014, 2016, 2018, None, 2022, 2024], dtype=pd.Int64Dtype())
    month = pd.Series([1, 7, 0, 12, 100, -3], dtype=pd.Int64Dtype())
    day = pd.Series([70, 12, -123, None, 1, -7], dtype=pd.Int64Dtype())
    hour = pd.Series([0, 2, 4, None, 40, -5], dtype=pd.Int64Dtype())
    minute = pd.Series([15, 0, -1, 0, 65, 0], dtype=pd.Int64Dtype())
    second = pd.Series([0, -1, 50, 3, 125, 1234], dtype=pd.Int64Dtype())
    nanosecond = pd.Series(
        [-1, 0, 123456789, 0, 250999, -102030405060], dtype=pd.Int64Dtype()
    )
    use_nanosecond = request.param
    if use_nanosecond:
        answer = pd.Series(
            [
                "2014-3-11 00:14:59.999999999",
                "2016-7-12 1:59:59",
                "2017-7-30 03:59:50.123456789",
                None,
                "2030-4-2 17:07:05.000250999",
                "2023-8-23 19:18:51.969594940",
            ]
        )
    else:
        answer = pd.Series(
            [
                "2014-3-11 00:15:00",
                "2016-7-12 1:59:59",
                "2017-7-30 3:59:50",
                None,
                "2030-4-2 17:07:05",
                "2023-8-23 19:20:34",
            ]
        )
    return year, month, day, hour, minute, second, nanosecond, use_nanosecond, answer


@pytest.mark.parametrize(
    "func",
    [
        pytest.param("TIMESTAMP_FROM_PARTS"),
        pytest.param("TIMESTAMP_NTZ_FROM_PARTS"),
        pytest.param("TIMESTAMP_LTZ_FROM_PARTS"),
        pytest.param("TIMESTAMP_TZ_FROM_PARTS"),
        pytest.param("TIMESTAMPFROMPARTS", marks=pytest.mark.slow),
        pytest.param("TIMESTAMPNTZFROMPARTS", marks=pytest.mark.slow),
        pytest.param("TIMESTAMPLTZFROMPARTS", marks=pytest.mark.slow),
        pytest.param("TIMESTAMPTZFROMPARTS", marks=pytest.mark.slow),
    ],
)
def test_timestamp_from_parts(
    func, timestamp_from_parts_data, local_tz, memory_leak_check
):
    (
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        use_nanosecond,
        answer,
    ) = timestamp_from_parts_data
    ns_str = ", NS" if use_nanosecond else ""
    use_case = func in {
        "TIMESTAMPFROMPARTS",
        "TIMESTAMP_NTZ_FROM_PARTS",
        "TIMESTAMPLTZFROMPARTS",
        "TIMESTAMP_TZ_FROM_PARTS",
    }
    if use_case:
        query = f"SELECT CASE WHEN YR < 0 THEN NULL ELSE {func}(YR, MO, DA, HO, MI, SE{ns_str}) END FROM table1"
    else:
        query = f"SELECT {func}(YR, MO, DA, HO, MI, SE{ns_str}) FROM table1"
    tz = (
        local_tz
        if func
        in {
            "TIMESTAMP_LTZ_FROM_PARTS",
            "TIMESTAMPLTZFROMPARTS",
            "TIMESTAMP_TZ_FROM_PARTS",
            "TIMESTAMPTZFROMPARTS",
        }
        else None
    )
    df = pd.DataFrame(
        {
            "YR": year,
            "MO": month,
            "DA": day,
            "HO": hour,
            "MI": minute,
            "SE": second,
            "NS": nanosecond,
        }
    )
    ctx = {"table1": df}
    py_output = pd.DataFrame({0: pd.Series([pd.Timestamp(s, tz=tz) for s in answer])})
    check_query(query, ctx, None, expected_output=py_output, check_names=False)


@pytest.fixture(
    params=[
        "MONTHNAME",
        "MONTH_NAME",
        "DAYNAME",
        "WEEKDAY",
        "LAST_DAY",
        "YEAROFWEEK",
        "YEAROFWEEKISO",
    ]
)
def date_only_single_arg_fns(request):
    return request.param


def date_only_single_arg_fns_time_input_handling(date_only_single_arg_fns, time_df):
    query = f"SELECT {date_only_single_arg_fns}(A) as output from table1"
    output = pd.DataFrame({"output": []})
    with pytest.raises(
        Exception, match=f"Time object is not supported by {date_only_single_arg_fns}"
    ):
        check_query(
            query,
            time_df,
            None,
            check_names=False,
            check_dtype=False,
            expected_output=output,
        )


@pytest.mark.parametrize("next_or_prev", ["NEXT", "PREVIOUS"])
def next_previous_day_time_input_handling(next_or_prev, time_df):
    query = f"SELECT {next_or_prev}_DAY(A, 'mo') as output from table1"
    output = pd.DataFrame({"output": []})
    with pytest.raises(
        Exception, match=f"Time object is not supported by {next_or_prev}_DAY"
    ):
        check_query(
            query,
            time_df,
            None,
            check_names=False,
            check_dtype=False,
            expected_output=output,
        )
