# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of Snowflake TO_X functions for date-related casting in BodoSQL
"""

import datetime

import bodosql
import pandas as pd
import pytest
from bodosql.tests.test_datetime_fns import dt_fn_dataframe  # noqa
from bodosql.tests.utils import (
    bodosql_use_date_type,
    check_query,
    make_tables_nullable,
)

from bodo.tests.test_bodosql_array_kernels.test_bodosql_snowflake_date_conversion_array_kernels import (  # pragma: no cover
    scalar_to_date_equiv_fn,
)
from bodo.tests.timezone_common import representative_tz  # noqa


@pytest.fixture(
    params=[
        ("2020-12-01T13:56:03.172:00",),
        ("2342-312",),
        ("2020-13-01",),
        ("-20200-15-15",),
        ("2100-12-01-01-01-01-01-01-01-01-01-01-01-01-100",),
        (pd.Series(["2022-02-18", "2022-14-18"] * 10),),
    ]
)
def invalid_to_date_args(request):
    """set of arguments which cause NA in try_to_date, and throw an error for to_date"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("TRY_TO_DATE", id="try_to_date"),
        pytest.param("TO_DATE", id="to_date"),
    ]
)
def test_fn(request):
    return request.param


def test_to_date_valid_strings(spark_info, dt_fn_dataframe, test_fn, memory_leak_check):
    """tests to_date on valid string values"""
    query = f"SELECT {test_fn}(datetime_strings) from table1"
    spark_query = f"SELECT TO_DATE(datetime_strings) from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_strings_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid string values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(datetime_strings) < DATE '2013-01-03' THEN {test_fn}(datetime_strings) END from table1"
    spark_query = f"SELECT CASE WHEN TO_DATE(datetime_strings) < DATE '2013-01-03' THEN TO_DATE(datetime_strings) END from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_digit_strings(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid digit string values in a case statment"""
    query = f"SELECT {test_fn}(digit_strings) from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
    expected_output = pd.DataFrame(
        {
            "foo": dt_fn_dataframe_nullable["table1"]["digit_strings"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
            )
        }
    )

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_valid_digit_strings_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid digit string values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(digit_strings) < DATE '2013-01-03' THEN {test_fn}(digit_strings) END from table1"

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
    expected_output = pd.DataFrame(
        {
            "foo": dt_fn_dataframe_nullable["table1"]["digit_strings"].apply(
                lambda val: scalar_to_date_equiv_fn(val)
                if not (scalar_to_date_equiv_fn(val) is None)
                and (
                    scalar_to_date_equiv_fn(val)
                    < pd.Timestamp("2013-01-03").to_datetime64()
                )
                else None
            )
        }
    )
    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_to_date_valid_datetime_types(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid datetime values"""
    query = f"SELECT {test_fn}(timestamps) from table1"
    spark_query = f"SELECT TO_DATE(timestamps) from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_valid_datetime_types_case(
    spark_info, dt_fn_dataframe, test_fn, memory_leak_check
):
    """tests to_date on valid datetime values in a case statment"""
    query = f"SELECT CASE WHEN {test_fn}(timestamps) < DATE '2013-01-03' THEN {test_fn}(timestamps) END from table1"
    spark_query = f"SELECT CASE WHEN TO_DATE(timestamps) < DATE '2013-01-03' THEN TO_DATE(timestamps) END from table1"
    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)

    check_query(
        query,
        dt_fn_dataframe_nullable,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )


def test_to_date_tz_aware(memory_leak_check):
    """tests to_date on valid datetime values in a case statment"""
    df = pd.DataFrame(
        {
            "timestamps": pd.date_range(
                "1/18/2022", periods=20, freq="10D5H", tz="US/PACIFIC"
            )
        }
    )
    ctx = {"table1": df}
    query = f"SELECT TO_DATE(timestamps) as timestamps from table1"
    expected_output = pd.DataFrame(
        {
            "timestamps": df["timestamps"]
            .dt.normalize()
            .apply(lambda t: t.tz_localize(None))
        }
    )

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


def test_to_date_tz_aware_case(memory_leak_check):
    """tests to_date on valid datetime values in a case statment"""
    df = pd.DataFrame(
        {
            "timestamps": pd.date_range(
                "1/18/2022", periods=30, freq="10D5H", tz="US/PACIFIC"
            ),
            "B": [True, False, True, False, True] * 6,
        }
    )
    ctx = {"table1": df}
    query = f"SELECT CASE WHEN B THEN TO_DATE(timestamps) END as timestamps from table1"
    to_date_series = (
        df["timestamps"].dt.normalize().apply(lambda t: t.tz_localize(None))
    )
    to_date_series[~df.B] = None
    expected_output = pd.DataFrame({"timestamps": to_date_series})

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


def test_try_to_date_tz_strings(tz_aware_df, memory_leak_check):
    """tests try_to_date on valid and invalid datetime values"""

    # Construct input dataframe of both valid and invalid datetime strings
    valid_datetimes = tz_aware_df["table1"]["A"]

    invalid_str_datetimes = pd.Series(
        [
            "",
            "invalid",
            "2020-100-17 00:00:00-00:00",
            "2021-0-0 00:00:00-00:00",
            "2000-0-50 00:00:00-00:00",
            "2000-13-32 00:00:00-00:00",
            "2020-1 00:00:00-00:00",
            "2022-1-2-3 00:00:00-00:00",
            "01/2020",
            "0111/01/1999",
            "01/100/2000",
        ]
    )
    df = pd.DataFrame(
        {
            "timestamps": pd.concat(
                [valid_datetimes.astype(str), invalid_str_datetimes]
            ).reset_index(drop=True)
        }
    )

    # Construct expected answer using Pandas
    valid_answers = valid_datetimes.dt.date
    invalid_answers = pd.Series([None] * len(invalid_str_datetimes))
    expected_output = pd.DataFrame(
        {
            "timestamps": pd.concat([valid_answers, invalid_answers])
            .reset_index(drop=True)
            .astype("datetime64[ns]")
        }
    )

    ctx = {"table1": df}
    query = f"SELECT TRY_TO_DATE(timestamps) as timestamps from table1"

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


# [BE-3774] Leaks Memory
def test_invalid_to_date_args(spark_info, dt_fn_dataframe, test_fn):
    """tests arguments which cause NA in try_to_date, and throw an error for to_date"""

    query = f"SELECT {test_fn}(invalid_dt_strings) from table1"

    if test_fn == "TRY_TO_DATE":
        expected_output = pd.DataFrame(
            {"foo": pd.Series([None] * len(dt_fn_dataframe["table1"]))}
        )
        dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
        check_query(
            query,
            dt_fn_dataframe_nullable,
            spark_info,
            check_dtype=False,
            check_names=False,
            expected_output=expected_output,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(Exception, match=msg):
            bc = bodosql.BodoSQLContext(dt_fn_dataframe)
            bc.sql(query)


def test_to_date_format_string_err(dt_fn_dataframe, test_fn, memory_leak_check):
    """
    Tests that using a format string with TO_DATE or TRY_TO_DATE fails with a reasonable error
    see https://bodo.atlassian.net/browse/BE-3614
    """

    query = f"SELECT {test_fn}(datetime_strings, 'foo') from table1"
    msg = f"Error, {test_fn} with two arguments not yet supported"
    with pytest.raises(Exception, match=msg):
        bc = bodosql.BodoSQLContext(dt_fn_dataframe)
        bc.sql(query)


_to_timestamp_string_data = [
    pytest.param(
        (
            pd.Series(
                [
                    "2019-1-1",
                    None,
                    "2020-4-25 5:30:00",
                    "2021-08-1 9:25:01.150",
                    "2022-12-30 20:59:59.99999",
                ]
            ),
            pd.Series(
                [
                    "2019-1-1",
                    None,
                    "2020-4-25 5:30:00",
                    "2021-8-1 9:25:01.150",
                    "2022-12-30 20:59:59.99999",
                ]
            ),
            None,
        ),
        id="string",
    ),
    pytest.param(
        (
            pd.Series(
                [
                    # Seconds
                    "1",
                    "1000",
                    "1000000",
                    "1000000000",
                    "-123456789",
                    # Milliseconds
                    "31536000000",
                    "1000000000000",
                    "-31536000000",
                    None,
                    # Microseconds
                    "31536000000000",
                    "1000000000000000",
                    "-1234567891234567",
                    None,
                    # Nanoseconds
                    "31536000000000000",
                    "1000000000000000000",
                    "-123456789123456789",
                    None,
                ]
            ),
            pd.Series(
                [
                    # Seconds
                    "1970-01-01 00:00:01",
                    "1970-01-01 00:16:40",
                    "1970-01-12 13:46:40",
                    "2001-09-09 01:46:40",
                    "1966-02-02 02:26:51",
                    # Milliseconds
                    "1971-01-01",
                    "2001-09-09 01:46:40",
                    "1969-01-01",
                    None,
                    # Microseconds
                    "1971-01-01",
                    "2001-09-09 01:46:40",
                    "1930-11-18 00:28:28.765433",
                    None,
                    # Nanoseconds
                    "1971-01-01",
                    "2001-09-09 01:46:40",
                    "1966-02-02 02:26:50.876543211",
                    None,
                ]
            ),
            None,
        ),
        id="numeric_string",
    ),
]

_to_timestamp_timestamp_data = [
    pytest.param(
        (
            pd.Series(
                [
                    pd.Timestamp("2018-10-1"),
                    None,
                    pd.Timestamp("2023-3-1 12:30:00"),
                    None,
                    pd.Timestamp("2025-7-4 9:00:13.250999"),
                ]
            ),
            pd.Series(
                [
                    "2018-10-1",
                    None,
                    "2023-3-1 12:30:00",
                    None,
                    "2025-7-4 9:00:13.250999",
                ]
            ),
            None,
        ),
        id="naive_timestamp",
    ),
    pytest.param(
        (
            pd.Series(
                [
                    pd.Timestamp("2018-10-1", tz="Australia/Sydney"),
                    None,
                    pd.Timestamp("2023-3-1 12:30:00", tz="Australia/Sydney"),
                    None,
                    pd.Timestamp("2025-7-4 9:00:13.250999", tz="Australia/Sydney"),
                ]
            ),
            pd.Series(
                [
                    "2018-10-1",
                    None,
                    "2023-3-1 12:30:00",
                    None,
                    "2025-7-4 9:00:13.250999",
                ]
            ),
            "Australia/Sydney",
        ),
        id="tz_timestamp",
    ),
    pytest.param(
        (
            pd.Series(
                [
                    datetime.date(2020, 7, 4),
                    None,
                    datetime.date(2023, 2, 28),
                    None,
                    datetime.date(2021, 12, 31),
                ]
            ),
            pd.Series(
                [
                    "2020-7-4 00:00:00",
                    None,
                    "2023-2-28 00:00:00",
                    None,
                    "2021-12-31 00:00:00",
                ]
            ),
            None,
        ),
        id="date",
    ),
]


@pytest.fixture(params=_to_timestamp_timestamp_data + _to_timestamp_string_data)
def to_timestamp_non_numeric_data(request):
    """Arguments for TO_TIMESTAMP. Each argument is accompanied by the expected
    output timestamps in string format, and a string indicating the timezone
    of the input data (or None if naive). This fixture only covers the non-numeric
    input types because the numerics will have their answer vary depending
    on the scale provided."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("TO_TIMESTAMP"),
        pytest.param("TO_TIMESTAMP_NTZ"),
        pytest.param("TO_TIMESTAMP_LTZ"),
        pytest.param("TO_TIMESTAMP_TZ"),
        pytest.param("TRY_TO_TIMESTAMP", marks=pytest.mark.slow),
        pytest.param("TRY_TO_TIMESTAMP_NTZ", marks=pytest.mark.slow),
        pytest.param("TRY_TO_TIMESTAMP_LTZ", marks=pytest.mark.slow),
        pytest.param("TRY_TO_TIMESTAMP_TZ", marks=pytest.mark.slow),
    ]
)
def to_timestamp_fn(request):
    return request.param


def test_to_timestamp_non_numeric(
    to_timestamp_fn, to_timestamp_non_numeric_data, local_tz, memory_leak_check
):
    use_case = to_timestamp_fn in {
        "TO_TIMESTAMP_NTZ",
        "TO_TIMESTAMP_TZ",
        "TRY_TO_TIMESTAMP",
        "TRY_TO_TIMESTAMP_LTZ",
    }
    data, answer, old_tz = to_timestamp_non_numeric_data
    if to_timestamp_fn.endswith("_LTZ"):
        tz = local_tz
    elif to_timestamp_fn.endswith("_TZ"):
        if old_tz is None:
            tz = local_tz
        else:
            tz = old_tz
    else:
        tz = None
    if use_case:
        query = (
            f"SELECT CASE WHEN b THEN NULL ELSE {to_timestamp_fn}(t) END FROM table1"
        )
    else:
        query = f"SELECT {to_timestamp_fn}(t) FROM table1"
    ctx = {
        "table1": pd.DataFrame({"t": data, "b": [i % 5 == 4 for i in range(len(data))]})
    }
    expected_output = pd.DataFrame(
        {0: pd.Series([None if s is None else pd.Timestamp(s, tz=tz) for s in answer])}
    )
    if use_case:
        expected_output[0][ctx["table1"]["b"]] = None
    with bodosql_use_date_type():
        check_query(
            query,
            ctx,
            None,
            expected_output=expected_output,
            check_names=False,
            only_jit_1DVar=True,
        )


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [
                        0,
                        None,
                        2048,
                        1073741824,
                        -536870912,
                    ],
                    dtype=pd.Int64Dtype(),
                ),
                pd.Series(
                    [
                        "1970-1-1",
                        None,
                        "1970-1-1 00:34:08",
                        "2004-01-10 13:37:04",
                        "1952-12-27 05:11:28",
                    ]
                ),
                "",
            ),
            id="integers-no_scale",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        123456789,
                        None,
                        12,
                        205891132094649,
                        -68630377364883,
                    ],
                    dtype=pd.Int64Dtype(),
                ),
                pd.Series(
                    [
                        "1970-01-01 00:02:03.456789",
                        None,
                        "1970-01 00:00:00.000012",
                        "1976-07-10 23:58:52.094649",
                        "1967-10-29 16:00:22.635117",
                    ]
                ),
                ", 6",
            ),
            id="integers-microsecond_scale",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        15.411,
                        None,
                        1234567890123.456789012,
                        314159265.3589793,
                        -98123456789.01234,
                    ]
                ),
                pd.Series(
                    [
                        "1970-01-01 00:00:00.015411",
                        None,
                        "2009-02-13 23:31:30.123456768",
                        "1970-01-04 15:15:59.265358979",
                        "1966-11-22 07:29:03.210987648",
                    ]
                ),
                ", 3",
            ),
            id="floats-millisecond_scale",
        ),
    ]
)
def to_timestamp_numeric_data(request):
    """Same as to_timestamp_non_numeric_data except for numeric arguments.
    Instead of providing an old timezone, if there is one, provides
    the string to use for the scale (e.g. '' for no scale, ', 9' for ns,
    etc.)"""
    return request.param


def test_to_timestamp_numeric(
    to_timestamp_fn, to_timestamp_numeric_data, local_tz, memory_leak_check
):
    use_case = to_timestamp_fn in {
        "TO_TIMESTAMP",
        "TO_TIMESTAMP_LTZ",
        "TRY_TO_TIMESTAMP_NTZ",
        "TRY_TO_TIMESTAMP_TZ",
    }
    data, answer, scale_str = to_timestamp_numeric_data
    if to_timestamp_fn.endswith("_LTZ") or to_timestamp_fn.endswith("_TZ"):
        tz = local_tz
    else:
        tz = None
    if use_case:
        query = f"SELECT CASE WHEN b THEN NULL ELSE {to_timestamp_fn}(t{scale_str}) END FROM table1"
    else:
        query = f"SELECT {to_timestamp_fn}(t{scale_str}) FROM table1"
    ctx = {
        "table1": pd.DataFrame({"t": data, "b": [i % 5 == 2 for i in range(len(data))]})
    }
    expected_output = pd.DataFrame(
        {0: pd.Series([None if s is None else pd.Timestamp(s, tz=tz) for s in answer])}
    )
    if use_case:
        expected_output[0][ctx["table1"]["b"]] = None
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_names=False,
        only_jit_1DVar=True,
    )
