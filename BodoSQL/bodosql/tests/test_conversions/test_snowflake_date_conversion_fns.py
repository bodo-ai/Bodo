"""
Test correctness of Snowflake TO_X functions for date-related casting in BodoSQL
"""

import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.timezone_common import representative_tz  # noqa
from bodosql.tests.test_datetime_fns import dt_fn_dataframe  # noqa
from bodosql.tests.test_kernels.test_snowflake_date_conversion_array_kernels import (  # pragma: no cover
    scalar_to_date_equiv_fn,
)
from bodosql.tests.utils import check_query, make_tables_nullable


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
        pytest.param("DATE", id="date"),
        pytest.param("TRY_TO_DATE", id="try_to_date"),
        pytest.param("TO_DATE", id="to_date", marks=pytest.mark.slow),
    ]
)
def test_fn(request):
    """
    Three different date casting function names
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param("DATETIME_STRINGS", id="valid_datetime_strings"),
        pytest.param("DIGIT_STRINGS", id="valid_digit_strings"),
        pytest.param("TIMESTAMPS", id="valid_timestamps"),
    ]
)
def date_casting_input_type(request):
    """
    Different input types in dt_fn_dataframe
    """
    return request.param


def test_date_casting_functions(
    dt_fn_dataframe, test_fn, date_casting_input_type, memory_leak_check
):
    """tests DATE/TO_DATE/TRY_TO_DATE on valid datetime string/digit string/timestamp values"""
    query = f"SELECT {test_fn}({date_casting_input_type}) from table1"

    expected_output = pd.DataFrame(
        {
            "FOO": dt_fn_dataframe["TABLE1"][date_casting_input_type].apply(
                lambda val: pd.NA
                if scalar_to_date_equiv_fn(val) is None
                else scalar_to_date_equiv_fn(val)
            )
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_date_casting_functions_case(
    dt_fn_dataframe, test_fn, date_casting_input_type, memory_leak_check
):
    """
    tests DATE/TO_DATE/TRY_TO_DATE on valid datetime string/digit string/timestamp values in a case statement
    """
    query = (
        f"SELECT CASE WHEN {test_fn}({date_casting_input_type}) < DATE '2013-01-03' "
        f"THEN {test_fn}({date_casting_input_type}) END from table1"
    )

    dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
    expected_output = pd.DataFrame(
        {
            "FOO": dt_fn_dataframe_nullable["TABLE1"][date_casting_input_type].apply(
                lambda val: scalar_to_date_equiv_fn(val)
                if scalar_to_date_equiv_fn(val) is not None
                and (scalar_to_date_equiv_fn(val) < pd.Timestamp("2013-01-03").date())
                else pd.NA
            )
        }
    )
    check_query(
        query,
        dt_fn_dataframe_nullable,
        None,
        check_names=False,
        expected_output=expected_output,
    )


def test_date_casting_functions_tz_aware(test_fn, memory_leak_check):
    """tests DATE/TO_DATE/TRY_TO_DATE on valid timestamp with timezone values"""
    df = pd.DataFrame(
        {
            "TIMESTAMPS": pd.date_range(
                "1/18/2022", periods=20, freq="10D5h", tz="US/Pacific", unit="ns"
            )
        }
    )
    ctx = {"TABLE1": df}
    query = f"SELECT {test_fn}(timestamps) as dates from table1"
    expected_output = pd.DataFrame({"DATES": df["TIMESTAMPS"].dt.date})

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


def test_date_casting_functions_tz_aware_case(test_fn, memory_leak_check):
    """tests DATE/TO_DATE/TRY_TO_DATE on valid datetime values in a case statement"""
    df = pd.DataFrame(
        {
            "TIMESTAMPS": pd.date_range(
                "1/18/2022", periods=30, freq="10D5h", tz="US/Pacific", unit="ns"
            ),
            "B": [True, False, True, False, True] * 6,
        }
    )
    ctx = {"TABLE1": df}
    query = (
        f"SELECT CASE WHEN B THEN {test_fn}(timestamps) END as timestamps from table1"
    )
    to_date_series = df["TIMESTAMPS"].dt.normalize().apply(lambda t: t.date())
    to_date_series[~df.B] = None
    expected_output = pd.DataFrame({"TIMESTAMPS": to_date_series})

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


def test_try_to_date_invalid_strings(tz_aware_df, memory_leak_check):
    """tests try_to_date on valid and invalid datetime values"""

    # Construct input dataframe of both valid and invalid datetime strings
    valid_datetimes = tz_aware_df["TABLE1"]["A"]

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
            "TIMESTAMPS": pd.concat(
                [valid_datetimes.astype(str), invalid_str_datetimes]
            ).reset_index(drop=True)
        }
    )

    # Construct expected answer using Pandas
    valid_answers = valid_datetimes.dt.date
    invalid_answers = pd.Series([pd.NA] * len(invalid_str_datetimes))
    expected_output = pd.DataFrame(
        {"DATES": pd.concat([valid_answers, invalid_answers]).reset_index(drop=True)}
    )

    ctx = {"TABLE1": df}
    query = "SELECT TRY_TO_DATE(timestamps) as dates from table1"

    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


# [BE-3774] Leaks Memory
def test_date_casting_functions_invalid_args(dt_fn_dataframe, test_fn):
    """
    tests arguments which cause NA in try_to_date, and throw an error for DATE/TO_DATE/TRY_TO_DATE
    """

    query = f"SELECT {test_fn}(invalid_dt_strings) from table1"

    if test_fn == "TRY_TO_DATE":
        expected_output = pd.DataFrame(
            {"FOO": pd.Series([pd.NA] * len(dt_fn_dataframe["TABLE1"]))}
        )
        dt_fn_dataframe_nullable = make_tables_nullable(dt_fn_dataframe)
        check_query(
            query,
            dt_fn_dataframe_nullable,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=expected_output,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(Exception, match=msg):
            bc = bodosql.BodoSQLContext(dt_fn_dataframe)
            bc.sql(query)


@pytest.fixture
def format_input_string_df():
    """
    Fixture containing a representative set of datetime.date object
    for use in testing, including None object.
    """
    return {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        "2017-03-26",
                        "2000-12-31",
                        "2003-09-06",
                        "2023-03-06",
                        "1980-10-14",
                        None,
                    ]
                    * 4
                ),
                "B": pd.Series(
                    [
                        "26-Mar-2017",
                        "31-Dec-2000",
                        "06-Sep-2003",
                        "06-Mar-2023",
                        "14-Oct-1980",
                        None,
                    ]
                    * 4
                ),
                "C": pd.Series(
                    [
                        "03/26/2017",
                        "12/31/2000",
                        "09/06/2003",
                        "03/06/2023",
                        "10/14/1980",
                        None,
                    ]
                    * 4
                ),
                "D": pd.Series(
                    [
                        "___March__26____17__",
                        "___December__31____00__",
                        "___September__06____03__",
                        "___March__06____23__",
                        "___October__14____80__",
                        None,
                    ]
                    * 4
                ),
                "E": pd.Series(
                    [
                        "Mon@#$26@#$2017@#$Mar",
                        "Tue@#$31@#$2000@#$Dec",
                        "Wed@#$06@#$2003@#$Sep",
                        "Thu@#$06@#$2023@#$Mar",
                        "Fri@#$14@#$1980@#$Oct",
                        None,
                    ]
                    * 4
                ),
            }
        )
    }


@pytest.mark.parametrize(
    "input_col, format_str",
    [
        pytest.param("A", "YYYY-MM-DD", id="YYYY-MM-DD"),
        pytest.param("B", "DD-MON-YYYY", id="DD-MON-YYYY"),
        pytest.param("C", "MM/DD/YYYY", id="MM/DD/YYYY", marks=pytest.mark.slow),
        pytest.param(
            "D", "___MMMM__DD____YY__", id="___MMMM__DD____YY__", marks=pytest.mark.slow
        ),
        pytest.param("E", "DY@#$DD@#$YYYY@#$MON", id="DY@#$DD@#$YYYY@#$MON"),
    ],
)
def test_date_casting_functions_with_valid_format(
    format_input_string_df, test_fn, input_col, format_str, memory_leak_check
):
    """
    Tests DATE/TO_DATE/TRY_TO_DATE with valid format strings
    """
    query = f"SELECT {test_fn}({input_col}, '{format_str}') from table1"
    expected_output = pd.DataFrame(
        {
            "FOO": pd.Series(
                [
                    datetime.date(2017, 3, 26),
                    datetime.date(2000, 12, 31),
                    datetime.date(2003, 9, 6),
                    datetime.date(2023, 3, 6),
                    datetime.date(1980, 10, 14),
                    None,
                ]
                * 4
            )
        }
    )

    check_query(
        query,
        format_input_string_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


# [BE-3774] Leaks Memory
@pytest.mark.parametrize(
    "input_col, format_str",
    [
        pytest.param("A", "MM/DD/YYYY", id="MM/DD/YYYY"),
        pytest.param("B", "YYYY-MM-DD", id="YYYY-MM-DD", marks=pytest.mark.slow),
        pytest.param("C", "DD-MON-YYYY", id="DD-MON-YYYY"),
        pytest.param("D", "DY@#$DD@#$YYYY@#$MON", id="DY@#$DD@#$YYYY@#$MON"),
        pytest.param(
            "E", "___MMMM__DD____YY__", id="___MMMM__DD____YY__", marks=pytest.mark.slow
        ),
    ],
)
def test_date_casting_functions_with_invalid_format(
    format_input_string_df, test_fn, input_col, format_str
):
    """
    Tests DATE/TO_DATE/TRY_TO_DATE can throw correct error when input strings don't match with format strings
    """
    query = f"SELECT {test_fn}({input_col}, '{format_str}') from table1"

    if test_fn == "TRY_TO_DATE":
        expected_output = pd.DataFrame({"FOO": pd.Series([pd.NA] * 24)})
        check_query(
            query,
            format_input_string_df,
            None,
            check_names=False,
            check_dtype=False,
            expected_output=expected_output,
        )
    else:
        msg = "Invalid input while converting to date value"
        with pytest.raises(Exception, match=msg):
            bc = bodosql.BodoSQLContext(format_input_string_df)
            bc.sql(query)


def test_date_casting_with_colon(
    dt_fn_dataframe, date_casting_input_type, memory_leak_check
):
    """tests ::DATE on valid datetime string/digit string/timestamp values"""
    query = f"SELECT {date_casting_input_type}::DATE from table1"

    expected_output = pd.DataFrame(
        {
            "FOO": dt_fn_dataframe["TABLE1"][date_casting_input_type].apply(
                lambda val: pd.NA
                if scalar_to_date_equiv_fn(val) is None
                else scalar_to_date_equiv_fn(val)
            )
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_date_casting_with_colon_tz_aware(memory_leak_check):
    """tests ::DATE on valid datetime values in a case statment"""
    df = pd.DataFrame(
        {
            "TIMESTAMPS": pd.date_range(
                "1/18/2022", periods=20, freq="10D5h", tz="US/Pacific", unit="ns"
            )
        }
    )
    ctx = {"TABLE1": df}
    query = "SELECT (timestamps)::DATE from table1"
    expected_output = pd.DataFrame({"FOO": df["TIMESTAMPS"].dt.date})

    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


# [BE-3774] Leaks Memory
def test_date_casting_with_colon_invalid_args(dt_fn_dataframe):
    """tests ::DATE throws correct error for invalid inputs"""

    query = "SELECT (invalid_dt_strings)::DATE from table1"

    msg = "Invalid input while converting to date value"
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
                ],
                dtype="datetime64[ns]",
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
                ],
                dtype="datetime64[ns, Australia/Sydney]",
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
        pytest.param("TRY_TO_TIMESTAMP", marks=pytest.mark.slow),
        pytest.param("TRY_TO_TIMESTAMP_NTZ", marks=pytest.mark.slow),
        pytest.param("TRY_TO_TIMESTAMP_LTZ", marks=pytest.mark.slow),
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
        # Any cast to TIMESTAMP_LTZ will be optimized out, so we convert the input here.
        # TIMESTAMP_LTZ is only defined to work with the local timezone.
        if old_tz is not None:
            data = data.dt.tz_localize(None).dt.tz_localize(local_tz)
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
        "TABLE1": pd.DataFrame({"T": data, "B": [i % 5 == 4 for i in range(len(data))]})
    }
    expected_output = pd.DataFrame(
        {
            0: pd.Series(
                [None if pd.isna(s) else pd.Timestamp(s, tz=tz) for s in answer],
                dtype=f"datetime64[ns, {tz}]" if tz else "datetime64[ns]",
            )
        }
    )
    if use_case:
        expected_output[0] = expected_output[0].where(~ctx["TABLE1"]["B"], other=None)

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
        "TABLE1": pd.DataFrame({"T": data, "B": [i % 5 == 2 for i in range(len(data))]})
    }
    expected_output = pd.DataFrame(
        {
            0: pd.Series(
                [None if pd.isna(s) else pd.Timestamp(s, tz=tz) for s in answer],
                dtype=f"datetime64[ns, {tz}]" if tz else "datetime64[ns]",
            )
        }
    )
    if use_case:
        expected_output[0] = expected_output[0].where(~ctx["TABLE1"]["B"], other=None)
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
                        "08/17/2000 06:45:00",
                        "12/31/2010 12:12:12",
                        "01/03/1970 23:10:01",
                        "02/28/2016 10:10:10",
                    ]
                    * 4,
                ),
                "MM/DD/YYYY HH24:MI:SS",
                pd.Series(
                    [
                        pd.Timestamp(2000, 8, 17, 6, 45, 0),
                        pd.Timestamp(2010, 12, 31, 12, 12, 12),
                        pd.Timestamp(1970, 1, 3, 23, 10, 1),
                        pd.Timestamp(2016, 2, 28, 10, 10, 10),
                    ]
                    * 4,
                    dtype="datetime64[ns]",
                ),
            ),
            id="format-1",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        "2022-11-17 08:26:51 AM",
                        "2023-04-05 03:59:14 PM",
                        "2021-09-29 11:12:27 AM",
                        "2023-02-14 06:37:05 PM",
                    ]
                    * 4,
                ),
                "YYYY-MM-DD HH12:MI:SS PM",
                pd.Series(
                    [
                        pd.Timestamp(2022, 11, 17, 8, 26, 51),
                        pd.Timestamp(2023, 4, 5, 15, 59, 14),
                        pd.Timestamp(2021, 9, 29, 11, 12, 27),
                        pd.Timestamp(2023, 2, 14, 18, 37, 5),
                    ]
                    * 4,
                    dtype="datetime64[ns]",
                ),
            ),
            id="format-2",
        ),
    ]
)
def to_timestamp_string_data_format_str(request):
    """
    String data with format strings to be converted to timestamp types.
    """
    return request.param


def test_to_timestamp_format_str(
    to_timestamp_fn, to_timestamp_string_data_format_str, memory_leak_check
):
    """
    Test TO_TIMESTAMP with optional format string argument
    """
    data, format_str, answer = to_timestamp_string_data_format_str
    query = f"SELECT {to_timestamp_fn}(t, '{format_str}') FROM table1"

    # Set proper timezone for the answer if LTZ
    if to_timestamp_fn in ("TO_TIMESTAMP_LTZ", "TRY_TO_TIMESTAMP_LTZ"):
        answer = answer.astype(pd.ArrowDtype(pa.timestamp("ns", "UTC")))

    ctx = {"TABLE1": pd.DataFrame({"T": data})}
    expected_output = pd.DataFrame({0: answer})
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "convert_calc, data, session_tz, answer",
    [
        pytest.param(
            "CONVERT_TIMEZONE('America/Los_Angeles', 'America/New_York', T)",
            pd.Series(
                [
                    pd.Timestamp("2024-01-01 12:00:00"),
                    pd.Timestamp("2024-02-04 11:15:10"),
                    None,
                    pd.Timestamp("2024-07-09 14:30:00"),
                    pd.Timestamp("2024-08-16 13:45:20"),
                ],
                dtype="datetime64[ns]",
            ),
            None,
            pd.Series(
                [
                    pd.Timestamp("2024-01-01 15:00:00"),
                    pd.Timestamp("2024-02-04 14:15:10"),
                    None,
                    pd.Timestamp("2024-07-09 17:30:00"),
                    pd.Timestamp("2024-08-16 16:45:20"),
                ],
                dtype="datetime64[ns]",
            ),
            id="3_arg-ntz-west_to_east",
        ),
        pytest.param(
            "CONVERT_TIMEZONE('UTC', 'America/New_York', T)",
            pd.Series(
                [
                    pd.Timestamp("2024-01-01 12:00:00", tz="America/Los_Angeles"),
                    pd.Timestamp("2024-02-04 11:15:10", tz="America/Los_Angeles"),
                    None,
                    pd.Timestamp("2024-07-09 14:30:00", tz="America/Los_Angeles"),
                    pd.Timestamp("2024-08-16 16:45:20", tz="America/Los_Angeles"),
                ],
                dtype="datetime64[ns, America/Los_Angeles]",
            ),
            "America/Los_Angeles",
            pd.Series(
                [
                    pd.Timestamp("2024-01-01 07:00:00"),
                    pd.Timestamp("2024-02-04 06:15:10"),
                    None,
                    pd.Timestamp("2024-07-09 10:30:00"),
                    pd.Timestamp("2024-08-16 12:45:20"),
                ],
                dtype="datetime64[ns]",
            ),
            id="3_arg-ltz_west-utc_to_east",
        ),
        pytest.param(
            "TO_CHAR(CONVERT_TIMEZONE('Europe/Berlin', T))",
            np.array(
                [
                    bodo.types.TimestampTZ.fromLocal("2024-01-01 12:00:00", 0),
                    bodo.types.TimestampTZ.fromLocal("2024-02-04 11:15:10", 60),
                    None,
                    bodo.types.TimestampTZ.fromLocal("2024-07-09 14:30:00", 195),
                    bodo.types.TimestampTZ.fromLocal("2024-08-16 13:45:20", -330),
                ]
            ),
            None,
            pd.Series(
                [
                    "2024-01-01 13:00:00 +0100",
                    "2024-02-04 11:15:10 +0100",
                    None,
                    "2024-07-09 13:15:00 +0200",
                    "2024-08-16 21:15:20 +0200",
                ]
            ),
            id="2_arg-tz-berlin",
        ),
        pytest.param(
            "TO_CHAR(CONVERT_TIMEZONE('America/Los_Angeles', T))",
            np.array(
                [
                    datetime.date(2024, 1, 2),
                    datetime.date(2024, 2, 15),
                    None,
                    datetime.date(2024, 4, 10),
                    datetime.date(2024, 6, 25),
                ]
            ),
            "America/New_York",
            pd.Series(
                [
                    "2024-01-01 21:00:00 -0800",
                    "2024-02-14 21:00:00 -0800",
                    None,
                    "2024-04-09 21:00:00 -0700",
                    "2024-06-24 21:00:00 -0700",
                ]
            ),
            id="2_arg-date-east_to_west",
        ),
    ],
)
def test_convert_timezone(convert_calc, data, session_tz, answer, memory_leak_check):
    """
    Tests the Snowflake function CONVERT_TIMEZONE. See documentation for description
    since it has several convoluted rules: https://docs.snowflake.com/en/sql-reference/functions/convert_timezone

    Answers verified on Snowflake
    """
    query = f"SELECT {convert_calc} as A from table1"
    ctx = {"TABLE1": pd.DataFrame({"T": data})}
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({"A": answer}),
        session_tz=session_tz,
        enable_timestamp_tz=True,
    )
