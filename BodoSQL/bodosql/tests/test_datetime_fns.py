# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL dateime functions with BodoSQL
"""

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

from bodo.tests.timezone_common import representative_tz  # noqa

EQUIVALENT_SPARK_DT_FN_MAP = {
    "WEEK": "WEEKOFYEAR",
    "CURDATE": "CURRENT_DATE",
}


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
        "DATEADD",
        "DATE_ADD",
        pytest.param("ADDDATE", marks=pytest.mark.slow),
    ]
)
def adddate_equiv_fns(request):
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
            "digit_strings": [
                None,
                "-13",
                "0",
                "-2",
                "5",
                "-66",
                "42",
                None,
                "1234",
                None,
            ],
            "valid_year_integers": pd.Series(
                [
                    2000,
                    2100,
                    None,
                    1999,
                    2020,
                    None,
                    2021,
                    1998,
                    2200,
                    2012,
                ],
                dtype=pd.Int64Dtype(),
            ),
            "mixed_integers": pd.Series(
                [None, 0, 1, -2, 3, -4, 5, -6, 7, None], dtype=pd.Int64Dtype()
            ),
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
            "DAYOFWEEK",
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


@pytest.fixture(
    params=[
        "CURRENT_TIMESTAMP",
        pytest.param("LOCALTIME", marks=pytest.mark.slow),
        pytest.param("LOCALTIMESTAMP", marks=pytest.mark.slow),
        pytest.param("NOW", marks=pytest.mark.slow),
    ]
)
def now_equivalent_fns(request):
    return request.param


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT A, GETDATE() from table1",
            id="no_case-just_getdate",
        ),
        pytest.param(
            "SELECT A, GETDATE() - interval '6' months from table1",
            id="no_case-minus_interval",
        ),
        pytest.param(
            "SELECT A, GETDATE() + interval '5' days from table1",
            id="no_case-plus_interval",
        ),
        pytest.param(
            "SELECT A, CASE WHEN EXTRACT(MONTH from GETDATE()) = A then 'y' ELSE 'n' END from table1",
            id="case",
            marks=pytest.mark.skip("[BE-3909] Fix GETDATE inside of CASE"),
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

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        only_jit_1DVar=True,
    )


def test_now_equivalents(basic_df, spark_info, now_equivalent_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the current time as a timestamp
    This one needs special handling, as the timestamps returned by each call will be
    slightly different, depending on when the function was run.
    """
    query = f"SELECT A, EXTRACT(DAY from {now_equivalent_fns}()), (EXTRACT(HOUR from {now_equivalent_fns}()) + EXTRACT(MINUTE from {now_equivalent_fns}()) + EXTRACT(SECOND from {now_equivalent_fns}()) ) >= 1  from table1"
    spark_query = "SELECT A, EXTRACT(DAY from NOW()), (EXTRACT(HOUR from NOW()) + EXTRACT(MINUTE from NOW()) + EXTRACT(SECOND from NOW()) ) >= 1  from table1"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_utc_timestamp(basic_df, spark_info, memory_leak_check):
    """tests utc_timestamp"""
    query = f"SELECT A, EXTRACT(DAY from UTC_TIMESTAMP()), (EXTRACT(HOUR from UTC_TIMESTAMP()) + EXTRACT(MINUTE from UTC_TIMESTAMP()) + EXTRACT(SECOND from UTC_TIMESTAMP()) ) >= 1  from table1"
    expected_output = pd.DataFrame(
        {
            "unkown_name1": basic_df["table1"]["A"],
            "unkown_name2": pd.Timestamp.now().day,
            "unkown_name5": True,
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


@pytest.mark.slow
@pytest.mark.skip(
    "Requires engine fix for overflow issue with pd.Timestamp.floor, see [BE-1022]"
)
def test_utc_date(basic_df, spark_info, memory_leak_check):
    """tests utc_date"""

    query = f"SELECT A, EXTRACT(day from UTC_DATE()), (EXTRACT(HOUR from UTC_DATE()) + EXTRACT(MINUTE from UTC_DATE()) + EXTRACT(SECOND from UTC_DATE()) ) = 0  from table1"
    expected_output = pd.DataFrame(
        {
            "unkown_name1": basic_df["table1"]["A"],
            "unkown_name2": pd.Timestamp.now().day,
            "unkown_name5": True,
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
    # TODO: add addition format charecters when/if they become supported
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


def test_monthname_cols(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests the monthname function on column inputs. Needed since the equivalent function has different syntax"""

    query = "SELECT MONTHNAME(timestamps) from table1"
    spark_query = "SELECT DATE_FORMAT(timestamps, 'MMMM') from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_monthname_scalars(basic_df, spark_info, memory_leak_check):
    """tests the monthname function on scalar inputs. Needed since the equivalent function has different syntax"""

    # since monthname is a fn we defined, don't need to worry about calcite performing optimizations
    query = "SELECT MONTHNAME(TIMESTAMP '2021-03-03'), MONTHNAME(TIMESTAMP '2021-03-13'), MONTHNAME(TIMESTAMP '2021-03-01')"
    spark_query = "SELECT DATE_FORMAT('2021-03-03', 'MMMM'), DATE_FORMAT('2021-03-13', 'MMMM'), DATE_FORMAT('2021-03-01', 'MMMM')"

    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
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
                    "DW": [None, 1, 7, 4, 3],
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
                    "DW": [None, 1, 7, 4, None],
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


@pytest.mark.skip(
    "Need pd.Timedelta * integer series support for column case, see BE-1054"
)
def test_timestamp_add_cols(
    spark_info, mysql_interval_str, dt_fn_dataframe, memory_leak_check
):
    query = f"SELECT timestampadd({mysql_interval_str}, small_positive_integers, timestamps) from table1"
    spark_interval = make_spark_interval(mysql_interval_str, "small_positive_integers")
    spark_query = f"SELECT timestamps + {spark_interval} from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.fixture
def dateadd_df():
    """Returns the context used by test snowflake_dateadd"""
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
                "DATEADD({!r}, col_int, col_dt)",
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
                "DATEADD({!r}, col_int, col_dt)",
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
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!r}, -25, col_dt) END",
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


def test_snowflake_dateadd(dateadd_df, dateadd_queries, spark_info, memory_leak_check):
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
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=answers,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_timestamp_add_scalar(
    spark_info, mysql_interval_str, dt_fn_dataframe, memory_leak_check
):
    query = f"SELECT CASE WHEN timestampadd({mysql_interval_str}, small_positive_integers, timestamps) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE timestampadd({mysql_interval_str}, small_positive_integers, timestamps) END from table1"
    spark_interval = make_spark_interval(mysql_interval_str, "small_positive_integers")
    spark_query = f"SELECT CASE WHEN ADD_TS < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE ADD_TS END from (SELECT timestamps + {spark_interval} as ADD_TS from table1)"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_adddate_cols_int_arg1(
    adddate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    "tests that date_add/adddate works when the second argument is an integer, on column values"
    query = f"SELECT {adddate_equiv_fns}({timestamp_date_string_cols}, positive_integers) from table1"
    spark_query = (
        f"SELECT DATE_ADD({timestamp_date_string_cols}, positive_integers) from table1"
    )

    # spark requires certain arguments of adddate to not
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
def test_adddate_scalar_int_arg1(
    adddate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    "tests that date_add/adddate works when the second argument is an integer, on scalar values"

    # Spark's date_add seems to truncate everything after the day in the scalar case, so we use normalized the output timestamp for bodosql
    query = f"SELECT CASE WHEN {adddate_equiv_fns}({timestamp_date_string_cols}, positive_integers) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE TO_DATE(ADDDATE({timestamp_date_string_cols}, positive_integers)) END from table1"
    spark_query = f"SELECT CASE WHEN DATE_ADD({timestamp_date_string_cols}, positive_integers) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE DATE_ADD({timestamp_date_string_cols}, positive_integers) END from table1"

    # spark requires certain arguments of adddate to not
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


def test_adddate_cols_td_arg1(
    adddate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    """tests that date_add/adddate works on timedelta 2nd arguments, with column inputs"""
    query = f"SELECT {adddate_equiv_fns}({timestamp_date_string_cols}, intervals) from table1"

    expected_output = pd.DataFrame(
        {
            "unknown_column_name": pd.to_datetime(
                dt_fn_dataframe["table1"][timestamp_date_string_cols]
            )
            + dt_fn_dataframe["table1"]["intervals"]
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
def test_adddate_td_scalars(
    adddate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    """tests that adddate works on timedelta 2nd arguments, with scalar inputs"""
    query = f"SELECT CASE WHEN {adddate_equiv_fns}({timestamp_date_string_cols}, intervals) < TIMESTAMP '1700-01-01' THEN TIMESTAMP '1970-01-01' ELSE {adddate_equiv_fns}({timestamp_date_string_cols}, intervals) END from table1"

    expected_output = pd.DataFrame(
        {
            "unknown_column_name": pd.to_datetime(
                dt_fn_dataframe["table1"][timestamp_date_string_cols]
            )
            + dt_fn_dataframe["table1"]["intervals"]
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


def test_yearweek(spark_info, dt_fn_dataframe, memory_leak_check):
    query = "SELECT YEARWEEK(timestamps) from table1"
    spark_query = "SELECT YEAR(timestamps) * 100 + WEEKOFYEAR(timestamps) from table1"

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


@pytest.mark.parametrize(
    "literal_str",
    [
        "MONTH",
        "WEEK",
        "DAY",
        "HOUR",
        "MINUTE",
        "SECOND",
        # Spark doesn't support millisecond, microsecond, or nanosecond.
        # TODO: Test
    ],
)
def test_date_trunc(spark_info, dt_fn_dataframe, literal_str, memory_leak_check):
    query = f"SELECT DATE_TRUNC('{literal_str}', TIMESTAMPS) as A from table1"
    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
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


def test_tz_aware_next_day(representative_tz, memory_leak_check):
    query = "SELECT next_day(A, B) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
        }
    )
    ctx = {"table1": df}
    out_series = df.apply(
        lambda row: row["A"].normalize()
        + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1),
        axis=1,
    )
    py_output = pd.DataFrame({"m": out_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.skip("[BE-4022] Support tz-aware data as the output of case")
def test_tz_aware_next_day_case(
    representative_tz,
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN next_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"table1": df}
    week_series = df.apply(
        lambda row: row["A"].normalize()
        + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"m": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


def test_tz_aware_previous_day(representative_tz, memory_leak_check):
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
        lambda row: row["A"].normalize()
        - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1),
        axis=1,
    )
    py_output = pd.DataFrame({"m": out_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.skip("[BE-4022] Support tz-aware data as the output of case")
def test_tz_aware_previous_day_case(
    representative_tz,
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN previous_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5H", periods=30, tz=representative_tz
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"table1": df}
    week_series = df.apply(
        lambda row: row["A"].normalize()
        - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"m": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)
