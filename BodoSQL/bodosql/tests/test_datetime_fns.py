# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL dateime functions with BodoSQL
"""

import bodosql
import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query

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
def dt_fn_dataframe():
    dt_strings = [
        "2011-01-01",
        "1971-02-02",
        "2021-03-03",
        "2021-05-31",
        "2020-12-01T13:56:03.172",
        "2007-01-01T03:30",
        "2001-12-01T12:12:02.21",
        "2100-10-01T13:00:33.1",
    ]
    timestamps = pd.Series([np.datetime64(x) for x in dt_strings])
    normalized_ts = timestamps.dt.normalize()
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
                np.timedelta64(4, "s"),
                np.timedelta64(3, "ms"),
                np.timedelta64(2000000, "us"),
            ],
            "datetime_strings": dt_strings,
            "positive_integers": [1, 2, 31, 400, 123, 13, 7, 80],
            "small_positive_integers": [1, 2, 3, 4, 5, 6, 7, 8],
            "dt_format_strings": [
                "%Y",
                "%a, %b, %c",
                "%D, %d, %f, %p, %S",
                "%Y, %M, %D",
                "%y",
                "%T",
                "%r",
                "%j",
            ],
            "valid_year_integers": [2000, 2100, 1999, 2020, 2021, 1998, 2200, 2012],
            "mixed_integers": [0, 1, -2, 3, -4, 5, -6, 7],
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
            marks=pytest.mark.skip(
                "Requires engine fix for overflow issue with pd.Timestamp.floor, see [BE-1022]"
            ),
        ),
        pytest.param(
            ("CURRENT_DATE", [], ("TIMESTAMP 2020-04-25", "TIMESTAMP 2020-04-25")),
            marks=pytest.mark.skip(
                "Requires engine fix for overflow issue with pd.Timestamp.floor, see [BE-1022]"
            ),
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
        dt_fn_dataframe,
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


@pytest.mark.slow
def test_to_date_invalid_col_check(spark_info, dt_fn_dataframe, memory_leak_check):
    import bodo

    query = f"SELECT TO_DATE(mixed_integers) from table1"

    msg = r".*TO_DATE cannot be called on an operand of type: BIGINT.*"
    with pytest.raises(bodo.utils.typing.BodoError, match=msg):
        bc = bodosql.BodoSQLContext(dt_fn_dataframe)
        bc.sql(query)


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
            .year.apply(lambda x: 1 if x > 2015 else 0)
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
