"""
Test correctness of SQL datetime functions with BodoSQL
"""

import datetime

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.conftest import (  # noqa
    date_df,
    day_part_strings,
    time_df,
    time_part_strings,
)
from bodo.tests.timezone_common import (  # noqa
    representative_tz,
)
from bodo.tests.utils import (
    dist_IR_contains,
    pytest_slow_unless_codegen,
)
from bodo.tests.utils_jit import DistTestPipeline
from bodosql.kernels.datetime_array_kernels import (
    standardize_snowflake_date_time_part_compile_time,
)
from bodosql.tests.test_kernels.test_datetime_array_kernels import (
    last_day_scalar_fn,
)
from bodosql.tests.timezone_utils import (
    generate_date_trunc_date_func,
    generate_date_trunc_func,
    generate_date_trunc_time_func,
)
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


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
        "TIMESTAMPS",
        pytest.param("TIMESTAMPS_NORMALIZED", marks=pytest.mark.slow),
        "DATETIME_STRINGS",
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
    # "1D2h37min48s" --> 1 day, 2 hours, 37 minutes, 48 seconds
    to_dst_series = pd.date_range(
        start="11/3/2021", freq="1D2h37min48s", periods=30, tz="US/Pacific", unit="ns"
    ).to_series()

    # Transition back from Daylight Savings
    from_dst_series = pd.date_range(
        start="03/1/2022", freq="0D12h30min1s", periods=60, tz="US/Pacific", unit="ns"
    ).to_series()

    # February is weird with leap years
    feb_leap_year_series = pd.date_range(
        start="02/20/2020", freq="1D0h30min0s", periods=20, tz="US/Pacific", unit="ns"
    ).to_series()

    second_quarter_series = pd.date_range(
        start="05/01/2015", freq="2D0h1min59s", periods=20, tz="US/Pacific", unit="ns"
    ).to_series()

    third_quarter_series = pd.date_range(
        start="08/17/2000", freq="10D1h1min10s", periods=20, tz="US/Pacific", unit="ns"
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

    return {"TABLE1": df}


@pytest.fixture
def dt_fn_dataframe():
    dt_strings = [
        "2011-01-01",
        "1971-02-02",
        "2021-03-03",
        "2021-05-31",
        None,
        "2020-12-01T13:56:03.172345689",
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
            "TIMESTAMPS": timestamps,
            "TIMESTAMPS_NORMALIZED": normalized_ts,
            "DATETIME_STRINGS": dt_strings,
            "INVALID_DT_STRINGS": invalid_dt_strings,
            "POSITIVE_INTEGERS": pd.Series(
                [1, 2, 31, 400, None, None, 123, 13, 7, 80], dtype=pd.Int64Dtype()
            ),
            "SMALL_POSITIVE_INTEGERS": pd.Series(
                [1, 2, 3, None, 4, 5, 6, None, 7, 8], dtype=pd.Int64Dtype()
            ),
            "DT_FORMAT_STRINGS": [
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
            "VALID_YEAR_INTEGERS": pd.Series(
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
            "MIXED_INTEGERS": pd.Series(
                [None, 0, 1, -2, 3, -4, 5, -6, 7, None], dtype=pd.Int64Dtype()
            ),
            "DIGIT_STRINGS": [
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
            "DAYS_OF_WEEK": [
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
    return {"TABLE1": df}


@pytest.fixture(
    params=[
        pytest.param((x, ["timestamps"], ("1", "2")), id=x, marks=pytest.mark.slow)
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
        pytest.param((x, ["timestamps"], ("1", "2")), id=x, marks=pytest.mark.slow)
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
            marks=pytest.mark.slow,
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
    """fixture that returns information used to test datetime functions
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
            "SELECT A, DATEFN from table1",
            id="no_case-just_getdate",
        ),
        pytest.param(
            "SELECT A, DATEFN - interval '6' months from table1",
            id="no_case-minus_interval-month",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT A, DATEFN + interval '5' weeks from table1",
            id="no_case-plus_interval-week",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT A, DATEFN - interval '8 weeks' from table1",
            id="no_case-minus_interval-week-sf-syntax",
        ),
        pytest.param(
            "SELECT A, DATEFN - interval '8' weeks from table1",
            id="no_case-minus_interval-week",
        ),
        pytest.param(
            "SELECT A, DATEFN + interval '5' days from table1",
            id="no_case-plus_interval-day",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT A, CASE WHEN EXTRACT(MONTH from DATEFN) = A then 'y' ELSE 'n' END from table1",
            id="case",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_getdate(query, spark_info, memory_leak_check):
    """Tests the snowflake GETDATE() function"""

    # Snowflake's GETDATE is equivalent to spark's CURRENT_TIMESTAMP (when
    # spark's timezone is set to UTC), not CURRENT_DATE. However, comparing the
    # current timestamp isn't useful, because there will be some delay between
    # the BodoSQL run and the spark run. To get around this, we need to use
    # `TO_DATE` to truncate first.
    # Ideally we should be able to mock out the clock source and test GETDATE
    # in isolation.
    spark_query = query.replace("DATEFN", "CURRENT_DATE()")
    query = query.replace("DATEFN", "TO_DATE(GETDATE())")
    ctx = {
        "TABLE1": pd.DataFrame(
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


def test_getdate_dist_len(spark_info, memory_leak_check):
    """Make sure GETDATE() doesn't create a distributed reduction (to avoid streaming
    hang)
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {"A": pd.Series(list(range(1, 13)), dtype=pd.Int32Dtype())}
        )
    }

    query = "select distinct A, to_date(getdate()) as B from table1"
    spark_query = query.replace("to_date(getdate())", "current_date()")

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        only_jit_1DVar=True,
    )

    # make sure scalar to array conversion doesn't require distributed len which leads
    # to dist_reduce() generation
    @bodo.jit(pipeline_class=DistTestPipeline, all_args_distributed_varlength=True)
    def bodo_func(bc, query):
        return bc.sql(query)

    bodo_func(bodosql.BodoSQLContext(ctx), query)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert not dist_IR_contains(f_ir, "dist_reduce")


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
    As the results depend on when the function was run, we pre compute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  DATE_TRUNC('DAY', {now_equiv_fns}()) AS date_trunc_res, "
        f"  EXTRACT(DAY from {now_equiv_fns}()) IN ({valid_days}) AS is_valid_day, "
        f"  EXTRACT(HOUR from {now_equiv_fns}()) IN ({valid_hours}) AS is_valid_hour, "
        f"  EXTRACT(MINUTE from {now_equiv_fns}()) IN ({valid_minutes}) AS is_valid_minute "
        f"FROM table1"
    )
    py_output = pd.DataFrame(
        {
            "A": basic_df["TABLE1"]["A"],
            "DATE_TRUNC_RES": current_time.normalize(),
            "IS_VALID_DAY": True,
            "IS_VALID_HOUR": True,
            "IS_VALID_MINUTE": True,
        }
    )
    # Make sure DATE_TRUNC_RES dtype is ns (not us)
    py_output["DATE_TRUNC_RES"] = py_output["DATE_TRUNC_RES"].astype(
        pd.DatetimeTZDtype(tz="UTC")
    )
    # Note: These tests can be very slow so we just run 1DVar. Slowness is a potential correctness issue.
    check_query(
        query,
        basic_df,
        None,
        expected_output=py_output,
        check_dtype=False,
        only_jit_1DVar=True,
    )


def test_now_equivalents_case(now_equiv_fns, memory_leak_check):
    """Tests the group of equivalent functions which return the current timestamp in case,
    without timezone info from the Snowflake Catalog.
    As the results depend on when the function was run, we pre compute a list of valid times.
    """
    current_time = pd.Timestamp.now(tz="UTC")
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  CASE WHEN A THEN DATE_TRUNC('DAY', {now_equiv_fns}()) END AS date_trunc_res, "
        f"  CASE WHEN A THEN EXTRACT(DAY from {now_equiv_fns}()) IN ({valid_days}) END AS is_valid_day, "
        f"  CASE WHEN A THEN EXTRACT(HOUR from {now_equiv_fns}()) IN ({valid_hours}) END AS is_valid_hour, "
        f"  CASE WHEN A THEN EXTRACT(MINUTE from {now_equiv_fns}()) IN ({valid_minutes}) END AS is_valid_minute "
        f"FROM table1"
    )

    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    ctx = {"TABLE1": df}
    D = pd.Series(
        current_time.normalize(), index=np.arange(len(df)), dtype="datetime64[ns, UTC]"
    )
    D[~df.A] = None
    S = pd.Series(True, index=np.arange(len(df)), dtype="boolean")
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "DATE_TRUNC_RES": D,
            "IS_VALID_DAY": S,
            "IS_VALID_HOUR": S,
            "IS_VALID_MINUTE": S,
        }
    )
    # Make sure DATE_TRUNC_RES dtype is ns (not us)
    py_output["DATE_TRUNC_RES"] = py_output["DATE_TRUNC_RES"].astype(
        pd.DatetimeTZDtype(tz="UTC")
    )
    # Note: These tests can be very slow so we just run 1DVar. Slowness is a potential correctness issue.
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        only_jit_1DVar=True,
    )


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
    As the results depend on when the function was run, we pre compute a list of valid times.
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
            "A": basic_df["TABLE1"]["A"],
            "IS_VALID_HOUR": True,
            "IS_VALID_MINUTE": True,
        }
    )

    # Note: These tests can be very slow so we just run 1DVar. Slowness is a potential correctness issue.
    check_query(
        query,
        basic_df,
        None,
        expected_output=py_output,
        check_dtype=False,
        only_jit_1DVar=True,
    )


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
    ctx = {"TABLE1": df}
    S = pd.Series(True, index=np.arange(len(df)), dtype="boolean")
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "IS_VALID_HOUR": S,
            "IS_VALID_MINUTE": S,
        }
    )
    # Note: These tests can be very slow so we just run 1DVar. Slowness is a potential correctness issue.
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.fixture(
    params=[
        "UTC_TIMESTAMP",
        pytest.param("SYSDATE", marks=pytest.mark.slow),
    ]
)
def sysdate_equiv_fns(request):
    return request.param


@pytest.mark.slow
def test_sysdate_equivalents_cols(
    basic_df, sysdate_equiv_fns, spark_info, memory_leak_check
):
    """
    Tests the group of equivalent functions which return the UTC timestamp.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now()
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  DATE_TRUNC('DAY', {sysdate_equiv_fns}()) AS date_trunc_res, "
        f"  EXTRACT(DAY from {sysdate_equiv_fns}()) IN ({valid_days}) AS is_valid_day, "
        f"  EXTRACT(HOUR from {sysdate_equiv_fns}()) IN ({valid_hours}) AS is_valid_hour, "
        f"  EXTRACT(MINUTE from {sysdate_equiv_fns}()) IN ({valid_minutes}) AS is_valid_minute "
        f"FROM table1"
    )
    py_output = pd.DataFrame(
        {
            "A": basic_df["TABLE1"]["A"],
            "DATE_TRUNC_RES": current_time.normalize(),
            "IS_VALID_DAY": True,
            "IS_VALID_HOUR": True,
            "IS_VALID_MINUTE": True,
        }
    )

    # Note: These tests can be very slow so we just run 1DVar. Slowness is a potential correctness issue.
    check_query(
        query,
        basic_df,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_sysdate_equivalents_case(sysdate_equiv_fns, spark_info, memory_leak_check):
    """
    Tests the group of equivalent functions which return the UTC timestamp in case.
    As the results depend on when the function was run, we precompute a list of valid times.
    """
    current_time = pd.Timestamp.now()
    valid_days, valid_hours, valid_minutes = compute_valid_times(current_time)
    query = (
        f"SELECT A, "
        f"  CASE WHEN A THEN DATE_TRUNC('DAY', {sysdate_equiv_fns}()) END AS date_trunc_res, "
        f"  CASE WHEN A THEN EXTRACT(DAY from {sysdate_equiv_fns}()) IN ({valid_days}) END AS is_valid_day, "
        f"  CASE WHEN A THEN EXTRACT(HOUR from {sysdate_equiv_fns}()) IN ({valid_hours}) END AS is_valid_hour, "
        f"  CASE WHEN A THEN EXTRACT(MINUTE from {sysdate_equiv_fns}()) IN ({valid_minutes}) END AS is_valid_minute "
        f"FROM table1"
    )

    df = pd.DataFrame({"A": [True, False, False, True, True] * 6})
    ctx = {"TABLE1": df}
    D = pd.Series(
        current_time.normalize(), index=np.arange(len(df)), dtype="datetime64[ns]"
    )
    D[~df.A] = None
    S = pd.Series(True, index=np.arange(len(df)), dtype="boolean")
    S[~df.A] = None
    py_output = pd.DataFrame(
        {
            "A": df.A,
            "DATE_TRUNC_RES": D,
            "IS_VALID_DAY": S,
            "IS_VALID_HOUR": S,
            "IS_VALID_MINUTE": S,
        }
    )

    # Make sure DATE_TRUNC_RES dtype is ns (not us)
    py_output["DATE_TRUNC_RES"] = py_output["DATE_TRUNC_RES"].astype("datetime64[ns]")

    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_utc_date(basic_df, spark_info, memory_leak_check):
    """tests utc_date"""

    query = "SELECT A as A, UTC_DATE() as B from table1"
    expected_output = pd.DataFrame(
        {
            "A": basic_df["TABLE1"]["A"],
            "B": pd.Timestamp.now(tz="UTC").date(),
        }
    )
    check_query(
        query,
        basic_df,
        spark_info,
        expected_output=expected_output,
    )


@pytest.fixture(
    params=[
        # check the values for which the format strings are the same
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
        ('% %a %\\, %%a, %%, %%%%, "%"', ' %a \\, %%a, %%, %%%%, ""'),
    ]
    # TODO: add addition format characters when/if they become supported
)
def python_mysql_dt_format_strings(request):
    """returns a tuple of python mysql string, and the equivalent python format string"""
    return request.param


@pytest.mark.slow
def test_date_format_timestamp(
    dt_fn_dataframe, python_mysql_dt_format_strings, memory_leak_check
):
    """
    tests the date format function with timestamp inputs
    """

    mysql_format_str = python_mysql_dt_format_strings[0]
    python_format_str = python_mysql_dt_format_strings[1]

    query = f"SELECT DATE_FORMAT(timestamps, '{mysql_format_str}') from table1"
    expected_output = pd.DataFrame(
        {
            "output": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.strftime(
                python_format_str
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
def test_date_format_date(date_df, python_mysql_dt_format_strings, memory_leak_check):
    """
    tests the date format function with date inputs
    """

    mysql_format_str = python_mysql_dt_format_strings[0]
    python_format_str = python_mysql_dt_format_strings[1]

    query = f"SELECT DATE_FORMAT(A, '{mysql_format_str}') from table1"
    expected_output = pd.DataFrame(
        {
            "output": pd.Series(
                [
                    None if date is None else date.strftime(python_format_str)
                    for date in date_df["TABLE1"]["A"]
                ]
            )
        }
    )
    check_query(
        query,
        date_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_microseconds(dt_fn_dataframe, memory_leak_check):
    """spark has no equivalent MICROSECOND function, so we need to test it manually"""

    query1 = "SELECT MICROSECOND(timestamps) as microsec_time from table1"

    expected_output = pd.DataFrame(
        {"MICROSEC_TIME": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.microsecond % 1000}
    )

    check_query(
        query1,
        dt_fn_dataframe,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_microsecond(tz_aware_df, memory_leak_check):
    """simplest test for microsecond on timezone aware data"""
    query = "SELECT MICROSECOND(A) as microsec_time from table1"
    expected_output = pd.DataFrame(
        {"MICROSEC_TIME": tz_aware_df["TABLE1"]["A"].dt.microsecond % 1000}
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

    micro_series = tz_aware_df["TABLE1"]["A"].dt.microsecond % 1000
    micro_series[micro_series <= 1] = -1

    expected_output = pd.DataFrame({"MICROSEC_TIME": micro_series})

    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_dayname_cols(dt_fn_dataframe, memory_leak_check):
    """tests the dayname function on column inputs. Needed since the equivalent function has different syntax"""
    query = "SELECT DAYNAME(timestamps) as OUTPUT from table1"
    py_output = pd.DataFrame(
        {
            "OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].map(
                lambda x: None if pd.isna(x) else x.day_name()[:3]
            ),
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


# Tests that implicit casting of string -> timestamp works
@pytest.mark.parametrize(
    "date_literal_strings",
    [
        ("TIMESTAMP '2021-03-03'", "TIMESTAMP '2021-03-13'", "TIMESTAMP '2021-03-01'"),
        ("'2021-03-03'", "'2021-03-13'", "'2021-03-01'"),
    ],
)
def test_dayname_scalars(basic_df, date_literal_strings, memory_leak_check):
    """tests the dayname function on scalar inputs. Needed since the equivalent function has different syntax"""

    # since dayname is a fn we defined, don't need to worry about calcite performing optimizations
    # Use basic_df so the input is expanded and we don't have to worry about empty arrays
    scalar1, scalar2, scalar3 = date_literal_strings
    query = f"SELECT A, DAYNAME({scalar1}) as B, DAYNAME({scalar2}) as C, DAYNAME({scalar3}) as D from table1"
    py_output = pd.DataFrame(
        {"A": basic_df["TABLE1"]["A"], "B": "Wed", "C": "Sat", "D": "Mon"}
    )

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


def test_dayname_date_cols(date_df, memory_leak_check):
    """tests the dayname function on column inputs of date objects."""
    query = "SELECT DAYNAME(A) AS OUTPUT from table1"
    py_output = pd.DataFrame({"OUTPUT": date_df["TABLE1"]["A"].map(day_name_func)})

    check_query(
        query,
        date_df,
        None,
        check_dtype=False,
        expected_output=py_output,
    )


def day_name_func(date):
    if date is None:
        return None
    dows = [
        "Mon",
        "Tue",
        "Wed",
        "Thu",
        "Fri",
        "Sat",
        "Sun",
    ]
    return dows[date.weekday()]


def test_dayname_date_scalars(basic_df, memory_leak_check):
    """tests the dayname function on scalar inputs of date objects."""

    # since dayname is a fn we defined, don't need to worry about calcite performing optimizations
    # Use basic_df so the input is expanded and we don't have to worry about empty arrays
    query = "SELECT DAYNAME(TO_DATE('2021-03-03')), DAYNAME(TO_DATE('2021-05-13')), DAYNAME(TO_DATE('2021-07-03'))"
    outputs = pd.DataFrame({"A": ["Wed"], "B": ["Thu"], "C": ["Sat"]})

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


@pytest.mark.parametrize(
    "fn_name", ["MONTHNAME", pytest.param("MONTH_NAME", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("wrap_case", [True, False])
def test_monthname_cols(fn_name, wrap_case, dt_fn_dataframe, memory_leak_check):
    """tests the monthname function on column inputs."""

    if wrap_case:
        query = f"SELECT CASE WHEN timestamps IS NULL THEN {fn_name}(timestamps) else {fn_name}(timestamps) END as output FROM table1"
    else:
        query = f"SELECT {fn_name}(timestamps) as output from table1"

    py_output = pd.DataFrame(
        {
            "output": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].map(
                lambda x: None if pd.isna(x) else x.month_name()[:3]
            )
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize("fn_name", ["MONTHNAME", "MONTH_NAME"])
def test_monthname_scalars(fn_name, basic_df, memory_leak_check):
    """tests the monthname function on scalar inputs"""

    # since monthname is a fn we defined, don't need to worry about calcite performing optimizations
    query = f"SELECT {fn_name}(TIMESTAMP '2021-03-03') as A, {fn_name}(TIMESTAMP '2021-05-13') as B, {fn_name}(TIMESTAMP '2021-10-01') as C"
    py_output = pd.DataFrame({"A": ["Mar"], "B": ["May"], "C": ["Oct"]})

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize(
    "fn_name", ["MONTHNAME", pytest.param("MONTH_NAME", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("wrap_case", [True, False])
def test_monthname_date_cols(fn_name, wrap_case, date_df, memory_leak_check):
    """tests the monthname function on column inputs of date objects."""

    if wrap_case:
        query = f"SELECT CASE WHEN A IS NULL THEN {fn_name}(A) else {fn_name}(A) END AS OUTPUT FROM table1"
    else:
        query = f"SELECT {fn_name}(A) AS OUTPUT from table1"

    outputs = pd.DataFrame({"OUTPUT": date_df["TABLE1"]["A"].map(month_name_func)})

    check_query(
        query,
        date_df,
        None,
        check_dtype=False,
        expected_output=outputs,
    )


def month_name_func(date):
    if date is None:
        return None
    mons = [
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
    ]
    return mons[date.month - 1]


@pytest.mark.parametrize("fn_name", ["MONTHNAME", "MONTH_NAME"])
def test_monthname_date_scalars(fn_name, basic_df, memory_leak_check):
    """tests the monthname function on scalar inputs of date objects."""

    query = f"SELECT {fn_name}(DATE '2021-03-03'), {fn_name}(DATE '2021-05-13'), {fn_name}(DATE '2021-07-01')"
    outputs = pd.DataFrame({"A": ["Mar"], "B": ["May"], "C": ["Jul"]})

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=outputs,
    )


def test_makedate_scalars(basic_df, dt_fn_dataframe, memory_leak_check):
    """tests makedate on scalar values"""

    query = "SELECT makedate(2000, 200), makedate(2010, 300), makedate(2020, 400)"
    output = pd.DataFrame(
        {
            "A": [datetime.date(2000, 7, 18)],
            "B": [datetime.date(2010, 10, 27)],
            "C": [datetime.date(2021, 2, 3)],
        }
    )

    check_query(
        query,
        basic_df,
        None,
        check_names=False,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(
            True,
            id="with_case",
        ),
    ],
)
def test_makedate_cols(dt_fn_dataframe, use_case, memory_leak_check):
    """tests makedate with case statement"""
    if use_case:
        query = "SELECT CASE WHEN makedate(valid_year_integers, positive_integers) > DATE '2211-01-01' THEN DATE '2000-01-01' ELSE makedate(valid_year_integers, positive_integers) END as OUTPUT from table1"
    else:
        query = "SELECT makedate(valid_year_integers, positive_integers) as OUTPUT from table1"

    def makedate_fn(dt_fn_dataframe):
        n = len(dt_fn_dataframe["TABLE1"]["VALID_YEAR_INTEGERS"])
        res = pd.Series([None] * n)
        for i in range(n):
            year = dt_fn_dataframe["TABLE1"]["VALID_YEAR_INTEGERS"][i]
            day = dt_fn_dataframe["TABLE1"]["POSITIVE_INTEGERS"][i]
            if pd.isna(year) or pd.isna(day):
                res[i] = None
            else:
                date = datetime.date(year=year, month=1, day=1) + pd.Timedelta(
                    day - 1, unit="D"
                )
                res[i] = date
        return res

    output = pd.DataFrame({"OUTPUT": makedate_fn(dt_fn_dataframe)})
    check_query(
        query,
        dt_fn_dataframe,
        None,
        expected_output=output,
    )


@pytest.mark.slow
def test_makedate_edgecases(spark_info, dt_fn_dataframe, memory_leak_check):
    """tests makedate on edgecases"""

    query = "SELECT makedate(valid_year_integers, mixed_integers), makedate(positive_integers, positive_integers) from table1"
    spark_query = "SELECT DATE_ADD(make_date(valid_year_integers, 1, 1), mixed_integers-1), DATE_ADD(make_date(positive_integers, 1, 1), positive_integers-1) from table1"

    # spark requires certain arguments of make_date to not
    # be of type bigint, but all pandas integer types are currently interpreted as bigint.
    cols_to_cast = {"TABLE1": [("POSITIVE_INTEGERS", "int"), ("MIXED_INTEGERS", "int")]}

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
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
    query = (
        f"SELECT EXTRACT({valid_extract_strings} from timestamps) AS OUTPUT from table1"
    )

    # spark does not allow the microsecond argument for extract, and to compensate, the
    # second argument returns a float. Therefore, in these cases we need to manually
    # generate the expected output
    if valid_extract_strings == "SECOND":
        expected_output = pd.DataFrame(
            {"OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.second}
        )
    elif valid_extract_strings == "MICROSECOND":
        expected_output = pd.DataFrame(
            {"OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.microsecond % 1000}
        )
    else:
        expected_output = None

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_extract_scalars(
    spark_info, dt_fn_dataframe, valid_extract_strings, memory_leak_check
):
    query = f"SELECT CASE WHEN EXTRACT({valid_extract_strings} from timestamps) < 0 THEN -1 ELSE EXTRACT({valid_extract_strings} from timestamps) END AS OUTPUT from table1"

    # spark does not allow the microsecond argument for extract, and to compensate, the
    # second argument returns a float. Therefore, in these cases we need to manually
    # generate the expected output
    if valid_extract_strings == "SECOND":
        expected_output = pd.DataFrame(
            {"OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.second}
        )
    elif valid_extract_strings == "MICROSECOND":
        expected_output = pd.DataFrame(
            {"OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.microsecond % 1000}
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
        "TABLE1": pd.DataFrame(
            {
                "COL_DT": pd.Series(
                    [
                        None,
                        pd.Timestamp("2010-01-17"),
                        pd.Timestamp("2011-02-26 03:36:01"),
                        pd.Timestamp("2012-05-09 16:43:16.123456"),
                        pd.Timestamp("2013-10-22 05:32:21.987654321"),
                    ],
                    dtype="datetime64[ns]",
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
    df = tz_aware_df["TABLE1"]
    py_output = pd.DataFrame(
        {
            "MY_YEAR": df.A.dt.year,
            "MY_Q": df.A.dt.quarter,
            "MY_MONS": df.A.dt.month,
            "MY_WK": df.A.map(lambda t: t.weekofyear),
            "MY_DAYOFMONTH": df.A.dt.day,
            "MY_HRS": df.A.dt.hour,
            "MY_MIN": df.A.dt.minute,
            "MY_S": df.A.dt.second,
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


def test_date_part_unquoted_timeunit(memory_leak_check):
    """
    Test DATE_PART works for unquoted time unit input
    """
    query_fmt = "DATE_PART({!s}, A) AS my_{}"
    selects = []
    for unit in [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "dow",
    ]:
        selects.append(query_fmt.format(unit, unit))
    query = f"SELECT {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [
                        None,
                        pd.Timestamp("2010-01-17"),
                        pd.Timestamp("2011-02-26 03:36:01"),
                        pd.Timestamp("2012-05-09 16:43:16.123456"),
                        pd.Timestamp("2013-10-22 05:32:21.987654321"),
                    ],
                    dtype="datetime64[ns]",
                )
            }
        )
    }
    df = ctx["TABLE1"]
    py_output = pd.DataFrame(
        {
            "MY_YEAR": df.A.dt.year,
            "MY_QUARTER": df.A.dt.quarter,
            "MY_MONTH": df.A.dt.month,
            "MY_WEEK": df.A.map(lambda t: t.weekofyear),
            "MY_DAY": df.A.dt.day,
            "MY_HOUR": df.A.dt.hour,
            "MY_MINUTE": df.A.dt.minute,
            "MY_SECOND": df.A.dt.second,
            "MY_DOW": pd.Series([None, 0, 6, 3, 2]),
        }
    )

    check_query(
        query,
        ctx,
        None,
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
        "TABLE1": pd.DataFrame(
            {
                "COL_INT": pd.Series([10, 1, None, -10, 100], dtype=pd.Int32Dtype()),
                "COL_DT": pd.Series(
                    [
                        None,
                        pd.Timestamp("2013-10-27"),
                        pd.Timestamp("2015-4-1 12:00:15"),
                        pd.Timestamp("2020-2-3 05:15:12.501"),
                        pd.Timestamp("2021-12-13 23:15:06.025999500"),
                    ],
                    dtype="datetime64[ns]",
                ),
            }
        )
    }


@pytest.fixture
def dateadd_fractional_df():
    """Returns the context used by test_snowflake_dateadd_fractional"""
    return {
        "TABLE1": pd.DataFrame(
            {
                "COL_INT": pd.Series(
                    [10.2, 0.5, None, -9.5, 99.8], dtype=pd.Float32Dtype()
                ),
                "COL_DT": pd.Series(
                    [
                        None,
                        pd.Timestamp("2013-10-27"),
                        pd.Timestamp("2015-4-1 12:00:15"),
                        pd.Timestamp("2020-2-3 05:15:12.501"),
                        pd.Timestamp("2021-12-13 23:15:06.025999500"),
                    ],
                    dtype="datetime64[ns]",
                ),
            }
        )
    }


@pytest.fixture(
    params=[
        pytest.param(
            (
                "DATEADD({!r}, col_int, col_dt)",
                ["YEAR", "MONTH", "WEEK", "DAY"],
                pd.DataFrame(
                    {
                        "YEAR": [
                            None,
                            pd.Timestamp("2014-10-27 00:00:00"),
                            None,
                            pd.Timestamp("2010-02-03 05:15:12.501000"),
                            pd.Timestamp("2121-12-13 23:15:06.025999500"),
                        ],
                        "MONTH": [
                            None,
                            pd.Timestamp("2013-11-27 00:00:00"),
                            None,
                            pd.Timestamp("2019-04-03 05:15:12.501000"),
                            pd.Timestamp("2030-04-13 23:15:06.025999500"),
                        ],
                        "WEEK": [
                            None,
                            pd.Timestamp("2013-11-03 00:00:00"),
                            None,
                            pd.Timestamp("2019-11-25 05:15:12.501000"),
                            pd.Timestamp("2023-11-13 23:15:06.025999500"),
                        ],
                        "DAY": [
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
                "TIMEADD({!s}, col_int, col_dt)",
                ["HOUR", "MINUTE", "SECOND"],
                pd.DataFrame(
                    {
                        "HOUR": [
                            None,
                            pd.Timestamp("2013-10-27 01:00:00"),
                            None,
                            pd.Timestamp("2020-02-02 19:15:12.501000"),
                            pd.Timestamp("2021-12-18 03:15:06.025999500"),
                        ],
                        "MINUTE": [
                            None,
                            pd.Timestamp("2013-10-27 00:01:00"),
                            None,
                            pd.Timestamp("2020-02-03 05:05:12.501000"),
                            pd.Timestamp("2021-12-14 00:55:06.025999500"),
                        ],
                        "SECOND": [
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
                ["MILLISECOND", "MICROSECOND", "NANOSECOND"],
                pd.DataFrame(
                    {
                        "MILLISECOND": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.001000"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:12.491000"),
                            pd.Timestamp("2021-12-13 23:15:06.125999500"),
                        ],
                        "MICROSECOND": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000001"),
                            None,
                            pd.Timestamp("2020-02-03 05:15:12.500990"),
                            pd.Timestamp("2021-12-13 23:15:06.026099500"),
                        ],
                        "NANOSECOND": [
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
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!s}, -25, col_dt) END",
                ["YEAR", "MONTH", "WEEK", "DAY"],
                pd.DataFrame(
                    {
                        "YEAR": [
                            None,
                            pd.Timestamp("1988-10-27 00:00:00"),
                            pd.Timestamp("1990-04-01 12:00:15"),
                            None,
                            pd.Timestamp("1996-12-13 23:15:06.025999500"),
                        ],
                        "MONTH": [
                            None,
                            pd.Timestamp("2011-09-27 00:00:00"),
                            pd.Timestamp("2013-03-01 12:00:15"),
                            None,
                            pd.Timestamp("2019-11-13 23:15:06.025999500"),
                        ],
                        "WEEK": [
                            None,
                            pd.Timestamp("2013-05-05 00:00:00"),
                            pd.Timestamp("2014-10-08 12:00:15"),
                            None,
                            pd.Timestamp("2021-06-21 23:15:06.025999500"),
                        ],
                        "DAY": [
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
                ["HOUR", "MINUTE", "SECOND"],
                pd.DataFrame(
                    {
                        "HOUR": [
                            None,
                            pd.Timestamp("2013-10-25 23:00:00"),
                            pd.Timestamp("2015-03-31 11:00:15"),
                            None,
                            pd.Timestamp("2021-12-12 22:15:06.025999500"),
                        ],
                        "MINUTE": [
                            None,
                            pd.Timestamp("2013-10-26 23:35:00"),
                            pd.Timestamp("2015-04-01 11:35:15"),
                            None,
                            pd.Timestamp("2021-12-13 22:50:06.025999500"),
                        ],
                        "SECOND": [
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
                "CASE WHEN col_int < 0 THEN NULL else DATEADD({!s}, -25, col_dt) END",
                ["MILLISECOND", "MICROSECOND", "NANOSECOND"],
                pd.DataFrame(
                    {
                        "MILLISECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.975000"),
                            pd.Timestamp("2015-04-01 12:00:14.975000"),
                            None,
                            pd.Timestamp("2021-12-13 23:15:06.000999500"),
                        ],
                        "MICROSECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999975"),
                            pd.Timestamp("2015-04-01 12:00:14.999975"),
                            None,
                            pd.Timestamp("2021-12-13 23:15:06.025974500"),
                        ],
                        "NANOSECOND": [
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


def test_snowflake_dateadd_fractional(
    dateadd_fractional_df, dateadd_queries, memory_leak_check
):
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
        dateadd_fractional_df,
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
        "TABLE1": pd.DataFrame(
            {
                "COL_INT": pd.Series([10, 1, None, -10, 100], dtype=pd.Int32Dtype()),
                "COL_DT": pd.Series(
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
                "DATEADD({!s}, col_int, col_dt)",
                ["YEAR", "QUARTER", "MONTH", "WEEK", "DAY"],
                pd.DataFrame(
                    {
                        "YEAR": [
                            None,
                            datetime.date(2014, 10, 27),
                            None,
                            datetime.date(2010, 2, 3),
                            datetime.date(2121, 12, 13),
                        ],
                        "QUARTER": [
                            None,
                            datetime.date(2014, 1, 27),
                            None,
                            datetime.date(2017, 8, 3),
                            datetime.date(2046, 12, 13),
                        ],
                        "MONTH": [
                            None,
                            datetime.date(2013, 11, 27),
                            None,
                            datetime.date(2019, 4, 3),
                            datetime.date(2030, 4, 13),
                        ],
                        "WEEK": [
                            None,
                            datetime.date(2013, 11, 3),
                            None,
                            datetime.date(2019, 11, 25),
                            datetime.date(2023, 11, 13),
                        ],
                        "DAY": [
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
                ["HOUR", "MINUTE", "SECOND"],
                pd.DataFrame(
                    {
                        "HOUR": [
                            None,
                            pd.Timestamp("2013-10-27 01:00:00"),
                            None,
                            pd.Timestamp("2020-02-02 14:00:00"),
                            pd.Timestamp("2021-12-17 04:00:00"),
                        ],
                        "MINUTE": [
                            None,
                            pd.Timestamp("2013-10-27 00:01:00"),
                            None,
                            pd.Timestamp("2020-02-02 23:50:00"),
                            pd.Timestamp("2021-12-13 01:40:00"),
                        ],
                        "SECOND": [
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
                "TIMESTAMPADD({!s}, col_int, col_dt)",
                ["MILLISECOND", "MICROSECOND", "NANOSECOND"],
                pd.DataFrame(
                    {
                        "MILLISECOND": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.001"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:59.990"),
                            pd.Timestamp("2021-12-13 00:00:00.100"),
                        ],
                        "MICROSECOND": [
                            None,
                            pd.Timestamp("2013-10-27 00:00:00.000001"),
                            None,
                            pd.Timestamp("2020-02-02 23:59:59.999990"),
                            pd.Timestamp("2021-12-13 00:00:00.000100"),
                        ],
                        "NANOSECOND": [
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
                "CASE WHEN col_int < 0 THEN DATE '1999-12-31' else DATEADD({!r}, -25, col_dt) END",
                ["YEAR", "QUARTER", "MONTH", "WEEK", "DAY"],
                pd.DataFrame(
                    {
                        "YEAR": [
                            None,
                            datetime.date(1988, 10, 27),
                            datetime.date(1990, 4, 1),
                            datetime.date(1999, 12, 31),
                            datetime.date(1996, 12, 13),
                        ],
                        "QUARTER": [
                            None,
                            datetime.date(2007, 7, 27),
                            datetime.date(2009, 1, 1),
                            datetime.date(1999, 12, 31),
                            datetime.date(2015, 9, 13),
                        ],
                        "MONTH": [
                            None,
                            datetime.date(2011, 9, 27),
                            datetime.date(2013, 3, 1),
                            datetime.date(1999, 12, 31),
                            datetime.date(2019, 11, 13),
                        ],
                        "WEEK": [
                            None,
                            datetime.date(2013, 5, 5),
                            datetime.date(2014, 10, 8),
                            datetime.date(1999, 12, 31),
                            datetime.date(2021, 6, 21),
                        ],
                        "DAY": [
                            None,
                            datetime.date(2013, 10, 2),
                            datetime.date(2015, 3, 7),
                            datetime.date(1999, 12, 31),
                            datetime.date(2021, 11, 18),
                        ],
                    }
                ),
            ),
            id="case-date_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN TIMESTAMP '1999-12-31' else TIMESTAMPADD({!s}, -25, col_dt) END",
                ["HOUR", "MINUTE", "SECOND"],
                pd.DataFrame(
                    {
                        "HOUR": [
                            None,
                            pd.Timestamp("2013-10-25 23:00:00"),
                            pd.Timestamp("2015-03-30 23:00:00"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2021-12-11 23:00:00"),
                        ],
                        "MINUTE": [
                            None,
                            pd.Timestamp("2013-10-26 23:35:00"),
                            pd.Timestamp("2015-03-31 23:35:00"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2021-12-12 23:35:00"),
                        ],
                        "SECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:35"),
                            pd.Timestamp("2015-03-31 23:59:35"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2021-12-12 23:59:35"),
                        ],
                    }
                ),
            ),
            id="case-time_units",
        ),
        pytest.param(
            (
                "CASE WHEN col_int < 0 THEN TIMESTAMP '1999-12-31' else DATEADD({!r}, -25, col_dt) END",
                ["MILLISECOND", "MICROSECOND", "NANOSECOND"],
                pd.DataFrame(
                    {
                        "MILLISECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.975"),
                            pd.Timestamp("2015-03-31 23:59:59.975"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2021-12-12 23:59:59.975"),
                        ],
                        "MICROSECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999975"),
                            pd.Timestamp("2015-03-31T23:59:59.999975"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2021-12-12T23:59:59.999975"),
                        ],
                        "NANOSECOND": [
                            None,
                            pd.Timestamp("2013-10-26 23:59:59.999999975"),
                            pd.Timestamp("2015-03-31 23:59:59.999999975"),
                            pd.Timestamp("1999-12-31 00:00:00"),
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


def test_snowflake_dateadd_date(
    dateadd_date_df, dateadd_date_queries, memory_leak_check
):
    """
    Tests the Snowflake version of DATEADD/TIMEADD/TIMESTAMPADD with date inputs for dt_val.
    Currently takes in the unit as a scalar string instead of a DT unit literal.
    """
    query_fmt, units, answers = dateadd_date_queries
    selects = []
    for unit in units:
        selects.append(query_fmt.format(unit))
    query = "SELECT " + ", ".join(selects) + " FROM table1"

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
        {"TABLE1": df},
        None,
        check_names=False,
        check_dtype=False,
        expected_output=answer,
        session_tz=time_zone,
    )


@pytest.fixture(
    params=[
        pytest.param(
            ("'mm'", 158, "2035-5-12 20:30:00", "2036-1-6 0:45:00"), id="month"
        ),
        pytest.param(
            ("'d'", 1, "2022-3-13 20:30:00", "2022-11-7 0:45:00"),
            id="day",
            marks=pytest.mark.slow,
        ),
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
        "TABLE1": pd.DataFrame(
            {
                "DT_COL": pd.Series(
                    [pd.Timestamp("2022-3-12 20:30:00", tz=request.param)] * 3
                    + [None]
                    + [pd.Timestamp("2022-11-6 0:45:00", tz=request.param)] * 3,
                    dtype=f"datetime64[ns, {request.param}]",
                ),
                "BOOL_COL": pd.Series([True] * 7),
            }
        )
    }
    calculation = f"DATEADD({unit}, {amt}, dt_col)"
    answer = pd.DataFrame(
        {
            0: pd.Series(
                [pd.Timestamp(springRes, tz=request.param)] * 3
                + [None]
                + [pd.Timestamp(fallRes, tz=request.param)] * 3,
                dtype=f"datetime64[ns, {request.param}]",
            )
        }
    )
    return ctx, calculation, answer, request.param


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_snowflake_tz_dateadd(tz_dateadd_data, case):
    ctx, calculation, answer, session_tz = tz_dateadd_data
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
        session_tz=session_tz,
    )


@pytest.mark.parametrize(
    "query, expected_output",
    [
        pytest.param(
            "SELECT DATEADD('MONTH', 10, '2022-06-30'::DATE)",
            pd.DataFrame({"A": pd.Series([pd.Timestamp("2023-04-30").date()])}),
            id="dateadd-date",
        ),
        pytest.param(
            "SELECT DATEADD('MONTH', 10, '2022-06-30')",
            pd.DataFrame(
                {"A": pd.Series([pd.Timestamp("2023-04-30")], dtype="datetime64[ns]")}
            ),
            id="dateadd-string",
        ),
        pytest.param(
            "SELECT TIMEADD('SECOND', 30, '2022-06-30 12:23:23'::TIMESTAMP)",
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [pd.Timestamp("2022-06-30 12:23:53")], dtype="datetime64[ns]"
                    )
                }
            ),
            id="timeadd-timestamp",
        ),
        pytest.param(
            "SELECT TIMEADD('SECOND', 30, '2022-06-30 12:23:23')",
            pd.DataFrame(
                {
                    "A": pd.Series(
                        [pd.Timestamp("2022-06-30 12:23:53")], dtype="datetime64[ns]"
                    )
                }
            ),
            id="dateadd-string",
        ),
        pytest.param(
            "SELECT TIMESTAMPADD('DAY', 5, '2022-06-30'::TIMESTAMP)",
            pd.DataFrame(
                {"A": pd.Series([pd.Timestamp("2022-07-05")], dtype="datetime64[ns]")}
            ),
            id="timeadd-timestamp",
        ),
        pytest.param(
            "SELECT TIMESTAMPADD('DAY', 5, '2022-06-30')",
            pd.DataFrame(
                {"A": pd.Series([pd.Timestamp("2022-07-05")], dtype="datetime64[ns]")}
            ),
            id="dateadd-string",
        ),
    ],
)
@pytest.mark.slow
def test_datedadd_date_literals(query, expected_output, basic_df, memory_leak_check):
    """
    Checks that calling DATEADD/TIMEADD/TIMESTAMPADD on datetime.date/string literals behaves as expected.
    """
    check_query(
        query,
        basic_df,
        spark=None,
        expected_output=expected_output,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_timestamp_add_scalar(mysql_interval_str, dt_fn_dataframe, memory_leak_check):
    interval_str_to_offset_fn = {
        "SECOND": lambda x: None if pd.isna(x) else pd.Timedelta(seconds=x),
        "MINUTE": lambda x: None if pd.isna(x) else pd.Timedelta(minutes=x),
        "HOUR": lambda x: None if pd.isna(x) else pd.Timedelta(hours=x),
        "DAY": lambda x: None if pd.isna(x) else pd.Timedelta(days=x),
        "WEEK": lambda x: None if pd.isna(x) else pd.Timedelta(weeks=x),
        "MONTH": lambda x: None if pd.isna(x) else pd.DateOffset(months=x),
        "YEAR": lambda x: None if pd.isna(x) else pd.DateOffset(years=x),
    }
    offset_fn = interval_str_to_offset_fn[mysql_interval_str]
    query = f"SELECT CASE WHEN timestampadd({mysql_interval_str}, small_positive_integers, timestamps) < TIMESTAMP '1970-01-01' THEN TIMESTAMP '1970-01-01' ELSE timestampadd({mysql_interval_str}, small_positive_integers, timestamps) END from table1"
    timestamps = dt_fn_dataframe["TABLE1"]["TIMESTAMPS"]
    integers = dt_fn_dataframe["TABLE1"]["SMALL_POSITIVE_INTEGERS"]
    offsets = integers.apply(offset_fn)
    result = pd.DataFrame(
        {"result": pd.Series([ts + offset for ts, offset in zip(timestamps, offsets)])}
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=result,
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
        "HOUR": unit,
        "MINUTE": f"'{unit}'",
        "SECOND": unit,
        "MILLISECOND": f"'{unit}'",
        "MICROSECOND": unit,
        "NANOSECOND": f"'{unit}'",
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

    # spark converts all inputs to DATE with DATE_SUB. This is incorrect
    # behavior, but for the sake of this test, we add a TO_DATE call on the
    # input so that it matches. We have tests of DATEADD preserving the right
    # type (test_snowflake_dateadd), which should compile to the same kernel as
    # DATE_SUB.
    if case:
        query = f"SELECT CASE WHEN positive_integers < 100 THEN TO_DATE({dateadd_fn}({dt_val}, {amt_val})) ELSE NULL END from table1"
        spark_query = f"SELECT CASE WHEN positive_integers < 100 THEN DATE_ADD({dt_val}, {amt_val}) ELSE NULL END from table1"
    else:
        query = f"SELECT TO_DATE({dateadd_fn}({dt_val}, {amt_val})) from table1"
        spark_query = f"SELECT DATE_ADD({dt_val}, {amt_val}) from table1"

    # spark requires certain arguments of adddate to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"TABLE1": [("MIXED_INTEGERS", "int")]}

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
        query = f"SELECT CASE WHEN N < 0 THEN NULL ELSE {dateadd_fn}(A, N) END AS RES from table1"
    else:
        query = f"SELECT {dateadd_fn}(A, N) AS RES from table1"

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
        ],
        dtype="datetime64[ns, US/Pacific]",
    )
    res = pd.Series(
        [
            None if s is None else pd.Timestamp(s, tz="US/Pacific")
            for s in adjusted_timestamp_strings
        ],
        dtype="datetime64[ns, US/Pacific]",
    )
    ctx = {
        "TABLE1": pd.DataFrame(
            {"A": tz_timestamps, "N": pd.Series(day_offsets, dtype=pd.Int32Dtype())}
        )
    }
    expected_output = pd.DataFrame({"RES": res})

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        expected_output=expected_output,
        session_tz="US/Pacific",
    )


@pytest.fixture
def date_add_sub_date_df():
    return {
        "TABLE1": pd.DataFrame(
            {
                "START_DT": [
                    datetime.date(2020, 6, 26),
                    datetime.date(2025, 5, 3),
                    datetime.date(1987, 3, 15),
                    datetime.date(2117, 8, 29),
                    datetime.date(1822, 12, 7),
                    datetime.date(1906, 4, 14),
                    None,
                    datetime.date(1700, 2, 4),
                ],
                # TODO: support pd.DateOffset array
                # "date_interval": [
                #     pd.DateOffset(days=10),
                #     pd.DateOffset(months=4),
                #     pd.DateOffset(years=6),
                #     None,
                #     None,
                #     pd.DateOffset(months=20),
                #     pd.DateOffset(days=100),
                #     pd.DateOffset(years=5),
                # ],
                "TIME_INTERVAL": [
                    pd.Timedelta(hours=10),
                    pd.Timedelta(minutes=4),
                    None,
                    pd.Timedelta(seconds=6),
                    pd.Timedelta(hours=1),
                    None,
                    pd.Timedelta(minutes=100),
                    pd.Timedelta(seconds=5),
                ],
            }
        )
    }


@pytest.fixture(
    params=[
        pytest.param(
            (
                "SELECT DATE_ADD(start_dt, date_interval) FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            datetime.date(2020, 7, 6),
                            datetime.date(2025, 9, 3),
                            datetime.date(1987, 3, 15),
                            None,
                            None,
                            datetime.date(1906, 4, 21),
                            None,
                            datetime.date(1705, 2, 4),
                        ]
                    }
                ),
            ),
            id="DATE_ADD-vector_date_interval",
            marks=pytest.mark.skip(reason="TODO: support pd.DateOffset array"),
        ),
        pytest.param(
            (
                "SELECT CASE WHEN start_dt IS NULL THEN DATE '1999-12-31' ELSE DATE_ADD(start_dt, INTERVAL 10 DAY) END FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            datetime.date(2020, 7, 6),
                            datetime.date(2025, 5, 13),
                            datetime.date(1987, 3, 25),
                            datetime.date(2117, 9, 8),
                            datetime.date(1822, 12, 17),
                            datetime.date(1906, 4, 24),
                            datetime.date(1999, 12, 31),
                            datetime.date(1700, 2, 14),
                        ]
                    }
                ),
            ),
            id="DATE_ADD-vector_scalar_date_interval_with_case",
        ),
        pytest.param(
            (
                "SELECT DATE_ADD(TO_DATE('2020-10-13'), INTERVAL 40 HOURS)",
                pd.DataFrame({"OUTPUTS": [pd.Timestamp("2020-10-14 16:00:00")]}),
            ),
            id="DATE_ADD-scalar_time_interval",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN time_interval IS NULL THEN TIMESTAMP '1999-12-31' ELSE ADDDATE(start_dt, time_interval) END FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            pd.Timestamp("2020-06-26 10:00:00"),
                            pd.Timestamp("2025-05-03 00:04:00"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            pd.Timestamp("2117-08-29 00:00:06"),
                            pd.Timestamp("1822-12-07 01:00:00"),
                            pd.Timestamp("1999-12-31 00:00:00"),
                            None,
                            pd.Timestamp("1700-02-04 00:00:05"),
                        ]
                    }
                ),
            ),
            id="ADDDATE-vector_time_interval_with_case",
        ),
        pytest.param(
            (
                "SELECT ADDDATE(TO_DATE('2022-3-5'), INTERVAL 8 MONTH)",
                pd.DataFrame({"OUTPUTS": [datetime.date(2022, 11, 5)]}),
            ),
            id="ADDDATE-scalar_date_interval",
        ),
        pytest.param(
            (
                "SELECT DATE_SUB(start_dt, date_interval) FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            datetime.date(2020, 7, 6),
                            datetime.date(2025, 9, 3),
                            datetime.date(1987, 3, 15),
                            None,
                            None,
                            datetime.date(1906, 4, 21),
                            None,
                            datetime.date(1705, 2, 4),
                        ]
                    }
                ),
            ),
            id="DATE_SUB-vector_date_interval",
            marks=pytest.mark.skip(reason="TODO: support pd.DateOffset array"),
        ),
        pytest.param(
            (
                "SELECT CASE WHEN start_dt IS NULL THEN DATE '1999-12-31' ELSE DATE_SUB(start_dt, INTERVAL 5 MONTH) END FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            datetime.date(2020, 1, 26),
                            datetime.date(2024, 12, 3),
                            datetime.date(1986, 10, 15),
                            datetime.date(2117, 3, 29),
                            datetime.date(1822, 7, 7),
                            datetime.date(1905, 11, 14),
                            datetime.date(1999, 12, 31),
                            datetime.date(1699, 9, 4),
                        ]
                    }
                ),
            ),
            id="DATE_SUB-vector_scalar_date_interval_with_case",
        ),
        pytest.param(
            (
                "SELECT DATE_SUB(TO_DATE('2011-1-18'), INTERVAL 80 MINUTES)",
                pd.DataFrame({"OUTPUTS": [pd.Timestamp("2011-1-17 22:40:00")]}),
            ),
            id="DATE_SUB-scalar_time_interval",
        ),
        pytest.param(
            (
                "SELECT SUBDATE(start_dt, time_interval) FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            pd.Timestamp("2020-06-25 14:00:00"),
                            pd.Timestamp("2025-05-02 23:56:00"),
                            None,
                            pd.Timestamp("2117-08-28 23:59:54"),
                            pd.Timestamp("1822-12-06 23:00:00"),
                            None,
                            None,
                            pd.Timestamp("1700-02-03 23:59:55"),
                        ]
                    }
                ),
            ),
            id="SUBDATE-vector_time_interval",
        ),
        pytest.param(
            (
                "SELECT SUBDATE(TO_DATE('2000-1-5'), INTERVAL 2 WEEK)",
                pd.DataFrame({"OUTPUTS": [datetime.date(1999, 12, 22)]}),
            ),
            id="SUBDATE-scalar_date_interval",
        ),
        pytest.param(
            (
                "SELECT CASE WHEN start_dt IS NULL THEN DATE '1999-12-31' ELSE start_dt + INTERVAL 20 WEEK END FROM table1",
                pd.DataFrame(
                    {
                        "OUTPUTS": [
                            datetime.date(2020, 11, 13),
                            datetime.date(2025, 9, 20),
                            datetime.date(1987, 8, 2),
                            datetime.date(2118, 1, 16),
                            datetime.date(1823, 4, 26),
                            datetime.date(1906, 9, 1),
                            datetime.date(1999, 12, 31),
                            datetime.date(1700, 6, 24),
                        ]
                    }
                ),
            ),
            id="date_add_date_interval_with_case",
        ),
        pytest.param(
            (
                "SELECT TO_DATE('1990-11-3') - INTERVAL 10 DAY",
                pd.DataFrame({"OUTPUTS": [datetime.date(1990, 10, 24)]}),
            ),
            id="date_minus_date_interval",
        ),
        pytest.param(
            (
                "SELECT INTERVAL 10 YEAR + TO_DATE('2004-5-23')",
                pd.DataFrame({"OUTPUTS": [datetime.date(2014, 5, 23)]}),
            ),
            id="date_interval_add_date",
        ),
    ],
)
def date_add_sub_date_query(request):
    return request.param


def test_date_add_sub_interval_with_date_input(
    date_add_sub_date_df,
    date_add_sub_date_query,
    memory_leak_check,
):
    """Tests the MySQL version of DATE_ADD and all of its equivalent functions
    with datetime.date input
    types. Meanings of the parametrized arguments:
    date_add_date_df: The fixture containing the datetime-equivialent data
    date_add_date_query: which function name is being used.
    """
    query, outputs = date_add_sub_date_query

    check_query(
        query,
        date_add_sub_date_df,
        None,
        check_names=False,
        expected_output=outputs,
    )


def test_subdate_cols_int_arg1(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    spark_info,
    memory_leak_check,
):
    "tests that date_sub/subdate works when the second argument is an integer, on column values"

    # spark converts all inputs to DATE with DATE_SUB. This is incorrect
    # behavior, but for the sake of this test, we add a TO_DATE call on the
    # input so that it matches. We have tests of DATEADD preserving the right
    # type (test_snowflake_dateadd), which should compile to the same kernel as
    # DATE_SUB.
    query = f"SELECT {subdate_equiv_fns}(TO_DATE({timestamp_date_string_cols}), positive_integers) from table1"
    spark_query = (
        f"SELECT DATE_SUB({timestamp_date_string_cols}, positive_integers) from table1"
    )

    # spark requires certain arguments of subdate to not
    # be of type bigint, but all pandas integer types are currently inerpreted as bigint.
    cols_to_cast = {"TABLE1": [("POSITIVE_INTEGERS", "int")]}

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
    test_query = f"SELECT CASE WHEN ({timestamp_date_string_cols}::DATE - INTERVAL (positive_integers) DAY < TIMESTAMP_NS '1970-01-01') THEN TIMESTAMP_NS '1970-01-01' ELSE ({timestamp_date_string_cols}::DATE - INTERVAL (positive_integers) DAY)::TIMESTAMP_NS END from table1"

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_names=False,
        equivalent_spark_query=test_query,
        use_duckdb=True,
    )


def test_subdate_cols_td_arg1(
    subdate_equiv_fns,
    dt_fn_dataframe,
    timestamp_date_string_cols,
    memory_leak_check,
):
    """tests that date_sub/subdate works on timedelta 2nd arguments, with column inputs"""
    if timestamp_date_string_cols == "DATETIME_STRINGS":
        # Invalid case
        return

    query = f"SELECT {subdate_equiv_fns}({timestamp_date_string_cols}, intervals) AS OUTPUT from table1"

    # add interval column separately here since PySpark fails on timedelta nulls in
    # other tests
    in_dfs = {"TABLE1": dt_fn_dataframe["TABLE1"].copy()}
    in_dfs["TABLE1"]["INTERVALS"] = [
        np.timedelta64(100, "h"),
        np.timedelta64(9, "h"),
        np.timedelta64(8, "W"),
        np.timedelta64(6, "h"),
        np.timedelta64(5, "m"),
        None,
        np.timedelta64(4, "s"),
        np.timedelta64(3, "ms"),
        np.timedelta64(2000000, "us"),
        None,
    ]

    expected_output = pd.DataFrame(
        {
            "OUTPUT": pd.to_datetime(
                in_dfs["TABLE1"][timestamp_date_string_cols], format="mixed"
            )
            - in_dfs["TABLE1"]["INTERVALS"]
        }
    )

    check_query(
        query,
        in_dfs,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_subdate_td_scalars(
    subdate_equiv_fns,
    dt_fn_dataframe,
    memory_leak_check,
):
    """tests that subdate works on timedelta 2nd arguments, with scalar inputs"""
    query = f"SELECT CASE WHEN {subdate_equiv_fns}(TIMESTAMPS, intervals) < TIMESTAMP '1700-01-01' THEN TIMESTAMP '1970-01-01' ELSE {subdate_equiv_fns}(TIMESTAMPS, intervals) END as OUTPUT from table1"

    # add interval column separately here since PySpark fails on timedelta nulls in
    # other tests
    in_dfs = {"TABLE1": dt_fn_dataframe["TABLE1"].copy()}
    in_dfs["TABLE1"]["INTERVALS"] = [
        np.timedelta64(100, "h"),
        np.timedelta64(9, "h"),
        np.timedelta64(8, "W"),
        np.timedelta64(6, "h"),
        np.timedelta64(5, "m"),
        None,
        np.timedelta64(4, "s"),
        np.timedelta64(3, "ms"),
        np.timedelta64(2000000, "us"),
        None,
    ]

    expected_output = pd.DataFrame(
        {
            "OUTPUT": pd.to_datetime(in_dfs["TABLE1"]["TIMESTAMPS"], format="mixed")
            - in_dfs["TABLE1"]["INTERVALS"]
        }
    )

    check_query(
        query,
        in_dfs,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "interval_amt",
    [
        pytest.param(100, id="integer"),
        pytest.param("INTERVAL '4' days + INTERVAL '6' hours", id="timedelta_add"),
        pytest.param(
            "INTERVAL '4' days - INTERVAL '6' hours",
            id="timedelta_sub",
            marks=pytest.mark.slow,
        ),
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
        {"TABLE1": table1},
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
        only_jit_1DVar=True,
        session_tz=tz,
    )


def test_yearweek(spark_info, dt_fn_dataframe, memory_leak_check):
    """Test for YEARWEEK, which returns a 6-character string
    with the date's year and week (1-53) concatenated together"""
    query = "SELECT YEARWEEK(timestamps) AS OUTPUT from table1"

    expected_output = pd.DataFrame(
        {
            "OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.year * 100
            + dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.isocalendar().week
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearweek(tz_aware_df, memory_leak_check):
    """Test for YEARWEEK on timezone aware data.
    YEARWEEK returns a 6-character string with the date's year
    and week (1-53) concatenated together"""

    query = "SELECT YEARWEEK(A) AS OUTPUT from table1"

    expected_output = pd.DataFrame(
        {
            "OUTPUT": tz_aware_df["TABLE1"]["A"].dt.year * 100
            + tz_aware_df["TABLE1"]["A"].dt.isocalendar().week
        }
    )
    check_query(
        query,
        tz_aware_df,
        spark=None,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


def test_yearweek_scalars(spark_info, dt_fn_dataframe, memory_leak_check):
    query = "SELECT CASE WHEN YEARWEEK(timestamps) = 0 THEN -1 ELSE YEARWEEK(timestamps) END AS OUTPUT from table1"

    expected_output = pd.DataFrame(
        {
            "OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.year * 100
            + dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.isocalendar().week
        }
    )

    check_query(
        query,
        dt_fn_dataframe,
        spark_info,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


# NOTE (allai5): since DATE_TRUNC and TRUNC (datetime version)
# map to the same codegen, it is believed that testing TRUNC
# with the below two DATE_TRUNC unit tests is sufficient.
@pytest.mark.parametrize("sql_func", ["DATE_TRUNC", "TRUNC"])
def test_date_trunc_time_part(sql_func, time_df, time_part_strings, memory_leak_check):
    query = f"SELECT {sql_func}('{time_part_strings}', A) as output from table1"
    scalar_func = generate_date_trunc_time_func(time_part_strings)
    output = pd.DataFrame({"OUTPUT": time_df["TABLE1"]["A"].map(scalar_func)})
    check_query(
        query,
        time_df,
        None,
        check_dtype=False,
        expected_output=output,
    )


@pytest.mark.parametrize("sql_func", ["DATE_TRUNC", "TRUNC"])
def test_date_trunc_day_part_handling(
    sql_func, time_df, day_part_strings, memory_leak_check
):
    query = f"SELECT {sql_func}('{day_part_strings}', A) as output from table1"
    output = pd.DataFrame({"OUTPUT": []})
    with pytest.raises(
        Exception,
        match="Unsupported unit for DATE_TRUNC with TIME input: ",
    ):
        check_query(
            query,
            time_df,
            None,
            check_dtype=False,
            expected_output=output,
        )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_date_trunc_date(date_df, day_part_strings, use_case, memory_leak_check):
    """
    test DATE_TRUNC works for datetime.date input
    """
    if use_case:
        query = f"SELECT CASE WHEN A IS NULL THEN NULL ELSE DATE_TRUNC('{day_part_strings}', A) END as output from table1"
    else:
        query = f"SELECT DATE_TRUNC('{day_part_strings}', A) as output from table1"
    scalar_func = generate_date_trunc_date_func(day_part_strings)
    output = pd.DataFrame({"OUTPUT": date_df["TABLE1"]["A"].map(scalar_func)})
    check_query(
        query,
        date_df,
        None,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
def test_date_trunc_time_part_handling(
    date_df, time_part_strings, use_case, memory_leak_check
):
    """
    test DATE_TRUNC can return the same date when date_or_time_expr is datetime.date
    and date_or_time_part is smaller than day.
    """
    if use_case:
        query = f"SELECT CASE WHEN A IS NULL THEN NULL ELSE DATE_TRUNC('{time_part_strings}', A) END AS OUTPUT from table1"
    else:
        query = f"SELECT DATE_TRUNC('{time_part_strings}', A) as output from table1"
    output = pd.DataFrame({"OUTPUT": date_df["TABLE1"]["A"]})
    check_query(
        query,
        date_df,
        None,
        expected_output=output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_date_trunc_timestamp(
    dt_fn_dataframe, date_trunc_literal, use_case, memory_leak_check
):
    """
    test DATE_TRUNC works for timestamp input
    """
    if use_case:
        query = f"SELECT CASE WHEN TIMESTAMPS IS NULL THEN NULL ELSE DATE_TRUNC('{date_trunc_literal}', TIMESTAMPS) END as output from table1"
    else:
        query = f"SELECT DATE_TRUNC('{date_trunc_literal}', TIMESTAMPS) as output from table1"
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    py_output = pd.DataFrame(
        {"OUTPUT": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].map(scalar_func)}
    )
    check_query(query, dt_fn_dataframe, None, expected_output=py_output)


def test_date_trunc_unquoted_timeunit(dt_fn_dataframe, memory_leak_check):
    """
    Test DATE_TRUNC works for unquoted time unit input
    """
    query_fmt = "DATE_TRUNC({!s}, TIMESTAMPS) AS my_{}"
    selects = []
    py_output = pd.DataFrame()
    for unit in [
        "year",
        "quarter",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    ]:
        selects.append(query_fmt.format(unit, unit))
        scalar_func = generate_date_trunc_func(unit)
        py_output[unit] = dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].map(scalar_func)
    query = f"SELECT {', '.join(selects)} FROM table1"
    check_query(
        query, dt_fn_dataframe, None, check_names=False, expected_output=py_output
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweekiso(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on columns.
    """
    query = "SELECT YEAROFWEEKISO(A) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame({"A": tz_aware_df["TABLE1"]["A"].dt.year})
    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_yearofweekiso(dt_fn_dataframe, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on columns.
    """
    query = "SELECT YEAROFWEEKISO(TIMESTAMPS) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {"A": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.isocalendar().year}
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweekiso(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on timezone-aware columns.
    """
    query = "SELECT YEAROFWEEKISO(A) as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {"A": tz_aware_df["TABLE1"]["A"].dt.isocalendar().year}
    )
    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_yearofweekiso_scalar(dt_fn_dataframe, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on scalars.
    """
    query = "SELECT CASE WHEN YEAROFWEEKISO(TIMESTAMPS) > 2015 THEN 1 ELSE 0 END as A from table1"
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {
            "A": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"]
            .dt.isocalendar()
            .year.apply(lambda x: 1 if pd.notna(x) and x > 2015 else 0)
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.tz_aware
def test_tz_aware_yearofweekiso_scalar(tz_aware_df, memory_leak_check):
    """
    Test Snowflake's yearofweekiso function on timezone-aware scalars.
    """
    query = (
        "SELECT CASE WHEN YEAROFWEEKISO(A) > 2015 THEN 1 ELSE 0 END as A from table1"
    )
    # Use expected output because this function isn't in SparkSQL
    expected_output = pd.DataFrame(
        {
            "A": tz_aware_df["TABLE1"]["A"]
            .dt.isocalendar()
            .year.apply(lambda x: 1 if pd.notna(x) and x > 2015 else 0)
        }
    )
    check_query(
        query,
        tz_aware_df,
        None,
        expected_output=expected_output,
        check_dtype=False,
    )


def test_weekiso(dt_fn_dataframe, memory_leak_check):
    query = "SELECT WEEKISO(timestamps) as expected from table1"

    expected_output = pd.DataFrame(
        {"EXPECTED": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.isocalendar().week}
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


def test_weekiso_scalar(dt_fn_dataframe, memory_leak_check):
    query = "SELECT CASE WHEN WEEKISO(timestamps) = 0 THEN -1 ELSE WEEKISO(timestamps) END as EXPECTED from table1"

    expected_output = pd.DataFrame(
        {"EXPECTED": dt_fn_dataframe["TABLE1"]["TIMESTAMPS"].dt.isocalendar().week}
    )

    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_dtype=False,
        only_python=True,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_weekiso(tz_aware_df, memory_leak_check):
    """simplest weekiso test on timezone aware data"""
    query = "SELECT WEEKISO(A) as EXPECTED from table1"

    expected_output = pd.DataFrame(
        {"EXPECTED": tz_aware_df["TABLE1"]["A"].dt.isocalendar().week}
    )
    check_query(
        query,
        tz_aware_df,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.tz_aware
def test_tz_aware_weekiso_case(tz_aware_df, memory_leak_check):
    """weekiso test in case statement on timezone aware data"""
    query = "SELECT CASE WHEN WEEKISO(A) > 2 THEN WEEKISO(A) ELSE 0 END as EXPECTED from table1"

    weekiso_series = tz_aware_df["TABLE1"]["A"].dt.isocalendar().week
    weekiso_series[weekiso_series <= 2] = 0

    expected_output = pd.DataFrame({"EXPECTED": weekiso_series})
    check_query(
        query,
        tz_aware_df,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


dm = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}


@pytest.mark.parametrize("next_or_prev", ["NEXT", "PREVIOUS"])
@pytest.mark.parametrize("dow_str", ["days_of_week", "su"])
def test_next_previous_day_cols(
    dt_fn_dataframe, next_or_prev, dow_str, memory_leak_check
):
    if dow_str in dm.keys():
        query = (
            f"SELECT {next_or_prev}_DAY(timestamps, '{dow_str}') AS OUTPUT from table1"
        )
    else:
        query = (
            f"SELECT {next_or_prev}_DAY(timestamps, {dow_str}) AS OUTPUT from table1"
        )

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
        dt_fn_dataframe["TABLE1"]["DAYS_OF_WEEK"]
        if dow_str == "days_of_week"
        else np.array([dow_str])
    )
    py_output = pd.DataFrame(
        {"OUTPUT": next_prev_day(dt_fn_dataframe["TABLE1"]["TIMESTAMPS"], dow_col)}
    )
    py_output = pd.DataFrame(
        {"OUTPUT": py_output.apply(lambda s: s.iloc[0].date(), axis=1)}
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.parametrize("next_or_prev", ["NEXT", "PREVIOUS"])
@pytest.mark.parametrize("dow_str", ["days_of_week", "su"])
def test_next_previous_day_scalars(
    dt_fn_dataframe, next_or_prev, dow_str, memory_leak_check
):
    if dow_str in dm.keys():
        query = f"SELECT CASE WHEN MONTH({next_or_prev}_DAY(timestamps, '{dow_str}')) < 4 THEN DATE '2021-05-31' ELSE {next_or_prev}_DAY(timestamps, '{dow_str}') END AS OUTPUT from table1"
    else:
        query = f"SELECT CASE WHEN MONTH({next_or_prev}_DAY(timestamps, {dow_str})) < 4 THEN DATE '2021-05-31' ELSE {next_or_prev}_DAY(timestamps, {dow_str}) END AS OUTPUT from table1"

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
        dt_fn_dataframe["TABLE1"]["DAYS_OF_WEEK"]
        if dow_str == "days_of_week"
        else np.array([dow_str])
    )
    py_output = pd.DataFrame(
        {
            "OUTPUT": next_prev_day_case(
                dt_fn_dataframe["TABLE1"]["TIMESTAMPS"], dow_col
            ).dt.date
        }
    )
    check_query(
        query,
        dt_fn_dataframe,
        None,
        check_dtype=False,
        expected_output=py_output,
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_day(tz_aware_df, memory_leak_check):
    query = "SELECT DAY(A) as m from table1"
    df = tz_aware_df["TABLE1"]
    py_output = pd.DataFrame({"M": df.A.dt.day})
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_day_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN DAY(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="145D27h37min48s",
                periods=30,
                tz="Poland",
                unit="ns",
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"TABLE1": df}

    day_series = df.A.dt.day
    day_series[~df.B] = None
    py_output = pd.DataFrame({"M": day_series})

    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_extract_yhms(tz_aware_df, memory_leak_check):
    query = "SELECT EXTRACT(YEAR from A) as my_yr, EXTRACT(HOUR from A) as h, \
                    EXTRACT(MINUTE from A) as m, EXTRACT(SECOND from A) as s \
                    from table1"
    df = tz_aware_df["TABLE1"]
    py_output = pd.DataFrame(
        {
            "MY_YR": df.A.dt.year,
            "H": df.A.dt.hour,
            "M": df.A.dt.minute,
            "S": df.A.dt.second,
        }
    )
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_year_hr_min_sec(tz_aware_df, memory_leak_check):
    query = "SELECT YEAR(A) as my_yr, HOUR(A) as h, MINUTE(A) as m, SECOND(A) as s from table1"
    df = tz_aware_df["TABLE1"]
    py_output = pd.DataFrame(
        {
            "MY_YR": df.A.dt.year,
            "H": df.A.dt.hour,
            "M": df.A.dt.minute,
            "S": df.A.dt.second,
        }
    )
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_month(tz_aware_df, memory_leak_check):
    query = "SELECT MONTH(A) as m from table1"
    df = tz_aware_df["TABLE1"]
    py_output = pd.DataFrame({"M": df.A.dt.month})
    check_query(query, tz_aware_df, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_month_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN MONTH(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"TABLE1": df}
    month_series = df.A.dt.month
    month_series[~df.B] = None
    py_output = pd.DataFrame({"M": month_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.fixture(
    params=[
        pytest.param(("US/Pacific", "1/1/2023", "h"), id="pacific-by_hour"),
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
    D = pd.date_range(start="1/1/2020", tz=tz, end=end, freq=freq, unit="ns")
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
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
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

    ctx = {"TABLE1": large_tz_df}

    py_output = pd.DataFrame(
        {
            "a": large_tz_df.A,
            "w": large_tz_df.A.dt.isocalendar().week,
            "q": large_tz_df.A.dt.quarter,
            "d": large_tz_df.A.map(lambda x: None if pd.isna(x) else x.day_name()[:3]),
            "m": large_tz_df.A.map(
                lambda x: None if pd.isna(x) else x.month_name()[:3]
            ),
            "m2": large_tz_df.A.map(
                lambda x: None if pd.isna(x) else x.month_name()[:3]
            ),
        }
    )
    if case:
        py_output["w"] = py_output["w"].where(large_tz_df["B"], None)
        py_output["q"] = py_output["q"].where(large_tz_df["B"], None)
        py_output["d"] = py_output["d"].where(large_tz_df["B"], None)
        py_output["m"] = py_output["m"].where(large_tz_df["B"], None)
        py_output["m2"] = py_output["m2"].where(large_tz_df["B"], None)
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
        pytest.param(True, id="case", marks=pytest.mark.slow),
    ],
)
def test_tz_aware_dayof_fns(large_tz_df, case, memory_leak_check):
    """Tests the BodoSQL functions DAYOFWEEK, DAYOFWEEKISO, DAYOFMONTH and
    DAYOFYEAR on timezone aware data with and without case statements. The queries
    are in the following forms:

    1. SELECT A, DAYOFWEEK(A), ... from table1
    2. SELECT A, CASE WHEN B THEN DAYOFWEEK(A) ELSE NULL END, ... from table1

    Note, the two DOY functions have the following correspondance to day names:
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

    ctx = {"TABLE1": large_tz_df}
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
        py_output["dow"] = py_output["dow"].where(large_tz_df["B"], None)
        py_output["dowiso"] = py_output["dowiso"].where(large_tz_df["B"], None)
        py_output["dom"] = py_output["dom"].where(large_tz_df["B"], None)
        py_output["doy"] = py_output["doy"].where(large_tz_df["B"], None)

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
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            ).to_series()
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({"M": df.A.dt.isocalendar().week})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_weekofyear_case(memory_leak_check):
    query = "SELECT CASE WHEN B THEN WEEKOFYEAR(A) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            ).to_series(),
            "B": [True, False] * 15,
        }
    )
    ctx = {"TABLE1": df}
    week_series = df.A.dt.isocalendar().week
    week_series[~df.B] = None
    py_output = pd.DataFrame({"M": week_series})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_next_day(memory_leak_check):
    query = "SELECT next_day(A, B) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz="Africa/Casablanca",
                unit="ns",
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
        }
    )
    ctx = {"TABLE1": df}
    out_series = df.apply(
        lambda row: (
            row["A"].normalize()
            + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    py_output = pd.DataFrame({"M": out_series.dt.date})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_next_day_case(
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN next_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz="Europe/Berlin",
                unit="ns",
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"TABLE1": df}
    week_series = df.apply(
        lambda row: (
            row["A"].normalize()
            + pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"M": week_series.dt.date})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_previous_day(memory_leak_check):
    query = "SELECT previous_day(A, B) as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022", freq="16D5h", periods=30, tz="Poland", unit="ns"
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
        }
    )
    ctx = {"TABLE1": df}
    out_series = df.apply(
        lambda row: (
            row["A"].normalize()
            - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    py_output = pd.DataFrame({"M": out_series.dt.date})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_tz_aware_previous_day_case(
    memory_leak_check,
):
    query = "SELECT CASE WHEN C THEN previous_day(A, B) END as m from table1"
    df = pd.DataFrame(
        {
            "A": pd.date_range(
                start="1/1/2022",
                freq="16D5h",
                periods=30,
                tz="Pacific/Honolulu",
                unit="ns",
            ).to_series(),
            "B": ["Monday", "Tuesday"] * 15,
            "C": [True, False, True, True, False] * 6,
        }
    )
    ctx = {"TABLE1": df}
    week_series = df.apply(
        lambda row: (
            row["A"].normalize()
            - pd.offsets.Week(n=1, weekday=0 if row["B"] == "Monday" else 1)
        ).tz_localize(None),
        axis=1,
    )
    week_series[~df.C] = None
    py_output = pd.DataFrame({"M": week_series.dt.date})
    check_query(query, ctx, None, expected_output=py_output, check_dtype=False)


@pytest.mark.tz_aware
def test_date_trunc_tz_aware(date_trunc_literal, memory_leak_check):
    query = f"SELECT DATE_TRUNC('{date_trunc_literal}', A) as output from table1"
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None] * 2,
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    py_output = pd.DataFrame({"OUTPUT": df["A"].map(scalar_func)})
    check_query(query, ctx, None, expected_output=py_output)


@pytest.mark.tz_aware
@pytest.mark.slow
def test_date_trunc_tz_aware_case(date_trunc_literal, memory_leak_check):
    query = f"SELECT CASE WHEN B THEN DATE_TRUNC('{date_trunc_literal}', A) END as output from table1"
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None] * 2,
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    scalar_func = generate_date_trunc_func(date_trunc_literal)
    S = df["A"].map(scalar_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(query, ctx, None, expected_output=py_output, session_tz="US/Pacific")


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_add_sub_interval_year(representative_tz, memory_leak_check):
    """
    Test +/- Interval Year on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT A + Interval 1 Year as output from table1"
    query2 = "SELECT A - Interval 2 Year as output from table1"
    py_output = pd.DataFrame({"OUTPUT": df.A + pd.DateOffset(years=1)})
    check_query(query1, ctx, None, expected_output=py_output)
    py_output = pd.DataFrame({"OUTPUT": df.A - pd.DateOffset(years=2)})
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
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Year END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Year END as output from table1"
    S = df.A + pd.DateOffset(years=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query1, ctx, None, expected_output=py_output, session_tz=representative_tz
    )
    S = df.A - pd.DateOffset(years=2)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query2, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_month(representative_tz, memory_leak_check):
    """
    Test +/- Interval Month on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT A + Interval 1 Month as output from table1"
    query2 = "SELECT A - Interval 2 Month as output from table1"
    py_output = pd.DataFrame({"OUTPUT": df.A + pd.DateOffset(months=1)})
    check_query(
        query1, ctx, None, expected_output=py_output, session_tz=representative_tz
    )
    py_output = pd.DataFrame({"OUTPUT": df.A - pd.DateOffset(months=2)})
    check_query(
        query2, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_add_sub_interval_month_case(representative_tz, memory_leak_check):
    """
    Test +/- Interval Month on tz-aware data with case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Month END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Month END as output from table1"
    S = df.A + pd.DateOffset(months=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query1, ctx, None, expected_output=py_output, session_tz=representative_tz
    )
    S = df.A - pd.DateOffset(months=2)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query2, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_day(representative_tz, memory_leak_check):
    """
    Test +/- Interval Day on tz-aware data.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT A + Interval 1 Day as output from table1"
    query2 = "SELECT A - Interval 2 Day as output from table1"
    # Function used to simulate the result of adding by a day
    scalar_add_func = interval_day_add_func(1)
    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)
    py_output = pd.DataFrame({"OUTPUT": df.A.map(scalar_add_func)})
    check_query(
        query1, ctx, None, expected_output=py_output, session_tz=representative_tz
    )
    py_output = pd.DataFrame({"OUTPUT": df.A.map(scalar_sub_func)})
    check_query(
        query2, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.tz_aware
def test_tz_aware_add_sub_interval_day_case(representative_tz, memory_leak_check):
    """
    Test +/- Interval Day on tz-aware data with case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz=representative_tz,
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT CASE WHEN B THEN A + Interval 1 Day END as output from table1"
    query2 = "SELECT CASE WHEN B THEN A - Interval 2 Day END as output from table1"
    # Function used to simulate the result of adding by a day
    scalar_add_func = interval_day_add_func(1)
    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)
    S = df.A.map(scalar_add_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query1, ctx, None, expected_output=py_output, session_tz=representative_tz
    )
    S = df.A.map(scalar_sub_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(
        query2, ctx, None, expected_output=py_output, session_tz=representative_tz
    )


@pytest.mark.tz_aware
@pytest.mark.slow
def test_tz_aware_subdate_integer(memory_leak_check):
    """
    Test subdate on tz-aware data with an integer argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT SUBDATE(A, 3) as output from table1"
    query2 = "SELECT DATE_SUB(A, 3) as output from table1"

    # Function used to simulate the result of subtracting 3 days
    scalar_sub_func = interval_day_add_func(-3)
    py_output = pd.DataFrame({"OUTPUT": df.A.map(scalar_sub_func)})
    check_query(query1, ctx, None, expected_output=py_output, session_tz="US/Pacific")
    check_query(query2, ctx, None, expected_output=py_output, session_tz="US/Pacific")


@pytest.mark.tz_aware
def test_tz_aware_subdate_integer_case(memory_leak_check):
    """
    Test subdate on tz-aware data with an integer argument and case.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT CASE WHEN B THEN SUBDATE(A, 3) END as output from table1"
    query2 = "SELECT CASE WHEN B THEN DATE_SUB(A, 3) END as output from table1"

    # Function used to simulate the result of subtracting 3 days
    scalar_sub_func = interval_day_add_func(-3)
    S = df.A.map(scalar_sub_func)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(query1, ctx, None, expected_output=py_output, session_tz="US/Pacific")
    check_query(query2, ctx, None, expected_output=py_output, session_tz="US/Pacific")


@pytest.mark.tz_aware
def test_tz_aware_subdate_interval_day(memory_leak_check):
    """
    Test subdate on tz-aware data with a Day Interval argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT SUBDATE(A, Interval 2 Days) as output from table1"
    query2 = "SELECT DATE_SUB(A, Interval 2 Days) as output from table1"

    # Function used to simulate the result of subtracting 2 days
    scalar_sub_func = interval_day_add_func(-2)

    py_output = pd.DataFrame({"OUTPUT": df.A.map(scalar_sub_func)})
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
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
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
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(query1, ctx, None, expected_output=py_output, session_tz="US/Pacific")
    check_query(query2, ctx, None, expected_output=py_output, session_tz="US/Pacific")


@pytest.mark.tz_aware
def test_tz_aware_subdate_interval_month(memory_leak_check):
    """
    Test subdate on tz-aware data with a Month Interval argument.
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT SUBDATE(A, Interval 4 Months) as output from table1"
    query2 = "SELECT DATE_SUB(A, Interval 4 Months) as output from table1"

    py_output = pd.DataFrame({"OUTPUT": df.A - pd.DateOffset(months=4)})
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
                    start="1/1/2022",
                    freq="16D5h",
                    periods=30,
                    tz="US/Pacific",
                    unit="ns",
                )
            )
            + [None, None],
            "B": [True, False, True, True] * 8,
        }
    )
    ctx = {"TABLE1": df}
    query1 = "SELECT CASE WHEN B THEN SUBDATE(A, Interval 1 Months) END as output from table1"
    query2 = "SELECT CASE WHEN B THEN DATE_SUB(A, Interval 1 Months) END as output from table1"

    S = df.A - pd.DateOffset(months=1)
    S[~df.B] = None
    py_output = pd.DataFrame({"OUTPUT": S})
    check_query(query1, ctx, None, expected_output=py_output, session_tz="US/Pacific")
    check_query(query2, ctx, None, expected_output=py_output, session_tz="US/Pacific")


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
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
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
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({0: pd.Series(list(answer))})
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        check_names=False,
    )


@pytest.fixture
def date_from_parts_float_data():
    years = pd.Series(
        [2014.5, 2018.1, 2021.0, 2024.3, 2025.4, 2027.2, 2029.6],
        dtype=pd.Float64Dtype(),
    )
    months = pd.Series([1.1, None, 7.2, 20.3, -1.4, 0.1, 6.5], dtype=pd.Float64Dtype())
    days = pd.Series([1.1, 12.2, 4.3, -4.5, 12.4, 0.6, 0], dtype=pd.Float64Dtype())
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
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
def test_date_from_float_parts(date_from_parts_float_data, use_case, memory_leak_check):
    if use_case:
        query = "SELECT CASE WHEN YR < 0 THEN NULL ELSE DATE_FROM_PARTS(YR, MO, DA) END FROM table1"
    else:
        query = "SELECT DATE_FROM_PARTS(YR, MO, DA) FROM table1"
    year, month, day, answer = date_from_parts_float_data
    df = pd.DataFrame(
        {
            "YR": year,
            "MO": month,
            "DA": day,
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame({0: pd.Series(list(answer))})
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        check_dtype=False,
        check_names=False,
    )


@pytest.fixture(
    params=[
        pytest.param(True, id="with_ns", marks=pytest.mark.slow),
        pytest.param(False, id="no_ns"),
    ]
)
def timestamp_from_parts_data(request):
    year = pd.Series(
        [2014, 2016, 2018, None, 2022, 2024, None, None, None], dtype=pd.Int64Dtype()
    )
    year_float = pd.Series(
        [
            2014.3,
            2015.6,
            2017.5,
            None,
            2022.2,
            2024.3,
            np.finfo(np.float64).max,
            2.0**70,
            -(3.0**50),
        ],
        dtype=pd.Float64Dtype(),
    )
    month = pd.Series([1, 7, 0, 12, 100, -3, 1, 1, 1], dtype=pd.Int64Dtype())
    month_float = pd.Series(
        [1.1, 7.2, 0.3, 12.4, 99.8, -2.5, 1, 1, 1], dtype=pd.Float64Dtype()
    )
    day = pd.Series([70, 12, -123, None, 1, -7, 1, 1, 1], dtype=pd.Int64Dtype())
    day_float = pd.Series(
        [70.1, 12.2, -123.3, None, 0.9, -6.6, 1, 1, 1], dtype=pd.Float64Dtype()
    )
    hour = pd.Series([0, 2, 4, None, 40, -5, 0, 0, 0], dtype=pd.Int64Dtype())
    hour_float = pd.Series(
        [0.1, 2.1, 4.1, None, 40.1, -5.1, 0, 0, 0], dtype=pd.Float64Dtype()
    )
    minute = pd.Series([15, 0, -1, 0, 65, 0, 0, 0, 0], dtype=pd.Int64Dtype())
    second = pd.Series([0, -1, 50, 3, 125, 1234, 0, 0, 0], dtype=pd.Int64Dtype())
    nanosecond = pd.Series(
        [-1, 0, 123456789, 0, 250999, -102030405060, 0, 0, 0], dtype=pd.Int64Dtype()
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
                None,
                None,
                None,
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
                None,
                None,
                None,
            ]
        )
    return (
        year,
        year_float,
        month,
        month_float,
        day,
        day_float,
        hour,
        hour_float,
        minute,
        second,
        nanosecond,
        use_nanosecond,
        answer,
    )


@pytest.mark.parametrize(
    "func",
    [
        pytest.param("TIMESTAMP_FROM_PARTS"),
        pytest.param("TIMESTAMP_NTZ_FROM_PARTS"),
        pytest.param("TIMESTAMP_LTZ_FROM_PARTS"),
        pytest.param("TIMESTAMPFROMPARTS", marks=pytest.mark.slow),
        pytest.param("TIMESTAMPNTZFROMPARTS", marks=pytest.mark.slow),
        pytest.param("TIMESTAMPLTZFROMPARTS", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "use_floats",
    [
        pytest.param(True, id="with_floats"),
        pytest.param(False, id="with_ints"),
    ],
)
def test_timestamp_from_parts(
    func, use_floats, timestamp_from_parts_data, local_tz, memory_leak_check
):
    (
        year,
        year_float,
        month,
        month_float,
        day,
        day_float,
        hour,
        hour_float,
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
        }
        else None
    )
    df = pd.DataFrame(
        {
            "YR": year if not use_floats else year_float,
            "MO": month if not use_floats else month_float,
            "DA": day if not use_floats else day_float,
            "HO": hour if not use_floats else hour_float,
            "MI": minute,
            "SE": second,
            "NS": nanosecond,
        }
    )
    ctx = {"TABLE1": df}
    py_output = pd.DataFrame(
        {
            0: pd.Series(
                [pd.Timestamp(s, tz=tz) for s in answer],
                dtype="datetime64[ns, UTC]" if tz else "datetime64[ns]",
            )
        }
    )
    check_query(query, ctx, None, expected_output=py_output, check_names=False)


@pytest.mark.parametrize(
    "func, use_case, use_timestamp",
    [
        pytest.param("TIMESTAMP_FROM_PARTS", False, False, id="no_case-time"),
        pytest.param("TIMESTAMPFROMPARTS", True, True, id="with_case-timestamp"),
        pytest.param("TIMESTAMP_NTZ_FROM_PARTS", True, False, id="with_case-time"),
        pytest.param("TIMESTAMPNTZFROMPARTS", False, True, id="no_case-timestamp"),
    ],
)
def test_timestamp_from_parts_datetime_overload(
    func, use_case, use_timestamp, memory_leak_check
):
    time_part = "TIME_FROM_PARTS(hours, minutes, seconds, nanoseconds)"
    if use_timestamp:
        time_part = (
            "TIMESTAMP_NTZ_FROM_PARTS(2020, 1, 1, hours, minutes, seconds, nanoseconds)"
        )
    fn_call = f"{func}(TO_DATE(date), {time_part})"

    query = f"SELECT {fn_call} as t FROM table1"
    if use_case:
        query = f"SELECT CASE WHEN {fn_call} > DATE '2211-01-01' THEN DATE '2211-01-01' ELSE {fn_call} END as t from table1"
    df = pd.DataFrame(
        {
            "DATE": ["1999/1/12", "2000/2/1", "2010/5/4", "2011/3/1", "2023/8/3"],
            "HOURS": [0, 1, 12, 6, 9],
            "MINUTES": [0, 2, 30, 50, 10],
            "SECONDS": [0, 3, 55, 20, 30],
            "NANOSECONDS": [0, 40001003, 550001004, 1234, 1200300],
        }
    )
    ctx = {"TABLE1": df}

    def row_to_timestamp(row):
        return pd.Timestamp(row["DATE"]) + pd.Timedelta(
            hours=row["HOURS"],
            minutes=row["MINUTES"],
            seconds=row["SECONDS"],
            nanoseconds=row["NANOSECONDS"],
        )

    py_output = pd.DataFrame(
        {"T": pd.Series([row_to_timestamp(df.loc[i]) for i in range(len(df))])}
    )
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


def date_only_single_arg_fns_time_input_handling(
    date_only_single_arg_fns, time_df, memory_leak_check
):
    query = f"SELECT {date_only_single_arg_fns}(A) as output from table1"
    output = pd.DataFrame({"OUTPUT": []})
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
def next_previous_day_time_input_handling(next_or_prev, time_df, memory_leak_check):
    query = f"SELECT {next_or_prev}_DAY(A, 'mo') as output from table1"
    output = pd.DataFrame({"OUTPUT": []})
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


def test_last_day_no_date_part(date_df, memory_leak_check):
    """
    Tests LAST_DAY function without specifying date units
    """
    query = "SELECT LAST_DAY(A) as output from table1"
    output = pd.DataFrame(
        {
            "OUTPUT": [
                last_day_scalar_fn(date_df["TABLE1"]["A"][i], "month")
                for i in range(len(date_df["TABLE1"]["A"]))
            ]
        }
    )
    check_query(
        query,
        date_df,
        None,
        expected_output=output,
    )


def test_last_day_date_part(date_df, day_part_strings, memory_leak_check):
    """
    Tests LAST_DAY function with specifying date units
    """
    query = f"SELECT LAST_DAY(B, '{day_part_strings}') as output from table1"
    unit = standardize_snowflake_date_time_part_compile_time(day_part_strings)(
        day_part_strings
    )
    output = pd.DataFrame(
        {
            "OUTPUT": [
                last_day_scalar_fn(date_df["TABLE1"]["B"][i], unit)
                for i in range(len(date_df["TABLE1"]["B"]))
            ]
        }
    )
    if unit == "day" or unit == "dd":
        with pytest.raises(
            Exception,
            match='Unsupported date/time unit "DAY" for function LAST_DAY',
        ):
            check_query(
                query,
                date_df,
                None,
                expected_output=pd.DataFrame({}),
            )
    else:
        check_query(
            query,
            date_df,
            None,
            expected_output=output,
        )


def test_last_day_time_part(date_df, time_part_strings, memory_leak_check):
    """
    Tests LAST_DAY function can throw correct error when input
    """
    query = f"SELECT LAST_DAY(B, '{time_part_strings}') as output from table1"
    with pytest.raises(
        Exception,
        match="Unsupported date/time unit .* for function LAST_DAY",
    ):
        check_query(
            query,
            date_df,
            None,
            check_names=False,
            expected_output=pd.DataFrame({}),
        )


@pytest.mark.parametrize("fn_name", ["CURDATE", "CURRENT_DATE"])
def test_current_date(fn_name, memory_leak_check):
    """
    Test CURRENT_DATE function and its alias CURDATE
    """
    query = f"SELECT {fn_name}() as output"
    check_query(
        query,
        {},
        None,
        expected_output=pd.DataFrame({"OUTPUT": [datetime.date.today()]}),
    )


def test_months_between(spark_info, date_df, memory_leak_check):
    query = "SELECT MONTHS_BETWEEN(B, A) from table1"

    check_query(
        query,
        date_df,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_add_months(spark_info, date_df, memory_leak_check):
    query = "SELECT ADD_MONTHS(A, -18) from table1"

    check_query(
        query,
        date_df,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_time_slice(memory_leak_check):
    ts = pd.Timestamp(2012, 1, 1, 12, 59, 59)
    df = pd.DataFrame({"A": pd.Series([ts] * 12, dtype="datetime64[ns]")})
    ctx = {"TABLE1": df}

    answer = pd.DataFrame(
        {
            "t1": pd.Series([pd.Timestamp(2012, 1, 1)] * 12, dtype="datetime64[ns]"),
            "t2": pd.Series([pd.Timestamp(2012, 1, 1)] * 12, dtype="datetime64[ns]"),
            "t3": pd.Series([pd.Timestamp(2012, 1, 1)] * 12, dtype="datetime64[ns]"),
            "t4": pd.Series([pd.Timestamp(2011, 12, 26)] * 12, dtype="datetime64[ns]"),
            "t5": pd.Series([pd.Timestamp(2012, 1, 1)] * 12, dtype="datetime64[ns]"),
            "t6": pd.Series(
                [pd.Timestamp(2012, 1, 1, 12)] * 12, dtype="datetime64[ns]"
            ),
            "t7": pd.Series(
                [pd.Timestamp(2012, 1, 1, 12, 59)] * 12, dtype="datetime64[ns]"
            ),
            "t8": pd.Series(
                [pd.Timestamp(2012, 1, 1, 12, 59, 59)] * 12, dtype="datetime64[ns]"
            ),
        }
    )

    time_units = ["YEAR", "QUARTER", "MONTH", "WEEK", "DAY", "HOUR", "MINUTE", "SECOND"]

    query = "SELECT "
    query += ", ".join(
        [
            f"TIME_SLICE(A, 1, '{unit}', 'START') as t{i + 1}"
            for i, unit in enumerate(time_units)
        ]
    )
    query += " FROM table1"

    check_query(
        query, ctx, None, check_names=False, check_dtype=False, expected_output=answer
    )
