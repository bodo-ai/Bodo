"""
Test SQL `time` support (non-constructor functions)
"""

import pandas as pd
import pytest

import bodo
from bodo.tests.conftest import (  # noqa
    day_part_strings,
    time_df,
    time_part_strings,
)
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.context import BodoSQLContext
from bodosql.tests.conftest import timeadd_arguments, timeadd_dataframe  # noqa
from bodosql.tests.test_kernels.test_datetime_array_kernels import (
    diff_fn,
)
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.parametrize(
    "precision",
    [
        0,
        pytest.param(3, marks=pytest.mark.slow),
        pytest.param(6, marks=pytest.mark.slow),
        9,
    ],
)
def test_time_array_box_unbox(precision, memory_leak_check):
    query = "select * from table1"
    df = pd.DataFrame(
        {
            "A": [bodo.types.Time(0, i, 0, precision=precision) for i in range(15)],
            "B": [bodo.types.Time(1, i, 1, precision=precision) for i in range(15)],
            "C": [bodo.types.Time(2, i, 2, precision=precision) for i in range(15)],
        },
    )
    ctx = {"TABLE1": df}
    expected_output = df
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "precision",
    [
        pytest.param(0, marks=pytest.mark.slow),
        pytest.param(3, marks=pytest.mark.slow),
        pytest.param(6, marks=pytest.mark.slow),
        pytest.param(9),
    ],
)
def test_time_box_array_unbox(precision, memory_leak_check):
    query = "select B from table1"
    df = pd.DataFrame(
        {
            "A": [bodo.types.Time(0, i, 0, precision=precision) for i in range(15)],
            "B": [bodo.types.Time(1, i, 1, precision=precision) for i in range(15)],
            "C": [bodo.types.Time(2, i, 2, precision=precision) for i in range(15)],
        },
    )
    ctx = {"TABLE1": df}
    expected_output = df[["B"]]
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "unit, test_fn_type, answer, error_msg",
    [
        pytest.param(
            "hour",
            "DATE_PART",
            pd.Series([12, 1, 9, 20, 23]),
            None,
            id="valid-hour-date_part",
        ),
        pytest.param(
            "minute",
            "MINUTE",
            pd.Series([30, 2, 59, 45, 50]),
            None,
            id="valid-minute-regular",
        ),
        pytest.param(
            "second",
            "DATE_PART",
            pd.Series([15, 3, 0, 1, 59]),
            None,
            id="valid-second-date_part",
        ),
        pytest.param(
            "millisecond",
            "EXTRACT",
            pd.Series([0, 4, 100, 123, 500]),
            None,
            id="valid-millisecond-extract",
        ),
        pytest.param(
            "microsecond",
            "MICROSECOND",
            pd.Series([0, 0, 250, 456, 0]),
            None,
            id="valid-microsecond-regular",
        ),
        pytest.param(
            "nanosecond",
            "EXTRACT",
            pd.Series([0, 0, 0, 789, 999]),
            None,
            id="valid-nanosecond-extract",
        ),
        pytest.param(
            "day",
            "DATE_PART",
            None,
            r"Cannot extract unit",
            id="invalid-day-date_part",
        ),
        pytest.param(
            "dayofyear",
            "DAYOFYEAR",
            None,
            r"Cannot extract unit",
            id="invalid-dayofyear-regular",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "dow",
            "EXTRACT",
            None,
            r"Cannot apply 'EXTRACT' to arguments of type",
            id="invalid-dow-extract",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "week",
            "WEEK",
            None,
            r"Cannot extract unit",
            id="invalid-week-regular",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "weekiso",
            "DATE_PART",
            None,
            r"Cannot extract unit",
            id="invalid-weekiso-date_part",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "month",
            "EXTRACT",
            None,
            r"Cannot apply 'EXTRACT' to arguments of type",
            id="invalid-month-extract",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "quarter",
            "DATE_PART",
            None,
            r"Cannot extract unit",
            id="invalid-quarter-date_part",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "year",
            "YEAR",
            None,
            r"Cannot extract unit",
            id="invalid-year-regular",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_time_extract(unit, answer, test_fn_type, error_msg, memory_leak_check):
    """Tests EXTRACT and EXTRACT-like functions on time data, checking that
    values larger than HOUR raise an exception"""
    if test_fn_type == "EXTRACT":
        query = f"SELECT EXTRACT({unit} FROM T) AS U FROM table1"
    elif test_fn_type == "DATE_PART":
        query = f"SELECT DATE_PART('{unit}', T) AS U FROM table1"
    else:
        query = f"SELECT {test_fn_type}(T) AS U FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "T": pd.Series(
                    [
                        bodo.types.Time(12, 30, 15, precision=9),
                        bodo.types.Time(1, 2, 3, 4, precision=9),
                        bodo.types.Time(9, 59, 0, 100, 250, precision=9),
                        bodo.types.Time(20, 45, 1, 123, 456, 789, precision=9),
                        bodo.types.Time(23, 50, 59, 500, 0, 999, precision=9),
                    ]
                )
            }
        )
    }
    if answer is None:
        bc = BodoSQLContext(ctx)
        with pytest.raises(Exception, match=error_msg):
            bc.sql(query)
    else:
        expected_output = pd.DataFrame({"U": answer})
        check_query(
            query, ctx, None, expected_output=expected_output, check_dtype=False
        )


@pytest.mark.parametrize(
    "query, answer",
    [
        pytest.param(
            "SELECT DATE_ADD(TI, TD) FROM table1",
            pd.Series(
                [
                    bodo.types.Time(13, 30, 0),
                    None,
                    bodo.types.Time(21, 40, 13),
                    bodo.types.Time(2, 3, 2, microsecond=1),
                    bodo.types.Time(5, 45, 3, nanosecond=625999250),
                ]
            ),
            id="date_add-timedelta_array",
        ),
        pytest.param(
            "SELECT DATE_SUB(TI, Interval '30' minutes) FROM table1",
            pd.Series(
                [
                    bodo.types.Time(12, 0, 0),
                    None,
                    bodo.types.Time(0, 30, 13),
                    bodo.types.Time(21, 30, 0),
                    bodo.types.Time(5, 15, 0, nanosecond=125999250),
                ]
            ),
            id="date_sub-interval_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT TI + Interval '3' hours FROM table1",
            pd.Series(
                [
                    bodo.types.Time(15, 30, 0),
                    None,
                    bodo.types.Time(4, 0, 13),
                    bodo.types.Time(1, 0, 0),
                    bodo.types.Time(8, 45, 0, nanosecond=125999250),
                ]
            ),
            id="addition-interval_scalar",
        ),
        pytest.param(
            "SELECT TI - TD FROM table1",
            pd.Series(
                [
                    bodo.types.Time(11, 30, 0),
                    None,
                    bodo.types.Time(4, 20, 13),
                    bodo.types.Time(17, 56, 57, microsecond=999999),
                    bodo.types.Time(5, 44, 56, nanosecond=625999250),
                ]
            ),
            id="subtraction-timedelta_array",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_time_plus_minus_intervals(query, answer, memory_leak_check):
    """Tests adding/subtracting intervals to/from TIME values.

    NOTE: intervals with units larger than hour __should__ cause
    an error at compile time, but currently BodoSQL is not able to
    know if this is the case since intervals are converted into
    timedeltas before the binop is computed, which means it is impossible
    to tell if the timedelta is from a unit that is too large or just
    a value from the legal units that has overflowed.

    Example: TIME + (1 day + 1 hour) should not be allowed (but currently is)
    Example: TIME + (25 hours) is allowed"""
    TI = pd.Series(
        [
            bodo.types.Time(12, 30, 0),
            None,
            bodo.types.Time(1, 0, 13),
            bodo.types.Time(22, 0, 0),
            bodo.types.Time(5, 45, 0, nanosecond=125999250),
        ]
    )
    TD = pd.Series(
        [
            pd.Timedelta(hours=1),
            pd.Timedelta(hours=2),
            pd.Timedelta(minutes=-200),
            pd.Timedelta(hours=4, minutes=3, seconds=2, microseconds=1),
            pd.Timedelta(seconds=3, milliseconds=500),
        ],
        dtype="timedelta64[ns]",
    )
    ctx = {"TABLE1": pd.DataFrame({"TI": TI, "TD": TD})}
    expected_output = pd.DataFrame({0: answer})
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
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
def test_timeadd(timeadd_dataframe, timeadd_arguments, use_case, memory_leak_check):
    unit, answer = timeadd_arguments
    # Decide which function to use based on the unit
    func = {
        "HOUR": "DATEADD",
        "MINUTE": "TIMEADD",
        "SECOND": "DATEADD",
        "MILLISECOND": "TIMEADD",
        "MICROSECOND": "DATEADD",
        "NANOSECOND": "TIMEADD",
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
    )


@pytest.mark.parametrize(
    "calculation, error_msg",
    [
        pytest.param(
            "DATEADD('yrs', 1, T)",
            "Unsupported unit for DATEADD with TIME input:",
            id="dateadd-year",
        ),
        pytest.param(
            "TIMEADD('mon', 6, T)",
            "Unsupported unit for DATEADD with TIME input:",
            id="timeadd-month",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TIMESTAMPADD(WEEK, -1, T)",
            "Invalid time unit input for TIMESTAMPADD: When arg2 is a time, the specified time unit must be smaller than day.",
            id="timestampadd-week",
            marks=pytest.mark.slow,
        ),
        pytest.param("DATE_ADD(10, T)", "", id="date_add-day"),
        pytest.param("ADDATE(13, T)", "", id="addate-day", marks=pytest.mark.slow),
        pytest.param(
            "DATE_SUB(T, 1)",
            "Cannot add/subtract days from TIME",
            id="date_sub-day",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SUBDATE(T, 2)", "Cannot add/subtract days from TIME", id="subdate-day"
        ),
        pytest.param(
            "DATEDIFF(QUARTER, T, T)",
            "Unsupported unit for DATEDIFF with TIME input: ",
            id="datediff-quarter",
        ),
        pytest.param(
            "TIMEDIFF('wy', T, T)",
            "Unsupported unit for DATEDIFF with TIME input: ",
            id="timediff-week",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "TIMESTAMPDIFF(DAY, T, T)",
            "Unsupported unit for DATEDIFF with TIME input: ",
            id="timestampdiff-day",
        ),
    ],
)
def test_timeadd_timediff_invalid_units(timeadd_dataframe, calculation, error_msg):
    """Tests various date/time addition/subtraction functions on TIME data with
    invalid units to ensure that they raise an error"""
    query = f"SELECT {calculation} FROM table1"
    bc = BodoSQLContext(timeadd_dataframe)
    with pytest.raises(Exception, match=error_msg):
        bc.sql(query)


@pytest.mark.parametrize(
    "query, expected_output",
    [
        pytest.param(
            "SELECT DATEDIFF(HOUR, TO_TIME('10:10:10'), TO_TIME('22:33:33'))",
            pd.DataFrame({"A": pd.Series([12])}),
            id="hour",
        ),
        pytest.param(
            "SELECT TIMEDIFF('MINUTE', TO_TIME('12:10:05'), TO_TIME('10:10:10'))",
            pd.DataFrame({"A": pd.Series([-120])}),
            id="minute",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF(SECOND, TO_TIME('22:33:33'), TO_TIME('12:10:05'))",
            pd.DataFrame({"A": pd.Series([-37408])}),
            id="second",
        ),
        pytest.param(
            "SELECT DATEDIFF('MILLISECOND', TO_TIME('10:10:10'), TO_TIME('12:10:05'))",
            pd.DataFrame({"A": pd.Series([7195000])}),
            id="millisecond",
        ),
        pytest.param(
            "SELECT TIMEDIFF(MICROSECOND, TO_TIME('22:33:33'), TO_TIME('12:10:05'))",
            pd.DataFrame({"A": pd.Series([-37408000000])}),
            id="microsecond",
        ),
        pytest.param(
            "SELECT TIMESTAMPDIFF('NANOSECOND', TO_TIME('22:33:33'), TO_TIME('10:10:10'))",
            pd.DataFrame({"A": pd.Series([-44603000000000])}),
            id="nanosecond",
        ),
    ],
)
@pytest.mark.slow
def test_datediff_time_literals(query, expected_output, basic_df, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF on bodo.types.Time literals behaves as expected.
    Tests all possible datetime parts except for time units.
    """

    check_query(
        query,
        basic_df,
        spark=None,
        expected_output=expected_output,
        check_names=False,
        check_dtype=False,
    )


def test_datediff_time_columns(time_df, time_part_strings, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF on columns behaves as expected
    """
    fn_name = {
        "HOUR": "DATEDIFF",
        "hr": "TIMEDIFF",
        "MINUTE": "TIMESTAMPDIFF",
        "min": "DATEDIFF",
        "SECOND": "TIMEDIFF",
        "ms": "TIMESTAMPDIFF",
        "microsecond": "DATEDIFF",
        "usec": "TIMEDIFF",
        "nanosecs": "TIMESTAMPDIFF",
    }[time_part_strings]
    query = f"SELECT {fn_name}('{time_part_strings}', A, B) as output from table1"
    output = pd.DataFrame(
        {
            "OUTPUT": [
                diff_fn(
                    time_part_strings,
                    time_df["TABLE1"]["A"][i],
                    time_df["TABLE1"]["B"][i],
                )
                for i in range(len(time_df["TABLE1"]["A"]))
            ]
        }
    )
    check_query(
        query,
        time_df,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=output,
    )


def test_datediff_time_day_part_handling(time_df, day_part_strings, memory_leak_check):
    """
    Checks that calling DATEDIFF/TIMEDIFF/TIMESTAMPDIFF throws an error when a date part is passed in as the unit
    """
    symbol_name, fn_name = {
        "quarter": ("QUARTER", "DATEDIFF"),
        "yyy": ("YEAR", "TIMEDIFF"),
        "MONTH": ("MONTH", "TIMESTAMPDIFF"),
        "mon": ("MONTH", "DATEDIFF"),
        "WEEK": ("WEEK", "TIMEDIFF"),
        "wk": ("WEEK", "TIMESTAMPDIFF"),
        "DAY": ("DAY", "DATEDIFF"),
        "dd": ("DAY", "TIMEDIFF"),
    }[day_part_strings]

    query = f"SELECT {fn_name}('{day_part_strings}', A, B) as output from table1"
    output = pd.DataFrame({"output": []})
    with pytest.raises(
        Exception,
        match=f"Unsupported unit for DATEDIFF with TIME input: {symbol_name}",
    ):
        check_query(
            query,
            time_df,
            None,
            check_names=False,
            check_dtype=False,
            expected_output=output,
        )


@pytest.mark.slow
def test_max_time_types(time_df, memory_leak_check):
    """
    Simple test to ensure that max is working on time types
    """
    query = "SELECT MAX(A) FROM table1"
    check_query(
        query,
        time_df,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"A": [bodo.types.Time(22, 13, 57)]}),
        is_out_distributed=False,
    )


def test_min_time_types(time_df, memory_leak_check):
    """
    Simple test to ensure that max is working on time types
    """
    query = "SELECT MIN(A) FROM table1"
    check_query(
        query,
        time_df,
        None,
        check_names=False,
        expected_output=pd.DataFrame({"A": [bodo.types.Time()]}),
        is_out_distributed=False,
    )
