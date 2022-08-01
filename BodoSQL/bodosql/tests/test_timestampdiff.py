import pandas as pd
import pytest

from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        "NANOSECOND",
        pytest.param("MICROSECOND", marks=pytest.mark.slow),
        "SECOND",
        pytest.param("MINUTE", marks=pytest.mark.slow),
        pytest.param("HOUR", marks=pytest.mark.slow),
        pytest.param("DAY", marks=pytest.mark.slow),
        "WEEK",
        pytest.param(
            "MONTH",
        ),
        pytest.param(
            "QUARTER",
            marks=(pytest.mark.slow,),
        ),
        pytest.param(
            "YEAR",
        ),
    ]
)
def interval_flag(request):
    return request.param


@pytest.fixture(
    params=[
        pd.Series(
            [
                pd.Timestamp(2021, 5, 19),
                pd.Timestamp(1999, 12, 31),
                pd.Timestamp(2020, 10, 11),
                pd.Timestamp(2025, 1, 1),
            ]
        )
    ]
)
def ts_series(request):
    return request.param


@pytest.fixture(
    params=[
        "1999-12-31",
        "2025-01-01",
    ]
)
def ts_scalar_str(request):
    return request.param


def test_timestampdiff_cols(interval_flag, ts_series, spark_info):
    "tests timestampdiff on column values"

    offset = get_offset(interval_flag)

    new_df = pd.DataFrame(
        {
            "A": ts_series,
            "B": ts_series + offset,
            "B_minus_1_ns": ts_series + offset - pd.Timedelta(1, unit="ns"),
        }
    )
    new_ctx = {"table1": new_df}

    # For any given offset, TIMESTAMPDIFF(OFFSET, A + offset, A) == 1,
    # but TIMESTAMPDIFF(OFFSET, A + offset - 1 ns, A) == 0
    query = f"SELECT TIMESTAMPDIFF({interval_flag}, B, A), TIMESTAMPDIFF({interval_flag}, A, B), TIMESTAMPDIFF({interval_flag}, B_minus_1_ns, A), TIMESTAMPDIFF({interval_flag}, A, B_minus_1_ns) from table1"

    expected_output = pd.DataFrame(
        {
            "0": -1,
            "1": 1,
            "0_1": 0,
            "0_2": 0,
        },
        index=pd.RangeIndex(len(ts_series)),
    )

    check_query(
        query,
        new_ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def test_timestampdiff_scalars(interval_flag, ts_series, spark_info):
    "tests timestampdiff on scalar values"
    offset = get_offset(interval_flag)
    new_df = pd.DataFrame(
        {
            "A": ts_series,
            "B": ts_series + offset,
            "B_minus_1_ns": ts_series + offset - pd.Timedelta(1, unit="ns"),
            "C": 1,
        }
    )
    new_ctx = {"table1": new_df}

    # For any given offset, TIMESTAMPDIFF(OFFSET, A + offset, A) == 1,
    # but TIMESTAMPDIFF(OFFSET, A + offset - 1 ns, A) == 0
    case_stmts = ", ".join(
        [
            f"CASE WHEN C < 0 THEN -10 ELSE TIMESTAMPDIFF({interval_flag}, {arg0}, {arg1}) END"
            for arg0, arg1 in [
                ("B", "A"),
                ("A", "B"),
                ("B_minus_1_ns", "A"),
                ("A", "B_minus_1_ns"),
            ]
        ]
    )
    query = f"SELECT {case_stmts} from table1"

    expected_output = pd.DataFrame(
        {
            "0": -1,
            "1": 1,
            "2": 0,
            "3": 0,
        },
        index=pd.RangeIndex(len(ts_series)),
    )

    check_query(
        query,
        new_ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_timestampdiff_col_scalar(interval_flag, ts_scalar_str, spark_info):
    "tests timestampdiff when one arg is column, and one arg is scalar"
    offset = get_offset(interval_flag)
    ts_series = pd.Series([pd.Timestamp(ts_scalar_str)] * 12)
    new_df = pd.DataFrame(
        {
            "B": ts_series + offset,
            "B_minus_1_ns": ts_series + offset - pd.Timedelta(1, unit="ns"),
        }
    )
    new_ctx = {"table1": new_df}

    # For any given offset, TIMESTAMPDIFF(OFFSET, A + offset, A) == 1,
    # but TIMESTAMPDIFF(OFFSET, A + offset - 1 ns, A) == 0
    query = f"SELECT TIMESTAMPDIFF({interval_flag}, B, TIMESTAMP '{ts_scalar_str}'), TIMESTAMPDIFF({interval_flag}, TIMESTAMP '{ts_scalar_str}', B), TIMESTAMPDIFF({interval_flag}, B_minus_1_ns, TIMESTAMP '{ts_scalar_str}'), TIMESTAMPDIFF({interval_flag}, TIMESTAMP '{ts_scalar_str}', B_minus_1_ns) from table1"

    expected_output = pd.DataFrame(
        {
            "0": -1,
            "1": 1,
            "0_1": 0,
            "0_2": 0,
        },
        index=pd.RangeIndex(len(ts_series)),
    )

    check_query(
        query,
        new_ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_timestampdiff_year_rounding(basic_df, spark_info):
    """Explicit test to insure that year rounding is working as expected"""

    args = ", ".join(
        [
            f"TIMESTAMPDIFF(YEAR, TIMESTAMP '{a}', TIMESTAMP '{b}'), TIMESTAMPDIFF(YEAR, TIMESTAMP '{b}', TIMESTAMP '{a}')"
            for a, b in [
                ("1999-10-10", "2000-10-10"),
                ("1999-10-10", "2000-10-09"),
                ("1999-10-10", "2000-09-10"),
            ]
        ]
    )
    query = "SELECT " + args

    expected_output = pd.DataFrame(
        {
            "0": 1,
            "1": -1,
            "0_1": 0,
            "0_2": 0,
            "0_3": 0,
            "0_4": 0,
        },
        index=pd.RangeIndex(1),
    )

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_timestampdiff_quarter_rounding(basic_df, spark_info):
    """Explicit test to insure that quarter rounding is working as expected"""

    args = ", ".join(
        [
            f"TIMESTAMPDIFF(QUARTER, TIMESTAMP '{a}', TIMESTAMP '{b}'), TIMESTAMPDIFF(QUARTER, TIMESTAMP '{b}', TIMESTAMP '{a}')"
            for a, b in [
                ("1999-10-10", "2000-10-10"),
                ("1999-10-10", "2000-10-09"),
                ("1999-07-10", "1999-10-10"),
                ("1999-10-10", "1999-8-10"),
            ]
        ]
    )
    query = "SELECT " + args

    expected_output = pd.DataFrame(
        {
            "0": 4,
            "1": -4,
            "2": 3,
            "3": -3,
            "4": 1,
            "5": -1,
            "6": 0,
            "7": 0,
        },
        index=pd.RangeIndex(1),
    )

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=expected_output,
    )


def get_offset(s):
    """helper function that returns timedelta/date_offset that is 1 * the specified time unit.
    Used with testing timestampdiff"""
    if s == "NANOSECOND":
        return pd.Timedelta(1, unit="ns")
    elif s == "MICROSECOND":
        return pd.Timedelta(1, unit="us")
    elif s == "SECOND":
        return pd.Timedelta(1, unit="s")
    elif s == "MINUTE":
        return pd.Timedelta(1, unit="minute")
    elif s == "HOUR":
        return pd.Timedelta(1, unit="hour")
    elif s == "DAY":
        return pd.Timedelta(1, unit="day")
    elif s == "WEEK":
        return pd.Timedelta(7, unit="day")
    elif s == "MONTH":
        return pd.DateOffset(months=1)
    elif s == "QUARTER":
        return pd.DateOffset(months=3)
    elif s == "YEAR":
        return pd.DateOffset(months=12)
