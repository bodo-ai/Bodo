"""
Test SQL `time` support (constructor functions)
"""

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.fixture(
    params=[
        pytest.param(
            "time", marks=pytest.mark.skip(reason="waiting for calcite support")
        ),
        "to_time",
        "try_to_time",
    ],
)
def to_time_fn(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [
                        "11:2:45.999",
                        "01:23:45",
                        None,
                        "01:23:45.67588",
                        "01:0:02",
                        "11:59:59",
                        "1:23:4.67588833",
                        "100",
                        "1000",
                        "1",
                        "4240",
                        "12345",
                        None,
                        "1200",
                        "0:0:0",
                        "12:30:5",
                        "22:15",
                        "16:15:14.1312",
                    ]
                ),
                pd.Series(
                    [
                        bodo.types.Time(11, 2, 45, millisecond=999, precision=9),
                        bodo.types.Time(1, 23, 45, precision=9),
                        None,
                        bodo.types.Time(1, 23, 45, microsecond=675880, precision=9),
                        bodo.types.Time(1, 0, 2, precision=9),
                        bodo.types.Time(11, 59, 59, precision=9),
                        bodo.types.Time(1, 23, 4, nanosecond=675888330, precision=9),
                        bodo.types.Time(0, 0, 100, precision=9),
                        bodo.types.Time(0, 0, 1000, precision=9),
                        bodo.types.Time(0, 0, 1, precision=9),
                        bodo.types.Time(0, 0, 4240, precision=9),
                        bodo.types.Time(0, 0, 12345, precision=9),
                        None,
                        bodo.types.Time(0, 0, 1200, precision=9),
                        bodo.types.Time(0, 0, 0, precision=9),
                        bodo.types.Time(12, 30, 5, precision=9),
                        bodo.types.Time(22, 15, 0, precision=9),
                        bodo.types.Time(16, 15, 14, microsecond=131200, precision=9),
                    ]
                ),
            ),
            id="string",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2020-12-31 01:15:59.995"),
                        pd.Timestamp("2021-4-1 12:00:00"),
                        None,
                        pd.Timestamp("2022-7-4 20:21:22"),
                        pd.Timestamp("2023-8-15 8:30:00.123456789"),
                        pd.Timestamp("2019-1-1"),
                    ],
                    dtype="datetime64[ns]",
                ),
                pd.Series(
                    [
                        bodo.types.Time(1, 15, 59, millisecond=995, precision=9),
                        bodo.types.Time(12, 0, 0, precision=9),
                        None,
                        bodo.types.Time(20, 21, 22, precision=9),
                        bodo.types.Time(8, 30, 0, nanosecond=123456789, precision=9),
                        bodo.types.Time(0, 0, 0, precision=9),
                    ]
                ),
            ),
            id="timestamp_ntz",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2020-12-31 01:15:59.995", tz="US/Pacific"),
                        pd.Timestamp("2021-4-1 12:00:00", tz="US/Pacific"),
                        None,
                        pd.Timestamp("2022-7-4 20:21:22", tz="US/Pacific"),
                        pd.Timestamp("2023-8-15 8:30:00.123456789", tz="US/Pacific"),
                        pd.Timestamp("2019-1-1", tz="US/Pacific"),
                    ],
                    dtype="datetime64[ns, US/Pacific]",
                ),
                pd.Series(
                    [
                        bodo.types.Time(1, 15, 59, millisecond=995, precision=9),
                        bodo.types.Time(12, 0, 0, precision=9),
                        None,
                        bodo.types.Time(20, 21, 22, precision=9),
                        bodo.types.Time(8, 30, 0, nanosecond=123456789, precision=9),
                        bodo.types.Time(0, 0, 0, precision=9),
                    ]
                ),
            ),
            id="timestamp_ltz",
        ),
    ],
)
def to_time_valid_data(request):
    data, result = request.param
    ctx = {"TABLE1": pd.DataFrame({"S": data, "B": [True] * len(data)})}
    return ctx, result


@pytest.fixture
def to_time_invalid_data():
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "S": pd.Series(
                    [
                        "NOT A TIME",
                        "12:00:00",
                        "1:2:3:4:5",
                        "12:00:00",
                        "1.1",
                        "12:00:00",
                        "-1",
                        "12:00:00",
                        "9:3032",
                        "12:00:00",
                        "9.30.23",
                        "12:00:00",
                        "12:400:00",
                        "12:00:00",
                        "01::20",
                        "12:00:00",
                        "12:13:14_fudge",
                        "12:00:00",
                        "1::2::03",
                        "12:00:00",
                        "12:13:14.a",
                        "12:00:00",
                        "12:13:14.123456789b",
                        "12:00:00",
                        ":1:23",
                        "12:00:00",
                        "10:29.2000",
                        "12:00:00",
                        "30:00:00",
                        "12:00:00",
                        "00:60:00",
                        "12:00:00",
                        "00:00:90",
                        "12:00:00",
                        "00:00:100",
                        "12:00:00",
                    ]
                ),
                "B": pd.Series([True] * 36),
            }
        )
    }
    answer = pd.DataFrame({"T": [None, bodo.types.Time(12, 0, 0, precision=9)] * 18})
    return ctx, answer


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_to_time_valid(to_time_fn, to_time_valid_data, use_case, memory_leak_check):
    ctx, result = to_time_valid_data
    if use_case:
        query = f"SELECT {to_time_fn}(S) as A FROM table1"
    else:
        query = f"SELECT CASE WHEN B THEN {to_time_fn}(S) END AS A FROM table1"
    ctx = ctx
    expected_output = pd.DataFrame({"A": result})
    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "arr, format_str, answer",
    [
        pytest.param(
            pd.Series(
                [
                    "12:50:45",
                    "02:43:30",
                    "10:23:00",
                    "20:00:10",
                ]
            ),
            "HH24:MI:SS",
            pd.Series(
                [
                    bodo.types.Time(12, 50, 45, precision=9),
                    bodo.types.Time(2, 43, 30, precision=9),
                    bodo.types.Time(10, 23, 0, precision=9),
                    bodo.types.Time(20, 0, 10, precision=9),
                ]
            ),
            id="format-24",
        ),
        pytest.param(
            pd.Series(
                [
                    "01:15:30 AM",
                    "11:45:15 PM",
                    "12:00:00 AM",
                    "12:30:45 PM",
                ]
            ),
            "HH12:MI:SS PM",
            pd.Series(
                [
                    bodo.types.Time(1, 15, 30, precision=9),
                    bodo.types.Time(23, 45, 15, precision=9),
                    bodo.types.Time(0, 0, 0, precision=9),
                    bodo.types.Time(12, 30, 45, precision=9),
                ]
            ),
            id="format-12",
        ),
    ],
)
def test_to_time_format_str(to_time_fn, arr, format_str, answer, memory_leak_check):
    query = f"SELECT {to_time_fn}(S, '{format_str}') as A FROM table1"
    ctx = {"TABLE1": pd.DataFrame({"S": arr})}
    expected_output = pd.DataFrame({"A": answer})

    check_query(query, ctx, None, expected_output=expected_output)


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_to_time_invalid(to_time_fn, to_time_invalid_data, use_case):
    ctx, answer = to_time_invalid_data
    if use_case:
        query = f"SELECT {to_time_fn}(S) AS T FROM table1"
    else:
        query = f"SELECT CASE WHEN B THEN {to_time_fn}(S) END AS T FROM table1"
    if "try" in to_time_fn:
        expected_output = answer
        check_query(
            query, ctx, None, expected_output=expected_output, run_dist_tests=False
        )
    else:
        with pytest.raises(ValueError, match="Invalid time string"):
            bc = bodosql.BodoSQLContext(ctx)

            @bodo.jit
            def impl(bc):
                return bc.sql(query)

            impl(bc)


@pytest.fixture(
    params=[
        pytest.param("TIME_FROM_PARTS", id="time_from_parts"),
        pytest.param("TIMEFROMPARTS", id="timefromparts", marks=pytest.mark.slow),
    ]
)
def time_from_parts_fn(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            (
                {
                    "TABLE1": pd.DataFrame(
                        {
                            "H": pd.Series(
                                [0, 10, None, 20, 30, 40, -1], dtype=pd.Int32Dtype()
                            ),
                            "M": pd.Series(
                                [30, -1, 0, 0, 90, 15, 59], dtype=pd.Int32Dtype()
                            ),
                            "S": pd.Series(
                                [0, 0, 1, 15, 100, -1, -1], dtype=pd.Int32Dtype()
                            ),
                        }
                    )
                },
                pd.DataFrame(
                    {
                        "A": pd.Series(
                            [
                                bodo.types.Time(0, 30, 0, precision=9),
                                bodo.types.Time(9, 59, 0, precision=9),
                                None,
                                bodo.types.Time(20, 0, 15, precision=9),
                                bodo.types.Time(7, 31, 40, precision=9),
                                bodo.types.Time(16, 14, 59, precision=9),
                                bodo.types.Time(23, 58, 59, precision=9),
                            ]
                        )
                    }
                ),
                "H, M, S",
            ),
            id="three_args",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                {
                    "TABLE1": pd.DataFrame(
                        {
                            "H": pd.Series(
                                [0, 12, 25, 23, 24, 0, 0, 0, 0, 12, 0],
                                dtype=pd.Int64Dtype(),
                            ),
                            "M": pd.Series(
                                [100, 0, 30, -1, 0, 0, -3, 0, 0, 500, 10**6],
                                dtype=pd.Int64Dtype(),
                            ),
                            "S": pd.Series(
                                [0, 12345, 0, 0, 0, 0, 65, 0, 0, -700, 0],
                                dtype=pd.Int64Dtype(),
                            ),
                            "NS": pd.Series(
                                [0, 0, 0, 0, 0, -1, 0, None, 2**44, -(2**30), 1234],
                                dtype=pd.Int64Dtype(),
                            ),
                        }
                    )
                },
                pd.DataFrame(
                    {
                        "A": pd.Series(
                            [
                                bodo.types.Time(1, 40, 0, precision=9),
                                bodo.types.Time(15, 25, 45, precision=9),
                                bodo.types.Time(1, 30, 0, precision=9),
                                bodo.types.Time(22, 59, 0, precision=9),
                                bodo.types.Time(0, 0, 0, precision=9),
                                bodo.types.Time(
                                    23, 59, 59, nanosecond=999999999, precision=9
                                ),
                                bodo.types.Time(23, 58, 5, precision=9),
                                None,
                                bodo.types.Time(
                                    hour=4,
                                    minute=53,
                                    second=12,
                                    nanosecond=186044416,
                                    precision=9,
                                ),
                                bodo.types.Time(
                                    hour=20,
                                    minute=8,
                                    second=18,
                                    nanosecond=926258176,
                                    precision=9,
                                ),
                                bodo.types.Time(
                                    10, 40, 0, nanosecond=1234, precision=9
                                ),
                            ]
                        )
                    }
                ),
                "H, M, S, NS",
            ),
            id="four_args",
        ),
        pytest.param(
            (
                {
                    "TABLE1": pd.DataFrame(
                        {
                            "H": pd.Series(
                                [0.1, 12.2, 25.3, 23.4, 23.5, 0.1, 0.3, 0, 0, 11.8, 0],
                                dtype=pd.Float64Dtype(),
                            ),
                            "M": pd.Series(
                                [
                                    99.99,
                                    0,
                                    29.6,
                                    -0.7,
                                    0,
                                    0,
                                    -3.4,
                                    0,
                                    0,
                                    500.01,
                                    10**6,
                                ],
                                dtype=pd.Float64Dtype(),
                            ),
                            "S": pd.Series(
                                [0, 12345.1, 0, 0, 0, 0, 65.2, 0, 0, -699.5, 0],
                                dtype=pd.Float64Dtype(),
                            ),
                            "NS": pd.Series(
                                [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    -1.1,
                                    0,
                                    None,
                                    2**44,
                                    -(2**30),
                                    1234.3,
                                ],
                                dtype=pd.Float64Dtype(),
                            ),
                        }
                    )
                },
                pd.DataFrame(
                    {
                        "A": pd.Series(
                            [
                                bodo.types.Time(1, 40, 0, precision=9),
                                bodo.types.Time(15, 25, 45, precision=9),
                                bodo.types.Time(1, 30, 0, precision=9),
                                bodo.types.Time(22, 59, 0, precision=9),
                                bodo.types.Time(0, 0, 0, precision=9),
                                bodo.types.Time(
                                    23, 59, 59, nanosecond=999999999, precision=9
                                ),
                                bodo.types.Time(23, 58, 5, precision=9),
                                None,
                                bodo.types.Time(
                                    hour=4,
                                    minute=53,
                                    second=12,
                                    nanosecond=186044416,
                                    precision=9,
                                ),
                                bodo.types.Time(
                                    hour=20,
                                    minute=8,
                                    second=18,
                                    nanosecond=926258176,
                                    precision=9,
                                ),
                                bodo.types.Time(
                                    10, 40, 0, nanosecond=1234, precision=9
                                ),
                            ]
                        )
                    }
                ),
                "H, M, S, NS",
            ),
            id="four_args_with_floats",
        ),
    ]
)
def time_from_parts_data(request):
    ctx, answer, args = request.param
    ctx["TABLE1"]["B"] = [True] * len(ctx["TABLE1"])
    return ctx, answer, args


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case", marks=pytest.mark.slow),
        pytest.param(True, id="with_case"),
    ],
)
def test_time_from_parts(
    time_from_parts_fn, use_case, time_from_parts_data, memory_leak_check
):
    ctx, answer, args = time_from_parts_data
    if use_case:
        query = (
            f"SELECT CASE WHEN B THEN {time_from_parts_fn}({args}) END AS A FROM table1"
        )
    else:
        query = f"SELECT {time_from_parts_fn}({args}) AS A FROM table1"
    check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_typing_issues=False,
        check_dtype=False,
    )
