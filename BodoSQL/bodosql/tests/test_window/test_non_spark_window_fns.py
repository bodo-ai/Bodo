import datetime

import numpy as np
import pandas as pd
import pytest

from bodo import Time
from bodo.tests.utils import nullable_float_arr_maker, pytest_slow_unless_window
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


@pytest.mark.tz_aware
def test_conditional_event_pure(memory_leak_check):
    """
    Tests CONDITIONAL_TRUE_EVENT and CONDITIONAL_CHANGE_EVENT in isolation
    (thus ensuring that groupby.window can be used)
    """
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P": ["A"] * 10,
                "O": pd.Series(list(range(10))),
                "I64_NULLABLE": pd.Series(
                    [100, None, 200, None, 100, None, None, None, -100, None],
                    dtype=pd.Int64Dtype(),
                ),
                "I64_NUMPY": np.array(
                    [1024, 1024, -1, -1, 0, 0, 0, 0, 8589934591, 8589934591],
                    dtype=np.int64,
                ),
                "BOOL_NULLABLE": pd.Series(
                    [True, False, True, True, True, None, False, True, False, False],
                    dtype=pd.BooleanDtype(),
                ),
                "BOOL_NUMPY": np.array(
                    [True, False, True, True, True, True, False, True, False, False],
                ),
                "BINARY_COL": pd.Series(
                    [b"", None, b"", None, b"", b"0", b"", b"1", b"1", b"0"]
                ),
                "STRING_COL": pd.Series(
                    ["A", "A", "B", "A", "A", "B", "C", "B", "A", "A"]
                ),
                "TIMESTAMP_NAIVE": pd.Series(
                    [
                        pd.Timestamp(f"201{y}-01-01")
                        for y in [0, 1, 0, 0, 1, 8, 0, 8, 8, 1]
                    ]
                ),
                "TIMESTAMP_LTZ": pd.Series(
                    [
                        pd.Timestamp(f"201{y}-01-01", tz="US/PACIFIC")
                        for y in [0, 1, 0, 0, 1, 8, 0, 8, 8, 1]
                    ]
                ),
                "DATE": pd.Series(
                    [
                        datetime.date.fromordinal(730119 + i)
                        for i in [0] * 2 + [10000] * 7 + [3000]
                    ]
                ),
            }
        ),
    }
    window = " partition by P order by O"
    selects = []
    for col in ctx["TABLE1"].columns[2:]:
        selects.append(f"CONDITIONAL_CHANGE_EVENT({col}) OVER ({window})")
    selects.append(f"CONDITIONAL_TRUE_EVENT(BOOL_NULLABLE) OVER ({window})")
    selects.append(f"CONDITIONAL_TRUE_EVENT(BOOL_NUMPY) OVER ({window})")
    selects.append(f"CONDITIONAL_TRUE_EVENT(STRING_COL = 'A') OVER ({window})")
    query = f"SELECT O, {', '.join(selects)} FROM table1"

    answer = pd.DataFrame(
        {
            "O": list(range(10)),
            "CHANGE_1": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3],
            "CHANGE_2": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3],
            "CHANGE_3": [0, 1, 2, 2, 2, 2, 3, 4, 5, 5],
            "CHANGE_4": [0, 1, 2, 2, 2, 2, 3, 4, 5, 5],
            "CHANGE_5": [0, 0, 0, 0, 0, 1, 2, 3, 3, 4],
            "CHANGE_6": [0, 0, 1, 2, 2, 3, 4, 5, 6, 6],
            "CHANGE_7": [0, 1, 2, 2, 3, 4, 5, 6, 6, 7],
            "CHANGE_8": [0, 1, 2, 2, 3, 4, 5, 6, 6, 7],
            "CHANGE_9": [0, 0, 1, 1, 1, 1, 1, 1, 1, 2],
            "TRUE_1": [1, 1, 2, 3, 4, 4, 4, 5, 5, 5],
            "TRUE_2": [1, 1, 2, 3, 4, 5, 5, 6, 6, 6],
            "TRUE_3": [1, 2, 2, 3, 4, 4, 4, 4, 5, 6],
        }
    )

    pandas_code = check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        only_jit_1DVar=True,
        return_codegen=True,
    )["pandas_code"]

    # All of the functions are CONDITIONAL_CHANGE_EVENT/CONDITIONAL_TRUE_EVENT
    # which support groupby.window, so there should be no groupby.apply calls
    count_window_applies(pandas_code, 0, ["CONDITIONAL_CHANGE_EVENT"])


def test_conditional_event_mixed(memory_leak_check):
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P": ["A"] * 10,
                "O": pd.Series(list(range(10))),
                "INT_COL": pd.Series(
                    [100, None, 200, None, 100, None, None, None, -100, None],
                    dtype=pd.Int64Dtype(),
                ),
                "BINARY_COL": pd.Series(
                    [b"", None, b"", None, b"", b"0", b"", b"1", b"1", b"0"]
                ),
                "STRING_COL": pd.Series(
                    ["A", "A", "B", "A", "A", "B", "C", "B", "A", "A"]
                ),
                "BOOL_COL": pd.Series(
                    [None, False, True, True, True, None, None, True, False, False],
                    dtype=pd.BooleanDtype(),
                ),
            }
        ),
    }
    selects = []
    window = "partition by P order by O"
    selects.append(f"CONDITIONAL_CHANGE_EVENT(INT_COL) OVER ({window})")
    selects.append(f"CONDITIONAL_CHANGE_EVENT(BINARY_COL) OVER ({window})")
    selects.append(f"CONDITIONAL_CHANGE_EVENT(STRING_COL) OVER ({window})")
    selects.append(f"CONDITIONAL_TRUE_EVENT(BOOL_COL) OVER ({window})")
    query = (
        f"SELECT O, {', '.join(selects)}, MODE(STRING_COL) OVER ({window}) FROM table1"
    )

    answer = pd.DataFrame(
        {
            "O": list(range(10)),
            "CHANGE_INT": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3],
            "CHANGE_BIN": [0, 0, 0, 0, 0, 1, 2, 3, 3, 4],
            "CHANGE_STR": [0, 0, 1, 2, 2, 3, 4, 5, 6, 6],
            "TRUE_BOOL": [0, 0, 1, 2, 3, 3, 3, 4, 4, 4],
            "MODE_STR": ["A"] * 10,
        }
    )

    pandas_code = check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        only_jit_1DVar=True,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. There are multiple window
    # functions being used with the same partition/orderby, but
    # only some of them can use groupby.window so groupby.window
    # will not be used here
    count_window_applies(
        pandas_code, 1, ["CONDITIONAL_CHANGE_EVENT", "CONDITIONAL_TRUE_EVENT", "MODE"]
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "data_col, partition_col, window_frame, answer",
    [
        pytest.param(
            pd.Series([1, 2, 3, 4, 5, 6], dtype=pd.Int32Dtype()),
            pd.Series(["A"] * 6),
            "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
            pd.Series([1, 1.5, 2, 2.5, 3, 3.5]),
            id="int32-single_partition-prefix",
        ),
        pytest.param(
            pd.Series(
                [10, 20, None, 30, 40, None, 50, 60, None], dtype=pd.UInt8Dtype()
            ),
            pd.Series(["A", "B", "C"] * 3),
            "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING",
            pd.Series([30, 40, None, 40, 50, None, 50, 60, None]),
            id="uint8-multiple_partitions-suffix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([0.275, 0.488, 0.04, 0.06, 0.92, 1.0, 0.1, None, 0.0, None]),
            pd.Series(["A"] * 10),
            "ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING",
            pd.Series([0.275, 0.1675, 0.275, 0.488, 0.1, 0.51, 0.51, 0.1, 0.05, 0.0]),
            id="float64-single_partition-rolling_5",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_median(data_col, partition_col, window_frame, answer):
    """Tests median (needs to be handled seperately due to PySpark limitations
    on calculating medians)"""
    assert len(data_col) == len(partition_col)
    query = f"SELECT A, B, C, MEDIAN(A) OVER (PARTITION BY B ORDER BY C {window_frame}) FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {"A": data_col, "B": partition_col, "C": list(range(len(data_col)))}
        )
    }
    expected_output = pd.DataFrame(
        {
            "A": data_col,
            "B": partition_col,
            "C": list(range(len(data_col))),
            "D": answer,
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "data_info, dtype",
    [
        pytest.param(
            [
                (0, [None, None, None]),
                (0, [10, -15, 5]),
                (10, [1, 2, None, 3, 4]),
                (16, [16, None]),
                (15, [1, 0, -4, 9, -16, 25]),
            ],
            pd.Int32Dtype(),
            id="int32",
        ),
        pytest.param(
            [
                (0, [None, None, None]),
                (0, [3.6, -3.6]),
                (2.5, [5.5, None, -3]),
                (-11.0, [-1.0, 2.0, -4.0, 8.0, -16.0]),
            ],
            np.float64,
            id="float64",
        ),
    ],
)
def test_ratio_to_report(data_info, dtype, memory_leak_check):
    query = "SELECT R, P, D, RATIO_TO_REPORT(D) OVER (PARTITION BY P) FROM table1"
    partitions = []
    data = []
    answers = []
    for i, (total, elems) in enumerate(data_info):
        for elem in elems:
            partitions.append(i)
            data.append(elem)
            if total == 0 or elem is None:
                answers.append(None)
            else:
                answers.append(elem / total)
    data = pd.Series(data, dtype=dtype)
    rows = np.arange(len(data))
    df = pd.DataFrame({"R": rows, "P": partitions, "D": data})
    answer = df.copy()
    answer["A"] = answers
    # Shuffle the rows of the input dataframe
    ordering = np.random.default_rng(42).permutation(np.arange(len(data)))
    ctx = {"TABLE1": df.iloc[ordering]}
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_approx_percentile(memory_leak_check):
    """Tests APPROX_PERCENTILE as a window function"""
    scale = 11
    partitions = [1, 1, 2, 1, 1, 2, 3, 2, 1, 1, 2, 3, 4, 3, 2, 1]
    df = pd.DataFrame(
        {
            "P": partitions * scale,
            "O": list(range(len(partitions) * scale)),
            "I": [
                None
                if i % len(partitions) in (5, 10, 12)
                else 10 * (2 ** (i / 10 + partitions[i % len(partitions)]))
                for i in range(len(partitions) * scale)
            ],
        }
    )

    selects = []
    for i, q in enumerate([0.5, 0.01, 0.9]):
        selects.append(f"APPROX_PERCENTILE(I, {q}) OVER (PARTITION BY P) as R{i}")

    query = f"SELECT P, {', '.join(selects)} FROM table1"

    answer = pd.DataFrame(
        {
            "P": ([1] * 7 + [2] * 5 + [3] * 3 + [4]) * scale,
            "R0": ([6755.880503] * 7 + [16634.929077] * 5 + [43899.841025] * 3 + [None])
            * scale,
            "R1": ([21.090957] * 7 + [52.038257] * 5 + [137.329783] * 3 + [None])
            * scale,
            "R2": ([9.732837e05] * 7 + [2.106636e06] * 5 + [4.148289e06] * 3 + [None])
            * scale,
        }
    )

    pandas_code = check_query(
        query,
        {"table1": df},
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        rtol=0.3,
    )


# In cases with multiple correct answers, the answer returned by our mode implementation
# is the value that was encountered first in the overall array
# due to the semantics of how Python iterates over hashmaps
@pytest.mark.slow
@pytest.mark.parametrize(
    "data_col, bounds, answer",
    [
        pytest.param(
            pd.Series([0, -1, -1, 2, 2, 2, 3, 3, 3, 3], dtype=pd.Int32Dtype()),
            ("UNBOUNDED PRECEDING", "CURRENT ROW"),
            pd.DataFrame(
                {0: pd.Series([0, 0, -1, -1, -1, 2, 2, 3, 3, 3], dtype=pd.Int32Dtype())}
            ),
            id="int32-prefix",
        ),
        pytest.param(
            pd.Series(
                [100, 1, 1, 100, None, None, 2, 2, 100, 2],
                dtype=pd.UInt8Dtype(),
            ),
            ("5 PRECEDING", "1 PRECEDING"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [None, 100, 100, 1, 100, None, None, 2, 2, 2],
                        dtype=pd.UInt8Dtype(),
                    )
                }
            ),
            id="uint8-before3",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "ABC",
                    "DEFGHIJKLMNOPQRSTUVWXYZDEFGHIJKLMNOPQRSTUVWXYZ",
                    "¡™£¢∞§¶•ªº⁄€‹›ﬁﬂ‡°·‚—±",
                    "ABC",
                    "ABC",
                    None,
                    "1!¡@2#™$3%£^4&¢*5(∞)6-§=7_¶+8[•]9{ª}0|\º1⁄2€3‹4›5ﬁ6ﬂ7‡8°9·0‚1—2±3",
                    "1!¡@2#™$3%£^4&¢*5(∞)6-§=7_¶+8[•]9{ª}0|\º1⁄2€3‹4›5ﬁ6ﬂ7‡8°9·0‚1—2±3",
                    None,
                    "ABC",
                ]
            ),
            None,
            pd.DataFrame(
                {
                    0: pd.Series(
                        ["ABC"] * 5
                        + [
                            "1!¡@2#™$3%£^4&¢*5(∞)6-§=7_¶+8[•]9{ª}0|\º1⁄2€3‹4›5ﬁ6ﬂ7‡8°9·0‚1—2±3"
                        ]
                        * 5
                    )
                }
            ),
            id="string-noframe",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([b"X", b"Y", b"X", b"Y", b"X", b"X", b"Y", b"Y", None, None]),
            ("1 PRECEDING", "1 FOLLOWING"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [b"X", b"X", b"Y", b"X", b"X", b"X", b"Y", b"Y", b"Y", None]
                    )
                }
            ),
            id="binary-rolling3",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [True, True, None, None, None, True, False, False, False, True],
                dtype=pd.BooleanDtype(),
            ),
            ("1 FOLLOWING", "UNBOUNDED FOLLOWING"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [True, None, None, None, None, False, False, True, True, None],
                        dtype=pd.BooleanDtype(),
                    )
                }
            ),
            id="boolean-exclusive_suffix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0]),
            ("UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
            pd.DataFrame({0: [1.0] * 5 + [3.0] * 5}),
            id="float-entire",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([pd.Timestamp(f"201{y}-7-4") for y in "0110143224"]),
            ("UNBOUNDED PRECEDING", "1 FOLLOWING"),
            pd.DataFrame(
                {0: pd.Series([pd.Timestamp(f"201{y}-7-4") for y in "0101144244"])}
            ),
            id="timestamp-overinclusive_prefix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [pd.Timestamp(f"201{y}-7-4", tz="US/Pacific") for y in "0110143224"]
            ),
            ("UNBOUNDED PRECEDING", "1 FOLLOWING"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [
                            pd.Timestamp(f"201{y}-7-4", tz="US/Pacific")
                            for y in "0101144244"
                        ]
                    )
                }
            ),
            id="tz-aware-overinclusive_prefix",
            marks=pytest.mark.tz_aware,
        ),
        pytest.param(
            pd.Series(
                [
                    datetime.date(2024, 1, 1),
                    datetime.date(2024, 1, 1),
                    None,
                    datetime.date(2022, 7, 4),
                    None,
                    datetime.date(1999, 12, 31),
                    datetime.date(1999, 12, 31),
                    datetime.date(1999, 12, 31),
                    datetime.date(2022, 7, 4),
                    datetime.date(2022, 7, 4),
                ]
            ),
            ("CURRENT ROW", "UNBOUNDED FOLLOWING"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [
                            datetime.date(2024, 1, 1),
                            datetime.date(2024, 1, 1),
                            datetime.date(2022, 7, 4),
                            datetime.date(2022, 7, 4),
                            None,
                            datetime.date(1999, 12, 31),
                            datetime.date(1999, 12, 31),
                            datetime.date(2022, 7, 4),
                            datetime.date(2022, 7, 4),
                            datetime.date(2022, 7, 4),
                        ]
                    )
                }
            ),
            id="date-suffix",
        ),
        pytest.param(
            pd.Series(
                [
                    Time(6, 10, 0, microsecond=500, precision=9),
                    Time(7, 45, 59, precision=9),
                    Time(7, 45, 59, precision=9),
                    Time(6, 10, 0, microsecond=500, precision=9),
                    Time(7, 45, 59, nanosecond=1, precision=9),
                    None,
                    Time(12, 30, 0, precision=9),
                    Time(10, 0, 1, precision=9),
                    Time(10, 0, 1, precision=9),
                    Time(12, 30, 0, precision=9),
                ]
            ),
            ("UNBOUNDED PRECEDING", "CURRENT ROW"),
            pd.DataFrame(
                {
                    0: pd.Series(
                        [
                            Time(6, 10, 0, microsecond=500, precision=9),
                            Time(6, 10, 0, microsecond=500, precision=9),
                            Time(7, 45, 59, precision=9),
                            Time(6, 10, 0, microsecond=500, precision=9),
                            Time(6, 10, 0, microsecond=500, precision=9),
                            None,
                            Time(12, 30, 0, precision=9),
                            Time(12, 30, 0, precision=9),
                            Time(10, 0, 1, precision=9),
                            Time(12, 30, 0, precision=9),
                        ]
                    )
                }
            ),
            id="time-prefix",
        ),
    ],
)
def test_mode(data_col, bounds, answer, memory_leak_check):
    """Tests the mode function with hardcoded answers"""
    if bounds == None:
        query = "select MODE(A) OVER (PARTITION BY B) from table1"
    else:
        query = f"select MODE(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN {bounds[0]} AND {bounds[1]}) from table1"

    assert len(data_col) == 10
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": data_col,
                "B": ["A"] * 5 + ["B"] * 5,
                "C": list(range(10)),
            }
        )
    }

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=answer,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_variance_stddev_nan(memory_leak_check):
    """Tests the 4 major var/std functions on data with both NULL and NaN"""
    params = [
        ("VAR_POP", "UNBOUNDED PRECEDING", "CURRENT ROW"),
        ("VAR_SAMP", "3 PRECEDING", "1 PRECEDING"),
        ("STDDEV_POP", "CURRENT ROW", "UNBOUNDED FOLLOWING"),
        ("STDDEV_SAMP", "1 PRECEDING", "1 FOLLOWING"),
    ]
    calculations = []
    for func, frame_start, frame_end in params:
        calculations.append(
            f"{func}(D) OVER (PARTITION BY P ORDER BY O ROWS BETWEEN {frame_start} AND {frame_end})"
        )
    query = f"SELECT O, {', '.join(calculations)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "P": [1] * 10,
                "O": range(10),
                "D": nullable_float_arr_maker(list(range(10)), [1, 9], [5, 6]),
            }
        )
    }
    expected_output = pd.DataFrame(
        {
            0: range(10),
            1: nullable_float_arr_maker(
                [0.0, 0.0, 1.0, 14 / 9, 35 / 16] + [0.0] * 5, [-1], [5, 6, 7, 8, 9]
            ),
            2: nullable_float_arr_maker(
                [0.0] * 3 + [2.0, 0.5, 1.0] + [0.0] * 4, [0, 1, 2], [6, 7, 8, 9]
            ),
            3: nullable_float_arr_maker(
                [0.0] * 7 + [0.5, 0.0, 0.0], [9], [0, 1, 2, 3, 4, 5, 6]
            ),
            4: nullable_float_arr_maker(
                [0.0, 2**0.5, 0.5**0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5**0.5, 0.0],
                [0, 9],
                [4, 5, 6, 7],
            ),
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_kurtosis_skew(memory_leak_check):
    """Tests the kurtosis and skew functions"""
    window_calls = [
        ("SKEW", "UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
        ("KURTOSIS", "UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
    ]
    selects = []
    for func, lower, upper in window_calls:
        selects.append(
            f"{func}(A) OVER (PARTITION BY P ORDER BY O ROWS BETWEEN {lower} AND {upper})"
        )
    query = f"SELECT {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [1, None, 2, None]
                    + [None, None, None, None]
                    + [10, 20, 40, None]
                    + [100, 175, 180, 170],
                    dtype=pd.Int32Dtype(),
                ),
                "P": [2.718281828] * 4 + [1.0] * 4 + [None] * 4 + [0.0] * 4,
                "O": list(range(16)),
            }
        )
    }
    answer = pd.DataFrame(
        {
            0: nullable_float_arr_maker(
                [0.0] * 8 + [0.9352195295828252] * 4 + [-1.9300313123985544] * 4,
                [0, 1, 2, 3, 4, 5, 6, 7],
                [-1],
            ),
            1: nullable_float_arr_maker(
                [0.0] * 12 + [3.7681402991281665] * 4,
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [-1],
            ),
        }
    )

    pandas_code = check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=answer,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly.
    count_window_applies(pandas_code, 1, ["KURTOSIS", "SKEW"])


def test_bool_agg(memory_leak_check):
    """Tests the boolean aggregation functions"""
    window_calls = [
        ("BOOLOR_AGG", "A", "2 PRECEDING", "CURRENT ROW"),
        ("BOOLAND_AGG", "B", "2 PRECEDING", "CURRENT ROW"),
        ("BOOLXOR_AGG", "A", "2 PRECEDING", "CURRENT ROW"),
        ("BOOLOR_AGG", "B", "CURRENT ROW", "1 FOLLOWING"),
        ("BOOLAND_AGG", "A", "CURRENT ROW", "1 FOLLOWING"),
        ("BOOLXOR_AGG", "B", "CURRENT ROW", "1 FOLLOWING"),
    ]
    selects = []
    for func, col, lower, upper in window_calls:
        selects.append(
            f"{func}({col}) OVER (PARTITION BY P ORDER BY O ROWS BETWEEN {lower} AND {upper})"
        )
    query = f"SELECT {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [None, 1, 0, None, 2, 3, 0, None], dtype=pd.Int32Dtype()
                ),
                "B": pd.Series(
                    [None, True, False, None, True, True, False, None],
                    dtype=pd.BooleanDtype(),
                ),
                "P": [pd.Timestamp("2023-1-1")] * 8,
                "O": list(range(8)),
            }
        )
    }
    answer = pd.DataFrame(
        {
            0: pd.Series(
                [None, True, True, True, True, True, True, True],
                dtype=pd.BooleanDtype(),
            ),
            1: pd.Series(
                [None, True, False, False, False, True, False, False],
                dtype=pd.BooleanDtype(),
            ),
            2: pd.Series(
                [None, True, True, True, True, False, False, True],
                dtype=pd.BooleanDtype(),
            ),
            3: pd.Series(
                [True, True, False, True, True, True, False, None],
                dtype=pd.BooleanDtype(),
            ),
            4: pd.Series(
                [True, False, False, True, True, False, False, None],
                dtype=pd.BooleanDtype(),
            ),
            5: pd.Series(
                [True, True, False, True, False, True, False, None],
                dtype=pd.BooleanDtype(),
            ),
        }
    )

    pandas_code = check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=answer,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly.
    count_window_applies(pandas_code, 1, ["BOOLOR_AGG", "BOOLAND_AGG", "BOOLXOR_AGG"])


@pytest.mark.parametrize(
    "data, dtype",
    [
        pytest.param(
            [42, 100, 60, 5, 15, 70, 3, 213, 6, None],
            pd.Int32Dtype(),
            id="int32",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            [41.5, 100.4, 60.0, 5.0, 15.21, 69.53, 3.2, 213.1, 5.8, None],
            pd.Float32Dtype(),
            id="floats",
        ),
        pytest.param(
            ["41.5", "100", "60", "5", "15", "69.53", "3.2", "213.1", "5.8", None],
            pd.StringDtype(),
            id="strings",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_bit_agg(data, dtype, memory_leak_check):
    """Tests the BITOR_AGG, BITAND_AGG, and BITXOR_AGG window functions.
        These operations perform bitwise or, and, and xor operations respectively,
        aggregating the result per window.

    Args:
        data (pd.Series): Input column
        memory_leak_check (): Fixture, see `conftest.py`.
    """
    bit_agg_funcs = ["BITOR_AGG", "BITAND_AGG", "BITXOR_AGG"]

    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3, 4], dtype=pd.Int32Dtype()),
                "B": pd.Series(data, dtype=dtype),
            }
        )
    }

    selects = []
    for func in bit_agg_funcs:
        selects.append(f"{func}(B) OVER (PARTITION BY A)")

    query = f"SELECT {', '.join(selects)} FROM table1"

    expected = pd.DataFrame(
        {
            0: pd.Series(
                [126, 126, 126, 79, 79, 79, 215, 215, 215, None],
                dtype=pd.Int32Dtype(),
            ),
            1: pd.Series(
                [32, 32, 32, 4, 4, 4, 0, 0, 0, None],
                dtype=pd.Int32Dtype(),
            ),
            2: pd.Series(
                [114, 114, 114, 76, 76, 76, 208, 208, 208, None],
                dtype=pd.Int32Dtype(),
            ),
        }
    )

    pandas_code = check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=expected,
        return_codegen=True,
    )["pandas_code"]

    # Verify that fusion is working correctly.
    count_window_applies(pandas_code, 1, bit_agg_funcs)
