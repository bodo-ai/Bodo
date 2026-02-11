import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bodo.tests.utils import (
    nullable_float_arr_maker,
    pytest_mark_multi_rank_nightly,
    pytest_slow_unless_window,
    temp_config_override,
)
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


@pytest.mark.tz_aware
@pytest_mark_multi_rank_nightly
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
                    ],
                    dtype="datetime64[ns]",
                ),
                "TIMESTAMP_LTZ": pd.Series(
                    [
                        pd.Timestamp(f"201{y}-01-01", tz="US/Pacific")
                        for y in [0, 1, 0, 0, 1, 8, 0, 8, 8, 1]
                    ],
                    dtype="datetime64[ns, US/Pacific]",
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


@pytest_mark_multi_rank_nightly
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
    "data_col, partition_col, answer",
    [
        pytest.param(
            pd.array([1, 2, 3, 4, 5, 6], dtype=pd.Int32Dtype()),
            pd.array(["A"] * 6),
            pd.array([3.5] * 6),
            id="int32",
        ),
        pytest.param(
            nullable_float_arr_maker(
                [0.275, 0.488, 0.06, 0.92, 1.0, 0.1, 0.0, 0.0, 0.0], [6, 8], [-1]
            ),
            pd.array(["A"] * 3 + ["B"] * 5 + ["C"]),
            nullable_float_arr_maker([0.275] * 3 + [0.51] * 5 + [0.0], [8], [-1]),
            id="float64",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_median(data_col, partition_col, answer):
    """Tests median (needs to be handled seperately due to PySpark limitations
    on calculating medians)"""
    assert len(data_col) == len(partition_col)
    query = "SELECT C, MEDIAN(A) OVER (PARTITION BY B) FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {"A": data_col, "B": partition_col, "C": list(range(len(data_col)))}
        )
    }
    expected_output = pd.DataFrame(
        {
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


def test_partitionless_ratio_to_report(memory_leak_check):
    query = "SELECT IDX, RATIO_TO_REPORT(S) OVER () AS WIN FROM TABLE1"
    n_rows = 10_000
    df = pd.DataFrame(
        {
            "IDX": range(n_rows),
            "S": pd.array(
                [
                    None if i % 7 == i % 6 else np.int64((np.tan(i) * 3) ** 2)
                    for i in range(n_rows)
                ],
                dtype=pd.Int64Dtype(),
            ),
        }
    )
    answer = pd.DataFrame({"IDX": df["IDX"], "WIN": df["S"] / df["S"].sum()})
    ordering = np.random.default_rng(42).permutation(np.arange(len(df)))
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

    check_query(
        query,
        {"TABLE1": df},
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
    "data_col, answer",
    [
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
            id="string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 1.0, 2.0, 3.0]),
            pd.DataFrame({0: [1.0] * 5 + [3.0] * 5}),
            id="float",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_mode(data_col, answer, memory_leak_check):
    """Tests the mode function with hardcoded answers"""
    query = "select MODE(A) OVER (PARTITION BY B) from table1"
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
    selects = []
    for func in ["SKEW", "KURTOSIS"]:
        selects.append(f"{func}(A) OVER (PARTITION BY P)")
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
        ("BOOLOR_AGG", "A"),
        ("BOOLOR_AGG", "B"),
        ("BOOLAND_AGG", "A"),
        ("BOOLAND_AGG", "B"),
        ("BOOLXOR_AGG", "A"),
        (
            "BOOLXOR_AGG",
            "B",
        ),
    ]
    selects = []
    for func, col in window_calls:
        selects.append(f"{func}({col}) OVER (PARTITION BY P)")
    query = f"SELECT O, {', '.join(selects)} FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [None, 1, 0, None, 2, 3, 0, None], dtype=pd.Int32Dtype()
                ),
                "B": pd.Series(
                    [None, False, True, None, True, True, False, None],
                    dtype=pd.BooleanDtype(),
                ),
                "P": pd.Series(
                    [pd.Timestamp("2023-3-14"), pd.Timestamp("2023-7-4")]
                    + [pd.Timestamp("2023-1-1")] * 6,
                    dtype="datetime64[ns]",
                ),
                "O": range(8),
            }
        )
    }
    answer = pd.DataFrame(
        {
            0: range(8),
            1: pd.Series(
                [None] + [True] * 7,
                dtype=pd.BooleanDtype(),
            ),
            2: pd.Series(
                [None, False] + [True] * 6,
                dtype=pd.BooleanDtype(),
            ),
            3: pd.Series(
                [None, True] + [False] * 6,
                dtype=pd.BooleanDtype(),
            ),
            4: pd.Series(
                [None] + [False] * 7,
                dtype=pd.BooleanDtype(),
            ),
            5: pd.Series(
                [None, True] + [False] * 6,
                dtype=pd.BooleanDtype(),
            ),
            6: pd.Series(
                [None] + [False] * 7,
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

    query = f"SELECT A, {', '.join(selects)} FROM table1"

    expected = pd.DataFrame(
        {
            0: ctx["TABLE1"]["A"],
            1: pd.Series(
                [126, 126, 126, 79, 79, 79, 215, 215, 215, None],
                dtype=pd.Int32Dtype(),
            ),
            2: pd.Series(
                [32, 32, 32, 4, 4, 4, 0, 0, 0, None],
                dtype=pd.Int32Dtype(),
            ),
            3: pd.Series(
                [114, 114, 114, 76, 76, 76, 208, 208, 208, None],
                dtype=pd.Int32Dtype(),
            ),
        }
    )
    with temp_config_override("bodo_use_decimal", True):
        check_query(
            query,
            ctx,
            None,
            check_dtype=False,
            check_names=False,
            expected_output=expected,
        )


@pytest.mark.parametrize(
    "data, value_dtype",
    [
        pytest.param(
            pd.array([1, 2, None, 4, 5, None, 7, 8, None, None], dtype=pd.Int64Dtype()),
            pa.int64(),
            id="integer",
        ),
        pytest.param(
            pd.array(
                ["Alpha", None, "Gamma", "Delta", None, "", "Theta", "Iota", None, None]
            ),
            pa.large_string(),
            id="string",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_object_agg(data, value_dtype, memory_leak_check):
    """Tests the OBJECT_AGG window function."""
    query = "SELECT P, O, OBJECT_AGG(K, V) OVER (PARTITION BY P) FROM table1"

    df = pd.DataFrame(
        {
            "P": [1, 2, 1, 2, 1, 3, 1, 3, 1, 4],
            "O": list(range(10)),
            "K": list("ABC") + [None] + list("EFGHIJ"),
            "V": data,
        }
    )

    json_data = []
    for i in range(len(df)):
        partition = df["P"].iloc[i]
        keys = df["K"][df["P"] == partition]
        values = df["V"][df["P"] == partition]
        keep = pd.notna(keys) & pd.notna(values)
        res = {}
        for j in range(len(keys)):
            if keep.iloc[j]:
                res[keys.iloc[j]] = values.iloc[j]
        json_data.append(res)

    expected = pd.DataFrame(
        {
            "P": df["P"],
            "O": df["O"],
            "J": pd.array(
                json_data, dtype=pd.ArrowDtype(pa.map_(pa.string(), value_dtype))
            ),
        }
    )

    check_query(
        query,
        {"TABLE1": df},
        None,
        check_dtype=False,
        check_names=False,
        convert_columns_to_pandas=True,
        expected_output=expected,
        use_dict_encoded_strings=False,
    )
