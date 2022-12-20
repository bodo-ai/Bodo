import numpy as np
import pandas as pd
import pytest
from bodosql.tests.test_window.window_common import count_window_applies
from bodosql.tests.utils import check_query


@pytest.mark.tz_aware
@pytest.mark.parametrize(
    "partition_col, answer",
    [
        pytest.param(
            pd.Series(["A"] * 10),
            pd.DataFrame(
                {
                    "O": list(range(10)),
                    "U8": [0, 0, 0, 0, 1, 2, 2, 3, 3, 3],
                    "I16": [0, 1, 2, 3, 3, 4, 4, 4, 5, 6],
                    "U32": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "I64": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3],
                    "BO": [0, 1, 2, 2, 2, 2, 3, 4, 5, 5],
                    "BI": [0, 0, 0, 0, 0, 1, 2, 3, 3, 4],
                    "S": [0, 0, 1, 2, 2, 3, 4, 5, 6, 6],
                    "TS": [0, 1, 2, 2, 3, 4, 5, 6, 6, 7],
                    "TZ": [0, 1, 2, 2, 3, 4, 5, 6, 6, 7],
                }
            ),
            id="single_partiton",
        ),
        pytest.param(
            pd.Series(["A"] * 5 + ["B"] * 5),
            pd.DataFrame(
                {
                    "O": list(range(10)),
                    "U8": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1],
                    "I16": [0, 1, 2, 3, 3, 0, 0, 0, 1, 2],
                    "U32": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                    "I64": [0, 0, 1, 1, 2, 0, 0, 0, 0, 0],
                    "BO": [0, 1, 2, 2, 2, 0, 0, 1, 2, 2],
                    "BI": [0, 0, 0, 0, 0, 0, 1, 2, 2, 3],
                    "S": [0, 0, 1, 2, 2, 0, 1, 2, 3, 3],
                    "TS": [0, 1, 2, 2, 3, 0, 1, 2, 2, 3],
                    "TZ": [0, 1, 2, 2, 3, 0, 1, 2, 2, 3],
                }
            ),
            id="two_partitions",
        ),
    ],
)
def test_conditional_change_event(partition_col, answer, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "P": partition_col,
                "O": pd.Series(list(range(10))),
                "U8": pd.Series(
                    [None, 1, None, 1, 2, 3, None, 4, None, None], dtype=pd.UInt8Dtype()
                ),
                "I16": pd.Series(
                    [4, 9, 16, 4, None, 9, 9, None, -25, 36], dtype=pd.Int16Dtype()
                ),
                "U32": pd.Series(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=pd.UInt32Dtype()
                ),
                "I64": pd.Series(
                    [100, None, 200, None, 100, None, None, None, -100, None],
                    dtype=pd.Int64Dtype(),
                ),
                "BO": pd.Series(
                    [True, False, True, True, True, None, False, True, False, False],
                    dtype=pd.BooleanDtype(),
                ),
                "BI": pd.Series(
                    [b"", None, b"", None, b"", b"0", b"", b"1", b"1", b"0"]
                ),
                "S": pd.Series(["A", "A", "B", "A", "A", "B", "C", "B", "A", "A"]),
                "TS": pd.Series(
                    [
                        pd.Timestamp(f"201{y}-01-01")
                        for y in [0, 1, 0, 0, 1, 8, 0, 8, 8, 1]
                    ]
                ),
                "TZ": pd.Series(
                    [
                        pd.Timestamp(f"201{y}-01-01", tz="US/PACIFIC")
                        for y in [0, 1, 0, 0, 1, 8, 0, 8, 8, 1]
                    ]
                ),
            }
        ),
    }
    selects = []
    for col in ["U8", "I16", "U32", "I64", "BO", "BI", "S", "TS", "TZ"]:
        selects.append(
            f"CONDITIONAL_CHANGE_EVENT({col}) OVER (partition by P order by O)"
        )
    query = f"SELECT O, {', '.join(selects)} FROM table1"

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

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, 1, ["CONDITIONAL_CHANGE_EVENT"])


@pytest.mark.parametrize(
    "partition_col, expressions, window, answer",
    [
        pytest.param(
            pd.Series(["A"] * 10),
            ["A = B", "A < B", "A * B > 0"],
            "partition by P order by O",
            pd.DataFrame(
                {
                    "O": list(range(10)),
                    "A = B": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                    "A < B": [0, 1, 2, 2, 3, 3, 3, 4, 5, 6],
                    "A * B > 0": [0, 0, 1, 1, 1, 1, 2, 3, 3, 4],
                }
            ),
            id="single_partiton",
        ),
        pytest.param(
            pd.Series(["A"] * 5 + ["B"] * 5),
            ["A = B", "A < B", "A * B > 0"],
            "partition by P order by O",
            pd.DataFrame(
                {
                    "O": list(range(10)),
                    "A = B": [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
                    "A < B": [0, 1, 2, 2, 3, 0, 0, 1, 2, 3],
                    "A * B > 0": [0, 0, 1, 1, 1, 0, 1, 2, 2, 3],
                }
            ),
            id="two_partition_vals",
        ),
        pytest.param(
            pd.Series([0, 1, 0, None, 0, 1, 0, 0, None, 1], dtype=pd.Int32Dtype()),
            ["COALESCE(P, 0) = 0", "A = B % 2", "TRUE"],
            "partition by A, B % 2 order by O",
            pd.DataFrame(
                {
                    "O": list(range(10)),
                    "COALESCE(P, 0) = 0": [1, 0, 1, 1, 2, 1, 1, 2, 1, 2],
                    "A = B % 2": [1, 0, 0, 0, 2, 0, 1, 0, 0, 0],
                    "TRUE": [1, 1, 1, 1, 2, 2, 1, 3, 2, 4],
                }
            ),
            id="two_partition_keys",
        ),
    ],
)
def test_conditional_true_event(
    partition_col, expressions, window, answer, memory_leak_check
):
    ctx = {
        "table1": pd.DataFrame(
            {
                "P": partition_col,
                "O": list(range(10)),
                "A": pd.Series([0, 0, 1, 0, 0, 1, 1, 1, 0, 1], dtype=pd.Int32Dtype()),
                "B": pd.Series(
                    [0, 1, 2, None, 4, 0, 1, 2, 3, 4], dtype=pd.Int32Dtype()
                ),
            }
        ),
    }
    selects = []
    for expr in expressions:
        selects.append(f"CONDITIONAL_TRUE_EVENT({expr}) OVER ({window})")
    query = f"SELECT O, {', '.join(selects)} FROM table1"

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

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, 1, ["CONDITIONAL_TRUE_EVENT"])


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
        "table1": pd.DataFrame(
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
    "data_col, partition_col, answer",
    [
        pytest.param(
            pd.Series([0, 1, -1, None] * 4, dtype=pd.Int32Dtype()),
            pd.Series(["A", "B", "C", "D"] * 4),
            pd.Series([None, 0.25, 0.25, None] * 4),
            id="int32-groups_of_4",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([0, 1, -1, None] * 4, dtype=pd.Int32Dtype()),
            pd.Series(["A"] * 16),
            pd.Series([None] * 16),
            id="int32-single_partition",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series([0, 1, -1, None] * 4, dtype=pd.Int32Dtype()),
            pd.Series(list("AABBCCCCCCDDEEEE")),
            pd.Series(
                [
                    0,
                    1,
                    1,
                    None,
                    0,
                    1,
                    -1,
                    None,
                    0,
                    1,
                    1,
                    None,
                    None,
                    None,
                    None,
                    None,
                ]
            ),
            id="int32-varying_groups",
        ),
        pytest.param(
            pd.Series(
                [None if i % 2 == 1 else i for i in range(16)], dtype=pd.UInt8Dtype()
            ),
            pd.Series(["A", "B", "C", "D"] * 4),
            pd.Series(
                [
                    0,
                    None,
                    1 / 16,
                    None,
                    1 / 6,
                    None,
                    3 / 16,
                    None,
                    1 / 3,
                    None,
                    5 / 16,
                    None,
                    1 / 2,
                    None,
                    7 / 16,
                    None,
                ]
            ),
            id="uint8-groups_of_4",
        ),
        pytest.param(
            pd.Series(
                [None if i % 2 == 1 else i for i in range(16)], dtype=pd.UInt8Dtype()
            ),
            pd.Series(["A"] * 16),
            pd.Series([None if i % 2 == 1 else i / 56 for i in range(16)]),
            id="uint8-single_partition",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [np.inf, 1, -2, 10, np.inf, 30, -np.inf, np.inf, 40, np.inf],
            ),
            pd.Series(["A"] * 5 + ["B"] * 5),
            pd.Series([None, 0, 0, 0, None, None, None, None, None, None]),
            id="float64-infinities",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_ratio_to_report(data_col, partition_col, answer, memory_leak_check):
    query = "SELECT A, B, RATIO_TO_REPORT(A) OVER (PARTITION BY B) FROM table1"

    assert len(data_col) == len(partition_col)
    ctx = {"table1": pd.DataFrame({"A": data_col, "B": partition_col})}

    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": data_col, "B": partition_col, "C": answer}),
        only_jit_1DVar=True,
    )


# In cases with multiple correct answers, the answer returned by our mode implementation
# is the value that was encountered first in the overall array
# due to the semantics of how Python iterates over hashmaps
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
        "table1": pd.DataFrame(
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
