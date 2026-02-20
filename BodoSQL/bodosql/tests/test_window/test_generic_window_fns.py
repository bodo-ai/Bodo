import functools
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from mpi4py import MPI

import bodo
from bodo.tests.utils import (
    pytest_mark_multi_rank_nightly,
    pytest_slow_unless_window,
    temp_env_override,
)
from bodosql.tests.test_window.window_common import (  # noqa
    all_numeric_window_col_names,
    all_numeric_window_df,
    all_window_col_names,
    all_window_df,
    count_window_applies,
    window_frames,
)
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query

# Skip unless any window-related files were changed
pytestmark = pytest_slow_unless_window


@pytest.mark.slow
@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["SUM", "AVG"], id="sum_avg"),
    ],
)
@pytest_mark_multi_rank_nightly
@pytest.mark.timeout(1300)
# passes in 18 minutes on 1 rank
def test_numeric_window_functions(
    funcs,
    all_numeric_window_df,
    all_numeric_window_col_names,
    window_frames,
    spark_info,
):
    """Tests sum, avg, stdev/variance and their variants with various
    combinations of window frames to test correctness and fusion"""
    selects = []
    for i in range(len(funcs)):
        for j, col in enumerate(all_numeric_window_col_names):
            selects.append(
                f"{funcs[i]}({col}) OVER ({window_frames[0][(i + j) % len(window_frames[0])]}) AS c_{funcs[i].lower()}_{col}"
            )
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    pandas_code = check_query(
        query,
        all_numeric_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
        convert_expected_output_to_nullable_float=False,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], funcs)


def test_two_arg_numeric_window_functions(memory_leak_check):
    """Tests covariance/correlation functions with and without a partition"""
    combinations = [
        ("COVAR_SAMP", "PARTITION BY P", "A"),
        ("COVAR_POP", "PARTITION BY P", "B"),
        ("CORR", "PARTITION BY P", "C"),
        ("COVAR_SAMP", "", "D"),
        ("COVAR_POP", "", "E"),
        ("CORR", "", "F"),
    ]
    selects = []
    for func, window, name in combinations:
        selects.append(f"{func}(X, Y) OVER ({window}) AS {name}")
    n_rows = 100
    df = pd.DataFrame(
        {
            "IDX": range(n_rows),
            "P": [(i**2) % 9 for i in range(n_rows)],
            "X": pd.array(
                [
                    None if i % 5 == 1 or (i**2) % 9 == 7 else Decimal(f"{i}.{i % 10}")
                    for i in range(n_rows)
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 1)),
            ),
            "Y": pd.array(
                [
                    None
                    if i % 7 == 3 or (i**2) % 9 == 7
                    else Decimal(f"{i * i}.{i % 10}")
                    for i in range(n_rows)
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 1)),
            ),
        }
    )
    partition_answers = {
        "A": {0: 85477.618498, 1: 79311.173718, 4: 100090.027125, 7: None},
        "B": {0: 81761.200302, 1: 73210.314201, 4: 93834.400430, 7: None},
        "C": {0: 0.966185, 1: 0.971490, 4: 0.972691, 7: None},
    }
    answer = pd.DataFrame(
        {
            "IDX": range(n_rows),
            "A": pd.array([partition_answers["A"][(i**2) % 9] for i in range(n_rows)]),
            "B": pd.array([partition_answers["B"][(i**2) % 9] for i in range(n_rows)]),
            "C": pd.array([partition_answers["C"][(i**2) % 9] for i in range(n_rows)]),
            "D": pd.array([85161.45263951733 for i in range(n_rows)]),
            "E": pd.array([83523.7323964497 for i in range(n_rows)]),
            "F": pd.array([0.9686966438205089 for i in range(n_rows)]),
        }
    )
    query = f"SELECT IDX, {', '.join(selects)} FROM table1"
    check_query(
        query,
        {"TABLE1": df},
        None,
        expected_output=answer,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
    )


@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["MIN", "MAX"], id="min-max"),
    ],
)
@pytest_mark_multi_rank_nightly
@pytest.mark.timeout(1200)
# NOTE (allai5): passes in 592.81 seconds on 1 rank on M1 as of 05/03/2023
def test_non_numeric_window_functions(
    funcs, all_window_df, all_window_col_names, window_frames, spark_info
):
    """Tests min, max, count, count(*) and count_if with various combinations of
    window frames to test correctness and fusion"""
    selects = []
    convert_columns_bytearray = []
    for i in range(len(funcs)):
        for j, col in enumerate(all_window_col_names):
            selects.append(
                f"{funcs[i]}({col}) OVER ({window_frames[0][(i + j) % len(window_frames[0])]}) AS C_{i}_{j}"
            )
            # If taking the min/max of a binary column, add the output to
            # the list of conversion columns
            if col == "BI" and funcs[i] in ("MIN", "MAX"):
                convert_columns_bytearray.append(f"C_{i}_{j}")
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    # TODO: Generate an expected output instead so we properly support TZ-Aware
    pandas_code = check_query(
        query,
        all_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
        use_duckdb=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], funcs)


@pytest.mark.parametrize(
    "window_calls",
    [
        pytest.param(
            [
                ("FIRST_VALUE(I64)", "UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
                ("FIRST_VALUE(BI)", "UNBOUNDED PRECEDING", "3 PRECEDING"),
                ("FIRST_VALUE(ST)", "1 PRECEDING", "UNBOUNDED FOLLOWING"),
                ("FIRST_VALUE(F64)", "1 FOLLOWING", "10 FOLLOWING"),
            ],
            id="first_value",
        ),
        pytest.param(
            [
                ("LAST_VALUE(ST)", "UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
                ("LAST_VALUE(DT)", "UNBOUNDED PRECEDING", "2 FOLLOWING"),
                ("LAST_VALUE(F64)", "1 FOLLOWING", "UNBOUNDED FOLLOWING"),
                ("LAST_VALUE(BI)", "3 PRECEDING", "3 FOLLOWING"),
            ],
            id="last_value",
        ),
        pytest.param(
            [
                ("NTH_VALUE(ST, 10)", "UNBOUNDED PRECEDING", "UNBOUNDED FOLLOWING"),
                ("NTH_VALUE(DT, 5)", "UNBOUNDED PRECEDING", "CURRENT ROW"),
                ("NTH_VALUE(I64, 3)", "1 FOLLOWING", "UNBOUNDED FOLLOWING"),
                ("NTH_VALUE(BI, 3)", "3 PRECEDING", "3 FOLLOWING"),
            ],
            id="nth_value",
        ),
    ],
)
def test_first_last_nth(window_calls, all_window_df, spark_info):
    """Tests first_value, last_value and nth_value."""
    selects = []
    convert_columns_bytearray = []
    for i in range(len(window_calls)):
        window_call, lower, upper = window_calls[i]
        selects.append(
            f"{window_call} OVER (PARTITION BY W3 ORDER BY W4 ROWS BETWEEN {lower} AND {upper}) AS C_{i}"
        )
        if "BI" in window_call:
            convert_columns_bytearray.append(f"C_{i}")
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    check_query(
        query,
        all_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
        use_duckdb=True,
    )


@pytest.mark.slow
@pytest_mark_multi_rank_nightly
def test_blended_fusion(memory_leak_check):
    """Tests fusion between RANK, AVG, MEDIAN, MODE and CONDITIONAL_CHANGE_EVENT.
    This allows window functions that are not tested together to have one
    test that checks that they can all be fused into the same closure."""
    ctx = {
        "TABLE1": pd.DataFrame(
            {
                "A": pd.Series(
                    [None, None, 2, 3, None, 5, None, 7, None, None],
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series(["A"] * 10),
                "C": pd.Series(list(range(10)), dtype=pd.Int32Dtype()),
            }
        )
    }
    selects = []
    funcs = [
        ("RANK()", False),
        ("AVG(A)", True),
        ("MEDIAN(A)", False),
        ("MODE(A)", False),
        ("CONDITIONAL_CHANGE_EVENT(A)", False),
        ("SUM(A)", True),
    ]
    for func, frame_syntax in funcs:
        suffix = (
            "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
            if frame_syntax
            else ""
        )
        selects.append(f"{func} OVER (PARTITION BY B ORDER BY C {suffix})")
    query = f"SELECT C, {', '.join(selects)} FROM table1"
    answer = pd.DataFrame(
        {
            "C": range(10),
            "RANK": range(1, 11),
            "AVG": [4.25] * 10,
            "MEDIAN": [4] * 10,
            "MODE": [2] * 10,
            "CONDITIONAL_CHANGE_EVENT": pd.Series(
                [0, 0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=pd.Int32Dtype()
            ),
            "SUM": [17] * 10,
        }
    )
    pandas_code = check_query(
        query,
        ctx,
        None,
        expected_output=answer,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(
        pandas_code, 1, ["RANK", "AVG", "MEDIAN", "MODE", "CONDITIONAL_CHANGE_EVENT"]
    )


@pytest.mark.parametrize(
    "func",
    [
        pytest.param("AVG"),
        pytest.param("STDDEV"),
        pytest.param("STDDEV_POP"),
        pytest.param("STDDEV_SAMP"),
        pytest.param("VARIANCE"),
        pytest.param("VARIANCE_POP"),
        pytest.param("VAR_SAMP"),
        pytest.param("VARIANCE_SAMP"),
        pytest.param("VAR_POP"),
    ],
)
def test_optimized_numeric_window_functions(func, all_numeric_window_df, spark_info):
    """Tests numeric window functions that can use groupby.window"""
    selects = []
    combinations = [
        ("U8", "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"),
        ("U8", "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"),
        ("U8", "ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING"),
        ("U8", "ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING"),
        ("I64", "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"),
        ("I64", "ROWS BETWEEN UNBOUNDED PRECEDING AND 2 PRECEDING"),
        ("I64", "ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING"),
        ("I64", "ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING"),
        ("F64", "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"),
        ("F64", "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING"),
        ("F64", "ROWS BETWEEN 3 PRECEDING AND UNBOUNDED FOLLOWING"),
        ("F64", "ROWS BETWEEN 10 PRECEDING AND CURRENT ROW"),
    ]
    for i, (col, frame) in enumerate(combinations):
        selects.append(
            f"{func}({col}) OVER (PARTITION BY W3 ORDER BY W4 {frame}) AS C{i}"
        )
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    check_query(
        query,
        all_numeric_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_expected_output_to_nullable_float=False,
    )


def test_count_fns(all_window_df, spark_info):
    """
    Tests the window functions COUNT(*), COUNT and COUNT_IF.
    """
    # Window frames to use for testing
    # 0: whole frame
    # 1: prefix frame
    # 2: suffix frame (exclusive of the current row)
    # 3: sliding frame
    frames = [
        "ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING",
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW",
        "ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING",
        "ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING",
    ]
    # Specify all combinations that are to be tested as a tuple of 3 components:
    # 1. The function name to test
    # 2. The column name to test the function with
    # 3. List of indices corresponding to which of the frames above to test with
    combinations = [
        ("COUNT", "*", [0, 1, 2, 3]),
        ("COUNT", "ST", [0, 1, 2, 3]),
        ("COUNT", "U8", [0, 1]),
        ("COUNT", "F64", [2, 3]),
        ("COUNT", "DA", [0, 3]),
        ("COUNT_IF", "BO", [0, 1, 2, 3]),
    ]
    selects = []
    for func, arg, frames_subset in combinations:
        for frame_idx in frames_subset:
            selects.append(
                f"{func}({arg}) OVER (PARTITION BY W2 ORDER BY W4 {frames[frame_idx]})"
            )
    query = f"SELECT W2, W4, {', '.join(selects)} FROM table1"
    check_query(
        query,
        all_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_any_value(all_window_df, all_window_col_names, spark_info):
    """Tests any_value by itself on multiple column types, using the optimized
    groupby.window codegen path"""
    data_cols = ["U8", "I64", "F64", "BO", "ST", "BI"]
    selects = [
        f"ANY_VALUE({col}) OVER (PARTITION BY W3) AS C_{col}" for col in data_cols
    ]
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    convert_columns_bytearray = ["C_BI"]
    spark_query = get_equivalent_spark_agg_query(query)
    check_query(
        query,
        all_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_all_null():
    """
    Tests that window functions using groupby.apply work correctly when the
    data is all-null.
    """
    selects = [
        "ROW_NUMBER() OVER (PARTITION BY P ORDER BY O) AS RN",
        "RANK() OVER (PARTITION BY P ORDER BY O) AS R",
        "DENSE_RANK() OVER (PARTITION BY P ORDER BY O) AS DR",
        "PERCENT_RANK() OVER (PARTITION BY P ORDER BY O) AS PR",
        "CUME_DIST() OVER (PARTITION BY P ORDER BY O) AS CD",
        "NTILE(2) OVER (PARTITION BY P ORDER BY O) AS NT",
        "FIRST_VALUE(O) OVER (PARTITION BY P ORDER BY O) AS FV",
        "LAST_VALUE(O) OVER (PARTITION BY P ORDER BY O) AS LV",
        "NTH_VALUE(O, 1) OVER (PARTITION BY P ORDER BY O) AS NV",
        "RATIO_TO_REPORT(O) OVER (PARTITION BY P ORDER BY O) AS RTR",
        "AVG(O) OVER (PARTITION BY P ORDER BY O) AS A",
        "VARIANCE(O) OVER (PARTITION BY P ORDER BY O) AS V",
        "VARIANCE_POP(O) OVER (PARTITION BY P ORDER BY O) AS VP",
        "STDDEV(O) OVER (PARTITION BY P ORDER BY O) AS S",
        "STDDEV_POP(O) OVER (PARTITION BY P ORDER BY O) AS SP",
        "SKEW(O) OVER (PARTITION BY P ORDER BY O) AS SK",
        "KURTOSIS(O) OVER (PARTITION BY P ORDER BY O) AS KU",
        "COUNT(*) OVER (PARTITION BY P ORDER BY O) AS CS",
        "COUNT(O) OVER (PARTITION BY P ORDER BY O) AS C",
        "COUNT_IF(B) OVER (PARTITION BY P ORDER BY O) AS CI",
        "BOOLOR_AGG(B) OVER (PARTITION BY P ORDER BY O) AS BO",
        "BOOLAND_AGG(B) OVER (PARTITION BY P ORDER BY O) AS BA",
        "BOOLXOR_AGG(B) OVER (PARTITION BY P ORDER BY O) AS BX",
        "BITOR_AGG(O) OVER (PARTITION BY P ORDER BY O) AS BIO",
        "BITAND_AGG(O) OVER (PARTITION BY P ORDER BY O) AS BIA",
        "BITXOR_AGG(O) OVER (PARTITION BY P ORDER BY O) AS BIX",
        "MEDIAN(O) OVER (PARTITION BY P ORDER BY O) AS ME",
        "MODE(O) OVER (PARTITION BY P ORDER BY O) AS MO",
        "MIN(O) OVER (PARTITION BY P ORDER BY O) AS MI",
        "MAX(O) OVER (PARTITION BY P ORDER BY O) AS MA",
        "SUM(O) OVER (PARTITION BY P ORDER BY O) AS SU",
        "LEAD(O) OVER (PARTITION BY P ORDER BY O) AS LL1",
        "LAG(O, 2) OVER (PARTITION BY P ORDER BY O) AS LL2",
        "LEAD(O, 3, -1) OVER (PARTITION BY P ORDER BY O) AS LL3",
    ]
    query = f"SELECT {', '.join(selects)} FROM table1"
    df = pd.DataFrame(
        {
            "P": [0] * 10,
            "O": pd.array([None] * 10, dtype=pd.Int32Dtype()),
            "B": pd.array([None] * 10, dtype=pd.BooleanDtype()),
        }
    )
    answer = pd.DataFrame(
        {
            "RN": list(range(1, 11)),
            "R": [1] * 10,
            "DR": [1] * 10,
            "PR": [0] * 10,
            "CD": [1] * 10,
            "NT": [1] * 5 + [2] * 5,
            "FV": [None] * 10,
            "LV": [None] * 10,
            "NV": [None] * 10,
            "RTR": [None] * 10,
            "A": [None] * 10,
            "V": [None] * 10,
            "VP": [None] * 10,
            "S": [None] * 10,
            "SP": [None] * 10,
            "SK": [None] * 10,
            "KU": [None] * 10,
            "CS": list(range(1, 11)),
            "C": [0] * 10,
            "CI": [0] * 10,
            "BO": [None] * 10,
            "BA": [None] * 10,
            "BX": [None] * 10,
            "BIO": [None] * 10,
            "BIA": [None] * 10,
            "BIX": [None] * 10,
            "ME": [None] * 10,
            "MO": [None] * 10,
            "MI": [None] * 10,
            "MA": [None] * 10,
            "SU": [None] * 10,
            "LL1": [None] * 10,
            "LL2": [None] * 10,
            "LL3": [None] * 7 + [-1] * 3,
        }
    )
    check_query(
        query,
        {"TABLE1": df},
        None,
        expected_output=answer,
        sort_output=True,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(7000),
                    "P": [str(i)[:2] for i in range(7000)],
                    "S": pd.array(
                        [None if i % 2 == 0 else i % 128 for i in range(7000)],
                        dtype=pd.Int8Dtype(),
                    ),
                }
            ),
            id="int8",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(9876),
                    "P": [int(np.tan(i)) for i in range(9876)],
                    "S": pd.array(
                        [np.tan(i) for i in range(9876)], dtype=pd.Float64Dtype()
                    ),
                }
            ),
            id="float64",
        ),
    ],
)
def test_simple_sum(df, spark_info, capfd):
    """Verifies that the correct path is taken for SUM"""

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT IDX, SUM(S) OVER (PARTITION BY P) as W FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(13000),
                    "P": [str(i)[:2] for i in range(13000)],
                    "S": pd.array(
                        [None if i % 2 == 0 else i % 128 for i in range(13000)],
                        dtype=pd.UInt8Dtype(),
                    ),
                }
            ),
            id="uint8",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "P": [round(1.5 * np.sin(i)) for i in range(1000)],
                    "S": pd.array(
                        [
                            [None, True, False][min(i % 3, i % 4, i % 5)]
                            for i in range(1000)
                        ],
                        dtype=pd.BooleanDtype(),
                    ),
                }
            ),
            id="bool",
        ),
    ],
)
def test_simple_count(df, spark_info, capfd):
    """Verifies that the correct path is taken for COUNT"""
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT IDX, COUNT(S) OVER (PARTITION BY P) as W FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "partition_col",
    [
        pytest.param("P1", id="single_partition"),
        pytest.param("P2", id="two_partitions"),
        pytest.param("P3", id="few_partitions", marks=pytest.mark.slow),
        pytest.param("P4", id="more_partitions"),
        pytest.param("P5", id="many_partitions", marks=pytest.mark.slow),
    ],
)
def test_simple_count_star(partition_col, spark_info, capfd):
    """Verifies that the correct path is taken for COUNT(*)"""
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = f"SELECT IDX, COUNT(*) OVER (PARTITION BY {partition_col}) as W FROM TABLE1"

    df = pd.DataFrame(
        {
            "IDX": range(12000),
            "P1": [17 for i in range(12000)],
            "P2": [min(i % 2, i % 3, i % 4, i % 5) for i in range(12000)],
            "P3": [min(i % 6, i % 7, i % 8) for i in range(12000)],
            "P4": [int(np.tan(i)) for i in range(12000)],
            "P5": [int((np.tan(i) * 3) ** 2) for i in range(12000)],
        }
    )

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
def test_multiple_sum_count(spark_info, capfd):
    """Verifies that the correct path is taken for SUM/COUNT when called multiple times"""
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    df = pd.DataFrame(
        {
            "IDX": range(4500),
            "P": [int(np.tan(i)) for i in range(4500)],
            "S": pd.array(
                [None if i % 7 == 6 else i % 255 for i in range(4500)],
                dtype=pd.UInt8Dtype(),
            ),
        }
    )
    terms = [
        "SUM(S) OVER (PARTITION BY P) as W1",
        "COUNT(S) OVER (PARTITION BY P) as W2",
        "COUNT(*) OVER (PARTITION BY P) as W3",
    ]
    query = f"SELECT IDX, {', '.join(terms)} FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "df",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(13000),
                    "P": [str(i)[:2] for i in range(13000)],
                    "S": pd.array(
                        [None if i % 2 == 0 else i % 128 for i in range(13000)],
                        dtype=pd.UInt8Dtype(),
                    ),
                }
            ),
            id="uint8",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "IDX": range(1000),
                    "P": [round(1.5 * np.sin(i)) for i in range(1000)],
                    "S": pd.array(
                        [
                            [None, True, False][min(i % 3, i % 4, i % 5)]
                            for i in range(1000)
                        ],
                        dtype=pd.BooleanDtype(),
                    ),
                }
            ),
            id="bool",
        ),
    ],
)
def test_simple_count(df, spark_info, capfd):
    """Verifies that the correct path is taken for COUNT"""

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT IDX, COUNT(S) OVER (PARTITION BY P) as W FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "partition_col",
    [
        pytest.param("P1", id="single_partition"),
        pytest.param("P2", id="two_partitions"),
        pytest.param("P3", id="few_partitions", marks=pytest.mark.slow),
        pytest.param("P4", id="more_partitions"),
        pytest.param("P5", id="many_partitions", marks=pytest.mark.slow),
    ],
)
def test_simple_count_star(partition_col, spark_info, capfd):
    """Verifies that the correct path is taken for COUNT(*)"""

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = f"SELECT IDX, COUNT(*) OVER (PARTITION BY {partition_col}) as W FROM TABLE1"

    df = pd.DataFrame(
        {
            "IDX": range(12000),
            "P1": [17 for i in range(12000)],
            "P2": [min(i % 2, i % 3, i % 4, i % 5) for i in range(12000)],
            "P3": [min(i % 6, i % 7, i % 8) for i in range(12000)],
            "P4": [int(np.tan(i)) for i in range(12000)],
            "P5": [int((np.tan(i) * 3) ** 2) for i in range(12000)],
        }
    )

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            spark_info,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "func_name, answer_func, out_dtype",
    [
        pytest.param(
            "COUNT_IF",
            lambda x: x.astype(pd.BooleanDtype()).sum(),
            pd.UInt64Dtype(),
            id="count_if",
        ),
        pytest.param(
            "BOOLOR_AGG",
            lambda x: None if x.count() == 0 else (x.sum() > 0),
            pd.BooleanDtype(),
            id="boolor_agg",
        ),
        pytest.param(
            "BOOLAND_AGG",
            lambda x: None
            if x.count() == 0
            else (x.astype(pd.BooleanDtype()).sum() == x.count()),
            pd.BooleanDtype(),
            id="booland_agg",
        ),
        pytest.param(
            "BITOR_AGG",
            lambda x: None
            if x.count() == 0
            else functools.reduce(lambda a, b: a | b, x[pd.notna(x)]),
            pd.Int64Dtype(),
            id="bitor_agg",
        ),
        pytest.param(
            "BITAND_AGG",
            lambda x: None
            if x.count() == 0
            else functools.reduce(lambda a, b: a & b, x[pd.notna(x)]),
            pd.Int64Dtype(),
            id="bitand_agg",
        ),
        pytest.param(
            "BITXOR_AGG",
            lambda x: None
            if x.count() == 0
            else functools.reduce(lambda a, b: a ^ b, x[pd.notna(x)]),
            pd.Int64Dtype(),
            id="bitxor_agg",
        ),
    ],
)
@pytest.mark.parametrize(
    "df, arg_strings",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "P": [0] * 500,
                    "IDX": range(500),
                    "D": pd.array(
                        [
                            [None, False, True][min(i % 3, i % 4, i % 5)]
                            for i in range(500)
                        ],
                        dtype=pd.BooleanDtype(),
                    ),
                }
            ),
            {
                "COUNT_IF": "D",
                "BOOLAND_AGG": "D",
                "BOOLOR_AGG": "D",
                "BITAND_AGG": "D::INTEGER",
                "BITOR_AGG": "D::INTEGER",
                "BITXOR_AGG": "D::INTEGER",
            },
            id="boolean-single_partition",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "P": [i // 100 for i in range(300)],
                    "IDX": range(300),
                    "D": pd.array(
                        [[None, False, True][i // 100] for i in range(300)],
                        dtype=pd.BooleanDtype(),
                    ),
                }
            ),
            {
                "COUNT_IF": "D",
                "BOOLAND_AGG": "D",
                "BOOLOR_AGG": "D",
                "BITAND_AGG": "D::INTEGER",
                "BITOR_AGG": "D::INTEGER",
                "BITXOR_AGG": "D::INTEGER",
            },
            id="boolean-two_partitions",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "P": [round(np.tan(i)) for i in range(200)],
                    "IDX": range(200),
                    "D": pd.array(
                        [[True, None, False][min(i % 3, i % 4)] for i in range(200)],
                        dtype=pd.BooleanDtype(),
                    ),
                }
            ),
            {
                "COUNT_IF": "D",
                "BOOLAND_AGG": "D",
                "BOOLOR_AGG": "D",
                "BITAND_AGG": "D::INTEGER",
                "BITOR_AGG": "D::INTEGER",
                "BITXOR_AGG": "D::INTEGER",
            },
            id="boolean-many_partitions",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "P": [0] * 500,
                    "IDX": range(500),
                    "D": pd.array(
                        [None if i % 2 == 0 else i % 128 for i in range(500)],
                        dtype=pd.Int8Dtype(),
                    ),
                }
            ),
            {
                "COUNT_IF": "D>0",
                "BOOLAND_AGG": "D",
                "BOOLOR_AGG": "D",
                "BITAND_AGG": "D",
                "BITOR_AGG": "D",
                "BITXOR_AGG": "D",
            },
            id="integer-single_partition",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "P": [round(np.tan(i)) for i in range(500)],
                    "IDX": range(500),
                    "D": pd.array(
                        [None if i % 2 == 0 else i % 128 for i in range(500)],
                        dtype=pd.Int64Dtype(),
                    ),
                }
            ),
            {
                "COUNT_IF": "D>0",
                "BOOLAND_AGG": "D",
                "BOOLOR_AGG": "D",
                "BITAND_AGG": "D",
                "BITOR_AGG": "D",
                "BITXOR_AGG": "D",
            },
            id="integer-many_partitions",
        ),
    ],
)
def test_simple_aggfuncs(func_name, answer_func, out_dtype, df, arg_strings, capfd):
    """Verifies that the correct path is taken for COUNT_IF, BOOLOR_AGG, BOOLAND_AGG,
    BITAND_AGG, BITOR_AGG, BITXOR_AGG"""

    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = f"SELECT IDX, {func_name}({arg_strings[func_name]}) OVER (PARTITION BY P) as W FROM TABLE1"

    gb_answers = df.groupby("P")["D"].apply(answer_func)
    answer_dict = dict(zip(gb_answers.index, gb_answers))

    answer = pd.DataFrame(
        {
            "IDX": df["IDX"],
            "W": pd.array([answer_dict[p] for p in df["P"]], dtype=out_dtype),
        }
    )

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            None,
            expected_output=answer,
            check_names=False,
            check_dtype=False,
            sort_output=True,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
@pytest.mark.parametrize(
    "data, expected_out",
    [
        pytest.param(
            pd.array(
                [None if i**2 % 7 < 3 else float(f"{i}.{i}") for i in range(10000)]
            ),
            pd.array([5000.2999808540435] * 10000),
            id="float64_arr",
        ),
        pytest.param(
            pd.array(
                [None if i**2 % 7 < 3 else Decimal(f"{i}.{i}") for i in range(10000)],
                dtype=pd.ArrowDtype(pa.decimal128(32, 5)),
            ),
            pd.array(
                [Decimal("5000.29998085404")] * 10000,
                dtype=pd.ArrowDtype(pa.decimal128(38, 11)),
            ),
            id="decimal_arr",
        ),
    ],
)
def test_avg_over_blank(data, expected_out, capfd):
    """Verifies that the correct path is taken for AVG"""

    df = pd.DataFrame({"A": data})

    expected_df = pd.DataFrame({"AVG": expected_out})
    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT AVG(A) OVER () FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            None,
            expected_output=expected_df,
            check_names=False,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
def test_countstar_over_blank(capfd):
    """Checks that count(*) over () works properly and takes the correct codepath"""

    df = pd.DataFrame(
        {
            "A": pd.array([None, 1, 2, 4, None, None, 2], dtype=pd.Int32Dtype),
            "B": pd.array([None] * 7, dtype=pd.StringDtype),
            "C": pd.array(
                [1.2, 2.1, None, None, 3.2, None, 4.2], dtype=pd.Float32Dtype
            ),
        }
    )

    expected_df = pd.DataFrame(
        {"ROW_COUNT": pd.array([len(df)] * len(df), dtype="uint64")}
    )
    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT COUNT(*) OVER () FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            None,
            expected_output=expected_df,
            check_names=False,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success


@pytest.mark.skipif(
    bodo.tests.utils.test_spawn_mode_enabled,
    reason="capfd doesn't work for spawn",
)
def test_multiple_over_blank(capfd):
    """Checks that multiple FUNC(X) OVER () work properly and takes the correct codepath"""
    from mpi4py import MPI

    from bodo.tests.utils import temp_env_override

    df = pd.DataFrame(
        {
            "IDX": range(1000),
            "DATA": pd.array(
                [None if i % 2 == i % 3 else str(i)[2:] for i in range(1000)]
            ),
        }
    )

    expected_df = pd.DataFrame(
        {
            "IDX": range(1000),
            "NON_NULL": pd.array([666 for i in range(1000)]),
            "TOTAL": pd.array([1000 for i in range(1000)]),
        }
    )
    expected_log_message = "[DEBUG] WindowState::FinalizeBuild: Finished"

    query = "SELECT IDX, COUNT(DATA) OVER () as NON_NULL, COUNT(*) OVER () as TOTAL FROM TABLE1"

    with temp_env_override(
        {
            "BODO_DEBUG_STREAM_GROUPBY_PARTITIONING": "1",
        }
    ):
        check_query(
            query,
            {"TABLE1": df},
            None,
            expected_output=expected_df,
            check_names=False,
            check_dtype=False,
        )

    comm = MPI.COMM_WORLD
    _, err = capfd.readouterr()
    assert_success = expected_log_message in err
    assert_success = comm.allreduce(assert_success, op=MPI.LAND)

    assert assert_success
