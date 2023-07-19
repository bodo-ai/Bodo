import pandas as pd
import pytest
from bodosql.tests.test_window.window_common import (  # noqa
    all_numeric_window_col_names,
    all_numeric_window_df,
    all_window_col_names,
    all_window_df,
    count_window_applies,
    window_frames,
)
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query


@pytest.mark.slow
@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["SUM", "AVG"], id="sum_avg"),
        pytest.param(["STDDEV", "STDDEV_POP", "STDDEV_SAMP"], id="stddev_variants"),
        pytest.param(
            ["VARIANCE", "VARIANCE_POP", "VAR_SAMP", "VARIANCE_SAMP", "VAR_POP"],
            id="variance_variants",
        ),
    ],
)
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
                f"{funcs[i]}({col}) OVER ({window_frames[0][(i+j) % len(window_frames[0])]}) AS c_{funcs[i].lower()}_{col}"
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


@pytest.mark.slow
def test_two_arg_numeric_window_functions(
    all_numeric_window_df,
    window_frames,
    spark_info,
):
    """Tests covariance functions with various combinations of window frames to
    test correctness and fusion"""
    combinations = [
        ("COVAR_SAMP", "W3", "I64"),
        ("COVAR_SAMP", "U8", "F64"),
        ("COVAR_POP", "W3", "I64"),
        ("COVAR_POP", "U8", "F64"),
        ("CORR", "W3", "I64"),
        ("CORR", "U8", "F64"),
    ]
    selects = []
    for i in range(len(combinations)):
        func, arg0, arg1 = combinations[i]
        selects.append(
            f"{func}({arg0}, {arg1}) OVER ({window_frames[0][i % len(window_frames[0])]}) AS c{i}"
        )
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    pandas_code = check_query(
        query,
        all_numeric_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], ["COVAR_SAMP", "COVAR_POP"])


@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["MIN", "MAX"], id="min-max"),
    ],
)
@pytest.mark.timeout(1200)
# NOTE (allai5): passes in 592.81 seconds on 1 rank on M1 as of 05/03/2023
def test_non_numeric_window_functions(
    funcs, all_window_df, all_window_col_names, window_frames, spark_info
):
    """Tests min, max, count, count(*) and count_if with various combinations of
    window frames to test correctness and fusion"""
    # Convert the spark input to tz-naive bc it can't handle timezones
    convert_columns_tz_naive = ["TZ"]
    selects = []
    convert_columns_bytearray = []
    for i in range(len(funcs)):
        for j, col in enumerate(all_window_col_names):
            selects.append(
                f"{funcs[i]}({col}) OVER ({window_frames[0][(i+j) % len(window_frames[0])]}) AS C_{i}_{j}"
            )
            # If taking the min/max of a binary column, add the output to
            # the list of conversion columns
            if col == "BI" and funcs[i] in ("MIN", "MAX"):
                convert_columns_bytearray.append(f"C_{i}_{j}")
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    # TODO: Generate an expected output instead so we properly support TZ-Aware
    pandas_code = check_query(
        query,
        all_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
        convert_columns_tz_naive=convert_columns_tz_naive,
        convert_columns_bytearray=convert_columns_bytearray,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], funcs)


@pytest.mark.timeout(700)
@pytest.mark.slow
def test_first_last_any_nth(
    all_window_df, all_window_col_names, window_frames, spark_info
):
    """Tests first_value, last_value, any_value and nth_value with various
    combinations of window frames to test correctness, fusion and optimization"""
    window_calls = [
        "FIRST_VALUE(I64)",
        "FIRST_VALUE(ST)",
        "LAST_VALUE(DA)",
        "LAST_VALUE(ST)",
        "LAST_VALUE(BI)",
        "ANY_VALUE(TZ)",
        "NTH_VALUE(DT, 2)",
        "NTH_VALUE(BI, 7)",
        "NTH_VALUE(I64, 25)",
    ]
    selects = []
    convert_columns_bytearray = []
    convert_columns_tz_naive = []
    for i in range(len(window_calls)):
        if "BI" in window_calls[i]:
            convert_columns_bytearray.append(f"C_{i}")
        if "TZ" in window_calls[i]:
            convert_columns_tz_naive.append(f"C_{i}")
        selects.append(
            f"{window_calls[i]} OVER ({window_frames[0][i%len(window_frames[0])]}) AS C_{i}"
        )
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    pandas_code = check_query(
        query,
        all_window_df,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
        convert_columns_bytearray=convert_columns_bytearray,
        convert_columns_tz_naive=convert_columns_tz_naive,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(
        pandas_code,
        window_frames[1],
        ["FIRST_VALUE", "LAST_VALUE", "ANY_VALUE", "NTH_VALUE"],
    )


@pytest.mark.slow
def test_first_value_last_value_optimized(
    all_window_df, all_window_col_names, spark_info
):
    """Tests first_value, last_value, any_value and nth_value with various
    combinations of window frames to test correctness, fusion and optimization"""
    selects = []
    window = "PARTITION BY U8 ORDER BY DT ASC NULLS FIRST"
    for col in ["I64", "ST", "BI", "DA"]:
        selects.append(
            f"FIRST_VALUE({col}) OVER ({window} ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {col}_FV"
        )
        selects.append(
            f"LAST_VALUE({col}) OVER ({window} ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS {col}_LV"
        )
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    convert_columns_bytearray = ["BI_FV", "BI_LV"]
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
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, 1, ["FIRST_VALUE", "LAST_VALUE"])

    # Verify that there are only the initial sorts, not the reverse sorts
    assert pandas_code.count("sort_values") == 1


@pytest.mark.slow
def test_blended_fusion(memory_leak_check):
    """Tests fusion between RANK, AVG, MEDIAN, MODE and CONDITIONAL_CHANGE_EVENT.
    This allows window functions that are not tested together to have one
    test that checks that they can all be fused into the same closure."""
    ctx = {
        "table1": pd.DataFrame(
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
        "RANK()",
        "AVG(A)",
        "MEDIAN(A)",
        "MODE(A)",
        "CONDITIONAL_CHANGE_EVENT(A)",
    ]
    for i in range(len(funcs)):
        selects.append(f"{funcs[i]} OVER (PARTITION BY B ORDER BY C)")
    query = f"SELECT C, {', '.join(selects)} FROM table1"
    answer = pd.DataFrame(
        {
            "C": pd.Series(list(range(10))),
            "RANK": pd.Series([i + 1 for i in range(10)]),
            "AVG": pd.Series(
                [None, None, 2, 2.5, 2.5, 10 / 3, 10 / 3, 4.25, 4.25, 4.25]
            ),
            "MEDIAN": pd.Series([None, None, 2, 2.5, 2.5, 3, 3, 4, 4, 4]),
            "MODE": pd.Series([None] * 2 + [2] * 8, dtype=pd.Int32Dtype()),
            "CONDITIONAL_CHANGE_EVENT": pd.Series(
                [0, 0, 0, 1, 1, 2, 2, 3, 3, 3], dtype=pd.Int32Dtype()
            ),
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
