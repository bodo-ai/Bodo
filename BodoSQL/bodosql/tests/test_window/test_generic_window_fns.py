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


@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["SUM", "AVG"], id="sum-avg"),
        pytest.param(["STDDEV", "STDDEV_POP", "STDDEV_SAMP"], id="stddev-variants"),
        pytest.param(
            ["VARIANCE", "VARIANCE_POP", "VAR_SAMP", "VARIANCE_SAMP", "VAR_POP"],
            id="variance-variants",
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
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], funcs)


@pytest.mark.parametrize(
    "funcs",
    [
        pytest.param(["MIN", "MAX"], id="min-max"),
        pytest.param(["COUNT(*)", "COUNT", "COUNT_IF"], id="count-count_if-count_star"),
    ],
)
@pytest.mark.timeout(600)
# passes in 468.71 seconds on 1 rank
def test_non_numeric_window_functions(
    funcs, all_window_df, all_window_col_names, window_frames, spark_info
):
    """Tests min, max, count, count(*) and count_if with various combinations of
    window frames to test correctness and fusion"""
    selects = []
    for i in range(len(funcs)):
        if funcs[i] == "COUNT(*)":
            selects.append(
                f"COUNT(*) OVER ({window_frames[0][i % len(window_frames[0])]})"
            )
        else:
            for j, col in enumerate(all_window_col_names):
                # Skip min/max if the datatype is string
                # [BE-780]/[BE-2116]/[BE-4033]: fix this
                if (
                    funcs[i] in ("MIN", "MAX")
                    and all_window_df["table1"][col].dtype.kind == "O"
                ):
                    continue
                # Skip count_if if the datatype is not a boolean
                if (
                    funcs[i] == "COUNT_IF"
                    and all_window_df["table1"][col].dtype.kind != "b"
                ):
                    continue
                selects.append(
                    f"{funcs[i]}({col}) OVER ({window_frames[0][(i+j) % len(window_frames[0])]})"
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
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, window_frames[1], funcs)


def test_first_last_any_nth(
    all_window_df, all_window_col_names, window_frames, spark_info
):
    """Tests first_value, last_value, any_value and nth_value with various
    combinations of window frames to test correctness, fusion and optimization"""
    batches = [
        ["FIRST_VALUE({:s})", "ANY_VALUE({:s})"],
        ["LAST_VALUE({:s})"],
        ["NTH_VALUE({:s}, 2)", "NTH_VALUE({:s}, 7)", "NTH_VALUE({:s}, 25)"],
    ]
    selects = []
    convert_columns_bytearray = []
    for i in range(len(batches)):
        funcs = batches[i]
        for j, col in enumerate(["I64", "DT", "ST", "BI"]):
            if type(all_window_df["table1"][col].iloc[0]) == bytes:
                convert_columns_bytearray.append(f"C_{i}_{j}")
            selects.append(
                f"{funcs[j % len(funcs)].format(col)} OVER ({window_frames[0][(i+j)%len(window_frames[0])]}) AS C_{i}_{j}"
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
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(
        pandas_code,
        window_frames[1],
        ["FIRST_VALUE", "LAST_VALUE", "ANY_VALUE", "NTH_VALUE"],
    )


def test_first_value_last_value_optimized(
    all_window_df, all_window_col_names, spark_info
):
    """Tests first_value, last_value, any_value and nth_value with various
    combinations of window frames to test correctness, fusion and optimization"""
    selects = []
    window = "PARTITION BY U8 ORDER BY DT ASC NULLS FIRST"
    for col in ["I64", "ST", "BI"]:
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


def test_blended_fusion(memory_leak_check):
    """Tests fusion between RANK, AVG, MEDIAN, MODE and CONDITIONAL_CHANGE_EVENT.
    This allows window funcitons that are not tested together to have one
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
