import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_window
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
    )


@pytest.mark.slow
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
        "idx",
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
            "IDX": list(range(10)),
            "P": [0] * 10,
            "O": pd.array([None] * 10, dtype=pd.Int32Dtype()),
            "B": pd.array([None] * 10, dtype=pd.BooleanDtype()),
        }
    )
    answer = pd.DataFrame(
        {
            "IDX": list(range(10)),
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
