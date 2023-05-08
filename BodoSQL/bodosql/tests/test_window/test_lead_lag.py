import pandas as pd
import pytest
from bodosql.tests.test_window.window_common import (  # noqa
    all_window_col_names,
    all_window_df,
    count_window_applies,
    null_respect_string,
    testing_locally,
    uint8_window_df,
)
from bodosql.tests.utils import check_query


@pytest.fixture(params=["LEAD", "LAG"])
def lead_or_lag(request):
    return request.param


def gen_lead_lag_queries(
    window,
    col_name,
    fill_value,
    null_respect_string,
    include_two_arg_test,
    lead_or_lag_value,
):
    """Helper function, that generates a string of lead/lag queries"""

    arg_list = [(f"{col_name}", "default"), (f"{col_name}, 6, NULL", "6_null")]
    if include_two_arg_test:
        arg_list.append((f"{col_name}, -3", "negative_3"))

    # TODO: fix for binary constants [BE-3304] and tz-aware constants
    if "X'" not in fill_value and fill_value != "NULL":
        arg_list.append((f"{col_name}, 5, {fill_value}", "5_with_fill"))

    lead_lag_queries = ", ".join(
        [
            f"{lead_or_lag_value}({args}) {null_respect_string} OVER {window} as {lead_or_lag_value}_{name}_{col_name}"
            for args, name in arg_list
        ]
    )
    lead_lag_names = [f"{lead_or_lag_value}_{name}_{col_name}" for _, name in arg_list]

    return lead_lag_queries, lead_lag_names


@pytest.mark.slow
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "cols_to_use, window_frame, nulls_handling",
    [
        pytest.param(
            ["U8", "I64", "F64", "BO"],
            "PARTITION BY W1 ORDER BY W4",
            "RESPECT NULLS",
            id="numerics",
        ),
        pytest.param(
            ["ST", "BI", "DT", "TZ", "DA"],
            "PARTITION BY W2 ORDER BY W3, W4",
            "IGNORE NULLS",
            id="non_numerics",
        ),
    ],
)
def test_lead_lag_handle_nulls(
    all_window_df,
    all_window_col_names,
    cols_to_use,
    window_frame,
    nulls_handling,
    spark_info,
):
    """Tests LEAD/LAG window functions with many queries in the same groupby-apply
    from different types. Each type is tested with either LEAD or LAG, and
    with 1 and 3 arguments (sometimes 2).

    Binary data has its 3-argument case skipped since for now."""
    selects = []
    include_two_arg_test = True
    cols_that_need_bytearray_conv = []
    # Columns that need tz info removed before passing it to spark
    cols_to_remove_tz = []
    for i, col in enumerate(cols_to_use):
        fill_value = all_window_col_names[col]
        lead_or_lag_value = "LEAD" if (i % 3) == 0 else "LAG"
        lead_lag_queries, lead_lag_names = gen_lead_lag_queries(
            f"({window_frame})",
            col,
            fill_value,
            nulls_handling,
            include_two_arg_test,
            lead_or_lag_value,
        )
        if col == "BI":
            cols_that_need_bytearray_conv.extend(lead_lag_names)
        if col == "TZ":
            cols_to_remove_tz.append("TZ")
        selects.append(lead_lag_queries)
        include_two_arg_test = not include_two_arg_test
    query = f"SELECT W4, {', '.join(selects)} FROM table1"
    pandas_code = check_query(
        query,
        all_window_df,
        spark_info,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        convert_columns_bytearray=cols_that_need_bytearray_conv,
        convert_columns_tz_naive=cols_to_remove_tz,
        only_jit_1DVar=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, 1, ["LEAD", "LAG"])


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "windows",
    [
        pytest.param(["PARTITION BY B ORDER BY C NULLS FIRST"] * 4, id="all_same"),
        pytest.param(
            [
                "PARTITION BY B ORDER BY A",
                "PARTITION BY 1 ORDER BY A",
            ]
            * 2,
            id="two_partitions",
        ),
        pytest.param(
            ["PARTITION BY B ORDER BY C NULLS FIRST"] * 2
            + ["PARTITION BY B ORDER BY C DESC NULLS FIRST"] * 2,
            id="two_order_asc",
        ),
        pytest.param(
            [
                "PARTITION BY B ORDER BY C ASC NULLS FIRST",
                "PARTITION BY C ORDER BY A DESC",
                "PARTITION BY C ORDER BY A DESC",
                "PARTITION BY C ORDER BY A ASC NULLS LAST",
                "PARTITION BY B ORDER BY C NULLS LAST, A DESC",
                "PARTITION BY C ORDER BY A DESC",
                "PARTITION BY B ORDER BY C NULLS LAST, A DESC",
                "PARTITION BY C ORDER BY A ASC NULLS LAST",
                "PARTITION BY B ORDER BY C ASC NULLS FIRST",
                "PARTITION BY B ORDER BY C NULLS LAST, A DESC",
            ],
            id="four_combinations",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_lead_lag_fusion(
    windows,
    spark_info,
    memory_leak_check,
):
    """Tests the lead and lag aggregation functions with various fusion cases.
    This also serves as more general test of if window fusion is working
    on the correct types of window combinations."""

    clauses = []
    for i in range(len(windows)):
        func = "LEAD" if i % 2 == 0 else "LAG"
        offset = f", {i % 4 + 1}" if i % 3 > 0 else ""
        default = f", -1" if i % 3 == 2 else ""
        calculation = f"{func}(A{offset}{default})"
        clauses.append(f"{calculation} OVER ({windows[i]}) AS C_{i}")
    query = f"SELECT A, B, C, {', '.join(clauses)} FROM table1"

    ctx = {
        "table1": pd.DataFrame(
            {
                "A": list(range(20)),
                "B": list("ABCDE") * 4,
                "C": (list("XXYXXYZYX") + [None]) * 2,
            }
        )
    }
    pandas_code = check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        only_jit_1DVar=True,
    )["pandas_code"]

    # Verify that fusion is working correctly. The term window_frames[1] refers
    # to how many distinct groupby-apply calls are expected after fusion.
    count_window_applies(pandas_code, len(set(windows)), ["LEAD", "LAG"])
