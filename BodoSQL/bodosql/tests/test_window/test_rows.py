"""
Tests the behavior of windowed aggregations functions with the OVER clause

Currently, all tests in this file only check the 1D Var case. This is to avoid
excessive memory leak, see [BS-530/BE-947]
"""


import numpy as np
import pandas as pd
import pytest
from bodosql.tests.test_window.window_common import (  # noqa
    null_respect_string,
    testing_locally,
)
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query


@pytest.fixture(
    params=[
        pytest.param(
            ("CURRENT ROW", "UNBOUNDED FOLLOWING"),
            id="suffix",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
        ),
        pytest.param(
            ("UNBOUNDED PRECEDING", "1 PRECEDING"),
            id="exclusive_prefix",
        ),
        pytest.param(
            ("1 PRECEDING", "1 FOLLOWING"),
            id="rolling_3",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
        ),
        pytest.param(
            ("CURRENT ROW", "1 FOLLOWING"),
            id="rolling2",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
        ),
        pytest.param(
            ("CURRENT ROW", "CURRENT ROW"),
            id="current_row",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
        ),
        pytest.param(
            ("1 FOLLOWING", "2 FOLLOWING"),
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="2_after",
        ),
        pytest.param(
            ("UNBOUNDED PRECEDING", "2 FOLLOWING"),
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="prefix_plus_2_after",
        ),
        pytest.param(
            ("3 PRECEDING", "UNBOUNDED FOLLOWING"),
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="suffix_plus_2_before",
        ),
    ]
)
def over_clause_bounds(request):
    """fixture containing the upper/lower bounds for the SQL OVER clause"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("MEDIAN", id="MEDIAN"),
        pytest.param("MAX", id="MAX"),
        pytest.param(
            "MIN",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="MIN",
        ),
        pytest.param("COUNT", marks=pytest.mark.slow, id="COUNT"),
        pytest.param("COUNT(*)", id="COUNT(*)"),
        pytest.param("SUM", marks=pytest.mark.slow, id="SUM"),
        pytest.param("AVG", id="AVG"),
        pytest.param("STDDEV", id="STDEV"),
        pytest.param(
            "STDDEV_POP",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="STDEV_POP",
        ),
        pytest.param("VARIANCE", marks=pytest.mark.slow, id="VARIANCE"),
        pytest.param("VAR_SAMP", marks=pytest.mark.slow, id="VAR_SAMP"),
        pytest.param("VARIANCE_SAMP", marks=pytest.mark.slow, id="VARIANCE_SAMP"),
        pytest.param(
            "VAR_POP",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="VAR_POP",
        ),
        pytest.param(
            "VARIANCE_POP",
            marks=[
                pytest.mark.skipif(not testing_locally, reason="Fix Memory Leak error"),
                pytest.mark.slow,
            ],
            id="VARIANCE_POP",
        ),
        pytest.param("FIRST_VALUE", id="FIRST_VALUE"),
        pytest.param("LAST_VALUE", id="LAST_VALUE"),
        pytest.param("ANY_VALUE", id="ANY_VALUE"),
    ]
)
def numeric_agg_funcs_subset(request):
    """subset of numeric aggregation functions, used for testing windowed behavior"""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("MAX", id="MAX"),
        pytest.param(
            "MIN",
            marks=pytest.mark.skipif(
                not testing_locally, reason="Fix Memory Leak error"
            ),
            id="MIN",
        ),
        pytest.param("COUNT", id="COUNT"),
        pytest.param("COUNT(*)", id="COUNT(*)"),
        pytest.param("FIRST_VALUE", id="FIRST_VALUE"),
        pytest.param("LAST_VALUE", id="LAST_VALUE"),
    ]
)
def non_numeric_agg_funcs_subset(request):
    """subset of non_numeric aggregation functions, used for testing windowed behavior"""
    return request.param


# TODO: fix memory leak issues with groupby apply, see [BS-530/BE-947]
def test_windowed_upper_lower_bound_numeric(
    bodosql_numeric_types,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    if numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    elif (
        numeric_agg_funcs_subset == "VARIANCE_POP"
        or numeric_agg_funcs_subset == "VARIANCE_SAMP"
    ):
        # spark doesn't support variance_pop/variance_samp
        agg_fn_call = f"VAR_{numeric_agg_funcs_subset[9:]}(A)"
    else:
        agg_fn_call = f"{numeric_agg_funcs_subset}(A)"

    window_ASC = f"(PARTITION BY B ORDER BY C ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY B ORDER BY C DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur
    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DESC FROM table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


# TODO: fix memory leak issues with groupby apply, see [BS-530/BE-947]
def test_windowed_upper_lower_bound_numeric_inside_case(
    bodosql_numeric_types,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    if numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    elif (
        numeric_agg_funcs_subset == "VARIANCE_POP"
        or numeric_agg_funcs_subset == "VARIANCE_SAMP"
    ):
        # spark doesn't support variance_pop/variance_samp
        agg_fn_call = f"VAR_{numeric_agg_funcs_subset[9:]}(A)"
    else:
        agg_fn_call = f"{numeric_agg_funcs_subset}(A)"

    window = f"(PARTITION BY B ORDER BY C ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur
    query = f"select A, B, C, CASE WHEN {agg_fn_call} OVER {window} > 0 THEN {agg_fn_call} OVER {window} ELSE -({agg_fn_call} OVER {window}) END FROM table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


# TODO: fix memory leak issues with groupby apply, see [BS-530/BE-947]
@pytest.mark.slow
def test_windowed_upper_lower_bound_timestamp(
    bodosql_datetime_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified on timestamp types"""

    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur.
    query = f"select A, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DESC FROM table1 ORDER BY C, A"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_windowed_upper_lower_bound_string(
    bodosql_string_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified on string types"""

    if non_numeric_agg_funcs_subset in ["MAX", "MIN"]:
        pytest.skip()
    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur.
    query = f"select A, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DESC FROM table1 ORDER BY C, A"

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
@pytest.mark.skip("Sorting doesn't work properly with binary data: BE-3279")
def test_windowed_upper_lower_bound_binary(
    bodosql_binary_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified on binary types"""
    if non_numeric_agg_funcs_subset in ["MAX", "MIN"]:
        pytest.skip()
    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur.
    query = f"select A, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DESC FROM table1 ORDER BY C, A"

    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=["A", "C"],
    )


@pytest.mark.slow
def test_windowed_upper_lower_bound_timedelta(
    bodosql_interval_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when both bounds are specified on timedelta types"""

    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur
    query = f"select A as A, C as C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DESC FROM table1 ORDER BY C, A"

    if (
        non_numeric_agg_funcs_subset == "COUNT"
        or non_numeric_agg_funcs_subset == "COUNT(*)"
    ):
        check_query(
            query,
            bodosql_interval_types,
            spark_info,
            sort_output=False,
            check_dtype=False,
            check_names=False,
            convert_columns_timedelta=["A", "C"],
            only_jit_1DVar=True,
        )
    else:
        # need to do a conversion, since spark timedeltas are converted to int64's
        check_query(
            query,
            bodosql_interval_types,
            spark_info,
            sort_output=False,
            check_dtype=False,
            check_names=False,
            convert_columns_timedelta=["A", "C", "WINDOW_AGG_ASC", "WINDOW_AGG_DESC"],
            only_jit_1DVar=True,
        )


@pytest.mark.skip("Defaults to Unbounded window in some case, TODO")
def test_windowed_only_upper_bound(
    basic_df,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when only the upper bound is specified"""

    if over_clause_bounds[0] == "1 FOLLOWING":
        # It seems like Calcite will rearange the window bounds to make sense, but mysql/spark don't?
        # However, we only see this issue for n = 3?
        pytest.skip("Skipped due to memory leak")

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur
    query = f"select A, B, C, {numeric_agg_funcs_subset}(A) OVER (PARTITION BY B ORDER BY C ASC ROWS {over_clause_bounds[0]}) as WINDOW_AGG_ASC, {numeric_agg_funcs_subset}(A) OVER (PARTITION BY B ORDER BY C DESC ROWS {over_clause_bounds[0]} ) as WINDOW_AGG_DESC FROM table1 ORDER BY B, C"

    # spark windowed min/max on integers returns an integer col.
    # pandas rolling min/max on integer series returns a float col
    # (and the method that we currently use returns a float col)
    cols_to_cast = [("WINDOW_AGG_ASC", "float64"), ("WINDOW_AGG_DESC", "float64")]

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        spark_output_cols_to_cast=cols_to_cast,
        only_jit_1DVar=True,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


@pytest.mark.skip("TODO")
def test_empty_window(
    basic_df,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when no bounds are specified"""
    query = f"select A, B, C, {numeric_agg_funcs_subset}(A) OVER () as WINDOW_AGG FROM table1"

    # spark windowed min/max on integers returns an integer col.
    # pandas rolling min/max on integer series returns a float col
    # (and the method that we currently use returns a float col)
    cols_to_cast = [("WINDOW_AGG", "float64")]

    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        spark_output_cols_to_cast=cols_to_cast,
        only_jit_1DVar=True,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


@pytest.mark.skip("TODO")
def test_nested_windowed_agg(
    basic_df,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests windowed aggregations works when performing aggregations, sorting by, and bounding by non constant values"""

    # doing an orderby and calculating extra rows in the query so it's easier to tell what the error is by visual comparison
    query = f"SELECT A, B, C, {numeric_agg_funcs_subset}(B) OVER (PARTITION BY A ORDER BY C ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) as WINDOW_AGG, CASE WHEN A > 1 THEN A * {numeric_agg_funcs_subset}(B) OVER (PARTITION BY A ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]}) ELSE -1 END AS NESTED_WINDOW_AGG from table1 ORDER BY A, C"

    # spark windowed min/max on integers returns an integer col.
    # pandas rolling min/max on integer series returns a float col
    # (and the method that we currently use returns a float col)
    cols_to_cast = [("WINDOW_AGG", "float64"), ("NESTED_WINDOW_AGG", "float64")]
    check_query(
        query,
        basic_df,
        spark_info,
        sort_output=False,
        check_dtype=False,
        check_names=False,
        spark_output_cols_to_cast=cols_to_cast,
        only_jit_1DVar=True,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


def test_row_number_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    """tests the row number aggregation function on numeric types"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_row_number_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """tests the row number aggregation function on datetime types"""

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_row_number_timedelta(bodosql_interval_types, spark_info, memory_leak_check):
    """tests the row_number aggregation functions on timedelta types"""

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) as ROW_NUM from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_timedelta=[
            "A",
            "B",
            "C",
        ],
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_row_number_string(bodosql_string_types, spark_info, memory_leak_check):
    """tests the row_number aggregation functions on timedelta types"""

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) as ROW_NUM from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_row_number_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """tests the row_number aggregation functions on binary types"""

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) as ROW_NUM from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=["A", "B", "C"],
    )


@pytest.mark.slow
def test_row_number_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """tests the row_number aggregation functions on boolean types"""

    query = f"select A, B, C, ROW_NUMBER() OVER (PARTITION BY B ORDER BY C) as ROW_NUM from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_nth_value_numeric(
    bodosql_numeric_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on numeric types"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_nth_value_datetime(
    bodosql_datetime_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on numeric types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_nth_value_timedelta(
    bodosql_interval_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on numeric types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
        convert_columns_timedelta=["A", "B", "C", "NTH_VALUE_1", "NTH_VALUE_3"],
    )


@pytest.mark.slow
def test_nth_value_string(
    bodosql_string_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on numeric types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_nth_value_binary(
    bodosql_binary_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on binary types"""
    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=["A", "B", "C", f"NTH_VALUE_1", f"NTH_VALUE_3"],
    )


@pytest.mark.slow
def test_nth_value_boolean(
    bodosql_boolean_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the Nth value aggregation functon on boolean types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    nth_val_queries = ", ".join(
        [
            f"NTH_VALUE(A, {x}) {null_respect_string} OVER {window} as NTH_VALUE_{name}"
            for x, name in [(1, "1"), (3, "3")]
        ]
    )
    query = f"select A, B, C, {nth_val_queries} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_first_last_value_binary(
    bodosql_binary_types,
    over_clause_bounds,
    spark_info,
    null_respect_string,
    memory_leak_check,
):
    """tests the first and last value aggregation functon on binary types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    query = f"select A, B, C, FIRST_VALUE(A) {null_respect_string} OVER {window} as FIRST_VALUE_A, LAST_VALUE(A) {null_respect_string} OVER {window} as LAST_VALUE_A from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=["A", "B", "C", "FIRST_VALUE_A", "LAST_VALUE_A"],
    )


@pytest.mark.slow
def test_first_last_value_boolean(
    bodosql_boolean_types,
    over_clause_bounds,
    null_respect_string,
    spark_info,
    memory_leak_check,
):
    """tests the first and last value aggregation functon on boolean types"""

    window = f"(PARTITION BY B ORDER BY C ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    query = f"select A, B, C, FIRST_VALUE(A) {null_respect_string} OVER {window} as FIRST_VALUE_A, LAST_VALUE(A) {null_respect_string} OVER {window} as LAST_VALUE_A from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


def test_ntile_numeric(
    bodosql_numeric_types,
    spark_info,
    memory_leak_check,
):
    """tests the ntile aggregation function with numeric data"""

    # remove once memory leak is resolved
    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_ntile_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    """tests the ntile aggregation function with datetime data"""

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_ntile_timedelta(bodosql_interval_types, spark_info, memory_leak_check):
    """tests the ntile aggregation with timedelta data"""

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_columns_timedelta=[
            "A",
            "B",
            "C",
        ],
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_ntile_string(bodosql_string_types, spark_info, memory_leak_check):
    """tests the ntile aggregation with string data"""

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.slow
def test_ntile_binary(bodosql_binary_types, spark_info, memory_leak_check):
    """tests the ntile aggregation with binary data"""

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        convert_columns_bytearray=["A", "B", "C"],
    )


@pytest.mark.slow
def test_ntile_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    """tests the ntile aggregation with boolean data"""

    fns = ", ".join(
        f"NTILE({x}) OVER (PARTITION BY B ORDER BY C) as NTILE_{x}" for x in [1, 3, 100]
    )

    query = f"select A, B, C, {fns} from table1 ORDER BY B, C"
    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "func",
    [
        "RANK",
        "DENSE_RANK",
        "PERCENT_RANK",
        "CUME_DIST",
    ],
)
@pytest.mark.parametrize("asc", ["ASC", "DESC"])
@pytest.mark.parametrize("nulls_pos", ["FIRST", "LAST"])
def test_rank_fns_complex(spark_info, asc, nulls_pos, func, memory_leak_check):
    tables = {}
    table0 = pd.DataFrame(
        {
            "A": np.repeat([1, 2, 3], 4),
            "B": [2, np.nan, 2, 4, np.nan, 3, 2, 5, 4, 2, np.nan, 2],
            "C": [1, 2, 1, 5, 3, 1, 4, 2, 5, 1, np.nan, 5],
        }
    )
    tables["table0"] = table0

    window = f"(PARTITION BY A ORDER BY B {asc} NULLS {nulls_pos}, C ASC NULLS LAST)"
    query = f"select A, B, C, {func}() OVER {window} from table0"

    # TODO: Currently, SparkSQL has a bug in how it orders by in within the window statement
    # (see https://bodo.atlassian.net/browse/BE-3091?focusedCommentId=16413)
    expected_output = None
    if asc == "ASC" and nulls_pos == "FIRST":
        expected_output = table0.copy()
        if func == "RANK":
            expected_output["rank"] = pd.Series([2, 1, 2, 4, 1, 3, 2, 4, 4, 2, 1, 3])
        elif func == "DENSE_RANK":
            expected_output["rank"] = pd.Series([2, 1, 2, 3, 1, 3, 2, 4, 4, 2, 1, 3])
        elif func == "PERCENT_RANK":
            expected_output["rank"] = (
                pd.Series([1, 0, 1, 3, 0, 2, 1, 3, 3, 1, 0, 2]) / 3
            )
        elif func == "CUME_DIST":
            expected_output["rank"] = (
                pd.Series([3, 1, 3, 4, 1, 3, 2, 4, 4, 2, 1, 3]) / 4
            )
    elif asc == "DESC" and nulls_pos == "LAST":
        expected_output = table0.copy()
        if func == "RANK":
            expected_output["rank"] = pd.Series([2, 4, 2, 1, 4, 2, 3, 1, 1, 2, 4, 3])
        elif func == "DENSE_RANK":
            expected_output["rank"] = pd.Series([2, 3, 2, 1, 4, 2, 3, 1, 1, 2, 4, 3])
        elif func == "PERCENT_RANK":
            expected_output["rank"] = (
                pd.Series([1, 3, 1, 0, 3, 1, 2, 0, 0, 1, 3, 2]) / 3
            )
        elif func == "CUME_DIST":
            expected_output["rank"] = (
                pd.Series([3, 4, 3, 1, 4, 2, 3, 1, 1, 2, 4, 3]) / 4
            )
    check_query(
        query,
        tables,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
    )


# NOTE: for the remaining rank tests on different dtypes, we only use RANK
# as they are all essentially the same under the hood.
def test_rank_numeric(bodosql_numeric_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rank_datetime(bodosql_datetime_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rank_timedelta(bodosql_interval_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_interval_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rank_string(bodosql_string_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select A, B, C, RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_string_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rank_binary(bodosql_binary_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_binary_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_rank_boolean(bodosql_boolean_types, spark_info, memory_leak_check):
    window = "(PARTITION BY B ORDER BY A DESC, C)"
    query = f"select RANK() OVER {window} from table1"

    check_query(
        query,
        bodosql_boolean_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT conditional_change_event(A) OVER (PARTITION BY B ORDER BY C NULLS FIRST) FROM table1",
                pd.Series([0, 0, 0, 0, 1, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0]),
            ),
            id="bool_string_int",
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(C % 2) OVER (PARTITION BY B ORDER BY C NULLS LAST) FROM table1",
                pd.Series([0, 1, 1, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]),
            ),
            id="int_string_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(B) OVER (PARTITION BY C % 5 ORDER BY C NULLS FIRST) FROM table1",
                pd.Series([0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 3]),
            ),
            id="string_int_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(C) OVER (PARTITION BY A ORDER BY C NULLS LAST) FROM table1",
                pd.Series([0, 1, 2, 3, 0, 1, 2, 3, 3, 3, 3, 3, 0, 1, 2, 3]),
            ),
            id="int_bool_int",
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(A) OVER (PARTITION BY B ORDER BY B) FROM table2",
                pd.Series([0] * 2 + [1] * 4 + [2] * 5 + [3] * 4 + [4] * 5),
            ),
            id="single_duplicates",
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(C) OVER (PARTITION BY B ORDER BY B) FROM table2",
                pd.Series([0] * 20),
            ),
            id="single_partition_all_null",
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(A) OVER (PARTITION BY B ORDER BY B) FROM table3",
                pd.Series([max(0, (i - 1) // 2) for i in range(200)]),
            ),
            id="longer_single_partition_unique_null_interleaved",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "SELECT conditional_change_event(A) OVER (PARTITION BY C ORDER BY C) FROM table3",
                pd.Series([0] * 200),
            ),
            id="longer_singleton_partitions",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_conditional_change_event(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series([True, False, None, True] * 4, dtype=pd.BooleanDtype()),
                "B": pd.Series((list("AABABCA") + [None]) * 2),
                "C": pd.Series(
                    [None if i % 4 == 0 else i for i in range(16, 0, -1)],
                    dtype=pd.Int32Dtype(),
                ),
            }
        ),
        "table2": pd.DataFrame(
            {
                "A": pd.Series(
                    [1] * 2 + [4] * 3 + [None] + [9] * 5 + [16, None] * 2 + [4] * 5,
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series(["A"] * 20),
                "C": pd.Series([None] * 20, dtype=pd.Int32Dtype()),
            }
        ),
        "table3": pd.DataFrame(
            {
                "A": pd.Series(
                    [i // 2 if i % 2 == 1 else None for i in range(200)],
                    dtype=pd.Int32Dtype(),
                ),
                "B": pd.Series(["A"] * 200),
                "C": pd.Series([chr(i) for i in range(32, 232)]),
            }
        ),
    }

    query, answer = args

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({0: answer}),
    )


@pytest.mark.parametrize(
    "windows",
    [
        pytest.param(
            ["PARTITION BY B ORDER BY C"] * 3,
            id="all_same",
        ),
        pytest.param(
            [
                "PARTITION BY B ORDER BY C",
                "PARTITION BY C ORDER BY B",
                "PARTITION BY B ORDER BY C",
            ],
            id="two_same",
        ),
    ],
)
def test_first_value_fusion(windows, basic_df, spark_info, memory_leak_check):
    import copy

    new_ctx = copy.deepcopy(basic_df)
    new_ctx["table1"]["D"] = new_ctx["table1"]["A"] + 10
    new_ctx["table1"]["E"] = new_ctx["table1"]["A"] * 2

    assert len(windows) == 3
    query = f"select FIRST_VALUE(A) OVER ({windows[0]}), FIRST_VALUE(D) OVER ({windows[1]}), FIRST_VALUE(E) OVER ({windows[2]}) from table1"

    codegen = check_query(
        query,
        new_ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_codegen=True,
    )["pandas_code"]

    # Check that the number of closures created corresponds to the number of
    # distinct windows
    expected_closures = len(set(windows))
    assert (
        codegen.count("def __bodo_dummy___sql_windowed_apply_fn") == expected_closures
    )


def test_first_value_optimized(spark_info, memory_leak_check):
    """
    Tests for an optimization with first_value when the
    window results in copying the first value of the group into
    every entry.
    """
    table = pd.DataFrame(
        {
            "A": [1, 2] * 10,
            "B": ["A", "B", "C", "D", "E"] * 4,
            "C": ["cq", "e22e", "r32", "#2431d"] * 5,
        }
    )
    ctx = {"table1": table}
    query = f"select FIRST_VALUE(C) OVER (PARTITION BY B ORDER BY A) as tmp from table1"
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
    )


def test_last_value_optimized(spark_info, memory_leak_check):
    """
    Tests for an optimization with last_value when the
    window results in copying the last value of the group into
    every entry.
    """
    table = pd.DataFrame(
        {
            "A": [1, 2] * 10,
            "B": ["A", "B", "C", "D", "E"] * 4,
            "C": ["cq", "e22e", "r32", "#2431d"] * 5,
        }
    )
    ctx = {"table1": table}
    query = f"select LAST_VALUE(C) OVER (PARTITION BY B ORDER BY A ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) as tmp from table1"
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
    )


def test_no_sort_permitted(spark_info, memory_leak_check):
    """tests that the row function works when not passed a sortstring"""

    table = pd.DataFrame(
        {
            "A": [1, 2] * 10,
            "B": ["A", "B", "C", "D", "E"] * 4,
            "C": ["cq", "e22e", "r32", "#2431d"] * 5,
        }
    )
    ctx = {"table1": table}

    query = "SELECT MAX(A) OVER (PARTITION BY B) from table1"
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                "SELECT conditional_true_event(A) OVER (PARTITION BY B ORDER BY C) FROM table1",
                pd.Series([1, 2, 2, 3, 3, 0, 1, 1, 2, 3, 0, 1, 2, 2, 3]),
            ),
            id="bool_string",
        ),
        pytest.param(
            (
                "SELECT conditional_true_event(A) OVER (PARTITION BY 0 ORDER BY C) FROM table1",
                pd.Series([1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 7, 7, 8, 9]),
            ),
            id="bool_singleton",
        ),
        pytest.param(
            (
                "SELECT conditional_true_event(A = B) OVER (PARTITION BY A ORDER BY C) FROM table2",
                pd.Series([1] * 6 + [2] * 3 + [0] * 14 + [1, 0]),
            ),
            id="strings_equal_string_int_larger",
        ),
        pytest.param(
            (
                "SELECT conditional_true_event(B <> LAG(B, 1)) OVER (PARTITION BY A ORDER BY B) FROM table2",
                pd.Series(
                    [
                        0,
                        0,
                        1,
                        1,
                        2,
                        2,
                        3,
                        3,
                        3,
                        0,
                        0,
                        1,
                        1,
                        1,
                        1,
                        2,
                        0,
                        0,
                        0,
                        1,
                        1,
                        0,
                        1,
                        2,
                        0,
                    ]
                ),
            ),
            id="equals_lag",
            marks=pytest.mark.skip(
                "[BE-3459] Nested window functions not supported yet"
            ),
        ),
    ],
)
def test_conditional_true_event(args, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series(
                    [True, False, None, True, True] * 3, dtype=pd.BooleanDtype()
                ),
                "B": pd.Series((list("AB") + [None]) * 5),
                "C": pd.Series(list(range(15))),
            }
        ),
        "table2": pd.DataFrame(
            {
                "A": pd.Series(list("AABAABCBAABCDCBAABCDECBDA")),
                "B": pd.Series(list("ABCDE") * 5),
                "C": pd.Series(list(range(25))),
            }
        ),
    }

    query, answer = args

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({0: answer}),
    )


def test_count_null(spark_info, memory_leak_check):
    """
    tests the null behavior of COUNT vs COUNT(*).

    Also doubles a test for having no orderby in window.
    """

    # Make sure each rank has some non-null data for type inference
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": [1, 1, 2, 2] * 4,
                "B": ["A", None, None, "B"] * 4,
            }
        )
    }

    query = "SELECT COUNT(B) OVER (PARTITION BY A), COUNT(*) OVER (PARTITION BY A) from table1"

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "SELECT COUNT_IF(A) OVER (PARTITION BY B) FROM table1",
            id="bool_string",
        ),
        pytest.param(
            "SELECT COUNT_IF(C % 2 = 1) OVER (PARTITION BY B) FROM table1",
            id="int_string",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(B = 'A' OR B = 'C') OVER (PARTITION BY C % 5) FROM table1",
            id="string_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(C = 0) OVER (PARTITION BY A), COUNT_IF(C < 4) OVER (PARTITION BY A) FROM table1",
            id="int_bool_multiple",
        ),
        pytest.param(
            "SELECT COUNT_IF(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) FROM table1",
            id="bool_string_int_sliding",
        ),
        pytest.param(
            "SELECT COUNT_IF(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM table1",
            id="bool_string_int_prefix_suffix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "SELECT COUNT_IF(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM table1",
            id="bool_string_int_cumulative_suffix",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_count_if(query, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series([True, False, None, True] * 25, dtype=pd.BooleanDtype()),
                "B": pd.Series(
                    ["A", "B", None, "A", "B", "C", "A", "A", "C", "A"] * 10
                ),
                "C": pd.Series(list(range(100)), dtype=pd.Int32Dtype()),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "select ANY_VALUE(A) OVER (PARTITION BY B ORDER BY C) from table1",
            id="float_string_int",
        ),
        pytest.param(
            "select ANY_VALUE(B) OVER (PARTITION BY C ORDER BY A ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) from table1",
            id="string_int_float_frame",
        ),
        pytest.param(
            "select ANY_VALUE(C) OVER (PARTITION BY A ORDER BY B ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING) from table1",
            id="int_float_string_frame",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "select ANY_VALUE(B) OVER (PARTITION BY A ORDER BY C) from table1",
            id="string_int_float",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_any_value(query, spark_info, memory_leak_check):
    ctx = {
        "table1": pd.DataFrame(
            {
                "A": pd.Series([i / 10 for i in range(10)] * 6),
                "B": pd.Series(list("ABCDEFGHIJ") * 6),
                "C": pd.Series(list(range(60))),
            }
        )
    }
    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=get_equivalent_spark_agg_query(query),
    )


# In cases with multiple correct answers, the chosen answer is the one that
# was encountered first in the overall array
@pytest.mark.parametrize(
    "data_col, bounds, answer",
    [
        pytest.param(
            pd.Series([0, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=pd.Int32Dtype()),
            ("UNBOUNDED PRECEDING", "CURRENT ROW"),
            pd.DataFrame(
                {0: pd.Series([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=pd.Int32Dtype())}
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
            pd.Series(list("abbaabacbb")),
            None,
            pd.DataFrame({0: pd.Series(["a"] * 5 + ["b"] * 5)}),
            id="string-noframe",
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
        ),
    ],
)
def test_mode(data_col, bounds, answer, spark_info, memory_leak_check):

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
        spark_info,
        check_dtype=False,
        check_names=False,
        sort_output=False,
        expected_output=answer,
    )


@pytest.mark.parametrize(
    "data_col, partition_col, answer",
    [
        pytest.param(
            pd.Series([0, 1, -1, None] * 4, dtype=pd.Int32Dtype()),
            pd.Series(["A", "B", "C", "D"] * 4),
            pd.Series([None, 0.25, 0.25, None] * 4),
            id="int32-groups_of_4",
        ),
        pytest.param(
            pd.Series([0, 1, -1, None] * 4, dtype=pd.Int32Dtype()),
            pd.Series(["A"] * 16),
            pd.Series([None] * 16),
            id="int32-single_partition",
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
def test_ratio_to_report(
    data_col, partition_col, answer, spark_info, memory_leak_check
):
    query = "SELECT A, B, RATIO_TO_REPORT(A) OVER (PARTITION BY B) FROM table1"

    assert len(data_col) == len(partition_col)
    ctx = {"table1": pd.DataFrame({"A": data_col, "B": partition_col})}

    check_query(
        query,
        ctx,
        spark_info,
        check_dtype=False,
        check_names=False,
        expected_output=pd.DataFrame({"A": data_col, "B": partition_col, "C": answer}),
    )


def test_row_number_orderby(datapath, memory_leak_check):
    """Test that row_number properly handles the orderby."""
    query = "select uuid, ROW_NUMBER() OVER(PARTITION BY store_id, ret_product_id ORDER BY last_seen DESC) as row_num from table1"

    parquet_path = datapath("sample-parquet-data/rphd_sample.pq")

    ctx = {
        "table1": pd.read_parquet(parquet_path)[
            ["uuid", "store_id", "ret_product_id", "last_seen"]
        ]
    }
    py_output = pd.DataFrame(
        {
            "uuid": [
                "67cd102b-e12f-49cb-88f5-c71d6be6642f",
                "ce4d3aa7-476b-4772-94b4-18224490c7a1",
                "bb9fb6cd-477d-4923-be3b-95615bbec5a5",
                "2adcb7de-464f-4c60-81a3-58dff3f8b1c9",
                "465cbfbb-c4c9-4837-83c7-c6be96597ca4",
                "fd5db816-902e-485b-b52d-a094709439a4",
            ],
            "row_num": [3, 5, 1, 6, 4, 2],
        }
    )
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
    )
