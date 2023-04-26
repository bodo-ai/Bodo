# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL queries containing qualify clauses in BodoSQL
"""

import os

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.test_window.test_lead_lag import lead_or_lag  # noqa
from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query

import bodo

# [BE-3894] TODO: refactor this file like how test_rows.py was refactored
# for window fusion


# Helper environment variable to allow for testing locally, while avoiding
# memory issues on CI
testing_locally = os.environ.get("BODOSQL_TESTING_LOCALLY", False)


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


def test_QUALIFY_no_bounds(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    A test to ensure qualify works for window functions that do not have specified bounds
    """

    df_dtype = bodosql_numeric_types["table1"]["A"].dtype
    if not (
        testing_locally
        or np.issubdtype(df_dtype, np.float64)
        or np.issubdtype(df_dtype, np.int64)
    ):
        pytest.skip("Skipped due to memory leak")

    spark_subquery = "SELECT A, B, C, MAX(A) OVER (PARTITION BY C ORDER BY B) as window_val from table1"
    bodosql_query = "SELECT * from table1 QUALIFY MAX(A) OVER (PARTITION BY C ORDER BY B) > 1 ORDER BY A, B, C"
    spark_query = (
        f"SELECT A, B, C from ({spark_subquery}) where window_val > 1 ORDER BY A, B, C"
    )

    res = check_query(
        bodosql_query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_seq_dataframe=True,
    )

    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter actually leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_numeric_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


def test_QUALIFY_upper_lower_bound_numeric(
    bodosql_numeric_types,
    numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """
    Tests qualify on numeric row functions
    Largely a copy of test_windowed_upper_lower_bound_numeric, but using qualify
    """

    if numeric_agg_funcs_subset in ("STDDEV", "VARIANCE", "VAR_SAMP", "VARIANCE_SAMP"):
        scalar_filter_val = "0"
    else:
        scalar_filter_val = "1"

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
    else:
        agg_fn_call = f"{numeric_agg_funcs_subset}(A)"

    window_ASC = f"(PARTITION BY B ORDER BY C ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY B ORDER BY C DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur
    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC FROM table1 QUALIFY WINDOW_AGG_ASC > {scalar_filter_val} OR {agg_fn_call} OVER {window_DESC} >= {scalar_filter_val} ORDER BY B, C"

    spark_subquery = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DES FROM table1"
    spark_query = f"select A, B, C, WINDOW_AGG_ASC FROM ({spark_subquery}) where WINDOW_AGG_ASC > {scalar_filter_val} OR WINDOW_AGG_DES >= {scalar_filter_val} ORDER BY B, C"

    res = check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_seq_dataframe=True,
    )

    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter acutally leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_numeric_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


@pytest.mark.slow
def test_QUALIFY_upper_lower_bound_timestamp(
    bodosql_datetime_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests Qualify works with windowed agregations when both bounds are specified on timestamp types

    Largley a copy of test_windowed_upper_lower_bound_timestamp in test_rows, that makes use of QUALIFY
    """

    # COUNT returns an integer value, all others return
    # timestamp. Make sure we use the correct type for comparison in the qualify clause
    if non_numeric_agg_funcs_subset.startswith("COUNT"):
        scalar_filter_val = "2"
    elif non_numeric_agg_funcs_subset == "MAX":
        # needed so filtering actually occurs
        scalar_filter_val = "TIMESTAMP '2020-12-01'"
    else:
        scalar_filter_val = "TIMESTAMP '2007-01-01'"

    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A DESC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur.
    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC FROM table1 QUALIFY WINDOW_AGG_ASC > {scalar_filter_val} OR {agg_fn_call} OVER {window_DESC} > {scalar_filter_val} ORDER BY C, A"

    spark_subquery = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DES FROM table1"
    spark_query = f"select A, B, C, WINDOW_AGG_ASC FROM ({spark_subquery}) where WINDOW_AGG_ASC > {scalar_filter_val} OR WINDOW_AGG_DES > {scalar_filter_val} ORDER BY C, A"

    res = check_query(
        query,
        bodosql_datetime_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_seq_dataframe=True,
    )

    # Only check when we have one rank, so we know we're checking the replicated dataframe
    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter acutally leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_datetime_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


@pytest.mark.slow
def test_QUALIFY_upper_lower_bound_string(
    bodosql_string_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """
    Tests Qualify works with windowed agregations when both bounds are specified on string types
    Largley a copy of test_windowed_upper_lower_bound_string in test_rows, that makes use of QUALIFY
    """

    # COUNT returns an integer value, all others return
    # string. Make sure we use the correct type for comparison in the qualify clause
    if non_numeric_agg_funcs_subset.startswith("COUNT"):
        scalar_filter_val = "2"
    elif non_numeric_agg_funcs_subset == "LAST_VALUE":
        scalar_filter_val = "'SpEaK'"
    else:
        scalar_filter_val = "'HeLlLllo'"

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
    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC FROM table1 QUALIFY WINDOW_AGG_ASC > {scalar_filter_val} OR {agg_fn_call} OVER {window_DESC} > {scalar_filter_val} ORDER BY C, A"

    spark_subquery = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DES FROM table1"
    spark_query = f"select A, B, C, WINDOW_AGG_ASC FROM ({spark_subquery}) where WINDOW_AGG_ASC > {scalar_filter_val} OR WINDOW_AGG_DES > {scalar_filter_val} ORDER BY C, A"

    res = check_query(
        query,
        bodosql_string_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        sort_output=False,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_seq_dataframe=True,
    )

    # Only check when we have one rank, so we know we're checking the replicated dataframe
    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter acutally leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_string_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


@pytest.mark.slow
def test_windowed_upper_lower_bound_binary(
    bodosql_binary_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """Tests Qualify works for window functions when both bounds are specified on binary types

    Mostly a copy of test_windowed_upper_lower_bound_binary from test_rows.py
    """

    cols_to_convert = ["A", "B", "C"]

    # COUNT returns an integer value, all others return
    # a binary type. Make sure we use the correct type for comparison in the qualify clause
    if non_numeric_agg_funcs_subset.startswith("COUNT"):
        scalar_filter_val = "1"
    else:
        cols_to_convert.append("WINDOW_AGG_ASC")
        pytest.skip("Currently unable to specify a binary literal in BodoSQL")

    if non_numeric_agg_funcs_subset in ["MAX", "MIN"]:
        pytest.skip()
    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # manually specified null order to avoid differnce with pySpark: https://bodo.atlassian.net/browse/BS-585
    window_ASC = f"(PARTITION BY C ORDER BY A ASC NULLS FIRST ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A DESC NULLS LAST ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC FROM table1 QUALIFY WINDOW_AGG_ASC > {scalar_filter_val} OR {agg_fn_call} OVER {window_DESC} > {scalar_filter_val}"

    spark_subquery = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DES FROM table1"
    spark_query = f"select A, B, C, WINDOW_AGG_ASC FROM ({spark_subquery}) where WINDOW_AGG_ASC > {scalar_filter_val} OR WINDOW_AGG_DES > {scalar_filter_val}"

    res = check_query(
        query,
        bodosql_binary_types,
        spark_info,
        equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
        sort_output=True,  # For binary, we have to sort the output dataframe in python, as the behavior for sorting behavior for spark differs: BE-3279
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
        return_seq_dataframe=True,
        convert_columns_bytearray=cols_to_convert,
    )

    # Only check when we have one rank, so we know we're checking the replicated dataframe
    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter acutally leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_binary_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


@pytest.mark.slow
def test_QUALIFY_upper_lower_bound_timedelta(
    bodosql_interval_types,
    non_numeric_agg_funcs_subset,
    over_clause_bounds,
    spark_info,
    memory_leak_check,
):
    """
    Tests QUALIFY works on windowed agregations when both bounds are specified on timedelta types

    Largely a copy of test_windowed_upper_lower_bound_timedelta from test_rows
    """

    if non_numeric_agg_funcs_subset.startswith("COUNT"):
        scalar_filter_val = "2"
        spark_scalar_filter_val = scalar_filter_val
    else:
        scalar_filter_val = "INTERVAL '3' DAYS"
        spark_scalar_filter_val = (
            "259200000000000"  # needed as sparksql converts input timedeltas to ints
        )

    if non_numeric_agg_funcs_subset == "COUNT(*)":
        agg_fn_call = "COUNT(*)"
    else:
        agg_fn_call = f"{non_numeric_agg_funcs_subset}(A)"

    # Switched partition/sortby to avoid null
    window_ASC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"
    window_DESC = f"(PARTITION BY C ORDER BY A ASC ROWS BETWEEN {over_clause_bounds[0]} AND {over_clause_bounds[1]})"

    # doing an orderby in the query so it's easier to tell what the error is by visual comparison
    # should an error occur.
    query = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC FROM table1 QUALIFY WINDOW_AGG_ASC > {scalar_filter_val} OR {agg_fn_call} OVER {window_DESC} > {scalar_filter_val} ORDER BY C, A"

    spark_subquery = f"select A, B, C, {agg_fn_call} OVER {window_ASC} as WINDOW_AGG_ASC, {agg_fn_call} OVER {window_DESC} as WINDOW_AGG_DES FROM table1"
    spark_query = f"select A, B, C, WINDOW_AGG_ASC FROM ({spark_subquery}) where WINDOW_AGG_ASC > {spark_scalar_filter_val} OR WINDOW_AGG_DES > {spark_scalar_filter_val} ORDER BY C, A"

    if (
        non_numeric_agg_funcs_subset == "COUNT"
        or non_numeric_agg_funcs_subset == "COUNT(*)"
    ):
        res = check_query(
            query,
            bodosql_interval_types,
            spark_info,
            equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
            sort_output=False,
            check_dtype=False,
            check_names=False,
            convert_columns_timedelta=["A", "B", "C"],
            only_jit_1DVar=True,
            return_seq_dataframe=True,
        )
    else:
        # need to do a conversion, since spark timedeltas are converted to int64's
        res = check_query(
            query,
            bodosql_interval_types,
            spark_info,
            equivalent_spark_query=get_equivalent_spark_agg_query(spark_query),
            sort_output=False,
            check_dtype=False,
            check_names=False,
            convert_columns_timedelta=["A", "B", "C", "WINDOW_AGG_ASC"],
            only_jit_1DVar=True,
            return_seq_dataframe=True,
        )

    if bodo.get_size() == 1:
        out_df = res["output_df"]
        # ensure that the qualify filter acutally leaves some output, so we can actually test correctness
        assert len(out_df) > 0, "Qualify filtered output is empty"
        # ensure that the qualify filter actually filters some of the output, so we can actually test correctness
        assert len(out_df) < len(
            bodosql_interval_types["table1"]
        ), f"Qualify filtered nothing. Output DF: {out_df}"


"""
    The following tests ensures that QUALIFY is evaluated in the proper ordering.
    From snowflake docs, the evaluation order of a query should be as follows
        From
        Where
        Group by
        Having
        Window
        QUALIFY
        Distinct
        Order by
        Limit

    Note: We can't test order by, as, in bodosql, if we don't provide an ordering in the WINDOW clause
    itself, then the ordering is undefined when evaluating the window function.
"""


def test_QUALIFY_eval_order_WHERE(spark_info, memory_leak_check):
    """Ensures that WHERE is evaluated before QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"table1": df}

    # If QUALIFY is evaluated first, MAX(A) OVER (PARTITION BY B) = 3 everywhere,
    # as B is one group. If evaluated after where, MAX(A) OVER (PARTITION BY B)
    # = 2 everywhere
    query = "SELECT A from table1 where A < 3 QUALIFY MAX(A) OVER (PARTITION BY B) = 3"

    expected_output = pd.DataFrame(
        {
            "A": [],
        }
    )

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_eval_order_GROUP_BY_HAVING(spark_info, memory_leak_check):

    """Ensures that Group by and HAVING are evaluated before QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1, 2, 3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"table1": df}

    # If QUALIFY is evaluated first, MAX(A) OVER (PARTITION BY B) = 3 everywhere,
    # as B is one group. If evaluated after HAVING/GROUP BY, MAX(A) OVER (PARTITION BY B)
    # = 2 everywhere, as the HAVING clause will have eliminated the A=3 rows

    # (GROUP BY A, B HAVING MAX(A) > 3) is just a fancy way of doing WHERE A > 3 in this case
    query = "SELECT A from table1 GROUP BY A, B HAVING MAX(A) > 3 QUALIFY MAX(A) OVER (PARTITION BY B) = 3"

    expected_output = pd.DataFrame(
        {
            "A": [],
        }
    )

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_eval_order_DISTINCT(spark_info, memory_leak_check):

    """Ensures that DISTINCT is evaluated after QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1] * 4 + [2] * 4 + [3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"table1": df}

    """
    If distinct is evaluated before QUALIFY, the dataframe will look something like:
    pd.DataFrame({
        "A" : [1,2,3]
    })
    And then the qualify filtering will reduce it to an empty dataframe.

    If qualify is evaluated first, then the dataframe will be unchanged,
    and the distinct clause will result in an output of
    pd.DataFrame({
        "A" : [1,2,3]
    })

    """

    query = "SELECT DISTINCT A from table1 QUALIFY COUNT(A) OVER (PARTITION BY B) > 3"

    expected_output = pd.DataFrame({"A": [1, 2, 3]})

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_eval_order_LIMIT(spark_info, memory_leak_check):

    """Ensures that LIMIT is evaluated after QUALIFY"""
    df = pd.DataFrame(
        {
            "A": [1] * 4 + [2] * 4 + [3] * 4,
            "B": [1] * 12,
        }
    )

    ctx = {"table1": df}

    """If limit is evalued first, we expect:
     df = pd.DataFrame({
        "A" : [1] * 3,
        "B" : [1] * 3,
    })
    and the qualify Count filtering will reduce it to an empty df.

    if qualify is evaluated first, we expect the output after evaluating the
    qualify to be unchanged, and then the limit will result in the output:
    df = pd.DataFrame({
        "A" : [1] * 3,
        "B" : [1] * 3,
    })

    """
    query = "SELECT A from table1 QUALIFY MAX(A) OVER (PARTITION BY B) <= 3 LIMIT 3"

    expected_output = pd.DataFrame({"A": [1] * 3})

    check_query(
        query,
        ctx,
        spark_info,
        expected_output=expected_output,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


def test_QUALIFY_nested_queries(spark_info, memory_leak_check):
    """stress test to ensure that qualify works with nested subqueries"""

    table1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7] * 3,
            "B": [1, 1, 2, 2, 3, 3, 4] * 3,
            "C": [1, 1, 1, 2, 2, 3, 3] * 3,
        }
    )

    ctx = {"table1": table1}

    bodosql_query1 = f"SELECT ROW_NUMBER() OVER (PARTITION BY B ORDER BY C, B, A) as w FROM table1 QUALIFY w < 10"
    bodosql_query2 = f"SELECT MAX(A) over (PARTITION BY C ORDER BY B, C, A) as x FROM table1 QUALIFY x in ({bodosql_query1})"
    bodosql_query = bodosql_query2

    spark_query_1 = f"SELECT * FROM (SELECT ROW_NUMBER() OVER (PARTITION BY B ORDER BY C, B, A) as w FROM table1) WHERE w < 10"
    spark_query_2 = f"SELECT * FROM (SELECT MAX(A) over (PARTITION BY C ORDER BY B, C, A) as x FROM table1 ) WHERE x in ({spark_query_1})"
    spark_query = spark_query_2

    check_query(
        bodosql_query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        sort_output=True,
        check_dtype=False,
        check_names=False,
        only_jit_1DVar=True,
    )


@pytest.mark.tz_aware
def test_qualify_tz_aware(memory_leak_check):
    """Tests that qualify is supported with tz-aware data."""
    query = "SELECT A, MIN(A) over (PARTITION BY C ORDER BY B ASC ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) as x FROM table1 QUALIFY x > A"
    tz = "US/Pacific"
    df = pd.DataFrame(
        {
            "A": [
                pd.Timestamp(year=2022, month=1, day=1, tz=tz),
                pd.Timestamp(year=2022, month=1, day=2, tz=tz),
                pd.Timestamp(year=2022, month=11, day=1, tz=tz),
                None,
                pd.Timestamp(year=2022, month=1, day=15, tz=tz),
            ]
            * 4,
            "B": np.arange(20),
            "C": ["left", "right", "left", "left"] * 5,
        }
    )
    ctx = {"table1": df}
    # Compute the expected output of the max. To do this we leverage
    # that the window is the current previous and next value and the
    # Order by keeps the DataFrame in order.
    x_list = []
    right_offset = 4
    right_modulo = 1
    for i in range(len(df)):
        # Determine if we are grouped by left or right
        group = df["C"].iat[i]
        if group == "right":
            prev = i - right_offset
            next = i + right_offset
        else:
            prev = i - 1
            next = i + 1
            if prev % right_offset == right_modulo:
                prev -= 1
            if next % right_offset == right_modulo:
                next += 1
        options = []
        if prev > 0:
            options.append(df["A"].iat[prev])
        options.append(df["A"].iat[i])
        if next < len(df):
            options.append(df["A"].iat[next])
        # Remove NA values as options
        options = [x for x in options if pd.notna(x)]
        x_list.append(min(options))
    py_output = pd.DataFrame(
        {
            "A": df["A"],
            "x": x_list,
        }
    )
    # Now apply the qualify filter
    filter = py_output["x"] > py_output["A"]
    py_output = py_output[filter]
    check_query(
        query,
        ctx,
        None,
        expected_output=py_output,
        only_jit_1DVar=True,
    )
