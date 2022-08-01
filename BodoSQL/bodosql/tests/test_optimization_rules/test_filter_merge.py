# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Test correctness of the filter merge optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        "Select B as X, C as Y,D as Z from (Select A, B, C from table1 where table1.A = 1) as t2 where t2.B = 2",
        pytest.param(
            "Select X as U, Y as W, Z as Q from (Select A as X, B as Y, C as Z from table1 where table1.B = 3) as t2 where t2.Z = 3",
            marks=pytest.mark.slow,
        ),
    ]
)
def filter_merge_queries(request):
    """fixture that supplies queries for the filter merge test

    This rule matches on any filter, whose input is a exactly one filter.

    It will combine the two filters into one filter, whose condition
    is an equal to both of the original filter conditions.

    Source code can be found here:
    https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/rel/rules/FilterMergeRule.java

    """
    return request.param


@pytest.mark.slow
def test_filter_merge(basic_df, spark_info, filter_merge_queries):
    """checking for alias bugs with filter_merge"""
    # need to have dtype = false, due to reduce Expr rule stuff where entire tables get optimized out
    check_query(
        filter_merge_queries,
        basic_df,
        spark_info,
        check_dtype=False,
    )
