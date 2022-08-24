# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the filter aggregate transpose optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        # this test get auto-optimized out
        pytest.param(
            "SELECT D as E, m as M from (SELECT B as D, Max(A) as m from table1 GROUP BY B) as t2 where t2.D= 0",
            marks=pytest.mark.slow,
        ),
        "SELECT D as E, m as M from (SELECT B as D, Max(A) as m from table1 GROUP BY B) as t2 where t2.m = 0",
        # test if agg is just a groupby
        "SELECT D as E FROM (SELECT A as D, B as C from table1 GROUP BY A, B) as t3 where t3.D = 0",
        # this one inserts an additional projection to reorder... which causes the filter to not get pushed
        pytest.param(
            "SELECT D as E FROM (SELECT B as C, A as D from table1 GROUP BY A, B) as t3 where t3.D = 0",
            marks=pytest.mark.slow,
        ),
    ]
)
def filter_aggregate_transpose_queries(request):
    """fixture that supplies queries for the filter_aggregate_transpose rule

    This rule matches on any Filter, whose input is exactly one aggregation.

    This rule will try to push as many filter conditions past an aggregation as it is able. It will not,
    for example, push any filters on columns that get aggregated, as such changes may effect the results of the
    aggregation.

    For example, assume a filter, "A" is on top of a aggregation "B". A has two conditions,
    one of which can be pushed, and the other can't. The end result after this optimization will
    be one filter, "A0" with the unpused condition on top of aggregation "B" whose input is filtered
    by "A1", the filter with the pushed condition.


    Source code here:
    https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/rel/rules/FilterAggregateTransposeRule.java
    """
    return request.param


@pytest.mark.slow
def test_filter_aggregate_transpose(
    basic_df, spark_info, filter_aggregate_transpose_queries
):
    """checks for bugs with filter agregate transpose"""
    check_query(
        filter_aggregate_transpose_queries,
        basic_df,
        spark_info,
    )
