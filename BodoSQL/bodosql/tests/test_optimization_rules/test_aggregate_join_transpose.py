# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the aggregate join transpose optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        # NOTE, due to project join implementation, I can't alias freely within the aggregation select, without causing the rule
        # to not be applied
        # These tests apply the rule, but don't do aliasing
        # tests just groupby
        "SELECT t3.B, t3.A from (SELECT t2.B, t1.A from (SELECT table1.A from table1) as t1, (SELECT table2.B from table2) as t2) as t3 GROUP BY t3.A, t3.B",
        # tests max aggregation function
        "SELECT Max(t3.B) as m, t3.A from (SELECT t1.A, t2.B from (SELECT table1.A from table1) as t1, (SELECT table2.B from table2) as t2) as t3 GROUP BY t3.A",
        # tests that distinct doesn't cause any issues (It should simply cause the rule to not be applied)
        "SELECT DISTINCT Max(t3.B) as m, t3.A from (SELECT t2.B, t1.A from (SELECT table1.A from table1) as t1, (SELECT table2.B from table2) as t2) as t3 GROUP BY t3.A",
    ]
)
def aggregate_join_transpose_queries(request):
    """fixture that supplies queriees for the aggregate join transpose rule

    This rule matches on an Aggregation whose inputs consist of exactly one Join node. The aggregations must also
    be allowed to be transposed. In the default configuration (the one we are using on master), pushing down aggregation functions is
    not supported, so the agg call list must be empty (meaning, it must be only a groupby), the aggregation must belong to the simple aggregation
    group (https://github.com/apache/calcite/blob/e5477e7cda4ed9747d970b830d1fe9c53c49f2f3/core/src/main/java/org/apache/calcite/rel/core/Aggregate.java#L473).
    the aggregation must support splitting, the aggregation cannot be distinct, and cannot have a filter.
    The Join must also an inner join, and the join condition must be an equality.

    In the extended configuration, all of the above holds, but the agg function can be not empty, and the aggregation functions will be pushed.

    This rule will perform the aggregation on each of the input tables before the join. I am actually very confused as to what this
    rule accomplishes in the default case. Given that the agg call list must be empty, pushing just a groupby shouldn't really do anything, and
    I don't even think it does that, due to a bug workaround in the source code. For the purposes of testing, I've used the extended version
    of the rule, which does push agg functions.

    Also, noticably, due to our implementation of Project Join transpose, if the select statment doing the aggregation does any amount of aliasing,
    this rule will not get applied, due to the header project that performs the aliasing.

    Source code here:
    https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/rel/rules/AggregateJoinTransposeRule.java

    """
    return request.param


@pytest.mark.slow
def test_aggregate_join_transpose(
    simple_join_fixture, spark_info, aggregate_join_transpose_queries
):
    """checks for bugs with the Aggregate Join transpose rule"""
    check_query(
        aggregate_join_transpose_queries,
        simple_join_fixture,
        spark_info,
        check_dtype=False
    )
