# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the filter into join optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        "SELECT A, B, C, D FROM table1 join table2 on TRUE where A = B",
        "SELECT A, B, C, D FROM (SELECT * FROM table1 join table2 on TRUE) where A = B",
        pytest.param(
            "SELECT A as B, B as C FROM (SELECT table1.A, table1.D, table2.B, table2.C from table1 join table2 on TRUE) WHERE A = D and B = C",
            marks=pytest.mark.slow,
        ),
    ]
)
def filter_into_join_queries(request):
    """fixture that supplies queries for the Flter Into Join tests

    This rule matches on a filter whose input is a join.

    This rule will push down all the conditions of the filter into the condition of the join itself, and, if possible, into each of the child nodes.

    Source code here:
    https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/rel/rules/FilterJoinRule.java#L333

    """
    return request.param


@pytest.mark.slow
def test_filter_into_join(simple_join_fixture, spark_info, filter_into_join_queries):
    """checks for bugs with the Filter into Join rule"""
    check_query(
        filter_into_join_queries,
        simple_join_fixture,
        spark_info,
    )
