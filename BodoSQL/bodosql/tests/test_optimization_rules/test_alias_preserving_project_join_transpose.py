# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the alias_preserving_project_join_transpose optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        "Select A as X, B as Y from table1 join table2 on True",
        # again, the inner project gets preemptively optimized out.
        pytest.param(
            "Select D as X, B as Y from (SELECT A as B, B as C, C as D, D as A from table1 join table2 on True) as t2",
            marks=pytest.mark.slow,
        ),
        # In the test case, the values from table2 are not used. However, the join still needs to be performed,
        # so we have the correct values for A and B. In this situation, calcite will add a projection to the table
        # whose values are not used (table2), that projects one random column from the table. This column will
        # then be used to perform the join.
        "SELECT A as X from table1 join table2 on True",
        # This doesn't optimize out the original project, which means the filter can't be pushed into the join
        pytest.param(
            "SELECT Y as X from (SELECT A as Y, B as C FROM table1 join table2 on TRUE) where C = Y",
            marks=pytest.mark.slow,
        ),
    ]
)
def alias_preserving_project_join_transpose_queries(request):
    """fixture that supplies queries for the project join transpose rule

    This rule matches on any project that has a join for a child.

    On a successful match, the project will determine which columns originate from the left table of the join, and which originate in the right
    It will then create a projection on the left/right tables that returns only the columns used in the join. These projections on top of the
    left/right table will not perform the proper aliasing. Therefore if the original projection aliases, the rule will leave a projection on top of the join,
    that does the proper aliasing. Ideally, we would like the projections that are pushed onto the left/right table to do the proper aliasing, and ommit
    the projection on top, as the projection on top often prevents any other nodes from being pushed past the join.

    This is a rule we reimplemented ourselves, in order to deal with aliasing issues, Source code can be found at:
    ../calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/BodoSQLRules/ProjectUnaliasedRemoveRule.java
    """
    return request.param


@pytest.mark.slow
def test_alias_preserving_project_join_transpose(
    simple_join_fixture, spark_info, alias_preserving_project_join_transpose_queries
):
    """checking for alias bugs with project_unaliased_remove"""
    check_query(
        alias_preserving_project_join_transpose_queries,
        simple_join_fixture,
        spark_info,
        check_dtype=False,
    )
