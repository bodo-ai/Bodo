# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the project unaliased remove rule optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        # this one gets automatically optimized
        pytest.param(
            "Select X, Y, Z from (Select A as X, B as Y, C as Z from table1) as t2",
            marks=pytest.mark.slow,
        ),
        # these ones also use filter_into_join, in order to prevent the automatic optimization that occurs above
        "Select A as X,B as Y,C as D from (Select A as B, B as C, C as A from table1 where table1.A = 1) as t2 where t2.C = 2",
        pytest.param(
            "Select A as Y, W as B, B as A from (Select A, 1 as B, 2 as Y, 3 as W from table1 where table1.B = 1) as t2 where t2.Y = 2",
            marks=pytest.mark.slow,
        ),
    ]
)
def project_unaliased_remove_rule_queries(request):
    """fixture that supplies queries for the project unaliased remove rule tests

    This rule matches on any project that simply returns it's child without performing any aliasing. This is determined by two functions,
    RexUtil.isIdentity (a calcite function), and isAlias (a function we defined).

    On a successful match, the project will be removed from the tree.

    This is a rule we reimplemented ourselves, in order to deal with aliasing issues. The code can be found at:
    ../calcite_sql/bodosql-calcite-application/src/main/java/com/bodosql/calcite/application/BodoSQLRules/ProjectUnaliasedRemoveRule.java

    """
    return request.param


@pytest.mark.slow
def test_project_unaliased_remove_rule(
    basic_df, spark_info, project_unaliased_remove_rule_queries, memory_leak_check
):
    """checking for alias bugs with project_unaliased_remove"""
    check_query(
        project_unaliased_remove_rule_queries,
        basic_df,
        spark_info,
        check_dtype=False,
    )
