# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Test correctness of the project merge optimization rule used by BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        # This automatically gets optimized out
        pytest.param(
            "SELECT A as X, A, D as Y from (SELECT A as D, D as A from table1)",
            marks=pytest.mark.slow,
        ),
        # I think this invokes the rule, but I'm uncertain, as I can't see intermediate steps.
        # In this case, even if aliasing was wrong, wouldn't matter because of how we handle
        # project join transpose
        "SELECT A as Y from (SELECT * FROM table1 join table2 on TRUE) where A = B",
    ]
)
def project_merge_queries(request):
    """fixture that supplies queries for the project merge test

    This rule matches on any project, whose input is a exactly one other project.

    This rule will do nothing to projections that are simply an identity,
    as it expects ProjectRemoveRule (another optimizaton(the one we replaced)) to remove those projections.

    This runs by default, as even in the unoptimized case we still seem to see this optimization occur. So, the only way
    to get this optimization to occur is in conjunction with project join transpose

    Source code here:
    https://github.com/apache/calcite/blob/master/core/src/main/java/org/apache/calcite/rel/rules/ProjectMergeRule.java
    """
    return request.param


@pytest.mark.slow
def test_project_merge(simple_join_fixture, spark_info, project_merge_queries):
    """checks for bugs with project_merge"""
    check_query(
        project_merge_queries,
        simple_join_fixture,
        spark_info,
    )
