"""
Test correctness of the filter optimization rules to extract common
predicates from a join.
"""

import pandas as pd
import pytest

from bodosql.tests.utils import check_query


@pytest.mark.slow
@pytest.mark.parametrize(
    "query_info",
    [
        # Simple condition to extract
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1)",
                1,
                1,
            ),
            id="query1",
        ),
        # Shared condition across 3 ORs
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1)",
                1,
                1,
            ),
            id="query2",
        ),
        # Shared condition that eliminates an OR
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1) OR (A > 1)",
                0,
                0,
            ),
            id="query3",
        ),
        # Shared condition that only selects a subset.
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (A > 1 AND A < C AND B < 5) OR (B < 5 AND A > 1) OR (B < C AND A > 1)",
                1,
                1,
            ),
            id="query4",
        ),
        # OR condition with no matches
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (C < B AND A < C) OR (B < 5 AND A > 1)",
                2,
                1,
            ),
            id="query5",
        ),
        # OR condition with no matches across all ORs. Any pair matches, but
        # nothing matches across all 3.
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (C > 10 AND A < C AND B < 5) OR (B < 5 AND A > 1) OR (A > 1 AND C > 10)",
                4,
                2,
            ),
            id="query6",
        ),
    ],
)
def test_logical_filter_rule(basic_df, spark_info, query_info, memory_leak_check):
    """
    Test that a common expression is extracted from an OR condition
    in a regular logical filter.
    """
    query, booland_count, boolor_count = query_info

    # Check the correctness of all queries. We check for the optimization
    # by validating the number of booland/| operations in the generated code.
    # TODO: Validate the actual generated plans.
    result1 = check_query(query, basic_df, spark_info, return_codegen=True)
    gen_code1 = result1["pandas_code"]
    assert (
        gen_code1.count("booland") == booland_count
    ), f"Expected {booland_count} booland after optimization"
    assert (
        gen_code1.count("boolor") == boolor_count
    ), f"Expected {boolor_count} boolor after optimization"


@pytest.mark.slow
@pytest.mark.parametrize(
    "query_info",
    [
        # Simple condition to extract
        pytest.param(
            (
                "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1)",
                0,
                1,
            ),
            id="query1",
        ),
        # Shared condition across 3 ORs
        pytest.param(
            (
                "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1) OR (t2.B < t1.C AND t1.A > 1)",
                0,
                2,
            ),
            id="query2",
        ),
        # Shared condition that eliminates an OR
        pytest.param(
            (
                "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1) OR (t1.A > 1)",
                0,
                0,
            ),
            id="query3",
        ),
        # Shared condition that only selects a subset.
        pytest.param(
            (
                "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C AND t2.B < 5) OR (t2.B < 5 AND t1.A > 1) OR (t2.B < t1.C AND t1.A > 1)",
                0,
                1,
            ),
            id="query4",
        ),
        # OR condition with no matches
        pytest.param(
            (
                "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.C < t2.B AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1)",
                0,
                1,
            ),
            id="query5",
        ),
        # OR condition with no matches across all ORs. Any pair matches, but
        # nothing matches across all 3.
        pytest.param(
            (
                "Select A, B FROM table1 WHERE (C > 10 AND A < C AND B < 5) OR (B < 5 AND A > 1) OR (A > 1 AND C > 10)",
                4,
                0,
            ),
            id="query6",
        ),
    ],
)
def test_join_filter_rule(spark_info, query_info, memory_leak_check):
    """
    Test that a common expression is extracted from an OR condition
    in a join expression.
    """
    query, booland_count, boolor_count = query_info

    ctx = {
        "TABLE1": pd.DataFrame(
            {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
        ),
        "TABLE2": pd.DataFrame(
            {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
        ),
    }

    result1 = check_query(query, ctx, spark_info, return_codegen=True)
    gen_code1 = result1["pandas_code"]
    assert (
        gen_code1.count("booland") == booland_count
    ), f"Expected {booland_count} booland after optimization"
    assert (
        gen_code1.count("|") == boolor_count
    ), f"Expected {boolor_count} | after optimization"
