# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of the filter optimization rules to extract common
predicates from a join.
"""
import pandas as pd
from bodosql.tests.utils import check_query


def test_logical_filter_rule(basic_df, spark_info, memory_leak_check):
    """
    Test that a common expression is extracted from an OR condition
    in a regular logical filter.
    """
    # Simple condition to extract
    query1 = "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1)"
    # Shared condition across 3 ORs
    query2 = "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1) OR (B < C AND A > 1)"
    # Shared condition that eliminates an OR
    query3 = "Select A, B FROM table1 WHERE (A > 1 AND A < C) OR (B < 5 AND A > 1) OR (A > 1)"
    # Shared condition that only selects a subset.
    query4 = "Select A, B FROM table1 WHERE (A > 1 AND A < C AND B < 5) OR (B < 5 AND A > 1) OR (B < C AND A > 1)"
    # OR condition with no matches
    query5 = "Select A, B FROM table1 WHERE (C < B AND A < C) OR (B < 5 AND A > 1)"
    # OR condition with no matches across all ORs. Any pair matches, but
    # nothing matches across all 3.
    query6 = "Select A, B FROM table1 WHERE (C > 10 AND A < C AND B < 5) OR (B < 5 AND A > 1) OR (A > 1 AND C > 10)"
    # Check the correctness of all queries. We check for the optimization
    # by validating the number of booland/| operations in the generated code.
    # TODO: Validate the actual generated plans.
    result1 = check_query(query1, basic_df, spark_info, return_codegen=True)
    gen_code1 = result1["pandas_code"]
    assert gen_code1.count("booland") == 1, "Expected 1 booland after optimization"
    assert gen_code1.count("|") == 1, "Expected 1 | after optimization"
    result2 = check_query(query2, basic_df, spark_info, return_codegen=True)
    gen_code2 = result2["pandas_code"]
    assert gen_code2.count("booland") == 1, "Expected 1 booland after optimization"
    assert gen_code2.count("|") == 2, "Expected 2 | after optimization"
    result3 = check_query(query3, basic_df, spark_info, return_codegen=True)
    gen_code3 = result3["pandas_code"]
    assert gen_code3.count("booland") == 1, "Expected 1 booland after optimization"
    assert gen_code3.count("|") == 1, "Expected 1 | after optimization"
    result4 = check_query(query4, basic_df, spark_info, return_codegen=True)
    gen_code4 = result4["pandas_code"]
    assert gen_code4.count("booland") == 2, "Expected 2 booland after optimization"
    assert gen_code4.count("|") == 2, "Expected 2 | after optimization"
    result5 = check_query(query5, basic_df, spark_info, return_codegen=True)
    gen_code5 = result5["pandas_code"]
    assert gen_code5.count("booland") == 2, "Expected 2 booland after no-optimization"
    assert gen_code5.count("|") == 1, "Expected 1 | after no-optimization"
    result6 = check_query(query6, basic_df, spark_info, return_codegen=True)
    gen_code6 = result6["pandas_code"]
    assert gen_code6.count("booland") == 4, "Expected 4 booland after no-optimization"
    assert gen_code6.count("|") == 2, "Expected 2 | after no-optimization"


def test_join_filter_rule(spark_info, memory_leak_check):
    """
    Test that a common expression is extracted from an OR condition
    in a join expression.
    """
    ctx = {
        "table1": pd.DataFrame(
            {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
        ),
        "table2": pd.DataFrame(
            {"A": [1, 2, 3] * 4, "B": [4, 5, 6, 7] * 3, "C": [7, 8, 9, 10, 11, 12] * 2}
        ),
    }

    # Simple condition to extract
    query1 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1)"
    # Shared condition across 3 ORs
    query2 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1) OR (t2.B < t1.C AND t1.A > 1)"
    # Shared condition that eliminates an OR
    query3 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1) OR (t1.A > 1)"
    # Shared condition that only selects a subset.
    query4 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.A > 1 AND t2.A < t1.C AND t2.B < 5) OR (t2.B < 5 AND t1.A > 1) OR (t2.B < t1.C AND t1.A > 1)"
    # OR condition with no matches
    query5 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.C < t2.B AND t2.A < t1.C) OR (t2.B < 5 AND t1.A > 1)"
    # OR condition with no matches across all ORs. Any pair matches, but
    # nothing matches across all 3.
    query6 = "Select t1.A, t2.B FROM table1 t1 inner join table2 t2 on (t1.C > 10 AND t2.A < t1.C AND t2.B < 5) OR (t2.B < 5 AND t1.A > 1) OR (t1.A > 1 AND t1.C > 10)"
    # Check the correctness of all queries. We check for the optimization
    # by validating the number of booland/| operations in the generated code.
    # The generated code has fewer booland than a regular filter because the
    # filters are pushed into the JOIN to the TABLE SCAN.
    # TODO: Validate the actual generated plans.
    result1 = check_query(query1, ctx, spark_info, return_codegen=True)
    gen_code1 = result1["pandas_code"]
    assert gen_code1.count("booland") == 0, "Expected 0 booland after optimization"
    assert gen_code1.count("|") == 1, "Expected 1 | after optimization"
    result2 = check_query(query2, ctx, spark_info, return_codegen=True)
    gen_code2 = result2["pandas_code"]
    assert gen_code2.count("booland") == 0, "Expected 0 booland after optimization"
    assert gen_code2.count("|") == 2, "Expected 2 | after optimization"
    result3 = check_query(query3, ctx, spark_info, return_codegen=True)
    gen_code3 = result3["pandas_code"]
    assert gen_code3.count("booland") == 0, "Expected 0 booland after optimization"
    assert gen_code3.count("|") == 1, "Expected 1 | after optimization"
    result4 = check_query(query4, ctx, spark_info, return_codegen=True)
    gen_code4 = result4["pandas_code"]
    assert gen_code4.count("booland") == 1, "Expected 1 booland after optimization"
    assert gen_code4.count("|") == 2, "Expected 2 | after optimization"
    result5 = check_query(query5, ctx, spark_info, return_codegen=True)
    gen_code5 = result5["pandas_code"]
    assert gen_code5.count("booland") == 2, "Expected 2 booland after no-optimization"
    assert gen_code5.count("|") == 1, "Expected 1 | after no-optimization"
    result6 = check_query(query6, ctx, spark_info, return_codegen=True)
    gen_code6 = result6["pandas_code"]
    assert gen_code6.count("booland") == 4, "Expected 4 booland after no-optimization"
    assert gen_code6.count("|") == 2, "Expected 2 | after no-optimization"
