# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Smoke tests for BodoSQL covering each major query feature
"""

import numpy as np
import pandas as pd
import pytest

from bodosql.tests.utils import check_query, get_equivalent_spark_agg_query


@pytest.fixture
def smoke_ctx():
    t1 = pd.DataFrame(
        {
            "A": pd.Series(
                [None if i % 7 == 5 else (i**2) % 10 for i in range(1000)],
                dtype=pd.Int32Dtype(),
            ),
            "B": pd.Series([str(i % 100) for i in range(1000)]),
            "C": pd.Series([bool(i % 6) for i in range(1000)]),
            "D": pd.Series([np.tan(i) for i in range(1000)]),
        }
    )
    t2 = pd.DataFrame(
        {
            "W": pd.Series(
                [None if i % 13 > 10 else (i**3) % 10 for i in range(100)],
                dtype=pd.Int32Dtype(),
            ),
            "X": pd.Series([str(99 - i) for i in range(100)]),
            "Y": pd.Series([bool(i % 4) for i in range(100)]),
            "Z": pd.Series([np.cos(i) for i in range(100)]),
        }
    )
    t = pd.Series(
        [(pd.Timestamp("2020-1-1") + pd.Timedelta(days=i)) for i in range(1461)]
    )
    t3 = pd.DataFrame(
        {
            "y": t.dt.year,
            "m": t.dt.month,
            "d": t.dt.day,
            "delta": pd.Series([int(0.5 + np.tan(i)) for i in range(1461)]),
        }
    )
    seasons = ["winter"] * 3 + ["spring"] * 3 + ["summer"] * 3 + ["fall"] * 3
    t3["s"] = t3["m"].apply(lambda m: seasons[m % 12])
    return {"table1": t1, "table2": t2, "table3": t3}


@pytest.mark.smoke
def test_smoke_project(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests selecting a subset of the columns
    """
    query = """
    SELECT A, C
    FROM table1
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_functions(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests calling several useful scalar functions
    """
    selects = [
        "COALESCE(A, -1)",
        "GREATEST(D, 0.0)",
        "LPAD(B, 3, '0')",
        "A + D",
        "IF(C, B, '')",
        "NULLIF(B, REVERSE(B))",
    ]
    query = f"SELECT {', '.join(selects)} FROM table1"
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_case(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using CASE statements
    """
    query = """
    SELECT
        CASE WHEN C THEN 'A' ELSE 'B' END,
        CASE WHEN A IS NULL THEN '' ELSE B END,
        CASE WHEN A = 0 THEN 1 WHEN A = 1 THEN 0 END
    FROM table1
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_raw_select(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using SELECT without a table provided
    """
    query = """
        SELECT
            'alphabet' LIKE 'a%',
            'alphabet' LIKE '%a',
            GREATEST(Timestamp '2022-1-1', Timestamp '2022-10-1', Timestamp '2022-1-1 10:35:32'),
            LEAST(Timestamp '2022-1-1', Timestamp '2022-10-1', Timestamp '2022-1-1 10:35:32')
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_distinct(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests selecting a subset of the columns with SELECT DISTINCT
    """
    query = """
    SELECT DISTINCT A, B
    FROM table1
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_sort(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using an ORDER BY clause
    """
    query = """
    SELECT *
    FROM table1
    ORDER BY D
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        sort_output=False,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_limit(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using an LIMIT clause
    """
    query = """
    SELECT *
    FROM table1
    ORDER BY D DESC
    LIMIT 10
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        sort_output=False,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_join(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using a simple JOIN clause
    """
    query = """
    SELECT *
    FROM table1
    INNER JOIN table2
    ON B=X
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_filter(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using a WHERE clause
    """
    query = """
    SELECT *
    FROM table1
    WHERE C
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_window(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests window functions
    """
    query = """
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY A ORDER BY D),
        NTILE(10) OVER (PARTITION BY A ORDER BY D)
    FROM table1
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_qualify(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests QUALIFY
    """
    query = """
    SELECT A, B, C, D
    FROM table1
    QUALIFY MIN(D) OVER (PARTITION BY B ORDER BY A ASC NULLS FIRST, D ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) = D;
    """
    spark_query = """
    SELECT A, B, C, D
    FROM (
        SELECT
            A, B, C, D,
            MIN(D) OVER (PARTITION BY B ORDER BY A ASC NULLS FIRST, D ASC ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS M
        FROM table1
    )
   WHERE M = D
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_groupby_aggregation(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests only groupby aggregation
    """
    agg_terms = [
        "A",
        "C",
        "COUNT(*)",
        "MIN(C)",
        "MAX(C)",
        "STDDEV_SAMP(D)",
        "STDDEV_POP(D)",
    ]
    query = f"SELECT {', '.join(agg_terms)} FROM table1 GROUP BY A, C"
    spark_query = get_equivalent_spark_agg_query(query)
    check_query(
        query,
        smoke_ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_nogroup_aggregation(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests only no-groupby aggregation
    """
    agg_terms = [
        "COUNT(*)",
        "COUNT(A)",
        "COUNT(DISTINCT B)",
        "COUNT_IF(C)",
        "VARIANCE_SAMP(D)",
        "VARIANCE_POP(D)",
        "MAX(D)",
    ]
    query = f"SELECT {', '.join(agg_terms)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    check_query(
        query,
        smoke_ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.smoke
def test_smoke_grouping_set_aggregation(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests a mix of concatenated groupby & no-groupby aggregation using GROUPING SETS.
    """
    agg_terms = ["A", "B", "COUNT(*)", "COUNT_IF(C)", "SUM(D)"]
    query = f"SELECT {', '.join(agg_terms)} FROM table1"
    spark_query = get_equivalent_spark_agg_query(query)
    query += " GROUP BY ROLLUP(A, B)"
    spark_query += " GROUP BY A, B WITH ROLLUP"
    check_query(
        query,
        smoke_ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_having(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests GROUP BY aggregation with a HAVING clause
    """
    query = """
    SELECT
        A,
        AVG(D),
        MIN(D),
        MAX(D)
    FROM table1
    GROUP BY A
    HAVING COUNT(*) >= 100;
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_pivot(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using PIVOT aggregations
    """
    query = """
        SELECT *
        FROM (SELECT y, s, delta FROM table3)
        PIVOT(SUM(delta) FOR s IN ('summer' as summer, 'fall' as fall, 'winter' as winter, 'spring' as spring))
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_setops(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using UNION ALL, INTERSECT and EXCEPT
    """
    query = """
    (
        (SELECT A, LEFT(B, 1) FROM table1)
        INTERSECT
        (SELECT A, RIGHT(B, 1) FROM table1)
    ) UNION ALL (
        (SELECT W, LEFT(X, 1) FROM table2)
        EXCEPT
        (SELECT W, RIGHT(X, 1) FROM table2)
    )
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke_subquery_ops(smoke_ctx, spark_info, memory_leak_check):
    """
    Tests using subquery operators ANY, ALL and IN
    """
    query = """
    (
        SELECT D, 'ALL'
        FROM table1
        WHERE D < ALL(SELECT Z FROM table2)
    ) UNION ALL (
        SELECT D, 'ANY'
        FROM table1
        WHERE D < ANY(SELECT Z FROM table2)
    ) UNION ALL (
        SELECT D, 'IN'
        FROM table1
        WHERE D::INTEGER IN (SELECT Z::INTEGER FROM table2)
    );
    """
    spark_query = """
    (
        SELECT D, 'ALL'
        FROM table1
        WHERE D < (SELECT MIN(Z) FROM table2)
    ) UNION ALL (
        SELECT D, 'ANY'
        FROM table1
        WHERE D < (SELECT MAX(Z) FROM table2)
    ) UNION ALL (
        SELECT D, 'IN'
        FROM table1
        WHERE CAST(D AS INTEGER) IN (SELECT CAST(Z AS INTEGER) FROM table2)
    )
    """
    check_query(
        query,
        smoke_ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )
