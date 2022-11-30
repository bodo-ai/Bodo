# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Smoke tests for BodoSQL as described in this document:
https://bodo.atlassian.net/wiki/spaces/B/pages/1199898627/Smoke+Tests+for+BodoSQL
"""

import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture
def named_params():
    """Named parameters used for smoke test 1"""
    return {
        "tsize": 90,
        "default_str": "foobar",
    }


@pytest.fixture
def smoke_ctx_1():
    """DataFrames used for smoke tests 1 and 2"""
    return {
        "table1": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [None if i % 21 == 0 else ((i**2) % 50 - 18) for i in range(100)],
                    dtype=pd.Int32Dtype(),
                ),
                "float_col": pd.Series(
                    [
                        (((i + 7) ** 5) % 12345)
                        * 10 ** (i % 15 - 7)
                        * (-1) ** ((i + 1) % 7)
                        for i in range(100)
                    ],
                    dtype=np.float64,
                ),
                "bool_col": pd.Series(
                    [i % 15 > 0 for i in range(100)], dtype=pd.BooleanDtype()
                ),
                "datetime_col": pd.Series(
                    (list(pd.date_range("1995", "2022", 9)) + [None]) * 10
                ),
                "string_col": pd.Series(
                    [
                        None if i % 18 == 0 else str((3 + 2 * (i % 8)) ** 5)
                        for i in range(100)
                    ]
                ),
            }
        ),
    }


@pytest.fixture
def smoke_ctx_2():
    """DataFrames used for smoke tests 3, 4 and 5"""
    return {
        "table1": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [(i**2) % 17 for i in range(100)], dtype=pd.Int32Dtype()
                ),
                "float_col": pd.Series(
                    [
                        (1 + ((i + 3) ** 3) % 13) * 10 ** (i % 5 - 3) * (-1) ** (i % 3)
                        for i in range(100)
                    ],
                    dtype=np.float64,
                ),
                "datetime_col": pd.Series(list(pd.date_range("2011", "2018", 5)) * 20),
                "string_col": pd.Series(
                    [
                        None if i % 7 == 0 else chr(65 + (i**2) % 8 + i // 48)
                        for i in range(100)
                    ]
                ),
            }
        ),
        "table2": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [i**2 for i in range(10)], dtype=pd.Int32Dtype()
                ),
                "float_col": pd.Series(
                    [
                        (1 + ((i + 4) ** 4) % 63) * 10 ** (i % 5 - 3) * (-1) ** (i % 5)
                        for i in range(10)
                    ],
                    dtype=np.float64,
                ),
                "datetime_col": pd.Series(pd.date_range("2000", "2018", 10)),
                "bool_col": pd.Series(
                    [True, False, None, True, True, False, False, None, None, True]
                ),
            }
        ),
        "table3": pd.DataFrame(
            {
                "string_col": pd.Series(
                    [
                        a + b + c + d
                        for a in ["", *"ALPHABETS♫UP"]
                        for b in ["", *"ÉPSI∫øN"]
                        for c in ["", *"ZE฿Rä"]
                        for d in "THETA"
                    ]
                )
            }
        ),
    }


@pytest.fixture
def smoke_ctx_3():
    """DataFrames used for smoke tests 6 and 7"""
    return {
        "table1": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [(i**2) % 10 for i in range(100)], dtype=pd.Int32Dtype()
                ),
                "datetime_col": pd.Series(
                    list(pd.date_range("2010-01-01", "2012-03-14", 100))
                ),
            }
        ),
        "table2": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [0, 1, 2, None, None, None, 6, 7, 8, 9], dtype=pd.Int32Dtype()
                ),
                "string_col": pd.Series(list("αBCDEFGHΣ∞")),
            }
        ),
    }


@pytest.fixture
def smoke_ctx_4():
    """DataFrames used for smoke tests 8 and 9"""
    return {
        "table1": pd.DataFrame(
            {
                "int32_col": pd.Series(
                    [None if i % 35 == 7 else i for i in range(300)],
                    dtype=pd.Int32Dtype(),
                ),
                "float_col": pd.Series(
                    [
                        i**2 / (2 ** (i % 10)) * (-1.0 if i % 28 == 0 else 1.0)
                        for i in range(300)
                    ]
                ),
                "string_col": pd.Series(
                    [
                        "abcde"[i % 5] + "∂é§œπä"[i % 6] + "qwertyu"[(i**2) % 7]
                        for i in range(300)
                    ]
                ),
                "datetime_col": pd.Series(
                    [
                        pd.Timestamp(f"201{(i**2)%7}-{1+(i**3)%10}-{10+(i**4)%20}")
                        for i in range(300)
                    ]
                ),
            }
        ),
    }


@pytest.mark.smoke
def test_smoke1(smoke_ctx_1, named_params, spark_info, memory_leak_check):
    """Tests various scalar functions, CASE, WHERE, ORDER BY, LIMIT and
    named parameters"""
    query = """
SELECT
   int32_col,
   float_col,
   string_col,
   bool_col,
   datetime_col,
   COALESCE(int32_col, -1) AS A1,
   COALESCE(datetime_col, TIMESTAMP '1999-12-31') AS A2,
   CAST(int32_col AS VARCHAR) AS A3,
   SUBSTRING(string_col, 2, 2) AS A4,
   EXTRACT(year from datetime_col) AS A5,
   CASE WHEN LENGTH(string_col) < 4 THEN 'fizzbuzz' ELSE @default_str END AS A6
FROM table1
WHERE bool_col
ORDER BY int32_col ASC NULLS FIRST, float_col DESC
LIMIT @tsize
    """

    spark_query = query.replace("VARCHAR", "STRING")
    check_query(
        query,
        smoke_ctx_1,
        spark_info,
        named_params=named_params,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.smoke
def test_smoke2(smoke_ctx_1, spark_info, memory_leak_check):
    """Tests SELECT DISINCT and WHERE"""
    query = """
SELECT DISTINCT
   int32_col,
   string_col
FROM table1
WHERE int32_col IN (0, 1, 3, 7)
    """
    check_query(
        query,
        smoke_ctx_1,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke3(smoke_ctx_2, spark_info, memory_leak_check):
    """Tests implicit cross joins and non-groupby aggregation"""
    query = """
SELECT
   SUM(table1.int32_col),
   MAX(DATEDIFF(table2.datetime_col, table1.datetime_col)),
   STDDEV(table1.float_col - table2.float_col),
   COUNT(table2.bool_col),
   COUNT(*)
FROM table1, table2
    """
    check_query(
        query,
        smoke_ctx_2,
        spark_info,
        check_names=False,
        check_dtype=False,
        is_out_distributed=False,
    )


@pytest.mark.smoke
def test_smoke4(smoke_ctx_2, spark_info, memory_leak_check):
    """Tests groupby aggregation, WHERE (with a subquery) and UNION"""
    query = """
    (SELECT
        string_col,
        datetime_col,
        COUNT(*),
        AVG(int32_col),
        MIN(int32_col),
        VARIANCE(float_col),
        COUNT(DISTINCT float_col)
    FROM table1
    WHERE table1.string_col IN (SELECT DISTINCT LEFT(table3.string_col, 1) FROM table3)
    GROUP BY string_col, datetime_col)
UNION
    (SELECT
        string_col,
        MAX(datetime_col),
        COUNT(*),
        STDDEV(int32_col),
        -13,
        STDDEV_SAMP(float_col),
        COUNT_IF(datetime_col < TIMESTAMP '2015-01-01')
    FROM table1
    GROUP BY string_col)
    """
    check_query(
        query,
        smoke_ctx_2,
        spark_info,
        check_names=False,
        check_dtype=False,
        only_jit_1DVar=True,
    )


@pytest.mark.smoke
def test_smoke5(smoke_ctx_2, spark_info, memory_leak_check):
    """Tests inner joints, groupby aggregation and HAVING with ORDER BY"""
    query = """
SELECT
   table1.float_col,
   MIN(table1.int32_col),
   MAX(table1.int32_col),
   COUNT(*) as num_rows
FROM table1 INNER JOIN table2 ON table1.int32_col < table2.int32_col
GROUP BY table1.float_col
HAVING num_rows > 9
ORDER BY num_rows, table1.float_col
    """
    check_query(
        query,
        smoke_ctx_2,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.smoke
def test_smoke6(smoke_ctx_3, spark_info, memory_leak_check):
    """Tests left joins, window functions and ORDER BY"""
    query = """
SELECT
   table1.int32_col,
   table2.string_col,
   IF(table2.string_col IS NULL, '-', REPEAT(table2.string_col, 5)),
   table1.datetime_col,
   FIRST_VALUE(table1.datetime_col) OVER (PARTITION BY table1.int32_col ORDER BY table1.datetime_col),
   LAG(table1.datetime_col, 1) OVER (PARTITION BY table1.int32_col ORDER BY table1.datetime_col)
FROM table1 LEFT JOIN table2 ON table1.int32_col = table2.int32_col
ORDER BY table2.string_col, table1.datetime_col
    """
    check_query(
        query,
        smoke_ctx_3,
        spark_info,
        check_names=False,
        check_dtype=False,
        sort_output=False,
    )


@pytest.mark.smoke
def test_smoke7(smoke_ctx_3, spark_info, memory_leak_check):
    """Tests window functions and QUALIFY"""
    query = """
SELECT
   int32_col,
   datetime_col,
   ROW_NUMBER() OVER (PARTITION BY 1 ORDER BY datetime_col),
   RANK() OVER (PARTITION BY 1 ORDER BY datetime_col),
   AVG(int32_col) OVER (PARTITION BY 1 ORDER BY int32_col ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
FROM table1
QUALIFY ROW_NUMBER() OVER (PARTITION BY 1 ORDER BY datetime_col) % 2 = 0
    """
    spark_query = """
SELECT
    *
FROM (
    SELECT
        int32_col,
        datetime_col,
        ROW_NUMBER() OVER (PARTITION BY 1 ORDER BY datetime_col) AS R,
        RANK() OVER (PARTITION BY 1 ORDER BY datetime_col),
        MAX(datetime_col) OVER (PARTITION BY 1 ORDER BY int32_col ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING)
    FROM table1)
WHERE R % 2 = 0 
"""
    check_query(
        query,
        smoke_ctx_3,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke8(smoke_ctx_4, spark_info, memory_leak_check):
    """Tests window functions, WHERE, GROUP BY, HAVING and QUALIFY"""
    query = """
SELECT
   string_col,
   NULLIF(POSITION('a' IN string_col), 0),
   MIN(float_col),
   MIN(MIN(float_col)) OVER (PARTITION BY 1 ORDER BY string_col ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS M
FROM table1
WHERE int32_col > 4 OR int32_col < 2
GROUP BY string_col
HAVING COUNT(int32_col) = COUNT(*)
QUALIFY M >= 0
    """
    spark_query = """
SELECT
    *
FROM (
    SELECT
        string_col,
        NULLIF(POSITION('a' IN string_col), 0),
        MIN(float_col),
        MIN(MIN(float_col)) OVER (PARTITION BY 1 ORDER BY string_col ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS M
    FROM table1
    WHERE int32_col > 4 OR int32_col < 2
    GROUP BY string_col
    HAVING COUNT(int32_col) = COUNT(*))
WHERE M >= 0
"""
    check_query(
        query,
        smoke_ctx_4,
        spark_info,
        equivalent_spark_query=spark_query,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.smoke
def test_smoke9(smoke_ctx_4, spark_info, memory_leak_check):
    """Tests WITH and PIVOT"""
    query = """
WITH T AS (
   SELECT int32_col % 10 AS n, EXTRACT(year FROM datetime_col) AS y
   FROM table1
   WHERE int32_col IS NOT NULL)
SELECT
    COALESCE(c10, 0),
    COALESCE(c11, 0),
    COALESCE(c12, 0),
    COALESCE(c13, 0),
    COALESCE(c14, 0)
FROM T
PIVOT(COUNT(*) for y IN (2010 AS c10, 2011 AS c11, 2012 AS c12, 2013 AS c13, 2014 AS c14))
    """
    check_query(
        query,
        smoke_ctx_4,
        spark_info,
        check_names=False,
        check_dtype=False,
    )
