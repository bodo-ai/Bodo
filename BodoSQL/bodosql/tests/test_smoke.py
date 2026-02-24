"""
Smoke tests for BodoSQL covering each major query feature
"""

import numpy as np
import pandas as pd
import pyarrow as pa
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
        [(pd.Timestamp("2020-1-1") + pd.Timedelta(days=i)) for i in range(1461)],
        dtype="datetime64[ns]",
    )
    t3 = pd.DataFrame(
        {
            "Y": t.dt.year,
            "M": t.dt.month,
            "D": t.dt.day,
            "DELTA": pd.Series([int(0.5 + np.tan(i)) for i in range(1461)]),
        }
    )
    seasons = ["winter"] * 3 + ["spring"] * 3 + ["summer"] * 3 + ["fall"] * 3
    t3["S"] = t3["M"].apply(lambda m: seasons[m % 12])
    return {
        "TABLE1": t1,
        "TABLE2": t2,
        "TABLE3": t3,
    }


@pytest.fixture
def smoke_shipping_ctx():
    shipping_log = pd.DataFrame(
        {
            "LOG_TS": pd.Series(
                [
                    pd.Timestamp(
                        1.5 * 10**18
                        + (10**5)
                        * (
                            10
                            + i * np.sin(i) * np.sin(i)
                            + (i**2) * np.cos(i) * np.cos(i)
                        )
                    )
                    for i in range(10**6)
                ],
                dtype="datetime64[ns]",
            ),
            "STORE_ID": pd.Series(
                [1234 + (i % np.round(np.sqrt(i + 1))) for i in range(10**6)],
                dtype=pd.Int32Dtype(),
            ),
            "LOC_ID": pd.Series(
                [np.round(101 + int(0.5 + np.tan(i))) % 100 for i in range(10**6)]
            ),
            "PRODUCT_ID": pd.Series(
                [
                    100 + (i % np.round(np.sqrt(np.sqrt((i + 1) * i * i))))
                    for i in range(10**6)
                ],
                dtype=pd.Int32Dtype(),
            ),
            "COST": pd.Series(
                [(np.tan(2 * i + 0.3) % 100000) for i in range(10**6)],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
        }
    )
    producers = pd.DataFrame(
        {
            "ID": pd.Series(list(range(1234, 2234))),
            "STORE_NAME": pd.Series([hex(i**2)[2:] for i in range(1234, 2234)]),
            "STORE_TYPE": pd.Series(
                [
                    ["GROCERY", "MANUFACTURING", "SOFTWARE", "TEXTILE", "HEALTH"][
                        min(i % 5, i % 7)
                    ]
                    for i in range(1234, 2234)
                ]
            ),
        }
    )
    destinations = pd.DataFrame(
        {
            "ID": pd.Series(list(range(100))),
            "LOC_NAME": pd.Series([hex((i + 101) ** 2)[2:] for i in range(100)]),
            "STATE": pd.Series(
                [
                    ["CA", "GA", "MI", "TX", "NJ", "WA", "OR", "NY", "PA", "IL"][i % 10]
                    for i in range(100)
                ]
            ),
        }
    )
    products = pd.DataFrame(
        {
            "ID": pd.Series(list(range(100, 29891))),
            "PRODUCT_NAME": pd.Series([hex(i**2 - i)[5:] for i in range(100, 29891)]),
        }
    )
    return {
        "SHIPPING_LOG": shipping_log,
        "DESTINATIONS": destinations,
        "PRODUCERS": producers,
        "PRODUCTS": products,
    }


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
@pytest.mark.timeout(2000)
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
    # Note: Bodo cast from float -> int always uses ROUND HALF UP,
    # so we match this in Spark explicitly.
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
        WHERE CAST(ROUND(D) AS INTEGER) IN (SELECT CAST(ROUND(Z) AS INTEGER) FROM table2)
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


@pytest.mark.smoke
def test_smoke_shipping_workload(smoke_shipping_ctx, memory_leak_check):
    """
    Tests joining the shipping table on the destinations, product, and producers
    tables to calculate the following for each industry in each state:
    - The total cost of shipments to each state relating to each industry
    - The number of unique stores shipping products to each state per industry
    - The number of unique products being shipped to each state per industry
    """
    # TODO: Fix cast with decimal and revert total_shipping
    query = """
    SELECT
        STATE,
        STORE_TYPE,
        CAST(SUM(COST) AS DOUBLE) as total_shipping,
        COUNT(DISTINCT(STORE_ID)) AS n_stores,
        COUNT(DISTINCT(PRODUCT_NAME)) AS n_products,
    FROM SHIPPING_LOG, DESTINATIONS, PRODUCERS, PRODUCTS
    WHERE SHIPPING_LOG.LOC_ID = DESTINATIONS.ID
    AND SHIPPING_LOG.STORE_ID = PRODUCERS.ID
    AND SHIPPING_LOG.PRODUCT_ID = PRODUCTS.ID
    AND STATE NOT IN ('OR', 'WA', 'CA')
    GROUP BY STATE, STORE_TYPE
    """
    answer = pd.DataFrame(
        {
            "STATE": [
                "GA",
                "GA",
                "GA",
                "GA",
                "GA",
                "IL",
                "IL",
                "IL",
                "IL",
                "IL",
                "MI",
                "MI",
                "MI",
                "MI",
                "MI",
                "NJ",
                "NJ",
                "NJ",
                "NJ",
                "NJ",
                "NY",
                "NY",
                "NY",
                "NY",
                "NY",
                "PA",
                "PA",
                "PA",
                "PA",
                "PA",
                "TX",
                "TX",
                "TX",
                "TX",
                "TX",
            ],
            "STORE_TYPE": [
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
                "GROCERY",
                "HEALTH",
                "MANUFACTURING",
                "SOFTWARE",
                "TEXTILE",
            ],
            # TODO: Fix cast with decimal and revert total_shipping
            "TOTAL_SHIPPING": pd.Series(
                [
                    7862256249.22,
                    2115206460.08,
                    6397246671.76,
                    5044534824.56,
                    3577084792.20,
                    15484.72,
                    4147.24,
                    12593.94,
                    10129.59,
                    6951.75,
                    3478345192.58,
                    939270608.80,
                    2830856662.50,
                    2220187767.58,
                    1586176358.19,
                    1017297617.27,
                    276399370.63,
                    842198000.42,
                    645098471.39,
                    466698931.61,
                    266503771.62,
                    74201019.76,
                    222603065.62,
                    179602373.10,
                    123701703.76,
                    27806732.46,
                    7701849.28,
                    22005496.56,
                    18904358.85,
                    11503049.28,
                    2086584328.56,
                    558195808.56,
                    1685087385.71,
                    1333389961.65,
                    946992900.22,
                ],
                dtype="Float64",
            ),
            "N_STORES": [
                314,
                85,
                256,
                200,
                143,
                313,
                84,
                255,
                198,
                142,
                314,
                85,
                256,
                201,
                143,
                312,
                84,
                253,
                199,
                141,
                309,
                84,
                250,
                197,
                140,
                308,
                84,
                254,
                199,
                141,
                313,
                85,
                255,
                200,
                142,
            ],
            "N_PRODUCTS": [
                24759,
                17272,
                23923,
                22739,
                20820,
                9034,
                3160,
                7743,
                6570,
                4877,
                19412,
                9680,
                17947,
                16055,
                13391,
                9030,
                3176,
                7874,
                6408,
                4975,
                6216,
                2062,
                5329,
                4370,
                3256,
                7092,
                2366,
                6019,
                4941,
                3687,
                13172,
                5094,
                11578,
                9854,
                7689,
            ],
        }
    )
    check_query(
        query,
        smoke_shipping_ctx,
        None,
        expected_output=answer,
        check_names=False,
        check_dtype=False,
    )
