import pytest
from bodosql.tests.utils import check_query


# TODO: fix memory leak issues with groupby apply, see [BS-530/BE-947]
def test_basic_pivot(basic_df, spark_info):
    """
    Basic test for PIVOT with 1 column.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS sum_a, AVG(C) AS avg_c
        FOR A IN (1 as single, 3 as triple)
    )
    """
    check_query(query, basic_df, spark_info, check_dtype=False)


# TODO: Determine memory leak issues in pivot
def test_multicol_pivot(basic_df, spark_info):
    """
    Basic test for PIVOT with mulitple columns.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS sum_a, AVG(C) AS avg_c
        FOR (A, B) IN ((1, 4) as col1, (2, 5) as col2)
    )
    """
    check_query(
        query, basic_df, spark_info, check_dtype=False, is_out_distributed=False
    )


# TODO: fix memory leak issues with groupby apply, see [BS-530/BE-947]
@pytest.mark.slow
def test_basic_null_pivot(basic_df, spark_info):
    """
    Basic test for PIVOT with 1 column without a match somewhere.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS sum_a, AVG(C) AS avg_c
        FOR A IN (1 as single, 7 as triple)
    )
    """
    check_query(query, basic_df, spark_info, convert_float_nan=True)


# TODO: Determine memory leak issues in pivot
@pytest.mark.slow
def test_multicol_null_pivot(basic_df, spark_info):
    """
    Basic test for PIVOT with mulitple columns without a match somewhere.
    """
    query = """
    SELECT * FROM table1
    PIVOT (
        SUM(A) AS sum_a, AVG(C) AS avg_c
        FOR (A, B) IN ((1, 4) as col1, (2, 15) as col2)
    )
    """
    check_query(
        query, basic_df, spark_info, convert_float_nan=True, is_out_distributed=False
    )
