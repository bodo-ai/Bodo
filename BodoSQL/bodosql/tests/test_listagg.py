"""
Test correctness of the LISTAGG SQL aggregation operations with and without groupby for BodoSQL
"""

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_groupby
from bodosql.tests.utils import check_query

# Skip unless any groupby-related files were changed
pytestmark = pytest_slow_unless_groupby


@pytest.mark.slow
def test_listagg_without_group_sorting_complex(listagg_data, memory_leak_check):
    """Full E2E test for listagg without groupby and with different sorting options"""

    query = """
SELECT
   LISTAGG(all_null_column) as agg_output_col0,
   LISTAGG(group_constant_str_col) as agg_output_col1,
   LISTAGG(group_constant_str_col2, ',') as agg_output_col2,
   LISTAGG(group_constant_str_col, '*') as agg_output_col3,
   LISTAGG(non_constant_str_col_with_nulls, '*') as agg_output_col3_2,
   LISTAGG(non_constant_str_col, ', ') WITHIN GROUP (ORDER BY order_col_1, order_col_3) as agg_output_col4,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_1 DESC NULLS LAST, order_col_2 ASC, order_col_3) as agg_output_col5,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_2 ASC NULLS FIRST, order_col_1 ASC NULLS FIRST, order_col_3) as agg_output_col6,
   LISTAGG(non_constant_str_col_with_nulls, ', ') WITHIN GROUP (ORDER BY order_col_2 ASC NULLS FIRST, order_col_1 ASC NULLS FIRST, order_col_3) as agg_output_col7,
   LISTAGG(all_null_column, '-') WITHIN GROUP (ORDER BY order_col_2 DESC NULLS LAST, order_col_3) as agg_output_col8
FROM table1\n"""

    expected = pd.DataFrame(
        {
            "AGG_OUTPUT_COL0": [""],
            "AGG_OUTPUT_COL1": ["aaaaaaœœœœœœeeeeee"],
            "AGG_OUTPUT_COL2": ["į,į,į,į,į,į,ë,ë,ë,ë,ë,ë,₠,₠,₠,₠,₠,₠"],
            "AGG_OUTPUT_COL3": ["a*a*a*a*a*a*œ*œ*œ*œ*œ*œ*e*e*e*e*e*e"],
            "AGG_OUTPUT_COL3_2": ["hi*hello*world*hi*hello*world*hi*hello*world"],
            "AGG_OUTPUT_COL4": ["B, B, B, D, D, D, F, F, F, A, C, E, A, C, E, A, C, E"],
            "AGG_OUTPUT_COL5": ["FFFDDDBBBAAACCCEEE"],
            "AGG_OUTPUT_COL6": ["BBBDDDFFFAAACCCEEE"],
            "AGG_OUTPUT_COL7": ["hello, hello, hello, world, world, world, hi, hi, hi"],
            "AGG_OUTPUT_COL8": [""],
        }
    )

    check_query(
        query,
        listagg_data,
        None,  # Spark info
        expected_output=expected,
        is_out_distributed=False,
    )


def test_listagg_no_withing_group_no_sep(listagg_data, spark_info, memory_leak_check):
    """Simple listagg test without sorting."""

    spark_equiv_query = """SELECT array_join(collect_list(table1.group_constant_str_col), '') FROM table1 group by key_col"""
    check_query(
        "SELECT listagg(group_constant_str_col) FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


@pytest.mark.slow
def test_listagg_no_withing_group_with_sep(listagg_data, spark_info, memory_leak_check):
    spark_equiv_query = """SELECT array_join(collect_list(table1.group_constant_str_col), ', ') FROM table1 group by key_col"""
    check_query(
        "SELECT listagg(group_constant_str_col, ', ') FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


@pytest.mark.slow
def test_listagg_no_within_group_with_other_aggregates(
    listagg_data, spark_info, memory_leak_check
):
    spark_equiv_query = """SELECT MAX(group_constant_str_col), key_col, array_join(collect_list(table1.group_constant_str_col), ''), MAX(group_constant_str_col2), array_join(collect_list(table1.group_constant_str_col2), 'å´îøü') FROM table1 group by key_col"""
    check_query(
        "SELECT MAX(group_constant_str_col), key_col, listagg(group_constant_str_col), MAX(group_constant_str_col2), listagg(group_constant_str_col2, 'å´îøü') FROM table1 group by key_col",
        listagg_data,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_equiv_query,
    )


def test_listagg_within_group_sorting_simple(listagg_data, memory_leak_check):
    """tests a simple withing group sorting with only one call"""
    expected = pd.DataFrame(
        {
            "agg_output_col": ["F, D, B, A, C, E"] * 3,
        }
    )

    check_query(
        "SELECT listagg(non_constant_str_col, ', ') WITHIN GROUP (ORDER BY order_col_1 DESC NULLS LAST, order_col_2 ASC NULLS FIRST) as agg_output_col FROM table1 group by key_col",
        listagg_data,
        None,  # Spark info
        check_names=False,
        check_dtype=False,
        expected_output=expected,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "with_having",
    [
        pytest.param(True, id="groupby-having"),
        pytest.param(False, id="groupby-no_having"),
    ],
)
def test_listagg_withing_group_sorting_complex(
    listagg_data, with_having, memory_leak_check
):
    """Full E2E test for listagg with groupby and with different sorting options"""

    query = """
SELECT
   LISTAGG(all_null_column) as agg_output_col0,
   LISTAGG(group_constant_str_col) as agg_output_col1,
   LISTAGG(group_constant_str_col2, ',') as agg_output_col2,
   LISTAGG(group_constant_str_col, '*') as agg_output_col3,
   LISTAGG(non_constant_str_col, ', ') WITHIN GROUP (ORDER BY order_col_1, order_col_3) as agg_output_col4,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_1 DESC NULLS LAST, order_col_2 ASC, order_col_3) as agg_output_col5,
   LISTAGG(non_constant_str_col) WITHIN GROUP (ORDER BY order_col_2 ASC NULLS FIRST, order_col_1 ASC NULLS FIRST, order_col_3) as agg_output_col6,
   LISTAGG(non_constant_str_col_with_nulls, ', ') WITHIN GROUP (ORDER BY order_col_2 ASC NULLS FIRST, order_col_1 ASC NULLS FIRST, order_col_3) as agg_output_col7,
   LISTAGG(all_null_column, '-') WITHIN GROUP (ORDER BY order_col_2 DESC NULLS LAST, order_col_3) as agg_output_col8
FROM table1
GROUP BY key_col"""

    if with_having:
        query += "\nHAVING LENGTH(LISTAGG(having_len_str, ',')) < 20"
        # Slightly different expected in having case
        expected = pd.DataFrame(
            {
                "AGG_OUTPUT_COL0": ["", ""],
                "AGG_OUTPUT_COL1": ["aaaaaa", "eeeeee"],
                "AGG_OUTPUT_COL2": ["į,į,į,į,į,į", "₠,₠,₠,₠,₠,₠"],
                "AGG_OUTPUT_COL3": ["a*a*a*a*a*a", "e*e*e*e*e*e"],
                "AGG_OUTPUT_COL4": ["B, D, F, A, C, E"] * 2,
                "AGG_OUTPUT_COL5": ["FDBACE"] * 2,
                "AGG_OUTPUT_COL6": ["BDFACE"] * 2,
                "AGG_OUTPUT_COL7": ["hello, world, hi"] * 2,
                "AGG_OUTPUT_COL8": ["", ""],
            }
        )
    else:
        expected = pd.DataFrame(
            {
                "AGG_OUTPUT_COL0": ["", "", ""],
                "AGG_OUTPUT_COL1": ["aaaaaa", "œœœœœœ", "eeeeee"],
                "AGG_OUTPUT_COL2": ["į,į,į,į,į,į", "ë,ë,ë,ë,ë,ë", "₠,₠,₠,₠,₠,₠"],
                "AGG_OUTPUT_COL3": ["a*a*a*a*a*a", "œ*œ*œ*œ*œ*œ", "e*e*e*e*e*e"],
                "AGG_OUTPUT_COL4": ["B, D, F, A, C, E"] * 3,
                "AGG_OUTPUT_COL5": ["FDBACE"] * 3,
                "AGG_OUTPUT_COL6": ["BDFACE"] * 3,
                "AGG_OUTPUT_COL7": ["hello, world, hi"] * 3,
                "AGG_OUTPUT_COL8": ["", "", ""],
            }
        )

    check_query(
        query,
        listagg_data,
        None,  # Spark info
        expected_output=expected,
    )
