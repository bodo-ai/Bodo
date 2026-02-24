"""
Test correctness of SQL arithmetic operations on BodoSQL
"""

import pandas as pd
import pytest

from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import (
    check_query,
    create_pyspark_schema_from_dataframe,
)

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


@pytest.mark.slow
def test_basic_arithmetic_select(bodosql_numeric_types, spark_info, memory_leak_check):
    """test select calls with an arithmetic expression"""
    check_query(
        "select A + 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 + A from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select A - 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 - A from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_column_arithmetic_select(bodosql_numeric_types, spark_info, memory_leak_check):
    """test select calls with an arithmetic expression"""
    check_query(
        "select A + C from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select C - A from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select A * (1 - C) from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_sum_arith_select(bodosql_numeric_types, spark_info, memory_leak_check):
    """test sum on a call with an arithmetic expression"""
    check_query(
        "select sum(A + 1) from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        is_out_distributed=False,
    )


@pytest.mark.slow
def test_nested_arithmetic_select(bodosql_numeric_types, spark_info, memory_leak_check):
    """test select calls with an arithmetic expression"""
    check_query(
        "select A + 1 + 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 + A + 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 + 1 + A from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select A - 1 - 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 - A - 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 - 1 - A from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_multicolumn_arithmetic_select(
    bodosql_numeric_types, spark_info, memory_leak_check
):
    """test select calls with an arithmetic expression"""
    check_query(
        "select A + 1, B from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )
    check_query(
        "select 1 + A, A - 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_select_col_arith(bodosql_numeric_types, spark_info, memory_leak_check):
    """test selecting a single column and performing arithmetic
    on that columns
    """
    check_query(
        "select A, A + 1 from table1",
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


def test_addition_constants(basic_df, spark_info, memory_leak_check):
    """
    Tests that addition works on integer constants
    """
    query = "Select A from table1 where B = .8 + 1 + 1 + -2 + 4 + .2"
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_addition_columns(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that addition works on columns
    """
    query = "Select A + B from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_subtraction_constants(basic_df, spark_info, memory_leak_check):
    """
    Tests that subtraction works on constants
    """
    query = "Select A from table1 where B = 10 - .001 - -2 - 1 - .2 - .7 - .3 -.8 - 2"
    check_query(query, basic_df, spark_info, check_dtype=False)


def test_subtraction_columns(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that subtraction works on columns
    """
    query = "Select B - A from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_multiplication_constants(basic_df, spark_info, memory_leak_check):
    """
    Tests that multiplication works on constants
    """
    query = "Select A from table1 where B = -1.0 * -1.0 * 2 * 3"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_multiply_columns(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that multiplication works on columns
    """
    query = "Select A * B from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


def test_division_constants(basic_df, spark_info, memory_leak_check):
    """
    Tests that division works on constants
    """
    query = "Select A from table1 where A = 24 / 2 / 4 / -1.0 / -1.0"
    check_query(query, basic_df, spark_info, check_dtype=False, check_names=False)


def test_division_columns(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that division works on columns
    """
    query = "Select B / A from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_op_order(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that order of operations is correct
    """
    query = "Select A from table1 where B = 5 + 10 / 2 - 3 / 3 - 5"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_op_order_cols(bodosql_numeric_types, spark_info, memory_leak_check):
    """
    Tests that order of operations is correct on columns
    """
    query = "Select A + C / B - A from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
def test_arith_ops_between_tables(basic_df, spark_info, memory_leak_check):
    """
    Tests that arith operations between tables work as intended
    """
    query = "Select table1.A + table2.C / table1.B - table2.A from table1, table2"
    newCtx = {"TABLE1": basic_df["TABLE1"], "TABLE2": basic_df["TABLE1"]}
    check_query(query, newCtx, spark_info, check_dtype=False, check_names=False)


@pytest.mark.skip("BS[112]")
def test_wraparound_unsigned_numerics(
    bodosql_numeric_types, spark_info, memory_leak_check
):
    """
    Tests that wraparound occurs when dealing with unsigned integer types
    """
    query = "Select A - B from table1"
    check_query(
        query, bodosql_numeric_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "interval",
    [
        "'4' years",
        "'5' months",
        "'3' days",
        "'17' hours",
        "'53' minutes",
        "'6' seconds",
        pytest.param(
            "'123' milliseconds",
            marks=pytest.mark.skip(
                "[BS-507] Support Intervals for values smaller than seconds"
            ),
        ),
        pytest.param(
            "'45' microseconds",
            marks=pytest.mark.skip(
                "[BS-507] Support Intervals for values smaller than seconds"
            ),
        ),
        pytest.param(
            "'4' nanoseconds",
            marks=pytest.mark.skip(
                "[BS-507] Support Intervals for values smaller than seconds"
            ),
        ),
    ],
)
def test_add_sub_intervals(
    bodosql_datetime_types, spark_info, memory_leak_check, interval
):
    """
    Tests support for + and - with various intervals.
    """
    # Place the - inside the ''
    negative_interval = interval[0] + "-" + interval[1:]
    query = f"Select A + interval {interval} from table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_dtype=False, check_names=False
    )
    query = f"Select A + interval {negative_interval} from table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_dtype=False, check_names=False
    )
    query = f"Select A - interval {interval} from table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_dtype=False, check_names=False
    )
    query = f"Select A - interval {negative_interval} from table1"
    check_query(
        query, bodosql_datetime_types, spark_info, check_dtype=False, check_names=False
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT -A, -B, -(B + -(A*A*A)) from table1", id="regular"),
        pytest.param(
            "SELECT CASE WHEN A > B THEN -A + -B ELSE -(A + -(B*B*B*B*B)) END from table1",
            id="case",
        ),
    ],
)
def test_negation(query, bodosql_numeric_types, spark_info, memory_leak_check):
    """Tests unary negation"""
    # Exact value depends on bitwidth so we have to pass the Spark schema
    pyspark_schemas = {}
    for table_name, df in bodosql_numeric_types.items():
        pyspark_schemas[table_name] = create_pyspark_schema_from_dataframe(df)

    check_query(
        query,
        bodosql_numeric_types,
        spark_info,
        check_dtype=False,
        check_names=False,
        pyspark_schemas=pyspark_schemas,
    )


def test_date_arithmetic(bodosql_date_types, memory_leak_check):
    """Tests +/- on date columns with integers. This is supported in Snowflake for
    date but not similar types like Timestamp.
    https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#simple-arithmetic-for-dates
    """
    query = "select A + 7 as col1, B - 2 as col2, 1 + C as col3 from table1"
    # Spark doesn't support date arithmetic with integers
    table1 = bodosql_date_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "COL1": table1.A + pd.Timedelta(days=7),
            "COL2": table1.B - pd.Timedelta(days=2),
            "COL3": pd.Timedelta(days=1) + table1.C,
        }
    )
    check_query(
        query,
        bodosql_date_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_date_arithmetic_case(bodosql_date_types, memory_leak_check):
    """Tests +/- on date columns with integers inside a case statement (to check
    the return type). This is supported in Snowflake for
    date but not similar types like Timestamp.
    https://docs.snowflake.com/en/sql-reference/data-types-datetime.html#simple-arithmetic-for-dates
    """

    query = "select CASE WHEN A > B THEN A + 7 WHEN B > C THEN B - 2 END as col1 from table1"
    # Spark doesn't support date arithmetic with integers
    table1 = bodosql_date_types["TABLE1"]
    S1 = table1.A + pd.Timedelta(days=7)
    S2 = table1.B - pd.Timedelta(days=2)
    expected_output = pd.DataFrame({"COL1": S1})
    # Replace S1 with S2 when not A > B
    filter1 = ~(table1.A > table1.B)
    expected_output["COL1"] = expected_output["COL1"].where(~filter1, S2)
    # Insert null for the else condition
    filter2 = filter1 & ~(table1.B > table1.C)
    expected_output["COL1"] = expected_output["COL1"].where(~filter2, None)
    check_query(
        query,
        bodosql_date_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_subtraction_between_dates(bodosql_date_types, memory_leak_check):
    """Tests - on two date columns. The output in SQL is an integer."""
    query = "select A - B as col1 from table1"
    # Spark doesn't support date arithmetic with integers
    table1 = bodosql_date_types["TABLE1"]
    expected_output = pd.DataFrame(
        {
            "COL1": (table1.A - table1.B).map(
                lambda a: pd.NA if pd.isna(a) else a.days
            ),
        }
    )
    check_query(
        query,
        bodosql_date_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_subtraction_between_dates_case(bodosql_date_types, memory_leak_check):
    """Tests - on two date columns with case. The output in SQL is an integer."""

    query = "select CASE WHEN A > B THEN A - B ELSE B - A END as col1 from table1"
    # Spark doesn't support date arithmetic with integers
    table1 = bodosql_date_types["TABLE1"]
    S1 = (table1.A - table1.B).map(lambda a: pd.NA if pd.isna(a) else a.days)
    S2 = (table1.B - table1.A).map(lambda a: pd.NA if pd.isna(a) else a.days)
    expected_output = pd.DataFrame({"COL1": S1})
    # Replace S1 with S2 when not A > B
    filter1 = ~(table1.A > table1.B)
    expected_output["COL1"] = expected_output["COL1"].where(~filter1, S2)
    check_query(
        query,
        bodosql_date_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )
