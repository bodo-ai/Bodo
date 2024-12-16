"""
Test that Dynamic Parameters can be used for the limit and offset values in
a SQL LIMIT expression.
"""

import pandas as pd

import bodo
import bodosql
from bodosql.tests.named_params_common import int_named_params  # noqa
from bodosql.tests.utils import check_query


def test_named_param_limit_unsigned(basic_df, int_named_params, memory_leak_check):
    """
    Checks using a named parameter
    inside a limit clause.
    """
    query = "select a from table1 limit @a"
    check_query(
        query,
        basic_df,
        None,
        named_params=int_named_params,
        expected_output=pd.DataFrame(
            {"A": basic_df["TABLE1"].A.head(int_named_params["a"])}
        ),
    )


def test_bind_variable_limit_unsigned(basic_df, int_named_params, memory_leak_check):
    """
    Checks using a named parameter
    inside a limit clause.
    """
    query = "select a from table1 limit ?"
    bind_variables = (int_named_params["a"], int_named_params["b"])

    check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        expected_output=pd.DataFrame(
            {"A": basic_df["TABLE1"].A.head(bind_variables[0])}
        ),
    )


def test_named_param_limit_offset(basic_df, int_named_params, memory_leak_check):
    """
    Checks using a named parameter
    inside limit and offset clauses.
    """
    query = "select A from table1 limit @a, @b"
    # Spark doesn't support offset so use an expected output
    a = int_named_params["a"]
    b = int_named_params["b"]
    expected_output = basic_df["TABLE1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        None,
        named_params=int_named_params,
        expected_output=expected_output,
    )


def test_bind_variable_limit_offset(basic_df, int_named_params, memory_leak_check):
    """
    Checks using a bind variable
    inside limit and offset clauses.
    """
    query = "select A from table1 limit ?, ?"
    # Spark doesn't support offset so use an expected output
    bind_variables = (int_named_params["a"], int_named_params["b"])
    a = bind_variables[0]
    b = bind_variables[1]
    expected_output = basic_df["TABLE1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        expected_output=expected_output,
    )


def test_limit_offset_keyword(basic_df, int_named_params, memory_leak_check):
    """
    Checks using a named parameter
    inside limit and offset clauses.
    """
    query = "select A from table1 limit @b offset @a"
    # Spark doesn't support offset so use an expected output
    a = int_named_params["a"]
    b = int_named_params["b"]
    expected_output = basic_df["TABLE1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        None,
        named_params=int_named_params,
        expected_output=expected_output,
    )


def test_bind_variable_limit_offset_keyword(
    basic_df, int_named_params, memory_leak_check
):
    """
    Checks using a bind variable
    inside limit and offset clauses.
    """
    query = "select A from table1 limit ? offset ?"
    # Spark doesn't support offset so use an expected output
    bind_variables = (int_named_params["b"], int_named_params["a"])
    a = bind_variables[1]
    b = bind_variables[0]
    expected_output = basic_df["TABLE1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        expected_output=expected_output,
    )


def test_limit_named_param_constant(basic_df, spark_info, memory_leak_check):
    """
    Checks that using a constant named_param compiles.
    """

    @bodo.jit
    def f(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select a from table1 limit @a", {"a": 10})

    df = basic_df["TABLE1"]
    py_output = pd.DataFrame({"A": df.A.head(10)})
    sql_output = f(df)
    pd.testing.assert_frame_equal(sql_output, py_output, check_column_type=False)
