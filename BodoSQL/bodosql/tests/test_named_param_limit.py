"""
Test that Named Parameters can be used for the limit and offset values in
a SQL LIMIT expression.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.
import re

import bodosql
import pandas as pd
import pytest
from bodosql.tests.named_params_common import int_named_params  # noqa
from bodosql.tests.utils import check_query

import bodo


def test_limit_unsigned(basic_df, spark_info, int_named_params):
    """
    Checks using a named parameter
    inside a limit clause.
    """
    query = "select a from table1 limit @a"
    check_query(query, basic_df, spark_info, named_params=int_named_params)


def test_limit_offset(basic_df, spark_info, int_named_params):
    """
    Checks using a named parameter
    inside limit and offset clauses.
    """
    query = "select A from table1 limit @a, @b"
    # Spark doesn't support offset so use an expected output
    a = int_named_params["a"]
    b = int_named_params["b"]
    expected_output = basic_df["table1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        spark_info,
        named_params=int_named_params,
        expected_output=expected_output,
    )


def test_limit_offset_keyword(basic_df, spark_info, int_named_params):
    """
    Checks using a named parameter
    inside limit and offset clauses.
    """
    query = "select A from table1 limit @b offset @b"
    # Spark doesn't support offset so use an expected output
    a = int_named_params["a"]
    b = int_named_params["b"]
    expected_output = basic_df["table1"].iloc[a : a + b, [0]]
    check_query(
        query,
        basic_df,
        spark_info,
        named_params=int_named_params,
        expected_output=expected_output,
    )


def test_limit_named_param_constant(basic_df, spark_info):
    """
    Checks that using a constant named_param compiles.
    """

    @bodo.jit
    def f(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select a from table1 limit @a", {"a": 10})

    df = basic_df["table1"]
    py_output = pd.DataFrame({"a": df.A.head(10)})
    if bodo.get_rank() == 0:
        # The warning is only produced on rank 0
        warning_prefix = re.escape(
            "The following named parameters: ['a'] were typed as literals"
        )
        with pytest.warns(bodosql.utils.BodoSQLWarning, match=warning_prefix):
            sql_output = f(df)
    else:
        sql_output = f(df)

    pd.testing.assert_frame_equal(sql_output, py_output)
