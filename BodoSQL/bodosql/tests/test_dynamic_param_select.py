"""
Test that Dynamic Parameters can be used in select expressions.
"""

import numpy as np
import pandas as pd

from bodosql.tests.named_params_common import (  # noqa
    named_params_all_column_types,
)
from bodosql.tests.utils import check_query


def test_select_named_param(
    basic_df, spark_info, named_params_all_column_types, memory_leak_check
):
    """
    Tests that selects works with named parameters
    """
    query = """
        SELECT
            @a as col1, @b as col2
        FROM
            table1
        """
    timedelta_columns = (
        ["col1", "col2"]
        if isinstance(named_params_all_column_types["a"], pd.Timedelta)
        else None
    )
    decimal_columns = (
        ["col1", "col2"]
        if isinstance(named_params_all_column_types["a"], float | np.floating)
        else None
    )
    check_query(
        query,
        basic_df,
        spark_info,
        convert_columns_decimal=decimal_columns,
        named_params=named_params_all_column_types,
        check_dtype=False,
        check_names=False,
        convert_columns_timedelta=timedelta_columns,
    )


def test_named_param_mixed_column_scalar(
    basic_df, spark_info, named_params_all_column_types, memory_leak_check
):
    """
    Tests that a mix of named parameters and columns work as expected.
    """
    query = """
        SELECT
            @a as col1, B as col2
        FROM
            table1
        """
    timedelta_columns = (
        ["col1"]
        if isinstance(named_params_all_column_types["a"], pd.Timedelta)
        else None
    )
    decimal_columns = (
        ["col1"]
        if isinstance(named_params_all_column_types["a"], float | np.floating)
        else None
    )
    check_query(
        query,
        basic_df,
        spark_info,
        convert_columns_decimal=decimal_columns,
        named_params=named_params_all_column_types,
        check_dtype=False,
        check_names=False,
        convert_columns_timedelta=timedelta_columns,
    )


def test_select_bind_variables(
    basic_df, named_params_all_column_types, memory_leak_check
):
    """
    Tests that selects works with bind variables.
    """
    query = """
        SELECT
            ? as COL1, ? as COL2
        FROM
            table1
        """
    bind_variables = (
        named_params_all_column_types["a"],
        named_params_all_column_types["b"],
    )
    num_rows = len(basic_df["TABLE1"])
    expected_output = pd.DataFrame(
        {"COL1": [bind_variables[0]] * num_rows, "COL2": [bind_variables[1]] * num_rows}
    )
    check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )


def test_bind_variables_mixed_column_scalar(
    basic_df, named_params_all_column_types, memory_leak_check
):
    """
    Tests that a mix of bind variables and columns work as expected.
    """
    query = """
        SELECT
            ? as COL1, B as col2
        FROM
            table1
        """
    bind_variables = (
        named_params_all_column_types["a"],
        named_params_all_column_types["b"],
    )
    num_rows = len(basic_df["TABLE1"])
    expected_output = pd.DataFrame(
        {"COL1": [bind_variables[0]] * num_rows, "COL2": basic_df["TABLE1"]["B"]}
    )
    check_query(
        query,
        basic_df,
        None,
        bind_variables=bind_variables,
        check_dtype=False,
        expected_output=expected_output,
    )
