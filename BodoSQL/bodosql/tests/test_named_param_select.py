"""
Test that Named Parameters can be used in select expressions.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import numpy as np
import pandas as pd
from bodosql.tests.named_params_common import (  # noqa
    named_params_all_column_types,
)
from bodosql.tests.utils import check_query


def test_select_named_param(basic_df, spark_info, named_params_all_column_types):
    """
    Tests that selects works with named parameters
    """
    query = f"""
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
        if isinstance(named_params_all_column_types["a"], (float, np.floating))
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


def test_mixed_column_scalar(basic_df, spark_info, named_params_all_column_types):
    """
    Tests that a mix of scalar and columns in result work
    """
    query = f"""
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
        if isinstance(named_params_all_column_types["a"], (float, np.floating))
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
