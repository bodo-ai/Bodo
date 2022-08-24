"""
Test that various builtin trig functions are properly supported in BODOSQL
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


import numpy as np
import pandas as pd
import pytest
from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param(np.float32, marks=pytest.mark.slow),
        np.float64,
    ]
)
def bodosql_trig_values(request):
    """certain Trig functions are only valid for a given range of inputs."""
    dtype = request.param
    trig_data = {
        "abs_lteq_1": [0, 1, -1, 0.5, -0.5, 0.3212, -0.78, None],
        "any_numeric": pd.Series(
            [0, 1, -1, 10000, -100000, 10, -15, None], dtype="Int64"
        ),
        "any_non_zero": [1, -1, 0.1, -0.1, 1234, -4321, 3, None],
    }
    return {"table1": pd.DataFrame(data=trig_data, dtype=dtype)}


@pytest.fixture(
    params=[
        (x, "any_numeric")
        for x in [
            "ACOS",
            "ACOSH",
            "ASIN",
            "ASINH",
            "COS",
            "COSH",
            "SIN",
            "SINH",
            "TAN",
            "TANH",
            "ATAN",
            "ATANH",
            "RADIANS",
            "DEGREES",
            "COT",
        ]
    ]
)
def single_op_trig_fn_info(request):
    """fixture that returns information to test a single operand function call that uses the
    bodosql_trig_values fixture.
    parameters are a tuple consisting of the string function name,
    and what column/scalar to use as its arguments"""
    return request.param


@pytest.fixture(params=[("ATAN2", "any_numeric", "any_numeric")])
def double_op_trig_fn_info(request):
    """fixture that returns information to test a two operand function call that uses the
    bodosql_trig_values fixture.
    parameters are a tuple consisting of the string function name,
    and what two columns/scalars to use as its arguments"""
    return request.param


def test_single_op_trig_fns_cols(
    single_op_trig_fn_info, bodosql_trig_values, spark_info, memory_leak_check
):
    """tests the behavior of trig functions with a single argument on columns"""
    fn_name = single_op_trig_fn_info[0]
    arg1 = single_op_trig_fn_info[1]
    query = f"SELECT {fn_name}({arg1}) from table1"
    check_query(
        query,
        bodosql_trig_values,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_double_op_trig_fns_cols(
    double_op_trig_fn_info, bodosql_trig_values, spark_info, memory_leak_check
):
    """tests the behavior of trig functions with two arguments on columns"""
    fn_name = double_op_trig_fn_info[0]
    arg1 = double_op_trig_fn_info[1]
    arg2 = double_op_trig_fn_info[2]
    query = f"SELECT {fn_name}({arg1}, {arg2}) from table1"
    check_query(
        query,
        bodosql_trig_values,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_single_op_trig_fns_scalars(
    single_op_trig_fn_info, bodosql_trig_values, spark_info, memory_leak_check
):
    """tests the behavior of trig functions with a single argument on scalars"""
    fn_name = single_op_trig_fn_info[0]
    arg1 = single_op_trig_fn_info[1]
    query = f"SELECT CASE when {fn_name}({arg1}) = 0 then 1 ELSE {fn_name}({arg1}) END from table1"
    check_query(
        query,
        bodosql_trig_values,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.slow
def test_double_op_trig_fns_scalars(
    double_op_trig_fn_info, bodosql_trig_values, spark_info, memory_leak_check
):
    """tests the behavior of trig functions with two arguments on scalars"""
    fn_name = double_op_trig_fn_info[0]
    arg1 = double_op_trig_fn_info[1]
    arg2 = double_op_trig_fn_info[2]
    query = f"SELECT CASE when {fn_name}({arg1}, {arg2}) = 0 then 1 ELSE {fn_name}({arg1}, {arg2}) END from table1"
    check_query(
        query,
        bodosql_trig_values,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_pi(basic_df, spark_info, memory_leak_check):
    """tests that the pi fn is working"""
    query = "SELECT A, PI from table1"
    spark_query = "SELECT A, PI() from table1"
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        equivalent_spark_query=spark_query,
    )
