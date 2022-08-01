"""
Test that various numeric builtin functions are properly supported in BODOSQL
"""
# Copyright (C) 2021 Bodo Inc. All rights reserved.


import numpy as np
import pandas as pd
import pytest

from bodosql.tests.utils import check_query


@pytest.fixture(
    params=[
        pytest.param((np.int8, np.float32), marks=pytest.mark.slow),
        pytest.param((np.int16, np.float32), marks=pytest.mark.slow),
        pytest.param((np.int32, np.float32), marks=pytest.mark.slow),
        pytest.param((np.int8, np.float64), marks=pytest.mark.slow),
        pytest.param((np.int16, np.float64), marks=pytest.mark.slow),
        pytest.param((np.int32, np.float64), marks=pytest.mark.slow),
        (np.int64, np.float64),
    ]
)
def bodosql_negative_numeric_types(request):
    """
    Fixture for dataframes with negative numeric BodoSQL types:


    """
    int_dtype = request.param[0]
    float_dtype = request.param[1]

    numeric_data = {
        "positive_ints": pd.Series([1, 2, 3, 4, 5, 6] * 2, dtype=int_dtype),
        "mixed_ints": pd.Series([-7, 8, -9, 10, -11, 12] * 2, dtype=int_dtype),
        "negative_ints": pd.Series([-13, -14, -15] * 4, dtype=int_dtype),
        "positive_floats": pd.Series(
            [1.2, 0.2, 0.03, 4.0, 0.001, 0.666] * 2, dtype=float_dtype
        ),
        "mixed_floats": pd.Series(
            [-0.7, 0.0, -9.223, 1.0, -0.11, 12.12] * 2, dtype=float_dtype
        ),
        "negative_floats": pd.Series([-13.0, -14.022, -1.5] * 4, dtype=float_dtype),
    }
    return {"table1": pd.DataFrame(numeric_data)}


@pytest.fixture(
    params=[
        {
            "table1": pd.DataFrame(
                {
                    "A": ["0", "1", "10", "01011", "111011", "1101011011"],
                    "B": ["0", "1", "10", "72121", "72121", "101101"],
                    "C": ["0", "1", "10", "8121", "12312", "33190"],
                    "D": ["0", "1", "10", "9AF12D", "1FF1B", "1AB021"],
                }
            )
        }
    ]
)
def bodosql_conv_df(request):
    """returns datframes used for testing conv
    A is in binary,
    B is in octal,
    C is in decimal,
    D is in hex,
    """
    return request.param


@pytest.fixture(
    params=[
        ("ABS", "ABS", "mixed_ints"),
        ("ABS", "ABS", "mixed_floats"),
        ("CEIL", "CEIL", "mixed_floats"),
        ("FLOOR", "FLOOR", "mixed_floats"),
        ("SIGN", "SIGN", "mixed_floats"),
        ("SIGN", "SIGN", "mixed_ints"),
        ("ROUND", "ROUND", "mixed_floats"),
        ("ROUND", "ROUND", "mixed_ints"),
    ]
    + [(x, x, "positive_floats") for x in ["LOG10", "LOG2", "LN", "EXP", "SQRT"]]
    + [
        ("LOG", "LOG10", "positive_floats"),
    ]
    # currently, behavior for log(0) differs from sparks behavior, see BS-374
    # + [(x, x, "negative_floats") for x in ["LOG10", "LOG2", "LN", "EXP", "SQRT"]]
)
def single_op_numeric_fn_info(request):
    """fixture that returns information to test a single operand function call that uses the
    bodosql_negative_numeric_types fixture.
    parameters are a tuple consisting of the string function name, the equivalent function name in Spark,
    and what columns/scalar to use as its argument"""
    return request.param


@pytest.fixture(
    params=[
        ("MOD", "MOD", "mixed_floats", "mixed_floats"),
        ("TRUNCATE", "ROUND", "mixed_floats", "3"),
    ]
    + [("ROUND", "ROUND", "mixed_floats", x) for x in ["2", "1", "3"]]
    + [
        ("POW", "POW", "positive_floats", "mixed_floats"),
        ("POWER", "POWER", "positive_floats", "mixed_floats"),
        ("POW", "POW", "mixed_floats", "mixed_ints"),
        ("POW", "POW", "mixed_floats", "mixed_floats"),
    ]
)
def double_op_numeric_fn_info(request):
    """fixture that returns information to test a double operand function call that uses the
    bodosql_negative_numeric_types fixture.
    parameters are a tuple consisting ofthe string function name, the equivalent function name in Spark,
    and what two columns/scalars to use as its arguments"""
    return request.param


def test_single_op_numeric_fns_cols(
    single_op_numeric_fn_info,
    bodosql_negative_numeric_types,
    spark_info,
    memory_leak_check,
):
    """tests the behavior of numeric functions with a single argument on columns"""
    fn_name = single_op_numeric_fn_info[0]
    spark_fn_name = single_op_numeric_fn_info[1]
    arg1 = single_op_numeric_fn_info[2]
    query = f"SELECT {fn_name}({arg1}) from table1"
    spark_query = f"SELECT {spark_fn_name}({arg1}) from table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_double_op_numeric_fns_cols(
    double_op_numeric_fn_info,
    bodosql_negative_numeric_types,
    spark_info,
    memory_leak_check,
):
    """tests the behavior of numeric functions with two arguments on columns"""
    fn_name = double_op_numeric_fn_info[0]
    spark_fn_name = double_op_numeric_fn_info[1]
    arg1 = double_op_numeric_fn_info[2]
    arg2 = double_op_numeric_fn_info[3]
    query = f"SELECT {fn_name}({arg1}, {arg2}) from table1"
    spark_query = f"SELECT {spark_fn_name}({arg1}, {arg2}) from table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_single_op_numeric_fns_scalars(
    single_op_numeric_fn_info,
    bodosql_negative_numeric_types,
    spark_info,
    memory_leak_check,
):
    """tests the behavior of numeric functions with a single argument on scalar values"""
    fn_name = single_op_numeric_fn_info[0]
    spark_fn_name = single_op_numeric_fn_info[1]
    arg1 = single_op_numeric_fn_info[2]
    query = f"SELECT CASE when {fn_name}({arg1}) = 0 then 1 ELSE {fn_name}({arg1}) END from table1"
    spark_query = f"SELECT CASE when {spark_fn_name}({arg1}) = 0 then 1 ELSE {spark_fn_name}({arg1}) END from table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.slow
def test_double_op_numeric_fns_scalars(
    double_op_numeric_fn_info,
    bodosql_negative_numeric_types,
    spark_info,
    memory_leak_check,
):
    """tests the behavior of numeric functions with two arguments on scalar values"""
    fn_name = double_op_numeric_fn_info[0]
    spark_fn_name = double_op_numeric_fn_info[1]
    arg1 = double_op_numeric_fn_info[2]
    arg2 = double_op_numeric_fn_info[3]
    query = f"SELECT CASE when {fn_name}({arg1}, {arg2}) = 0 then 1 ELSE {fn_name}({arg1}, {arg2}) END from table1"
    spark_query = f"SELECT CASE when {spark_fn_name}({arg1}, {arg2}) = 0 then 1 ELSE {spark_fn_name}({arg1}, {arg2}) END from table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


def test_rand(basic_df, spark_info, memory_leak_check):
    """tests the behavior of rand"""
    query = "Select (A >= 0.0 AND A < 1.0) as cond, B from (select RAND() as A, B from table1)"
    # Currenly having an issue when running as distributed, see BS-383
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_python=True,
    )


def test_conv_columns(bodosql_conv_df, spark_info, memory_leak_check):
    """tests that the CONV function works as intended for columns"""
    query = "SELECT CONV(A, 2, 10), CONV(B, 8, 2), CONV(C, 10, 10), CONV(D, 16, 8) from table1"
    check_query(
        query,
        bodosql_conv_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.slow
def test_conv_scalars(bodosql_conv_df, spark_info, memory_leak_check):
    """tests that the CONV function works as intended for scalars"""
    query = (
        "SELECT CASE WHEN A > B THEN CONV(A, 2, 10) ELSE CONV(B, 8, 10) END from table1"
    )
    check_query(
        query,
        bodosql_conv_df,
        spark_info,
        check_dtype=False,
        check_names=False,
    )


@pytest.mark.parametrize(
    "query",
    [
        pytest.param("SELECT LOG(A, B) FROM table1", id="all_vector"),
        pytest.param("SELECT LOG(A, 2.0) FROM table1", id="vector_scalar"),
        pytest.param("SELECT LOG(100, B) FROM table1", id="scalar_vector"),
        pytest.param("SELECT LOG(72.0, 2.0) FROM table1", id="all_scalar"),
    ],
)
def test_log_hybrid(query, spark_info):
    """Testing log seperately since it reverses the order of the arguments"""
    ctx = {
        "table1": pd.DataFrame(
            {"A": [1.0, 2.0, 0.5, 64.0, 100.0], "B": [2.0, 3.0, 4.0, 5.0, 10.0]}
        )
    }
    # Spark switches the order of the arguments
    lhs, rest = query.split("(")
    args, rhs = rest.split(")")
    arg0, arg1 = args.split(", ")
    spark_query = f"{lhs}({arg1}, {arg0}){rhs}"
    check_query(
        query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )
