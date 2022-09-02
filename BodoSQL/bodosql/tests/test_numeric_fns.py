"""
Test that various numeric builtin functions are properly supported in BODOSQL
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.


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
        ("CBRT", "CBRT", "mixed_floats"),
        ("CBRT", "CBRT", "mixed_ints"),
        ("FACTORIAL", "FACTORIAL", "positive_ints"),
        ("FLOOR", "FLOOR", "mixed_floats"),
        ("ROUND", "ROUND", "mixed_floats"),
        ("ROUND", "ROUND", "mixed_ints"),
        ("SIGN", "SIGN", "mixed_floats"),
        ("SIGN", "SIGN", "mixed_ints"),
        # the second argument to POW for SQUARE (2) is provided below
        ("SQUARE", "POW", "mixed_floats"),
        ("SQUARE", "POW", "mixed_ints"),
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
        ("TRUNC", "ROUND", "mixed_floats", "3"),
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
    if fn_name == "SQUARE":
        if arg1[-5:] == "_ints" and any(
            bodosql_negative_numeric_types["table1"].dtypes == np.int8
        ):
            spark_query = (
                f"SELECT CAST({spark_fn_name}({arg1}, 2) AS TINYINT) from table1"
            )
        else:
            spark_query = f"SELECT {spark_fn_name}({arg1}, 2) from table1"
    else:
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
    spark_query = query
    if fn_name == "TRUNC" or fn_name == "TRUNCATE":
        inner_case = f"(CASE WHEN {arg1} > 0 THEN FLOOR({arg1} * POW(10, {arg2})) / POW(10, {arg2}) ELSE CEIL({arg1} * POW(10, {arg2})) / POW(10, {arg2}) END)"
        spark_query = f"SELECT {inner_case} from table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
    )


@pytest.mark.parametrize(
    "query_args",
    [
        pytest.param(("A", "B", "C", "D"), id="all_vector"),
        pytest.param(("A", "-2", "12", "20"), id="vector_scalar"),
        pytest.param(("20", "B", "C", "D"), id="scalar_vector"),
        pytest.param(("0.5", "-0.5", "2", "12"), id="all_scalar"),
    ],
)
def test_width_bucket_cols(query_args, spark_info, memory_leak_check):
    t0 = pd.DataFrame(
        {
            "A": [-1, -0.5, 0, 0.5, 1, 2.5, None, 10, 15, 200],
            "B": [0, 0, 0, 0, 1, 1, None, -1, 2, 20],
            "C": [2, 2, 2, None, 3, 4, 4, 5, 10, 300],
            "D": pd.Series([2, None, 2, 2, 4, 5, 10, 20, 5, 20], dtype="Int32"),
        }
    )
    ctx = {"table0": t0}
    A, B, C, D = query_args
    query = f"SELECT WIDTH_BUCKET({A}, {B}, {C}, {D}) from table0"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


def test_width_bucket_scalars(spark_info, memory_leak_check):
    t0 = pd.DataFrame(
        {
            "A": [-1, -0.5, 0, 0.5, 1, 2.5, None, -2, 15, 200],
            "B": [-2, -1, 0, 0, 1, 1, None, -1, 2, 20],
            "C": [2, 2, 2, None, 3, 4, 4, 5, 10, 300],
            "D": pd.Series([2, None, 2, 2, 4, 5, 10, 20, 5, 20], dtype="Int32"),
        }
    )
    ctx = {"table0": t0}
    query = f"SELECT CASE WHEN A <= 0.0 THEN WIDTH_BUCKET(-A, B, C, D) ELSE WIDTH_BUCKET(A, B, C, 2*D) END FROM table0"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "query_args",
    [
        pytest.param(("A", "B", "C", "D"), id="all_vector"),
        pytest.param(
            ("A", "142.78966505413766", "3.7502297731338663", "D"), id="scalar_vector"
        ),
        pytest.param(
            (
                "70.80417858598695",
                "-8.853993015501311",
                "139.9669747821279",
                "-43.29468080693516",
            ),
            id="all_scalar",
        ),
    ],
)
def test_haversine_cols(query_args, spark_info, memory_leak_check):
    ctx = {
        "table0": pd.DataFrame(
            {
                "A": [
                    7.7784067526128275,
                    20.87186811910824,
                    -69.00052792241254,
                    -105.7424091178466,
                    160.27692403891982,
                    -79.0359589304318,
                    -157.73081325445796,
                    -129.67223135825714,
                    -63.614597645943014,
                    -9.860404086579484,
                ],
                "B": [
                    -16.882366885350347,
                    -125.47584643591107,
                    174.6591236400272,
                    -1.4210895317999706,
                    -44.35890275974883,
                    9.634780842740671,
                    -105.48041299330141,
                    None,
                    19.318514807820073,
                    -47.91512496347812,
                ],
                "C": [
                    -47.319485128431594,
                    142.78966505413766,
                    None,
                    -27.30646075379275,
                    151.58354214722291,
                    3.7502297731338663,
                    -109.64993293339201,
                    -118.41046850400079,
                    69.93292834796553,
                    93.08586960310635,
                ],
                "D": [
                    70.80417858598695,
                    -8.853993015501311,
                    139.9669747821279,
                    -43.29468080693516,
                    145.21628623385664,
                    None,
                    76.16981197115778,
                    69.89499853983904,
                    98.42455790747928,
                    65.19331155605875,
                ],
            }
        )
    }
    LAT1, LON1, LAT2, LON2 = query_args
    query = f"select haversine({LAT1}, {LON1}, {LAT2}, {LON2}) from table0"
    equiv_query = f"SELECT 2 * 6371 * ASIN(SQRT(POW(SIN((RADIANS({LAT2}) - RADIANS({LAT1})) / 2),2) + (COS(RADIANS({LAT1})) * COS(RADIANS({LAT2})) * POW(SIN((RADIANS({LON2}) - RADIANS({LON1})) / 2),2)))) FROM table0"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=equiv_query,
    )


def test_haversine_scalars(spark_info, memory_leak_check):
    ctx = {
        "table0": pd.DataFrame(
            {
                "A": [
                    7.7784067526128275,
                    20.87186811910824,
                    -69.00052792241254,
                    -105.7424091178466,
                    160.27692403891982,
                    -79.0359589304318,
                    -157.73081325445796,
                    -129.67223135825714,
                    -63.614597645943014,
                    -9.860404086579484,
                ],
                "B": [
                    -16.882366885350347,
                    -125.47584643591107,
                    174.6591236400272,
                    -1.4210895317999706,
                    -44.35890275974883,
                    9.634780842740671,
                    -105.48041299330141,
                    None,
                    19.318514807820073,
                    -47.91512496347812,
                ],
                "C": [
                    -47.319485128431594,
                    142.78966505413766,
                    None,
                    -27.30646075379275,
                    151.58354214722291,
                    3.7502297731338663,
                    -109.64993293339201,
                    -118.41046850400079,
                    69.93292834796553,
                    93.08586960310635,
                ],
                "D": [
                    70.80417858598695,
                    -8.853993015501311,
                    139.9669747821279,
                    -43.29468080693516,
                    145.21628623385664,
                    None,
                    76.16981197115778,
                    69.89499853983904,
                    98.42455790747928,
                    65.19331155605875,
                ],
            }
        )
    }
    query = (
        f"select case when A < 0.0 then haversine(A, B, C, D) else 0.0 end from table0"
    )
    equiv_query = f"SELECT CASE WHEN A < 0.0 THEN 2 * 6371 * ASIN(SQRT(POW(SIN((RADIANS(C) - RADIANS(A)) / 2),2) + (COS(RADIANS(A)) * COS(RADIANS(C)) * POW(SIN((RADIANS(D) - RADIANS(B)) / 2),2)))) ELSE 0.0 END FROM table0"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=equiv_query,
    )


def test_haversine_calc(spark_info, memory_leak_check):
    ctx = {
        "table0": pd.DataFrame(
            {
                "A": [
                    7.7784067526128275,
                    20.87186811910824,
                    -69.00052792241254,
                    -105.7424091178466,
                    160.27692403891982,
                    -79.0359589304318,
                    -157.73081325445796,
                    -129.67223135825714,
                    -63.614597645943014,
                    -9.860404086579484,
                ],
                "B": [
                    -16.882366885350347,
                    -125.47584643591107,
                    174.6591236400272,
                    -1.4210895317999706,
                    -44.35890275974883,
                    9.634780842740671,
                    -105.48041299330141,
                    None,
                    19.318514807820073,
                    -47.91512496347812,
                ],
                "C": [
                    -47.319485128431594,
                    142.78966505413766,
                    None,
                    -27.30646075379275,
                    151.58354214722291,
                    3.7502297731338663,
                    -109.64993293339201,
                    -118.41046850400079,
                    69.93292834796553,
                    93.08586960310635,
                ],
                "D": [
                    70.80417858598695,
                    -8.853993015501311,
                    139.9669747821279,
                    -43.29468080693516,
                    145.21628623385664,
                    None,
                    76.16981197115778,
                    69.89499853983904,
                    98.42455790747928,
                    65.19331155605875,
                ],
            }
        )
    }
    query = f"select haversine(A + B, B - C, C + D, D - A) from table0"
    equiv_query = f"SELECT 2 * 6371 * ASIN(SQRT(POW(SIN((RADIANS(C + D) - RADIANS(A + B)) / 2),2) + (COS(RADIANS(A + B)) * COS(RADIANS(C + D)) * POW(SIN((RADIANS(D - A) - RADIANS(B - C)) / 2),2)))) FROM table0"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=equiv_query,
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
    if fn_name == "SQUARE":
        if arg1[-5:] == "_ints" and any(
            bodosql_negative_numeric_types["table1"].dtypes == np.int8
        ):
            spark_query = f"SELECT CASE when CAST({spark_fn_name}({arg1}, 2) AS TINYINT) = 0 then 1 ELSE CAST({spark_fn_name}({arg1}, 2)  AS TINYINT) END from table1"
        else:
            spark_query = f"SELECT CASE when {spark_fn_name}({arg1}, 2) = 0 then 1 ELSE {spark_fn_name}({arg1}, 2) END from table1"
    else:
        spark_query = f"SELECT CASE when {spark_fn_name}({arg1}) = 0 then 1 ELSE {spark_fn_name}({arg1}) END from table1"

    query = f"SELECT CASE when {fn_name}({arg1}) = 0 then 1 ELSE {fn_name}({arg1}) END from table1"
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
    if fn_name == "TRUNC" or fn_name == "TRUNCATE":
        inner_case = f"(CASE WHEN {arg1} > 0 THEN FLOOR({arg1} * POW(10, {arg2})) / POW(10, {arg2}) ELSE CEIL({arg1} * POW(10, {arg2})) / POW(10, {arg2}) END)"
        spark_query = f"SELECT CASE when {inner_case} = 0 then 1 ELSE {inner_case} END from table1"
    elif fn_name == "MOD":
        # for now we skip mod tests due to [BE-3533] due to issues casting optional types
        # (mod will return an optional type since we may receive a value of 0
        # for the second arg and MOD(x, 0) is None)
        return
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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(("A", "B"), id="all_vector"),
        pytest.param(("A", "0.0"), id="vector_scalar"),
        pytest.param(("100", "B"), id="scalar_vector"),
        pytest.param(("72.0", "0.0"), id="all_scalar"),
    ],
)
def test_div0_cols(args, spark_info, memory_leak_check):
    ctx = {}
    ctx["table1"] = pd.DataFrame(
        {
            "A": [10.0, 12, np.nan, 32, 24, np.nan, 8, np.nan, 14, 28],
            "B": [1.0, 0, 4, 0, 2, 0, 3, 0, np.nan, 0],
        }
    )
    A, B = args
    query = f"select div0({A}, {B}) from table1"
    # TODO: Spark does not interpret NaNs as NULL, but we do (from Pandas behavior).
    # The following is equiv spark query of above.
    spark_query = f"select case when ((({A} is not NULL) and (not isnan({A}))) and ({B} = 0)) then 0 else ({A} / {B}) end from table1"
    check_query(
        query,
        ctx,
        spark_info,
        equivalent_spark_query=spark_query,
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )


def test_div0_scalars(spark_info):
    df = pd.DataFrame(
        {
            "A": [10.0, 12, np.nan, 32, 24, np.nan, 8, np.nan, 14, 28],
            "B": [1.0, -12, 4, 0, 2, 0, -8, 0, np.nan, 0],
        }
    )
    ctx = {"table1": df}

    def _py_output(df):
        a, b = df["A"], df["B"]
        ret = np.empty(a.size)
        sum_ = a + b
        ret[a > b] = ((a - b) / sum_)[a > b]
        ret[b > a] = ((b - a) / sum_)[b > a]
        ret[pd.isna(a) | pd.isna(b)] = np.nan
        ret[sum_ == 0] = 0
        return pd.DataFrame({"out": ret})

    output = _py_output(df)
    query = (
        "SELECT CASE WHEN A > B THEN DIV0(A-B, A+B) ELSE DIV0(B-A, A+B) END FROM table1"
    )
    check_query(
        query,
        ctx,
        spark_info,
        expected_output=output,
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )
