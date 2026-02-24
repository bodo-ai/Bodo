"""
Test that various numeric builtin functions are properly supported in BODOSQL
"""

import sys
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import pytest_slow_unless_codegen
from bodosql.tests.utils import check_query

# Skip unless any codegen files were changed
pytestmark = pytest_slow_unless_codegen


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
    Fixture for DataFrames with negative numeric BodoSQL types:
    """
    if sys.platform == "win32":
        pytest.skip("Spark does not support unsigned int columns on Windows.")

    int_dtype = request.param[0]
    float_dtype = request.param[1]
    numeric_data = {
        "POSITIVE_INTS": pd.Series([1, 2, 3, 4, 5, 6] * 2, dtype=int_dtype),
        "UNSIGNED_INT32S": pd.Series([1, 3, 7, 14, 0, 11] * 2, dtype=np.uint32),
        "UNSIGNED_INT64S": pd.Series([12, 2**50, 0, 78, 390] * 2, dtype="UInt64"),
        "MIXED_INTS": pd.Series([-7, 8, -9, 10, -11, 12] * 2, dtype=int_dtype),
        "NEGATIVE_INTS": pd.Series([-13, -14, -15] * 4, dtype=int_dtype),
        "POSITIVE_FLOATS": pd.Series(
            [1.2, 0.2, 0.03, 4.0, 0.001, 0.666] * 2, dtype=float_dtype
        ),
        "MIXED_FLOATS": pd.Series(
            [-0.7, 0.0, -9.223, 1.0, -0.11, 12.12] * 2, dtype=float_dtype
        ),
        "NEGATIVE_FLOATS": pd.Series([-13.0, -14.022, -1.5] * 4, dtype=float_dtype),
    }
    return {"TABLE1": pd.DataFrame(numeric_data)}


@pytest.fixture(
    params=[
        {
            "TABLE1": pd.DataFrame(
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
    """returns DataFrames used for testing conv
    A is in binary,
    B is in octal,
    C is in decimal,
    D is in hex,
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(("ABS", "ABS", "MIXED_INTS"), marks=pytest.mark.slow),
        ("ABS", "ABS", "MIXED_FLOATS"),
        pytest.param(("CBRT", "CBRT", "MIXED_FLOATS"), marks=pytest.mark.slow),
        ("CBRT", "CBRT", "MIXED_INTS"),
        pytest.param(
            ("FACTORIAL", "FACTORIAL", "POSITIVE_INTS"), marks=pytest.mark.slow
        ),
        pytest.param(("SIGN", "SIGN", "MIXED_FLOATS"), marks=pytest.mark.slow),
        ("SIGN", "SIGN", "MIXED_INTS"),
        # the second argument to POW for SQUARE (2) is provided below
        pytest.param(("SQUARE", "POW", "MIXED_FLOATS"), marks=pytest.mark.slow),
        ("SQUARE", "POW", "MIXED_INTS"),
    ]
    + [(x, x, "POSITIVE_FLOATS") for x in ["LOG10", "LOG2", "LN", "EXP", "SQRT"]]
    + [
        pytest.param(("LOG", "LOG10", "POSITIVE_FLOATS"), marks=pytest.mark.slow),
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
        ("MOD", "MOD", "MIXED_FLOATS", "MIXED_FLOATS"),
        pytest.param(
            ("MOD", "MOD", "MIXED_INTS", "MIXED_INTS"), marks=pytest.mark.slow
        ),
        ("MOD", "MOD", "MIXED_INTS", "MIXED_FLOATS"),
        pytest.param(
            ("MOD", "MOD", "UNSIGNED_INT64S", "UNSIGNED_INT32S"), marks=pytest.mark.slow
        ),
        ("POW", "POW", "POSITIVE_FLOATS", "MIXED_FLOATS"),
        pytest.param(
            ("POWER", "POWER", "POSITIVE_FLOATS", "MIXED_FLOATS"),
            marks=pytest.mark.slow,
        ),
        ("POW", "POW", "MIXED_FLOATS", "MIXED_INTS"),
        pytest.param(
            ("POW", "POW", "MIXED_FLOATS", "MIXED_FLOATS"), marks=pytest.mark.slow
        ),
    ]
)
def double_op_numeric_fn_info(request):
    """fixture that returns information to test a double operand function call that uses the
    bodosql_negative_numeric_types fixture.
    parameters are a tuple consisting of the string function name, the equivalent function name in Spark,
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
        if arg1[-5:] == "_INTS" and any(
            bodosql_negative_numeric_types["TABLE1"].dtypes == np.int8
        ):
            spark_query = (
                f"SELECT CAST({spark_fn_name}({arg1}, 2) AS DOUBLE) from table1"
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
        convert_expected_output_to_nullable_float=False,
    )


@pytest.mark.parametrize(
    "query_args",
    [
        pytest.param(("A", "B", "C", "D"), id="all_vector"),
        pytest.param(
            ("A", "-2", "12", "20"), id="vector_scalar", marks=pytest.mark.slow
        ),
        pytest.param(("20", "B", "C", "D"), id="scalar_vector"),
        pytest.param(
            ("0.5", "-0.5", "2", "12"), id="all_scalar", marks=pytest.mark.slow
        ),
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
    ctx = {"TABLE0": t0}
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
    ctx = {"TABLE0": t0}
    query = "SELECT CASE WHEN A <= 0.0 THEN WIDTH_BUCKET(-A, B, C, D) ELSE WIDTH_BUCKET(A, B, C, 2*D) END FROM table0"
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
            ("A", "142.78966505413766", "3.7502297731338663", "D"),
            id="scalar_vector",
            marks=pytest.mark.slow,
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
        "TABLE0": pd.DataFrame(
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


@pytest.mark.slow
def test_haversine_scalars(spark_info, memory_leak_check):
    ctx = {
        "TABLE0": pd.DataFrame(
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
        "select case when A < 0.0 then haversine(A, B, C, D) else 0.0 end from table0"
    )
    equiv_query = "SELECT CASE WHEN A < 0.0 THEN 2 * 6371 * ASIN(SQRT(POW(SIN((RADIANS(C) - RADIANS(A)) / 2),2) + (COS(RADIANS(A)) * COS(RADIANS(C)) * POW(SIN((RADIANS(D) - RADIANS(B)) / 2),2)))) ELSE 0.0 END FROM table0"
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
        "TABLE0": pd.DataFrame(
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
    query = "select haversine(A + B, B - C, C + D, D - A) from table0"
    equiv_query = "SELECT 2 * 6371 * ASIN(SQRT(POW(SIN((RADIANS(C + D) - RADIANS(A + B)) / 2),2) + (COS(RADIANS(A + B)) * COS(RADIANS(C + D)) * POW(SIN((RADIANS(D - A) - RADIANS(B - C)) / 2),2)))) FROM table0"
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
        if arg1[-5:] == "_INTS" and any(
            bodosql_negative_numeric_types["TABLE1"].dtypes == np.int8
        ):
            spark_query = f"SELECT CASE WHEN CAST({spark_fn_name}({arg1}, 2) AS DOUBLE) = 0 THEN 1 ELSE CAST({spark_fn_name}({arg1}, 2)  AS DOUBLE) END FROM table1"
        else:
            spark_query = f"SELECT CASE WHEN {spark_fn_name}({arg1}, 2) = 0 THEN 1 ELSE {spark_fn_name}({arg1}, 2) END FROM table1"
    else:
        spark_query = f"SELECT CASE WHEN {spark_fn_name}({arg1}) = 0 THEN 1 ELSE {spark_fn_name}({arg1}) END FROM table1"

    query = f"SELECT CASE WHEN {fn_name}({arg1}) = 0 THEN 1 ELSE {fn_name}({arg1}) END FROM table1"
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
    query = f"SELECT CASE WHEN {fn_name}({arg1}, {arg2}) = 0 THEN -1 ELSE {fn_name}({arg1}, {arg2}) END FROM table1"
    spark_query = f"SELECT CASE when {spark_fn_name}({arg1}, {arg2}) = 0 THEN -1 ELSE {spark_fn_name}({arg1}, {arg2}) END from table1"
    if fn_name == "TRUNC" or fn_name == "TRUNCATE":
        inner_case = f"(CASE WHEN {arg1} > 0 THEN FLOOR({arg1} * POW(10, {arg2})) / POW(10, {arg2}) ELSE CEIL({arg1} * POW(10, {arg2})) / POW(10, {arg2}) END)"
        spark_query = f"SELECT CASE when {inner_case} = 0 THEN -1 ELSE {inner_case} END FROM table1"
    elif fn_name == "MOD":
        # MOD may return null depending on value passed, so EQUAL_NULL should be used, which properly handles null values, unlike '='.
        query = f"SELECT CASE WHEN EQUAL_NULL(MOD({arg1}, {arg2}), 0) THEN -1 ELSE {fn_name}({arg1}, {arg2}) END FROM table1"
    check_query(
        query,
        bodosql_negative_numeric_types,
        spark_info,
        check_names=False,
        check_dtype=False,
        equivalent_spark_query=spark_query,
        convert_expected_output_to_nullable_float=False,
    )


def test_rand(basic_df, spark_info, memory_leak_check):
    """tests the behavior of rand"""
    query = "Select (A >= 0.0 AND A < 1.0) as cond, B from (select RAND() as A, B from table1)"
    # Currently having an issue when running as distributed, see BS-383
    check_query(
        query,
        basic_df,
        spark_info,
        check_dtype=False,
        check_names=False,
        only_python=True,
    )


@pytest.mark.parametrize(
    "is_integer, use_case",
    [
        pytest.param(True, True, id="integer-with_case"),
        pytest.param(False, False, id="float-no_case"),
    ],
)
def test_uniform_distribution(is_integer, use_case, memory_leak_check):
    """Tests the behavior of UNIFORM by generating a million columns
    (with and without CASE) where the seed is distinct for each row then
    making sure that the values meet certain criteria:
    - The lower and upper bounds match the provided inputs LO and HI
    - [Integers] The entire domain is chosen at some point
    - [Floats] Each output is distinct
    - The average of the outputs is approximately (LO + HI) / 12
    - The standard deviation of the outputs is approximately sqrt((HI - LO) ** 2 / 12)
    - The skew of the outputs is approximately zero
    """
    lo = 0 if is_integer else 0.0
    hi = 99 if is_integer else 99.0
    if use_case:
        calculation = f"CASE WHEN A >= 0 THEN UNIFORM({lo}, {hi}, A) ELSE NULL END"
    else:
        calculation = f"UNIFORM({lo}, {hi}, A)"
    query = f"""
        WITH table2 AS (SELECT {calculation} AS U FROM table1)
        SELECT
            MIN(U),
            MAX(U),
            COUNT(DISTINCT U),
            AVG(U),
            STDDEV_POP(U),
            SKEW(U)
        FROM table2
    """
    n = 10**6
    ctx = {"TABLE1": pd.DataFrame({"A": np.arange(n)})}
    if is_integer:
        expected_distinct = 100
        expected_min = 0
        expected_max = 99
    else:
        expected_distinct = n
        expected_min = 0.0
        expected_max = 99.0
    answer = pd.DataFrame(
        {
            "MIN": expected_min,
            "MAX": expected_max,
            "DISTINCT": expected_distinct,
            "MEAN": 49.5,
            "STDV": 28.5788,
            "SKEW": 0.0,
        },
        index=np.arange(1),
    )
    np.random.seed(42)
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        atol=0.1,
        rtol=0.1,
        is_out_distributed=False,
    )


def test_uniform_determinism(memory_leak_check):
    """Tests the consistency of UNIFORM by generating rows based on a set of
    gen values (including some duplicates) to verify that it is consistent
    across multiple ranks, multiple sessions, and with duplicate values.

    The output values were generated by observing what np.random.randint
    outputs for the hardcoded input seeds.
    """
    query = "SELECT UNIFORM(0, 100, A) FROM table1"
    ctx = {"TABLE1": pd.DataFrame({"A": [1, 2, 3, 1000] * 10})}
    answer = pd.DataFrame({0: [37, 40, 24, 51] * 10})
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
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
        pytest.param(
            "SELECT LOG(100, B) FROM table1", id="scalar_vector", marks=pytest.mark.slow
        ),
        pytest.param(
            "SELECT LOG(72.0, 2.0) FROM table1", id="all_scalar", marks=pytest.mark.slow
        ),
    ],
)
def test_log_hybrid(query, spark_info, memory_leak_check):
    """Testing log seperately since it reverses the order of the arguments"""
    ctx = {
        "TABLE1": pd.DataFrame(
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
        pytest.param(("A", "B"), id="all_vector", marks=pytest.mark.slow),
        pytest.param(("A", "0.0"), id="vector_scalar"),
        pytest.param(("100", "B"), id="scalar_vector"),
        pytest.param(("72.0", "0.0"), id="all_scalar", marks=pytest.mark.slow),
    ],
)
def test_div0_cols(args, spark_info, memory_leak_check):
    ctx = {}
    ctx["TABLE1"] = pd.DataFrame(
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
    ctx = {"TABLE1": df}

    def _py_output(df):
        a, b = df["A"], df["B"]
        ret = np.empty(a.size)
        sum_ = a + b
        ret[a > b] = ((a - b) / sum_)[a > b]
        ret[b > a] = ((b - a) / sum_)[b > a]
        ret[pd.isna(a) | pd.isna(b)] = np.nan
        ret[sum_ == 0] = 0
        return pd.DataFrame({"OUT": ret})

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


@pytest.mark.parametrize(
    "df, ans",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array([10.0, 12, None, 32, 24, None, 8, None, 14, 28]),
                    "B": pd.array([1.0, -12, 4, 0, None, 0, -8, 0, None, 0]),
                }
            ),
            pd.array([10.0, -1.0, None, 0.0, 0.0, None, -1.0, None, 0.0, 0.0]),
            id="floats",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            Decimal("-1.2345"),
                            Decimal("-5.6789"),
                            Decimal("-2.9999"),
                            Decimal("-313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    ),
                    "B": pd.array(
                        [
                            Decimal("2.57"),
                            Decimal("0"),
                            None,
                            Decimal("5.25"),
                            Decimal("0"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    ),
                }
            ),
            pd.array(
                [
                    Decimal("-0.480350194553"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("-59.659458304762"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 14)),
            ),
            id="decimals",
        ),
    ],
)
def test_div0null_cols(df, ans, request):
    # Skip decimal test as it is not supported for COALESCE
    if request.node.callspec.id == "decimals":
        pytest.skip("[BSE-3740] Decimal type not supported for COALESCE")

    ctx = {"TABLE1": df}
    query = "SELECT DIV0NULL(A, B) AS RES FROM table1"
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"RES": ans}),
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "df, ans",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array([10.0, 12, None, 32, 24, None, 8, None, 14, 28]),
                    "B": pd.array([1.0, -12, 4, 0, None, 0, -8, 0, None, 0]),
                }
            ),
            pd.array([13.0, 15.0, None, 35.0, 0.0, None, 11.0, None, 0.0, 31.0]),
            id="floats",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": pd.array(
                        [
                            Decimal("-1.2345"),
                            Decimal("-5.6789"),
                            Decimal("-2.9999"),
                            Decimal("-313.2121561"),
                            None,
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    ),
                    "B": pd.array(
                        [
                            Decimal("2.57"),
                            Decimal("0"),
                            None,
                            Decimal("5.25"),
                            Decimal("0"),
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(13, 7)),
                    ),
                }
            ),
            pd.array(
                [
                    Decimal("1.7655"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("-310.2121561"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 14)),
            ),
            id="decimals",
        ),
    ],
)
def test_div0null_scalars(df, ans, request):
    # Skip decimal test as it is not supported for COALESCE
    if request.node.callspec.id == "decimals":
        pytest.skip("[BSE-3740] Decimal type not supported for COALESCE")

    ctx = {"TABLE1": df}
    query = "SELECT CASE WHEN B IS NULL THEN DIV0NULL(A+3, B) ELSE DIV0NULL(A+3, 1) END AS RES FROM table1"
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"RES": ans}),
        check_dtype=False,
        check_names=False,
        sort_output=False,
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        pytest.param("TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TO_DECIMAL", marks=pytest.mark.slow),
        "TRY_TO_NUMBER",
        pytest.param("TRY_TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TRY_TO_DECIMAL", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "values, expected_output",
    [
        pytest.param(["1"], 1, id="int", marks=pytest.mark.slow),
        pytest.param(["1.0"], 1, id="float"),
        pytest.param(
            ["1.5", "1", "0"], 2, id="float_with_scale", marks=pytest.mark.slow
        ),
        pytest.param(["'1.23456789'"], 1, id="str", marks=pytest.mark.slow),
        pytest.param(["'1.23456789'", "5", "4"], 1.2346, id="str_with_scale"),
        pytest.param(["NULL"], None, id="null", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, marks=pytest.mark.slow),
        True,
    ],
)
def test_to_number_scalar(fn_name, values, expected_output, use_case):
    args_string = ", ".join(values)
    if use_case and args_string != "NULL":
        query = f"SELECT (CASE WHEN {fn_name}({args_string}) > 0 THEN {fn_name}({args_string}) ELSE 0 END) as A"
    else:
        query = f"SELECT {fn_name}({args_string}) as A"
    ctx = {}
    check_query(
        query,
        ctx,
        None,
        expected_output=pd.DataFrame({"A": [expected_output]}),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        pytest.param("TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TO_DECIMAL", marks=pytest.mark.slow),
        "TRY_TO_NUMBER",
        pytest.param("TRY_TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TRY_TO_DECIMAL", marks=pytest.mark.slow),
    ],
)
def test_to_number_columns(fn_name):
    query = (
        f"SELECT {fn_name}(A) as A, {fn_name}(B) as B, {fn_name}(C) as C FROM table1"
    )

    df = pd.DataFrame(
        {
            "A": [str(i) for i in range(30)],
            "B": list(range(30)),
            "C": [float(i) + 0.2 for i in range(30)],
        }
    )

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=df.astype("int64"),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        pytest.param("TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TO_DECIMAL", marks=pytest.mark.slow),
        "TRY_TO_NUMBER",
        pytest.param("TRY_TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TRY_TO_DECIMAL", marks=pytest.mark.slow),
    ],
)
def test_to_number_columns_with_scale(fn_name):
    query = f"SELECT {fn_name}(A, 6) as A, {fn_name}(B, 3, 0) as B, {fn_name}(C, 10, 3) as C, {fn_name}(D, 10, 3) as D FROM table1"

    float_series = [
        123.456,
        10.3,
        1234567.123,
        None,
        -1234567.1237,
        1234567.1231,
        0.0,
        None,
        1.0,
        1234567.12399,
    ]
    # Manually converting to avoid string NaN issues
    str_float_series = [
        "123.456",
        "10.3",
        "1234567.123",
        None,
        "-1234567.1237",
        "1234567.1231",
        "0.0",
        None,
        "1.0",
        "1234567.12399",
    ]

    df = pd.DataFrame(
        {
            "A": [str(i) for i in range(10)],
            "B": [1 for i in range(10)],
            "C": float_series,
            "D": str_float_series,
        }
    )

    float_series_out = pd.Series(
        [
            123.456,
            10.3,
            1234567.123,
            None,
            -1234567.124,
            1234567.123,
            0.0,
            None,
            1.0,
            1234567.124,
        ]
    )
    expected_output = pd.DataFrame(
        {
            "A": df["A"].astype(pd.Int32Dtype()),
            "B": df["B"].astype(pd.Int16Dtype()),
            "C": float_series_out,
            "D": float_series_out,
        }
    )

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
        # From manual inspection, output df.dtypes == expected_output.dtypes,
        # but I still get a dtypes error from somewhere in _test_equal_guard:
        # Attributes of DataFrame.iloc[:, 0] (column name="A") are different
        # Attribute "dtype" are different
        # [left]:  object
        # [right]: Int32
        # For right now, I'm just going to keep check_dtype=False
        check_dtype=False,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        "TRY_TO_NUMBER",
    ],
)
def test_to_number_optional(fn_name):
    """Test TRY_TO_NUMBER and TO_NUMBER in optional argument case"""
    query = f"""SELECT case when {fn_name}(A) in (0, 1, 2, 3, 4, 5)
            then 'USA' else 'international' end as origin_zip_type
            FROM table1 """

    df = pd.DataFrame({"A": [str(i) for i in (1, 22, 3, 99, 44, 5, 0)]})
    expected_output = pd.DataFrame(
        {
            "ORIGIN_ZIP_TYPE": [
                "USA",
                "international",
                "USA",
                "international",
                "international",
                "USA",
                "USA",
            ]
        }
    )

    ctx = {"TABLE1": df}
    check_query(
        query,
        ctx,
        None,
        expected_output=expected_output,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        "TRY_TO_NUMBER",
    ],
)
def test_to_number_optional_invalid_str(fn_name):
    query = f"""SELECT case when {fn_name}(A) in (0, 1, 2, 3, 4, 5)
            then 'USA' else 'international' end as origin_zip_type
            FROM table1 """

    df = pd.DataFrame({"A": [str(i) for i in (1, "$$", 3, 99, 44, 5, "-4#")]})
    expected_output = pd.DataFrame(
        {
            "ORIGIN_ZIP_TYPE": [
                "USA",
                "international",
                "USA",
                "international",
                "international",
                "USA",
                "international",
            ]
        }
    )
    ctx = {"TABLE1": df}
    if "TRY" in fn_name:
        check_query(
            query,
            ctx,
            None,
            expected_output=expected_output,
        )
    else:
        with pytest.raises(ValueError, match="unable to convert string literal"):
            bc = bodosql.BodoSQLContext({"TABLE1": df})

            @bodo.jit
            def impl(bc):
                return bc.sql(query)

            impl(bc)


@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        pytest.param("TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TO_DECIMAL", marks=pytest.mark.slow),
        "TRY_TO_NUMBER",
        pytest.param("TRY_TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TRY_TO_DECIMAL", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "invalid_str",
    [
        "NOT A NUMBER",
        pytest.param("1.0.0", marks=pytest.mark.slow),
        pytest.param("1-000", marks=pytest.mark.slow),
        pytest.param(".", marks=pytest.mark.slow),
        pytest.param("-1.23-", marks=pytest.mark.slow),
        pytest.param("--1.23", marks=pytest.mark.slow),
    ],
)
def test_to_number_invalid(fn_name, invalid_str):
    query = f"SELECT {fn_name}('{invalid_str}') as A"
    ctx = {}
    if "TRY" in fn_name:
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame(
                {"A": pd.array([None], dtype=pd.ArrowDtype(pa.int64()))}
            ),
            check_dtype=False,
        )
    else:
        with pytest.raises(ValueError, match="unable to convert string literal"):
            bc = bodosql.BodoSQLContext()

            @bodo.jit
            def impl(bc):
                return bc.sql(query)

            impl(bc)


@pytest.mark.parametrize(
    "fn_name",
    [
        "TO_NUMBER",
        pytest.param("TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TO_DECIMAL", marks=pytest.mark.slow),
        "TRY_TO_NUMBER",
        pytest.param("TRY_TO_NUMERIC", marks=pytest.mark.slow),
        pytest.param("TRY_TO_DECIMAL", marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize(
    "invalid_str",
    [
        "10000.0",
        pytest.param("1231249", marks=pytest.mark.slow),
        pytest.param("-9999999.2", marks=pytest.mark.slow),
    ],
)
def test_to_number_out_of_bounds(fn_name, invalid_str):
    query = f"SELECT {fn_name}('{invalid_str}', 4) as A"
    ctx = {}
    if "TRY" in fn_name:
        check_query(
            query,
            ctx,
            None,
            expected_output=pd.DataFrame(
                {"A": pd.array([None], dtype=pd.ArrowDtype(pa.int64()))}
            ),
            check_dtype=False,
        )
    else:
        with pytest.raises(
            ValueError, match="too many digits to the left of the decimal"
        ):
            bc = bodosql.BodoSQLContext()

            @bodo.jit
            def impl(bc):
                return bc.sql(query)

            impl(bc)


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series(
                    [-123456] * 5 + [None, 0, 156, 155, 165, 175], dtype=pd.Int32Dtype()
                ),
                pd.Series(
                    [1, -1, -2, None, -3, 1, 2, -5, -1, -1, -1], dtype=pd.Int32Dtype()
                ),
                pd.Series(
                    [
                        -123456,
                        -123460,
                        -123500,
                        None,
                        -123000,
                        None,
                        0,
                        0,
                        160,
                        170,
                        180,
                    ],
                    dtype=pd.Int32Dtype(),
                ),
            ),
            id="integers-with_scale",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        15.57407724654902,
                        -218.5039863261519,
                        None,
                        -0.1425465430742778,
                        11578.212823495775,
                        0.5,
                        1.5,
                        2.5,
                        -0.5,
                        -1.5,
                        -2.5,
                    ]
                ),
                0,
                pd.Series(
                    [16.0, -219.0, None, 0.0, 11578, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0]
                ),
            ),
            id="floats-no_scale",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        15.57407724654902,
                        -21.85039863261519,
                        None,
                        -0.1425465430742778,
                        11578.212823495775,
                        0.05,
                        -12.3495,
                        -47.65,
                        25.0,
                    ]
                    * 3
                ),
                [1] * 9 + [4] * 9 + [-1] * 9,
                pd.Series(
                    [
                        15.6,
                        -21.9,
                        None,
                        -0.1,
                        11578.2,
                        0.1,
                        -12.3,
                        -47.7,
                        25.0,
                        15.5741,
                        -21.8504,
                        None,
                        -0.1425,
                        11578.2128,
                        0.05,
                        -12.3495,
                        -47.65,
                        25.0,
                        20.0,
                        -20.0,
                        None,
                        0.0,
                        11580.0,
                        0.0,
                        -10.0,
                        -50.0,
                        30.0,
                    ]
                ),
            ),
            id="floats-with_scale",
        ),
    ]
)
def round_data(request):
    """Tests the behavior of the ROUND function. Hardcoded answers obtained
    from Snowflake to ensure that the correct rounding behavior is achieved"""
    data, scale, answer = request.param
    ctx = {"TABLE1": pd.DataFrame({"A": data, "S": scale})}
    scale_str = "" if isinstance(scale, int) and scale == 0 else ", S"
    return ctx, scale_str, answer


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case", marks=pytest.mark.slow),
    ],
)
def test_round(round_data, use_case, spark_info, memory_leak_check):
    ctx, scale_str, answer = round_data
    if use_case:
        query = f"SELECT CASE WHEN A < -999999 THEN NULL ELSE ROUND(A{scale_str}) END FROM table1"
    else:
        query = f"SELECT ROUND(A{scale_str}) FROM table1"
    check_query(
        query,
        ctx,
        spark_info,
        check_names=False,
        check_dtype=False,
        expected_output=pd.DataFrame({0: answer}),
    )


@pytest.mark.slow
def test_floor_ceil(memory_leak_check):
    """
    Tests the rounding functions FLOOR and CEIL with 1 argument and with
    2 arguments.
    """
    query = "SELECT FLOOR(X), CEIL(X), FLOOR(X, P), CEIL(X, P) FROM table1"
    ctx = {
        "TABLE1": pd.DataFrame(
            {"X": [2.71828] * 3 + [123.456] * 3, "P": [1, -1, 3] * 2}
        )
    }
    expected_output = pd.DataFrame(
        {
            0: [2.0] * 3 + [123.0] * 3,
            1: [3.0] * 3 + [124.0] * 3,
            2: [2.7, 0.0, 2.718, 123.4, 120.0, 123.456],
            3: [2.8, 10.0, 2.719, 123.5, 130.0, 123.456],
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.parametrize(
    "use_case",
    [
        pytest.param(False, id="no_case"),
        pytest.param(True, id="with_case"),
    ],
)
def test_random(use_case, memory_leak_check):
    """Tests the behavior of RANDOM by generating a million columns
    (with and without CASE) then making sure that the values meet
    certain criteria:
    - The smallest value generated is near INT_MIN for int64
    - The largest value generated is near INT_MAX for int64
    - At most 5 of the values are duplicates
    - The mean is within a certain distance of zero
    - The skew is approximately zero"""
    n = 10**6
    if use_case:
        calculation = "CASE WHEN A >= 0 THEN RANDOM() ELSE RANDOM() END"
    else:
        calculation = "RANDOM()"
    query = f"""
        WITH table2 AS (SELECT {calculation} AS R FROM table1)
        SELECT
            MIN(R) <= -9131138316486227968,
            MAX(R) >= 9131138316486227968,
            COUNT(DISTINCT R) >= {n - 5},
            AVG(R) <= 100000000000000000,
            AVG(R) >= -100000000000000000,
            SKEW(R)
        FROM table2
    """
    ctx = {"TABLE1": pd.DataFrame({"A": np.arange(n)})}
    answer = pd.DataFrame(
        {
            "MIN": True,
            "MAX": True,
            "DISTINCT": True,
            "MEAN_HI": True,
            "MEAN_LO": True,
            "SKEW": 0.0,
        },
        index=np.arange(1),
    )
    check_query(
        query,
        ctx,
        None,
        check_dtype=False,
        check_names=False,
        expected_output=answer,
        atol=0.1,
        is_out_distributed=False,
    )


def test_trunc_truncate_single_arg(memory_leak_check):
    """
    Tests TRUNC and TRUNCATE on numeric values with a single argument
    """
    t0 = pd.DataFrame({"A": [100, 100.123, 100.5, -100.5, -100.123, -100]})
    ctx = {"TABLE0": t0}
    query = "SELECT TRUNC(A) as trunc_out, TRUNCATE(A) as truncate_out from table0"
    expected_output = pd.DataFrame(
        {
            "TRUNC_OUT": [100, 100, 100, -100, -100, -100],
            "TRUNCATE_OUT": [100, 100, 100, -100, -100, -100],
        }
    )
    check_query(
        query,
        ctx,
        None,
        check_names=False,
        check_dtype=False,
        expected_output=expected_output,
    )
