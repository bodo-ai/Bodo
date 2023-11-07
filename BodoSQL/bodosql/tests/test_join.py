# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL join queries on BodoSQL
"""

import copy
import io
from datetime import date

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.tests.timezone_common import representative_tz  # noqa
from bodo.tests.user_logging_utils import (
    check_logger_msg,
    create_string_io_logger,
    set_logging_stream,
)
from bodo.tests.utils import pytest_slow_unless_join
from bodosql.tests.utils import check_efficient_join, check_query

# Skip unless any join-related files were changed
pytestmark = pytest_slow_unless_join


@pytest.fixture(params=["INNER", "LEFT", "RIGHT", "FULL OUTER"])
def join_type(request):
    return request.param


def test_join(
    join_dataframes, spark_info, join_type, comparison_ops, memory_leak_check
):
    """test simple join queries"""
    # For nullable integers convert the pyspark output from
    # float to object
    if any(
        [
            isinstance(x, pd.core.arrays.integer.IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        convert_float_nan = True
    else:
        convert_float_nan = False
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    if comparison_ops == "<=>":
        # TODO: Add support for <=> from general-join cond
        return
    query = f"select table1.B, C, D from table1 {join_type} join table2 on table1.A {comparison_ops} table2.A"
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
        return_codegen=True,
        convert_float_nan=convert_float_nan,
        convert_columns_bytearray=convert_columns_bytearray,
    )
    pandas_code = result["pandas_code"]
    if comparison_ops == "=":
        check_efficient_join(pandas_code)


def test_multitable_join_cond(join_dataframes, spark_info, memory_leak_check):
    """tests selecting from multiple tables based upon a where clause"""

    if any(
        [
            isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                ),
            )
            or x in (np.float32, np.float64)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A", "B"]
    else:
        convert_columns_bytearray = None
    check_query(
        "select table1.A, table2.B from table1, table2 where table2.B > table2.A",
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_join_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Test that checks that joining two tables that share a column name
    can be merged if aliased.
    """
    if any(
        [
            isinstance(x, pd.core.arrays.integer.IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        convert_float_nan = True
    else:
        convert_float_nan = False
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["A1", "A2"]
    else:
        convert_columns_bytearray = None
    query = """SELECT
                 t1.A as A1,
                 t2.A as A2
               FROM
                 table1 t1,
                 table2 t2
    """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=False,
        convert_float_nan=convert_float_nan,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_natural_join(join_dataframes, spark_info, join_type, memory_leak_check):
    """test simple natural join queries"""
    # For nullable integers convert the pyspark output from
    # float to object
    if any(
        [
            isinstance(x, pd.core.arrays.integer.IntegerDtype)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        convert_float_nan = True
    else:
        convert_float_nan = False
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select table1.B, C, D from table1 NATURAL {join_type} join table2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=False,
        check_names=False,
        convert_float_nan=convert_float_nan,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_and_join(join_dataframes, spark_info, memory_leak_check):
    """
    Query that demonstrates that a join with an AND expression
    will merge on a common column, rather than just merge the entire tables.
    """
    if any(
        [
            isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                ),
            )
            or x in (np.float32, np.float64)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    query = """
        SELECT
            table1.A, table2.B
        from
            table1, table2
        where
            (table1.A = table2.A and table1.B = table2.B)
        """
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        return_codegen=True,
        check_dtype=check_dtype,
        # TODO[BE-3478]: enable dict-encoded string test when fixed
        use_dict_encoded_strings=False,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_or_join(join_dataframes, spark_info, memory_leak_check):
    """
    Query that demonstrates that a join with an OR expression and common conds
    will merge on the common cond, rather than just merge the entire tables.
    """

    if isinstance(join_dataframes["table1"]["A"][0], bytes):
        byte_array_cols = ["A", "B"]
    else:
        byte_array_cols = []

    if any(
        [
            isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                ),
            )
            or x in (np.float32, np.float64)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    query = """
        SELECT
            table1.A, table2.B
        from
            table1, table2
        where
            (table1.A = table2.A or table1.B = table2.B)
        """
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        return_codegen=True,
        check_dtype=check_dtype,
        convert_columns_bytearray=byte_array_cols,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_join_types(join_dataframes, spark_info, join_type, memory_leak_check):
    """test all possible join types"""
    if any(
        [
            isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                ),
            )
            or x in (np.float32, np.float64)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select table2.B, C, D from table1 {join_type} join table2 on table1.A = table2.A"
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_join_different_size_tables(
    join_dataframes, spark_info, join_type, memory_leak_check
):
    """tests that join operations still works when the dataframes have different sizes"""
    if any(
        [
            isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.Float32Dtype,
                    pd.Float64Dtype,
                ),
            )
            or x in (np.float32, np.float64)
            for x in join_dataframes["table1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["B", "C", "D"]
    else:
        convert_columns_bytearray = None
    df = pd.DataFrame({"A": [1, 2, 3]})
    copied_join_dataframes = copy.copy(join_dataframes)

    copied_join_dataframes["table3"] = df
    query = f"select table2.B, C, D from table1 {join_type} join table2 on table1.A = table2.A"
    result = check_query(
        query,
        copied_join_dataframes,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_nested_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work properly"""

    # for context, the nested right join should create a number of null values in table4.A,
    # which we then use in the join condition for the top level join
    # the null values in table4.A shouldn't match to anything, and shouldn't raise an error

    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["T1", "T2"]
    else:
        convert_columns_bytearray = None

    query = f"""
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3
    JOIN
        (select table1.A from table1 RIGHT join table2 on table1.A = table2.A) as table4
    ON
        table4.A = table3.Y
    """
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_nested_or_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work with implicit joins using 'or'"""

    # for context, the nested outer join should create a number of null values in table4.A and table4.B,
    # which we then use in the join condition for the top level join
    # assumedly, the null values in table4.A/B shouldn't match to anything, and shouldn't raise an error
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["T1", "T2"]
    else:
        convert_columns_bytearray = None

    query = f"""
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A) as table4
    WHERE
        table3.Y = table4.A or table3.Y = table4.B
    """
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_nested_and_join(join_dataframes, spark_info, memory_leak_check):
    """tests that nested joins work with implicit joins using 'and'"""

    # for context, the nested right join should create a number of null values in table4.A,
    # which we then use in the join condition for the top level join
    # assumedly, the null values in table4.A should match to anything, and shouldn't raise an error
    query = f"""
    SELECT
        table3.Y as T1, table4.A as T2
    FROM
        table3, (select table1.A, table2.B from table1 FULL OUTER join table2 on table1.A = table2.A) as table4
    WHERE
        table3.Y = table4.A and table3.Y = table4.B
    """
    result = check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=False,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_join_boolean(bodosql_boolean_types, spark_info, join_type, memory_leak_check):
    """test all possible join types on boolean table"""

    newCtx = {
        "table1": bodosql_boolean_types["table1"],
        "table2": bodosql_boolean_types["table1"],
    }
    query = f"select table1.B, table2.C from table1 {join_type} join table2 on table1.A"
    result = check_query(
        query,
        newCtx,
        spark_info,
        check_names=False,
        return_codegen=True,
        check_dtype=False,
    )
    pandas_code = result["pandas_code"]
    check_efficient_join(pandas_code)


def test_multikey_join_types(join_dataframes, spark_info, join_type, memory_leak_check):
    """test that for all possible join types "and equality conditions" turn into multikey join"""
    # Note: We don't check the generated code because column ordering isn't deterministic
    # Join code doesn't properly trim the filter yet, so outer joins will drop any NA columns
    # when applying the filter.
    # TODO: Trim filter to just column not used in the key
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select C, D from table1 {join_type} join table2 on table1.A = table2.A and table1.B = table2.B"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_trimmed_multikey_cond_inner_join(
    join_dataframes, spark_info, memory_leak_check
):
    """test that with inner join, equality conditions that are used in AND become keys and don't appear in the filter."""
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        convert_columns_bytearray = ["C", "D"]
    else:
        convert_columns_bytearray = None
    query = f"select C, D from table1 inner join table2 on table1.A = table2.A and table1.B < table2.B"
    # Note: We don't check the generated code because column ordering isn't deterministic
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_nonascii_in_implicit_join(spark_info, memory_leak_check):
    """
    Tests using non-ascii in an implicit join via select distinct.
    """
    ctx = {
        "table1": pd.DataFrame(
            {
                "D": pd.Series(list(pd.date_range("2011", "2018", 5)) * 20),
                "S": pd.Series(
                    [
                        None if i % 7 == 0 else chr(65 + (i**2) % 8 + i // 48)
                        for i in range(100)
                    ]
                ),
            }
        ),
        "table2": pd.DataFrame(
            {
                "T": pd.Series(
                    [
                        a + b + c + d
                        for a in ["", *"ALPHABETS♫UP"]
                        for b in ["", *"ÉPSI∫øN"]
                        for c in ["", *"ZE฿Rä"]
                        for d in "THETA"
                    ]
                )
            }
        ),
    }

    query = """
    SELECT
        S,
        D,
        COUNT(*)
    FROM table1
    WHERE s IN (SELECT DISTINCT LEFT(t, 1) FROM table2)
    GROUP BY s, d
    """

    check_query(query, ctx, spark_info, check_names=False, check_dtype=False)


def test_tz_aware_join(representative_tz, memory_leak_check):
    """
    Test join, including non-equality, on tz_aware data
    """
    df = pd.DataFrame(
        {
            "A": list(
                pd.date_range(
                    start="1/1/2022", freq="4D7H", periods=30, tz=representative_tz
                )
            )
            + [None] * 4,
            # Note: B's and A's will overlap.
            "B": [None] * 14
            + list(
                pd.date_range(
                    start="1/1/2022", freq="12D21H", periods=20, tz=representative_tz
                )
            ),
            "C": pd.date_range(
                start="3/1/2022", freq="1H", periods=34, tz=representative_tz
            ),
            "D": pd.date_range(
                start="1/1/2022", freq="14D20T", periods=34, tz=representative_tz
            ),
        }
    )
    query = """
        select
            t1.A as A,
            t2.B as B,
            t1.C as C,
            t2.D as D
        FROM
            table1 t1
        JOIN table2 t2
            on t1.A = t2.B and t2.C > t1.B
    """
    ctx = {
        "table1": df,
        "table2": df,
    }
    py_output = df.merge(df, left_on="A", right_on="B")
    # Drop nulls to match SQL
    py_output = py_output[py_output.A_x.notna()]
    # Add the comparison
    py_output = py_output[py_output.C_y > py_output.B_x]
    py_output = py_output[["A_x", "B_y", "C_x", "D_y"]]
    py_output.columns = ["A", "B", "C", "D"]
    check_query(query, ctx, None, expected_output=py_output)


def test_join_pow(spark_info, join_type, memory_leak_check):
    """
    Make sure pow() works inside join conditions
    """
    df1 = pd.DataFrame({"A": [2, 4, 3] * 4, "B": [3.1, 2.2, 0.1] * 4})
    df2 = pd.DataFrame({"C": [1, 2] * 3, "D": [1.1, 3.3] * 3})
    query1 = f"select * from t1 {join_type} join t2 on pow(t1.A - t2.C, 2) > 11"
    query2 = f"select * from t1 {join_type} join t2 on pow(pow(t1.A - t2.C, 2) + pow(t1.B - t2.D,2),.5)<2"
    ctx = {
        "t1": df1,
        "t2": df2,
    }
    check_query(query1, ctx, spark_info, check_dtype=False, check_names=False)
    check_query(query2, ctx, spark_info, check_dtype=False, check_names=False)


def test_interval_join_compilation(memory_leak_check):
    """
    Tests that the Interval Join detection code correctly determines that
    Interval Join should be used in this case. This is useful for ensuring:
        * That Bodo can handle BodoSQL column names (EXPR$1, ...)
        * That BodoSQL performs the casts as a projection before the join.
          Interval join currently does not support operations inside of the condition
    """
    if bodo.bodosql_use_streaming_plan:
        # Ignore this test when using streaming plan
        return

    df1 = pd.DataFrame(
        {
            "P": [date(2023, 1, 1)],
        }
    )
    df2 = pd.DataFrame(
        {
            "L": pd.date_range(start="2023-01-01", periods=10, freq="D").to_series(),
            "R": pd.date_range(start="2023-01-01", periods=10, freq="D").to_series(),
        }
    )
    bc = bodosql.BodoSQLContext({"t1": df1, "t2": df2})
    query = (
        "select P, L from t1 inner join t2 on t1.P >= t2.L::date and t1.P < t2.R::date"
    )

    def impl(bc, query):
        return bc.sql(query)

    stream = io.StringIO()
    logger = create_string_io_logger(stream)
    with set_logging_stream(logger, 2):
        bodo.jit((bodo.typeof(bc), numba.types.literal(query)))(impl)
        check_logger_msg(stream, "Using optimized interval range join")


@pytest.mark.slow
def test_join_div(spark_info, join_type, memory_leak_check):
    """
    Make sure div operation works inside join conditions
    """
    df1 = pd.DataFrame({"A": [2, 4, 3] * 4, "B": [3.1, 2.2, 0.1] * 4})
    df2 = pd.DataFrame({"A": [1, 2] * 4, "D": [1.1, 3.3] * 4})
    query1 = f"select B from t1 {join_type} join t2 on true where t1.B / t2.D > 2.0"
    ctx = {
        "t1": df1,
        "t2": df2,
    }
    check_query(query1, ctx, spark_info, check_dtype=False, check_names=False)
