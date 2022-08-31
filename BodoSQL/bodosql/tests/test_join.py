# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of SQL join queries on BodoSQL
"""

import copy

import pandas as pd
import pytest
from bodosql.tests.utils import (
    bodo_version_older,
    check_efficient_join,
    check_query,
)

import bodo


@pytest.fixture(params=["INNER", "LEFT", "RIGHT", "FULL OUTER"])
def join_type(request):
    return request.param


def test_join(join_dataframes, spark_info, comparison_ops, memory_leak_check):
    """test simple join queries"""
    # For nullable intergers convert the pyspark output from
    # float to object
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
    query = f"select table1.B, C, D from table1 join table2 on table1.A {comparison_ops} table2.A"
    result = check_query(
        query,
        join_dataframes,
        spark_info,
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
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
        if (
            join_type == "FULL OUTER"
            and bodo.get_size() >= 3
            and bodo_version_older(2022, 6, 2)
        ):
            pytest.skip(
                reason="Requires next mini-release for engine changes to support metadata fixes",
            )
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


def test_join_differentsize_tables(
    join_dataframes, spark_info, join_type, memory_leak_check
):
    """tests that join operations still works when the dataframes have different sizes"""
    if any(
        [
            isinstance(x, pd.core.arrays.integer._IntegerDtype)
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
        if (
            join_type == "FULL OUTER"
            and bodo.get_size() >= 3
            and bodo_version_older(2022, 6, 2)
        ):
            pytest.skip(
                reason="Requires next mini-release for engine changes to support metadata fixes",
            )
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

    # for context, the nested outter join should create a number of null values in table4.A and table4.B,
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
    if join_type != "INNER":
        # [BS-664] Support outer join without equality condition
        return
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
        if (
            join_type in ("FULL OUTER", "LEFT")
            and bodo.get_size() >= 3
            and bodo_version_older(2022, 6, 2)
        ):
            pytest.skip(
                reason="Requires next mini-release for engine changes to support metadata fixes",
            )
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
def test_trimmed_multikey_cond_innerjoin(
    join_dataframes, spark_info, memory_leak_check
):
    """test that with inner join, equality conditions that are used in AND become keys and don't appear in the filter."""
    if any(
        [
            isinstance(join_dataframes["table1"][colname].values[0], bytes)
            for colname in join_dataframes["table1"].columns
        ]
    ):
        if bodo_version_older(2022, 6, 3):
            pytest.skip(
                reason="Requires next mini-release for engine changes to support general merge conditions",
            )
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
