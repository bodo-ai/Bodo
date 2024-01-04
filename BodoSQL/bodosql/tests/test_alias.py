# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Test file for tests related to alias operations. Some tests
    looks at the plans that are generated and check correctness.
"""
import pandas as pd
import pytest

from bodosql.tests.utils import check_query


def test_aliasing_numeric(bodosql_numeric_types, memory_leak_check):
    """test aliasing in queries"""
    table1 = bodosql_numeric_types["TABLE1"]
    check_query(
        "select A as testCol, C from table1",
        bodosql_numeric_types,
        None,
        check_dtype=False,
        expected_output=pd.DataFrame({"TESTCOL": table1["A"], "C": table1["C"]}),
    )
    check_query(
        "select sum(B) as testCol from table1",
        bodosql_numeric_types,
        None,
        check_dtype=False,
        is_out_distributed=False,
        expected_output=pd.DataFrame({"TESTCOL": [table1["B"].sum()]}),
    )
    expected_output = (
        table1.groupby("A")
        .agg({"B": "sum", "C": "sum"})
        .rename({"B": "TESTCOL1", "C": "TESTCOL2"}, axis=1)
    )
    check_query(
        "select sum(B) as testCol1, sum(C) as testCol2 from table1 group by A",
        bodosql_numeric_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )
    filtered_output = table1[(table1.A > 1) & (table1.A < 3)]
    grouped_output = filtered_output.groupby("A", as_index=False).agg({"B": "sum"})
    expected_output = (
        grouped_output.rename({"B": "TESTCOL"}, axis=1)
        .sort_values("TESTCOL", ascending=False)
        .head(10)
    )
    check_query(
        "select A, sum(b) as testCol from table1 where (A > 1 and A < 3) group by A order by testCol desc limit 10",
        bodosql_numeric_types,
        None,
        check_dtype=False,
        expected_output=expected_output,
    )


@pytest.mark.slow
def test_as_on_colnames(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that the as operator is working correctly for aliasing columns
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray1 = ["X"]
        convert_columns_bytearray2 = ["X", "Y"]
        convert_columns_bytearray3 = ["A", "Y"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None
    query1 = """
        SELECT
            A as X
        FROM
            table1
        """
    query2 = """
        SELECT
            A as X, B as Y
        FROM
            table1
        """
    query3 = """
        SELECT
            table1.A, table2.D as Y
        FROM
            table1, table2
        """
    check_query(
        query1,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray1,
    )
    check_query(
        query2,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray2,
    )
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_as_on_tablenames(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that the as operator is working correctly for aliasing table names
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray1 = ["A"]
        convert_columns_bytearray2 = ["A", "B"]
        convert_columns_bytearray3 = ["A", "D"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None
    query1 = """
        SELECT
            X.A
        FROM
            table1 as X
        """
    query2 = """
        SELECT
            X.A, X.B
        FROM
            table1 as X
        """
    query3 = """
        SELECT
            X.A, Y.D
        FROM
            table1 as X, table2 as Y
        """
    check_query(
        query1,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray1,
    )
    check_query(
        query2,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray2,
    )
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_cyclic_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that aliasing that could be interpreted as cyclic works as intended
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray = ["B", "C", "A"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
            A as B, B as C, C as A
        FROM
            table1
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_names=False,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_col_aliased_to_tablename(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that bodosql works correctly when the column names are aliased to table names
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray = ["TABLE2", "TABLE1"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
           TABLE1.C as TABLE2, TABLE2.A as TABLE1
        FROM
            TABLE1, TABLE2
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


@pytest.mark.slow
def test_table_aliased_to_colname(join_dataframes, spark_info, memory_leak_check):
    """
    Tests that bodosql works correctly when the table names are aliased to column names
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray = ["C", "A"]
    else:
        convert_columns_bytearray = None
    query = """
        SELECT
           A.C, C.A
        FROM
            table1 as A, table2 as C
        """
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray,
    )


def test_multi_table_renamed_projection(join_dataframes, spark_info, memory_leak_check):
    """
    Test that verifies that aliased projections from two different tables
    behave as expected.
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray1 = ["A", "Y"]
        convert_columns_bytearray2 = ["Y", "B"]
        convert_columns_bytearray3 = ["B", "Y", "D"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
        convert_columns_bytearray3 = None

    query = "SELECT table1.A, table2.D as Y FROM table1, table2"
    query2 = "SELECT table1.A as Y, table2.B FROM table1, table2"
    query3 = "SELECT table1.B, table2.A as Y, table2.D FROM table1, table2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray1,
    )
    check_query(
        query2,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray2,
    )
    check_query(
        query3,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray3,
    )


@pytest.mark.slow
def test_implicit_table_alias(join_dataframes, spark_info, memory_leak_check):
    """
    Test that aliasing tables with the implicit syntax works as intended
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
            for x in join_dataframes["TABLE1"].dtypes
        ]
    ):
        check_dtype = False
    else:
        check_dtype = True
    if any(
        [
            isinstance(join_dataframes["TABLE1"][colname].values[0], bytes)
            for colname in join_dataframes["TABLE1"].columns
        ]
    ):
        convert_columns_bytearray1 = ["A"]
        convert_columns_bytearray2 = ["Y", "B"]
    else:
        convert_columns_bytearray1 = None
        convert_columns_bytearray2 = None
    query = "SELECT T1.A from (SELECT * from table1) T1"
    query2 = "SELECT T1.Y, T2.B from (SELECT table2.A as Y FROM table2) T1, (SELECT table1.B FROM table1) T2"
    check_query(
        query,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray1,
    )
    check_query(
        query2,
        join_dataframes,
        spark_info,
        check_dtype=check_dtype,
        convert_columns_bytearray=convert_columns_bytearray2,
    )


@pytest.mark.slow
def test_unreserved_kw(spark_info, memory_leak_check):
    """Test that language/lead/user/method/rank"""
    query = "SELECT t.LANGUAGE, t.LEAD, t.USER, t.METHOD, t.RANK AS A FROM table1 t"
    check_query(
        query,
        {
            "TABLE1": pd.DataFrame(
                {
                    "LANGUAGE": ["A", "B", "C", "D", "E"],
                    "LEAD": ["A", "B", "C", "D", "E"],
                    "USER": ["A", "B", "C", "D", "E"],
                    "METHOD": ["A", "B", "C", "D", "E"],
                    "RANK": ["A", "B", "C", "D", "E"],
                }
            )
        },
        spark_info,
        expected_output=pd.DataFrame(
            {
                "A": ["A", "B", "C", "D", "E"],
                "LEAD": ["A", "B", "C", "D", "E"],
                "USER": ["A", "B", "C", "D", "E"],
                "METHOD": ["A", "B", "C", "D", "E"],
                "RANK": ["A", "B", "C", "D", "E"],
            }
        ),
        check_names=False,
    )


@pytest.mark.slow
def test_unreserved_kw_pt2(spark_info, memory_leak_check):
    """Test that "OUT", "FILTER", "CONDITION", "TRANSLATION", "POSITION"
    can be columns
    """
    query = "SELECT t.OUT, t.FILTER, t.CONDITION, t.TRANSLATION, t.POSITION AS A FROM table1 t"
    check_query(
        query,
        {
            "TABLE1": pd.DataFrame(
                {
                    "OUT": ["A", "B", "C", "D", "E"],
                    "FILTER": ["A", "B", "C", "D", "E"],
                    "CONDITION": ["A", "B", "C", "D", "E"],
                    "TRANSLATION": ["A", "B", "C", "D", "E"],
                    "POSITION": ["A", "B", "C", "D", "E"],
                }
            )
        },
        spark_info,
        expected_output=pd.DataFrame(
            {
                "OUT": ["A", "B", "C", "D", "E"],
                "FILTER": ["A", "B", "C", "D", "E"],
                "CONDITION": ["A", "B", "C", "D", "E"],
                "TRANSLATION": ["A", "B", "C", "D", "E"],
                "POSITION": ["A", "B", "C", "D", "E"],
            }
        ),
        check_names=False,
    )


@pytest.mark.slow
def test_unreserved_kw_pt3(spark_info, memory_leak_check):
    """Test that "ROW_NUMBER", "INTERVAL", "PERCENT", "COUNT", "TRANSLATE", "ROLLUP", "MATCHES", "ABS", "LAG", "MATCH_NUMBER",
    can be columns, aliases, or table names
    """
    query = "SELECT abs.AB as row_number, abs.count, abs.percent, abs.TRANSLATE as interval, abs.rollup, abs.matches as match_number FROM LAG abs"
    check_query(
        query,
        {
            "LAG": pd.DataFrame(
                {
                    "AB": ["A", "B", "C", "D", "E"],
                    "COUNT": ["A", "B", "C", "D", "E"],
                    "PERCENT": ["A", "B", "C", "D", "E"],
                    "TRANSLATE": ["A", "B", "C", "D", "E"],
                    "ROLLUP": ["A", "B", "C", "D", "E"],
                    "MATCHES": ["A", "B", "C", "D", "E"],
                }
            )
        },
        spark_info,
        expected_output=pd.DataFrame(
            {
                "AB": ["A", "B", "C", "D", "E"],
                "COUNT": ["A", "B", "C", "D", "E"],
                "PERCENT": ["A", "B", "C", "D", "E"],
                "TRANSLATE": ["A", "B", "C", "D", "E"],
                "ROLLUP": ["A", "B", "C", "D", "E"],
                "MATCHES": ["A", "B", "C", "D", "E"],
            }
        ),
        check_names=False,
    )
