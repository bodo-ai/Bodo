# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test SQL operations that should produce understandable errors on BodoSQL
"""
import re
from decimal import Decimal

import bodosql
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_type_error(memory_leak_check):
    """
    Checks that incomparable types raises a BodoError.
    """

    def impl(df):
        query = "Select A from table1 where A > date '2021-09-05'"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Cannot apply '>' to arguments of type '<BIGINT> > <DATE>'",
    ):
        impl(df)


@pytest.mark.slow
def test_type_error_jit(memory_leak_check):
    """
    Checks that incomparable types raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        query = "Select A from table1 where A > date '2021-09-05'"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Cannot apply '>' to arguments of type '<BIGINT> > <DATE>'",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_format_str(memory_leak_check):
    """
    Checks that an invalid format string raises a BodoError.
    """

    def impl(df):
        query = "SELECT STR_TO_DATE(A, '%Y-%n-%d:%h') from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4})
    with pytest.raises(
        BodoError,
        match=r"STR_TO_DATE contains an invalid format string",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_format_str_jit(memory_leak_check):
    """
    Checks that an invalid format string raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        query = "SELECT STR_TO_DATE(A, '%Y-%n-%d:%h') from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4})
    with pytest.raises(
        BodoError,
        match=r"STR_TO_DATE contains an invalid format string",
    ):
        impl(df)


@pytest.mark.slow
def test_unsupported_format_str(memory_leak_check):
    """
    Checks that an unsupported format string raises a BodoError.
    """

    def impl(df):
        query = "SELECT STR_TO_DATE(A, '%l') from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4})
    with pytest.raises(
        BodoError,
        match=r"STR_TO_DATE contains an unsupported escape character %l",
    ):
        impl(df)


@pytest.mark.slow
def test_unsupported_format_str_jit(memory_leak_check):
    """
    Checks that an unsupported format string raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        query = "SELECT STR_TO_DATE(A, '%l') from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4})
    with pytest.raises(
        BodoError,
        match=r"STR_TO_DATE contains an unsupported escape character %l",
    ):
        impl(df)


@pytest.mark.slow
def test_unsupported_types(memory_leak_check):
    """
    Checks unsupported types throw an exception in Python.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})

    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [Decimal(1.2)] * 4})
    with pytest.raises(
        BodoError,
        match="Pandas column 'B' with type DecimalArrayType\\(38, 18\\) not supported in BodoSQL.",
    ):
        impl(df)


@pytest.mark.slow
def test_unsupported_types_jit(memory_leak_check):
    """
    Checks unsupported types throw an exception in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        # Bodo doesn't try assign a type until there is a sql call
        return bc.sql("select * from table1")

    df = pd.DataFrame({"A": [1, 2, 3, 4], "B": [Decimal(1.2)] * 4})
    with pytest.raises(
        BodoError,
        match="Pandas column 'B' with type DecimalArrayType\\(38, 18\\) not supported in BodoSQL.",
    ):
        impl(df)


@pytest.mark.slow
def test_bad_bodosql_context(memory_leak_check):
    """
    Checks that a bad bodosql context raises a BodoError.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"table1": filename})
        return bc

    filename = "myfile.pq"
    with pytest.raises(
        BodoError,
        match=r"BodoSQLContext\(\): 'table' values must be DataFrames",
    ):
        impl(filename)


@pytest.mark.slow
def test_bad_bodosql_context_jit(memory_leak_check):
    """
    Checks that a bad bodosql context raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(filename):
        bc = bodosql.BodoSQLContext({"table1": filename})
        # Bodo doesn't try assign a type until there is a sql call
        return bc.sql("select * from table1")

    filename = "myfile.pq"
    with pytest.raises(
        BodoError,
        match=r"BodoSQLContext\(\): 'table' values must be DataFrames",
    ):
        impl(filename)


@pytest.mark.slow
def test_query_syntax_error(memory_leak_check):
    """
    Checks that a bad bodosql query raises a BodoError.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("selct * from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Non-query expression encountered in illegal context",
    ):
        impl(df)


@pytest.mark.slow
def test_query_syntax_error_jit(memory_leak_check):
    """
    Checks that a bad bodosql query raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("selct * from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Non-query expression encountered in illegal context",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_table(memory_leak_check):
    """
    Checks that a missing table raises a BodoError.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select * from table_1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Object 'table_1' not found",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_table_jit(memory_leak_check):
    """
    Checks that a missing table raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select * from table_1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Object 'table_1' not found",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_column(memory_leak_check):
    """
    Checks that a missing column raises a BodoError.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select B from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'B' not found in any table",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_column_jit(memory_leak_check):
    """
    Checks that a missing column raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select B from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'B' not found in any table",
    ):
        impl(df)


@pytest.mark.slow
def test_empty_query_python(memory_leak_check):

    df = pd.DataFrame({"A": np.arange(100)})
    msg = "BodoSQLContext passed empty query string"
    with pytest.raises(
        BodoError,
        match=msg,
    ):
        bc = bodosql.BodoSQLContext({"table1": df})
        bc.sql("")

    with pytest.raises(
        BodoError,
        match=msg,
    ):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.convert_to_pandas("")


@pytest.mark.slow
def test_empty_query_jit(memory_leak_check):
    @bodo.jit
    def impl1():
        bc = bodosql.BodoSQLContext({"table1": df})
        bc.sql("")

    @bodo.jit
    def impl2():
        bc = bodosql.BodoSQLContext({"table1": df})
        bc.convert_to_pandas("")

    df = pd.DataFrame({"A": np.arange(100)})
    msg = "BodoSQLContext passed empty query string"

    with pytest.raises(
        BodoError,
        match=msg,
    ):
        impl1()

    with pytest.raises(
        BodoError,
        match=msg,
    ):
        impl2()


@pytest.mark.slow
@pytest.mark.parametrize("fn_name", ["NVL", "IFNULL"])
def test_str_date_select_cond_fns(fn_name, memory_leak_check):
    """
    Many sql dialects play fast and loose with what is a string and date/timestamp
    This test checks that our user defined cond functions in calcite do not accept
    a string and timestamp.
    """
    df = pd.DataFrame(
        {
            "B": [
                pd.Timestamp("2021-09-26"),
                pd.Timestamp("2021-03-25"),
                pd.Timestamp("2020-01-26"),
            ]
            * 4,
            "A": ["2021-09-26", "2021-03-25", "2020-01-26"] * 4,
        }
    )

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(f"select {fn_name}(B, A) from table1")

    with pytest.raises(
        BodoError,
        match=re.escape(
            f"Cannot infer return type for {fn_name}; operand types: [TIMESTAMP(0), VARCHAR]"
        ),
    ):
        impl(df)


@pytest.mark.slow
@pytest.mark.skip("Need to fix operand typchecking for IF, see BS-581")
def test_str_date_select_IF(memory_leak_check):
    """
    Many sql dialects play fast and loose with what is a string and date/timestamp
    This test checks that a IF with a string and timestamp is not accepted.
    """
    df = pd.DataFrame(
        {
            "C": [True, False, True] * 4,
            "B": [
                pd.Timestamp("2021-09-26"),
                pd.Timestamp("2021-03-25"),
                pd.Timestamp("2020-01-26"),
            ]
            * 4,
            "A": ["2021-09-26", "2021-03-25", "2020-01-26"] * 4,
        }
    )

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select IF(C, B, A) from table1")

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Cannot infer return type for IF; operand types: [BOOLEAN, TIMESTAMP(0), VARCHAR]"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_non_existant_fn(memory_leak_check):
    """
    Checks that a non existent function raises a BodoError in JIT.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select NON_EXISTANT_FN(A) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "No match found for function signature NON_EXISTANT_FN(<NUMERIC>)"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_non_existant_fn_jit(memory_leak_check):
    """
    Checks that a non existent function raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select NON_EXISTANT_FN(A) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "No match found for function signature NON_EXISTANT_FN(<NUMERIC>)"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_wrong_types_fn(memory_leak_check):
    """
    Checks that supplying the incorrect types to a fn raises a BodoError in JIT.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select LCASE(INTERVAL 10 DAYS) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Cannot apply 'LCASE' to arguments of type 'LCASE(<INTERVAL DAY>)'. Supported form(s): 'LCASE(<STRING>)'"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_wrong_types_fn_jit(memory_leak_check):
    """
    Checks that supplying the incorrect types to a fn raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select LCASE(INTERVAL 10 DAYS) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Cannot apply 'LCASE' to arguments of type 'LCASE(<INTERVAL DAY>)'. Supported form(s): 'LCASE(<STRING>)'"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_wrong_number_of_args(memory_leak_check):
    """
    Checks that supplying the wrong number of arguments to a fn raises a BodoError in JIT.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select CIELING(A, 10) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "No match found for function signature CIELING(<NUMERIC>, <NUMERIC>)"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_wrong_number_of_args_jit(memory_leak_check):
    """
    Checks that supplying the wrong number of arguments to a fn raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select CIELING(A, 10) from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "No match found for function signature CIELING(<NUMERIC>, <NUMERIC>)"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_coalesece():
    """
    Checks that supplying non unifiable types raises a BodoError in JIT
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(
            "select COALESCE(0.23, INTERVAL 5 DAYS, 'hello world') from table1"
        )

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Illegal mixing of types in CASE or COALESCE statement",
    ):
        impl(df)


@pytest.mark.slow
def test_coalesece_jit():
    """
    Checks that supplying non unifiable types raises a BodoError in JIT
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(
            "select COALESCE(0.23, INTERVAL 5 DAYS, 'hello world') from table1"
        )

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Illegal mixing of types in CASE or COALESCE statement",
    ):
        impl(df)


@pytest.mark.slow
def test_row_fn():
    """
    Checks that incorrectly formated windowed aggregations. Raise a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(
            "select LEAD(A, 1) over (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) from table1"
        )

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"ROW/RANGE not allowed with RANK, DENSE_RANK or ROW_NUMBER functions",
    ):
        impl(df)


@pytest.mark.slow
def test_row_fn_jit():
    """
    Checks that incorrectly formated windowed aggregations. Raise a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(
            "select LEAD(A, 1) over (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) from table1"
        )

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"ROW/RANGE not allowed with RANK, DENSE_RANK or ROW_NUMBER functions",
    ):
        impl(df)


@pytest.mark.slow
def test_lead_lag_no_2nd_arg():
    """tests that lead and lag throw a reasonable error when not supplied a second argument"""

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select LEAD(A) over (PARTITION BY B ORDER BY C) from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"BodoSQL does not support LEAD or LAG without a second argument. Please supply a second argument",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_fn():
    """
    Checks that incorrectly formated windowed aggregations. Raise a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select INT(A, 1) from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"[\s | .]*Incorrect syntax near the keyword 'INT' at line 1, column 8[\s | .]*Was expecting one of[\s | .]*",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_fn_jit():
    """
    Checks that incorrectly formated windowed aggregations. Raise a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select INT(A, 1) from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"[\s | .]*Incorrect syntax near the keyword 'INT' at line 1, column 8[\s | .]*Was expecting one of[\s | .]*",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_comma():
    """
    Checks that improper comma placement raises a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select A,B,C, from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r'[\s | .]*Encountered "from" at line 1, column 15.\sWas expecting one of:[\s | .]*',
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_comma_jit():
    """
    Checks that improper comma placement raises a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select A,B,C, from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r'[\s | .]*Encountered "from" at line 1, column 15.\sWas expecting one of:[\s | .]*',
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_named_param():
    """
    Checks that not specifying named parameters raises a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select @a from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"SQL query contains a unregistered parameter: '@a'",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_named_param_jit():
    """
    Checks that not specifying named parameters raises a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select @a from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"SQL query contains a unregistered parameter: '@a'",
    ):
        impl(df)


@pytest.mark.slow
def test_multi_table_colname():
    """
    Checks ambiguous column selection raises a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df, "table2": df})
        return bc.sql("select A from table1, table2")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'A' is ambiguous",
    ):
        impl(df)


@pytest.mark.slow
def test_multi_table_colname_jit():
    """
    Checks ambiguous column selection raises a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"table1": df, "table2": df})
        return bc.sql("select A from table1, table2")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'A' is ambiguous",
    ):
        impl(df)


@pytest.mark.slow
def test_string_row_min_max_error(bodosql_string_types):
    """tests that row functions with unsupported types raises an error"""

    def impl_max(df):
        query = "SELECT MAX(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    def impl_min(df):
        query = "SELECT MIN(A) OVER (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) from table1"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r".*Windowed aggregation function MAX not supported for SQL type VARCHAR.*",
    ):
        impl_max(bodosql_string_types["table1"])

    with pytest.raises(
        BodoError,
        match=r".*Windowed aggregation function MIN not supported for SQL type VARCHAR.*",
    ):
        impl_min(bodosql_string_types["table1"])


@pytest.mark.slow
def test_qualify_no_groupby_err():
    """tests that a reasonable error is thrown when using non grouped expressions in window functions"""
    table1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7] * 3,
            "B": [1, 1, 2, 2, 3, 3, 4] * 3,
            "C": [1, 1, 1, 2, 2, 3, 3] * 3,
        }
    )

    def impl(df):
        query = "SELECT A from table1 GROUP BY A HAVING MAX(A) > 3 QUALIFY MAX(A) OVER (PARTITION BY B) = 3"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r".*Expression 'B' is not being grouped*",
    ):
        impl(table1)


@pytest.mark.slow
def test_qualify_no_window_err():
    """tests that a reasonable error is thrown when qualify contains no window functions"""

    table1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7] * 3,
            "B": [1, 1, 2, 2, 3, 3, 4] * 3,
            "C": [1, 1, 1, 2, 2, 3, 3] * 3,
        }
    )

    def impl(df):
        query = "SELECT A from table1 QUALIFY A > 3"
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r".*QUALIFY clause must contain at least one windowed function*",
    ):
        impl(table1)
