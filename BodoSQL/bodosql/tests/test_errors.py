"""
Test SQL operations that should produce understandable errors on BodoSQL
"""

import re

import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_type_error(memory_leak_check):
    """
    Checks that incomparable types raises a BodoError.
    """

    def impl(df):
        query = "Select A from table1 where A > date '2021-09-05'"
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": np.arange(100, dtype=np.int64)})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": np.arange(100, dtype=np.int64)})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    df = pd.DataFrame({"A": ["2003-02-01:11", "2013-02-11:11", "2011-11-01:02"] * 4})
    with pytest.raises(
        BodoError,
        match=r"STR_TO_DATE contains an unsupported escape character %l",
    ):
        impl(df)


@pytest.mark.slow
def test_bad_bodosql_context(memory_leak_check):
    """
    Checks that a bad bodosql context raises a BodoError.
    """

    def impl(filename):
        bc = bodosql.BodoSQLContext({"TABLE1": filename})
        return bc

    filename = "myfile.pq"
    with pytest.raises(
        ValueError,
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
        bc = bodosql.BodoSQLContext({"TABLE1": filename})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("selct * from table1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        ValueError,
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select * from table_1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Object 'TABLE_1' not found",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_table_jit(memory_leak_check):
    """
    Checks that a missing table raises a BodoError in JIT.
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select * from table_1")

    df = pd.DataFrame({"A": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Object 'TABLE_1' not found",
    ):
        impl(df)


@pytest.mark.slow
def test_missing_column(memory_leak_check):
    """
    Checks that a missing column raises a BodoError.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        ValueError,
        match=msg,
    ):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        bc.sql("")

    with pytest.raises(
        ValueError,
        match=msg,
    ):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.convert_to_pandas("")


@pytest.mark.slow
def test_empty_query_jit(memory_leak_check):
    @bodo.jit
    def impl1():
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        bc.sql("")

    @bodo.jit
    def impl2():
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(f"select {fn_name}(B, A) from table1")

    # Does not explicitly check the types of the two arguments since changes to
    # type handling could cause their precision info to change, which may cause
    # a failure even though we still got the overall error we expected.
    with pytest.raises(
        BodoError,
        match=re.escape(f"Cannot infer return type for {fn_name}; operand types:"),
    ):
        impl(df)


@pytest.mark.slow
@pytest.mark.skip("Need to fix operand typechecking for IF, see BS-581")
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select IF(C, B, A) from table1")

    with pytest.raises(
        BodoError,
        match=re.escape(
            "Cannot infer return type for IF; operand types: [BOOLEAN, TIMESTAMP(9), VARCHAR]"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_non_existant_fn(memory_leak_check):
    """
    Checks that a non existent function raises a BodoError in JIT.
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select LOWER(A, 10) from table1")

    df = pd.DataFrame({"A": ["A"] * 10})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Invalid number of arguments to function 'LOWER'. Was expecting 1 arguments"
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select LOWER(A, 10) from table1")

    df = pd.DataFrame({"A": ["A"] * 10})
    with pytest.raises(
        BodoError,
        match=re.escape(
            "Invalid number of arguments to function 'LOWER'. Was expecting 1 arguments"
        ),
    ):
        impl(df)


@pytest.mark.slow
def test_coalesece(memory_leak_check):
    """
    Checks that supplying non unifiable types raises a BodoError in JIT
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
def test_coalesece_jit(memory_leak_check):
    """
    Checks that supplying non unifiable types raises a BodoError in JIT
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
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
def test_row_fn(memory_leak_check):
    """
    Checks that incorrectly formatted windowed aggregations. Raise a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(
            "select LEAD(A, 1) over (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) from table1"
        )

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"ROW/RANGE not allowed with RANK, DENSE_RANK, ROW_NUMBER or PERCENTILE_CONT/DISC functions",
    ):
        impl(df)


@pytest.mark.slow
def test_row_fn_jit(memory_leak_check):
    """
    Checks that incorrectly formatted windowed aggregations. Raise a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(
            "select LEAD(A, 1) over (PARTITION BY B ORDER BY C ROWS BETWEEN 1 PRECEDING and 1 FOLLOWING) from table1"
        )

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"ROW/RANGE not allowed with RANK, DENSE_RANK, ROW_NUMBER or PERCENTILE_CONT/DISC functions",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_fn(memory_leak_check):
    """
    Checks that incorrectly formatted windowed aggregations. Raise a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select INT(A, 1) from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"[\s | .]*No match found for function signature INT\(<NUMERIC>, <NUMERIC>\)[\s | .]*",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_syntax_fn_jit(memory_leak_check):
    """
    Checks that incorrectly formatted windowed aggregations. Raise a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select INT(A, 1) from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"[\s | .]*No match found for function signature INT\(<NUMERIC>, <NUMERIC>\)[\s | .]*",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_named_param(memory_leak_check):
    """
    Checks that not specifying named parameters raises a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select @a from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Named parameter \"a\" specified in query, but type information is not available",
    ):
        impl(df)


@pytest.mark.slow
def test_invalid_named_param_jit(memory_leak_check):
    """
    Checks that not specifying named parameters raises a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select @a from table1")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Named parameter \"a\" specified in query, but type information is not available",
    ):
        impl(df)


@pytest.mark.slow
def test_multi_table_colname(memory_leak_check):
    """
    Checks ambiguous column selection raises a JIT error
    """

    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df, "TABLE2": df})
        return bc.sql("select A from table1, table2")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'A' is ambiguous",
    ):
        impl(df)


@pytest.mark.slow
def test_multi_table_colname_jit(memory_leak_check):
    """
    Checks ambiguous column selection raises a JIT error
    """

    @bodo.jit
    def impl(df):
        bc = bodosql.BodoSQLContext({"TABLE1": df, "TABLE2": df})
        return bc.sql("select A from table1, table2")

    df = pd.DataFrame({"A": np.arange(100), "B": np.arange(100), "C": np.arange(100)})
    with pytest.raises(
        BodoError,
        match=r"Column 'A' is ambiguous",
    ):
        impl(df)


@pytest.mark.slow
def test_qualify_no_groupby_err(memory_leak_check):
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r".*Expression 'B' is not being grouped*",
    ):
        impl(table1)


@pytest.mark.slow
def test_qualify_no_window_err(memory_leak_check):
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
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r"QUALIFY expression '`TABLE1`.`A` > 3' must contain a window function",
    ):
        impl(table1)


@pytest.mark.slow
def test_lag_respect_nulls_no_window():
    """Tests LEAD/LAG with RESPECT NULLS but no OVER clause
    This correctly validates in Calcite, so we have to handle it in our codegen.
    I'm not sure if that's an error or not"""

    table1 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7] * 3,
            "B": [1, 1, 2, 2, 3, 3, 4] * 3,
            "C": [1, 1, 1, 2, 2, 3, 3] * 3,
        }
    )

    def impl(df):
        query = "SELECT LAG(A) RESPECT NULLS from table1"
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql(query)

    with pytest.raises(
        BodoError,
        match=r".*LAG requires OVER clause*",
    ):
        impl(table1)
