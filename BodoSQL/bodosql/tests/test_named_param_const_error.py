"""
Test that Named Parameters can't be used where constants are
required.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import bodosql
import pandas as pd
import pytest

import bodo


@pytest.mark.slow
def test_named_param_extract(bodosql_datetime_types):
    """
    Checks that Named Params cannot be used in
    extract because the name must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select Extract(@a from A) from table1", {"a": a})

    with pytest.raises(bodo.utils.typing.BodoError, match="Unable to parse SQL Query"):
        impl(bodosql_datetime_types["table1"], "year")


@pytest.mark.slow
def test_named_param_order_by(basic_df):
    """
    Checks that Named Params cannot be used in
    ASC/DESC for order by because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select A from table1 order by A @a", {"a": a})

    with pytest.raises(bodo.utils.typing.BodoError, match="Unable to parse SQL Query"):
        impl(basic_df["table1"], "ASC")


@pytest.mark.slow
def test_named_param_str_to_date():
    """
    Checks that Named Params cannot be used in
    the format string for str_to_date
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select str_to_date(A, @a) from table1", {"a": a})

    with pytest.raises(bodo.utils.typing.BodoError, match="Unable to parse SQL Query"):
        impl(pd.DataFrame({"A": ["2017-08-29", "2017-09-29"] * 4}), "%Y-%m-%d")


@pytest.mark.slow
def test_named_param_date_format(bodosql_datetime_types):
    """
    Checks that Named Params cannot be used in
    the format string for date_format
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"table1": df})
        return bc.sql("select date_format(A, @a) from table1", {"a": a})

    with pytest.raises(bodo.utils.typing.BodoError, match="Unable to parse SQL Query"):
        impl(bodosql_datetime_types["table1"], "%Y %m %d")
