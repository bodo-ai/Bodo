"""
Test that Named Parameters can't be used where constants are
required.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.utils.typing import BodoError


@pytest.mark.slow
def test_named_param_extract(bodosql_datetime_types, memory_leak_check):
    """
    Checks that Named Params cannot be used in
    extract because the name must be a literal identifier.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select Extract(@a from A) from table1", {"a": a})

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        impl(bodosql_datetime_types["TABLE1"], "year")


@pytest.mark.slow
def test_named_param_order_by(basic_df, memory_leak_check):
    """
    Checks that Named Params cannot be used in
    ASC/DESC for order by because it must be a literal identifier.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 order by A @a", {"a": a})

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        impl(basic_df["TABLE1"], "ASC")


@pytest.mark.slow
def test_named_param_str_to_date(memory_leak_check):
    """
    Checks that Named Params cannot be used in
    the format string for str_to_date
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select str_to_date(A, @a) from table1", {"a": a})

    with pytest.raises(BodoError, match="Failure in compiling or validating SQL Query"):
        impl(pd.DataFrame({"A": ["2017-08-29", "2017-09-29"] * 4}), "%Y-%m-%d")


@pytest.mark.slow
def test_named_param_date_format(bodosql_datetime_types, memory_leak_check):
    """
    Checks that Named Params cannot be used in
    the format string for date_format
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select date_format(A, @a) from table1", {"a": a})

    with pytest.raises(BodoError, match="Failure in compiling or validating SQL Query"):
        impl(bodosql_datetime_types["TABLE1"], "%Y %m %d")


@pytest.mark.slow
def test_bind_variable_extract(bodosql_datetime_types, memory_leak_check):
    """
    Checks that bind variables cannot be used in
    extract because the name must be a literal identifier.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select Extract(? from A) from table1", None, (a,))

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        impl(bodosql_datetime_types["TABLE1"], "year")


@pytest.mark.slow
def test_bind_variable_order_by(basic_df, memory_leak_check):
    """
    Checks that bind variables cannot be used in
    ASC/DESC for order by because it must be a literal identifier.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 order by A ?", None, (a,))

    with pytest.raises(BodoError, match="Failure encountered while parsing SQL Query"):
        impl(basic_df["TABLE1"], "ASC")


@pytest.mark.slow
def test_bind_variable_str_to_date(memory_leak_check):
    """
    Checks that bind variables cannot be used in
    the format string for str_to_date
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select str_to_date(A, ?) from table1", None, (a,))

    with pytest.raises(BodoError, match="Failure in compiling or validating SQL Query"):
        impl(pd.DataFrame({"A": ["2017-08-29", "2017-09-29"] * 4}), "%Y-%m-%d")


@pytest.mark.slow
def test_bind_variable_date_format(bodosql_datetime_types, memory_leak_check):
    """
    Checks that Bind Variables cannot be used in
    the format string for date_format
    because it must be a literal.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select date_format(A, ?) from table1", None, (a,))

    with pytest.raises(BodoError, match="Failure in compiling or validating SQL Query"):
        impl(bodosql_datetime_types["TABLE1"], "%Y %m %d")
