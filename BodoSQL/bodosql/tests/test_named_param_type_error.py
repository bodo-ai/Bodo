"""
Test that Named Parameters are either rejected or do an implicit cast
in situations where types don't match.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.

import pandas as pd
import pytest

import bodo
import bodosql
from bodo.utils.typing import BodoError
from bodosql.tests.named_params_common import *  # noqa
from bodosql.tests.utils import check_query


@pytest.mark.slow
def test_string_limit(basic_df, spark_info, memory_leak_check):
    """
    Check that limit enforces the integer requirement.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 limit @a", {"a": a})

    with pytest.raises(
        BodoError, match=r"Failure in compiling or validating SQL Query"
    ):
        impl(basic_df["TABLE1"], "no limit")


@pytest.mark.slow
def test_string_offset(basic_df, spark_info, memory_leak_check):
    """
    Check that offset enforces the integer requirement.
    """

    @bodo.jit
    def impl(df, a):
        bc = bodosql.BodoSQLContext({"TABLE1": df})
        return bc.sql("select A from table1 limit @a, 4", {"a": a})

    with pytest.raises(
        BodoError, match=r"Failure in compiling or validating SQL Query"
    ):
        impl(basic_df["TABLE1"], "no limit")


@pytest.mark.slow
def test_string_cast(
    bodosql_string_types, spark_info, int_named_params, memory_leak_check
):
    """
    Check that limit a string function will cast a non-string
    named Parameter.
    """
    query = "select CONCAT(A, @a) from table1"
    expected_output = pd.DataFrame(
        {"Result": bodosql_string_types["TABLE1"]["A"] + str(int_named_params["a"])}
    )
    check_query(
        query,
        bodosql_string_types,
        spark_info,
        named_params=int_named_params,
        expected_output=expected_output,
        check_names=False,
    )
