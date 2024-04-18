# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Explicitly test for different Dynamic Parameters passing
patterns within JIT code.
"""
import numpy as np
import pandas as pd
import pytest

import bodo
import bodosql


@pytest.fixture
def simple_context():
    return bodosql.BodoSQLContext({"TABLE1": pd.DataFrame({"A": np.arange(10)})})


def test_literal_list(simple_context, memory_leak_check):
    """
    Test passing bind variables as a literal list.
    """

    @bodo.jit
    def impl(bc, var):
        return bc.sql("SELECT A FROM TABLE1 WHERE A > ?", dynamic_params_list=[var])

    output = impl(simple_context, 4).reset_index(drop=True)
    expected_output = pd.DataFrame({"A": np.arange(5, 10)})
    pd.testing.assert_frame_equal(output, expected_output)


def test_literal_tuple(simple_context, memory_leak_check):
    """
    Test passing bind variables as a literal tuple.
    """

    @bodo.jit
    def impl(bc, var):
        return bc.sql("SELECT A FROM TABLE1 WHERE A > ?", dynamic_params_list=(var,))

    output = impl(simple_context, 4).reset_index(drop=True)
    expected_output = pd.DataFrame({"A": np.arange(5, 10)})
    pd.testing.assert_frame_equal(output, expected_output)


def test_tuple_argument(simple_context, memory_leak_check):
    """
    Test passing bind variables as a tuple argument.
    """

    @bodo.jit
    def impl(bc, bind_variables):
        return bc.sql(
            "SELECT A FROM TABLE1 WHERE A > ?", dynamic_params_list=bind_variables
        )

    output = impl(simple_context, (4,)).reset_index(drop=True)
    expected_output = pd.DataFrame({"A": np.arange(5, 10)})
    pd.testing.assert_frame_equal(output, expected_output)
