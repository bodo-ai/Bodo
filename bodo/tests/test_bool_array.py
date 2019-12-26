# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (
    count_array_REPs,
    count_parfor_REPs,
    count_parfor_OneDs,
    count_array_OneDs,
    dist_IR_contains,
    get_start_end,
    check_func,
)

np.random.seed(0)


def test_unary_ufunc():
    ufunc = np.invert

    def test_impl(A):
        return ufunc(A.values)

    A = pd.Series([False, True, True, False, False])
    check_func(test_impl, (A,))


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_cmp(op):
    """Test comparison of two boolean arrays
    """
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A1, A2):\n"
    func_text += "  return A1.values {} A2.values\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1 = pd.Series([False, True, True, None, True, True, False])
    A2 = pd.Series([True, True, None, False, False, False, True])
    check_func(test_impl, (A1, A2))


def test_cmp_scalar():
    """Test comparison of boolean array and a scalar
    """

    def test_impl1(A):
        return A.values == True

    def test_impl2(A):
        return True != A.values

    A = pd.Series([False, True, True, None, True, True, False])
    check_func(test_impl1, (A,))
    check_func(test_impl2, (A,))
