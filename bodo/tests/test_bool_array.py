# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import pandas as pd
import numpy as np
import pytest

import numba
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[pd.array([True, False, True, pd.NA, False]),]
)
def bool_arr_value(request):
    return request.param


def test_unbox(bool_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (bool_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (bool_arr_value,))


def test_boolean_dtype():
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.BooleanDtype(),))

    # constructor
    def impl2():
        return pd.BooleanDtype()

    check_func(impl2, ())


def test_unary_ufunc():
    ufunc = np.invert

    def test_impl(A):
        return ufunc(A.values)

    A = pd.Series([False, True, True, False, False], dtype="boolean")
    check_func(test_impl, (A,))


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_cmp(op):
    """Test comparison of two boolean arrays
    """
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A1, A2):\n"
    func_text += "  return A1.values {} A2.values\n".format(op_str)
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1 = pd.Series([False, True, True, None, True, True, False], dtype="boolean")
    A2 = pd.Series([True, True, None, False, False, False, True], dtype="boolean")
    check_func(test_impl, (A1, A2))


def test_cmp_scalar():
    """Test comparison of boolean array and a scalar
    """

    def test_impl1(A):
        return A.values == True

    def test_impl2(A):
        return True != A.values

    A = pd.Series([False, True, True, None, True, True, False], dtype="boolean")
    check_func(test_impl1, (A,))
    check_func(test_impl2, (A,))
