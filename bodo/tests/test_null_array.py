# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Tests for the null array type, which is an array of all nulls
that can be cast to any type. See null_arr_ext.py for the
core implementation.
"""
import pandas as pd

import bodo
from bodo.tests.utils import check_func


def test_nullable_bool_cast(memory_leak_check):
    """
    Tests casting a nullable array to a boolean array.
    """

    def impl(n):
        null_arr = bodo.libs.null_arr_ext.init_null_array(n)
        return null_arr.astype(pd.BooleanDtype())

    n = 10
    arr = pd.array([None] * n, dtype=pd.BooleanDtype())
    check_func(impl, [n], py_output=arr)
