# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of struct values.
"""
import operator
from collections import namedtuple
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.array(
            [
                {"X": 1, "Y": 3.1},
                {"X": 2, "Y": 1.1},
                None,
                {"X": -1, "Y": -1.1},
                {"X": 3, "Y": 4.0},
                {"X": -3, "Y": -1.2},
                {"X": 5, "Y": 9.0},
            ]
        ),
        np.array(
            [
                {"X": 1, "Y": 3},
                {"X": 2, "Y": 1},
                None,
                {"X": -1, "Y": -1},
                {"X": 3, "Y": 4},
                {"X": -3, "Y": -1},
                {"X": 5, "Y": 9},
            ]
        ),
    ]
)
def struct_arr_value(request):
    return request.param


def test_unbox(struct_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (struct_arr_value,))
    check_func(impl2, (struct_arr_value,))


def test_getitem_int(struct_arr_value, memory_leak_check):
    def test_impl(A, i):
        return A[i]

    i = 1
    assert bodo.jit(test_impl)(struct_arr_value, i) == test_impl(struct_arr_value, i)
