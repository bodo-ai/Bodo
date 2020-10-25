# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of tuple values.
"""
import operator
from collections import namedtuple

import numba
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        # heterogeneous values
        np.array(
            [
                (1, 3.1),
                (2, 1.1),
                None,
                (-1, 7.8),
                (3, 4.0),
                (-3, -1.2),
                (None, 9.0),
            ]
        ),
        # homogeneous values
        np.array(
            [
                (1.1, 3.1),
                (2.1, 1.1),
                None,
                (-1.1, -1.1),
                (3.1, 4.1),
                (-3.1, -1.1),
                (5.1, 9.1),
            ]
        ),
    ]
)
def tuple_arr_value(request):
    return request.param


def test_unbox(tuple_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (tuple_arr_value,))
    check_func(impl2, (tuple_arr_value,))
