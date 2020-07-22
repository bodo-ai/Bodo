# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of map values.
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
        # simple types to handle in C
        np.array(
            [
                {1: 1.4, 2: 3.1},
                {7: -1.2},
                None,
                {11: 3.4, 21: 3.1, 9: 8.1},
                {4: 9.4, 6: 4.1},
                {7: -1.2},
                {},
            ]
        ),
        # nested type
        np.array(
            [
                {1: [3, 1, None], 2: [2, 1]},
                {3: [5], 7: None},
                None,
                {4: [9, 2], 6: [8, 1]},
                {7: [2]},
                {},
                {21: None, 9: []},
            ]
        ),
    ]
)
def map_arr_value(request):
    return request.param


def test_unbox(map_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (map_arr_value,))
    check_func(impl2, (map_arr_value,))
