# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of map values.
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
                {11: [2, -1]},
                {1: [-1]},
                {},
                {21: None, 9: []},
            ]
        ),
    ]
)
def map_arr_value(request):
    return request.param


# there is a memory leak probably due to the decref issue in to_arr_obj_if_list_obj()
# TODO: fix leak and enable test
# def test_unbox(map_arr_value, memory_leak_check):
@pytest.mark.slow
def test_unbox(map_arr_value):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (map_arr_value,))
    check_func(impl2, (map_arr_value,))
