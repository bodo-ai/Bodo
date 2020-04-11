# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Tests for array of list of fixed size items.
"""
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func

np.random.seed(0)


@pytest.fixture(
    params=[
        np.array([[1, 3], [2], None, [4, 5, 6], [], [1, 1]]),
        np.array([[2.0, -3.2], [2.2], None, [4.1, 5.2, 6.3], [], [1.1, 1.2]]),
    ]
)
def list_item_arr_value(request):
    return request.param


def test_unbox(list_item_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (list_item_arr_value,))
    check_func(impl2, (list_item_arr_value,))
