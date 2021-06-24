# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""Test Bodo's binary array data type
"""
import numpy as np
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        np.array([b"abc", b"c", np.nan, b"ccdefg" b"abcde", b"poiu"], object),
    ]
)
def binary_arr_value(request):
    return request.param


def test_unbox(binary_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (binary_arr_value,))
    check_func(impl2, (binary_arr_value,))
