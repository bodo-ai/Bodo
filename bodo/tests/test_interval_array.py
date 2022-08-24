# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Tests for interval arrays
"""
import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[pd.arrays.IntervalArray.from_arrays(np.arange(11), np.arange(11) + 1)]
)
def interval_array_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(interval_array_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (interval_array_value,))
    check_func(impl2, (interval_array_value,))


@pytest.mark.slow
def test_nbytes(interval_array_value, memory_leak_check):
    """Test IntervalArrayType nbytes"""

    def impl(arr):
        return arr.nbytes

    check_func(impl, (interval_array_value,))
