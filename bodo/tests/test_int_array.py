import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import (count_array_REPs, count_parfor_REPs,
    count_parfor_OneDs, count_array_OneDs, dist_IR_contains, get_start_end,
    check_func)


@pytest.fixture(params = [
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int32),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, -3, 2, 3, 10], np.int64),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint8),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint32),
        np.array([False, True, True, False, False])),
    pd.arrays.IntegerArray(np.array([1, 4, 2, 3, 10], np.uint64),
        np.array([False, True, True, False, False])),
    # large array
    pd.arrays.IntegerArray(np.random.randint(0, 100, 1211),
        np.random.ranf(1211)<.3),
])
def int_arr_value(request):
    return request.param


def test_unbox(int_arr_value):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (int_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (int_arr_value,))
