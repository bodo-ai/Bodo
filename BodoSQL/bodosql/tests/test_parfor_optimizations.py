"""Tests that various optimizations inside of parfors work properly
with Bodo data types. These tests should be used to check that specific
compiler optimizations (i.e. dce) are working properly.
"""

import pandas as pd
from numba import prange

import bodo
import bodosql
from bodo.tests.test_parfor_optimizations import _check_num_parfors
from bodo.tests.utils import ParforTestPipeline


def test_parfor_fusion_scalar_optional_getitem(memory_leak_check):
    """
    Tests that scalar_optional_getitem can be used to compute parfor
    fusion.
    """
    out_arr_typ = bodo.types.IntegerArrayType(bodo.types.int64)

    def impl(arr):
        n1 = len(arr)
        arr2 = bodo.utils.utils.alloc_type(n1, out_arr_typ, (-1))
        for i in prange(n1):
            arr2[i] = bodosql.kernels.add_numeric(
                bodo.utils.indexing.scalar_optional_getitem(arr, i), 1
            )
        n2 = len(arr2)
        arr3 = bodo.utils.utils.alloc_type(n2, out_arr_typ, (-1))
        for i in prange(n2):
            arr3[i] = bodosql.kernels.add_numeric(
                3, bodo.utils.indexing.scalar_optional_getitem(arr2, i)
            )
        return arr3

    input_arr = pd.array([0, 1, 2, 3, 4, 5] * 10 + [None, None, None], dtype="Int64")
    bodo_func = bodo.jit(pipeline_class=ParforTestPipeline)(impl)
    bodo_func(input_arr)
    _check_num_parfors(bodo_func, 1)
