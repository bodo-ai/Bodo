# Copyright (C) 2019 Bodo Inc. All rights reserved.

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import SeriesOptTestPipeline, check_func, dist_IR_contains


@pytest.fixture(
    params=[
        pytest.param(
            pa.array(
                ["abc", "b", None, "abc", None, "b", "cde"],
                type=pa.dictionary(pa.int32(), pa.string()),
            )
        ),
    ]
)
def dict_arr_value(request):
    return request.param


@pytest.mark.slow
def test_unbox(dict_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (dict_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (dict_arr_value,))


@pytest.mark.slow
def test_len(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (dict_arr_value,))


@pytest.mark.slow
def test_shape(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    # PyArrow doesn't support shape
    assert bodo.jit(test_impl)(dict_arr_value) == (len(dict_arr_value),)


@pytest.mark.slow
def test_dtype(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    # PyArrow doesn't support dtype
    assert bodo.jit(test_impl)(dict_arr_value) == pd.StringDtype()


@pytest.mark.slow
def test_ndim(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    # PyArrow doesn't support ndim
    assert bodo.jit(test_impl)(dict_arr_value) == 1


@pytest.mark.slow
def test_copy(dict_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    # PyArrow doesn't support copy
    np.testing.assert_array_equal(bodo.jit(test_impl)(dict_arr_value), dict_arr_value)


@pytest.mark.slow
def test_cmp_opt(dict_arr_value, memory_leak_check):
    """test optimizaton of comparison operators (eq, ne) for dict array"""

    def impl1(A, val):
        return A == val

    def impl2(A, val):
        return val == A

    def impl3(A, val):
        return A != val

    def impl4(A, val):
        return val != A

    # convert to Pandas array since PyArrow doesn't support cmp operators
    pd_arr = pd.array(dict_arr_value.to_numpy(False), "string")

    for val in ("abc", "defg"):
        check_func(impl1, (dict_arr_value, val), py_output=(pd_arr == val))
        check_func(impl2, (dict_arr_value, val), py_output=(val == pd_arr))
        check_func(impl3, (dict_arr_value, val), py_output=(pd_arr != val))
        check_func(impl4, (dict_arr_value, val), py_output=(val != pd_arr))

    # make sure IR has the optimized functions
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func(dict_arr_value, "abc")
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_eq")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func("abc", dict_arr_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_eq")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl3)
    bodo_func(dict_arr_value, "abc")
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_ne")
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl4)
    bodo_func("abc", dict_arr_value)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "dict_arr_ne")


@pytest.mark.slow
def test_int_convert_opt(memory_leak_check):
    """test optimizaton of integer conversion for dict array"""

    def impl(A):
        return pd.Series(A).astype("Int32")

    data = ["14", None, "-3", "11", "-155", None]
    A = pa.array(data, type=pa.dictionary(pa.int32(), pa.string()))
    check_func(
        impl, (A,), py_output=pd.Series(pd.array(data, "string")).astype("Int32")
    )
    # make sure IR has the optimized function
    bodo_func = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl)
    bodo_func(A)
    f_ir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    assert dist_IR_contains(f_ir, "convert_dict_arr_to_int")


@pytest.mark.slow
def test_gatherv_rm(dict_arr_value, memory_leak_check):
    """make sure gatherv() is removed in non-distributed pipeline without errors"""

    @bodo.jit(distributed=False)
    def impl(A):
        return pd.Series(A).unique()

    res = impl(dict_arr_value)
    pd.testing.assert_series_equal(
        pd.Series(pd.Series(dict_arr_value).unique()), pd.Series(res)
    )


def test_dict_array_unify(dict_arr_value):
    """Tests that unifying dict arrays works as expected."""
    # TODO: Add memory leak check, casting bug

    @bodo.jit
    def impl(A):
        # This condition is False at runtime, so unifying
        # the arrays will require casting the series.
        if len(A) > 30:
            A = pd.Series(A).sort_values().values
        return A

    bodo_out = impl(dict_arr_value)
    # Map NaN to None to match arrow
    bodo_out[pd.isna(bodo_out)] = None
    py_output = dict_arr_value.to_numpy(False)
    np.testing.assert_array_equal(py_output, bodo_out)
