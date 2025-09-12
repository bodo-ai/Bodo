import operator

import numba  # noqa TID253
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from bodo.tests.utils import check_func, get_num_test_workers


@pytest.fixture(
    params=[
        pd.array([True, False, True, pd.NA, False]),
    ]
)
def bool_arr_value(request):
    return request.param


@pytest.mark.slow
def test_np_where(memory_leak_check):
    def impl(arr):
        return np.where(arr)

    # Doesn't work with null values in Python
    A = pd.array([True, True, False, True] * 10)

    check_func(impl, (A,))


@pytest.mark.slow
def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = pd.array([True, False, True, False] * 20)

    check_func(impl, (A,))


@pytest.mark.slow
def test_np_repeat(bool_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (bool_arr_value,))


@pytest.mark.slow
def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = pd.array([True, False, True, False, False] * 10)
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_unbox(bool_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (bool_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (bool_arr_value,))


@pytest.mark.slow
def test_unbox_arrow_ext(memory_leak_check):
    """Make sure boxing/unboxing works for ArrowExtensionArray input"""

    # unbox and box
    def impl(arr_arg):
        return arr_arg

    A = pd.arrays.ArrowExtensionArray(pa.array([True, None, False, True, True]))
    check_func(impl, (A,))


@pytest.mark.slow
def test_boolean_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.BooleanDtype(),))

    # constructor
    def impl2():
        return pd.BooleanDtype()

    check_func(impl2, ())


@pytest.mark.slow
def test_unary_ufunc(memory_leak_check):
    ufunc = np.invert

    def test_impl(A):
        return ufunc(A.values)

    A = pd.Series([False, True, True, False, False], dtype="boolean")
    check_func(test_impl, (A,))


@pytest.mark.parametrize("op", [operator.eq, operator.ne])
def test_cmp(op, memory_leak_check):
    """Test comparison of two boolean arrays"""
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A1, A2):\n"
    func_text += f"  return A1.values {op_str} A2.values\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1 = pd.Series([False, True, True, None, True, True, False], dtype="boolean")
    A2 = pd.Series([True, True, None, False, False, False, True], dtype="boolean")
    check_func(test_impl, (A1, A2))


@pytest.mark.slow
def test_cmp_scalar(memory_leak_check):
    """Test comparison of boolean array and a scalar"""

    def test_impl1(A):
        return A.values == True

    def test_impl2(A):
        return True != A.values

    A = pd.Series([False, True, True, None, True, True, False], dtype="boolean")
    check_func(test_impl1, (A,))
    check_func(test_impl2, (A,))


@pytest.mark.slow
def test_max(memory_leak_check):
    def test_impl(A):
        return max(A)

    # Doesn't work with a null value in python
    A = pd.array([True, False, True, False])
    check_func(test_impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_max(bool_arr_value, memory_leak_check):
    def test_impl(A):
        return np.max(A)

    check_func(test_impl, (bool_arr_value,))


def test_min(memory_leak_check):
    def test_impl(A):
        return min(A)

    # Doesn't work with a null value in python
    A = pd.array([True, False, True, False])
    check_func(test_impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_min(bool_arr_value, memory_leak_check):
    def test_impl(A):
        return np.min(A)

    check_func(test_impl, (bool_arr_value,))


@pytest.mark.slow
def test_sum(memory_leak_check):
    def test_impl(A):
        return sum(A)

    # Doesn't work with a null value in python
    A = pd.array([True, False, True, False])
    check_func(test_impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_sum(bool_arr_value, memory_leak_check):
    def test_impl(A):
        return np.sum(A)

    check_func(test_impl, (bool_arr_value,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_prod(bool_arr_value, memory_leak_check):
    def test_impl(A):
        return np.prod(A)

    check_func(test_impl, (bool_arr_value,))


@pytest.mark.slow
def test_constant_lowering(bool_arr_value, memory_leak_check):
    def impl():
        return bool_arr_value

    check_func(impl, (), check_dtype=False, only_seq=True)


@pytest.mark.smoke
def test_setitem_int(bool_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    # get a non-null value
    bool_arr_value._mask[0] = False
    val = bool_arr_value[0]
    check_func(test_impl, (bool_arr_value, val))


@pytest.mark.smoke
def test_setitem_arr(bool_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(bool_arr_value), 11)
    val = np.random.randint(0, 2, 11, np.bool_)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # BooleanArray as value, reuses the same idx
    val = pd.arrays.BooleanArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single boolean as a value, reuses the same idx
    val = True
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = np.random.ranf(len(bool_arr_value)) < 0.2
    val = np.random.randint(0, 2, idx.sum(), np.bool_)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # BooleanArray as value, reuses the same idx
    val = pd.arrays.BooleanArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single boolean as a value, reuses the same idx
    val = True
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = slice(1, 4)
    val = np.random.randint(0, 2, 3, np.bool_)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # BooleanArray as value, reuses the same idx
    val = pd.arrays.BooleanArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single boolean as a value, reuses the same idx
    val = True
    check_func(test_impl, (bool_arr_value, idx, val), dist_test=False, copy_input=True)


@pytest.mark.slow
def test_bool_arr_nbytes(bool_arr_value, memory_leak_check):
    """Test BooleanArrayType nbytes"""

    def impl(A):
        return A.nbytes

    # 1 byte per rank for data and 1 per null bitmap
    py_out = 2 * get_num_test_workers()
    check_func(impl, (bool_arr_value,), py_output=py_out, only_1D=True)
    check_func(impl, (bool_arr_value,), py_output=2, only_seq=True)


def test_or_null(memory_leak_check):
    """
    Checks or null behavior inside boolean arrays
    """

    def test_impl(arr1, arr2):
        return arr1 | arr2

    arr1 = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    arr2 = pd.array([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (arr1, arr2))


@pytest.mark.slow
def test_or_null_numpy(memory_leak_check):
    """
    Checks or null behavior inside boolean arrays
    """

    def test_impl(arr1, arr2):
        return arr1 | arr2

    arr1 = pd.array([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    arr2 = np.array([True, False] * 3)

    check_func(test_impl, (arr1, arr2))
    check_func(test_impl, (arr2, arr1))


@pytest.mark.slow
def test_or_null_scalar(memory_leak_check):
    """
    Checks or null behavior inside boolean arrays
    """

    def test_impl(arr1, arr2):
        return arr1 | arr2

    arr = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")

    check_func(test_impl, (arr, True))
    check_func(test_impl, (arr, False))
    check_func(test_impl, (True, arr))
    check_func(test_impl, (False, arr))


@pytest.mark.slow
def test_and_null(memory_leak_check):
    """
    Checks and null behavior inside boolean arrays
    """

    def test_impl(arr1, arr2):
        return arr1 & arr2

    arr1 = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")
    arr2 = pd.array([True, False, None] * 3, dtype="boolean")

    check_func(test_impl, (arr1, arr2))


@pytest.mark.slow
def test_and_null_numpy(memory_leak_check):
    """
    Checks and null behavior inside boolean arrays with numpy
    """

    def test_impl(arr1, arr2):
        return arr1 & arr2

    arr1 = pd.array([True] * 2 + [False] * 2 + [None] * 2, dtype="boolean")
    arr2 = np.array([True, False] * 3)

    check_func(test_impl, (arr1, arr2))
    check_func(test_impl, (arr2, arr1))


@pytest.mark.slow
def test_and_null_scalar(memory_leak_check):
    """
    Checks and null behavior inside boolean arrays with scalars
    """

    def test_impl(arr1, arr2):
        return arr1 & arr2

    arr = pd.array([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean")

    check_func(test_impl, (arr, True))
    check_func(test_impl, (arr, False))
    check_func(test_impl, (True, arr))
    check_func(test_impl, (False, arr))


@pytest.mark.parametrize(
    "arr",
    [
        pytest.param(pd.array([True] * 10, dtype="boolean"), id="true"),
        pytest.param(
            pd.array([True, False] + [True] * 10, dtype="boolean"),
            id="false",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.array([True] * 10 + [None], dtype="boolean"),
            id="true-na",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.array([True, False] + [True] * 10 + [None], dtype="boolean"),
            id="false-na",
        ),
        pytest.param(
            pd.array([None] * 10, dtype="boolean"), id="all-na", marks=pytest.mark.slow
        ),
        pytest.param(pd.array([], dtype="boolean"), id="empty", marks=pytest.mark.slow),
    ],
)
@pytest.mark.slow
def test_all(arr, memory_leak_check):
    """Test BooleanArray.all()"""

    def impl(A):
        return A.all()

    check_func(impl, (arr,))


def test_to_numpy(memory_leak_check):
    """Test BooleanArray.to_numpy()"""

    # Note: we don't test with NAs because currently we cast null to False,
    # which won't match pandas
    arr = pd.array([True] * 10 + [False] * 10, dtype="boolean")

    def impl(A):
        return A.to_numpy()

    check_func(impl, (arr,))


def test_arr_astype_same(memory_leak_check):
    """Test BooleanArray.astype(pd.BooleanDtype()) is supported."""

    def impl(A):
        return A.astype(pd.BooleanDtype())

    arr = pd.array([True] * 10 + [False] * 10, dtype="boolean")
    check_func(impl, (arr,))


def test_series_astype_same(memory_leak_check):
    """Test BooleanSeries.astype(pd.BooleanDtype()) is supported."""

    def impl(S):
        return S.astype(pd.BooleanDtype())

    S = pd.Series([True] * 10 + [False] * 10, dtype="boolean")
    check_func(impl, (S,))
