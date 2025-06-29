import operator

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, get_num_test_workers


def get_random_integerarray(tot_size):
    np.random.seed(0)
    return pd.arrays.IntegerArray(
        np.random.randint(0, 100, tot_size), np.random.ranf(tot_size) < 0.3
    )


@pytest.fixture(
    params=[
        pytest.param(
            pd.arrays.IntegerArray(
                np.array([1, -3, 2, 3, 10], np.int8),
                np.array([False, True, True, False, False]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            np.array([1, None, None, 3, 10], object),
        ),
        pytest.param(
            pd.arrays.IntegerArray(
                np.array([1, -3, 2, 3, 10], np.int64),
                np.array([False, True, True, False, False]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    pd.arrays.IntegerArray(
                        np.array([1, 4, 2, 3, 10], np.uint8),
                        np.array([False, True, True, False, False]),
                    )
                )
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.arrays.IntegerArray(
                np.array([1, 4, 2, 3, 10], np.uint32),
                np.array([False, True, True, False, False]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.arrays.IntegerArray(
                np.array([1, 4, 2, 3, 10], np.uint64),
                np.array([False, True, True, False, False]),
            ),
            marks=pytest.mark.slow,
        ),
        # large array
        get_random_integerarray(1211),
    ]
)
def int_arr_value(request):
    return request.param


def test_np_repeat(int_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (int_arr_value,))


def test_np_where(memory_leak_check):
    def impl(arr):
        return np.where(arr)

    # Doesn't work with null values in Python
    A = pd.arrays.IntegerArray(
        np.array([1, 3, 0, 3, 10], np.int8),
        np.array([False, False, False, False, False]),
    )

    check_func(impl, (A,))


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = pd.array(np.array([1, 4, 2, 3, 10, 122, 4], np.uint8))
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    # Doesn't work with null values in Numpy
    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10] * 10, np.int8),
        np.array([False, False, False, False, False] * 10),
    )

    check_func(impl, (A,))


def test_max(memory_leak_check):
    def impl(arr):
        return max(arr)

    # Doesn't work with null values in Python
    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, False, False, False, False]),
    )

    check_func(impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_max(int_arr_value, memory_leak_check):
    def impl(arr):
        return np.max(arr)

    check_func(impl, (int_arr_value,))


def test_min(memory_leak_check):
    def impl(arr):
        return min(arr)

    # Doesn't work with null values in Python
    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, False, False, False, False]),
    )

    check_func(impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_min(int_arr_value, memory_leak_check):
    def impl(arr):
        return np.min(arr)

    check_func(impl, (int_arr_value,))


def test_sum(memory_leak_check):
    def impl(arr):
        return sum(arr)

    # Doesn't work with null values in Python
    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, False, False, False, False]),
    )

    check_func(impl, (A,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_sum(int_arr_value, memory_leak_check):
    def impl(arr):
        return np.sum(arr)

    check_func(impl, (int_arr_value,))


@pytest.mark.skip("Reduce not supported in Pandas")
def test_np_prod(int_arr_value, memory_leak_check):
    def impl(arr):
        return np.prod(arr)

    check_func(impl, (int_arr_value,))


@pytest.mark.slow
def test_unbox(int_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (int_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (int_arr_value,))


@pytest.mark.slow
def test_int_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.Int32Dtype(),))
    check_func(impl, (pd.Int8Dtype(),))
    check_func(impl, (pd.UInt32Dtype(),))

    # constructors
    def impl2():
        return pd.Int8Dtype()

    check_func(impl2, ())

    def impl3():
        return pd.UInt32Dtype()

    check_func(impl3, ())


@pytest.mark.smoke
def test_setitem_int(int_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    # get a non-null value
    val = int_arr_value[0]
    check_func(test_impl, (int_arr_value, val), copy_input=True, only_seq=True)


@pytest.mark.smoke
def test_setitem_arr(int_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    dtype = pd.array(int_arr_value).dtype.numpy_dtype
    np.random.seed(0)
    n = len(int_arr_value)
    idx = np.random.choice(n, n, replace=False)
    val = np.random.randint(0, 50, n, dtype)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # IntegerArray as value, reuses the same idx
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single integer as a value, reuses the same idx
    val = 4
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = np.random.ranf(len(int_arr_value)) < 0.2
    val = np.random.randint(0, 50, idx.sum(), dtype)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # IntegerArray as value, reuses the same idx
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single integer as a value, reuses the same idx
    val = 4
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    idx = slice(1, 4)
    val = np.random.randint(0, 50, 3, dtype)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # IntegerArray as value, reuses the same idx
    val = pd.arrays.IntegerArray(val, np.random.ranf(len(val)) < 0.2)
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)

    # Single integer as a value, reuses the same idx
    val = 4
    check_func(test_impl, (int_arr_value, idx, val), dist_test=False, copy_input=True)


@pytest.mark.slow
def test_len(memory_leak_check):
    def test_impl(A):
        return len(A)

    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_shape(memory_leak_check):
    def test_impl(A):
        return A.shape

    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_list(memory_leak_check):
    def test_impl(A):
        return list(A)

    A = pd.array([3, 1, 2], "Int64")
    check_func(test_impl, (A,), only_seq=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    # avoiding isnat since only supported for datetime/timedelta
    "ufunc",
    [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 1 and f != np.isnat],
)
def test_unary_ufunc(ufunc):
    def test_impl(A):
        return ufunc(A)

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, -3, 10], np.int32),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


def test_unary_ufunc_explicit_np(memory_leak_check):
    def test_impl(A):
        return np.negative(A)

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, -3, 10], np.int32),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ufunc", [f for f in numba.np.ufunc_db.get_ufuncs() if f.nin == 2 and f.nout == 1]
)
def test_binary_ufunc(ufunc, memory_leak_check):
    # Need suppport for floating array when doing true division
    if ufunc == np.true_divide:
        check_dtype = False
    else:
        check_dtype = True

    # Bodo's overflow behavior can be different on Python 3.11 (TODO: investigate why)
    if ufunc == np.power:
        return

    def test_impl(A1, A2):
        return ufunc(A1, A2)

    A1 = pd.arrays.IntegerArray(
        np.array([1, 1, 1, 2, 10], np.int32),
        np.array([False, True, True, False, False]),
    )
    A2 = pd.arrays.IntegerArray(
        np.array([4, 2, 1, 1, 12], np.int32),
        np.array([False, False, True, True, False]),
    )
    arr = np.array([1, 3, 7, 11, 2])
    check_func(test_impl, (A1, A2), check_dtype=check_dtype)
    check_func(test_impl, (A1, arr), check_dtype=check_dtype)
    check_func(test_impl, (arr, A2), check_dtype=check_dtype)


def test_add(memory_leak_check):
    def test_impl(A, other):
        return A + other

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, -3, 10], np.int64),
        np.array([False, True, True, False, False]),
    )

    check_func(test_impl, (A, A))
    check_func(test_impl, (A, 2))
    check_func(test_impl, (2, A))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
)
def test_binary_op(op, memory_leak_check):
    # Pandas doesn't support these operators yet, but Bodo does to be able to
    # replace all numpy arrays
    if op in (
        operator.lshift,
        operator.rshift,
        operator.and_,
        operator.or_,
        operator.xor,
    ):
        return
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += f"  return A {op_str} other\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    # TODO: use int32 when Numba #4449 is resolved
    A1 = pd.arrays.IntegerArray(
        np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]),
    )
    A2 = pd.arrays.IntegerArray(
        np.array([4, 2, 1, 1, 12], np.int64),
        np.array([False, False, True, True, False]),
    )
    check_func(test_impl, (A1, A2))
    check_func(test_impl, (A1, 2))
    check_func(test_impl, (2, A2))


def test_div_by_zero(memory_leak_check):
    """make sure division by zero doesn't throw an error (returns inf in Pandas)"""

    def impl1(A1, A2):
        return A1 / A2

    def impl2(A1, A2):
        return A1 // A2

    def impl3(A1, A2):
        A1 /= A2
        return A1

    def impl4(A1, A2):
        A1 //= A2
        return A1

    A1 = pd.array([1, 2, None, 3, 11, 0], "Int64")
    A2 = pd.array([1, None, 3, 0, 11, 4], "Int64")

    check_func(impl1, (A1, A2))
    check_func(impl2, (A1, A2))
    # TODO: support inplace truediv (difficult due to type change to float)
    # check_func(impl3, (A1, A2), copy_input=True)
    check_func(impl4, (A1, A2), copy_input=True)


def test_inplace_iadd(memory_leak_check):
    def test_impl(A, other):
        A += other
        return A

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, -3, 10], np.int32),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A, A), copy_input=True)
    check_func(test_impl, (A, 2), copy_input=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
)
def test_inplace_binary_op(op, memory_leak_check):
    # Numba can't handle itruediv
    # pandas doesn't support the others
    if op in (
        operator.ilshift,
        operator.irshift,
        operator.iand,
        operator.ior,
        operator.ixor,
        operator.itruediv,
    ):
        return
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += f"  A {op_str} other\n"
    func_text += "  return A\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1 = pd.arrays.IntegerArray(
        np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]),
    )
    A2 = pd.arrays.IntegerArray(
        np.array([4, 2, 1, 1, 12], np.int64),
        np.array([False, False, True, True, False]),
    )
    # TODO: test inplace change properly
    check_func(test_impl, (A1, A2), copy_input=True)
    check_func(test_impl, (A1, 2), copy_input=True)


@pytest.mark.skip(reason="pd.arrays.IntegerArray doesn't support unary op")
@pytest.mark.parametrize("op", (operator.neg, operator.invert, operator.pos))
def test_unary_op(op, memory_leak_check):
    # TODO: fix operator.pos
    if op == operator.pos:
        return

    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A):\n"
    func_text += f"  return {op_str} A\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_dtype(int_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    # Bodo returns Pandas dtype so convert others
    dtype = int_arr_value.dtype
    if dtype == np.object_:
        dtype = pd.Int64Dtype()
    if isinstance(dtype, pd.ArrowDtype):
        dtype = pd.core.arrays.integer.IntegerDtype._get_dtype_mapping()[
            dtype.numpy_dtype
        ]

    check_func(test_impl, (int_arr_value,), py_output=dtype)


@pytest.mark.slow
def test_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    A = pd.arrays.IntegerArray(
        np.array([1, 1, 1, 2, 10], np.int64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_copy(int_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (int_arr_value,))


def test_astype_fast(memory_leak_check):
    def test_impl(A, dtype):
        return A.astype(dtype)

    A = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int8),
        np.array([False, True, True, False, False]),
    )
    dtype = pd.Int8Dtype()
    check_func(test_impl, (A, dtype))


@pytest.mark.parametrize("dtype", [pd.Int8Dtype(), np.float64])
def test_astype(int_arr_value, dtype, memory_leak_check):
    def test_impl(A, dtype):
        return A.astype(dtype)

    if isinstance(int_arr_value, np.ndarray):
        pytest.skip("Numpy arrays don't support Pandas dtypes")

    check_func(test_impl, (int_arr_value, dtype))


def test_astype_str(int_arr_value, memory_leak_check):
    def test_impl(A):
        return A.astype("float64")

    check_func(test_impl, (int_arr_value,))


def test_unique(int_arr_value, memory_leak_check):
    def test_impl(A):
        return A.unique()

    if isinstance(int_arr_value, np.ndarray):
        pytest.skip("Numpy arrays don't have unique()")

    # only sequential check since not directly parallelized
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_extension_array_equal(
        bodo_func(int_arr_value), test_impl(int_arr_value), check_dtype=False
    )


def test_sum_method(int_arr_value, memory_leak_check):
    """test IntegerArray.sum()"""

    def test_impl(A):
        return A.sum()

    if not isinstance(int_arr_value, pd.arrays.IntegerArray):
        pytest.skip("Only actual IntegerArray has sum()")

    check_func(test_impl, (int_arr_value,))


@pytest.mark.slow
def test_constant_lowering(int_arr_value, memory_leak_check):
    def impl():
        return int_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(int_arr_value), check_dtype=False
    )


@pytest.mark.slow
def test_int_arr_nbytes(memory_leak_check):
    """Test IntegerArrayType nbytes"""

    def impl(A):
        return A.nbytes

    arr = pd.arrays.IntegerArray(
        np.array([1, -3, 2, 3, 10], np.int64),
        np.array([False, True, True, False, False]),
    )
    n_pes = get_num_test_workers()
    py_out = 40 + n_pes  # 1 extra byte for null_bit_map per rank
    check_func(impl, (arr,), py_output=py_out, only_1D=True)
    check_func(impl, (arr,), py_output=41, only_seq=True)
