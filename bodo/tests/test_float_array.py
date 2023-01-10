# Copyright (C) 2022 Bodo Inc. All rights reserved.

import operator

import numba
import numpy as np
import pandas as pd
import pytest
from numba.core import types

import bodo
from bodo.tests.utils import check_func

nullable_float_marker = pytest.mark.skipif(
    not bodo.libs.float_arr_ext._use_nullable_float,
    reason="nullable float not fully supported yet",
)


@pytest.fixture(
    params=[
        pytest.param(pd.Float32Dtype(), id="float32"),
        pytest.param(pd.Float64Dtype(), id="float64"),
    ]
)
def float_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param([], id="empty"),
        pytest.param([float(i) for i in range(30)], id="all_floats"),
        pytest.param([1.0, None, 3.0] * 10, id="floats_and_nulls"),
        pytest.param([None] * 30, id="all_nulls"),
        pytest.param([1.0, None, np.nan] * 10, id="floats_nulls_and_nans"),
        pytest.param([np.nan] * 30, id="all_nans"),
    ]
)
def nullable_float_values(request):
    return request.param


@pytest.mark.slow
def test_float_arr_unbox(nullable_float_values, float_dtype, memory_leak_check):
    def impl(arr):
        return 42

    arr = pd.array(nullable_float_values, dtype=float_dtype)

    check_func(impl, (arr,))


@pytest.mark.slow
def test_float_arr_box_unbox(nullable_float_values, float_dtype, memory_leak_check):
    def impl(arr):
        return arr

    check_func(impl, (pd.array(nullable_float_values, dtype=float_dtype),))


@pytest.mark.slow
@pytest.mark.parametrize(
    "target_dtype",
    [
        pytest.param(pd.Float32Dtype(), id="Float32"),
        pytest.param(pd.Float64Dtype(), id="Float64"),
        pytest.param(pd.Int32Dtype(), id="Int32"),
    ],
)
@pytest.mark.parametrize(
    "copy",
    [
        pytest.param(True, id="copy"),
        pytest.param(False, id="no_copy"),
    ],
)
def test_float_arr_astype(float_dtype, target_dtype, copy, memory_leak_check):
    def impl(val, dtype):
        return val.astype(dtype, copy=copy)

    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)
    check_func(impl, (arr, target_dtype), dist_test=False)

    series = pd.Series(arr)
    check_func(impl, (series, target_dtype), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "index_fn",
    [
        pytest.param(
            lambda arr: arr[0],
            id="arr[0]",
            marks=pytest.mark.skip(reason="NA/NaN indexing mismatch"),
        ),
        pytest.param(lambda arr: arr[1], id="arr[1]"),
        pytest.param(lambda arr: arr[-1], id="arr[-1]"),
    ],
)
def test_float_arr_scalar_indexing(index_fn, float_dtype, memory_leak_check):
    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)
    check_func(index_fn, (arr,), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "mask",
    [
        pytest.param([True] * 30, id="all_true"),
        pytest.param([False] * 30, id="all_false"),
        pytest.param([True, False] * 15, id="alternating"),
    ],
)
def test_float_arr_bool_array_indexing(float_dtype, mask, memory_leak_check):
    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)

    def impl(arr, mask):
        return arr[mask]

    check_func(impl, (arr, mask), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "index_arr",
    [
        pytest.param([0] * 30, id="repeating"),
        pytest.param(list(range(0, 30, 4)), id="subset"),
        pytest.param(list(range(30))[::-1], id="reordered"),
    ],
)
def test_float_arr_int_array_indexing(float_dtype, index_arr, memory_leak_check):
    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)

    def impl(arr, index_arr):
        return arr[index_arr]

    check_func(impl, (arr, index_arr), dist_test=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "iloc_fn",
    [
        pytest.param(
            lambda arr: arr.iloc[0],
            id="iloc[0]",
            marks=pytest.mark.skip(reason="NA/NaN indexing mismatch"),
        ),
        pytest.param(lambda arr: arr.iloc[1:3], id="iloc[1:3]"),
        pytest.param(
            lambda arr: arr.iloc[1:3].astype(np.float32),
            id="iloc[1:3].astype(np.float32)",
            marks=nullable_float_marker,
        ),
    ],
)
def test_float_arr_series_iloc(iloc_fn, float_dtype, memory_leak_check):
    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)
    series = pd.Series(arr)
    check_func(iloc_fn, (series,), check_dtype=False, dist_test=False)


@nullable_float_marker
def test_float_arr_coerce_scalar(memory_leak_check):
    arr_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

    n = 50
    scalar_float = 1.0
    full_output = pd.array([scalar_float] * n, dtype=pd.Float64Dtype())
    null_output = pd.arrays.FloatingArray(
        np.array([np.NaN] * 50), np.array([True] * 50)
    )

    def impl1(arg, len):
        return bodo.utils.conversion.coerce_scalar_to_array(arg, len, arr_dtype)

    def impl2(arg, len, flag):
        # Test optional type
        optional_arg = arg if flag else None
        return bodo.utils.conversion.coerce_scalar_to_array(
            optional_arg, len, arr_dtype
        )

    def impl3(len):
        return bodo.utils.conversion.coerce_scalar_to_array(
            scalar_float, len, arr_dtype
        )

    def impl4(len):
        return bodo.utils.conversion.coerce_scalar_to_array(None, len, arr_dtype)

    check_func(impl1, (scalar_float, n), py_output=full_output)
    check_func(impl1, (None, n), py_output=null_output)
    check_func(impl2, (scalar_float, n, True), py_output=full_output)
    check_func(impl2, (scalar_float, n, False), py_output=null_output)
    check_func(impl3, (n,), py_output=full_output)
    check_func(impl4, (n,), py_output=null_output)


@pytest.mark.slow
def test_float_arr_isna(nullable_float_values, float_dtype, memory_leak_check):
    def impl(arr):
        return pd.isna(arr)

    check_func(impl, (pd.array(nullable_float_values, dtype=float_dtype),))


FLOAT_SCALAR = 1.0

INT_SCALAR = 1

FLOAT_ARR = np.array([float(i) for i in range(30)], dtype=np.float32)

INT_ARR = np.array([i for i in range(30)], dtype=np.int32)

NULLABLE_FLOAT_ARR = pd.array(
    [float(i) if i % 2 else None for i in range(30)],
    dtype=pd.Float32Dtype(),
)

NULLABLE_INT_ARR = (
    pd.array([i if i % 2 else None for i in range(30)], dtype=pd.Int32Dtype()),
)

FLOAT_SERIES = pd.Series(FLOAT_ARR)

INT_SERIES = pd.Series(INT_ARR)

NULLABLE_FLOAT_SERIES = pd.Series(NULLABLE_FLOAT_ARR)

NULLABLE_INT_SERIES = pd.Series(NULLABLE_INT_ARR)

ARR_BINOP_ARGS = [
    pytest.param(
        (NULLABLE_FLOAT_ARR, NULLABLE_FLOAT_ARR),
        id="nullable_float_arr-nullable_float_arr",
    ),
    pytest.param((NULLABLE_FLOAT_ARR, FLOAT_ARR), id="nullable_float_arr-float_arr"),
    pytest.param(
        (NULLABLE_FLOAT_ARR, NULLABLE_INT_ARR),
        id="nullable_float_arr-nullable_int_arr",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
    pytest.param((NULLABLE_FLOAT_ARR, INT_ARR), id="nullable_float_arr-int_arr"),
    pytest.param(
        (NULLABLE_FLOAT_ARR, FLOAT_SCALAR), id="nullable_float_arr-float_scalar"
    ),
    pytest.param((NULLABLE_FLOAT_ARR, INT_SCALAR), id="nullable_float_arr-int_scalar"),
]

SERIES_BINOP_ARGS = [
    pytest.param(
        (NULLABLE_FLOAT_ARR, INT_SERIES),
        id="nullable_float_arr-int_series",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
    pytest.param(
        (NULLABLE_FLOAT_ARR, NULLABLE_INT_SERIES),
        id="nullable_float_arr-nullable_int_series",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
    pytest.param(
        (NULLABLE_FLOAT_ARR, FLOAT_SERIES), id="nullable_float_arr-float_series"
    ),
    pytest.param(
        (NULLABLE_FLOAT_ARR, NULLABLE_FLOAT_SERIES),
        id="nullable_float_arr-nullable_float_series",
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, FLOAT_ARR), id="nullable_float_series-float_arr"
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, NULLABLE_FLOAT_ARR),
        id="nullable_float_series-nullable_float_arr",
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, FLOAT_SCALAR),
        id="nullable_float_series-float_scalar",
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, FLOAT_SERIES),
        id="nullable_float_series-float_series",
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, NULLABLE_FLOAT_SERIES),
        id="nullable_float_series-nullable_float_series",
    ),
    pytest.param((NULLABLE_FLOAT_SERIES, INT_ARR), id="nullable_float_series-int_arr"),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, NULLABLE_INT_ARR),
        id="nullable_float_series-nullable_int_arr",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, INT_SCALAR), id="nullable_float_series-int_scalar"
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, INT_SERIES),
        id="nullable_float_series-int_series",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
    pytest.param(
        (NULLABLE_FLOAT_SERIES, NULLABLE_INT_SERIES),
        id="nullable_float_series-nullable_int_series",
        marks=pytest.mark.skip(reason="pandas doesn't support yet"),
    ),
]


@nullable_float_marker
@pytest.mark.slow
@pytest.mark.parametrize(
    # avoiding isnat since only supported for datetime/timedelta
    # avoiding invert/logical not since only supported for int/bool
    "ufunc",
    [
        f
        for f in numba.np.ufunc_db.get_ufuncs()
        if f.nin == 1 and f not in (np.isnat, np.invert, np.logical_not)
    ],
)
def test_float_arr_unary_ufunc(ufunc):
    def test_impl(A):
        return ufunc(A)

    A = NULLABLE_FLOAT_ARR

    check_func(test_impl, (A,))


@pytest.mark.slow
@pytest.mark.parametrize("binop_args", ARR_BINOP_ARGS)
@pytest.mark.parametrize(
    # avoiding int operations
    # avoiding known pandas diff for np.power
    "ufunc",
    [
        f
        for f in numba.np.ufunc_db.get_ufuncs()
        if f.nin == 2
        and f.nout == 1
        and f
        not in (
            np.gcd,
            np.lcm,
            np.ldexp,
            np.bitwise_and,
            np.bitwise_or,
            np.bitwise_xor,
            np.logical_and,
            np.logical_or,
            np.logical_xor,
            np.left_shift,
            np.right_shift,
            np.power,
        )
    ],
)
def test_float_arr_binary_ufunc(binop_args, ufunc, memory_leak_check):
    def test_impl(A1, A2):
        return ufunc(A1, A2)

    A1, A2 = binop_args

    check_func(test_impl, (A1, A2))
    check_func(test_impl, (A2, A1))


@pytest.mark.slow
@pytest.mark.parametrize("binop_args", ARR_BINOP_ARGS + SERIES_BINOP_ARGS)
@pytest.mark.parametrize(
    # avoiding int operations
    "op",
    [
        op
        for op in numba.core.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
        if op
        not in (
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.pow,
        )
    ],
)
def test_float_arr_binary_op(binop_args, op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += f"  return A {op_str} other\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1, A2 = binop_args

    check_func(test_impl, (A1, A2), check_dtype=False)
    check_func(test_impl, (A2, A1), check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize("binop_args", ARR_BINOP_ARGS)
@pytest.mark.parametrize(
    # avoiding int operations
    "op",
    [
        op
        for op in numba.core.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
        if op
        not in (
            operator.imod,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ior,
            operator.ixor,
            operator.ipow,
        )
    ],
)
def test_float_arr_binary_inplace_op(binop_args, op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A, other):\n"
    func_text += "  A = A + 0\n"  # universal copy (.copy will not work on scalars)
    func_text += f"  A {op_str} other\n"
    func_text += "  return A\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A1, A2 = binop_args

    check_func(test_impl, (A1, A2), check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize("op", (operator.neg, operator.pos))
def test_float_arr_unary_op(op, memory_leak_check):
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(A):\n"
    func_text += f"  return {op_str} A\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    test_impl = loc_vars["test_impl"]

    A = pd.arrays.FloatingArray(
        np.array([1.0, 1.0, 1.1, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_sum(memory_leak_check):
    arr = pd.array(
        [float(i) if i % 2 else None for i in range(30)], dtype=pd.Float32Dtype()
    )

    def impl(arr):
        return arr.sum()

    check_func(impl, (arr,))


@pytest.mark.smoke
def test_float_arr_ops_fast(float_dtype, memory_leak_check):
    arr1 = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)
    arr2 = pd.array(
        [float(i + 30) if i % 3 else None for i in range(30)], dtype=float_dtype
    )

    def impl(a, b):
        return a + b

    check_func(impl, (arr1, 1.0))
    check_func(impl, (arr1, arr2))


@pytest.mark.slow
def test_float_arr_unique(memory_leak_check):
    arr = pd.array(
        [float(i) if i % 2 else None for i in range(30)], dtype=pd.Float32Dtype()
    )

    def impl(arr):
        return arr.unique()

    check_func(impl, (arr,), dist_test=False)


@pytest.mark.slow
def test_float_arr_getitem(nullable_float_values, float_dtype, memory_leak_check):
    arr = pd.array(nullable_float_values, dtype=float_dtype)

    if len(nullable_float_values) == 0:
        pytest.skip(reason="empty array")

    def impl(arr):
        return arr[0]

    check_func(
        impl,
        (arr,),
        py_output=arr[0] if arr[0] is not pd.NA else np.nan,
    )


@pytest.mark.slow
def test_float_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.Float32Dtype(),))
    check_func(impl, (pd.Float64Dtype(),))

    # constructors
    def impl2():
        return pd.Float32Dtype()

    check_func(impl2, ())

    def impl3():
        return pd.Float64Dtype()

    check_func(impl3, ())


@pytest.mark.slow
def test_float_arr_len(memory_leak_check):
    def test_impl(A):
        return len(A)

    A = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_shape(memory_leak_check):
    def test_impl(A):
        return A.shape

    A = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_dtype(nullable_float_values, float_dtype, memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (pd.array(nullable_float_values, dtype=float_dtype),))


@pytest.mark.slow
def test_float_arr_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    A = pd.arrays.FloatingArray(
        np.array([1.0, 1.0, 1.0, 2.0, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    check_func(test_impl, (A,))


@pytest.mark.slow
def test_float_arr_copy(nullable_float_values, float_dtype, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (pd.array(nullable_float_values, dtype=float_dtype),))


@pytest.mark.slow
def test_float_arr_constant_lowering(
    nullable_float_values, float_dtype, memory_leak_check
):
    arr = pd.array(nullable_float_values, dtype=float_dtype)

    def impl():
        return arr

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(arr), check_dtype=False
    )


@pytest.mark.slow
def test_float_arr_nbytes(memory_leak_check):
    def impl(A):
        return A.nbytes

    arr = pd.arrays.FloatingArray(
        np.array([1.0, -3.14, 2.0, 3.14, 10.0], np.float64),
        np.array([False, True, True, False, False]),
    )
    py_out = 40 + bodo.get_size()  # 1 extra byte for null_bit_map per rank
    check_func(impl, (arr,), py_output=py_out, only_1D=True)
    check_func(impl, (arr,), py_output=41, only_seq=True)


@pytest.mark.slow
def test_float_arr_series_conversion(float_dtype, memory_leak_check):
    def impl(A):
        return pd.Series(A)

    arr = pd.array([float(i) if i % 2 else None for i in range(30)], dtype=float_dtype)

    check_func(impl, (arr,))
