# Copyright (C) 2022 Bodo Inc. All rights reserved.
import operator
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
from bodo.tests.utils import check_func, pytest_mark_one_rank
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        pytest.param(
            np.array(
                [
                    Decimal("1.6"),
                    None,
                    Decimal("-0.222"),
                    Decimal("1111.316"),
                    Decimal("1234.00046"),
                    Decimal("5.1"),
                    Decimal("-11131.0056"),
                    Decimal("0.0"),
                ]
                * 10
            ),
            marks=pytest.mark.slow,
        ),
        np.array(
            [
                Decimal("1.6"),
                None,
                Decimal("-0.222"),
                Decimal("1111.316"),
                Decimal("1234.00046"),
                Decimal("5.1"),
                Decimal("-11131.0056"),
                Decimal("0.0"),
            ]
        ),
    ]
)
def decimal_arr_value(request):
    return request.param


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = np.array(
        [
            Decimal("1.6"),
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
        * 20
    )

    check_func(impl, (A,))


def test_np_repeat(decimal_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (decimal_arr_value,))


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = np.array(
        [
            Decimal("1.6"),
            Decimal("-0.222"),
            Decimal("5.1"),
            Decimal("1111.316"),
            Decimal("-0.2220001"),
            Decimal("-0.2220"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
            Decimal("5.11"),
            Decimal("0.00"),
            Decimal("0.01"),
            Decimal("0.03"),
            Decimal("0.113"),
            Decimal("1.113"),
        ]
    )
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_unbox(decimal_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (decimal_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (decimal_arr_value,))


@pytest.mark.slow
def test_len(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_shape(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_dtype(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_ndim(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_decimal_coerce(memory_leak_check):
    ts = Decimal("4.5")

    def f(df, ts):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": 1 + np.arange(6)})
    check_func(f, (df1, ts))


def test_series_astype_str(decimal_arr_value, memory_leak_check):
    """test decimal conversion to string.
    Using a checksum for checking output since Bodo's output can have extra 0 digits
    """

    def test_impl(A):
        S2 = A.astype(str).values
        s = 0.0
        for i in bodo.prange(len(S2)):
            val = 0
            if not (
                bodo.libs.array_kernels.isna(S2, i) or S2[i] == "None" or S2[i] == "nan"
            ):
                val = float(S2[i])
            s += val
        return s

    S = pd.Series(decimal_arr_value)
    check_func(test_impl, (S,))


@pytest.mark.slow
@pytest.mark.parametrize(
    "decimal_value",
    [
        # long value to exercise both 64-bit slots
        Decimal("422222222.511133333444411"),
        # short value to test an empty 64-bit slot
        Decimal("4.5"),
    ],
)
def test_decimal_constant_lowering(decimal_value, memory_leak_check):
    def f():
        return decimal_value

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == pa.scalar(decimal_value, pa.decimal128(38, 18))


def test_join(decimal_arr_value, memory_leak_check):
    """test joining dataframes with decimal data columns
    TODO: add decimal array to regular df tests and remove this
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on="A")

    # double the size of the input array to avoid issues on 3 processes
    decimal_arr_value = np.concatenate((decimal_arr_value, decimal_arr_value))
    n = len(decimal_arr_value)
    df1 = pd.DataFrame({"A": np.arange(n), "B": decimal_arr_value})
    df2 = pd.DataFrame({"A": np.arange(n) + 3, "C": decimal_arr_value})
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_constructor(memory_leak_check):
    def test_impl1():
        return Decimal("1.1")

    def test_impl2():
        return Decimal()

    def test_impl3():
        return Decimal(1)

    check_func(
        test_impl1, (), py_output=pa.scalar(Decimal("1.1"), pa.decimal128(38, 18))
    )
    check_func(test_impl2, (), py_output=pa.scalar(Decimal(), pa.decimal128(38, 18)))
    check_func(test_impl3, (), py_output=pa.scalar(Decimal(1), pa.decimal128(38, 18)))


# TODO: fix memory leak and add memory_leak_check
@pytest.mark.slow
def test_constant_lowering(decimal_arr_value):
    def impl():
        return decimal_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(decimal_arr_value), check_dtype=False
    )


@pytest.mark.slow
def test_constructor_error(memory_leak_check):
    """Test that an invalid constructor throws a BodoError"""

    def impl():
        return Decimal([1.1, 2.2, 3.2])

    with pytest.raises(BodoError, match=r"decimal.Decimal\(\) value type must be"):
        bodo.jit(impl)()


def test_decimal_ops(memory_leak_check):
    def test_impl_eq(d1, d2):
        return d1 == d2

    def test_impl_ne(d1, d2):
        return d1 != d2

    def test_impl_gt(d1, d2):
        return d1 > d2

    def test_impl_ge(d1, d2):
        return d1 >= d2

    def test_impl_lt(d1, d2):
        return d1 < d2

    def test_impl_le(d1, d2):
        return d1 <= d2

    test_funcs = [
        test_impl_eq,
        test_impl_ne,
        test_impl_gt,
        test_impl_ge,
        test_impl_lt,
        test_impl_le,
    ]

    d1 = Decimal("-1.1")
    d2 = Decimal("100.2")

    for func in test_funcs:
        check_func(func, (d1, d1))
        check_func(func, (d1, d2))
        check_func(func, (d2, d1))


@pytest.mark.smoke
def test_setitem_int(decimal_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    val = decimal_arr_value[0]
    check_func(test_impl, (decimal_arr_value, val))


@pytest.mark.smoke
def test_setitem_arr(decimal_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(decimal_arr_value), 11)
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(11)])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = np.random.ranf(len(decimal_arr_value)) < 0.2
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(idx.sum())])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = slice(1, 4)
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(3)])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )


@pytest.mark.slow
def test_decimal_arr_nbytes(memory_leak_check):
    """Test DecimalArrayType nbytes"""

    def impl(A):
        return A.nbytes

    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
    )
    py_out = 128 + bodo.get_size()  # 1 extra byte for null_bit_map per rank
    check_func(impl, (arr,), py_output=py_out, only_1D=True, only_1DVar=True)
    check_func(impl, (arr,), py_output=129, only_seq=True)


@pytest.mark.parametrize(
    "value",
    [
        # int32 array
        pd.array([1, -1, 3, 4, -2, 0, None, 4], "Int32"),
        # int64 array
        pd.array([5, -1, 0, None, -2, 10, None, 12], "Int64"),
        # numpy array
        np.array([5, -21131, 0, 7, -2, 10, 12340, 12]),
        # float32 array
        pd.array([5.1, -1.1, 0.54, None, -2.1, 101.1, None, 1.234], "Float32"),
        # float64 array
        pd.array(
            [1.111, -1.12, 1000000.54, None, -2000.1, -101.1, None, 1.234], "Float64"
        ),
        # decimal array
        np.array(
            [
                Decimal("1.62"),
                Decimal("-1.222"),
                Decimal("1.316"),
                Decimal("-4.00046"),
                Decimal("5.14"),
                None,
                Decimal("-131.0056"),
                Decimal("0.0"),
            ]
        ),
        # int32 scalar
        np.int32(1),
        # int64 scalar
        -1,
        # float32 scalar
        np.float32(1.4),
        # float64 scalar
        -0.4,
        # decimal scalar
        Decimal("1.45"),
    ],
)
def test_decimal_comparison(value, memory_leak_check):
    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
    )
    pa_val = (
        pa.scalar(value)
        if bodo.utils.typing.is_scalar_type(bodo.typeof(value))
        else pa.array(value)
    )

    for op, pa_op in (
        (operator.gt, pa.compute.greater),
        (operator.ge, pa.compute.greater_equal),
        (operator.eq, pa.compute.equal),
        (operator.ne, pa.compute.not_equal),
        (operator.lt, pa.compute.less),
        (operator.le, pa.compute.less_equal),
    ):

        def impl(a, b):
            return op(a, b)

        py_output = pd.array(pa_op(pa.array(arr), pa_val), pd.BooleanDtype())
        check_func(impl, (arr, value), py_output=py_output)
        py_output = pd.array(pa_op(pa_val, pa.array(arr)), pd.BooleanDtype())
        check_func(impl, (value, arr), py_output=py_output)


def test_decimal_comparison_error_checking():
    """Make sure decimal comparison with invalid type raises an error"""
    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
        ]
    )
    other = np.array(["1.2", "3.1", None], object)

    def impl(a, b):
        return a < b

    with pytest.raises(BodoError, match=r"Invalid decimal comparison with"):
        bodo.jit(impl)(arr, other)


@pytest_mark_one_rank
def test_decimal_scalar_int_cast(memory_leak_check):
    """Test casting decimal scalars to integers"""

    def impl(a, b, flag):
        if flag:
            c = a
        else:
            c = b
        return c

    # cast to decimal type since wider
    a = pa.scalar(-123, pa.decimal128(20, 0))
    b = np.int32(-3)
    check_func(impl, (a, b, True), py_output=a, only_seq=True)
    check_func(
        impl,
        (a, b, False),
        py_output=pa.scalar(int(b), pa.decimal128(20, 0)),
        only_seq=True,
    )

    # cast to int type since wider
    a = pa.scalar(-123, pa.decimal128(3, 0))
    b = -3
    check_func(impl, (a, b, True), py_output=a.cast(pa.int64()).as_py(), only_seq=True)
    check_func(impl, (a, b, False), py_output=b, only_seq=True)


@pytest_mark_one_rank
def test_setitem_cast_decimal(memory_leak_check):
    """Test automatic cast of decimal to int/float in array setitem"""

    def test_impl(A, val):
        A[2] = val
        return A

    # Int case
    arr = pd.array([1, 2, 3, 4, 5], "Int64")
    val = Decimal("123")
    # Pandas doesn't support Decimal setitem yet so create output manually
    out = arr.copy()
    out[2] = int(val)
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_extension_array_equal(bodo_func(arr.copy(), val), out)

    # Float case
    arr = pd.array([1.2, 2.3, 3.1, 4.1, 5.4], "Float32")
    val = Decimal("12.3")
    # Pandas doesn't support Decimal setitem yet so create output manually
    out = arr.copy()
    out[2] = float(val)
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_extension_array_equal(bodo_func(arr.copy(), val), out)


@pytest.fixture(
    params=[
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            )
        ),
    ]
)
def precision_scale_decimal_array(request):
    return request.param


def test_box_arrow_array_precision_scale(precision_scale_decimal_array):
    """
    Test that we can box/unbox an arrow decimal array without 38, 18 precision and scale.
    """

    def impl(arr):
        return arr

    check_func(
        impl, (precision_scale_decimal_array,), py_output=precision_scale_decimal_array
    )


def test_cast_decimal_to_decimal_scalar(
    precision_scale_decimal_array, memory_leak_check
):
    # This takes the unsafe cast route because the leading digits increase.
    def impl1(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = bodo.libs.bodosql_array_kernels.numeric_to_decimal(
                    arr[i], 28, 3, False
                )

        return out_arr

    # This takes the safe cast route because the leading digits decrease, so we
    # need to ensure the data fits.
    def impl2(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 15, 1)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = bodo.libs.bodosql_array_kernels.numeric_to_decimal(
                    arr[i], 15, 1, False
                )
        return out_arr

    py_output1 = pd.array(
        ["1", "1.55", "1.56", "10.56", "1000.5", None, None, "10004.1", "-11.41"],
        dtype=pd.ArrowDtype(pa.decimal128(28, 3)),
    )
    check_func(impl1, (precision_scale_decimal_array,), py_output=py_output1)
    py_output2 = pd.array(
        ["1", "1.6", "1.6", "10.6", "1000.5", None, None, "10004.1", "-11.4"],
        dtype=pd.ArrowDtype(pa.decimal128(15, 1)),
    )
    check_func(impl2, (precision_scale_decimal_array,), py_output=py_output2)


def test_cast_decimal_to_decimal_scalar_loss_null_on_error(
    precision_scale_decimal_array, memory_leak_check
):
    """
    Test that when decimal scalars are truncated we correctly output NULL
    values if error on null is set.
    """

    def impl(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = bodo.libs.bodosql_array_kernels.numeric_to_decimal(
                    arr[i], 4, 3, True
                )

        return out_arr

    py_output = pd.array(
        ["1", "1.55", "1.56", None, None, None, None, None, None],
        dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
    )
    check_func(impl, (precision_scale_decimal_array,), py_output=py_output)


@pytest_mark_one_rank
def test_cast_decimal_to_decimal_scalar_error(precision_scale_decimal_array):
    """
    Test that decimals that don't fit raise an error when numeric_to_decimal
    doesn't have null on error set.
    """

    @bodo.jit
    def impl(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = bodo.libs.bodosql_array_kernels.numeric_to_decimal(
                    arr[i], 4, 3, False
                )

        return out_arr

    with pytest.raises(Exception, match=r"Number out of representable range"):
        impl(precision_scale_decimal_array)


def test_cast_decimal_to_decimal_array(
    precision_scale_decimal_array, memory_leak_check
):
    # This takes the unsafe cast route because the leading digits increase.
    def impl1(arr):
        return bodo.libs.bodosql_array_kernels.numeric_to_decimal(arr, 28, 3, False)

    # This takes the safe cast route because the leading digits decrease, so we
    # need to ensure the data fits.
    def impl2(arr):
        return bodo.libs.bodosql_array_kernels.numeric_to_decimal(arr, 15, 1, False)

    py_output1 = pd.array(
        ["1", "1.55", "1.56", "10.56", "1000.5", None, None, "10004.1", "-11.41"],
        dtype=pd.ArrowDtype(pa.decimal128(28, 3)),
    )
    check_func(impl1, (precision_scale_decimal_array,), py_output=py_output1)
    py_output2 = pd.array(
        ["1", "1.6", "1.6", "10.6", "1000.5", None, None, "10004.1", "-11.4"],
        dtype=pd.ArrowDtype(pa.decimal128(15, 1)),
    )
    check_func(impl2, (precision_scale_decimal_array,), py_output=py_output2)


@pytest_mark_one_rank
def test_cast_decimal_to_decimal_array_loss_null_on_error(
    precision_scale_decimal_array,
):
    """
    Test that decimals that don't fit raise an error when numeric_to_decimal
    doesn't have null on error set.
    """

    @bodo.jit
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.numeric_to_decimal(arr, 4, 3, False)

    with pytest.raises(Exception, match=r"Number out of representable range"):
        impl(precision_scale_decimal_array)
