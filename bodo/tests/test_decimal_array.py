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
    arr = pd.array(arr, dtype=pd.ArrowDtype(pa.decimal128(38, 12)))
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
    """test joining DataFrames with decimal data columns
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

    def test_impl4():
        return Decimal(" -1.135E  ")

    check_func(
        test_impl1, (), py_output=pa.scalar(Decimal("1.1"), pa.decimal128(38, 18))
    )
    check_func(test_impl2, (), py_output=pa.scalar(Decimal(), pa.decimal128(38, 18)))
    check_func(test_impl3, (), py_output=pa.scalar(Decimal(1), pa.decimal128(38, 18)))
    check_func(
        test_impl4, (), py_output=pa.scalar(Decimal("-1.135"), pa.decimal128(38, 18))
    )


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


def test_cast_decimal_to_decimal_array_loss_null_on_error(
    precision_scale_decimal_array, memory_leak_check
):
    """
    Test that when decimal scalars are truncated we correctly output NULL
    values if error on null is set.
    """

    def impl(arr):
        return bodo.libs.bodosql_array_kernels.numeric_to_decimal(arr, 4, 3, True)

    py_output = pd.array(
        ["1", "1.55", "1.56", None, None, None, None, None, None],
        dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
    )
    check_func(impl, (precision_scale_decimal_array,), py_output=py_output)


@pytest_mark_one_rank
def test_cast_decimal_to_decimal_array_error(
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


@pytest.mark.parametrize(
    "arg0, arg1, answer",
    [
        pytest.param(
            pa.scalar(Decimal("85.23"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("104.68"), pa.decimal128(5, 2)),
            id="scalar-scalar-same_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("543"), pa.decimal128(3, 0)),
            pa.scalar(Decimal("16.45"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("559.45"), pa.decimal128(6, 2)),
            id="scalar-scalar-smaller_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("1.23456"), pa.decimal128(6, 5)),
            pa.scalar(Decimal("9876543210"), pa.decimal128(10, 0)),
            pa.scalar(Decimal("9876543211.23456"), pa.decimal128(16, 5)),
            id="scalar-scalar-larger_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("-85.23"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("-65.78"), pa.decimal128(38, 2)),
            id="scalar-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789012345678901234567890"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("00.12"), pa.decimal128(4, 2)),
            pa.scalar(
                Decimal("123456789012345678901234567890.12"), pa.decimal128(38, 2)
            ),
            id="scalar-scalar-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("112112321123.4321123454321"), pa.decimal128(30, 18)),
            pa.scalar(Decimal("-12345.6789"), pa.decimal128(35, 4)),
            pa.scalar(Decimal("112112308777.7532123454321"), pa.decimal128(38, 18)),
            id="scalar-scalar-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("-5.00"), Decimal("4.56"), Decimal("9.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("-6.45"), None, Decimal("0.01"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("-11.45"), None, Decimal("10.00"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            id="array-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("994"), Decimal("456"), Decimal("-12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 0)),
            ),
            pd.array(
                [Decimal("9.45"), None, Decimal("0.01"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("1003.45"), None, Decimal("-11.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            id="array-array-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1.234"), Decimal("5.678"), Decimal("-9.101"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
            ),
            pd.array(
                [Decimal("9999.9"), None, Decimal("-9999.9"), Decimal("0")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 1)),
            ),
            pd.array(
                [Decimal("10001.134"), None, Decimal("-10009.001"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            id="array-array-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345678901234567890"),
                    Decimal("-1024"),
                    Decimal("1000000000000000000000000000000000001"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197531"),
                    None,
                    Decimal("-3000000000000000000000000000000000076"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("109877654220987765421"),
                    None,
                    Decimal("-2000000000000000000000000000000000075"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-array-same_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345678901234567"),
                    Decimal("-1024"),
                    Decimal("123456789123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197"),
                    None,
                    Decimal("-7654321987654321.987654321123456789"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("109877654220987764"),
                    None,
                    Decimal("115802467135802467.012345678876543211"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("987654.32109876543210"),
                    Decimal("3.1415926"),
                    Decimal("-123456.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 18)),
            ),
            pd.array(
                [
                    Decimal("98765432109876543210"),
                    None,
                    Decimal("-12345678901234567890"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 0)),
            ),
            pd.array(
                [
                    Decimal("98765432109877530864.32109876543210"),
                    None,
                    Decimal("-12345678901234691346.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("0.99"), pa.decimal128(4, 2)),
            pd.array(
                [Decimal("1.99"), Decimal("99.99"), Decimal("-99.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("2.98"), Decimal("100.98"), Decimal("-99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="scalar-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("15"), Decimal("-99"), Decimal("56"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(2, 0)),
            ),
            pa.scalar(Decimal("98.12"), pa.decimal128(4, 2)),
            pd.array(
                [Decimal("113.12"), Decimal("-0.88"), Decimal("154.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1.2345"), Decimal("-9.9999"), Decimal("5.4321"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 4)),
            ),
            pa.scalar(Decimal("999.5"), pa.decimal128(4, 1)),
            pd.array(
                [Decimal("1000.7345"), Decimal("989.5001"), Decimal("1004.9321"), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            id="array-scalar-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("9" * 36), Decimal("-" + "9" * 36), Decimal("5" * 36), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pa.scalar(Decimal("9" * 36), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("1" + "9" * 35 + "8"),
                    Decimal("0"),
                    Decimal("1" + "5" * 35 + "4"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789"), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("9" * 20 + ".25"),
                    Decimal("9" * 25 + ".5"),
                    Decimal("9" * 30 + ".75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                [
                    Decimal("100000000000123456788.25"),
                    Decimal("10000000000000000123456788.5"),
                    Decimal("1000000000000000000000123456788.75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("987654321.123456789"), pa.decimal128(38, 18)),
            pd.array(
                [
                    Decimal("9" * 10 + ".3"),
                    Decimal("9" * 14 + ".4"),
                    Decimal("9" * 18 + ".5"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    Decimal("10987654320.423456789"),
                    Decimal("100000987654320.523456789"),
                    Decimal("1000000000987654320.623456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="scalar-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("1" + "5" * 36), pa.decimal128(38, 0)),
            pa.scalar(Decimal("-1" + "3" * 36), pa.decimal128(38, 1)),
            pa.scalar(Decimal("2" * 36), pa.decimal128(38, 1)),
            id="scalar-scalar-unsafe_rescale_edgecase",
        ),
        pytest.param(
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            None,
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="null-array",
        ),
    ],
)
def test_decimal_addition(arg0, arg1, answer, memory_leak_check):
    """Test adding decimals"""

    def impl(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.add_numeric(arg0, arg1)

    check_func(impl, (arg0, arg1), py_output=answer)


@pytest.mark.parametrize(
    "arg0, arg1",
    [
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-same_scale-overflow_on_add",
        ),
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            id="scalar-scalar-larger_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
        pytest.param(
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="array-array-same_scale-overflow_on_add",
        ),
        pytest.param(
            pd.array(
                [Decimal("800000000000000001"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 20)),
            ),
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
    ],
)
def test_decimal_addition_error(arg0, arg1):
    """Test adding decimals in ways that will cause error"""

    def impl(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.add_numeric(arg0, arg1)

    with pytest.raises(ValueError, match="Number out of representable range"):
        check_func(impl, (arg0, arg1), py_output=-1)


@pytest.mark.parametrize(
    "arg0, arg1, answer",
    [
        pytest.param(
            pa.scalar(Decimal("104.68"), pa.decimal128(5, 2)),
            pa.scalar(Decimal("85.23"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(4, 2)),
            id="scalar-scalar-smaller_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("559.45"), pa.decimal128(6, 2)),
            pa.scalar(Decimal("543"), pa.decimal128(6, 2)),
            pa.scalar(Decimal("16.45"), pa.decimal128(6, 2)),
            id="scalar-scalar-same_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("1.23456"), pa.decimal128(6, 5)),
            pa.scalar(Decimal("9876543211.23456"), pa.decimal128(16, 5)),
            pa.scalar(Decimal("-9876543210"), pa.decimal128(10, 0)),
            id="scalar-scalar-larger_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("-65.78"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("-85.23"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(38, 2)),
            id="scalar-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(
                Decimal("123456789012345678901234567890.12"), pa.decimal128(38, 2)
            ),
            pa.scalar(Decimal("123456789012345678901234567890"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("00.12"), pa.decimal128(4, 2)),
            id="scalar-scalar-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("112112308777.7532123454321"), pa.decimal128(38, 18)),
            pa.scalar(Decimal("112112321123.4321123454321"), pa.decimal128(30, 18)),
            pa.scalar(Decimal("-12345.6789"), pa.decimal128(38, 18)),
            id="scalar-scalar-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("-11.45"), None, Decimal("10.00"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("-5.00"), Decimal("4.56"), Decimal("9.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("-6.45"), None, Decimal("0.01"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            id="array-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1003.45"), None, Decimal("-11.99"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            pd.array(
                [Decimal("994"), Decimal("456"), Decimal("-12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 0)),
            ),
            pd.array(
                [Decimal("9.45"), None, Decimal("0.01"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            id="array-array-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("10001.134"), None, Decimal("-10009.001"), Decimal("0")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            pd.array(
                [Decimal("1.234"), Decimal("5.678"), Decimal("-9.101"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
            ),
            pd.array(
                [Decimal("9999.9"), None, Decimal("-9999.9"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            id="array-array-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("109877654220987765421"),
                    None,
                    Decimal("-2000000000000000000000000000000000075"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("12345678901234567890"),
                    Decimal("-1024"),
                    Decimal("1000000000000000000000000000000000001"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197531"),
                    None,
                    Decimal("-3000000000000000000000000000000000076"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-array-same_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("109877654220987764"),
                    None,
                    Decimal("115802467135802467.012345678876543211"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("12345678901234567"),
                    None,
                    Decimal("123456789123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197"),
                    None,
                    Decimal("-7654321987654321.987654321123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("98765432109877530864.32109876543210"),
                    None,
                    Decimal("-12345678901234691346.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("987654.32109876543210"),
                    Decimal("3.1415926"),
                    Decimal("-123456.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 18)),
            ),
            pd.array(
                [
                    Decimal("98765432109876543210"),
                    None,
                    Decimal("-12345678901234567890"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("2.98"), Decimal("100.98"), Decimal("-99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            pa.scalar(Decimal("0.99"), pa.decimal128(5, 2)),
            pd.array(
                [Decimal("1.99"), Decimal("99.99"), Decimal("-99.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("113.12"), Decimal("-0.88"), Decimal("154.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            pa.scalar(Decimal("99"), pa.decimal128(2, 0)),
            pd.array(
                [Decimal("14.12"), Decimal("-99.88"), Decimal("55.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1000.7345"), Decimal("989.5001"), Decimal("1004.9321"), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            pa.scalar(Decimal("1.2345"), pa.decimal128(5, 4)),
            pd.array(
                [Decimal("999.5"), Decimal("988.2656"), Decimal("1003.6976"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            id="array-scalar-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("9" * 36), Decimal("-" + "9" * 36), Decimal("5" * 36), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pa.scalar(Decimal("-" + "9" * 36), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("1" + "9" * 35 + "8"),
                    Decimal("0"),
                    Decimal("1" + "5" * 35 + "4"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789"), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("-" + "9" * 20 + ".25"),
                    Decimal("-" + "9" * 25 + ".5"),
                    Decimal("-" + "9" * 30 + ".75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                [
                    Decimal("100000000000123456788.25"),
                    Decimal("10000000000000000123456788.5"),
                    Decimal("1000000000000000000000123456788.75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("987654321.123456789"), pa.decimal128(38, 18)),
            pd.array(
                [
                    Decimal("-" + "9" * 10 + ".3"),
                    Decimal("-" + "9" * 14 + ".4"),
                    Decimal("-" + "9" * 18 + ".5"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    Decimal("10987654320.423456789"),
                    Decimal("100000987654320.523456789"),
                    Decimal("1000000000987654320.623456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="scalar-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("1" + "5" * 36), pa.decimal128(38, 0)),
            pa.scalar(Decimal("1" + "3" * 36), pa.decimal128(38, 1)),
            pa.scalar(Decimal("2" * 36), pa.decimal128(38, 1)),
            id="scalar-scalar-unsafe_rescale_edgecase",
        ),
        pytest.param(
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            None,
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="null-array",
        ),
    ],
)
def test_decimal_subtraction(arg0, arg1, answer, memory_leak_check):
    """Test subtracting decimals"""

    def impl(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.subtract_numeric(arg0, arg1)

    check_func(impl, (arg0, arg1), py_output=answer)


@pytest.mark.parametrize(
    "arg0, arg1",
    [
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("-87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-same_scale-overflow_on_add",
        ),
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("-800000000000000001"), pa.decimal128(38, 20)),
            id="scalar-scalar-larger_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            pa.scalar(Decimal("-87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
        pytest.param(
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            pd.array(
                [Decimal("-87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="array-array-same_scale-overflow_on_add",
        ),
        pytest.param(
            pd.array(
                [Decimal("800000000000000001"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 20)),
            ),
            pd.array(
                [Decimal("-87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
    ],
)
def test_decimal_subtraction_error(arg0, arg1):
    """Test subtracting decimals in ways that will cause error"""

    def impl(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.subtract_numeric(arg0, arg1)

    with pytest.raises(ValueError, match="Number out of representable range"):
        check_func(impl, (arg0, arg1), py_output=-1)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
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
            ),
            pd.array(
                [
                    "1",
                    "2.4025",
                    "2.4336",
                    "111.5136",
                    "1001000.25",
                    None,
                    None,
                    "100082016.81",
                    "130.1881",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            id="array_array",
        ),
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
            ),
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "2.0",
                    "3.1",
                    "3.12",
                    "21.12",
                    "2001.0",
                    None,
                    None,
                    "20008.2",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
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
            ),
            pd.array(
                [
                    "2.0",
                    "3.1",
                    "3.12",
                    "21.12",
                    "2001.0",
                    None,
                    None,
                    "20008.2",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="scalar_array",
        ),
        pytest.param(
            pa.scalar(Decimal("2.3"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("4.5"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("10.3500"), pa.decimal128(8, 4)),
            id="scalar_scalar",
        ),
    ],
)
def test_decimal_array_multiplication(arg1, arg2, expected, memory_leak_check):
    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.multiply_decimals(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


def test_decimal_array_multiplication_overflow_handling():
    """
    Test that an appropriate error is raised when one or more
    decimal multiplications overflow.
    """

    @bodo.jit(distributed=["arr1", "arr2"])
    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.multiply_decimals(arr1, arr2)

    arr1 = pd.array(
        [
            "1",
            "99.9999999",  # This will overflow
            None,
            None,
            "99.9999999",
            "1",  # This won't overflow
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )
    arr2 = pd.array(
        [
            "1",
            "99.9999999",
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)

    arr1 = Decimal("99999999999999.9999999")
    arr2 = Decimal("99999999999999.9999999")
    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
            pd.array(
                [
                    "2.1",
                    "0.005",
                    "2.4336",
                    "0.000001",
                    "1001000.25",
                    "1.4",
                    None,
                    "100082016.81",
                    "130.1881",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    "0.47619048",
                    "310.00000000",
                    "0.64102564",
                    "10560000.00000000",
                    "0.00099950",
                    None,
                    None,
                    "0.00009996",
                    "-0.08764242",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_array",
        ),
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
            ),
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "0.50000000",
                    "0.77500000",
                    "0.78000000",
                    "5.28000000",
                    "500.25000000",
                    None,
                    None,
                    "5002.05000000",
                    "-5.70500000",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
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
            ),
            pd.array(
                [
                    "2.00000000",
                    "1.29032258",
                    "1.28205128",
                    "0.18939394",
                    "0.00199900",
                    None,
                    None,
                    "0.00019992",
                    "-0.17528484",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="scalar_array",
        ),
        pytest.param(
            pa.scalar(Decimal("-2.12"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("-1.345"), pa.decimal128(5, 3)),
            pa.scalar(Decimal("1.57620818"), pa.decimal128(13, 8)),
            id="scalar_scalar",
        ),
    ],
)
def test_decimal_array_division(arg1, arg2, expected, memory_leak_check):
    """Test decimal division"""

    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.divide_decimals(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest_mark_one_rank
def test_decimal_array_division_error_handling():
    """
    Test that an appropriate error is raised when there is division by zero or overflow
    """

    @bodo.jit(distributed=["arr1", "arr2"])
    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.divide_decimals(arr1, arr2)

    arr1 = pd.array(
        [
            "1",
            "9999999999999999999.99",  # This will overflow
            None,
            None,
            "9999999.99",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
    )
    arr2 = pd.array(
        [
            "1",
            "0.0000000000000001",
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)

    arr1 = pd.array(
        [
            "1",
            "9999999999999999999.99",
            None,
            None,
            "9999999.99",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
    )
    arr2 = pd.array(
        [
            "1",
            "0.0",  # Division by zero
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(RuntimeError, match="Decimal division by zero error"):
        out = impl(arr1, arr2)
        print(out)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    2,
                    2.55,
                    2.56,
                    11.56,
                    1001.5,
                    None,
                    None,
                    10005.1,
                    -10.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
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
            ),
            pd.array(
                [
                    2,
                    2.55,
                    2.56,
                    11.56,
                    1001.5,
                    None,
                    None,
                    10005.1,
                    -10.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1.0),
            float(16.50),
            id="scalar-decimal_first",
        ),
        pytest.param(
            float(1.0),
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(16.50),
            id="scalar-float_first",
        ),
        pytest.param(
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            float(1.0),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_addition(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float addition works correctly.
    """

    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.add_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    0,
                    0.55,
                    0.56,
                    9.56,
                    999.5,
                    None,
                    None,
                    10003.1,
                    -12.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
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
            ),
            pd.array(
                [
                    0,
                    -0.55,
                    -0.56,
                    -9.56,
                    -999.5,
                    None,
                    None,
                    -10003.1,
                    12.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1.0),
            float(14.50),
            id="scalar-decimal_first",
        ),
        pytest.param(
            float(1.0),
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(-14.50),
            id="scalar-float_first",
        ),
        pytest.param(
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    -1.5,
                    -99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            float(1.0),
            pd.array(
                [
                    1.5,
                    99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    1.5,
                    99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    -1.5,
                    -99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_subtraction(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float subtraction works correctly.
    """

    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.subtract_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
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
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1.0),
            float(15.50),
            id="scalar-decimal_first",
        ),
        pytest.param(
            float(1.0),
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(15.50),
            id="scalar-float_first",
        ),
        pytest.param(
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_multiplication(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float multiplication works correctly.
    """

    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.multiply_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
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
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
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
            ),
            pd.array(
                [
                    1 / 1,
                    1 / 1.55,
                    1 / 1.56,
                    1 / 10.56,
                    1 / 1000.5,
                    None,
                    None,
                    1 / 10004.1,
                    1 / -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1.0),
            float(15.50),
            id="scalar-decimal_first",
        ),
        pytest.param(
            float(1.0),
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1 / 15.50),
            id="scalar-float_first",
        ),
        pytest.param(
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1 / 2.5,
                    1 / 100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            float(1.0),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1 / 2.5,
                    1 / 100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_division(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float division works correctly.
    """

    def impl(arr1, arr2):
        return bodo.libs.bodosql_array_kernels.divide_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


def test_str_to_decimal_scalar(memory_leak_check):
    """
    Test converting a string scalar to decimal.
    """

    def impl(s):
        return bodo.libs.bodosql_array_kernels.string_to_decimal(s, 4, 2, True)

    check_func(
        impl, ("1.12",), py_output=pa.scalar(Decimal("1.12"), pa.decimal128(4, 2))
    )
    check_func(impl, ("E",), py_output=pa.scalar(Decimal("0"), pa.decimal128(4, 2)))
    check_func(impl, ("-E",), py_output=pa.scalar(Decimal("-0"), pa.decimal128(4, 2)))
    check_func(impl, ("+E",), py_output=pa.scalar(Decimal("+0"), pa.decimal128(4, 2)))
    check_func(
        impl, ("1.1E-1",), py_output=pa.scalar(Decimal("0.11"), pa.decimal128(4, 2))
    )
    check_func(
        impl, ("-1.1E",), py_output=pa.scalar(Decimal("-1.1"), pa.decimal128(4, 2))
    )
    # Check rounding scale
    check_func(
        impl, ("1.125",), py_output=pa.scalar(Decimal("1.13"), pa.decimal128(4, 2))
    )
    # Check for too large a leading digit
    check_func(impl, ("-100",), py_output=None)
    # Check for None -> None
    check_func(impl, (None,), py_output=None)


def test_str_to_decimal_array(memory_leak_check):
    """
    Test converting a string array to decimal.
    """

    def impl(arr):
        return bodo.libs.bodosql_array_kernels.string_to_decimal(arr, 4, 2, True)

    arr = pd.array(
        [
            "1.12",
            "E",
            "-E",
            "+E",
            "1.1E-1",
            "-1.1E",
            "1.125",
            "-100",
            None,
        ]
        * 2
    )
    py_output = pd.array(
        [
            "1.12",
            "0",
            "0",
            "0",
            "0.11",
            "-1.1",
            "1.13",
            None,
            None,
        ]
        * 2,
        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
    )
    check_func(impl, (arr,), py_output=py_output)


def test_str_to_decimal_scalar_edge_case(memory_leak_check):
    """
    Tests an edge case where using the full precision + scale doesn't
    fit in a decimal 128, but the scale can be rounded.
    """

    def impl(s):
        return bodo.libs.bodosql_array_kernels.string_to_decimal(s, 38, 4, True)

    leading = "9" + "0" * 33
    value = f"{leading}.12345"
    check_func(
        impl,
        (value,),
        py_output=pa.scalar(Decimal(f"{leading}.1235"), pa.decimal128(38, 4)),
    )


def test_str_to_decimal_error():
    """
    Test that with null_on_error=False strings that don't fit or don't parse raise
    an error.
    """

    @bodo.jit
    def impl(s):
        return bodo.libs.bodosql_array_kernels.string_to_decimal(s, 4, 2, False)

    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl("-100")
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl("1 0")
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl(pd.array(["-100"] * 5))
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl(pd.array(["1 0"] * 5))
