# Copyright (C) 2022 Bodo Inc. All rights reserved.
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
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
    assert val_ret == decimal_value


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

    check_func(test_impl1, ())
    check_func(test_impl2, ())
    check_func(test_impl3, ())


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
