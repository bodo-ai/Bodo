# Copyright (C) 2019 Bodo Inc. All rights reserved.
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


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

    check_func(impl, (decimal_arr_value,), dist_test=False)


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
        ]
    )
    check_func(impl, (arr,), sort_output=True)


def test_unbox(decimal_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (decimal_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (decimal_arr_value,))


def test_len(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (decimal_arr_value,))


def test_shape(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (decimal_arr_value,))


def test_ndim(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (decimal_arr_value,))


def test_decimal_coerce(memory_leak_check):
    ts = Decimal("4.5")

    def f(df, ts):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": 1 + np.arange(6)})
    check_func(f, (df1, ts))


# TODO: Add memory_leak_check when bug is resolved.
def test_series_astype_str(decimal_arr_value):
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


@pytest.mark.parametrize(
    "decimal_value",
    [
        # long value to exercise both 64-bit slots
        Decimal("422222222.511133333444411"),
        # short value to test an empty 64-bit slot
        Decimal("4.5"),
    ],
)
# TODO: Add memory_leak_check when bug is resolved.
def test_decimal_constant_lowering(decimal_value):
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


def test_setitem_none_int(decimal_arr_value, memory_leak_check):
    def test_impl(A, i):
        A[i] = None
        return A

    i = 0
    check_func(
        test_impl, (decimal_arr_value.copy(), i), copy_input=True, dist_test=False
    )


def test_setitem_optional_int(decimal_arr_value, memory_leak_check):
    def test_impl(A, i, flag, val):
        if flag:
            x = None
        else:
            x = val
        A[i] = x
        return A

    check_func(
        test_impl,
        (decimal_arr_value.copy(), 1, False, Decimal("5.9")),
        copy_input=True,
        dist_test=False,
    )
    check_func(
        test_impl,
        (decimal_arr_value.copy(), 0, True, Decimal("5.9")),
        copy_input=True,
        dist_test=False,
    )


# TODO: fix memory leak and add memory_leak_check
def test_constant_lowering(decimal_arr_value):
    def impl():
        return decimal_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(decimal_arr_value), check_dtype=False
    )
