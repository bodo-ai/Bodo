# Copyright (C) 2019 Bodo Inc. All rights reserved.
from decimal import Decimal
import pandas as pd
import numpy as np
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


def test_unbox(decimal_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (decimal_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (decimal_arr_value,))


def test_len(decimal_arr_value):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (decimal_arr_value,))


def test_shape(decimal_arr_value):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (decimal_arr_value,))


def test_ndim(decimal_arr_value):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (decimal_arr_value,))


def test_decimal_coerce():
    ts = Decimal("4.5")

    def f(df, ts):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": 1 + np.arange(6)})
    check_func(f, (df1, ts))


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
def test_decimal_constant_lowering(decimal_value):
    def f():
        return decimal_value

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == decimal_value


def test_join(decimal_arr_value):
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
