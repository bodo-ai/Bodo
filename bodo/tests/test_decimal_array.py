# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
from decimal import Decimal
import pandas as pd
import numpy as np
import pytest

import numba
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


def test_unbox(decimal_arr_value):
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
