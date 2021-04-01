# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's string array data type
"""
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        # unicode
        pytest.param(
            pd.array(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    pd.NA,
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                    "한국,가,고싶다ㅠ",
                    "🢇🄐,🏈𠆶💑😅",
                ],
            ),
            marks=pytest.mark.slow,
        ),
        # ASCII array
        pd.array(["AB", "", "ABC", pd.NA, "abcd"]),
    ]
)
def str_arr_value(request):
    return request.param


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = pd.array(["AB", "", "ABC", "abcd", "PQ", "DDE"] * 8)

    check_func(impl, (A,))


def test_np_repeat(str_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (str_arr_value,))


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = pd.array(["AB", "", "ABC", "abcd", "ab", "AB"])

    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_unbox(str_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (str_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (str_arr_value,))


# TODO: fix memory leak and add memory_leak_check
@pytest.mark.slow
def test_constant_lowering(str_arr_value):
    def impl():
        return str_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(str_arr_value), check_dtype=False
    )


@pytest.mark.slow
def test_string_dtype(memory_leak_check):
    # unbox and box
    def impl(d):
        return d

    check_func(impl, (pd.StringDtype(),))

    # constructor
    def impl2():
        return pd.StringDtype()

    check_func(impl2, ())


@pytest.mark.smoke
def test_getitem_int(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a integer ind
    """

    def test_impl(A, i):
        return A[i]

    check_func(test_impl, (str_arr_value, 4))


def test_getitem_int_arr(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with an integer list ind
    """

    def test_impl(A, ind):
        return A[ind]

    ind = np.array([0, 2, 3])
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    check_func(
        test_impl,
        (str_arr_value, list(ind)),
        check_dtype=False,
        dist_test=False,
    )


def test_getitem_bool(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a boolean ind
    """

    def test_impl(A, ind):
        return A[ind]

    np.random.seed(0)
    ind = np.random.ranf(len(str_arr_value)) < 0.5
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    check_func(
        test_impl,
        (str_arr_value, list(ind)),
        check_dtype=False,
        dist_test=False,
    )


def test_getitem_slice(str_arr_value, memory_leak_check):
    """
    Test operator.getitem on String array with a slice ind
    """

    def test_impl(A, ind):
        return A[ind]

    ind = slice(1, 4)
    # Pandas outputs differs (object vs string)
    # TODO [BE-483]: Fix distributed
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)
    # Test with a slice with non step > 1
    # TODO [BE-483]: Fix distributed
    ind = slice(1, 4, 2)
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)


@pytest.mark.smoke
def test_setitem_int(memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    A = pd.array(["AB", "", "한국", pd.NA, "abcd"])
    idx = 2
    val = "국한"  # same size as element 2 but different value
    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        pd.array(bodo_func(A.copy(), idx, val), "string"), test_impl(A.copy(), idx, val)
    )


@pytest.mark.slow
def test_setitem_none_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n - 1)
        for i in range(n):
            if i == idx:
                A[i] = None
                continue
            A[i] = "A"
        return A

    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        pd.array(bodo_func(8, 1), "string"), pd.array(["A", None] + ["A"] * 6, "string")
    )


@pytest.mark.slow
def test_setitem_optional_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n - 1)
        for i in range(n):
            if i == idx:
                value = None
            else:
                value = "A"
            A[i] = value
        return A

    bodo_func = bodo.jit(test_impl)
    pd.util.testing.assert_extension_array_equal(
        pd.array(bodo_func(8, 1), "string"), pd.array(["A", None] + ["A"] * 6, "string")
    )


@pytest.mark.slow
def test_dtype(memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (pd.array(["AA", "B"] * 4),))


@pytest.mark.slow
def test_ndim(memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (pd.array(["AA", "B"] * 4),))


def test_astype_str(memory_leak_check):
    def test_impl(A):
        return A.astype(str)

    check_func(test_impl, (pd.array(["AA", "B"] * 4),))
