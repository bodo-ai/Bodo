# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's string array data type
"""
import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


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
        pd.array(["AB", "", "ABC", pd.NA, "C", "D", "abcd", "ABCD"]),
    ]
)
def str_arr_value(request):
    return request.param


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = pd.array(["AB", "", "ABC", "abcd", "PQ", "DDE"] * 8)

    check_func(impl, (A,))


def test_hash(str_arr_value, memory_leak_check):
    def impl(S):
        return S.map(lambda x: None if pd.isna(x) else hash(x))

    # check_dtype=False because None converts the output to Float in Pandas
    # dist_test = False because the randomness causes different inputs on each core.
    check_func(impl, (pd.Series(str_arr_value),), check_dtype=False, dist_test=False)


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
    check_func(test_impl, (str_arr_value, -1))


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
    # Check nullable with pd.array and insert NA values
    ind = pd.array(ind)
    ind[1] = None
    ind[3] = None
    check_func(test_impl, (str_arr_value, ind), check_dtype=False, dist_test=False)


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
    check_func(test_impl, (A, idx, val), copy_input=True)


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

    py_output = pd.array(["A", None] + ["A"] * 6, "string")
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)


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

    py_output = pd.array(["A", None] + ["A"] * 6, "string")
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)


def test_setitem_slice(memory_leak_check):
    """
    Test operator.setitem with a slice index. String arrays
    should only have setitem used during initialization, so
    we create a new string array in the test.
    """

    def test_impl(val):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = "AB"
        A[1] = "CD"
        A[2:7] = val
        A[7] = "GH"
        return A

    values = (pd.array(["IJ"] * 5), ["IJ"] * 5, "IJ")
    py_output = pd.array(["AB", "CD"] + ["IJ"] * 5 + ["GH"], "string")
    for val in values:
        check_func(test_impl, (val,), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_slice_optional(memory_leak_check):
    """
    Test operator.setitem with a slice index and an optional type.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl(val, flag):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = "AB"
        A[1] = "CD"
        if flag:
            A[2:7] = val
        else:
            A[2:7] = None
        A[7] = "GH"
        return A

    values = (pd.array(["IJ"] * 5), ["IJ"] * 5, "IJ")
    py_output_flag = pd.array(["AB", "CD"] + ["IJ"] * 5 + ["GH"], "string")
    py_output_no_flag = pd.array(["AB", "CD"] + [None] * 5 + ["GH"], "string")
    for val in values:
        check_func(test_impl, (val, True), dist_test=False, py_output=py_output_flag)
        check_func(
            test_impl, (val, False), dist_test=False, py_output=py_output_no_flag
        )


@pytest.mark.slow
def test_setitem_slice_none(memory_leak_check):
    """
    Test operator.setitem with a slice index and None.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl():
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        A[0] = "AB"
        A[1] = "CD"
        A[2:7] = None
        A[7] = "GH"
        return A

    py_output = pd.array(["AB", "CD"] + [None] * 5 + ["GH"], "string")
    check_func(test_impl, (), dist_test=False, py_output=py_output)


def test_setitem_bool(memory_leak_check):
    """
    Test operator.setitem with a bool index.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(val, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        A[idx] = val
        return A

    values = (pd.array(["IJ"] * 5), np.array(["IJ"] * 5), "IJ")
    py_output = pd.array(["AB"] * 2 + ["IJ"] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    for val in values:
        check_func(test_impl, (val, idx), dist_test=False, py_output=py_output)
        check_func(test_impl, (val, array_idx), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_bool_optional(memory_leak_check):
    """
    Test operator.setitem with a bool index and an optional type.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(val, idx, flag):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        if flag:
            A[idx] = val
        else:
            A[idx] = None
        return A

    values = (pd.array(["IJ"] * 5), np.array(["IJ"] * 5), "IJ")
    py_output_flag = pd.array(["AB"] * 2 + ["IJ"] * 5 + ["AB"], "string")
    py_output_no_flag = pd.array(["AB"] * 2 + [None] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    for val in values:
        check_func(
            test_impl, (val, idx, False), dist_test=False, py_output=py_output_no_flag
        )
        check_func(
            test_impl,
            (val, array_idx, False),
            dist_test=False,
            py_output=py_output_no_flag,
        )
        check_func(
            test_impl, (val, idx, True), dist_test=False, py_output=py_output_flag
        )
        check_func(
            test_impl, (val, array_idx, True), dist_test=False, py_output=py_output_flag
        )


@pytest.mark.slow
def test_setitem_bool_none(memory_leak_check):
    """
    Test operator.setitem with a bool index and None.
    The bool setitem index is used with Series.loc/Series.iloc, so
    it modifies the array in place. However, the size of the elements
    should be the same.
    """

    def test_impl(idx):
        A = bodo.libs.str_arr_ext.pre_alloc_string_array(8, -1)
        for i in range(8):
            A[i] = "AB"
        A[idx] = None
        return A

    py_output = pd.array(["AB"] * 2 + [None] * 5 + ["AB"], "string")
    idx = [False, False, True, True, True, True, True, False]
    array_idx = np.array(idx)
    check_func(test_impl, (idx,), dist_test=False, py_output=py_output)
    check_func(test_impl, (array_idx,), dist_test=False, py_output=py_output)


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


def test_tolist(memory_leak_check):
    def impl(A):
        return A.tolist()

    # NOTE: only Numpy array has tolist(), not Pandas arrays
    check_func(impl, (np.array(["A", "BCD"] * 4),), only_seq=True)
    check_func(impl, (np.arange(11),), only_seq=True)


@pytest.mark.slow
def test_astype_str(memory_leak_check):
    def test_impl(A):
        return A.astype(str)

    check_func(test_impl, (pd.array(["AA", "B"] * 4),))


@pytest.mark.slow
def test_str_array_setitem_unsupported(memory_leak_check):
    """
    Checks that string array setitem with unsupported index, value
    pairs throw BodoErrors. String Array setitem shouldn't occur
    after initialization, but since these tests should error at compile
    time, this shouldn't be an issue.
    """

    def impl(arr, idx, val):
        arr[idx] = val

    err_msg = "StringArray setitem with index .* and value .* not supported yet."

    arr = pd.array(["AB", "", "ABC", pd.NA, "abcd"])
    # Check integer with a non-string
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, 2, ["erw", "qwewq"])
    # Check slice with a numpy str array
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, slice(0, 2), np.array(["erw", "qwe"]))

    bool_list = np.array([True, False, True, False, True])
    # Check boolean array with list
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, bool_list, ["erw", "qwewq"])
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, np.array(bool_list), ["erw", "qwewq"])

    # Check integer array index (not supported yet)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl)(arr, np.array([1, 2]), ["erw", "qwewq"])
