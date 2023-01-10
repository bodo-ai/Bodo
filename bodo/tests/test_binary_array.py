# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's binary array data type
"""

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import check_func, generate_comparison_ops_func


@pytest.fixture(
    params=[
        np.array(
            [b"", b"abc", b"c", np.nan, b"ccdefg", b"abcde", b"poiu", bytes(3)] * 2,
            object,
        ),
    ]
)
def binary_arr_value(request):
    return request.param


def test_unbox(binary_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl, (binary_arr_value,))
    check_func(impl2, (binary_arr_value,))


@pytest.mark.slow
def test_len(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (binary_arr_value,))


@pytest.mark.slow
def test_shape(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (binary_arr_value,))


@pytest.mark.slow
def test_ndim(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (binary_arr_value,))


def test_copy(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return A.copy()

    check_func(test_impl, (binary_arr_value,))


def test_constant_lowering(binary_arr_value, memory_leak_check):
    def test_impl():
        return binary_arr_value

    check_func(test_impl, (), only_seq=True)


@pytest.mark.slow
def test_hex_method(binary_arr_value, memory_leak_check):
    def test_impl(A):
        return pd.Series(A).apply(lambda x: None if pd.isna(x) else x.hex())

    check_func(test_impl, (binary_arr_value,))


@pytest.mark.slow
def test_bytes_hash(binary_arr_value, memory_leak_check):
    """
    Test support for Bytes.__hash using nunique.
    """

    def test_impl(A):
        return pd.Series(A).map(lambda x: None if pd.isna(x) else hash(x))

    # check_dtype=False because None converts the output to Float in Pandas
    # dist_test = False because the randomness causes different inputs on each core.
    # only_seq=True because the hash function has randomness between cores
    check_func(
        test_impl,
        (binary_arr_value,),
        check_dtype=False,
        only_seq=True,
    )


@pytest.mark.slow
def test_bytes_comparison_ops(cmp_op, memory_leak_check):
    """
    Test logical comparisons between bytes values.
    """
    func = generate_comparison_ops_func(cmp_op)
    arg1 = b"abc"
    arg2 = b"c0e"
    check_func(func, (arg1, arg1))
    check_func(func, (arg1, arg2))
    check_func(func, (arg2, arg1))


@pytest.mark.slow
def test_binary_bytes_comparison_ops(binary_arr_value, cmp_op, memory_leak_check):
    """
    Test logical comparisons between bytes and binary array.
    """
    func = generate_comparison_ops_func(cmp_op)
    arg1 = b"abc"
    # Generate a py_output value because pandas doesn't handle null
    pandas_func = generate_comparison_ops_func(cmp_op, check_na=True)
    py_output = pd.array([None] * len(binary_arr_value), dtype="boolean")
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(binary_arr_value[i], arg1)
    check_func(func, (binary_arr_value, arg1), py_output=py_output)
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(arg1, binary_arr_value[i])
    check_func(func, (arg1, binary_arr_value), py_output=py_output)


def test_binary_binary_comparison_ops(binary_arr_value, cmp_op, memory_leak_check):
    """
    Test logical comparisons between binary array and binary array.
    """
    func = generate_comparison_ops_func(cmp_op)
    arg1 = np.array([b"", b"c", np.nan, bytes(2)] * 4, object)
    # Generate a py_output value because pandas doesn't handle null
    pandas_func = generate_comparison_ops_func(cmp_op, check_na=True)
    py_output = pd.array([None] * len(binary_arr_value), dtype="boolean")
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(binary_arr_value[i], arg1[i])
    check_func(func, (binary_arr_value, arg1), py_output=py_output)
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(arg1[i], binary_arr_value[i])
    check_func(func, (arg1, binary_arr_value), py_output=py_output)


# TODO: [BE-1157] Fix Memory leak
def test_get_item(binary_arr_value):
    def test_impl(A, idx):
        return A[idx]

    np.random.seed(0)

    # A single integer
    idx = 0
    check_func(test_impl, (binary_arr_value, idx))

    # Array of integers
    idx = np.random.randint(0, len(binary_arr_value), 11)
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)

    # Array of booleans
    idx = np.random.ranf(len(binary_arr_value)) < 0.2
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)

    # Slice
    idx = slice(1, 4)
    check_func(test_impl, (binary_arr_value, idx), dist_test=False)


# TODO: [BE-1157] Fix Memory leak
def test_bytes_fromhex():
    def impl(hex_val):
        return bytes.fromhex(hex_val)

    check_func(impl, ("121134a320",))
    check_func(impl, ("1211 34a302",))
    check_func(impl, (b"HELLO".hex(),))
    # Test for a trailing space
    check_func(impl, ("1e21\t",))


@pytest.mark.slow
def test_binary_series_apply(binary_arr_value, memory_leak_check):
    def test_impl1(S):
        return S.apply(lambda x: None if pd.isna(x) else x)

    def test_impl2(S):
        return S.map(lambda x: None if pd.isna(x) else x)

    S = pd.Series(binary_arr_value)
    check_func(
        test_impl1,
        (S,),
    )
    check_func(
        test_impl2,
        (S,),
    )


@pytest.mark.slow
def test_binary_dataframe_apply(binary_arr_value, memory_leak_check):
    def test_impl(df):
        return df.apply(lambda x: None if pd.isna(x["A"]) else x["A"], axis=1)

    df = pd.DataFrame(
        {
            "A": binary_arr_value,
            "B": 1,
        }
    )
    check_func(
        test_impl,
        (df,),
    )


# Binary setitem tests. Modified from their counterparts in test_string_array.py


@pytest.fixture(
    params=[
        pytest.param(np.array([b"IJ"] * 5, object), id="array"),
        pytest.param([b"IJ"] * 5, id="list"),
        pytest.param(b"IJ", id="scalar"),
    ],
)
def setitem_val(request):
    return request.param


@pytest.mark.slow
def test_setitem_slice(memory_leak_check, setitem_val):
    """
    Test operator.setitem with a slice index. String arrays
    should only have setitem used during initialization, so
    we create a new string array in the test.
    """

    def test_impl(val):
        A = bodo.libs.str_arr_ext.pre_alloc_binary_array(8, -1)
        A[0] = b"Afdhui"
        A[1] = b"CD"
        A[2:7] = val
        A[7] = b"GH"
        return A

    py_output = np.array([b"Afdhui", b"CD"] + [b"IJ"] * 5 + [b"GH"], object)
    check_func(test_impl, (setitem_val,), dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_slice_optional(memory_leak_check, setitem_val):
    """
    Test operator.setitem with a slice index and an optional type.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl(val, flag):
        A = bodo.libs.str_arr_ext.pre_alloc_binary_array(8, -1)
        A[0] = b"AB"
        A[1] = b"CD"
        if flag:
            A[2:7] = val
        else:
            A[2:7] = None
        A[7] = b"GH"
        return A

    py_output_flag = np.array([b"AB", b"CD"] + [b"IJ"] * 5 + [b"GH"], object)
    py_output_no_flag = np.array([b"AB", b"CD"] + [None] * 5 + [b"GH"], object)

    check_func(
        test_impl, (setitem_val, True), dist_test=False, py_output=py_output_flag
    )
    check_func(
        test_impl, (setitem_val, False), dist_test=False, py_output=py_output_no_flag
    )


@pytest.mark.slow
def test_setitem_slice_none(memory_leak_check):
    """
    Test operator.setitem with a slice index and None.
    String arrays should only have setitem used during
    initialization, so we create a new string array in the test.
    """

    def test_impl():
        A = bodo.libs.str_arr_ext.pre_alloc_binary_array(8, -1)
        A[0] = b"AB"
        A[1] = b"CD"
        A[2:7] = None
        A[7] = b"GH"
        return A

    py_output = np.array([b"AB", b"CD"] + [None] * 5 + [b"GH"], object)
    check_func(test_impl, (), dist_test=False, py_output=py_output)


@pytest.mark.smoke
def test_setitem_int(memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    A = np.array([b"AB", b"", b"121", np.nan, b"abcd", bytes(3)], object)
    idx = 2
    val = b"212"  # same size as element 2 but different value
    check_func(test_impl, (A, idx, val), copy_input=True)


@pytest.mark.slow
def test_setitem_none_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_binary_array(n, n - 1)
        for i in range(n):
            if i == idx:
                A[i] = None
                continue
            A[i] = b"A"
        return A

    py_output = np.array([b"A", None] + [b"A"] * 6, object)
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)


@pytest.mark.slow
def test_setitem_optional_int(memory_leak_check):
    def test_impl(n, idx):
        A = bodo.libs.str_arr_ext.pre_alloc_binary_array(n, n - 1)
        for i in range(n):
            if i == idx:
                value = None
            else:
                value = b"A"
            A[i] = value
        return A

    py_output = np.array([b"A", None] + [b"A"] * 6, object)
    check_func(test_impl, (8, 1), copy_input=True, dist_test=False, py_output=py_output)
