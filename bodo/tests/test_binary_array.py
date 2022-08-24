# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's binary array data type
"""
import operator

import numba
import numpy as np
import pandas as pd
import pytest

from bodo.tests.utils import check_func


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


def generate_comparison_ops_func(op, check_na=False):
    """
    Generates a comparison function. If check_na,
    then we are being called on a scalar value because Pandas
    can't handle NA values in the array. If so, we return None
    if either input is NA.
    """
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def test_impl(a, b):\n"
    if check_na:
        func_text += f"  if pd.isna(a) or pd.isna(b):\n"
        func_text += f"    return None\n"
    func_text += f"  return a {op_str} b\n"
    loc_vars = {}
    exec(func_text, {"pd": pd}, loc_vars)
    return loc_vars["test_impl"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", (operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt)
)
def test_bytes_comparison_ops(op, memory_leak_check):
    """
    Test logical comparisons between bytes values.
    """
    func = generate_comparison_ops_func(op)
    arg1 = b"abc"
    arg2 = b"c0e"
    check_func(func, (arg1, arg1))
    check_func(func, (arg1, arg2))
    check_func(func, (arg2, arg1))


@pytest.mark.slow
@pytest.mark.parametrize(
    "op", (operator.eq, operator.ne, operator.ge, operator.gt, operator.le, operator.lt)
)
def test_binary_bytes_comparison_ops(binary_arr_value, op, memory_leak_check):
    """
    Test logical comparisons between bytes and binary array.
    """
    func = generate_comparison_ops_func(op)
    arg1 = b"abc"
    # Generate a py_output value because pandas doesn't handle null
    pandas_func = generate_comparison_ops_func(op, check_na=True)
    py_output = pd.array([None] * len(binary_arr_value), dtype="boolean")
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(binary_arr_value[i], arg1)
    check_func(func, (binary_arr_value, arg1), py_output=py_output)
    # Update py_output
    for i in range(len(binary_arr_value)):
        py_output[i] = pandas_func(arg1, binary_arr_value[i])
    check_func(func, (arg1, binary_arr_value), py_output=py_output)


@pytest.mark.parametrize(
    "op",
    (
        operator.eq,
        pytest.param(operator.ne, marks=pytest.mark.slow),
        pytest.param(operator.ge, marks=pytest.mark.slow),
        pytest.param(operator.gt, marks=pytest.mark.slow),
        pytest.param(operator.le, marks=pytest.mark.slow),
        pytest.param(operator.lt, marks=pytest.mark.slow),
    ),
)
def test_binary_binary_comparison_ops(binary_arr_value, op, memory_leak_check):
    """
    Test logical comparisons between binary array and binary array.
    """
    func = generate_comparison_ops_func(op)
    arg1 = np.array([b"", b"c", np.nan, bytes(2)] * 4, object)
    # Generate a py_output value because pandas doesn't handle null
    pandas_func = generate_comparison_ops_func(op, check_na=True)
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
