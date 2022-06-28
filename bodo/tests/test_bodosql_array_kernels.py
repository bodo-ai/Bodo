# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL
"""

import re

import pandas as pd
import pytest

import bodo
from bodo.libs import bodosql_array_kernels
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


def vectorized_sol(args, scalar_fn, dtype):
    """Creates a py_output for a vectorized function using its arguments and the
       a function that is applied to the scalar values

    Args:
        args (any list): a list of arguments, each of which is either a scalar
        or vector (vectors must be the same size)
        scalar_fn (function): the function that is applied to scalar values
        corresponding to each row
        dtype (dtype): the dtype of the final output array

    Returns:
        scalar or Series: the result of applying scalar_fn to each row of the
        vectors with scalar args broadcasted (or just the scalar output if
        all of the arguments are scalar)
    """
    length = -1
    for arg in args:
        if isinstance(arg, (pd.core.arrays.base.ExtensionArray, pd.Series)):
            length = len(arg)
            break
    if length == -1:
        return scalar_fn(*args)
    arglist = []
    for arg in args:
        if isinstance(arg, (pd.core.arrays.base.ExtensionArray, pd.Series)):
            arglist.append(arg)
        else:
            arglist.append([arg] * length)
    return pd.Series([scalar_fn(*params) for params in zip(*arglist)], dtype=dtype)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.array(["alpha", "beta", "gamma", "delta", "epsilon"]),
                pd.array([2, 4, 8, 16, 32]),
                pd.array(["_", "_", "_", "AB", "123"]),
            ),
        ),
        pytest.param(
            (
                pd.array([None, "words", "words", "words", "words", "words"]),
                pd.array([16, None, 16, 0, -5, 16]),
                pd.array(["_", "_", None, "_", "_", ""]),
            ),
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, "_"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 0, "_"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), None, "_"),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, ""),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, None),
            marks=pytest.mark.slow,
        ),
        pytest.param(("words", 20, "0123456789"), marks=pytest.mark.slow),
        pytest.param((None, 20, "0123456789"), marks=pytest.mark.slow),
        pytest.param(
            ("words", pd.array([2, 4, 8, 16, 32]), "0123456789"), marks=pytest.mark.slow
        ),
        pytest.param(
            (None, 20, pd.array(["A", "B", "C", "D", "E"])), marks=pytest.mark.slow
        ),
        pytest.param(
            (
                "words",
                30,
                pd.array(["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "", None]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "words",
                pd.array([-10, 0, 10, 20, 30]),
                pd.array([" ", " ", " ", "", None]),
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param((None, None, None), marks=pytest.mark.slow),
        pytest.param(
            (
                pd.array(["A", "B", "C", "D", "E"]),
                pd.Series([2, 4, 6, 8, 10]),
                pd.Series(["_"] * 5),
            ),
        ),
    ],
)
def test_lpad_rpad(args):
    def impl1(arr, length, lpad_string):
        return bodo.libs.bodosql_array_kernels.lpad(arr, length, lpad_string)

    def impl2(arr, length, rpad_string):
        return bodo.libs.bodosql_array_kernels.rpad(arr, length, rpad_string)

    # Simulates LPAD on a single element
    def lpad_scalar_fn(elem, length, pad):
        if pd.isna(elem) or pd.isna(length) or pd.isna(pad):
            return None
        elif pad == "":
            return elem
        elif length <= 0:
            return ""
        elif len(elem) > length:
            return elem[:length]
        else:
            return (pad * length)[: length - len(elem)] + elem

    # Simulates RPAD on a single element
    def rpad_scalar_fn(elem, length, pad):
        if pd.isna(elem) or pd.isna(length) or pd.isna(pad):
            return None
        elif pad == "":
            return elem
        elif length <= 0:
            return ""
        elif len(elem) > length:
            return elem[:length]
        else:
            return elem + (pad * length)[: length - len(elem)]

    arr, length, pad_string = args
    lpad_answer = vectorized_sol(
        (arr, length, pad_string), lpad_scalar_fn, pd.StringDtype()
    )
    rpad_answer = vectorized_sol(
        (arr, length, pad_string), rpad_scalar_fn, pd.StringDtype()
    )
    check_func(
        impl1,
        (arr, length, pad_string),
        py_output=lpad_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        (arr, length, pad_string),
        py_output=rpad_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_lpad_rpad():
    def impl1(arr, length, lpad_string, flag1, flag2):
        B = length if flag1 else None
        C = lpad_string if flag2 else None
        return bodosql_array_kernels.lpad(arr, B, C)

    def impl2(val, length, lpad_string, flag1, flag2, flag3):
        A = val if flag1 else None
        B = length if flag2 else None
        C = lpad_string if flag3 else None
        return bodosql_array_kernels.rpad(A, B, C)

    arr, length, pad_string = pd.array(["A", "B", "C", "D", "E"]), 3, " "
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1 and flag2:
                answer = pd.array(["  A", "  B", "  C", "  D", "  E"])
            else:
                answer = pd.array([None] * 5, dtype=pd.StringDtype())
            check_func(
                impl1,
                (arr, length, pad_string, flag1, flag2),
                py_output=answer,
                check_dtype=False,
            )

    val, length, pad_string = "alpha", 10, "01"
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            for flag3 in [True, False]:
                if flag1 and flag2 and flag3:
                    answer = "alpha01010"
                else:
                    answer = None
                check_func(
                    impl2,
                    (val, length, pad_string, flag1, flag2, flag3),
                    py_output=answer,
                )


@pytest.mark.slow
def test_error_lpad_rpad():
    def impl1(arr, length, lpad_string):
        return bodosql_array_kernels.lpad(arr, length, lpad_string)

    def impl2(arr):
        return bodosql_array_kernels.lpad(arr, "$", " ")

    def impl3(arr):
        return bodosql_array_kernels.lpad(arr, 42, 0)

    def impl4(arr, length, lpad_string):
        return bodosql_array_kernels.rpad(arr, length, lpad_string)

    def impl5(arr):
        return bodosql_array_kernels.rpad(arr, "$", " ")

    def impl6(arr):
        return bodosql_array_kernels.rpad(arr, 42, 0)

    err_msg1 = re.escape(
        "LPAD length argument must be an integer, integer column, or null"
    )
    err_msg2 = re.escape(
        "LPAD lpad_string argument must be a string, string column, or null"
    )
    err_msg3 = re.escape("LPAD can only be applied to strings, string columns, or null")
    err_msg4 = re.escape(
        "RPAD length argument must be an integer, integer column, or null"
    )
    err_msg5 = re.escape(
        "RPAD rpad_string argument must be a string, string column, or null"
    )
    err_msg6 = re.escape("RPAD can only be applied to strings, string columns, or null")

    A1 = pd.array(["A", "B", "C", "D", "E"])
    A2 = pd.array([1, 2, 3, 4, 5])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(A1, "_", "X")

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl2)(A1)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl1)(A1, 10, 2)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl3)(A1)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl1)(A2, 10, "_")

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl4)(A1, "_", "X")

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl5)(A1)

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl4)(A1, 10, 2)

    with pytest.raises(BodoError, match=err_msg5):
        bodo.jit(impl6)(A1)

    with pytest.raises(BodoError, match=err_msg6):
        bodo.jit(impl4)(A2, 10, "_")
