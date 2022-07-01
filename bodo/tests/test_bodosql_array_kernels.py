# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL
"""

import re

import numpy as np
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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [1, None, 3, None, 5, None, 7, None], dtype=pd.Int32Dtype()
                    )
                ),
                pd.Series(
                    pd.array(
                        [2, 3, 5, 7, None, None, None, None], dtype=pd.Int32Dtype()
                    )
                ),
            ),
            id="int_series_2",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [1, 2, None, None, 3, 4, None, None], dtype=pd.Int32Dtype()
                    )
                ),
                None,
                pd.Series(
                    pd.array(
                        [None, None, None, None, None, None, None, None],
                        dtype=pd.Int32Dtype(),
                    )
                ),
                pd.Series(
                    pd.array(
                        [None, 5, None, 6, None, None, None, 7], dtype=pd.Int32Dtype()
                    )
                ),
                42,
                pd.Series(
                    pd.array(
                        [8, 9, 10, None, None, None, None, 11], dtype=pd.Int32Dtype()
                    )
                ),
            ),
            id="int_series_scalar_6",
        ),
        pytest.param((None, None, 3, 4, 5, None), id="int_scalar_6"),
        pytest.param(
            (None, None, None, None, None, None),
            id="all_null_6",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [None, "AB", None, "CD", None, "EF", None, "GH"],
                        dtype=pd.StringDtype(),
                    )
                ),
                pd.Series(
                    pd.array(
                        ["IJ", "KL", None, None, "MN", "OP", None, None],
                        dtype=pd.StringDtype(),
                    )
                ),
            ),
            id="string_series_2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [1, None, 3, None, 5, None, 7, None], dtype=pd.Int16Dtype()
                    )
                ),
                pd.Series(
                    pd.array(
                        [2, 3, 5, 2**38, None, None, None, None],
                        dtype=pd.Int64Dtype(),
                    )
                ),
            ),
            id="mixed_int_series_2",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [4, None, 64, None, 256, None, 1024, None],
                        dtype=pd.UInt16Dtype(),
                    )
                ),
                pd.Series(
                    pd.array(
                        [1.1, 1.2, 1.3, 1.4, None, None, None, None],
                        dtype=np.float64,
                    )
                ),
            ),
            id="int_float_series_2",
        ),
        pytest.param((42,), id="int_1", marks=pytest.mark.slow),
        pytest.param((42,), id="none_1", marks=pytest.mark.slow),
        pytest.param(
            (pd.array([1, 2, 3, 4, 5]),), id="int_array_1", marks=pytest.mark.slow
        ),
    ],
)
def test_coalesce(args):
    def impl1(A, B):
        return bodo.libs.bodosql_array_kernels.coalesce((A, B))

    def impl2(A, B, C, D, E, F):
        return bodo.libs.bodosql_array_kernels.coalesce((A, B, C, D, E, F))

    def impl3(A):
        return bodo.libs.bodosql_array_kernels.coalesce((A,))

    def coalesce_scalar_fn(*args):
        for arg in args:
            if not pd.isna(arg):
                return arg

    coalesce_answer = vectorized_sol(args, coalesce_scalar_fn, None)

    if len(args) == 2:
        check_func(
            impl1, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
        )
    elif len(args) == 6:
        check_func(
            impl2, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
        )
    elif len(args) == 1:
        check_func(
            impl3, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
        )


@pytest.mark.slow
def test_option_coalesce():
    def impl1(arr, scale1, scale2, flag1, flag2):
        A = scale1 if flag1 else None
        B = scale2 if flag2 else None
        return bodosql_array_kernels.coalesce((A, arr, B))

    arr, scale1, scale2 = pd.array(["A", None, "C", None, "E"]), "", " "
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1:
                answer = pd.Series(["", "", "", "", ""])
            elif flag2:
                answer = pd.Series(["A", " ", "C", " ", "E"])
            else:
                answer = pd.Series(["A", None, "C", None, "E"])
            check_func(
                impl1,
                (arr, scale1, scale2, flag1, flag2),
                py_output=answer,
                check_dtype=False,
                reset_index=True,
            )


@pytest.mark.slow
def test_error_coalesce():
    def impl1(A, B, C):
        return bodosql_array_kernels.coalesce((A, B, C))

    def impl2(A, B, C):
        return bodosql_array_kernels.coalesce([A, B, C])

    def impl3():
        return bodosql_array_kernels.coalesce(())

    # Note: not testing non-constant tuples because the kernel is only used
    # by BodoSQL in cases where we do the code generation and can guarantee
    # that the tuple is constant

    err_msg1 = re.escape("Cannot coalesce columns with different dtypes")
    err_msg2 = re.escape("Coalesce argument must be a tuple")
    err_msg3 = re.escape("Cannot coalesce 0 columns")

    A = pd.Series(["A", "B", "C", "D", "E"])
    B = pd.Series(["D", "E", "F", "G", "H"])
    C = pd.Series([123, 456, 789, 123, 456])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(A, B, C)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl2)(A, B, C)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl3)()


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                pd.Series([1, -4, 3, 14, 5, 0]),
            ),
            id="all_vector_no_null",
        ),
        pytest.param(
            (
                pd.Series(pd.array(["AAAAA", "BBBBB", "CCCCC", None] * 3)),
                pd.Series(pd.array([2, 4, None] * 4)),
            ),
            id="all_vector_some_null",
        ),
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                4,
            ),
            id="vector_string_scalar_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                pd.Series(pd.array(list(range(-2, 11)))),
            ),
            id="scalar_string_vector_int",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                6,
            ),
            id="all_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]),
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "alphabet",
                None,
            ),
            id="scalar_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_left_right(args):
    def impl1(arr, n_chars):
        return bodo.libs.bodosql_array_kernels.left(arr, n_chars)

    def impl2(arr, n_chars):
        return bodo.libs.bodosql_array_kernels.right(arr, n_chars)

    # Simulates LEFT on a single row
    def left_scalar_fn(elem, n_chars):
        if pd.isna(elem) or pd.isna(n_chars):
            return None
        elif n_chars <= 0:
            return ""
        else:
            return elem[:n_chars]

    # Simulates RIGHT on a single row
    def right_scalar_fn(elem, n_chars):
        if pd.isna(elem) or pd.isna(n_chars):
            return None
        elif n_chars <= 0:
            return ""
        else:
            return elem[-n_chars:]

    arr, n_chars = args
    left_answer = vectorized_sol((arr, n_chars), left_scalar_fn, pd.StringDtype())
    right_answer = vectorized_sol((arr, n_chars), right_scalar_fn, pd.StringDtype())
    check_func(
        impl1,
        (arr, n_chars),
        py_output=left_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        (arr, n_chars),
        py_output=right_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_left_right():
    def impl1(scale1, scale2, flag1, flag2):
        arr = scale1 if flag1 else None
        n_chars = scale2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.left(arr, n_chars)

    def impl2(scale1, scale2, flag1, flag2):
        arr = scale1 if flag1 else None
        n_chars = scale2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.right(arr, n_chars)

    scale1, scale2 = "alphabet soup", 10
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1 and flag2:
                answer1 = "alphabet s"
                answer2 = "habet soup"
            else:
                answer1 = None
                answer2 = None
            check_func(
                impl1,
                (scale1, scale2, flag1, flag2),
                py_output=answer1,
                check_dtype=False,
            )
            check_func(
                impl2,
                (scale1, scale2, flag1, flag2),
                py_output=answer2,
                check_dtype=False,
            )


@pytest.mark.slow
def test_error_left_right():
    def impl1(arr, n_chars):
        return bodo.libs.bodosql_array_kernels.left(arr, n_chars)

    def impl2(arr, n_chars):
        return bodo.libs.bodosql_array_kernels.right(arr, n_chars)

    err_msg1 = re.escape(
        "LEFT n_chars argument must be an integer, integer column, or null"
    )
    err_msg2 = re.escape("LEFT can only be applied to strings, string columns, or null")
    err_msg3 = re.escape(
        "RIGHT n_chars argument must be an integer, integer column, or null"
    )
    err_msg4 = re.escape(
        "RIGHT can only be applied to strings, string columns, or null"
    )

    A = pd.Series(["A", "B", "C", "D", "E"])
    B = pd.Series([123, 456, 789, 123, 456])

    with pytest.raises(BodoError, match=err_msg1):
        bodo.jit(impl1)(A, A)

    with pytest.raises(BodoError, match=err_msg2):
        bodo.jit(impl1)(B, B)

    with pytest.raises(BodoError, match=err_msg3):
        bodo.jit(impl2)(A, A)

    with pytest.raises(BodoError, match=err_msg4):
        bodo.jit(impl2)(B, B)
