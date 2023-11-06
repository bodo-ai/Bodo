# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL variadic functions
"""

import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from numba.core import types

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import (
    SeriesOptTestPipeline,
    check_func,
    dist_IR_count,
    find_nested_dispatcher_and_args,
    pytest_slow_unless_codegen,
)

# Skip unless any library or BodoSQL codegen or files were changed
pytestmark = pytest_slow_unless_codegen


def coalesce_expected_output(args):
    def coalesce_scalar_fn(*args):
        for arg in args:
            if not pd.isna(arg):
                return arg

    return vectorized_sol(args, coalesce_scalar_fn, None)


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
def test_coalesce(args, memory_leak_check):
    """Test BodoSQL COALESCE kernel"""
    n_args = len(args)
    args_str = ", ".join(f"A{i}" for i in range(n_args))
    test_impl = f"def impl({args_str}):\n"
    series_str = (
        "pd.Series"
        if any(
            isinstance(a, (pd.Series, pd.core.arrays.base.ExtensionArray)) for a in args
        )
        else ""
    )
    test_impl += f"  return {series_str}(bodo.libs.bodosql_array_kernels.coalesce(({args_str},)))"
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)
    impl = impl_vars["impl"]

    coalesce_answer = coalesce_expected_output(args)

    check_func(
        impl, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
    )


def test_coalesce_str_array_optimized(memory_leak_check):
    """Test that the BodoSQL COALESCE kernel doesn't produce intermediate allocations
    when processing string arrays/Series."""
    S0 = pd.Series(
        ["afa", "erwoifnewoi", "Rer", pd.NA, "مرحبا, العالم ، هذا هو بودو"] * 5
    )
    S1 = pd.Series(["a", "b", "c", "d", pd.NA] + (["a", pd.NA, "a", pd.NA] * 5))
    scalar = "مرحبا, العالم ، هذا"
    args = (S0, S1, scalar)
    args_str = ", ".join(f"A{i}" for i in range(len(args)))
    test_impl = f"def impl({args_str}):\n"
    series_str = (
        "pd.Series"
        if any(
            isinstance(a, (pd.Series, pd.core.arrays.base.ExtensionArray)) for a in args
        )
        else ""
    )
    test_impl += f"  return {series_str}(bodo.libs.bodosql_array_kernels.coalesce(({args_str},)))"
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)
    impl = impl_vars["impl"]

    coalesce_answer = coalesce_expected_output(args)

    check_func(
        impl, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
    )

    # Verify get_str_arr_item_copy is in the IR and there is no intermediate
    # allocation. This function is not inlined so we must traverse several steps to get to
    # the actual IR in question.
    bodo_func = bodo.jit(parallel=True)(impl)
    # Find the coalesce dispatcher in the IR
    arg_typs = tuple([bodo.typeof(arg) for arg in args])
    dispatcher, used_sig = find_nested_dispatcher_and_args(
        bodo_func, arg_typs, ("coalesce", "bodo.libs.bodosql_array_kernels")
    )
    # Note: infrastructure doesn't handle defaults, so we need to manually add this to
    # the signature
    used_sig = used_sig + (types.none, types.int64)
    # Find the coalesce_util dispatcher in the IR
    dispatcher, used_sig = find_nested_dispatcher_and_args(
        dispatcher,
        used_sig,
        ("coalesce_util", "bodo.libs.bodosql_variadic_array_kernels"),
    )
    # Verify get_str_arr_item_copy is in the IR. find_nested_dispatcher_and_args
    # will throw an assertion error if it doesn't exist.
    find_nested_dispatcher_and_args(
        dispatcher,
        used_sig,
        ("get_str_arr_item_copy", "bodo.libs.str_arr_ext"),
        return_dispatcher=False,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series((list(range(9)) + [None]) * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, None] * 10, dtype=pd.Int32Dtype()),
                pd.Series(["A", None] * 10),
                pd.Series([1, None] * 10, dtype=pd.Int32Dtype()),
                "B",
                pd.Series([2, None] * 10, dtype=pd.Int32Dtype()),
                None,
                None,
                pd.Series(["D", None] * 10),
                None,
                "E",
                None,
                None,
                6,
                pd.Series(["G", None] * 10),
                7,
                "H",
                8,
                None,
            ),
            id="vector_vector_no_default",
        ),
        pytest.param(
            (
                pd.Series((list(range(9)) + [None]) * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, None] * 10, dtype=pd.Int32Dtype()),
                pd.Series(["A", None] * 10),
                pd.Series([1, None] * 10, dtype=pd.Int32Dtype()),
                "B",
                pd.Series([2, None] * 10, dtype=pd.Int32Dtype()),
                None,
                None,
                pd.Series(["D", None] * 10),
                None,
                "E",
                None,
                None,
                6,
                pd.Series(["G", None] * 10),
                7,
                "H",
                8,
                None,
                pd.Series(
                    [
                        "a",
                        None,
                        "c",
                        None,
                        "e",
                        None,
                        "g",
                        None,
                        "i",
                        None,
                        "k",
                        None,
                        "m",
                        None,
                        "o",
                        None,
                        "q",
                        None,
                        "s",
                        None,
                    ]
                ),
            ),
            id="vector_vector_vector_default",
        ),
        pytest.param(
            (
                pd.Series((list(range(9)) + [None]) * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, None] * 10, dtype=pd.Int32Dtype()),
                pd.Series(["A", None] * 10),
                pd.Series([1, None] * 10, dtype=pd.Int32Dtype()),
                "B",
                pd.Series([2, None] * 10, dtype=pd.Int32Dtype()),
                None,
                None,
                "E",
                None,
                pd.Series(["D", None] * 10),
                None,
                None,
                6,
                pd.Series(["G", None] * 10),
                7,
                "H",
                8,
                None,
                "J",
            ),
            id="vector_scalar_scalar_default",
        ),
        pytest.param(
            (
                pd.Series((list(range(9)) + [None]) * 2, dtype=pd.Int32Dtype()),
                pd.Series([0, None] * 10, dtype=pd.Int32Dtype()),
                pd.Series(["A", None] * 10),
                pd.Series([1, None] * 10, dtype=pd.Int32Dtype()),
                "B",
                pd.Series([2, None] * 10, dtype=pd.Int32Dtype()),
                None,
                None,
                None,
                None,
                pd.Series(["D", None] * 10),
                None,
                "E",
                6,
                pd.Series(["G", None] * 10),
                7,
                "H",
                8,
                None,
                None,
            ),
            id="vector_null_null_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (1, 1, "X"),
            id="scalar_scalar_match",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (-2.0, -3.0, "X"),
            id="scalar_scalar_no_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("foo", "bar", "Y", "N"),
            id="scalar_scalar_scalar_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (np.uint8(42), np.uint8(255), "_", pd.Series(list("uvwxyz"))),
            id="scalar_scalar_vector_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("foo", pd.Series(["foo", "bar", "fizz", "buzz", "foo", None] * 2), ":)"),
            id="scalar_null_scalar_no_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "foo",
                pd.Series(["foo", "bar", "fizz", "buzz", "foo", None] * 2),
                ":)",
                ":(",
            ),
            id="scalar_vector_scalar_scalar_default",
        ),
        pytest.param(
            (
                None,
                pd.Series(["A", None, "C", None, "E", None, "G", None] * 2),
                pd.Series([1, 2, None, None] * 4, dtype=pd.Int32Dtype()),
                pd.Series(["A", "B", "C", "D", None, None, None, None] * 2),
                pd.Series([3, 4, None, None] * 4, dtype=pd.Int32Dtype()),
            ),
            id="null_four_vector_no_default",
        ),
        pytest.param(
            (
                None,
                pd.Series(["A", None, "C", None, "E", None, "G", None] * 2),
                pd.Series([1, 2, None, None] * 4, dtype=pd.Int32Dtype()),
                pd.Series(["A", "B", "C", "D", None, None, None, None] * 2),
                pd.Series([3, 4, None, None] * 4, dtype=pd.Int32Dtype()),
                42,
            ),
            id="null_four_vector_scalar_default",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series(["A", None, "C", None, "E", None, "G", None] * 2),
                pd.Series([1, 2, None, None] * 4, dtype=pd.Int32Dtype()),
                None,
                16,
                pd.Series(["A", "B", "C", "D", None, None, None, None] * 2),
                pd.Series([3, 4, None, None] * 4, dtype=pd.Int32Dtype()),
                42,
            ),
            id="null_null",
        ),
        pytest.param(
            (None, "A", 42, 16), id="null_no_match_scalar", marks=pytest.mark.slow
        ),
        pytest.param(
            (None, "A", 42, pd.Series([0, 1, 2, 3, 4, 5])),
            id="null_no_match_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if i % 5 == 0 else chr(65 + ((i + 13) ** 2) % 15)
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 6 == 0 else chr(65 + ((i + 14) ** 2) % 15)
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 7 == 0 else 65 + ((i + 14) ** 2) % 15
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 8 == 0 else chr(65 + ((i + 15) ** 2) % 15)
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 9 == 0 else 65 + ((i + 15) ** 2) % 15
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 11 == 0 else chr(65 + ((i + 16) ** 2) % 15)
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 13 == 0 else 65 + ((i + 16) ** 2) % 15
                        for i in range(500)
                    ]
                ),
                pd.Series(
                    [
                        None if i % 17 == 0 else 65 + ((i + 17) ** 2) % 15
                        for i in range(500)
                    ]
                ),
            ),
            id="all_vector_large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        None if i % 5 == 0 else chr(65 + ((i + 13) ** 2) % 26)
                        for i in range(500)
                    ]
                ),
                *("A", 0, "E", 1, "I", 2, "O", 3, "U", 4, "Y", 5, None, -2, -1),
            ),
            id="vector_all_scalar_large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        (lambda x: x / 10 if x < 3 else None)(((i + 20) ** 2) % 4)
                        for i in range(500)
                    ]
                ),
                0.0,
                pd.Series((list("AEIOU") + [None] * 5) * 50),
                0.1,
                pd.Series(([None] * 5 + list("AEIOU")) * 50),
                None,
                pd.Series(["alpha", "beta", "gamma", None, None] * 100),
                pd.Series([None, "delta", None, "epsilon", None] * 100),
            ),
            id="vector_scalar_vectors_large",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        *pd.date_range("2018-01-01", "2018-06-02", freq="M"),
                        None,
                        None,
                        *pd.date_range("2018-03-01", "2018-07-02", freq="M"),
                    ]
                    * 2
                ),
                np.datetime64(pd.Timestamp("2018-02-28"), "ns"),
                np.datetime64(pd.Timestamp("2018-02-01"), "ns"),
                np.datetime64(pd.Timestamp("2018-03-31"), "ns"),
                np.datetime64(pd.Timestamp("2018-03-01"), "ns"),
                np.datetime64(pd.Timestamp("2018-05-30"), "ns"),
                np.datetime64(pd.Timestamp("2018-05-01"), "ns"),
                pd.Series(pd.date_range("2018", "2019-11-01", freq="M")),
            ),
            id="date_vector_output",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                np.datetime64(pd.Timestamp("2018-01-01"), "ns"),
                np.datetime64(pd.Timestamp("2018-01-01"), "ns"),
                np.datetime64(pd.Timestamp("2019-06-20"), "ns"),
            ),
            id="date_scalar_output",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([0, 127, 128, 255, None] * 5, dtype=pd.UInt8Dtype()),
                pd.Series([0, 127, -128, -1, None] * 5, dtype=pd.Int8Dtype()),
                pd.Series([255] * 25, dtype=pd.UInt8Dtype()),
                pd.Series([-128, -1, 128, -1, -(2**34)] * 5, dtype=pd.Int64Dtype()),
                pd.Series([2**63 - 1] * 25, dtype=pd.Int64Dtype()),
            ),
            id="all_vector_multiple_types",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_decode(args, memory_leak_check):
    """Test BodoSQL DECODE kernel"""
    # generate test function for the number of args
    n_args = len(args)
    args_str = ", ".join(f"A{i}" for i in range(n_args))
    test_impl = f"def impl({args_str}):\n"
    series_str = "pd.Series" if any(isinstance(a, pd.Series) for a in args) else ""
    test_impl += (
        f"  return {series_str}(bodo.libs.bodosql_array_kernels.decode(({args_str})))"
    )
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)
    impl = impl_vars["impl"]

    def decode_scalar_fn(*args):
        for i in range(1, len(args) - 1, 2):
            if (pd.isna(args[0]) and pd.isna(args[i])) or (
                not pd.isna(args[0]) and not pd.isna(args[i]) and args[0] == args[i]
            ):
                return args[i + 1]
        if len(args) % 2 == 0:
            return args[-1]

    decode_answer = vectorized_sol(args, decode_scalar_fn, None)
    check_func(impl, args, py_output=decode_answer, check_dtype=False, reset_index=True)


def test_dict_arr_coalesce_null(memory_leak_check):
    """Test coalesce behavior with dictionary encoded arrays and a scalar NULL."""

    def impl(arr0, arr1, scalar):
        return pd.Series(bodo.libs.bodosql_array_kernels.coalesce((arr0, arr1, scalar)))

    arr0 = pd.arrays.ArrowStringArray(
        pa.array(
            ["afa", "erwoifnewoi", "Rer", None, "مرحبا, العالم ، هذا هو بودو"] * 5,
            type=pa.dictionary(pa.int32(), pa.string()),
        )
    )
    arr1 = pd.arrays.ArrowStringArray(
        pa.array(
            ["a", "b", "c", "d", None] + (["a", None, "a", None] * 5),
            type=pa.dictionary(pa.int32(), pa.string()),
        )
    )

    args = (arr0, arr1, None)
    coalesce_answer = coalesce_expected_output(args)
    check_func(
        impl, args, py_output=coalesce_answer, check_dtype=False, reset_index=True
    )


@pytest.mark.slow
def test_dict_arr_coalesce_optional():
    """Test coalesce behavior with dictionary encoded arrays and various optional types"""
    # Note: We remove a memory leak check because we leak memory when an optional scalar
    # is followed by an array that changes the expected output from dict encoding to
    # regular string array. This occurs because the cast from dict array -> string array
    # leaks memory.

    def impl(arr, scalar, flag, other):
        A = scalar if flag else None
        return pd.Series(bodo.libs.bodosql_array_kernels.coalesce((arr, A, other)))

    main_arr = pd.arrays.ArrowStringArray(
        pa.array(
            ["afa", "erwoifnewoi", "Rer", None, "مرحبا, العالم ، هذا هو بودو"] * 5,
            type=pa.dictionary(pa.int32(), pa.string()),
        )
    )
    main_scalar = "هو بودو"
    other_dict_arr = pd.arrays.ArrowStringArray(
        pa.array(
            ["a", "b", "c", "d", None] + (["a", None, "a", None] * 5),
            type=pa.dictionary(pa.int32(), pa.string()),
        )
    )
    other_regular_arr = pd.Series(
        ["a", "b", "c", "d", None] + (["a", None, "a", None] * 5)
    )
    other_scalar = "bef"
    for other in (other_dict_arr, other_regular_arr, other_scalar, None):
        for flag in [True, False]:
            args = (main_arr, main_scalar, flag, other)
            expected_args = (main_arr, main_scalar if flag else None, other)
            coalesce_answer = coalesce_expected_output(expected_args)
            check_func(
                impl,
                args,
                py_output=coalesce_answer,
                check_dtype=False,
                reset_index=True,
            )


@pytest.mark.slow
def test_option_with_arr_coalesce():
    """tests coalesce behavior with optionals when supplied an array argument"""
    # Note: We remove a memory leak check because we leak memory when an optional scalar
    # is followed by an array that changes the expected output from dict encoding to
    # regular string array. This occurs because the cast from dict array -> string array
    # leaks memory.

    def impl1(arr, scale1, scale2, flag1, flag2):
        A = scale1 if flag1 else None
        B = scale2 if flag2 else None
        return pd.Series(bodo.libs.bodosql_array_kernels.coalesce((A, arr, B)))

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
def test_option_no_arr_coalesce(memory_leak_check):
    """tests coalesce behavior with optionals when supplied no array argument"""

    def impl1(scale1, scale2, flag1, flag2):
        A = scale1 if flag1 else None
        B = scale2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.coalesce((A, B))

    scale1, scale2 = "A", "B"
    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if flag1:
                answer = "A"
            elif flag2:
                answer = "B"
            else:
                answer = None
            check_func(
                impl1,
                (scale1, scale2, flag1, flag2),
                py_output=answer,
                check_dtype=False,
                reset_index=True,
            )


@pytest.mark.slow
@pytest.mark.parametrize(
    "flags",
    [
        [True, True, True, True, True],
        [True, True, True, True, False],
        [False, True, True, False, True],
        [False, True, True, False, False],
        [True, False, True, False, True],
        [True, False, True, False, False],
        [False, False, True, True, True],
        [False, False, True, True, False],
        [True, False, True, True, True],
        [True, False, True, True, False],
        [False, False, False, True, True],
        [False, False, False, True, False],
        [False, True, False, True, True],
        [False, True, False, True, False],
    ],
)
def test_option_decode(flags, memory_leak_check):
    def impl1(A, B, C, D, E, F, flag0, flag1, flag2, flag3, flag4):
        arg0 = B if flag0 else None
        arg1 = C if flag1 else None
        arg2 = D if flag2 else None
        arg3 = E if flag3 else None
        arg4 = F if flag4 else None
        return pd.Series(
            bodo.libs.bodosql_array_kernels.decode((A, arg0, arg1, arg2, arg3, arg4))
        )

    def impl2(A, B, C, D, E, flag0, flag1, flag2, flag3):
        arg0 = B if flag0 else None
        arg1 = C if flag1 else None
        arg2 = D if flag2 else None
        arg3 = E if flag3 else None
        return pd.Series(
            bodo.libs.bodosql_array_kernels.decode((A, arg0, arg1, arg2, arg3))
        )

    def decode_scalar_fn(*args):
        for i in range(1, len(args) - 1, 2):
            if (pd.isna(args[0]) and pd.isna(args[i])) or (
                not pd.isna(args[0]) and not pd.isna(args[i]) and args[0] == args[i]
            ):
                return args[i + 1]
        if len(args) % 2 == 0:
            return args[-1]

    A = pd.Series(["A", "E", None, "I", "O", "U", None, None])
    B, C, D, E, F = "A", "a", "E", "e", "y"

    flag0, flag1, flag2, flag3, flag4 = flags

    arg0 = B if flag0 else None
    arg1 = C if flag1 else None
    arg2 = D if flag2 else None
    arg3 = E if flag3 else None
    arg4 = F if flag4 else None

    args = (A, B, C, D, E, F, flag0, flag1, flag2, flag3, flag4)
    nulled_args = (A, arg0, arg1, arg2, arg3, arg4)
    decode_answer_A = vectorized_sol(nulled_args, decode_scalar_fn, pd.StringDtype())
    check_func(
        impl1,
        args,
        py_output=decode_answer_A,
        check_dtype=False,
        reset_index=True,
        dist_test=False,
    )

    if flag4:
        args = (A, B, C, D, E, flag0, flag1, flag2, flag3)
        nulled_args = (A, arg0, arg1, arg2, arg3)
        decode_answer_B = vectorized_sol(
            nulled_args, decode_scalar_fn, pd.StringDtype()
        )
        check_func(
            impl2,
            args,
            py_output=decode_answer_B,
            check_dtype=False,
            reset_index=True,
            dist_test=False,
        )


def test_option_concat_ws(memory_leak_check):
    """
    Test calling concat_ws with optional values in tuple and an optional
    separator.
    """

    def impl(arr1, A, B, arr2, sep, flag0, flag1, flag2):
        arg0 = arr1
        arg1 = A if flag0 else None
        arg2 = B if flag1 else None
        arg3 = arr2
        arg4 = sep if flag2 else None
        return pd.Series(
            bodo.libs.bodosql_array_kernels.concat_ws((arg0, arg1, arg2, arg3), arg4)
        )

    def concat_ws_scalar_fn(*args):
        for i in range(len(args)):
            if pd.isna(args[i]):
                return None
        sep = args[-1]
        cols = list(args[:-1])
        return sep.join(cols)

    arr1 = pd.array(["cat", "dog", "cat", "lava", None] * 4)
    arr2 = pd.array([None, "elmo", "fire", "lamp", None] * 4)
    A = "flag"
    B = "window"
    sep = "-"
    for flag2 in [True, False]:
        for flag1 in [True, False]:
            for flag0 in [True, False]:
                args = (arr1, A, B, arr2, sep, flag0, flag1, flag2)
                arg0 = arr1
                arg1 = A if flag0 else None
                arg2 = B if flag1 else None
                arg3 = arr2
                arg4 = sep if flag2 else None
                concat_ws_answer = vectorized_sol(
                    (arg0, arg1, arg2, arg3, arg4),
                    concat_ws_scalar_fn,
                    pd.StringDtype(),
                )
                check_func(impl, args, py_output=concat_ws_answer, check_dtype=False)


def test_concat_ws_fusion(memory_leak_check):
    """
    Tests that multiple calls to concat_ws with the same constant separator are
    fused into a single call in Series Pass.
    """

    def impl1(arr):
        return bodo.libs.bodosql_array_kernels.concat_ws(
            (
                bodo.libs.bodosql_array_kernels.concat_ws(
                    (
                        bodo.libs.bodosql_array_kernels.concat_ws(
                            (
                                bodo.libs.bodosql_array_kernels.concat_ws(
                                    ("%", " "), ""
                                ),
                                arr,
                            ),
                            "",
                        ),
                        " ",
                    ),
                    "",
                ),
                "%",
            ),
            "",
        )

    def impl2(arr):
        return bodo.libs.bodosql_array_kernels.concat_ws(
            (
                bodo.libs.bodosql_array_kernels.concat_ws(
                    (
                        bodo.libs.bodosql_array_kernels.concat_ws(
                            (
                                bodo.libs.bodosql_array_kernels.concat_ws(
                                    ("%", " "), ""
                                ),
                                arr,
                            ),
                            # This should prevent fusion
                            "/",
                        ),
                        " ",
                    ),
                    "",
                ),
                "%",
            ),
            "",
        )

    arr = pd.Series(
        ["abc", "b", None, "abc", None, "b", "cde", "Y432^23", "R3qr32&&", "4342*"] * 10
    ).values
    expected_output1 = (
        pd.Series(arr).map(lambda x: None if pd.isna(x) else f"% {x} %").values
    )
    expected_output2 = (
        pd.Series(arr).map(lambda x: None if pd.isna(x) else f"% /{x} %").values
    )
    # First check correctness
    check_func(impl1, (arr,), py_output=expected_output1, check_dtype=False)
    check_func(impl2, (arr,), py_output=expected_output2, check_dtype=False)
    # Now check for fusion.
    bodo_func1 = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl1)
    bodo_func1(arr)
    f_ir = bodo_func1.overloads[bodo_func1.signatures[0]].metadata["preserved_ir"]
    concat_ws_calls = dist_IR_count(f_ir, "concat_ws")
    assert concat_ws_calls == 1, f"Expected 1 concat_ws call, got {concat_ws_calls}"
    # Only partial fusion should be possible because of different separators
    bodo_func2 = bodo.jit(pipeline_class=SeriesOptTestPipeline)(impl2)
    bodo_func2(arr)
    f_ir = bodo_func2.overloads[bodo_func2.signatures[0]].metadata["preserved_ir"]
    concat_ws_calls = dist_IR_count(f_ir, "concat_ws")
    assert concat_ws_calls == 3, f"Expected 3 concat_ws call, got {concat_ws_calls}"


def test_concat_ws_binary(memory_leak_check):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.concat_ws((A, B), b"_"))

    A = pd.Series([b"A", b"B", None, b"C", b"D"] * 2)
    B = pd.Series([b"E", b"F", b"G", None, b"H"] * 2)
    expected_output = pd.Series([b"A_E", b"B_F", None, None, b"D_H"] * 2)
    check_func(impl, (A, B), py_output=expected_output, check_dtype=False)


@pytest.mark.slow
@pytest.mark.parametrize(
    "args, answer",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        datetime.date(2020, 1, 1),
                        None,
                        datetime.date(2022, 12, 31),
                        None,
                        datetime.date(2024, 7, 4),
                        None,
                    ]
                ),
                pd.Series(
                    [
                        pd.Timestamp("2018-3-14"),
                        pd.Timestamp("2019-4-1"),
                        None,
                        None,
                        None,
                        pd.Timestamp("2020-10-31"),
                    ]
                ),
            ),
            pd.Series(
                [
                    pd.Timestamp("2020-1-1"),
                    pd.Timestamp("2019-4-1"),
                    pd.Timestamp("2022-12-31"),
                    None,
                    pd.Timestamp("2024-7-4"),
                    pd.Timestamp("2020-10-31"),
                ]
            ),
            id="date_col-naive_col",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        datetime.date(2022, 1, 1),
                        None,
                        datetime.date(2024, 12, 31),
                        None,
                        datetime.date(2020, 7, 4),
                        None,
                    ]
                ),
                pd.Series(
                    [
                        pd.Timestamp("2018-4-14", tz="US/Pacific"),
                        pd.Timestamp("2019-10-1", tz="US/Pacific"),
                        None,
                        None,
                        None,
                        pd.Timestamp("2020-3-31", tz="US/Pacific"),
                    ]
                ),
            ),
            pd.Series(
                [
                    pd.Timestamp("2022-1-1", tz="US/Pacific"),
                    pd.Timestamp("2019-10-1", tz="US/Pacific"),
                    pd.Timestamp("2024-12-31", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2020-7-4", tz="US/Pacific"),
                    pd.Timestamp("2020-3-31", tz="US/Pacific"),
                ]
            ),
            id="date_col-tz_col",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2015-5-5"),
                        None,
                        pd.Timestamp("2023-11-24"),
                        None,
                        pd.Timestamp("2020-6-30"),
                        None,
                    ]
                ),
                datetime.date(1999, 12, 31),
            ),
            pd.Series(
                [
                    pd.Timestamp("2015-5-5"),
                    pd.Timestamp("1999-12-31"),
                    pd.Timestamp("2023-11-24"),
                    pd.Timestamp("1999-12-31"),
                    pd.Timestamp("2020-6-30"),
                    pd.Timestamp("1999-12-31"),
                ]
            ),
            id="naive_col-date_scalar",
        ),
    ],
)
def test_coalesce_date_timestamp(args, answer, memory_leak_check):
    """Test BodoSQL COALESCE kernel on combinations of date and timestamp values"""
    n_args = len(args)
    args_str = ", ".join(f"A{i}" for i in range(n_args))
    test_impl = f"def impl({args_str}):\n"
    test_impl += (
        f"  return pd.Series(bodo.libs.bodosql_array_kernels.coalesce(({args_str},)))"
    )
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)
    impl = impl_vars["impl"]
    check_func(impl, args, py_output=answer, reset_index=True)


@pytest.mark.parametrize("func", ["least", "greatest"])
@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param((1, -2, 3), (-2, 3), id="integer-scalars"),
        pytest.param(
            (1.0, None, -1.0),
            (None, None),
            id="floats-null-scalars",
        ),
        pytest.param(
            (None, "d", ""),
            (None, None),
            id="strings-null-scalars",
        ),
        pytest.param(
            (
                pd.Series([1, 0, None, -1, -1] * 6),
                pd.Series([None, -100, 100, 2, 5] * 6),
                3,
            ),
            (
                pd.Series([None, -100, None, -1, -1] * 6),
                pd.Series([None, 3, None, 3, 5] * 6),
            ),
            id="integers-null-mix",
        ),
        pytest.param(
            (
                pd.Series(["abc", "asdf10", None, None, "00000"] * 3),
                "Ã",
                pd.Series(["@#$#@!", "`12`~", "`12`", "AAAAA", "Å"] * 3),
            ),
            (
                pd.Series(["@#$#@!", "`12`~", None, None, "00000"] * 3),
                pd.Series(["Ã", "Ã", None, None, "Å"] * 3),
            ),
            id="strings-null-mix",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2022-02-14"),
                        pd.Timestamp("2022-02-15"),
                        pd.Timestamp("2032-02-14"),
                    ]
                    * 5
                ),
                pd.Timestamp("2017-08-17"),
                pd.Series(
                    [
                        pd.Timestamp("2012-02-14"),
                        pd.Timestamp("2032-02-15"),
                        pd.Timestamp("2022-02-14"),
                    ]
                    * 5
                ),
            ),
            (
                pd.Series(
                    [
                        pd.Timestamp("2012-02-14"),
                        pd.Timestamp("2017-08-17"),
                        pd.Timestamp("2017-08-17"),
                    ]
                    * 5
                ),
                pd.Series(
                    [
                        pd.Timestamp("2022-02-14"),
                        pd.Timestamp("2032-02-15"),
                        pd.Timestamp("2032-02-14"),
                    ]
                    * 5
                ),
            ),
            id="timestamp-mix",
        ),
        pytest.param(
            (
                pd.Timestamp("2023-08-17", tz="Poland"),
                pd.Timestamp("2013-08-17", tz="Poland"),
                pd.Timestamp("2003-08-17", tz="Poland"),
            ),
            (
                pd.Timestamp("2003-08-17", tz="Poland"),
                pd.Timestamp("2023-08-17", tz="Poland"),
            ),
            id="tz-aware-timestamp-scalars",
        ),
    ],
)
def test_least_greatest(func, args, answers, request, memory_leak_check):
    """
    Tests the least and greatest variadic bodosql kernels with a mix
    of columns and scalars, with string, integer, float, timestamp data with null values.
    """
    is_scalar = "scalar" in request.node.name

    expected_output = answers[0] if func == "least" else answers[1]

    n_args = len(args)
    args_str = ", ".join(f"A{i}" for i in range(n_args))
    test_impl = f"def impl({args_str}):\n"

    if is_scalar:
        test_impl += f"  return bodo.libs.bodosql_array_kernels.{func}(({args_str}))"
    else:
        test_impl += (
            f"  return pd.Series(bodo.libs.bodosql_array_kernels.{func}(({args_str})))"
        )
    impl_vars = {}
    exec(test_impl, {"bodo": bodo, "pd": pd}, impl_vars)
    impl = impl_vars["impl"]

    check_func(
        impl,
        args,
        py_output=expected_output,
        check_dtype=False,
    )


@pytest.mark.parametrize("func", ["least", "greatest"])
@pytest.mark.parametrize(
    "args, answers",
    [
        pytest.param(
            (-100, 1000),
            (-100, 1000),
            id="integer-scalars",
        ),
        pytest.param(
            (-12.11111, 10000.0),
            (-12.11111, 10000.0),
            id="floats-scalars",
        ),
        pytest.param(
            ("abc", "1230123"),
            ("1230123", "abc"),
            id="strings-scalars",
        ),
        pytest.param(
            (pd.Timestamp("2022-08-17"), pd.Timestamp("2000-08-17")),
            (pd.Timestamp("2000-08-17"), pd.Timestamp("2022-08-17")),
            id="timestamp-scalars",
        ),
    ],
)
def test_least_greatest_optional(func, args, answers, request, memory_leak_check):
    """
    Tests the least and greatest variadic bodosql kernels with
    optional types, using string, numeric, and timestamp data.
    """

    def least_impl(arg1, arg2, flag1, flag2):
        A = arg1 if flag1 else None
        B = arg2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.least((A, B))

    def greatest_impl(arg1, arg2, flag1, flag2):
        A = arg1 if flag1 else None
        B = arg2 if flag2 else None
        return bodo.libs.bodosql_array_kernels.greatest((A, B))

    arg1, arg2 = args
    impl = least_impl if func == "least" else greatest_impl

    for flag1 in [True, False]:
        for flag2 in [True, False]:
            if not (flag1 and flag2):
                answer = None
            else:
                answer = answers[0] if func == "least" else answers[1]
            check_func(
                impl,
                (arg1, arg2, flag1, flag2),
                py_output=answer,
                check_dtype=False,
                reset_index=True,
            )


@pytest.mark.parametrize(
    "test",
    [
        pytest.param(0, id="test_a"),
        pytest.param(1, id="test_b"),
        pytest.param(2, id="test_c"),
    ],
)
def test_row_number(test, memory_leak_check):
    def impl1(df):
        return bodo.libs.bodosql_array_kernels.row_number(
            df, ["A", "B"], [True, False], ["first", "last"]
        )

    def impl2(df):
        return bodo.libs.bodosql_array_kernels.row_number(df, ["C"], [True], ["first"])

    def impl3(df):
        return bodo.libs.bodosql_array_kernels.row_number(
            df, ["B", "C"], [False, True], ["last", "last"]
        )

    df = pd.DataFrame(
        {
            "A": pd.Series([1, None, 0, 5] * 4, dtype=pd.Int32Dtype()).values,
            "B": pd.Series([2, 8, None, 4], dtype=pd.Int32Dtype()).repeat(4).values,
            "C": [str(i) for i in range(16)],
        }
    )

    res1 = pd.Series([11, 3, 7, 15, 9, 1, 5, 13, 12, 4, 8, 16, 10, 2, 6, 14])
    res2 = pd.Series([1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8])
    res3 = pd.Series([9, 10, 11, 12, 1, 2, 3, 4, 15, 16, 13, 14, 5, 6, 7, 8])

    impls = [impl1, impl2, impl3]
    results = [res1, res2, res3]

    impl = impls[test]
    res = results[test]
    check_func(
        impl,
        (df,),
        py_output=pd.DataFrame({"ROW_NUMBER": res}),
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args, scalar_tup, use_map_arrays, answer",
    [
        pytest.param(
            (
                pd.Series(
                    [1, None, 4, None, 16, None, 64, None, 256], dtype=pd.Int64Dtype()
                ),
            ),
            (False,),
            False,
            pd.Series([[1], [None], [4], [None], [16], [None], [64], [None], [256]]),
            id="integer-1",
        ),
        pytest.param(
            (
                pd.Series(
                    [1, None, 4, None, 16, None, 64, None, 256], dtype=pd.Int32Dtype()
                ),
                np.int8(-1),
            ),
            (False, True),
            False,
            pd.Series(
                [
                    [1, -1],
                    [None, -1],
                    [4, -1],
                    [None, -1],
                    [16, -1],
                    [None, -1],
                    [64, -1],
                    [None, -1],
                    [256, -1],
                ]
            ),
            id="integer-2",
        ),
        pytest.param(
            (
                None,
                pd.Series([1, 2, 1, 2, 1], dtype=pd.UInt32Dtype()),
                np.uint16(3),
                np.array([4, 5, -6, 7, 8], dtype=np.int8),
            ),
            (True, False, True, False),
            False,
            pd.Series(
                [
                    [None, 1, 3, 4],
                    [None, 2, 3, 5],
                    [None, 1, 3, -6],
                    [None, 2, 3, 7],
                    [None, 1, 3, 8],
                ]
            ),
            id="integer-4",
        ),
        pytest.param(
            ("",),
            (True,),
            False,
            pd.array([""]),
            id="string_scalar-1",
        ),
        pytest.param(
            (pd.Series(["", None, "A", None, "AB", None, "ABC"] * 3),),
            (False,),
            False,
            pd.Series([[""], [None], ["A"], [None], ["AB"], [None], ["ABC"]] * 3),
            id="string-1",
        ),
        pytest.param(
            (pd.Series(["", None, "A", None, "AB", None, "ABC"] * 3), "foo"),
            (False, True),
            False,
            pd.Series(
                [
                    ["", "foo"],
                    [None, "foo"],
                    ["A", "foo"],
                    [None, "foo"],
                    ["AB", "foo"],
                    [None, "foo"],
                    ["ABC", "foo"],
                ]
                * 3
            ),
            id="string-2",
        ),
        pytest.param(
            (
                "K",
                pd.Series(["1", None, "2", None, "3"]),
                None,
                pd.Series(["", "A", "BC", "DEF", "GHIJ"]),
            ),
            (True, False, True, False),
            False,
            pd.Series(
                [
                    ["K", "1", None, ""],
                    ["K", None, None, "A"],
                    ["K", "2", None, "BC"],
                    ["K", None, None, "DEF"],
                    ["K", "3", None, "GHIJ"],
                ]
            ),
            id="string-4",
        ),
        pytest.param(
            (
                b"1",
                pd.Series([b"23", None, b"456", None, b""] * 2),
            ),
            (True, False),
            False,
            pd.Series(
                [[b"1", b"23"], [b"1", None], [b"1", b"456"], [b"1", None], [b"1", b""]]
                * 2
            ),
            id="binary-2",
            marks=pytest.mark.skip(
                reason="[BSE-1776] TODO: support array_construct when inputs are binary"
            ),
        ),
        pytest.param(
            (
                np.array([True, False, True, False] * 2, dtype=np.bool_),
                pd.Series([True, False, None, True] * 2, dtype=pd.BooleanDtype()),
            ),
            (False, False),
            False,
            pd.Series([[True, True], [False, False], [True, None], [False, True]] * 2),
            id="boolean-2",
        ),
        pytest.param(
            (
                np.float64(3.14),
                pd.Series([-1.0, 0.0, None, 2.71828] * 2, dtype=pd.Float32Dtype()),
            ),
            (True, False),
            False,
            pd.Series([[3.14, -1.0], [3.14, 0.0], [3.14, None], [3.14, 2.71828]] * 2),
            id="float-2",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        Decimal("0.0"),
                        None,
                        Decimal("-1024.2048"),
                        None,
                        Decimal("1.23456789"),
                    ]
                ),
                Decimal("0.025"),
            ),
            (False, True),
            False,
            pd.Series(
                [
                    [Decimal("0.0"), Decimal("0.025")],
                    [None, Decimal("0.025")],
                    [Decimal("-1024.2048"), Decimal("0.025")],
                    [None, Decimal("0.025")],
                    [Decimal("1.23456789"), Decimal("0.025")],
                ]
            ),
            id="decimal-2",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        Decimal("0.0"),
                        None,
                        Decimal("-1024.2048"),
                        None,
                        Decimal("1.23456789"),
                    ]
                ),
                np.int64(0),
            ),
            (False, True),
            False,
            pd.Series(
                [
                    [Decimal("0.0"), Decimal("0.0")],
                    [None, Decimal("0.0")],
                    [Decimal("-1024.2048"), Decimal("0.0")],
                    [None, Decimal("0.0")],
                    [Decimal("1.23456789"), Decimal("0.0")],
                ]
            ),
            id="decimal_int-2-upcasting",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        Decimal("0.0"),
                        None,
                        Decimal("-1024.2048"),
                        None,
                        Decimal("1.23456789"),
                    ]
                ),
                np.float64(3.1415),
            ),
            (False, True),
            False,
            pd.Series(
                [
                    [0.0, 3.1415],
                    [None, 3.1415],
                    [-1024.2048, 3.1415],
                    [None, 3.1415],
                    [1.23456789, 3.1415],
                ]
            ),
            id="decimal_float-2-upcasting",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        datetime.date(2023, 10, 25),
                        None,
                        datetime.date(1999, 12, 31),
                        None,
                        datetime.date(2008, 4, 1),
                    ]
                ),
                datetime.date(1999, 12, 31),
            ),
            (False, True),
            False,
            pd.Series(
                [
                    [datetime.date(2023, 10, 25), datetime.date(1999, 12, 31)],
                    [None, datetime.date(1999, 12, 31)],
                    [datetime.date(1999, 12, 31), datetime.date(1999, 12, 31)],
                    [None, datetime.date(1999, 12, 31)],
                    [datetime.date(2008, 4, 1), datetime.date(1999, 12, 31)],
                ]
            ),
            id="date-2",
        ),
        pytest.param(
            (
                pd.Timestamp("2015-3-14 9:26:53.59"),
                pd.Series(
                    [
                        pd.Timestamp("2023-1-1"),
                        None,
                        pd.Timestamp("1999-12-31 23:59:59.999250"),
                        None,
                        pd.Timestamp("2024-7-4 6:30:00"),
                    ]
                ),
            ),
            (True, False),
            False,
            pd.Series(
                [
                    [pd.Timestamp("2015-3-14 9:26:53.59"), pd.Timestamp("2023-1-1")],
                    [pd.Timestamp("2015-3-14 9:26:53.59"), None],
                    [
                        pd.Timestamp("2015-3-14 9:26:53.59"),
                        pd.Timestamp("1999-12-31 23:59:59.999250"),
                    ],
                    [pd.Timestamp("2015-3-14 9:26:53.59"), None],
                    [
                        pd.Timestamp("2015-3-14 9:26:53.59"),
                        pd.Timestamp("2024-7-4 6:30:00"),
                    ],
                ]
            ),
            id="timestamp_ntz-2",
        ),
        pytest.param(
            (
                pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"),
                pd.Series(
                    [
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                        None,
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                        None,
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                    ]
                ),
            ),
            (True, False),
            False,
            pd.Series(
                [
                    [
                        pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"),
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                    ],
                    [pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"), None],
                    [
                        pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"),
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                    ],
                    [pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"), None],
                    [
                        pd.Timestamp("2015-3-14 9:26:53.59", tz="US/Pacific"),
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                    ],
                ]
            ),
            id="timestamp_ltz-2",
            marks=pytest.mark.skip(
                reason="[BSE-1777] TODO: fix array_construct when inputs are tz-aware timestamps, specifically when mixing vectors with scalars"
            ),
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2024-7-4 6:30:00"),
                        None,
                        pd.Timestamp("1999-12-31 23:59:59.999250"),
                        pd.Timestamp("2023-1-1"),
                        None,
                    ]
                ),
                pd.Series(
                    [
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                        None,
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                        None,
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                    ]
                ),
            ),
            (False, False),
            False,
            pd.Series(
                [
                    [
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                    ],
                    [None, None],
                    [
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                    ],
                    [None, None],
                    [
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                    ],
                ]
            ),
            id="timestamp_mixed-2",
            marks=pytest.mark.skip(
                reason="[BSE-1778] TODO: fix array_construct when inputs are tz-naive and tz-aware timestamps"
            ),
        ),
        pytest.param(
            (pd.Series([[1], [2, None], [], None, [5, 6], []] * 3),),
            (False,),
            False,
            pd.Series([[[1]], [[2, None]], [[]], [None], [[5, 6]], [[]]] * 3),
            id="nested_integer-1",
        ),
        pytest.param(
            (pd.Series([["A"], ["BC", None], [], None, ["DEF", "", "GH"], []] * 3),),
            (False,),
            False,
            pd.Series(
                [[["A"]], [["BC", None]], [[]], [None], [["DEF", "", "GH"]], [[]]] * 3
            ),
            id="nested_string-1",
        ),
        pytest.param(
            (
                pd.Series([[1], [2, None], [], None, [5, 6], []] * 3),
                pd.Series([[], [], [7], [8], [None, 10], [11, 12]] * 3),
            ),
            (False, False),
            False,
            pd.Series(
                [
                    [[1], []],
                    [[2, None], []],
                    [[], [7]],
                    [None, [8]],
                    [[5, 6], [None, 10]],
                    [[], [11, 12]],
                ]
                * 3
            ),
            id="nested_integer-2",
            marks=pytest.mark.skip(
                reason="[BSE-1780] TODO: fix array_construct when inputs are multiple arrays"
            ),
        ),
        pytest.param(
            (
                pd.Series([["A"], ["BC", None], [], None, ["DEF", "", "GH"], []] * 3),
                pd.Series([["A", "BC"], [], ["DEF", ""], [None], None, ["GH"]] * 3),
            ),
            (False, False),
            False,
            pd.Series(
                [
                    [["A"], ["A", "BC"]],
                    [["BC", None], []],
                    [[], ["DEF", ""]],
                    [None, [None]],
                    [["DEF", "", "GH"], None],
                    [[], ["GH"]],
                ]
                * 3
            ),
            id="nested_string-2",
            marks=pytest.mark.skip(
                reason="[BSE-1780] TODO: fix array_construct when inputs are multiple arrays"
            ),
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"X": 1, "Y": 3.1},
                        {"X": 3, "Y": -1.3},
                        {"X": 4, "Y": 0.4},
                    ]
                    * 10
                    + [
                        {"X": -2, "Y": 2.2},
                        None,
                    ]
                ),
                np.array(
                    [
                        {"X": 5, "Y": 10.1},
                        {"X": 7, "Y": 10.3},
                        {"X": -8, "Y": 10.4},
                    ]
                    * 10
                    + [
                        None,
                        {"X": None, "Y": 10.2},
                    ]
                ),
            ),
            (False, False),
            False,
            pd.Series(
                [
                    [{"X": 1, "Y": 3.1}, {"X": 5, "Y": 10.1}],
                    [{"X": 3, "Y": -1.3}, {"X": 7, "Y": 10.3}],
                    [{"X": 4, "Y": 0.4}, {"X": -8, "Y": 10.4}],
                ]
                * 10
                + [[{"X": -2, "Y": 2.2}, None], [None, {"X": None, "Y": 10.2}]]
            ),
            id="struct_array-int_float-2",
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"name": "Daenerys", "house": "Targaryen"},
                        {"name": "Sansa", "house": "Stark"},
                        {"name": "Tyrion", "house": "Lannister"},
                        {"name": "Arya", "house": "Stark"},
                    ]
                    * 3
                    + [None]
                ),
                None,
                None,
                np.array(
                    [
                        {"name": "Jaime", "house": "Lannister"},
                        {"name": "Olenna", "house": "Tyrell"},
                        {"name": "Jaime", "house": "Lannister"},
                        {"name": "Olenna", "house": "Tyrell"},
                    ]
                    * 3
                    + [{"name": "Jon", "house": None}]
                ),
            ),
            (False, True, True, False),
            False,
            pd.Series(
                [
                    [
                        {"name": "Daenerys", "house": "Targaryen"},
                        None,
                        None,
                        {"name": "Jaime", "house": "Lannister"},
                    ],
                    [
                        {"name": "Sansa", "house": "Stark"},
                        None,
                        None,
                        {"name": "Olenna", "house": "Tyrell"},
                    ],
                    [
                        {"name": "Tyrion", "house": "Lannister"},
                        None,
                        None,
                        {"name": "Jaime", "house": "Lannister"},
                    ],
                    [
                        {"name": "Arya", "house": "Stark"},
                        None,
                        None,
                        {"name": "Olenna", "house": "Tyrell"},
                    ],
                ]
                * 3
                + [[None, None, None, {"name": "Jon", "house": None}]]
            ),
            id="struct_array_without-scalars-strings-4",
        ),
        pytest.param(
            (
                np.array(
                    [
                        {"name": "Daenerys", "house": "Targaryen"},
                        {"name": "Sansa", "house": "Stark"},
                        None,
                        {"name": "Tyrion", "house": "Lannister"},
                        {"name": "Arya", "house": "Stark"},
                    ]
                    * 3
                ),
                None,
                {"name": "Cersei", "house": "Lannister"},
                np.array(
                    [
                        {"name": "Bran", "house": "Stark"},
                        {"name": "Jaime", "house": "Lannister"},
                        None,
                        {"name": "Olenna", "house": "Tyrell"},
                        None,
                    ]
                    * 3
                ),
            ),
            (False, True, True, False),
            False,
            pd.Series(
                [
                    [
                        {"name": "Daenerys", "house": "Targaryen"},
                        None,
                        {"name": "Cersei", "house": "Lannister"},
                        {"name": "Bran", "house": "Stark"},
                    ],
                    [
                        {"name": "Sansa", "house": "Stark"},
                        None,
                        {"name": "Cersei", "house": "Lannister"},
                        {"name": "Jaime", "house": "Lannister"},
                    ],
                    [None, None, {"name": "Cersei", "house": "Lannister"}, None],
                    [
                        {"name": "Tyrion", "house": "Lannister"},
                        None,
                        {"name": "Cersei", "house": "Lannister"},
                        {"name": "Olenna", "house": "Tyrell"},
                    ],
                    [
                        {"name": "Arya", "house": "Stark"},
                        None,
                        {"name": "Cersei", "house": "Lannister"},
                        None,
                    ],
                ]
                * 3
            ),
            id="struct_array_with_scalars-strings-4",
            marks=pytest.mark.skip(
                reason="[BSE-1781] TODO: fix array_construct and array_position when inputs are mix of struct arrays and scalars"
            ),
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {"O": "A"},
                        {"P": "B", "Q": "C", "R": "D"},
                        None,
                        {},
                        {"S": "", "T": "EFGH", "U": "IJ"},
                    ]
                    * 3
                ).values,
            ),
            (False,),
            True,
            pd.Series(
                [
                    [{"O": "A"}],
                    [{"P": "B", "Q": "C", "R": "D"}],
                    [None],
                    [{}],
                    [{"S": "", "T": "EFGH", "U": "IJ"}],
                ]
                * 3
            ),
            id="map_array-string-1",
            marks=pytest.mark.skip(
                reason="[BSE-1782] TODO: fix array_construct when inputs are map arrays with simple keys"
            ),
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {0: 9},
                        {1: 10, 2: 7, 3: 5},
                        None,
                        {},
                        {4: 6, 5: 6},
                    ]
                    * 3
                ),
                pd.Series(
                    [
                        {6: 8},
                        None,
                        {7: 8, 8: 6, 9: 5},
                        {10: 7},
                        {},
                    ]
                    * 3
                ),
            ),
            (False, False),
            True,
            pd.Series(
                [
                    [{0: 9}, {6: 8}],
                    [{1: 10, 2: 7, 3: 5}, None],
                    [None, {7: 8, 8: 6, 9: 5}],
                    [{}, {10: 7}],
                    [{4: 6, 5: 6}, {}],
                ]
                * 3
            ),
            id="map_array-int_int-2",
            marks=pytest.mark.skip(
                reason="[BSE-1782] TODO: fix array_construct when inputs are map arrays with simple keys"
            ),
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {b"nyc": 9},
                        {b"sf": 10, b"la": 7, b"sac": 5},
                        None,
                        {},
                        {b"pit": 6, b"phi": 6},
                    ]
                    * 3
                ),
                pd.Series(
                    [
                        {b"chi": 8},
                        None,
                        {b"aus": 8, b"dal": 6, b"hou": 5},
                        {b"atl": 7},
                        {},
                    ]
                    * 3
                ),
            ),
            (False, False),
            True,
            pd.Series(
                [
                    [{b"nyc": 9}, {b"chi": 8}],
                    [{b"sf": 10, b"la": 7, b"sac": 5}, None],
                    [None, {b"aus": 8, b"dal": 6, b"hou": 5}],
                    [{}, {b"atl": 7}],
                    [{b"pit": 6, b"phi": 6}, {}],
                ]
                * 3
            ),
            id="map_array-binary_int-2",
            marks=pytest.mark.skip(
                reason="[BSE-1783] TODO: fix array_construct when inputs are map arrays with binary keys"
            ),
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {0: 9},
                        {1: 10, 2: 7, 3: 5},
                        None,
                        {},
                        {4: 6, 5: 6},
                    ]
                    * 3
                ),
                {0: 1, 2: 3},
            ),
            (False, True),
            True,
            pd.Series(
                [
                    [{0: 9}, {0: 1, 2: 3}],
                    [{1: 10, 2: 7, 3: 5}, {0: 1, 2: 3}],
                    [None, {0: 1, 2: 3}],
                    [{}, {0: 1, 2: 3}],
                    [{4: 6, 5: 6}, {0: 1, 2: 3}],
                ]
                * 3
            ),
            id="map_array_with_scalars-int_int-2",
            marks=pytest.mark.skip(
                reason="[BSE-1782] TODO: fix array_construct when inputs are map arrays with simple keys"
            ),
        ),
    ],
)
def test_array_construct(args, scalar_tup, use_map_arrays, answer, memory_leak_check):
    def impl_scalar(A):
        return bodo.libs.bodosql_array_kernels.array_construct((A,), scalar_tup)

    def impl1(A):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.array_construct((A,), scalar_tup)
        )

    def impl2(A, B):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.array_construct((A, B), scalar_tup)
        )

    def impl4(A, B, C, D):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.array_construct((A, B, C, D), scalar_tup)
        )

    if len(args) == 1 and not isinstance(args[0], pd.Series):
        impl = impl_scalar
        is_out_distributed = False
    else:
        implementations = {
            1: impl1,
            2: impl2,
            4: impl4,
        }
        impl = implementations[len(args)]
        is_out_distributed = None

    check_func(
        impl,
        args,
        is_out_distributed=is_out_distributed,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
        use_map_arrays=use_map_arrays,
    )


@pytest.mark.parametrize(
    "is_none_0, is_none_1",
    [
        pytest.param(False, False, id="scalar-scalar"),
        pytest.param(True, False, id="null-scalar"),
        pytest.param(False, True, id="scalar-null"),
        pytest.param(True, True, id="null-null"),
    ],
)
def test_array_construct_optional(is_none_0, is_none_1, memory_leak_check):
    def impl(A, B, is_none_0, is_none_1):
        arg0 = None if is_none_0 else A
        arg1 = None if is_none_1 else B
        return bodo.libs.bodosql_array_kernels.array_construct(
            (arg0, arg1), (True, True)
        )

    answer = pd.array([None if is_none_0 else "A", None if is_none_1 else "B"])

    check_func(
        impl,
        ("A", "B", is_none_0, is_none_1),
        py_output=answer,
        check_dtype=False,
        dist_test=False,
    )
