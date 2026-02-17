"""Test Bodo's array kernel utilities for BodoSQL variadic functions"""

import datetime
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numba.core import types

import bodo
import bodosql
from bodo.tests.utils import (
    check_func,
    dist_IR_count,
    find_nested_dispatcher_and_args,
    pytest_slow_unless_codegen,
)
from bodo.tests.utils_jit import SeriesOptTestPipeline
from bodo.utils.typing import BodoError, ColNamesMetaType, MetaType
from bodosql.kernels.array_kernel_utils import vectorized_sol

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
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            bodo.types.TimestampTZ.fromLocal("2024-03-06 12:00:00", 0),
                            None,
                            bodo.types.TimestampTZ.fromLocal("2024-03-06 13:00:00", 60),
                            None,
                            bodo.types.TimestampTZ.fromLocal(
                                "2024-03-06 14:00:00", -300
                            ),
                            None,
                            bodo.types.TimestampTZ.fromLocal(
                                "2024-03-06 15:00:00", -45
                            ),
                            None,
                        ]
                    )
                ),
                pd.Series(
                    pd.array(
                        [
                            bodo.types.TimestampTZ.fromLocal("2024-07-04", 30),
                            bodo.types.TimestampTZ.fromLocal("2024-07-04", 0),
                            None,
                            None,
                            bodo.types.TimestampTZ.fromLocal("2024-07-04", -30),
                            bodo.types.TimestampTZ.fromLocal("2024-07-04", 90),
                            None,
                            None,
                        ]
                    )
                ),
            ),
            id="timestamp_tz_vector_2",
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
    test_impl += f"  return {series_str}(bodosql.kernels.coalesce(({args_str},)))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)
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
    test_impl += f"  return {series_str}(bodosql.kernels.coalesce(({args_str},)))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)
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
        bodo_func, arg_typs, ("coalesce", "bodosql.kernels")
    )
    # Note: infrastructure doesn't handle defaults, so we need to manually add this to
    # the signature
    used_sig = used_sig + (types.none, types.int64)
    # Find the coalesce_util dispatcher in the IR
    dispatcher, used_sig = find_nested_dispatcher_and_args(
        dispatcher,
        used_sig,
        ("coalesce_util", "bodosql.kernels.variadic_array_kernels"),
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
                        *pd.date_range(
                            "2018-01-01", "2018-06-02", freq="ME", unit="ns"
                        ),
                        None,
                        None,
                        *pd.date_range(
                            "2018-03-01", "2018-07-02", freq="ME", unit="ns"
                        ),
                    ]
                    * 2,
                    dtype="datetime64[ns]",
                ),
                np.datetime64(pd.Timestamp("2018-02-28"), "ns"),
                np.datetime64(pd.Timestamp("2018-02-01"), "ns"),
                np.datetime64(pd.Timestamp("2018-03-31"), "ns"),
                np.datetime64(pd.Timestamp("2018-03-01"), "ns"),
                np.datetime64(pd.Timestamp("2018-05-30"), "ns"),
                np.datetime64(pd.Timestamp("2018-05-01"), "ns"),
                pd.Series(
                    pd.date_range("2018", "2019-11-01", freq="ME", unit="ns"),
                    dtype="datetime64[ns]",
                ),
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
        pytest.param(
            (
                pd.Series(["A", None, "C", None, "E"]),
                pd.Series(["E", None, None, "C", "E"]),
                np.array(
                    [
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-07 10:13:42", -300),
                        bodo.types.TimestampTZ.fromLocal("2024-03-07 10:14:00", -360),
                        bodo.types.TimestampTZ.fromLocal("2024-03-07 10:15:00", -240),
                        None,
                    ]
                ),
                "A",
                bodo.types.TimestampTZ.fromLocal("2024-03-07 10:13:42", -300),
            ),
            id="all_vector_timestamp_tz",
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
    test_impl += f"  return {series_str}(bodosql.kernels.decode(({args_str})))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)
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
        return pd.Series(bodosql.kernels.coalesce((arr0, arr1, scalar)))

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
        return pd.Series(bodosql.kernels.coalesce((arr, A, other)))

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
        return pd.Series(bodosql.kernels.coalesce((A, arr, B)))

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
        return bodosql.kernels.coalesce((A, B))

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
        return pd.Series(bodosql.kernels.decode((A, arg0, arg1, arg2, arg3, arg4)))

    def impl2(A, B, C, D, E, flag0, flag1, flag2, flag3):
        arg0 = B if flag0 else None
        arg1 = C if flag1 else None
        arg2 = D if flag2 else None
        arg3 = E if flag3 else None
        return pd.Series(bodosql.kernels.decode((A, arg0, arg1, arg2, arg3)))

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


@pytest.mark.parametrize("flag0", [True, False])
@pytest.mark.parametrize("flag1", [True, False])
@pytest.mark.parametrize("flag2", [True, False])
@pytest.mark.slow
def test_option_concat_ws(flag0, flag1, flag2, memory_leak_check):
    """
    Test calling concat_ws with optional values in tuple and an optional separator.
    """

    def impl(arr1, A, B, arr2, sep, flag0, flag1, flag2):
        arg0 = arr1
        arg1 = A if flag0 else None
        arg2 = B if flag1 else None
        arg3 = arr2
        arg4 = sep if flag2 else None
        return pd.Series(bodosql.kernels.concat_ws((arg0, arg1, arg2, arg3), arg4))

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
        return bodosql.kernels.concat_ws(
            (
                bodosql.kernels.concat_ws(
                    (
                        bodosql.kernels.concat_ws(
                            (
                                bodosql.kernels.concat_ws(("%", " "), ""),
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
        return bodosql.kernels.concat_ws(
            (
                bodosql.kernels.concat_ws(
                    (
                        bodosql.kernels.concat_ws(
                            (
                                bodosql.kernels.concat_ws(("%", " "), ""),
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
        return pd.Series(bodosql.kernels.concat_ws((A, B), b"_"))

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
                    ],
                    dtype="datetime64[ns]",
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
                ],
                dtype="datetime64[ns]",
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
                    ],
                    dtype="datetime64[ns, US/Pacific]",
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
                ],
                dtype="datetime64[ns, US/Pacific]",
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
                    ],
                    dtype="datetime64[ns]",
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
                ],
                dtype="datetime64[ns]",
            ),
            id="naive_col-date_scalar",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        pd.Timestamp("2018-4-14", tz="US/Pacific"),
                        pd.Timestamp("2019-10-1", tz="US/Pacific"),
                        None,
                        None,
                        None,
                        pd.Timestamp("2020-3-31", tz="US/Pacific"),
                    ],
                    dtype="datetime64[ns, US/Pacific]",
                ),
                pd.Timestamp("2020-01-01", tz="US/Pacific"),
            ),
            pd.Series(
                [
                    pd.Timestamp("2018-4-14", tz="US/Pacific"),
                    pd.Timestamp("2019-10-1", tz="US/Pacific"),
                    pd.Timestamp("2020-01-01", tz="US/Pacific"),
                    pd.Timestamp("2020-01-01", tz="US/Pacific"),
                    pd.Timestamp("2020-01-01", tz="US/Pacific"),
                    pd.Timestamp("2020-3-31", tz="US/Pacific"),
                ],
                dtype="datetime64[ns, US/Pacific]",
            ),
            id="pd_datetime_tz_col-ts_tz_col",
        ),
    ],
)
def test_coalesce_date_timestamp(args, answer, memory_leak_check):
    """Test BodoSQL COALESCE kernel on combinations of date and timestamp values"""
    n_args = len(args)
    args_str = ", ".join(f"A{i}" for i in range(n_args))
    test_impl = f"def impl({args_str}):\n"
    test_impl += f"  return pd.Series(bodosql.kernels.coalesce(({args_str},)))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)
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
                    * 5,
                    dtype="datetime64[ns]",
                ),
                pd.Timestamp("2017-08-17"),
                pd.Series(
                    [
                        pd.Timestamp("2012-02-14"),
                        pd.Timestamp("2032-02-15"),
                        pd.Timestamp("2022-02-14"),
                    ]
                    * 5,
                    dtype="datetime64[ns]",
                ),
            ),
            (
                pd.Series(
                    [
                        pd.Timestamp("2012-02-14"),
                        pd.Timestamp("2017-08-17"),
                        pd.Timestamp("2017-08-17"),
                    ]
                    * 5,
                    dtype="datetime64[ns]",
                ),
                pd.Series(
                    [
                        pd.Timestamp("2022-02-14"),
                        pd.Timestamp("2032-02-15"),
                        pd.Timestamp("2032-02-14"),
                    ]
                    * 5,
                    dtype="datetime64[ns]",
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
        pytest.param(
            (
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 0),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", -60),
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 30),
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 60),
                    ]
                ),
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", -60),
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 0),
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 300),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 480),
                    ]
                ),
            ),
            (
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 0),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 300),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 480),
                    ]
                ),
                pd.Series(
                    [
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", -60),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", -60),
                        None,
                        bodo.types.TimestampTZ.fromLocal("2024-03-14", 60),
                    ]
                ),
            ),
            id="timestamp_tz",
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
        test_impl += f"  return bodosql.kernels.{func}(({args_str}))"
    else:
        test_impl += f"  return pd.Series(bodosql.kernels.{func}(({args_str})))"
    impl_vars = {}
    exec(test_impl, {"bodosql": bodosql, "pd": pd}, impl_vars)
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
        return bodosql.kernels.least((A, B))

    def greatest_impl(arg1, arg2, flag1, flag2):
        A = arg1 if flag1 else None
        B = arg2 if flag2 else None
        return bodosql.kernels.greatest((A, B))

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
        return bodosql.kernels.row_number(
            df, ["A", "B"], [True, False], ["first", "last"]
        )

    def impl2(df):
        return bodosql.kernels.row_number(df, ["C"], [True], ["first"])

    def impl3(df):
        return bodosql.kernels.row_number(
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
    "keep_mode, keys_to_filter, json_data, scalars, answer",
    [
        pytest.param(
            True,
            ("A",),
            pd.array(
                [
                    {"A": 0, "B": 10, "C": ""},
                    {"A": 1, "B": 20, "C": "A"},
                    {"A": 2, "B": 40, "C": "AB"},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": 80, "C": "ABC"},
                    {"A": 4, "B": None, "C": "A"},
                    {"A": 5, "B": 320, "C": None},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.int32()),
                            pa.field("B", pa.int32()),
                            pa.field("C", pa.string()),
                        ]
                    )
                ),
            ),
            (False, True),
            pd.array(
                [{"A": 0}, {"A": 1}, {"A": 2}] * 10
                + [None, {"A": None}, {"A": 4}, {"A": 5}],
                dtype=pd.ArrowDtype(pa.struct([pa.field("A", pa.int32())])),
            ),
            id="struct-keep_literal_string",
        ),
        pytest.param(
            True,
            ("A",),
            pd.array(
                [
                    {"A": 0, "B": 1, "C": 2},
                    {"B": 3, "C": 4},
                    {"A": 5, "C": 6},
                    {"A": 7, "B": 8},
                    {"A": 9},
                    {"B": 10},
                    {"C": 11},
                ]
                + [
                    None,
                    {"A": None, "B": None},
                    {},
                    {"A": 16, "B": None},
                    {"B": 42},
                    {"A": 42},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            (False, True),
            pd.array(
                [
                    {"A": 0},
                    {},
                    {"A": 5},
                    {"A": 7},
                    {"A": 9},
                    {},
                    {},
                ]
                + [
                    None,
                    {"A": None},
                    {},
                    {"A": 16},
                    {},
                    {"A": 42},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            id="map-keep_literal_string",
        ),
        pytest.param(
            False,
            ("A",),
            pd.Series(
                [
                    {"A": 0, "B": 10, "C": ""},
                    {"A": 1, "B": 20, "C": "A"},
                    {"A": 2, "B": 40, "C": "AB"},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": 80, "C": "ABC"},
                    {"A": 4, "B": None, "C": "A"},
                    {"A": 5, "B": 320, "C": None},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.int32()),
                            pa.field("B", pa.int32()),
                            pa.field("C", pa.string()),
                        ]
                    )
                ),
            ).values,
            (False, True),
            pd.Series(
                [
                    {"B": 10, "C": ""},
                    {"B": 20, "C": "A"},
                    {"B": 40, "C": "AB"},
                ]
                * 10
                + [
                    None,
                    {"B": 80, "C": "ABC"},
                    {"B": None, "C": "A"},
                    {"B": 320, "C": None},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct([pa.field("B", pa.int32()), pa.field("C", pa.string())])
                ),
            ).values,
            id="struct-drop_literal_string",
        ),
        pytest.param(
            False,
            ("D", "E", "a", "b,c"),
            pd.Series(
                [
                    {"A": 0, "B": 10, "C": ""},
                    {"A": 1, "B": 20, "C": "A"},
                    {"A": 2, "B": 40, "C": "AB"},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": 80, "C": "ABC"},
                    {"A": 4, "B": None, "C": "A"},
                    {"A": 5, "B": 320, "C": None},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.int32()),
                            pa.field("B", pa.int32()),
                            pa.field("C", pa.string()),
                        ]
                    )
                ),
            ).values,
            (False, True, True, True, True),
            pd.Series(
                [
                    {"A": 0, "B": 10, "C": ""},
                    {"A": 1, "B": 20, "C": "A"},
                    {"A": 2, "B": 40, "C": "AB"},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": 80, "C": "ABC"},
                    {"A": 4, "B": None, "C": "A"},
                    {"A": 5, "B": 320, "C": None},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("A", pa.int32()),
                            pa.field("B", pa.int32()),
                            pa.field("C", pa.string()),
                        ]
                    )
                ),
            ).values,
            id="struct-drop_nothing",
        ),
        pytest.param(
            False,
            ("A",),
            pd.Series(
                [
                    {"A": 0, "B": 1, "C": 2},
                    {"B": 3, "C": 4},
                    {"A": 5, "C": 6},
                    {"A": 7, "B": 8},
                    {"A": 9},
                    {"B": 10},
                    {"C": 11},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": None},
                    {},
                    {"A": 16, "B": None},
                    {"B": 42},
                    {"A": 42},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ).values,
            (False, True),
            pd.Series(
                [
                    {"B": 1, "C": 2},
                    {"B": 3, "C": 4},
                    {"C": 6},
                    {"B": 8},
                    {},
                    {"B": 10},
                    {"C": 11},
                ]
                * 10
                + [None, {"B": None}, {}, {"B": None}, {"B": 42}, {}],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ).values,
            id="map-drop_literal_string",
        ),
        pytest.param(
            False,
            (pd.Series(["A", "B"] * 43),),
            pd.Series(
                [
                    {"A": 0, "B": 1, "C": 2},
                    {"B": 3, "C": 4},
                    {"A": 5, "C": 6},
                    {"A": 7, "B": 8},
                    {"A": 9},
                    {"B": 10},
                    {"C": 11},
                    {"C": 12, "A": 13, "B": 14},
                ]
                * 10
                + [
                    None,
                    {"A": None, "B": None},
                    {},
                    {"A": 16, "B": None},
                    {"B": 42},
                    {"A": 42},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ).values,
            (False, False),
            pd.Series(
                [
                    {"B": 1, "C": 2},
                    {"C": 4},
                    {"C": 6},
                    {"A": 7},
                    {},
                    {},
                    {"C": 11},
                    {"C": 12, "A": 13},
                ]
                * 10
                + [
                    None,
                    {"A": None},
                    {},
                    {"A": 16},
                    {"B": 42},
                    {"A": 42},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="map-drop_column",
        ),
        pytest.param(
            False,
            (pd.Series(["A"] * 10),),
            pd.Series(
                [{"A": 0}] * 10,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ).values,
            (False, False),
            pd.Series(
                [{}] * 10,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="map-drop_all",
        ),
        pytest.param(
            True,
            (pd.Series(["B"] * 10),),
            pd.Series(
                [{"A": 0}] * 10,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ).values,
            (False, False),
            pd.Series(
                [{}] * 10,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
            ),
            id="map-keep_none",
        ),
        pytest.param(
            False,
            ("A",),
            pd.Series(
                [{"A": 0}] * 10,
                dtype=pd.ArrowDtype(pa.struct([pa.field("A", pa.int32())])),
            ).values,
            (False, True),
            pd.Series(
                [{}] * 10,
                dtype=pd.ArrowDtype(pa.struct([])),
            ),
            id="struct-drop_all",
        ),
        pytest.param(
            True,
            ("B",),
            pd.Series(
                [{"A": 0}] * 10,
                dtype=pd.ArrowDtype(pa.struct([pa.field("A", pa.int32())])),
            ).values,
            (False, True),
            pd.Series(
                [{}] * 10,
                dtype=pd.ArrowDtype(pa.struct([])),
            ),
            id="struct-keep_none",
        ),
    ],
)
def test_object_filter_keys(
    keep_mode, keys_to_filter, scalars, json_data, answer, memory_leak_check
):
    raw_arg_text = ", ".join(f"arg{i}" for i in range(len(keys_to_filter) + 1))
    arg_text = ["arg0"]
    for i in range(len(keys_to_filter)):
        if isinstance(keys_to_filter[i], str):
            arg_text.append(repr(keys_to_filter[i]))
        else:
            arg_text.append(f"arg{i + 1}")
    func_text = f"def impl({raw_arg_text}):\n"
    func_text += f"   res = bodosql.kernels.object_filter_keys(({', '.join(arg_text)},), {keep_mode}, scalars)\n"
    if any(isinstance(arg, pd.Series) for arg in keys_to_filter + (json_data, answer)):
        func_text += "   res = pd.Series(res)\n"
    func_text += "   return res\n"
    loc_vars = {}
    exec(
        func_text,
        {"bodosql": bodosql, "pd": pd, "scalars": MetaType(scalars)},
        loc_vars,
    )
    impl = loc_vars["impl"]

    check_func(
        impl,
        (json_data, *keys_to_filter),
        py_output=answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "values, keys, scalars, answer",
    [
        pytest.param(
            (pd.Series([1, 2, 3, None, 5, 6, 7]),),
            ("i",),
            (False,),
            pd.Series([{"i": i} for i in [1, 2, 3, None, 5, 6, 7]]),
            id="1-int_vector",
        ),
        pytest.param(
            (
                "John",
                pd.Series(["Smith", "Jones", "Stewart", None, "Irons"]).values,
            ),
            ("first", "last"),
            (True, False),
            pd.Series(
                [
                    {"first": "John", "last": surname}
                    for surname in ["Smith", "Jones", "Stewart", None, "Irons"]
                ]
            ),
            id="2-string_scalar-string_vector",
        ),
        pytest.param(
            (
                pd.Series([10, 11, 16] * 10 + [None, 23]).values,
                pd.Series(
                    [
                        {"A": 1, "B": 1},
                        {"A": 2, "B": 4},
                        {"A": 5, "B": 25},
                    ]
                    * 10
                    + [None, {"A": 4, "B": None}]
                ).values,
            ),
            ("id", "data"),
            (False, False),
            pd.Series(
                [
                    {"id": 10, "data": {"A": 1, "B": 1}},
                    {"id": 11, "data": {"A": 2, "B": 4}},
                    {"id": 16, "data": {"A": 5, "B": 25}},
                ]
                * 10
                + [{"id": None, "data": None}, {"id": 23, "data": {"A": 4, "B": None}}],
            ),
            id="2-int_vector-struct_vector",
        ),
        pytest.param(
            (
                pd.Series([10, 11, 16] * 10 + [None, 23]).values,
                pd.Series(
                    [
                        {"A": 1},
                        {"B": 2, "C": 4},
                        {"D": 5, "E": 25, "F": None},
                    ]
                    * 10
                    + [None, {"G": 4}],
                    dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.int64())),
                ).values,
            ),
            ("id", "data"),
            (False, False),
            pd.Series(
                [
                    {"id": 10, "data": {"A": 1}},
                    {"id": 11, "data": {"B": 2, "C": 4}},
                    {"id": 16, "data": {"D": 5, "E": 25, "F": None}},
                ]
                * 10
                + [{"id": None, "data": None}, {"id": 23, "data": {"G": 4}}],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("id", pa.int64()),
                            pa.field("data", pa.map_(pa.large_string(), pa.int64())),
                        ]
                    )
                ),
            ),
            id="2-int_vector-map_vector",
        ),
        pytest.param(
            (pd.Series([1, 2, 3, None, 5, 6, 7]), None),
            ("i", "j"),
            (False, True),
            pd.Series([{"i": i, "j": None} for i in [1, 2, 3, None, 5, 6, 7]]),
            id="2-int_vector-null",
        ),
        pytest.param(
            (
                pd.Series([[0], None, [1, 2], [], [3, None]]),
                pd.Series([0, None], dtype=pd.Int32Dtype()).values,
            ),
            ("A", "B"),
            (False, True),
            pd.Series(
                [
                    {"A": [0], "B": [0, None]},
                    {"A": None, "B": [0, None]},
                    {"A": [1, 2], "B": [0, None]},
                    {"A": [], "B": [0, None]},
                    {"A": [3, None], "B": [0, None]},
                ]
            ),
            id="2-int_array_vector-int_array_scalar",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        {"address": 5000, "street": "Forbes"},
                        {"address": 4200, "street": "Fifth"},
                        {"address": 6525, "street": "Penn"},
                    ]
                    * 10
                    + [{"address": 5607, "street": None}, None],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("address", pa.int64()),
                                pa.field("street", pa.string()),
                            ]
                        )
                    ),
                ).values,
                {"lat": 40.44, "lon": -79.99},
            ),
            ("data", "metadata"),
            (False, True),
            pd.Series(
                [
                    {
                        "data": {"address": 5000, "street": "Forbes"},
                        "metadata": {"lat": 40.44, "lon": -79.99},
                    },
                    {
                        "data": {"address": 4200, "street": "Fifth"},
                        "metadata": {"lat": 40.44, "lon": -79.99},
                    },
                    {
                        "data": {"address": 6525, "street": "Penn"},
                        "metadata": {"lat": 40.44, "lon": -79.99},
                    },
                ]
                * 10
                + [
                    {
                        "data": {"address": 5607, "street": None},
                        "metadata": {"lat": 40.44, "lon": -79.99},
                    },
                    {"data": None, "metadata": {"lat": 40.44, "lon": -79.99}},
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "data",
                                pa.struct(
                                    [
                                        pa.field("address", pa.int64()),
                                        pa.field("street", pa.string()),
                                    ]
                                ),
                            ),
                            pa.field("metadata", pa.map_(pa.string(), pa.float64())),
                        ]
                    )
                ),
            ),
            id="2-struct_vector-struct_scalar",
        ),
        pytest.param(
            (
                pd.Series(
                    [
                        [
                            {"species": "Tiger", "colors": [None, "orange", "black"]},
                            {"species": "Lion", "colors": ["yellow"]},
                        ],
                        [{"species": "Black Widow", "colors": ["black", "red"]}],
                        [{"species": "Flamingo", "colors": ["pink"]}, None],
                        [
                            {"species": "Ladybug", "colors": ["red", "black"]},
                            {"species": "grashopper", "colors": None},
                        ],
                        [{"species": "Elephant", "colors": ["grey"]}],
                    ]
                    * 5
                    + [[], None],
                    dtype=pd.ArrowDtype(
                        pa.list_(
                            pa.struct(
                                [
                                    pa.field("species", pa.string()),
                                    pa.field("colors", pa.list_(pa.string())),
                                ]
                            )
                        )
                    ),
                ),
            ),
            ("facts",),
            (False,),
            pd.Series(
                [
                    {
                        "facts": [
                            {"species": "Tiger", "colors": [None, "orange", "black"]},
                            {"species": "Lion", "colors": ["yellow"]},
                        ],
                    },
                    {
                        "facts": [
                            {"species": "Black Widow", "colors": ["black", "red"]}
                        ],
                    },
                    {
                        "facts": [{"species": "Flamingo", "colors": ["pink"]}, None],
                    },
                    {
                        "facts": [
                            {"species": "Ladybug", "colors": ["red", "black"]},
                            {"species": "grashopper", "colors": None},
                        ],
                    },
                    {
                        "facts": [{"species": "Elephant", "colors": ["grey"]}],
                    },
                ]
                * 5
                + [{"facts": []}, {"facts": None}],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field(
                                "facts",
                                pa.list_(
                                    pa.struct(
                                        [
                                            pa.field("species", pa.string()),
                                            pa.field("colors", pa.list_(pa.string())),
                                        ]
                                    )
                                ),
                            )
                        ]
                    )
                ),
            ),
            id="1-struct_array_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    [[1, 2, 3], [4, None, 6], [7, 8, 9], None, [13, 14, 15]]
                ).values,
                None,
                pd.Series(
                    [
                        ["Alpha", None, "Gamma", "Delta"],
                        ["Beta"],
                        None,
                        ["Epsilon", "Theta"],
                        ["Pi"],
                    ]
                ),
            ),
            ("scores", "hash", "letters"),
            (False, True, False),
            pd.Series(
                [
                    {
                        "scores": [1, 2, 3],
                        "hash": None,
                        "letters": ["Alpha", None, "Gamma", "Delta"],
                    },
                    {
                        "scores": [4, None, 6],
                        "hash": None,
                        "letters": ["Beta"],
                    },
                    {
                        "scores": [7, 8, 9],
                        "hash": None,
                        "letters": None,
                    },
                    {
                        "scores": None,
                        "hash": None,
                        "letters": ["Epsilon", "Theta"],
                    },
                    {
                        "scores": [13, 14, 15],
                        "hash": None,
                        "letters": ["Pi"],
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.struct(
                        [
                            pa.field("scores", pa.large_list(pa.int64())),
                            pa.field("hash", pa.null()),
                            pa.field("letters", pa.large_list(pa.large_string())),
                        ]
                    )
                ),
            ),
            id="3-int_array_vector-null-string_array_vector",
        ),
    ],
)
def test_object_construct_keep_null(values, keys, scalars, answer, memory_leak_check):
    keys_meta = ColNamesMetaType(keys)
    scalars_meta = MetaType(scalars)

    def impl1(v0):
        return pd.Series(
            bodosql.kernels.object_construct_keep_null((v0,), keys_meta, scalars_meta)
        )

    def impl2(v0, v1):
        return pd.Series(
            bodosql.kernels.object_construct_keep_null(
                (v0, v1), keys_meta, scalars_meta
            )
        )

    def impl3(v0, v1, v2):
        return pd.Series(
            bodosql.kernels.object_construct_keep_null(
                (v0, v1, v2), keys_meta, scalars_meta
            )
        )

    mixed_scalar_vector = any(scalars) != all(scalars)

    if len(values) == 1:
        impl = impl1
    elif len(values) == 2:
        impl = impl2
    elif len(values) == 3:
        impl = impl3

    check_func(
        impl,
        values,
        check_dtype=False,
        py_output=answer,
        dist_test=not mixed_scalar_vector,
    )


@pytest.mark.parametrize(
    "values, keys, scalars, answer",
    [
        pytest.param(
            (pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, None, 11], dtype=pd.Int32Dtype()),),
            ("i",),
            (False,),
            pd.Series(
                [{"i": i} for i in range(1, 10)] + [{}, {"i": 11}],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            id="1-int_vector",
        ),
        pytest.param(
            # TODO: fix this test with gatherv
            (
                pd.array(
                    [1, None, 3, None, 5, None, 7, None] * 5, dtype=pd.Int32Dtype()
                ),
                pd.array(
                    [8, 9, None, None, 12, 13, None, None] * 5, dtype=pd.Int32Dtype()
                ),
                pd.array(
                    [16, 17, 18, 19, None, None, None, None] * 5, dtype=pd.Int32Dtype()
                ),
            ),
            ("alpha", "beta", "gamma"),
            (False, False, False),
            pd.Series(
                [
                    {"alpha": 1, "beta": 8, "gamma": 16},
                    {"beta": 9, "gamma": 17},
                    {"alpha": 3, "gamma": 18},
                    {"gamma": 19},
                    {"alpha": 5, "beta": 12},
                    {"beta": 13},
                    {"alpha": 7},
                    {},
                ]
                * 5,
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
            ),
            id="3-int_vector-int_vector-int_vector",
        ),
        pytest.param(
            (
                "hymenoptera",
                pd.Series(
                    [
                        "hylaeus",
                        None,
                        "ophion",
                        None,
                        "vespula",
                        None,
                        "proctorenyxa incredibilis",
                        None,
                        "cleptes",
                    ]
                ).values,
            ),
            ("order", "genus"),
            (True, False),
            pd.Series(
                [
                    {"order": "hymenoptera", "genus": "hylaeus"},
                    {"order": "hymenoptera"},
                    {"order": "hymenoptera", "genus": "ophion"},
                    {"order": "hymenoptera"},
                    {"order": "hymenoptera", "genus": "vespula"},
                    {"order": "hymenoptera"},
                    {"order": "hymenoptera", "genus": "proctorenyxa incredibilis"},
                    {"order": "hymenoptera"},
                    {"order": "hymenoptera", "genus": "cleptes"},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.string())),
            ),
            id="2-string_scalar-string_vector",
        ),
        pytest.param(
            (
                pd.Series([[1]] * 3 + [[]] * 3 + [[2, 3, 4]] * 3 + [None] * 3).values,
                pd.Series([[5, 6], None, [7, None]] * 4).values,
                pd.array([8, 9, 10, 11]),
            ),
            ("A1", "A2", "A3"),
            (False, False, True),
            pd.Series(
                [
                    {"A1": [1], "A2": [5, 6], "A3": [8, 9, 10, 11]},
                    {"A1": [1], "A3": [8, 9, 10, 11]},
                    {"A1": [1], "A2": [7, None], "A3": [8, 9, 10, 11]},
                    {"A1": [], "A2": [5, 6], "A3": [8, 9, 10, 11]},
                    {"A1": [], "A3": [8, 9, 10, 11]},
                    {"A1": [], "A2": [7, None], "A3": [8, 9, 10, 11]},
                    {"A1": [2, 3, 4], "A2": [5, 6], "A3": [8, 9, 10, 11]},
                    {"A1": [2, 3, 4], "A3": [8, 9, 10, 11]},
                    {"A1": [2, 3, 4], "A2": [7, None], "A3": [8, 9, 10, 11]},
                    {"A2": [5, 6], "A3": [8, 9, 10, 11]},
                    {"A3": [8, 9, 10, 11]},
                    {"A2": [7, None], "A3": [8, 9, 10, 11]},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.large_list(pa.int32()))),
            ),
            id="3-int_array_vector-int_array_vector-int_array_scalar",
        ),
        pytest.param(
            (
                datetime.date(2023, 1, 14),
                pd.Series(
                    [
                        None if i % 2 == 1 else datetime.date.fromordinal(738534 + 2**i)
                        for i in range(10)
                    ]
                ).values,
            ),
            ("start_date", "end_date"),
            (True, False),
            pd.Series(
                [
                    {
                        "start_date": datetime.date(2023, 1, 14),
                        "end_date": datetime.date(2023, 1, 15),
                    },
                    {"start_date": datetime.date(2023, 1, 14)},
                    {
                        "start_date": datetime.date(2023, 1, 14),
                        "end_date": datetime.date(2023, 1, 18),
                    },
                    {"start_date": datetime.date(2023, 1, 14)},
                    {
                        "start_date": datetime.date(2023, 1, 14),
                        "end_date": datetime.date(2023, 1, 30),
                    },
                    {"start_date": datetime.date(2023, 1, 14)},
                    {
                        "start_date": datetime.date(2023, 1, 14),
                        "end_date": datetime.date(2023, 3, 19),
                    },
                    {"start_date": datetime.date(2023, 1, 14)},
                    {
                        "start_date": datetime.date(2023, 1, 14),
                        "end_date": datetime.date(2023, 9, 27),
                    },
                    {"start_date": datetime.date(2023, 1, 14)},
                ],
                dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.date32())),
            ),
            id="2-date_scalar-date_vector",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {"male": 1, "female": 1},
                        None,
                        None,
                        None,
                        {"male": 0, "female": 1},
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("male", pa.int32()),
                                pa.field("female", pa.int32()),
                            ]
                        )
                    ),
                ),
                pd.array(
                    [
                        {"male": 1, "female": 0},
                        {"male": 1, "female": 3},
                        None,
                        None,
                        None,
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("male", pa.int32()),
                                pa.field("female", pa.int32()),
                            ]
                        )
                    ),
                ),
                pd.array(
                    [
                        {"male": 0, "female": 1},
                        None,
                        {"male": 1, "female": 1},
                        None,
                        {"male": 4, "female": 0},
                    ],
                    dtype=pd.ArrowDtype(
                        pa.struct(
                            [
                                pa.field("male", pa.int32()),
                                pa.field("female", pa.int32()),
                            ]
                        )
                    ),
                ),
            ),
            ("parents", "siblings", "children"),
            (False, False, False),
            pd.Series(
                [
                    {
                        "parents": {"male": 1, "female": 1},
                        "siblings": {"male": 1, "female": 0},
                        "children": {"male": 0, "female": 1},
                    },
                    {"siblings": {"male": 1, "female": 3}},
                    {"children": {"male": 1, "female": 1}},
                    {},
                    {
                        "parents": {"male": 0, "female": 1},
                        "children": {"male": 4, "female": 0},
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(
                        pa.string(),
                        pa.struct(
                            [
                                pa.field("male", pa.int32()),
                                pa.field("female", pa.int32()),
                            ]
                        ),
                    )
                ),
            ),
            id="3-struct-vector_vector_vector",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {"male": 1, "female": 1},
                        None,
                        None,
                        None,
                        {"female": 1},
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
                pd.array(
                    [
                        {"male": 1},
                        {"male": 1, "female": 3},
                        None,
                        None,
                        None,
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
                pd.array(
                    [
                        {"female": 1},
                        {},
                        {"male": 1, "female": 1},
                        None,
                        {"male": 4},
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
            ),
            ("parents", "siblings", "children"),
            (False, False, False),
            pd.Series(
                [
                    {
                        "parents": {"male": 1, "female": 1},
                        "siblings": {"male": 1},
                        "children": {"female": 1},
                    },
                    {"siblings": {"male": 1, "female": 3}, "children": {}},
                    {"children": {"male": 1, "female": 1}},
                    {},
                    {"parents": {"female": 1}, "children": {"male": 4}},
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(pa.string(), pa.map_(pa.string(), pa.int64()))
                ),
            ),
            id="3-map-vector_vector_vector",
        ),
        pytest.param(
            (
                pd.array(
                    [
                        {"A": 1, "E": 1, "I": 1, "O": 1, "U": 1, "Y": None},
                        None,
                        {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5},
                        {"Q": 10, "Z": 10, "J": 8, "X": 8},
                        {},
                    ],
                    dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int64())),
                ),
                {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4},
            ),
            ("points", "char_counts"),
            (False, True),
            pd.Series(
                [
                    {
                        "points": {"A": 1, "E": 1, "I": 1, "O": 1, "U": 1, "Y": None},
                        "char_counts": {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4},
                    },
                    {"char_counts": {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4}},
                    {
                        "points": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5},
                        "char_counts": {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4},
                    },
                    {
                        "points": {"Q": 10, "Z": 10, "J": 8, "X": 8},
                        "char_counts": {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4},
                    },
                    {
                        "points": {},
                        "char_counts": {"C": 1, "I": 1, "O": 1, "R": 1, "S": 4},
                    },
                ],
                dtype=pd.ArrowDtype(
                    pa.map_(pa.string(), pa.map_(pa.string(), pa.int64()))
                ),
            ),
            id="2-map-vector_scalar",
        ),
    ],
)
def test_object_construct(values, keys, scalars, answer, memory_leak_check):
    keys_meta = ColNamesMetaType(keys)
    scalars_meta = MetaType(scalars)

    def impl1(v0):
        return pd.DataFrame(
            {"res": bodosql.kernels.object_construct((v0,), keys_meta, scalars_meta)}
        )

    def impl2(v0, v1):
        return pd.DataFrame(
            {"res": bodosql.kernels.object_construct((v0, v1), keys_meta, scalars_meta)}
        )

    def impl3(v0, v1, v2):
        return pd.DataFrame(
            {
                "res": bodosql.kernels.object_construct(
                    (v0, v1, v2), keys_meta, scalars_meta
                )
            }
        )

    mixed_scalar_vector = any(scalars) != all(scalars)

    if len(values) == 1:
        impl = impl1
    elif len(values) == 2:
        impl = impl2
    elif len(values) == 3:
        impl = impl3

    check_func(
        impl,
        values,
        check_dtype=False,
        py_output=pd.DataFrame({"res": answer}),
        dist_test=not mixed_scalar_vector,
        convert_to_nullable_float=True,
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
def test_object_construct_keep_null_optional(is_none_0, is_none_1, memory_leak_check):
    names = ColNamesMetaType(("A", "B"))
    scalars = MetaType((True, True))

    def impl(A, B, is_none_0, is_none_1):
        arg0 = None if is_none_0 else A
        arg1 = None if is_none_1 else B
        return bodosql.kernels.object_construct_keep_null((arg0, arg1), names, scalars)

    answer = {"A": None if is_none_0 else 42, "B": None if is_none_1 else True}

    check_func(
        impl,
        (42, True, is_none_0, is_none_1),
        py_output=answer,
        check_dtype=False,
        dist_test=False,
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
@pytest.mark.parametrize(
    "first_val, second_val",
    [
        pytest.param(True, False, id="boolean"),
        pytest.param(np.int8(15), np.int64(-(2**60 - 1)), id="integers"),
        pytest.param("alpha", "beta", id="strings"),
        pytest.param(pd.array([3.14, 2.7]), pd.array([-1.0]), id="float_arrays"),
        pytest.param({"A": 0, "B": 1}, {"A": 2, "E": 3, "I": 4}, id="int_map"),
    ],
)
def test_object_construct_optional(
    first_val, second_val, is_none_0, is_none_1, memory_leak_check
):
    names = ColNamesMetaType(("first_key", "second_key"))
    scalars = MetaType((True, True))

    def impl(A, B, is_none_0, is_none_1):
        arg0 = None if is_none_0 else A
        arg1 = None if is_none_1 else B
        return bodosql.kernels.object_construct((arg0, arg1), names, scalars)

    answer = {}
    if not is_none_0:
        answer["first_key"] = first_val
    if not is_none_1:
        answer["second_key"] = second_val

    check_func(
        impl,
        (first_val, second_val, is_none_0, is_none_1),
        py_output=answer,
        check_dtype=False,
        dist_test=False,
    )


@pytest.mark.parametrize(
    "args, scalar_tup, answer",
    [
        pytest.param(
            (
                pd.Series(
                    [1, None, 4, None, 16, None, 64, None, 256], dtype=pd.Int64Dtype()
                ),
            ),
            (False,),
            pd.Series(
                [[1], [None], [4], [None], [16], [None], [64], [None], [256]],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
            ),
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
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int32())),
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
            pd.Series(
                [
                    [None, 1, 3, 4],
                    [None, 2, 3, 5],
                    [None, 1, 3, -6],
                    [None, 2, 3, 7],
                    [None, 1, 3, 8],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
            id="integer-4",
        ),
        pytest.param(
            ("",),
            (True,),
            pd.array([""]),
            id="string_scalar-1",
        ),
        pytest.param(
            (pd.Series(["", None, "A", None, "AB", None, "ABC"] * 3),),
            (False,),
            pd.Series(
                [[""], [None], ["A"], [None], ["AB"], [None], ["ABC"]] * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string-1",
        ),
        pytest.param(
            (pd.Series(["", None, "A", None, "AB", None, "ABC"] * 3), "foo"),
            (False, True),
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
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
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
            pd.Series(
                [
                    ["K", "1", None, ""],
                    ["K", None, None, "A"],
                    ["K", "2", None, "BC"],
                    ["K", None, None, "DEF"],
                    ["K", "3", None, "GHIJ"],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.string())),
            ),
            id="string-4",
        ),
        pytest.param(
            (
                b"1",
                pd.Series([b"23", None, b"456", None, b""] * 2),
            ),
            (True, False),
            pd.Series(
                [[b"1", b"23"], [b"1", None], [b"1", b"456"], [b"1", None], [b"1", b""]]
                * 2
            ),
            id="binary-2",
        ),
        pytest.param(
            (
                np.array([True, False, True, False] * 2, dtype=np.bool_),
                pd.Series([True, False, None, True] * 2, dtype=pd.BooleanDtype()),
            ),
            (False, False),
            pd.Series(
                [[True, True], [False, False], [True, None], [False, True]] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.bool_())),
            ),
            id="boolean-2",
        ),
        pytest.param(
            (
                np.float64(3.14),
                pd.Series([-1.0, 0.0, None, 2.71828] * 2, dtype=pd.Float32Dtype()),
            ),
            (True, False),
            pd.Series(
                [[3.14, -1.0], [3.14, 0.0], [3.14, None], [3.14, 2.71828]] * 2,
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
            ),
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
            pd.Series(
                [
                    [Decimal("0.0"), Decimal("0.025")],
                    [None, Decimal("0.025")],
                    [Decimal("-1024.2048"), Decimal("0.025")],
                    [None, Decimal("0.025")],
                    [Decimal("1.23456789"), Decimal("0.025")],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.decimal128(38, 18))),
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
            pd.Series(
                [
                    [Decimal("0.0"), Decimal("0.0")],
                    [None, Decimal("0.0")],
                    [Decimal("-1024.2048"), Decimal("0.0")],
                    [None, Decimal("0.0")],
                    [Decimal("1.23456789"), Decimal("0.0")],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.decimal128(38, 18))),
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
            pd.Series(
                [
                    [0.0, 3.1415],
                    [None, 3.1415],
                    [-1024.2048, 3.1415],
                    [None, 3.1415],
                    [1.23456789, 3.1415],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.float64())),
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
            pd.Series(
                [
                    [datetime.date(2023, 10, 25), datetime.date(1999, 12, 31)],
                    [None, datetime.date(1999, 12, 31)],
                    [datetime.date(1999, 12, 31), datetime.date(1999, 12, 31)],
                    [None, datetime.date(1999, 12, 31)],
                    [datetime.date(2008, 4, 1), datetime.date(1999, 12, 31)],
                ],
                dtype=pd.ArrowDtype(pa.large_list(pa.date32())),
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
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            (True, False),
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
                    ],
                    dtype="datetime64[ns, US/Pacific]",
                ),
            ),
            (True, False),
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
                    ],
                    dtype="datetime64[ns]",
                ),
                pd.Series(
                    [
                        pd.Timestamp("2023-1-1", tz="US/Pacific"),
                        None,
                        pd.Timestamp("1999-12-31 23:59:59.999250", tz="US/Pacific"),
                        None,
                        pd.Timestamp("2024-7-4 6:30:00", tz="US/Pacific"),
                    ],
                    dtype="datetime64[ns, US/Pacific]",
                ),
            ),
            (False, False),
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
            pd.Series([[[1]], [[2, None]], [[]], [None], [[5, 6]], [[]]] * 3),
            id="nested_integer-1",
        ),
        pytest.param(
            (pd.Series([["A"], ["BC", None], [], None, ["DEF", "", "GH"], []] * 3),),
            (False,),
            pd.Series(
                [[["A"]], [["BC", None]], [[]], [None], [["DEF", "", "GH"]], [[]]] * 3
            ),
            id="nested_string-1",
        ),
        pytest.param(
            (
                pd.Series(
                    [[1], [2, None], [], None, [5, 6], []] * 3,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
                pd.Series(
                    [[], [], [7], [8], [None, 10], [11, 12]] * 3,
                    dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
                ),
            ),
            (False, False),
            pd.Series(
                [
                    [[1], []],
                    [[2, None], []],
                    [[], [7]],
                    [None, [8]],
                    [[5, 6], [None, 10]],
                    [[], [11, 12]],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.int64()))),
            ),
            id="nested_integer-2",
        ),
        pytest.param(
            (
                pd.Series(
                    [1, 2, 3] * 3,
                    dtype=np.int64,
                ),
                pd.Series(
                    [4, 5, 6] * 3,
                    dtype=np.int64,
                ),
            ),
            (True, True),
            pd.Series(
                [[1, 2, 3] * 3, [4, 5, 6] * 3],
                dtype=pd.ArrowDtype(pa.large_list(pa.int64())),
            ),
            id="nested_integer_scalar-2",
        ),
        pytest.param(
            (
                pd.Series(
                    [["A"], ["BC", None], [], None, ["DEF", "", "GH"], []] * 3,
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
                pd.Series(
                    [["A", "BC"], [], ["DEF", ""], [None], None, ["GH"]] * 3,
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
            ),
            (False, False),
            pd.Series(
                [
                    [["A"], ["A", "BC"]],
                    [["BC", None], []],
                    [[], ["DEF", ""]],
                    [None, [None]],
                    [["DEF", "", "GH"], None],
                    [[], ["GH"]],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.large_list(pa.large_string()))),
            ),
            id="nested_string-2",
        ),
        pytest.param(
            (
                pd.Series(
                    ["a", "b", "c"],
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
                pd.Series(
                    ["d", "e", "f"],
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
                pd.Series(
                    ["g", "h", "i"],
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
                pd.Series(
                    ["j", "k", "l"],
                    dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
                ),
            ),
            (True, True, True, True),
            pd.Series(
                [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"], ["j", "k", "l"]],
                dtype=pd.ArrowDtype(pa.large_list(pa.large_string())),
            ),
            id="nested_string_scalar-4",
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
                    * 3,
                    dtype=pd.ArrowDtype(pa.map_(pa.large_string(), pa.large_string())),
                ),
            ),
            (False,),
            pd.Series(
                [
                    [{"O": "A"}],
                    [{"P": "B", "Q": "C", "R": "D"}],
                    [None],
                    [{}],
                    [{"S": "", "T": "EFGH", "U": "IJ"}],
                ]
                * 3,
                dtype=pd.ArrowDtype(
                    pa.large_list(pa.map_(pa.large_string(), pa.large_string()))
                ),
            ),
            id="map_array-string-1",
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
            pd.Series(
                [
                    [{0: 9}, {6: 8}],
                    [{1: 10, 2: 7, 3: 5}, None],
                    [None, {7: 8, 8: 6, 9: 5}],
                    [{}, {10: 7}],
                    [{4: 6, 5: 6}, {}],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.map_(pa.int64(), pa.int64()))),
            ),
            id="map_array-int_int-2",
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
                    * 3,
                    dtype=pd.ArrowDtype(pa.map_(pa.int32(), pa.int32())),
                ),
                {0: 1, 2: 3},
            ),
            (False, True),
            pd.Series(
                [
                    [{0: 9}, {0: 1, 2: 3}],
                    [{1: 10, 2: 7, 3: 5}, {0: 1, 2: 3}],
                    [None, {0: 1, 2: 3}],
                    [{}, {0: 1, 2: 3}],
                    [{4: 6, 5: 6}, {0: 1, 2: 3}],
                ]
                * 3,
                dtype=pd.ArrowDtype(pa.large_list(pa.map_(pa.int32(), pa.int32()))),
            ),
            id="map_array_with_scalars-int_int-2",
        ),
    ],
)
def test_array_construct(args, scalar_tup, answer, memory_leak_check):
    def impl_scalar(A):
        return bodosql.kernels.array_construct((A,), scalar_tup)

    def impl1(A):
        return pd.Series(bodosql.kernels.array_construct((A,), scalar_tup))

    def impl2(A, B):
        return pd.Series(bodosql.kernels.array_construct((A, B), scalar_tup))

    def impl4(A, B, C, D):
        return pd.Series(bodosql.kernels.array_construct((A, B, C, D), scalar_tup))

    dist_test = True
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
        # If every input is marked as scalar, we don't want to distributed the
        # input, and there's no distributed semantics to test.
        dist_test = not all(scalar_tup)

    check_func(
        impl,
        args,
        is_out_distributed=is_out_distributed,
        dist_test=dist_test,
        py_output=answer,
        check_dtype=False,
        reset_index=True,
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
        return bodosql.kernels.array_construct((arg0, arg1), (True, True))

    answer = pd.array([None if is_none_0 else "A", None if is_none_1 else "B"])

    check_func(
        impl,
        ("A", "B", is_none_0, is_none_1),
        py_output=answer,
        check_dtype=False,
        dist_test=False,
    )


def test_object_filter_keys_errors(memory_leak_check):
    scalars = MetaType((False, True))
    with pytest.raises(
        BodoError, match=re.escape("keep_keys argument must be a const bool")
    ):

        @bodo.jit
        def impl(B, A):
            return bodosql.kernels.object_filter_keys((A, "A"), B, scalars)

        impl(True, pd.Series([{"a": 0}]))

    with pytest.raises(
        BodoError,
        match=re.escape("unsupported on struct arrays with non-constant keys"),
    ):

        @bodo.jit
        def impl(B, A):
            return bodosql.kernels.object_filter_keys((A, B), True, scalars)

        impl(
            pd.Series(["a"]),
            pd.Series(
                [{"a": 0}], dtype=pd.ArrowDtype(pa.struct([pa.field("a", pa.int32())]))
            ),
        )
