# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL numeric functions
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["10", "11", "12", "13", "14", "15"])),
                pd.Series(pd.array([10, 10, 10, 16, 16, 16])),
                pd.Series(pd.array([2, 10, 16, 2, 10, 16])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "11111",
                pd.Series(
                    pd.array(
                        [2, 2, 2, 2, 8, 8, 8, 8, 10, 10, 10, 10, 16, 16, 16, 16, 10, 10]
                    )
                ),
                pd.Series(
                    pd.array(
                        [2, 8, 10, 16, 2, 8, 10, 16, 2, 8, 10, 16, 2, 8, 10, 16, 17, -1]
                    )
                ),
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(pd.array(["2", "4", None, "8", "16", "32", "64", None])),
                pd.Series(pd.array([3, None, None, None, 16, 7, 36, 3])),
                10,
            ),
            id="vector_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "FGHIJ",
                pd.Series(pd.array([20, 21, 22, 23, 24, 25])),
                10,
            ),
            id="scalar_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("ff", 16, 2),
            id="all_scalar",
        ),
    ],
)
def test_conv(args):
    def impl(arr, old_base, new_base):
        return bodo.libs.bodosql_array_kernels.conv(arr, old_base, new_base)

    # Simulates CONV on a single row
    def conv_scalar_fn(elem, old_base, new_base):
        if (
            pd.isna(elem)
            or pd.isna(old_base)
            or pd.isna(new_base)
            or old_base <= 1
            or new_base not in [2, 8, 10, 16]
        ):
            return None
        else:
            old = int(elem, base=old_base)
            if new_base == 2:
                return "{:b}".format(old)
            if new_base == 8:
                return "{:o}".format(old)
            if new_base == 10:
                return "{:d}".format(old)
            if new_base == 16:
                return "{:x}".format(old)
            return None

    conv_answer = vectorized_sol(args, conv_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        args,
        py_output=conv_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([1.0, 2.0, 3.0, 4.0, 8.0]),
                pd.Series([6.0, 0.0, 2.0, 0.0, 0.0]),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([1.1, None, 3.6, 10.0, 16.0, 17.3, 101.0]),
                0.0,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (1.0, 0.0),
            id="all_scalar_no_null",
        ),
        pytest.param((None, 5.6), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_div0(args):
    def impl(a, b):
        return bodo.libs.bodosql_array_kernels.div0(a, b)

    def div0_scalar_fn(a, b):
        if pd.isna(a) or pd.isna(b):
            return None
        elif b != 0:
            return a / b
        else:
            return 0

    a, b = args
    expected_output = vectorized_sol(args, div0_scalar_fn, np.float64)

    check_func(
        impl,
        (
            a,
            b,
        ),
        py_output=expected_output,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([1.0, 2.0, 3.0, 4.0, 8.0]),
                pd.Series([6.0, 2.0, 2.0, 10.5, 2.0]),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series([1.1, None, 3.6, 10.0, 16.0, 17.3, 101.0]),
                2.0,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (64.0, 4.0),
            id="all_scalar_no_null",
        ),
        pytest.param((None, 5.6), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_log(args):
    def impl(arr, base):
        return bodo.libs.bodosql_array_kernels.log(arr, base)

    # Simulates LOG on a single row
    def log_scalar_fn(elem, base):
        if pd.isna(elem) or pd.isna(base):
            return None
        else:
            return np.log(elem) / np.log(base)

    log_answer = vectorized_sol(args, log_scalar_fn, np.float64)
    check_func(
        impl,
        args,
        py_output=log_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param(
            pd.Series([1, 0, 2345678, -910, None], dtype=pd.Int64Dtype()),
            id="vector_int",
        ),
        pytest.param(
            pd.Series(pd.array([0, 1, 32, 127, -126, 125], dtype=pd.Int8Dtype())),
            id="vector_int8",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [0, 1, 32, 127, 128, 129, 251, 252, 253, 254, 255],
                    dtype=pd.UInt8Dtype(),
                )
            ),
            id="vector_uint8",
        ),
        pytest.param(
            pd.Series(
                pd.array([0, 1, 100, 1000, 32767, 32768, 65535], dtype=pd.UInt16Dtype())
            ),
            id="vector_uint16",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [0, 100, 32767, 32768, 65535, 4294967295], dtype=pd.UInt32Dtype()
                )
            ),
            id="vector_uint32",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                pd.array(
                    [
                        0,
                        100,
                        32767,
                        4294967295,
                        9223372036854775806,
                        9223372036854775807,
                    ],
                    dtype=pd.UInt64Dtype(),
                )
            ),
            id="vector_uint64",
        ),
        pytest.param(
            pd.Series([-1.0, 0.0, -123.456, 4096.1, None], dtype=np.float64),
            id="vector_float",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            42,
            id="scalar_int",
        ),
        pytest.param(-12.345, id="scalar_float", marks=pytest.mark.slow),
    ],
)
def test_negate(numbers):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.negate(arr)

    # Simulates -X on a single row
    def negate_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return -elem

    if (
        isinstance(numbers, pd.Series)
        and not isinstance(numbers.dtype, np.dtype)
        and numbers.dtype
        in (pd.UInt8Dtype(), pd.UInt16Dtype(), pd.UInt32Dtype(), pd.UInt64Dtype())
    ):
        dtype = {
            pd.UInt8Dtype(): pd.Int16Dtype(),
            pd.UInt16Dtype(): pd.Int32Dtype(),
            pd.UInt32Dtype(): pd.Int64Dtype(),
            pd.UInt64Dtype(): pd.Int64Dtype(),
        }[numbers.dtype]
        negate_answer = vectorized_sol(
            (pd.Series(pd.array(list(numbers), dtype=dtype)),), negate_scalar_fn, dtype
        )
    else:
        negate_answer = vectorized_sol((numbers,), negate_scalar_fn, None)

    check_func(
        impl,
        (numbers,),
        py_output=negate_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_conv_option():
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodo.libs.bodosql_array_kernels.conv(arg0, arg1, arg2)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                answer = "101010" if flag0 and flag1 and flag2 else None
                check_func(impl, ("42", 10, 2, flag0, flag1, flag2), py_output=answer)


@pytest.mark.slow
def test_div0_option():
    def impl(a, b, flag0, flag1):
        arg0 = a if flag0 else None
        arg1 = b if flag1 else None
        return bodo.libs.bodosql_array_kernels.div0(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 0.0 if flag0 and flag1 else None
            check_func(impl, (8.0, 0.0, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_log_option():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.log(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 3.0 if flag0 and flag1 else None
            check_func(impl, (8.0, 2.0, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_negate_option():
    def impl(A, flag0):
        arg = A if flag0 else None
        return bodo.libs.bodosql_array_kernels.negate(arg)

    for flag0 in [True, False]:
        answer = -42 if flag0 else None
        check_func(impl, (42, flag0), py_output=answer)
