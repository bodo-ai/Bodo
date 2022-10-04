# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL miscellaneous functions
"""


import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(
            (
                pd.Series([0, 42, None], dtype=pd.Int32Dtype()).repeat(3),
                pd.Series([0, 42, None] * 3, dtype=pd.Int32Dtype()),
            ),
            id="vector_vector",
        ),
        pytest.param(
            (pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype()), 0),
            id="vector_scalar_zero",
        ),
        pytest.param(
            (0, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype())),
            id="scalar_vector_zero",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (3, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int32Dtype())),
            id="scalar_vector_nonzero",
        ),
        pytest.param(
            (
                pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int16Dtype()),
                np.uint8(255),
            ),
            id="vector_scalar_nonzero",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int64Dtype()), None),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, pd.Series([0, 1, -1, None, 2, -2, 0, None], dtype=pd.Int64Dtype())),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (15, -42),
            id="scalar_scalar_nonzero",
        ),
        pytest.param((0, 13), id="scalar_scalar_mixed", marks=pytest.mark.slow),
        pytest.param((0, 0), id="scalar_scalar_zero_zero", marks=pytest.mark.slow),
        pytest.param((0, None), id="scalar_scalar_zero_null", marks=pytest.mark.slow),
        pytest.param(
            (64, None), id="scalar_scalar_nonzero_null", marks=pytest.mark.slow
        ),
        pytest.param((None, 0), id="scalar_scalar_null_zero", marks=pytest.mark.slow),
        pytest.param(
            (None, -15), id="scalar_scalar_null_nonzero", marks=pytest.mark.slow
        ),
        pytest.param(
            (None, None), id="scalar_scalar_null_null", marks=pytest.mark.slow
        ),
        pytest.param(
            (
                pd.Series([0, 1, 127, 128, 255, None] * 5, dtype=pd.UInt8Dtype()),
                pd.Series([0, 1, 127, -128, -1, None], dtype=pd.Int8Dtype()).repeat(5),
            ),
            id="mixed_int_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([True, False, None], dtype=pd.BooleanDtype()).repeat(3),
                pd.Series([True, False, None] * 3, dtype=pd.BooleanDtype()),
            ),
            id="boolean_vector_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def boolean_numerical_scalar_vector(request):
    return request.param


def test_booland(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.booland(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.booland(A, B)

    def booland_scalar_fn(A, B):
        if pd.notna(A) and pd.notna(B) and A != 0 and B != 0:
            return True
        elif (pd.notna(A) and A == 0) or (pd.notna(B) and B == 0):
            return False
        else:
            return None

    booland_answer = vectorized_sol(
        boolean_numerical_scalar_vector, booland_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=booland_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_boolor(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.boolor(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.boolor(A, B)

    def boolor_scalar_fn(A, B):
        if (pd.notna(A) and A != 0) or (pd.notna(B) and B != 0):
            return True
        elif pd.notna(A) and A == 0 and pd.notna(B) and B == 0:
            return False
        else:
            return None

    boolor_answer = vectorized_sol(
        boolean_numerical_scalar_vector, boolor_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=boolor_answer,
        check_dtype=False,
        reset_index=True,
        sort_output=False,
    )


def test_boolxor(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.boolxor(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in boolean_numerical_scalar_vector):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.boolxor(A, B)

    def boolxor_scalar_fn(A, B):
        if pd.isna(A) or pd.isna(B):
            return None
        else:
            return (A == 0) != (B == 0)

    boolxor_answer = vectorized_sol(
        boolean_numerical_scalar_vector, boolxor_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl,
        boolean_numerical_scalar_vector,
        py_output=boolxor_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_boolnot(boolean_numerical_scalar_vector, memory_leak_check):
    def impl(A):
        return pd.Series(bodo.libs.bodosql_array_kernels.boolnot(A))

    impl_scalar = lambda A: bodo.libs.bodosql_array_kernels.boolnot(A)

    def boolnot_scalar_fn(A):
        if pd.isna(A):
            return None
        if A == 0:
            return True
        else:
            return False

    boolxor_answer_0 = vectorized_sol(
        (boolean_numerical_scalar_vector[0],), boolnot_scalar_fn, pd.BooleanDtype()
    )
    boolxor_answer_1 = vectorized_sol(
        (boolean_numerical_scalar_vector[1],), boolnot_scalar_fn, pd.BooleanDtype()
    )

    check_func(
        impl
        if isinstance(boolean_numerical_scalar_vector[0], pd.Series)
        else impl_scalar,
        (boolean_numerical_scalar_vector[0],),
        py_output=boolxor_answer_0,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl
        if isinstance(boolean_numerical_scalar_vector[1], pd.Series)
        else impl_scalar,
        (boolean_numerical_scalar_vector[1],),
        py_output=boolxor_answer_1,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array([True, False, True, False, True, None])),
                pd.Series(pd.array([None, None, 2, 3, 4, -1])),
                pd.Series(pd.array([5, 6, None, None, 9, -1])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(pd.array([True, True, True, False, False])),
                pd.Series(pd.array(["A", "B", "C", "D", "E"])),
                "-",
            ),
            id="vector_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series(pd.array([False, True, False, True, False])), 1.0, -1.0),
            id="vector_scalar_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(pd.array([True, True, False, False, True])),
                pd.Series(pd.array(["A", "B", "C", "D", "E"])),
                None,
            ),
            id="vector_vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, 42, 16),
            id="all_scalar_no_null",
        ),
        pytest.param(
            (None, 42, 16),
            id="all_scalar_with_null_cond",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, None, 16),
            id="all_scalar_with_null_branch",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (True, 13, None),
            id="all_scalar_with_unused_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (False, None, None),
            id="all_scalar_both_null_branch",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, None, None),
            id="all_scalar_all_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_cond(args, memory_leak_check):
    def impl(arr, ifbranch, elsebranch):
        return pd.Series(
            bodo.libs.bodosql_array_kernels.cond(arr, ifbranch, elsebranch)
        )

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arr, ifbranch, elsebranch: bodo.libs.bodosql_array_kernels.cond(
            arr, ifbranch, elsebranch
        )

    # Simulates COND on a single row
    def cond_scalar_fn(arr, ifbranch, elsebranch):
        return ifbranch if ((not pd.isna(arr)) and arr) else elsebranch

    cond_answer = vectorized_sol(args, cond_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=cond_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series([0, 42, None] * 3, dtype=pd.Int32Dtype()),
                pd.Series(
                    [0, 0, 0, 42, 42, 42, None, None, None], dtype=pd.Int32Dtype()
                ),
            ),
            id="int32_vector_vector",
        ),
        pytest.param(
            (pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()), 42),
            id="int32_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                0,
                pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()),
            ),
            id="int32_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()), None),
            id="int32_vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series([0, 36, 42, None, -42, 1], dtype=pd.Int32Dtype()),
            ),
            id="int32_null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((42, 42), id="int64_scalar_scalar"),
        pytest.param((39, None), id="int64_scalar_null"),
        pytest.param((None, 42), id="int64_null_scalar"),
        pytest.param((None, None), id="int64_null_null"),
        pytest.param(
            (
                pd.Series([0, 1, 127, 128, 255, None] * 5, dtype=pd.UInt8Dtype()),
                pd.Series([0, 1, 127, -128, -1, None], dtype=pd.Int8Dtype()).repeat(5),
            ),
            id="uint8_int8_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(["A", "B", "a", "AAA", None] * 4),
                pd.Series(["A", "B", "a", "AAA", None]).repeat(4),
            ),
            id="string_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series([True, False, None, True] * 4),
                pd.Series([True, False, None, False]).repeat(4),
            ),
            id="boolean_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    (list(pd.date_range("2018", "2019", periods=3)) + [None]) * 4
                ),
                pd.Series(
                    list(pd.date_range("2018", "2019", periods=3)) + [None]
                ).repeat(4),
            ),
            id="date_vector_vector",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_bool_equal_null(args, memory_leak_check):
    def impl(A, B):
        return pd.Series(bodo.libs.bodosql_array_kernels.equal_null(A, B))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda A, B: bodo.libs.bodosql_array_kernels.equal_null(A, B)

    def equal_null_scalar_fn(A, B):
        if (pd.isna(A) and pd.isna(B)) or (pd.notna(A) and pd.notna(B) and A == B):
            return True
        else:
            return False

    equal_null_answer = vectorized_sol(args, equal_null_scalar_fn, None)

    check_func(
        impl,
        args,
        py_output=equal_null_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    [
                        b"sxcsdasdfdf",
                        None,
                        b"",
                        b"asadf1234524asdfa",
                        b"\0\0\0\0",
                        None,
                        b"hello world",
                    ]
                    * 2
                ),
                pd.Series(
                    [
                        b"sxcsdasdfdf",
                        b"239i1u8yighjbfdnsma4",
                        b"i12u3gewqds",
                        None,
                        b"1203-94euwidsfhjk",
                        None,
                        b"hello world",
                    ]
                    * 2
                ),
                None,
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                12345678.123456789,
                pd.Series(
                    [
                        12345678.123456789,
                        None,
                        1,
                        2,
                        3,
                        None,
                        4,
                        12345678.123456789,
                        5,
                    ]
                    * 2
                ),
                None,
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            pd.Timestamp("2022-01-02 00:00:00"),
                            None,
                            pd.Timestamp("2002-01-02 00:00:00"),
                            pd.Timestamp("2022"),
                            None,
                            pd.Timestamp("2122-01-12 00:00:00"),
                            pd.Timestamp("2022"),
                            pd.Timestamp("2022-01-02 00:01:00"),
                            pd.Timestamp("2022-11-02 00:00:00"),
                        ]
                        * 2
                    )
                ),
                pd.Timestamp("2022"),
                None,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series(
                    pd.array(
                        [
                            b"12345678.123456789",
                            None,
                            b"a",
                            b"b",
                            b"c",
                            b"d",
                            b"e",
                            b"12345678.123456789",
                            b"g",
                        ]
                        * 2
                    )
                ),
                pd.StringDtype(),
            ),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            pd.Timedelta(minutes=40),
                            pd.Timedelta(hours=2),
                            pd.Timedelta(5),
                            pd.Timedelta(days=3),
                            pd.Timedelta(days=13),
                            pd.Timedelta(weeks=3),
                            pd.Timedelta(seconds=3),
                            None,
                            None,
                        ]
                        * 2
                    )
                ),
                None,
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param((-426472, 2, pd.Int64Dtype()), id="all_scalar_not_null"),
        pytest.param(
            ("hello world", None, pd.StringDtype()),
            id="all_scalar_null_arg1",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, b"0923u8hejrknsd", None),
            id="all_scalar_null_arg0",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, None, None),
            id="all_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_nullif(args, memory_leak_check):
    def impl(arg0, arg1):
        return pd.Series(bodo.libs.bodosql_array_kernels.nullif(arg0, arg1))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl = lambda arg0, arg1: bodo.libs.bodosql_array_kernels.nullif(arg0, arg1)

    # Simulates NULLIF on a single row
    def nullif_scalar_fn(arg0, arg1):
        if pd.isna(arg0) or arg0 == arg1:
            return None
        else:
            return arg0

    arg0, arg1, out_dtype = args

    nullif_answer = vectorized_sol((arg0, arg1), nullif_scalar_fn, out_dtype)

    check_func(
        impl,
        (arg0, arg1),
        py_output=nullif_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                -42.16,
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                10875.3115512,
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                pd.Series(
                    np.array(
                        [1.0, None, 3.0, 4.0, 5.0, 6.0, None, 8.0], dtype=np.float64
                    )
                ),
                None,
            ),
            id="vector_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                None,
                pd.Series(
                    np.array(
                        [None, 4.0, 9.0, 16.0, 25.0, 36.0, None, 64.0], dtype=np.float64
                    )
                ),
            ),
            id="null_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((100.0, 95.2), id="all_scalar_no_null"),
        pytest.param((10.03, None), id="all_scalar_with_null"),
    ],
)
def test_regr_valxy(args, memory_leak_check):
    def impl1(y, x):
        return pd.Series(bodo.libs.bodosql_array_kernels.regr_valx(y, x))

    def impl2(y, x):
        return pd.Series(bodo.libs.bodosql_array_kernels.regr_valy(y, x))

    # avoid Series conversion for scalar output
    if all(not isinstance(arg, pd.Series) for arg in args):
        impl1 = lambda y, x: bodo.libs.bodosql_array_kernels.regr_valx(y, x)
        impl2 = lambda y, x: bodo.libs.bodosql_array_kernels.regr_valy(y, x)

    # Simulates REGR_VALX on a single row
    def regr_valx_scalar_fn(y, x):
        if pd.isna(y) or pd.isna(x):
            return None
        else:
            return x

    # Simulates REGR_VALY on a single row
    def regr_valy_scalar_fn(y, x):
        if pd.isna(y) or pd.isna(x):
            return None
        else:
            return y

    regr_valx_answer = vectorized_sol(args, regr_valx_scalar_fn, None)
    regr_valy_answer = vectorized_sol(args, regr_valy_scalar_fn, None)
    check_func(
        impl1,
        args,
        py_output=regr_valx_answer,
        check_dtype=False,
        reset_index=True,
    )
    check_func(
        impl2,
        args,
        py_output=regr_valy_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_bool_fns(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.booland(arg0, arg1),
            bodo.libs.bodosql_array_kernels.boolor(arg0, arg1),
            bodo.libs.bodosql_array_kernels.boolxor(arg0, arg1),
            bodo.libs.bodosql_array_kernels.boolnot(arg0),
            bodo.libs.bodosql_array_kernels.equal_null(arg0, arg1),
        )

    for A in [0, 16]:
        for B in [0, 16]:
            for flag0 in [True, False]:
                for flag1 in [True, False]:
                    a = A if flag0 else None
                    b = B if flag1 else None
                    A0 = (
                        True
                        if a == 16 and b == 16
                        else (False if a == 0 or b == 0 else None)
                    )
                    A1 = (
                        True
                        if a == 16 or b == 16
                        else (False if a == 0 and b == 0 else None)
                    )
                    A2 = None if a == None or b == None else (a != b)
                    A3 = None if a == None else not a
                    A4 = a == b
                    check_func(
                        impl, (A, B, flag0, flag1), py_output=(A0, A1, A2, A3, A4)
                    )


@pytest.mark.slow
def test_cond_option(memory_leak_check):
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return bodo.libs.bodosql_array_kernels.cond(arg0, arg1, arg2)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                answer = "A" if flag0 and flag1 else None
                check_func(
                    impl, (True, "A", "B", flag0, flag1, flag2), py_output=answer
                )


@pytest.mark.slow
def test_option_nullif(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.nullif(arg0, arg1)

    A, B = 0.1, 0.5
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = None if not flag0 else 0.1
            check_func(impl, (A, B, flag0, flag1), py_output=answer)


@pytest.mark.slow
def test_option_regr_valxy(memory_leak_check):
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.regr_valx(arg0, arg1),
            bodo.libs.bodosql_array_kernels.regr_valy(arg0, arg1),
        )

    A, B = 0.1, 0.5
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (0.5, 0.1) if flag0 and flag1 else (None, None)
            check_func(impl, (A, B, flag0, flag1), py_output=answer)
