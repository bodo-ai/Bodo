# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL miscellaneous functions
"""


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
def test_cond(args):
    def impl(arr, ifbranch, elsebranch):
        return bodo.libs.bodosql_array_kernels.cond(arr, ifbranch, elsebranch)

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
def test_nullif(args):
    def impl(arg0, arg1):
        return bodo.libs.bodosql_array_kernels.nullif(arg0, arg1)

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


@pytest.mark.slow
def test_cond_option():
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
def test_option_nullif():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.nullif(arg0, arg1)

    A, B = 0.1, 0.5
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = None if not flag0 else 0.1
            check_func(impl, (A, B, flag0, flag1), py_output=answer)
