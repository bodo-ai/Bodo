# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL
"""

import re

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs import bodosql_array_kernels
from bodo.tests.utils import check_func, gen_nonascii_list
from bodo.utils.typing import BodoError


def vectorized_sol(args, scalar_fn, dtype, manual_coercion=False):
    """Creates a py_output for a vectorized function using its arguments and the
       a function that is applied to the scalar values

    Args:
        args (any list): a list of arguments, each of which is either a scalar
        or vector (vectors must be the same size)
        scalar_fn (function): the function that is applied to scalar values
        corresponding to each row
        dtype (dtype): the dtype of the final output array
        manual_coercion (boolean, optional): whether to manually coerce the
        non-null elements of the output array to the dtype

    Returns:
        scalar or Series: the result of applying scalar_fn to each row of the
        vectors with scalar args broadcasted (or just the scalar output if
        all of the arguments are scalar)
    """
    length = -1
    for arg in args:
        if isinstance(arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray)):
            length = len(arg)
            break
    if length == -1:
        return dtype(scalar_fn(*args)) if manual_coercion else scalar_fn(*args)
    arglist = []
    for arg in args:
        if isinstance(arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray)):
            arglist.append(arg)
        else:
            arglist.append([arg] * length)
    if manual_coercion:
        return pd.Series([dtype(scalar_fn(*params)) for params in zip(*arglist)])
    else:
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
    err_msg3 = re.escape("LPAD arr argument must be a string, string column, or nul")
    err_msg4 = re.escape(
        "RPAD length argument must be an integer, integer column, or null"
    )
    err_msg5 = re.escape(
        "RPAD rpad_string argument must be a string, string column, or null"
    )
    err_msg6 = re.escape("RPAD arr argument must be a string, string column, or nul")

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


@pytest.fixture(
    params=[
        pytest.param(
            pd.concat(
                [
                    pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
                    pd.Series([None, None]),
                    pd.Series(pd.date_range("1970-01-01", "2108-01-01", periods=20)),
                ]
            ),
            id="vector",
        ),
        pytest.param(pd.Timestamp("2000-10-29"), id="scalar"),
    ],
)
def dates_scalar_vector(request):
    return request.param


def test_last_day(dates_scalar_vector):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.last_day(arr)

    # Simulates LAST_DAY on a single row
    def last_day_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return np.datetime64(
                elem + pd.tseries.offsets.MonthEnd(n=0, normalize=True)
            )

    last_day_answer = vectorized_sol((dates_scalar_vector,), last_day_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=last_day_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_dayname(dates_scalar_vector):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.dayname(arr)

    # Simulates DAYNAME on a single row
    def dayname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.day_name()

    dayname_answer = vectorized_sol((dates_scalar_vector,), dayname_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=dayname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_monthname(dates_scalar_vector):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.monthname(arr)

    # Simulates MONTHNAME on a single row
    def monthname_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.month_name()

    monthname_answer = vectorized_sol((dates_scalar_vector,), monthname_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=monthname_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_weekday(dates_scalar_vector):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.weekday(arr)

    # Simulates WEEKDAY on a single row
    def weekday_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.weekday()

    weekday_answer = vectorized_sol((dates_scalar_vector,), weekday_scalar_fn, None)
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=weekday_answer,
        check_dtype=False,
        reset_index=True,
    )


def test_yearofweekiso(dates_scalar_vector):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.yearofweekiso(arr)

    # Simulates YEAROFWEEKISO on a single row
    def yearofweekiso_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem.isocalendar()[0]

    yearofweekiso_answer = vectorized_sol(
        (dates_scalar_vector,), yearofweekiso_scalar_fn, None
    )
    check_func(
        impl,
        (dates_scalar_vector,),
        py_output=yearofweekiso_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array([2001, 2002, 2003, 2004, 2005, None, 2007])),
                pd.Series(pd.array([None, 32, 90, 180, 150, 365, 225])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                2007,
                pd.Series(pd.array([1, 10, 40, None, 80, 120, 200, 350, 360, None])),
            ),
            id="scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((2018, 300), id="all_scalar"),
    ],
)
def test_makedate(args):
    def impl(year, day):
        return bodo.libs.bodosql_array_kernels.makedate(year, day)

    # Simulates MAKEDATE on a single row
    def makedate_scalar_fn(year, day):
        if pd.isna(year) or pd.isna(day):
            return None
        else:
            return np.datetime64(
                pd.Timestamp(year=year, month=1, day=1)
                + pd.Timedelta(day - 1, unit="D")
            )

    makedate_answer = vectorized_sol(args, makedate_scalar_fn, None)
    check_func(
        impl,
        args,
        py_output=makedate_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_calendar_optional():
    def impl(A, B, C, flag0, flag1, flag2):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        return (
            bodo.libs.bodosql_array_kernels.last_day(arg0),
            bodo.libs.bodosql_array_kernels.dayname(arg0),
            bodo.libs.bodosql_array_kernels.monthname(arg0),
            bodo.libs.bodosql_array_kernels.weekday(arg0),
            bodo.libs.bodosql_array_kernels.yearofweekiso(arg0),
            bodo.libs.bodosql_array_kernels.makedate(arg1, arg2),
        )

    A, B, C = pd.Timestamp("2018-04-01"), 2005, 365
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                a0 = np.datetime64("2018-04-30") if flag0 else None
                a1 = "Sunday" if flag0 else None
                a2 = "April" if flag0 else None
                a3 = 6 if flag0 else None
                a4 = 2018 if flag0 else None
                a5 = np.datetime64("2005-12-31") if flag1 and flag2 else None
                check_func(
                    impl,
                    (A, B, C, flag0, flag1, flag2),
                    py_output=(a0, a1, a2, a3, a4, a5),
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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is 🟥🟧🟨🟩🟦🟪",
                            "so very very delicious",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array([5, -5, 3, -8, 10, 20, 1])),
                pd.Series(pd.array([10, 5, 12, 4, 2, 5, -1])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo 🟥🟧🟨🟩🟦🟪",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array([0, 1, -2, 4, -8, 16, -32])),
                5,
            ),
            id="scalar_vector_mix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is 🟥🟧🟨🟩🟦🟪 so very delicious", 10, 8),
            id="all_scalar_no_null",
        ),
        pytest.param(
            ("alphabet soup is so very delicious", None, 8),
            id="all_scalar_some_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_substring(args):
    def impl(arr, start, length):
        return bodo.libs.bodosql_array_kernels.substring(arr, start, length)

    # Simulates SUBSTRING on a single row
    def substring_scalar_fn(elem, start, length):
        if pd.isna(elem) or pd.isna(start) or pd.isna(length):
            return None
        elif length <= 0:
            return ""
        elif start < 0 and start + length >= 0:
            return elem[start:]
        else:
            if start > 0:
                start -= 1
            return elem[start : start + length]

    arr, start, length = args
    substring_answer = vectorized_sol(
        (arr, start, length), substring_scalar_fn, pd.StringDtype()
    )
    check_func(
        impl,
        (arr, start, length),
        py_output=substring_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious 🟥🟧🟨🟩🟦🟪",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                pd.Series(pd.array(["a", "b", "e", " ", " ", "o", "r"])),
                pd.Series(pd.array([1, 4, 3, 0, 1, -1, None])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "alphabet soup is",
                            "so very very delicious 🟥🟧🟨🟩🟦🟪",
                            "aaeaaeieaaeioiea",
                            "alpha beta gamma delta epsilon",
                            None,
                            "foo",
                            "bar",
                        ]
                    )
                ),
                " ",
                pd.Series(pd.array([1, 2, -1, 4, 5, 1, 0])),
            ),
            id="scalar_vector_mix",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is so very delicious", "o", 3),
            id="all_scalar_no_null",
        ),
        pytest.param(
            ("alphabet soup is so very delicious", None, 3),
            id="all_scalar_some_null",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_substring_index(args):
    def impl(arr, delimiter, occurrences):
        return bodo.libs.bodosql_array_kernels.substring_index(
            arr, delimiter, occurrences
        )

    # Simulates SUBSTRING_INDEX on a single row
    def substring_index_scalar_fn(elem, delimiter, occurrences):
        if pd.isna(elem) or pd.isna(delimiter) or pd.isna(occurrences):
            return None
        elif delimiter == "" or occurrences == 0:
            return ""
        elif occurrences >= 0:
            return delimiter.join(elem.split(delimiter)[:occurrences])
        else:
            return delimiter.join(elem.split(delimiter)[occurrences:])

    arr, delimiter, occurrences = args
    substring_index_answer = vectorized_sol(
        (arr, delimiter, occurrences), substring_index_scalar_fn, pd.StringDtype()
    )
    check_func(
        impl,
        (arr, delimiter, occurrences),
        py_output=substring_index_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_substring():
    def impl(A, B, C, D, E, flag0, flag1, flag2, flag3, flag4):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        arg3 = D if flag3 else None
        arg4 = E if flag4 else None
        return (
            bodo.libs.bodosql_array_kernels.substring(arg0, arg1, arg2),
            bodo.libs.bodosql_array_kernels.substring_index(arg0, arg3, arg4),
        )

    A, B, C, D, E = "alpha beta gamma", 7, 4, " ", 1
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    for flag4 in [True, False]:
                        a0 = "beta" if flag0 and flag1 and flag2 else None
                        a1 = "alpha" if flag0 and flag3 and flag4 else None
                        check_func(
                            impl,
                            (A, B, C, D, E, flag0, flag1, flag2, flag3, flag4),
                            py_output=(a0, a1),
                        )


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
def test_option_with_arr_coalesce():
    """tests coalesce behavior with optionals when suplied an array argument"""

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
def test_option_no_arr_coalesce():
    """tests coalesce behavior with optionals when suplied no array argument"""

    def impl1(scale1, scale2, flag1, flag2):
        A = scale1 if flag1 else None
        B = scale2 if flag2 else None
        return bodosql_array_kernels.coalesce((A, B))

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

    err_msg1 = re.escape("Cannot call COALESCE on columns with different dtypes")
    err_msg2 = re.escape("Coalesce argument must be a tuple")
    err_msg3 = re.escape("Cannot coalesce 0 columns")

    A = pd.Series(["A", "B", "C", "D", "E"])
    B = pd.Series(["D", "E", "F"] + gen_nonascii_list(2))
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
        pytest.param(
            (
                pd.Series(gen_nonascii_list(6)),
                None,
            ),
            id="nonascii_vector_null",
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
    err_msg2 = re.escape("LEFT arr argument must be a string, string column, or null")
    err_msg3 = re.escape(
        "RIGHT n_chars argument must be an integer, integer column, or null"
    )
    err_msg4 = re.escape("RIGHT arr argument must be a string, string column, or null")

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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["A", "BCD", "EFGH🐍", None, "I", "J"])),
                pd.Series(pd.array([2, 6, -1, 3, None, 3])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (pd.Series(pd.array(["", "A✓", "BC", "DEF", "GHIJ", None])), 10),
            id="vector_scalar",
        ),
        pytest.param(
            ("Ʃ = alphabet", pd.Series(pd.array([-5, 0, 1, 5, 2]))),
            id="scalar_vector",
        ),
        pytest.param(("racecars!", 4), id="all_scalar_no_null"),
        pytest.param((None, None), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_repeat(args):
    def impl(arr, repeats):
        return bodo.libs.bodosql_array_kernels.repeat(arr, repeats)

    # Simulates REPEAT on a single row
    def repeat_scalar_fn(elem, repeats):
        if pd.isna(elem) or pd.isna(repeats):
            return None
        else:
            return elem * repeats

    strings, numbers = args
    repeat_answer = vectorized_sol(
        (strings, numbers), repeat_scalar_fn, pd.StringDtype()
    )
    check_func(
        impl,
        (strings, numbers),
        py_output=repeat_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "numbers",
    [
        pytest.param(
            pd.Series(pd.array([2, 6, -1, 3, None, 3])),
            id="vector",
        ),
        pytest.param(
            4,
            id="scalar",
        ),
    ],
)
def test_space(numbers):
    def impl(n_chars):
        return bodo.libs.bodosql_array_kernels.space(n_chars)

    # Simulates SPACE on a single row
    def space_scalar_fn(n_chars):
        if pd.isna(n_chars):
            return None
        else:
            return " " * n_chars

    space_answer = vectorized_sol((numbers,), space_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (numbers,),
        py_output=space_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "strings",
    [
        pytest.param(
            pd.Series(pd.array(["A", "BƬCD", "EFGH", None, "I", "J✖"])),
            id="vector",
        ),
        pytest.param("racecarsƟ", id="scalar"),
        pytest.param(
            pd.Series(pd.array(gen_nonascii_list(6))),
            id="vector",
        ),
        pytest.param(gen_nonascii_list(1)[0], id="scalar"),
    ],
)
def test_reverse(strings):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.reverse(arr)

    # Simulates REVERSE on a single row
    def reverse_scalar_fn(elem):
        if pd.isna(elem):
            return None
        else:
            return elem[::-1]

    reverse_answer = vectorized_sol((strings,), reverse_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (strings,),
        py_output=reverse_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["alphabet", "s🟦o🟦u🟦p", "is", "delicious", None])),
                pd.Series(pd.array(["a", "", "4", "ic", " "])),
                pd.Series(pd.array(["_", "X", "5", "", "."])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(
                    pd.array(
                        [
                            "i'd like to buy",
                            "the world a coke",
                            "and",
                            None,
                            "keep it company",
                        ]
                    )
                ),
                pd.Series(pd.array(["i", " ", "", "$", None])),
                "🟩",
            ),
            id="vector_vector_scalar",
        ),
        pytest.param(
            (
                pd.Series(pd.array(["oohlala", "books", "oooo", "ooo", "ooohooooh"])),
                "oo",
                pd.Series(pd.array(["", "OO", "*", "#O#", "!"])),
            ),
            id="vector_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "♪♪♪ I'd like to teach the world to sing ♫♫♫",
                " ",
                pd.Series(pd.array(["_", "  ", "", ".", None])),
            ),
            id="scalar_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("alphabet soup is so very delicious", "so", "SO"), id="all_scalar_no_null"
        ),
        pytest.param(
            ("Alpha", None, "Beta"), id="all_scalar_with_null", marks=pytest.mark.slow
        ),
    ],
)
def test_replace(args):
    def impl(arr, to_replace, replace_with):
        return bodo.libs.bodosql_array_kernels.replace(arr, to_replace, replace_with)

    # Simulates REPLACE on a single row
    def replace_scalar_fn(elem, to_replace, replace_with):
        if pd.isna(elem) or pd.isna(to_replace) or pd.isna(replace_with):
            return None
        elif to_replace == "":
            return elem
        else:
            return elem.replace(to_replace, replace_with)

    replace_answer = vectorized_sol(args, replace_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        args,
        py_output=replace_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_reverse_repeat_replace_space():
    def impl(A, B, C, D, flag0, flag1, flag2, flag3):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        arg2 = C if flag2 else None
        arg3 = D if flag3 else None
        return (
            bodo.libs.bodosql_array_kernels.reverse(arg0),
            bodo.libs.bodosql_array_kernels.replace(arg0, arg1, arg2),
            bodo.libs.bodosql_array_kernels.repeat(arg2, arg3),
            bodo.libs.bodosql_array_kernels.space(arg3),
        )

    A, B, C, D = "alphabet soup", "a", "_", 4
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            for flag2 in [True, False]:
                for flag3 in [True, False]:
                    a0 = "puos tebahpla" if flag0 else None
                    a1 = "_lph_bet soup" if flag0 and flag1 and flag2 else None
                    a2 = "____" if flag2 and flag3 else None
                    a3 = "    " if flag3 else None
                    check_func(
                        impl,
                        (A, B, C, D, flag0, flag1, flag2, flag3),
                        py_output=(a0, a1, a2, a3),
                    )


@pytest.mark.parametrize(
    "s",
    [
        pytest.param(
            pd.Series(pd.array(["alphabet", "ɲɳ", "Ʃ=sigma", "", " yay "])),
            id="vector",
        ),
        pytest.param(
            "Apple",
            id="scalar",
        ),
    ],
)
def test_ord_ascii(s):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.ord_ascii(arr)

    # Simulates ORD/ASCII on a single row
    def ord_ascii_scalar_fn(elem):
        if pd.isna(elem) or len(elem) == 0:
            return None
        else:
            return ord(elem[0])

    ord_answer = vectorized_sol((s,), ord_ascii_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        (s,),
        py_output=ord_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "n",
    [
        pytest.param(
            pd.Series(pd.array([65, 100, 110, 0, 33])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_char(n):
    def impl(arr):
        return bodo.libs.bodosql_array_kernels.char(arr)

    # Simulates CHAR on a single row
    def char_scalar_fn(elem):
        if pd.isna(elem) or elem < 0 or elem > 127:
            return None
        else:
            return chr(elem)

    chr_answer = vectorized_sol((n,), char_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (n,),
        py_output=chr_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_ord_ascii_char():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.ord_ascii(arg0),
            bodo.libs.bodosql_array_kernels.char(arg1),
        )

    A, B = "A", 97
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            a0 = 65 if flag0 else None
            a1 = "a" if flag1 else None
            check_func(impl, (A, B, flag0, flag1), py_output=(a0, a1))


@pytest.mark.parametrize(
    "days",
    [
        pytest.param(
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_int_to_days(days):
    def impl(days):
        return bodo.libs.bodosql_array_kernels.int_to_days(days)

    # Simulates int_to_days on a single row
    def itd_scalar_fn(days):
        if pd.isna(days):
            return None
        else:
            return np.timedelta64(days, "D")

    itd_answer = vectorized_sol(
        (days,), itd_scalar_fn, np.timedelta64, manual_coercion=True
    )
    check_func(
        impl,
        (days,),
        py_output=itd_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_int_to_days():
    def impl(A, flag):
        arg = A if flag else None
        return bodo.libs.bodosql_array_kernels.int_to_days(arg)

    for flag in [True, False]:
        answer = np.timedelta64(pd.Timedelta(days=10)) if flag else None
        check_func(impl, (10, flag), py_output=answer)


@pytest.mark.parametrize(
    "seconds",
    [
        pytest.param(
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32, 100000])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_second_timestamp(seconds):
    def impl(seconds):
        return bodo.libs.bodosql_array_kernels.second_timestamp(seconds)

    # Simulates second_timestamp on a single row
    def second_scalar_fn(seconds):
        if pd.isna(seconds):
            return None
        else:
            return pd.Timestamp(seconds, unit="s")

    second_answer = vectorized_sol(
        (seconds,), second_scalar_fn, np.datetime64, manual_coercion=True
    )
    check_func(
        impl,
        (seconds,),
        py_output=second_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "days",
    [
        pytest.param(
            pd.Series(pd.array([0, 1, -2, 4, 8, None, -32, 10000])),
            id="vector",
        ),
        pytest.param(
            42,
            id="scalar",
        ),
    ],
)
def test_day_timestamp(days):
    def impl(days):
        return bodo.libs.bodosql_array_kernels.day_timestamp(days)

    # Simulates day_timestamp on a single row
    def days_scalar_fn(days):
        if pd.isna(days):
            return None
        else:
            return pd.Timestamp(days, unit="D")

    days_answer = vectorized_sol(
        (days,), days_scalar_fn, np.datetime64, manual_coercion=True
    )
    check_func(
        impl,
        (days,),
        py_output=days_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_timestamp():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.second_timestamp(arg0),
            bodo.libs.bodosql_array_kernels.day_timestamp(arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            A0 = np.datetime64(pd.Timestamp(1000000, unit="s")) if flag0 else None
            A1 = np.datetime64(pd.Timestamp(10000, unit="D")) if flag1 else None
            check_func(
                impl,
                (1000000, 10000, flag0, flag1),
                py_output=(A0, A1),
            )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.concat(
                    [
                        pd.Series(
                            pd.date_range("2018-01-01", "2019-01-01", periods=20)
                        ),
                        pd.Series([None, None]),
                    ]
                ),
                pd.Series(pd.date_range("2005-01-01", "2020-01-01", periods=22)),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                pd.Series(pd.date_range("2018-01-01", "2019-01-01", periods=20)),
                pd.Timestamp("2018-06-05"),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.Timestamp("2000-10-29"), pd.Timestamp("1992-03-25")), id="all_scalar"
        ),
    ],
)
def test_month_diff(args):
    def impl(arr0, arr1):
        return bodo.libs.bodosql_array_kernels.month_diff(arr0, arr1)

    # Simulates month diff on a single row
    def md_scalar_fn(ts1, ts2):
        if pd.isna(ts1) or pd.isna(ts2):
            return None
        else:
            floored_delta = (ts1.year - ts2.year) * 12 + (ts1.month - ts2.month)
            remainder = ((ts1 - pd.DateOffset(months=floored_delta)) - ts2).value
            remainder = 1 if remainder > 0 else (-1 if remainder < 0 else 0)
            if floored_delta > 0 and remainder < 0:
                actual_month_delta = floored_delta - 1
            elif floored_delta < 0 and remainder > 0:
                actual_month_delta = floored_delta + 1
            else:
                actual_month_delta = floored_delta
            return -actual_month_delta

    days_answer = vectorized_sol(args, md_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=days_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_month_diff():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.month_diff(arg0, arg1)

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = 42 if flag0 and flag1 else None
            check_func(
                impl,
                (pd.Timestamp("2007-01-01"), pd.Timestamp("2010-07-04"), flag0, flag1),
                py_output=answer,
            )


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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                np.array(
                    [
                        15.112345,
                        1234567890,
                        np.NAN,
                        17,
                        -13.6413,
                        1.2345,
                        12345678910111213.141516171819,
                    ]
                ),
                pd.Series(pd.array([3, 4, 6, None, 0, -1, 5])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                12345678.123456789,
                pd.Series(pd.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),
            ),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param((-426472, 2), id="all_scalar_not_null"),
        pytest.param((None, 5), id="all_scalar_with_null", marks=pytest.mark.slow),
    ],
)
def test_format(args):
    def impl(arr, places):
        return bodo.libs.bodosql_array_kernels.format(arr, places)

    # Simulates FORMAT on a single row
    def format_scalar_fn(elem, places):
        if pd.isna(elem) or pd.isna(places):
            return None
        elif places <= 0:
            return "{:,}".format(round(elem))
        else:
            return (f"{{:,.{places}f}}").format(elem)

    arr, places = args
    format_answer = vectorized_sol((arr, places), format_scalar_fn, pd.StringDtype())
    check_func(
        impl,
        (arr, places),
        py_output=format_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_option_format():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return bodo.libs.bodosql_array_kernels.format(arg0, arg1)

    A, B = 12345678910.111213, 4
    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = "12,345,678,910.1112" if flag0 and flag1 else None
            check_func(impl, (A, B, flag0, flag1), py_output=answer)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["ABC", "25", "X", None, "A"])),
                pd.Series(pd.array(["abc", "123", "X", "B", None])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (pd.Series(pd.array(["ABC", "ACB", "ABZ", "AZB", "ACE", "ACX"])), "ACE"),
            id="vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(("alphabet", "soup"), id="all_scalar"),
    ],
)
def test_strcmp(args):
    def impl(arr0, arr1):
        return bodo.libs.bodosql_array_kernels.strcmp(arr0, arr1)

    # Simulates STRCMP on a single row
    def strcmp_scalar_fn(arr0, arr1):
        if pd.isna(arr0) or pd.isna(arr1):
            return None
        else:
            return -1 if arr0 < arr1 else (1 if arr0 > arr1 else 0)

    strcmp_answer = vectorized_sol(args, strcmp_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=strcmp_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.Series(pd.array(["alpha", "beta", "gamma", None, "epsilon"])),
                pd.Series(pd.array(["a", "b", "c", "t", "n"])),
            ),
            id="all_vector",
        ),
        pytest.param(
            (
                "alphabet soup is delicious",
                pd.Series(pd.array([" ", "ici", "x", "i", None])),
            ),
            id="scalar_vector",
        ),
        pytest.param(
            ("The quick brown fox jumps over the lazy dog", "x"),
            id="all_scalar",
        ),
    ],
)
def test_instr(args):
    def impl(arr0, arr1):
        return bodo.libs.bodosql_array_kernels.instr(arr0, arr1)

    # Simulates INSTR on a single row
    def instr_scalar_fn(elem, target):
        if pd.isna(elem) or pd.isna(target):
            return None
        else:
            return elem.find(target) + 1

    instr_answer = vectorized_sol(args, instr_scalar_fn, pd.Int32Dtype())
    check_func(
        impl,
        args,
        py_output=instr_answer,
        check_dtype=False,
        reset_index=True,
    )


@pytest.mark.slow
def test_strcmp_instr_option():
    def impl(A, B, flag0, flag1):
        arg0 = A if flag0 else None
        arg1 = B if flag1 else None
        return (
            bodo.libs.bodosql_array_kernels.strcmp(arg0, arg1),
            bodo.libs.bodosql_array_kernels.instr(arg0, arg1),
        )

    for flag0 in [True, False]:
        for flag1 in [True, False]:
            answer = (1, 0) if flag0 and flag1 else None
            check_func(impl, ("a", "Z", flag0, flag1), py_output=answer)


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
def test_negate_option():
    def impl(A, flag0):
        arg = A if flag0 else None
        return bodo.libs.bodosql_array_kernels.negate(arg)

    for flag0 in [True, False]:
        answer = -42 if flag0 else None
        check_func(impl, (42, flag0), py_output=answer)


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
