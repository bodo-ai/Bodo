# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Test Bodo's array kernel utilities for BodoSQL string functions
"""


import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.libs.bodosql_array_kernels import *
from bodo.tests.utils import check_func, gen_nonascii_list


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


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            (
                pd.array(["alpha", "beta", "gamma", "delta", "epsilon"]),
                pd.array([2, 4, 8, 16, 32]),
                pd.array(["_", "_", "_", "AB", "123"]),
            ),
            id="all_vector_no_null",
        ),
        pytest.param(
            (
                pd.array([None, "words", "words", "words", "words", "words"]),
                pd.array([16, None, 16, 0, -5, 16]),
                pd.array(["_", "_", None, "_", "_", ""]),
            ),
            id="all_vector_with_null",
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, "_"),
            id="vector_scalar_scalar_A",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 0, "_"),
            id="vector_scalar_scalar_B",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, ""),
            id="vector_sscalar_scalar_C",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), None, "_"),
            id="vector_null_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (pd.array(["alpha", "beta", "gamma", "delta", "epsilon", None]), 20, None),
            id="vector_scalar_null",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            ("words", 20, "0123456789"), id="all_scalar_no_null", marks=pytest.mark.slow
        ),
        pytest.param(
            (None, 20, "0123456789"), id="all_scalar_with_null", marks=pytest.mark.slow
        ),
        pytest.param(
            ("words", pd.array([2, 4, 8, 16, 32]), "0123456789"),
            id="scalar_vector_scalar",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (None, 20, pd.array(["A", "B", "C", "D", "E"])),
            id="null_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "words",
                30,
                pd.array(["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "", None]),
            ),
            id="scalar_scalar_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            (
                "words",
                pd.array([-10, 0, 10, 20, 30]),
                pd.array([" ", " ", " ", "", None]),
            ),
            id="scalar_vector_vector",
            marks=pytest.mark.slow,
        ),
        pytest.param((None, None, None), id="all_null", marks=pytest.mark.slow),
        pytest.param(
            (
                pd.array(["A", "B", "C", "D", "E"]),
                pd.Series([2, 4, 6, 8, 10]),
                pd.Series(["_"] * 5),
            ),
            id="series_test",
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
def test_option_char_ord_ascii():
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
def test_option_lpad_rpad():
    def impl1(arr, length, lpad_string, flag1, flag2):
        B = length if flag1 else None
        C = lpad_string if flag2 else None
        return bodo.libs.bodosql_array_kernels.lpad(arr, B, C)

    def impl2(val, length, lpad_string, flag1, flag2, flag3):
        A = val if flag1 else None
        B = length if flag2 else None
        C = lpad_string if flag3 else None
        return bodo.libs.bodosql_array_kernels.rpad(A, B, C)

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
