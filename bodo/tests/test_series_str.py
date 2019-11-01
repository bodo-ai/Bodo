# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import pandas as pd
import numpy as np
import pytest

import numba
import bodo
from bodo.tests.utils import check_func, _test_equal


def test_len():
    def test_impl(S):
        return S.str.len()

    S = pd.Series(
        [" bbCD ", "ABC", " mCDm ", np.nan, "abcffcc", "", "A"],
        [4, 3, 5, 1, 0, -3, 2],
        name="A",
    )
    check_func(test_impl, (S,), check_dtype=False)


def test_split():
    def test_impl(S):
        return S.str.split(",")

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    S = pd.Series(
        ["ABCC", "ABBD", "AA", "C,ABB, D", "B,B,CC"], [3, 5, 1, 0, 2], name="A"
    )
    # TODO: support distributed
    # check_func(test_impl, (S,))
    pd.testing.assert_series_equal(bodo.jit(test_impl)(S), test_impl(S))


def test_get():
    def test_impl(S):
        B = S.str.split(",")
        return B.str.get(1)

    # TODO: support and test NA
    S = pd.Series(
        ["AB,CC", "C,ABB,D", "LLL,JJ", "C,D", "C,ABB,D"], [4, 3, 5, 1, 0], name="A"
    )
    # TODO: support distributed
    # check_func(test_impl, (S,))
    pd.testing.assert_series_equal(bodo.jit(test_impl)(S), test_impl(S))


def test_replace_regex():
    def test_impl(S):
        return S.str.replace("AB*", "EE", regex=True)

    S = pd.Series(
        ["ABCC", "CABBD", np.nan, "CCD", "C,ABB,D"], [4, 3, 5, 1, 0], name="A"
    )
    check_func(test_impl, (S,))


def test_replace_noregex():
    def test_impl(S):
        return S.str.replace("AB", "EE", regex=False)

    S = pd.Series(["ABCC", "CABBD", np.nan, "AA", "C,ABB,D"], [4, 3, 5, 1, 0], name="A")
    check_func(test_impl, (S,))


def test_contains_regex():
    def test_impl(S):
        return S.str.contains("AB*", regex=True)

    S = pd.Series(
        ["ABCC", "CABBD", np.nan, "AA", "C,ABB,D", "AAcB", "BBC", "AbC"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="A",
    )
    check_func(test_impl, (S,))


def test_contains_noregex():
    def test_impl(S):
        return S.str.contains("AB", regex=False)

    S = pd.Series(
        ["ABCC", "CABBD", np.nan, "AA", "C,ABB,D", "AAcB", "BBC", "AbC"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="A",
    )
    check_func(test_impl, (S,))


def test_count_noflag():
    def test_impl(S):
        return S.str.count("A")

    S = pd.Series(
        ["AAABCC", "CABBD", np.nan, "AA", "C,ABB,D"], [4, 3, 5, 1, 0], name="A"
    )
    check_func(test_impl, (S,), check_dtype=False)


def test_count_flag():
    import re

    # TODO: the flag does not work inside numba
    flag = re.IGNORECASE.value

    def test_impl(S):
        return S.str.count("A", flag)

    S = pd.Series(
        ["AAABCC", "CABBD", np.nan, "Aaba", "C,BB,D"], [4, 3, 5, 1, 0], name="A"
    )
    check_func(test_impl, (S,), check_dtype=False)


def test_find():
    def test_impl(S):
        return S.str.find("AB")

    S = pd.Series(["ABCC", "CABBD", np.nan, "AA", "C,ABB,D"], [4, 3, 5, 1, 0], name="A")
    check_func(test_impl, (S,), check_dtype=False)


def test_rfind():
    def test_impl(S):
        return S.str.rfind("AB")

    S = pd.Series(
        ["ABCC", "CABBDAB", np.nan, "ABAB", "C,BB,D"], [4, 3, 5, 1, 0], name="A"
    )
    check_func(test_impl, (S,), check_dtype=False)


@pytest.mark.slow
def test_center():
    def test_impl(S):
        return S.str.center(5, "*")

    S = pd.Series(
        ["ABCDDC", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_ljust():
    def test_impl(S):
        return S.str.ljust(5, "*")

    S = pd.Series(
        ["ABCDDC", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_rjust():
    def test_impl(S):
        return S.str.rjust(5, "*")

    S = pd.Series(
        ["ABCDDC", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S,))


@pytest.mark.slow
def test_pad():
    def test_impl_default(S):
        return S.str.pad(5)

    def test_impl_left(S):
        return S.str.pad(5, "left", "*")

    def test_impl_right(S):
        return S.str.pad(5, "right", "*")

    def test_impl_both(S):
        return S.str.pad(5, "both", "*")

    S = pd.Series(
        ["ABCDDC", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl_default, (S,))
    check_func(test_impl_left, (S,))
    check_func(test_impl_right, (S,))
    check_func(test_impl_both, (S,))


@pytest.mark.slow
def test_zfill():
    def test_impl(S):
        return S.str.zfill(10)

    S = pd.Series(
        ["ABCDDCABABAAB", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S,))


def test_slice():
    def test_impl(S):
        return S.str.slice(step=2)

    S = pd.Series(
        ["ABCDDCABABAAB", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S,))


def test_startswith():
    def test_impl(S):
        return S.str.startswith("AB")

    S = pd.Series(
        ["AB", "ABb", "abab", "C,ABB, D", np.nan, "AA", "abc", "ABCa"],
        [3, 5, 1, 0, 2, 4, 6, 7],
        name="A",
    )
    check_func(test_impl, (S,))


def test_endswith():
    def test_impl(S):
        return S.str.startswith("AB")

    S = pd.Series(
        ["AB", "ABb", "abab", "C,ABB, D", np.nan, "AA", "abc", "ABCa"],
        [3, 5, 1, 0, 2, 4, 6, 7],
        name="A",
    )
    check_func(test_impl, (S,))


def test_isupper():
    def test_impl(S):
        return S.str.isupper()

    S = pd.Series(
        ["AB", "ABb", "abab", "C,ABB, D", np.nan, "AA", "abc", "ABCa"],
        [3, 5, 1, 0, 2, 4, 6, 7],
        name="A",
    )
    check_func(test_impl, (S,))


##############  list of string array tests  #################


@pytest.fixture(
    params=[
        np.array([["a", "bc"], ["a"], ["aaa", "b", "cc"]] * 2),
        # empty strings, empty lists, NA
        np.array([["a", "bc"], ["a"], [], ["aaa", "", "cc"], [""], np.nan] * 2),
        # large array
        np.array([["a", "bc"], ["a"], [], ["aaa", "", "cc"], [""], np.nan] * 1000),
    ]
)
def list_str_arr_value(request):
    return request.param


def test_list_str_arr_unbox(list_str_arr_value):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (list_str_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (list_str_arr_value,))


def test_getitem_int(list_str_arr_value):
    def test_impl(A, i):
        return A[i]

    bodo_func = bodo.jit(test_impl)
    i = 2
    assert bodo_func(list_str_arr_value, i) == test_impl(list_str_arr_value, i)


def test_getitem_bool(list_str_arr_value):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    np.random.seed(0)
    ind = np.random.ranf(len(list_str_arr_value)) < 0.2
    # TODO: parallel test
    _test_equal(bodo_func(list_str_arr_value, ind), test_impl(list_str_arr_value, ind))


def test_getitem_slice(list_str_arr_value):
    def test_impl(A, ind):
        return A[ind]

    bodo_func = bodo.jit(test_impl)
    ind = slice(1, 4)
    # TODO: parallel test
    _test_equal(bodo_func(list_str_arr_value, ind), test_impl(list_str_arr_value, ind))


def test_copy(list_str_arr_value):
    def test_impl(A):
        return A.copy()

    _test_equal(bodo.jit(test_impl)(list_str_arr_value), list_str_arr_value)
