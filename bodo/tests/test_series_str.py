# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import itertools
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


def test_extract():
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd])(?P<C>\d+)")

    S = pd.Series(
        ["a1", "b1", np.nan, "a2", "c2", "ddd", "d1", "d222"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    check_func(test_impl, (S,))


def test_extract_noexpand():
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd]+)\d+", expand=False)

    # when regex group has no name, Series name should be used
    def test_impl_noname(S):
        return S.str.extract(r"([abd]+)\d+", expand=False)

    S = pd.Series(
        ["a1", "b1", np.nan, "a2", "cc2", "ddd", "ddd1", "d222"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    check_func(test_impl, (S,))
    check_func(test_impl_noname, (S,))


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


def test_pad_fill_fast():
    # this function increases coverage for not slow test suite
    def test_impl1(S):
        return S.str.center(1, "a")
    def test_impl2(S):
        return S.str.rjust(1, "a")
    def test_impl3(S):
        return S.str.ljust(1, "a")
    def test_impl4(S):
        return S.str.pad(1, "left", "a")
    def test_impl5(S):
        return S.str.zfill(1)

    S = pd.Series(
        ["AB,C", "AB", "A", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl1, (S,))
    check_func(test_impl2, (S,))
    check_func(test_impl3, (S,))
    check_func(test_impl4, (S,))
    check_func(test_impl5, (S,))


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


@pytest.mark.parametrize("ind", [slice(2), 2])
def test_getitem(ind):
    def test_impl(S, ind):
        return S.str[ind]

    S = pd.Series(
        ["ABCDDCABABAAB", "ABBD", "AA", "C,ABB, D", np.nan], [3, 5, 1, 0, 2], name="A"
    )
    check_func(test_impl, (S, ind))


##############  list of string array tests  #################


@pytest.fixture(
    params=[
        pytest.param(
            np.array([["a", "bc"], ["a"], ["aaa", "b", "cc"]] * 2),
            marks=pytest.mark.slow,
        ),
        # empty strings, empty lists, NA
        pytest.param(
            np.array([["a", "bc"], ["a"], [], ["aaa", "", "cc"], [""], np.nan] * 2),
            marks=pytest.mark.slow,
        ),
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


def test_flatten1():
    """tests flattening array of string lists after split call when split view
    optimization is applied
    """

    def impl(S):
        A = S.str.split(",")
        return pd.Series(list(itertools.chain(*A)))

    S = pd.Series(
        ["AB,CC", "C,ABB,D", "CAD", "CA,D", "AA,,D"], [3, 1, 2, 0, 4], name="A"
    )
    check_func(impl, (S,))


def test_flatten2():
    """tests flattening array of string lists after split call when split view
    optimization is not applied
    """

    def impl(S):
        A = S.str.split()
        return pd.Series(list(itertools.chain(*A)))

    S = pd.Series(
        ["AB  CC", "C ABB  D", "CAD", "CA\tD", "AA\t\tD"], [3, 1, 2, 0, 4], name="A"
    )
    check_func(impl, (S,))


def test_join():
    """test the functionality of bodo's join with NaN
    """

    def test_impl(S):
        return S.str.join("-")

    S = pd.Series(
        [
            [
                "AAAAAA",
                "BERQBBBBB",
                "1111ASDDDDDDD11",
                "222222TTTTTTT",
                "CCCQWEQWEQWEQWEWQEQWEQWECCC",
            ],
            np.nan,
            ["KKKKKK", "LALLLLL", "MMMQWEMMM", "APPQWEQWEPPP", "!@###@@^%$%$#"],
            np.nan,
            ["1234567", "QWERQWER", "HAPPYCODING", "%)(*&&*())", "{}_)(*#(#))"],
        ]
    )
    check_func(test_impl, (S,))


@pytest.fixture(
    params=[
        pd.Series(["ABCDEFGH", "1", "AB", "ABC", "!@##@!##!@#!@#@!#!$!@$"]),
        pd.Series(["123456789"] * 5),
        pd.Series(["ABCDEFGH"] * 1000),
    ]
)
def test_sr(request):
    return request.param


def test_join_string(test_sr):
    """test the functionality of bodo's join with just a string
    """

    def test_impl(test_sr):
        return test_sr.str.join("-")

    def test_impl2(test_sr):
        return test_sr.str.join("*****************")

    check_func(test_impl, (test_sr,))
    check_func(test_impl2, (test_sr,))


def test_join_splitview():
    """test the functionality of bodo's join with split view type as an input
    """

    def test_impl(S):
        B = S.str.split(",")
        return B.str.join("-")

    S = pd.Series(["AB,CC", "C,ABB,D", "LLL,JJ", "C,D", "C,ABB,D"], name="A")

    check_func(test_impl, (S,))
