# Copyright (C) 2019 Bodo Inc. All rights reserved.
import itertools
import pandas as pd
import numpy as np
import pytest

import bodo
from bodo.tests.utils import check_func, _test_equal


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    "ABCDD,OSAJD",
                    "a1b2d314f,sdf234",
                    "22!@#,$@#$",
                    np.nan,
                    "A,C,V,B,B",
                    "AA",
                    "",
                ],
                [4, 3, 5, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    np.nan,
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                ],
                [4, 3, 5, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    np.nan,
                    "こんにち,は世界",
                    "大处着眼，小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                    "한국,가,고싶다ㅠ",
                ],
                [4, 3, 5, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "😀🐍,⚡😅😂",
                    "🌶🍔,🏈💔💑💕",
                    "𠁆𠁪,𠀓𠄩𠆶",
                    np.nan,
                    "🏈,💔,𠄩,😅",
                    "🠂,🠋🢇🄐,🞧",
                    "🢇🄐,🏈𠆶💑😅",
                ],
                [4, 3, 5, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pd.Series(
            [
                "A",
                " bbCD",
                " mCDm",
                "C,ABB, D",
                "B,B,CC",
                "ABBD",
                "ABCDD,OSAJD",
                "a1b2d314f,sdf234",
                "C,ABB,D",
                "¿abc¡Y tú, quién te cre\t\tes?",
                "오늘도 피츠버그의 날씨는 매\t우, 구림",
                np.nan,
                "🏈,💔,𠄩,😅",
                "大处着眼，小处着手。",
                "🠂,🠋🢇🄐,🞧",
                "россия очень, холодная страна",
                "",
                " ",
            ],
            [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            name="A",
        ),
    ]
)
def test_unicode(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            pd.Series(
                [
                    "AOSJD,OSAJD",
                    "a1b2d314f,sdf234",
                    "22!@#,$@#$",
                    "A,C,V,B,B",
                    "HELLO, WORLD",
                    "aAbB,ABC",
                ],
                [4, 3, 1, 0, 2, 5],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                    "Español es agra,dable escuchar",
                ],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    "こんにち,は世界",
                    "大处着眼, 小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                    "한국,가,고싶다ㅠ",
                ],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pytest.param(
            pd.Series(
                ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧", "🢇🄐,🏈𠆶💑😅"],
                [4, 3, 1, 0, -3, 2],
                name="A",
            ),
            marks=pytest.mark.slow,
        ),
        pd.Series(
            [
                "A",
                " bbCD",
                " mCDm",
                "C,ABB, D",
                "B,B,CC",
                "ABBD",
                "ABCDD,OSAJD",
                "a1b2d314f,sdf234",
                "C,ABB,D",
                "¿abc¡Y tú, quién te cre\t\tes?",
                "오늘도 피츠버그의 날씨는 매\t우, 구림",
                "🏈,💔,𠄩,😅",
                "🠂,🠋🢇🄐,🞧",
                "россия очень, холодная страна",
                " ",
                "",
            ],
            [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            name="A",
        ),
    ]
)
def test_unicode_no_nan(request):
    return request.param


def test_len(test_unicode):
    def test_impl(S):
        return S.str.len()

    check_func(test_impl, (test_unicode,), check_dtype=False)


def test_split(test_unicode_no_nan):
    def test_impl(S):
        return S.str.split(",")

    # TODO: more split tests similar to the ones test_hiframes
    # TODO: support and test NA
    # TODO: support distributed
    # check_func(test_impl, (S,))

    pd.testing.assert_series_equal(
        bodo.jit(test_impl)(test_unicode_no_nan), test_impl(test_unicode_no_nan)
    )


def test_get():
    def test_impl(S):
        B = S.str.split(",")
        return B.str.get(1)

    # TODO: support and test NA
    # TODO: support distributed
    S = pd.Series(
        [
            "A,B",
            " bb,CD",
            " mCD,m",
            "C,ABB, D",
            "B,B,CC",
            "AB,BD",
            "ABCDD,OSAJD",
            "a1b2d314f,sdf234",
            "C,ABB,D",
            "¿abc¡Y tú, quién te cre\t\tes?",
            "오늘도 피츠버그의 날씨는 매\t우, 구림",
            "🏈,💔,𠄩,😅",
            "🠂,🠋🢇🄐,🞧",
            "россия очень, холодная страна",
        ],
        [4, 3, 5, 1, 0, -3, 2, 6, 7, 8, 9, 10, 11, 12],
        name="A",
    )
    # check_func(test_impl, (S,))
    pd.testing.assert_series_equal(
        bodo.jit(test_impl)(S), test_impl(S), check_dtype=False
    )


def test_replace_regex(test_unicode):
    def test_impl(S):
        return S.str.replace("AB*", "EE", regex=True)

    def test_impl2(S):
        return S.str.replace("피츠*", "뉴욕의", regex=True)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_replace_noregex(test_unicode):
    def test_impl(S):
        return S.str.replace("AB", "EE", regex=False)

    def test_impl2(S):
        return S.str.replace("피츠버그의", "뉴욕의", regex=True)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_contains_regex(test_unicode):
    def test_impl(S):
        return S.str.contains("AB*", regex=True)

    def test_impl2(S):
        return S.str.contains("피츠버*", regex=True)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_contains_noregex(test_unicode):
    def test_impl(S):
        return S.str.contains("AB", regex=False)

    def test_impl2(S):
        return S.str.contains("피츠버그", regex=False)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_extract(test_unicode):
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd])(?P<C>\d+)")

    def test_impl2(S):
        return S.str.extract(r"(?P<BBB>[아])(?P<C>\d+)")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_extract_noexpand(test_unicode):
    def test_impl(S):
        return S.str.extract(r"(?P<BBB>[abd]+)\d+", expand=False)

    def test_impl2(S):
        return S.str.extract(r"(?P<BBB>[아])(?P<C>\d+)", expand=False)

    # when regex group has no name, Series name should be used
    def test_impl_noname(S):
        return S.str.extract(r"([abd]+)\d+", expand=False)

    check_func(test_impl, (test_unicode,))
    check_func(test_impl_noname, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_extractall():
    """Test Series.str.extractall() with various input cases
    """
    # ascii input with non-string index, single named group
    def test_impl1(S):
        return S.str.extractall(r"(?P<BBB>[abd]+)\d+")

    S = pd.Series(
        ["a1b1", "b1", np.nan, "a2", "c2", "ddd", "dd4d1", "d22c2"],
        [4, 3, 5, 1, 0, 2, 6, 11],
        name="AA",
    )
    check_func(test_impl1, (S,))

    # unicode input with string index, multiple unnamed group
    def test_impl2(S):
        return S.str.extractall(r"([чен]+)\d+([ст]+)\d+")

    S2 = pd.Series(
        ["чьь1т33", "ьнн2с222", "странаст2", np.nan, "ьнне33ст3"],
        ["е3", "не3", "н2с2", "AA", "C"],
    )
    check_func(test_impl2, (S2,))


def test_count_noflag(test_unicode):
    def test_impl(S):
        return S.str.count("A")

    def test_impl2(S):
        return S.str.count("피츠")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


def test_count_flag(test_unicode):
    import re

    # TODO: the flag does not work inside numba
    flag = re.IGNORECASE.value

    def test_impl(S):
        return S.str.count("A", flag)

    def test_impl2(S):
        return S.str.count("피츠", flag)

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


def test_find(test_unicode):
    def test_impl(S):
        return S.str.find("AB")

    def test_impl2(S):
        return S.str.find("🍔")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


def test_rfind(test_unicode):
    def test_impl(S):
        return S.str.rfind("AB")

    def test_impl2(S):
        return S.str.rfind("дн")

    check_func(test_impl, (test_unicode,), check_dtype=False)
    check_func(test_impl2, (test_unicode,), check_dtype=False)


def test_pad_fill_fast(test_unicode):
    # this function increases coverage for not slow test suite
    def test_impl1(S):
        return S.str.center(1, "필")

    def test_impl2(S):
        return S.str.rjust(1, "d")

    def test_impl3(S):
        return S.str.ljust(1, "a")

    def test_impl4(S):
        return S.str.pad(1, "left", "🍔")

    def test_impl5(S):
        return S.str.zfill(1)

    check_func(test_impl1, (test_unicode,))
    check_func(test_impl2, (test_unicode,))
    check_func(test_impl3, (test_unicode,))
    check_func(test_impl4, (test_unicode,))
    check_func(test_impl5, (test_unicode,))


@pytest.mark.slow
def test_center(test_unicode):
    def test_impl(S):
        return S.str.center(5, "*")

    def test_impl2(S):
        return S.str.center(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_ljust(test_unicode):
    def test_impl(S):
        return S.str.ljust(5, "*")

    def test_impl2(S):
        return S.str.ljust(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_rjust(test_unicode):
    def test_impl(S):
        return S.str.rjust(5, "*")

    def test_impl2(S):
        return S.str.rjust(5, "🍔")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


@pytest.mark.slow
def test_pad(test_unicode):
    def test_impl_default(S):
        return S.str.pad(5)

    def test_impl_left(S):
        return S.str.pad(5, "left", "*")

    def test_impl_right(S):
        return S.str.pad(5, "right", "*")

    def test_impl_both(S):
        return S.str.pad(5, "both", "*")

    def test_impl_both2(S):
        return S.str.pad(5, "both", "🍔")

    check_func(test_impl_default, (test_unicode,))
    check_func(test_impl_left, (test_unicode,))
    check_func(test_impl_right, (test_unicode,))
    check_func(test_impl_both, (test_unicode,))
    check_func(test_impl_both2, (test_unicode,))


@pytest.mark.slow
def test_zfill(test_unicode):
    def test_impl(S):
        return S.str.zfill(10)

    check_func(test_impl, (test_unicode,))


def test_slice(test_unicode):
    def test_impl(S):
        return S.str.slice(step=2)

    check_func(test_impl, (test_unicode,))


def test_startswith(test_unicode):
    def test_impl(S):
        return S.str.startswith("AB")

    def test_impl2(S):
        return S.str.startswith("테스팅")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_endswith(test_unicode):
    def test_impl(S):
        return S.str.endswith("AB")

    def test_impl2(S):
        return S.str.endswith("테스팅")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_isupper(test_unicode):
    def test_impl(S):
        return S.str.isupper()

    check_func(test_impl, (test_unicode,))


@pytest.mark.parametrize("ind", [slice(2), 2])
def test_getitem(ind, test_unicode):
    def test_impl(S, ind):
        return S.str[ind]

    check_func(test_impl, (test_unicode, ind))


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
        np.array(
            [
                ["a", "bc"],
                ["a"],
                [],
                ["aaa", "", "cc"],
                [""],
                np.nan,
                [
                    "¿abc¡Y tú, quién te crees?",
                    "ÕÕÕú¡úú,úũ¿ééé",
                    "россия очень, холодная страна",
                    "مرحبا, العالم ، هذا هو بودو",
                    "Γειά σου ,Κόσμε",
                ],
                [
                    "아1, 오늘 저녁은 뭐먹지",
                    "나,는 유,니,코,드 테스팅 중",
                    "こんにち,は世界",
                    "大处着眼，小处着手。",
                    "오늘도 피츠버그의 날씨는 매우, 구림",
                ],
                ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧"],
            ]
            * 1000
        ),
    ]
)
def list_str_arr_value(request):
    return request.param


def test_list_str_arr_unbox(list_str_arr_value, memory_leak_check):
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


def test_flatten1(test_unicode_no_nan):
    """tests flattening array of string lists after split call when split view
    optimization is applied
    """

    def impl(S):
        A = S.str.split(",")
        return pd.Series(list(itertools.chain(*A)))

    check_func(impl, (test_unicode_no_nan,))


def test_flatten2(test_unicode_no_nan):
    """tests flattening array of string lists after split call when split view
    optimization is not applied
    """

    def impl(S):
        A = S.str.split()
        return pd.Series(list(itertools.chain(*A)))

    check_func(impl, (test_unicode_no_nan,))


def test_flatten3(test_unicode_no_nan):
    """tests flattening array without the "list" call
    """

    def impl(S):
        A = S.str.split(",")
        return pd.Series(itertools.chain(*A))

    check_func(impl, (test_unicode_no_nan,))


def test_flatten4(test_unicode_no_nan):
    """tests flattening array with "from_iterable"
    """

    def impl(S):
        A = S.str.split(",")
        res = pd.Series(itertools.chain.from_iterable(A))
        return res

    check_func(impl, (test_unicode_no_nan,))


def test_join():
    """test the functionality of bodo's join with NaN
    """

    def test_impl(S):
        return S.str.join("-")

    S = pd.Series(
        [
            ["ABCDD,OSAJD", "a1b2d314f,sdf234", "22!@#,$@#$", "A,C,V,B,B", ""],
            [
                "¿abc¡Y tú, quién te crees?",
                "ÕÕÕú¡úú,úũ¿ééé",
                "россия очень, холодная страна",
                "مرحبا, العالم ، هذا هو بودو",
                "Γειά σου ,Κόσμε",
            ],
            [
                "아1, 오늘 저녁은 뭐먹지",
                "나,는 유,니,코,드 테스팅 중",
                "こんにち,は世界",
                "大处着眼，小处着手。",
                "오늘도 피츠버그의 날씨는 매우, 구림",
            ],
            np.nan,
            ["😀🐍,⚡😅😂", "🌶🍔,🏈💔💑💕", "𠁆𠁪,𠀓𠄩𠆶", "🏈,💔,𠄩,😅", "🠂,🠋🢇🄐,🞧"],
        ]
    )
    check_func(test_impl, (S,))


def test_join_string(test_unicode):
    """test the functionality of bodo's join with just a string
    """

    def test_impl(test_unicode):
        return test_unicode.str.join("-")

    def test_impl2(test_unicode):
        return test_unicode.str.join("*****************")

    check_func(test_impl, (test_unicode,))
    check_func(test_impl2, (test_unicode,))


def test_join_splitview(test_unicode_no_nan):
    """test the functionality of bodo's join with split view type as an input
    """

    def test_impl(S):
        B = S.str.split(",")
        return B.str.join("-")

    check_func(test_impl, (test_unicode_no_nan,))


def test_join_splitview_nan_entry():
    """test the functionality of bodo's join with split view type as an input
    """

    def test_impl(S):
        B = S.str.split(",")
        return B.str.join("-")

    S = pd.Series(["ABCDD,OSAJD", "a1b2d314f,sdf234", np.nan], [4, 3, 1], name="A")
    check_func(test_impl, (S,))
